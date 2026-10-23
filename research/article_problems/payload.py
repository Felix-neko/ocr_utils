"""Вариант прогона (модель, путь, рассуждения) и сборка запроса по статье: тело, payload, ключ и папка кэша."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ocr_utils.external_ocr_services.cache import entry_dir, request_key
from research.article_problems import PROMPT_VERSION
from research.article_problems.articles import Article

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
# Суффикс batch-варианта модели у OpenRouter: такой слаг принимает только Batch API.
BATCH_SUFFIX = ":batch"
# Закреплённый провайдер синхронного пути по модели: у DeepSeek — родной эндпоинт без квантования (как в
# external_ocr_services). Модели не из словаря идут без закрепления — маршрут выбирает OpenRouter.
SYNC_PROVIDERS = {"deepseek/deepseek-v4.1-flash": "deepseek"}
JSON_MODE = "json_object"
# Потолок ответа: случаи с цитатами редко длиннее 5 тыс. токенов; с рассуждениями они входят в тот же потолок.
MAX_TOKENS_PLAIN = 16000
MAX_TOKENS_REASONING = 32000

# Справочные цены, $ за млн токенов (каталог OpenRouter /api/v1/models на 2026-09-29) — только для
# прикидки до отправки; настоящая цена берётся из usage.cost ответов.
PRICE_PER_M = {
    "deepseek/deepseek-v4.1-flash:batch": (0.112, 0.336),
    "deepseek/deepseek-v4.1-flash": (0.15, 0.60),  # провайдер DeepSeek, дневной тариф
    "stealth/space-bunny-alpha": (0.0, 0.0),  # анонимная модель, бесплатна на время теста
}
# Для прикидки: символов русского текста на токен и средний размер ответа.
CHARS_PER_TOKEN = 3.2
EST_ANSWER_TOKENS = 1000
EST_REASONING_TOKENS = {"off": 0, "low": 3800, "high": 8000}  # low — по пилоту v2 (55 статей)


class Transport(StrEnum):
    """Как запрос попадает к модели."""

    SYNC = "sync"  # chat/completions, ответ сразу; для пилота и итераций промпта
    BATCH = "batch"  # Batch API OpenRouter, ответ до 24 ч, дешевле


class ReasoningLevel(StrEnum):
    """Уровень рассуждений модели (``reasoning.effort`` OpenRouter; ``off`` → ``none``)."""

    OFF = "off"
    LOW = "low"
    HIGH = "high"


@dataclass(frozen=True)
class Variant:
    """Вариант прогона: всё, что вместе с текстом статьи определяет ответ модели.

    Args:
        model: Базовый слаг модели OpenRouter (без ``:batch``).
        transport: Синхронный запрос или Batch API.
        reasoning: Уровень рассуждений.
        prompt_version: Версия системного промпта (файл ``prompts/system_v{N}.md``).
    """

    model: str = DEFAULT_MODEL
    transport: Transport = Transport.SYNC
    reasoning: ReasoningLevel = ReasoningLevel.OFF
    prompt_version: int = PROMPT_VERSION

    @property
    def model_slug(self) -> str:
        """Слаг, который уходит в OpenRouter: для Batch API — с ``:batch``."""
        return self.model + BATCH_SUFFIX if self.transport == Transport.BATCH else self.model

    @property
    def stage(self) -> str:
        """Имя этапа в папке кэша: ``problems.v1.sync.off``."""
        return f"problems.v{self.prompt_version}.{self.transport.value}.{self.reasoning.value}"

    @property
    def label(self) -> str:
        """Метка прогона для папки результатов: ``v1_sync_off``; у модели не по умолчанию — с её именем."""
        label = f"v{self.prompt_version}_{self.transport.value}_{self.reasoning.value}"
        return label if self.model == DEFAULT_MODEL else f"{label}_{self.model.split('/')[-1]}"

    @property
    def max_tokens(self) -> int:
        """Потолок ответа: с рассуждениями больше, они входят в тот же счёт."""
        return MAX_TOKENS_PLAIN if self.reasoning == ReasoningLevel.OFF else MAX_TOKENS_REASONING


def system_prompt(version: int) -> str:
    """Текст системного промпта нужной версии.

    Args:
        version: Номер версии (``prompts/system_v{version}.md``).
    """
    return (PROMPTS_DIR / f"system_v{version}.md").read_text(encoding="utf-8")


def user_message(article: Article) -> str:
    """Сообщение пользователя: паспорт статьи и её текст целиком.

    Args:
        article: Статья.
    """
    header = [
        f"Журнал «Материально-техническое снабжение», {article.year} год, № {article.issue}.",
        f"Рубрика: {article.rubric or '—'}",
        f"Заголовок: {article.title or '—'}",
    ]
    if article.authors:
        header.append("Авторы: " + "; ".join(article.authors))
    return "\n".join(header) + "\n\n---\n\n" + article.text


def request_body(article: Article, variant: Variant) -> dict:
    """Тело запроса chat/completions без модели и провайдера — общее для обоих путей.

    Args:
        article: Статья.
        variant: Вариант прогона.

    Returns:
        ``messages``, ``temperature``, ``max_tokens``, ``response_format``, ``reasoning`` в форме HTTP API.
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt(variant.prompt_version)},
            {"role": "user", "content": user_message(article)},
        ],
        "temperature": 0,
        "max_tokens": variant.max_tokens,
        # Схема описана словами в промпте: родной эндпоинт DeepSeek structured outputs не умеет.
        "response_format": {"type": JSON_MODE},
        # effort none выключает thinking целиком (по докам OpenRouter); low/high — уровень.
        "reasoning": {"effort": "none" if variant.reasoning == ReasoningLevel.OFF else variant.reasoning.value},
    }


def sync_payload(article: Article, variant: Variant) -> dict:
    """Payload синхронного запроса: тело + модель + закреплённый провайдер (если он задан для модели).

    Args:
        article: Статья.
        variant: Вариант прогона (транспорт ``SYNC``).
    """
    payload = {"model": variant.model_slug, **request_body(article, variant)}
    provider = SYNC_PROVIDERS.get(variant.model)
    if provider is not None:
        # Без откатов на чужие (квантованные) копии модели.
        payload["provider"] = {"order": [provider], "allow_fallbacks": False}
    return payload


def cache_payload(article: Article, variant: Variant) -> dict:
    """Payload, по которому считается ключ кэша: для sync — как уходит, для batch — тело с моделью.

    Args:
        article: Статья.
        variant: Вариант прогона.
    """
    if variant.transport == Transport.SYNC:
        return sync_payload(article, variant)
    return {"model": variant.model_slug, **request_body(article, variant)}


def cache_entry(cache_root: Path, article: Article, variant: Variant) -> Path:
    """Папка записи кэша статьи для варианта: ``cache/1968/09/A15/problems.v1.sync.off.json_object.<ключ>``.

    Args:
        cache_root: Корень кэша.
        article: Статья.
        variant: Вариант прогона.
    """
    key = request_key(cache_payload(article, variant))
    return entry_dir(cache_root, article.rel, variant.stage, JSON_MODE, key)


def request_info(article: Article, variant: Variant) -> dict:
    """Паспорт запроса для ``request.json`` кэша: откуда статья и каким вариантом спрошена.

    Args:
        article: Статья.
        variant: Вариант прогона.
    """
    return {
        "article": article.key,
        "title": article.title,
        "rubric": article.rubric,
        "source": str(article.source_path),
        "line_start": article.line_start,
        "variant": variant.label,
        "model": variant.model_slug,
        "prompt_version": variant.prompt_version,
    }


def estimate_cost_usd(articles: list[Article], variant: Variant) -> float:
    """Прикидка цены прогона до отправки: токены по длине текста, справочная цена модели.

    Args:
        articles: Статьи прогона.
        variant: Вариант прогона.

    Returns:
        Долларов; 0, если цены модели нет в ``PRICE_PER_M``.
    """
    price_in, price_out = PRICE_PER_M.get(variant.model_slug, (0.0, 0.0))
    prompt_chars = len(system_prompt(variant.prompt_version))
    tokens_in = sum((prompt_chars + len(user_message(article))) / CHARS_PER_TOKEN for article in articles)
    tokens_out = len(articles) * (EST_ANSWER_TOKENS + EST_REASONING_TOKENS[variant.reasoning.value])
    return (tokens_in * price_in + tokens_out * price_out) / 1e6
