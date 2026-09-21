"""Вспомогательный текстовый запрос по статьям без заголовка: оглавление и строки-кандидаты → какая строка — название.

Сверка выпуска с оглавлением (``reconcile``) ставит `#` по id и названию и восстанавливает заголовок
из тела в окне страницы статьи (строго, ослабленно ≥ 0.6, из маркера) и из колонтитула. Что осталось
без заголовка, показывается той же модели **текстом, без картинок**: всё оглавление выпуска (контекст —
что уже привязано), непривязанные статьи и для каждой пронумерованные строки-кандидаты из
транскрипции полос вокруг её страницы (заголовки `##`, жирные и курсивные абзацы, маркеры разделов,
первые абзацы). Модель отвечает номером строки и уверенностью; принимается только ``high`` и только
строка из окна статьи (или заголовочная, если окна нет). Строки, которой нет в транскрипции, этот
запрос не найдёт — это осознанный предел: картинки полос сюда не шлются.

Один запрос на выпуск (до ``BATCH_MAX_ARTICLES`` статей, дальше ещё запрос), кэш — как у проверки
стыков: ``cache/{год}/{выпуск}/_headings/heading.json_object.{ключ}/``, своя версия промпта
``HEADING_PROMPT_VERSION``. Применение — подсказки сверке (``reconcile_headings(hints=…)``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ocr_utils.external_ocr_services.boundary import CheckerStats
from ocr_utils.external_ocr_services.cache import CacheVerdict, cache_for, entry_dir, request_key
from ocr_utils.external_ocr_services.client import OpenRouterClient, OpenRouterError
from ocr_utils.external_ocr_services.models import JsonMode, ModelSpec
from ocr_utils.external_ocr_services.ocr import provider_field, reasoning_field
from ocr_utils.external_ocr_services.prompts import HEADING_PROMPT_VERSION, heading_prompts

logger = logging.getLogger(__name__)

HEADING_STAGE = "heading"  # имя этапа в папке кэша: cache/{год}/{выпуск}/_headings/heading.…
HEADING_PAGE = "_headings"  # псевдополоса выпуска в кэше — запрос общий на все статьи без заголовка
HEADING_MAX_TOKENS = 200  # ответ на одну статью — короткий JSON
BATCH_MAX_ARTICLES = 10  # статей в одном запросе; больше — ещё запрос
LINES_PER_ARTICLE = 20  # строк-кандидатов на статью в запросе
LINE_MAX_CHARS = 200  # длиннее — обрезается: название не бывает длиннее


class Confidence(StrEnum):
    """Уверенность модели в выборе строки; принимается только ``HIGH``."""

    HIGH = "high"
    LOW = "low"


class HintStatus(StrEnum):
    """Что стало с ответом модели по статье."""

    ACCEPTED = "accepted"  # строка принята — станет `#`
    REJECTED = "rejected"  # строка отклонена: уверенность low, номер вне списка или строка вне окна статьи
    NONE = "none"  # модель не нашла названия среди строк
    ERROR = "error"  # запрос или разбор не удался


@dataclass(frozen=True)
class CandidateLine:
    """Строка-кандидат из транскрипции: номер в запросе, индекс блока выпуска, полоса, текст."""

    number: int  # номер в списке запроса (с 1)
    block: int  # индекс блока в тексте выпуска (до правок сверки)
    page_file: str
    page_number: int | None
    text: str
    in_window: bool  # с полосы в окне «страница статьи ±1» (или заголовочная строка, если окна нет)


@dataclass
class HeadingQuery:
    """Статья без заголовка и её строки-кандидаты."""

    article_id: str
    title: str
    authors: list[str]
    page: str | None  # страница по оглавлению как напечатана
    lines: list[CandidateLine] = field(default_factory=list)


@dataclass
class HeadingVerdict:
    """Ответ модели по статье и учёт запроса."""

    article_id: str
    line: int | None = None  # номер строки в запросе
    block: int | None = None  # индекс блока выпуска, если строка принята
    confidence: str = ""
    notes: str = ""
    status: HintStatus = HintStatus.NONE
    cache_entry: str = ""
    cache_hit: bool = False
    cost_usd: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict:
        """Для sidecar: пустые поля опущены, перечисление строкой."""
        data = {key: value for key, value in vars(self).items() if value not in ("", None, False, 0.0)}
        data["status"] = self.status.value
        return data


def parse_verdicts(text: str) -> dict[str, dict]:
    """Разобрать ответ: ``{"articles": [{"article_id", "line", "confidence", "notes"}]}`` → по id статьи.

    Args:
        text: Сырой ответ модели (JSON, возможно в ограждении).

    Returns:
        ``{article_id: запись}``.

    Raises:
        ValueError: JSON не разобран или нет списка ``articles``.
    """
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("в ответе нет JSON")
    payload = json.JSONDecoder().raw_decode(stripped[start:])[0]
    items = payload.get("articles") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("в ответе нет списка articles")
    return {str(item.get("article_id")): item for item in items if isinstance(item, dict) and item.get("article_id")}


class HeadingChecker:
    """Сопоставление статей без заголовка со строками текста той же моделью — текстом, один запрос на выпуск.

    Args:
        client: Клиент OpenRouter.
        spec: Модель из реестра (та же, что распознаёт полосы).
        cache_dir: Кэш запросов (``--cache-dir``); ``None`` — без кэша.
        max_tokens: Потолок ответа на одну статью (умножается на число статей в запросе).
    """

    def __init__(
        self, client: OpenRouterClient, spec: ModelSpec, cache_dir: Path | None, max_tokens: int = HEADING_MAX_TOKENS
    ):
        self.client = client
        self.spec = spec
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.max_tokens = max_tokens
        self.stats = CheckerStats()

    def payload(self, toc: list[dict], queries: list[HeadingQuery]) -> dict[str, Any]:
        """Тело запроса: системный промпт и текст с оглавлением и строками-кандидатами.

        Args:
            toc: Все статьи оглавления ``{"id", "title", "authors", "page"}``.
            queries: Статьи запроса по порядку.
        """
        described = [
            {
                "article_id": query.article_id,
                "title": query.title,
                "authors": query.authors,
                "page": query.page,
                "lines": [
                    {"number": line.number, "page_number": line.page_number, "text": line.text} for line in query.lines
                ],
            }
            for query in queries
        ]
        system, user = heading_prompts(toc, described)
        payload: dict[str, Any] = {
            "model": self.spec.openrouter_id,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": self.max_tokens * max(1, len(queries)),
            "response_format": {"type": JsonMode.JSON_OBJECT.value},
        }
        # Рассуждения — как у полос (по реестру выключены): иначе модель думает в счёт потолка ответа.
        reasoning = reasoning_field(self.spec, None)
        if reasoning is not None:
            payload["reasoning"] = reasoning
        provider = provider_field(self.spec)
        if provider:
            payload["provider"] = provider
        return payload

    def check_issue(self, issue_key: str, toc: list[dict], queries: list[HeadingQuery]) -> dict[str, HeadingVerdict]:
        """Показать модели все статьи выпуска без заголовка.

        Args:
            issue_key: «год/выпуск» — папка записи в кэше.
            toc: Все статьи оглавления.
            queries: Статьи без заголовка со строками-кандидатами; без строк — пропускаются.

        Returns:
            ``{article_id: вердикт}`` для статей со строками.
        """
        verdicts: dict[str, HeadingVerdict] = {}
        usable = [query for query in queries if query.lines]
        for start in range(0, len(usable), BATCH_MAX_ARTICLES):
            verdicts.update(self._request(issue_key, toc, usable[start : start + BATCH_MAX_ARTICLES]))
        return verdicts

    def _request(self, issue_key: str, toc: list[dict], queries: list[HeadingQuery]) -> dict[str, HeadingVerdict]:
        """Один запрос (из кэша или в сеть) на пачку статей.

        Args:
            issue_key: «год/выпуск».
            toc: Все статьи оглавления.
            queries: Статьи пачки.
        """
        payload = self.payload(toc, queries)
        self.stats.requests += 1
        cache = cache_for(self.cache_dir) if self.cache_dir is not None else None
        entry = None
        cache_entry = ""
        key = request_key(payload)
        if cache is not None:
            entry = entry_dir(
                self.cache_dir, Path(issue_key) / HEADING_PAGE, HEADING_STAGE, JsonMode.JSON_OBJECT.value, key
            )
            cache_entry = entry.relative_to(self.cache_dir).as_posix()
            cached = cache.lookup(entry)
            if cached is not None:
                self.stats.cache_hits += 1
                return self._finish(queries, cached.response.text, cached.response.cost_usd, cache_entry, True)
        try:
            response = self.client.chat(payload)
        except OpenRouterError as error:
            if cache is not None and entry is not None:
                cache.record_error(entry, str(error), error.body)
            self.stats.errors += 1
            return {
                query.article_id: HeadingVerdict(
                    query.article_id, status=HintStatus.ERROR, cache_entry=cache_entry, error=str(error)
                )
                for query in queries
            }
        if cache is not None and entry is not None and response.text.strip():
            info = {
                "issue": issue_key,
                "stage": HEADING_STAGE,
                "articles": [
                    {"article_id": query.article_id, "title": query.title, "lines": len(query.lines)}
                    for query in queries
                ],
                "heading_prompt_version": HEADING_PROMPT_VERSION,
                "model": self.spec.name,
                "openrouter_id": self.spec.openrouter_id,
                "key": key,
            }
            cache.store(entry, info, payload, response, CacheVerdict.OK)
        self.stats.cost_usd += response.cost_usd or 0.0
        return self._finish(queries, response.text, response.cost_usd, cache_entry, False)

    def _finish(
        self, queries: list[HeadingQuery], text: str, cost: float | None, cache_entry: str, cache_hit: bool
    ) -> dict[str, HeadingVerdict]:
        """Разобрать ответ пачки и решить по каждой статье: принять, отклонить, нет ответа.

        Args:
            queries: Статьи пачки.
            text: Сырой ответ.
            cost: Цена ответа (делится между статьями поровну для sidecar).
            cache_entry: Папка записи кэша.
            cache_hit: Ответ взят из кэша.
        """
        share = (cost or 0.0) / max(1, len(queries))
        try:
            parsed = parse_verdicts(text)
        except (ValueError, TypeError) as error:
            self.stats.errors += 1
            return {
                query.article_id: HeadingVerdict(
                    query.article_id,
                    status=HintStatus.ERROR,
                    cache_entry=cache_entry,
                    cache_hit=cache_hit,
                    error=f"ответ не разобран: {error}",
                )
                for query in queries
            }
        result: dict[str, HeadingVerdict] = {}
        for query in queries:
            item = parsed.get(query.article_id) or {}
            verdict = HeadingVerdict(
                query.article_id,
                confidence=str(item.get("confidence") or ""),
                notes=str(item.get("notes") or ""),
                cache_entry=cache_entry,
                cache_hit=cache_hit,
                cost_usd=share,
            )
            line_number = item.get("line")
            line = next((line for line in query.lines if line.number == line_number), None) if line_number else None
            if line_number is None or not item:
                verdict.status = HintStatus.NONE
            elif line is None or verdict.confidence != Confidence.HIGH or not line.in_window:
                # Номер вне списка, уверенность low или строка с полосы вне окна статьи — не верим.
                verdict.line, verdict.status = (
                    line_number if isinstance(line_number, int) else None
                ), HintStatus.REJECTED
            else:
                verdict.line, verdict.block, verdict.status = line.number, line.block, HintStatus.ACCEPTED
            result[query.article_id] = verdict
        return result
