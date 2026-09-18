"""Реестр OpenRouter-моделей боевого прогона: короткое имя, id, режим JSON, рассуждения, провайдеры.

Цены справочные (каталог OpenRouter на 2026-09-13, $ за миллион токенов) — для прикидки до прогона;
реальная стоимость каждой полосы берётся из ``usage.cost`` ответа и пишется в .meta.json. Локальных
движков здесь нет: они остались в исследовательском стенде.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class JsonMode(StrEnum):
    """Как просить структурированный ответ; значения — как в ``response_format.type`` OpenRouter.

    ``JSON_SCHEMA`` — строгая схема через ``response_format`` (не у всех провайдеров);
    ``JSON_OBJECT`` — «верни валидный JSON», сама схема описана словами в промпте;
    ``NONE`` — только промпт, без ``response_format``. Цепочка запасных ходов идёт в этом порядке.
    """

    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"
    NONE = "none"


class Reasoning(StrEnum):
    """Уровень рассуждений модели.

    ``OFF`` — выключить thinking (уходит как ``reasoning.effort: none``); ``LOW`` / ``MEDIUM`` —
    ``reasoning.effort``; ``NONE`` — параметр модели не знаком, не слать вовсе. На OCR рассуждения
    не помогают, а стоят втрое. Потолок ``reasoning.max_tokens`` штатный SDK OpenRouter не
    передаёт, поэтому у моделей с невыключаемым thinking (Gemini) остаётся только уровень.
    """

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    NONE = "none"


DEFAULT_MODEL = "deepseek-v41-flash"


@dataclass(frozen=True)
class ModelSpec:
    """Запись реестра: как зовут модель у нас и у OpenRouter, как с ней разговаривать и почём.

    Args:
        name: Короткое имя в CLI (``--model``) и в meta полосы.
        openrouter_id: Идентификатор модели у OpenRouter (``vendor/model``).
        price_in_per_m: Справочная цена входа, $ за миллион токенов (для прикидки, не для учёта).
        price_out_per_m: То же для выхода.
        json_mode: С какого режима JSON начинать; при отказе провайдера цепочка идёт к более простым.
        reasoning: Уровень рассуждений по умолчанию; ``--reasoning`` в CLI его переопределяет.
        image_detail: Значение ``detail`` у ``image_url``; DeepSeek по ``low`` ужимает картинку
            до 512×512 и теряет текст, поэтому ``high``. ``None`` — не слать.
        provider_order: Предпочтительные провайдеры OpenRouter по убыванию (с откатом на остальных).
        provider_ignore: Провайдеры, которых не брать никогда (квантованные копии читают хуже).
        notes: Заметка для вывода ``models``: что известно о модели по стенду.
    """

    name: str
    openrouter_id: str
    price_in_per_m: float
    price_out_per_m: float
    json_mode: JsonMode = JsonMode.JSON_SCHEMA
    reasoning: Reasoning = Reasoning.OFF
    image_detail: str | None = "high"
    provider_order: tuple[str, ...] = ()
    provider_ignore: tuple[str, ...] = ()
    notes: str = ""


# fmt: off
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "deepseek-v41-flash", "deepseek/deepseek-v4.1-flash", 0.15, 0.60,
        # Родной эндпоинт умеет response_format, но не structured_outputs — отсюда json_object. Он же
        # первый по порядку: самый дешёвый и без квантования. Relace отдаёт fp4-копию — не берём.
        json_mode=JsonMode.JSON_OBJECT, provider_order=("deepseek",), provider_ignore=("relace",),
        notes="нативное зрение, потолок 1024 токенов на картинку ≈ 1300 px: полоса идёт тайлами",
    ),
    ModelSpec("gemini-31-flash-lite", "google/gemini-3.1-flash-lite", 0.25, 1.50, reasoning=Reasoning.LOW, notes="плитки 768 px; thinking не выключается, только low (потолок через SDK не передать)"),
    ModelSpec("qwen38-flash", "qwen/qwen3.8-flash", 0.15, 0.47, notes="MoE-flash Qwen3.8, thinking выключаем"),
)
# fmt: on

# Быстрый доступ по короткому имени — для ``resolve`` и CLI.
REGISTRY: dict[str, ModelSpec] = {spec.name: spec for spec in MODELS}


def resolve(name: str) -> ModelSpec:
    """Спецификация по короткому имени; неизвестное имя — ошибка со списком известных.

    Args:
        name: Короткое имя из реестра (``--model`` в CLI), например ``deepseek-v41-flash``.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"неизвестная модель {name!r}; известны: {', '.join(sorted(REGISTRY))}") from None
