"""Реестр OpenRouter-моделей боевого прогона: короткое имя, id, режим JSON, рассуждения, провайдеры.

Цены справочные (каталог OpenRouter на 2026-09-13, $ за миллион токенов) — для прикидки до прогона;
реальная стоимость каждой полосы берётся из ``usage.cost`` ответа и пишется в .meta.json. Локальных
движков здесь нет: они остались в исследовательском стенде.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Как просить структурированный ответ: строгая схема через response_format (не у всех
# провайдеров), «верни валидный JSON» со схемой в промпте, либо только промпт.
JsonMode = Literal["json_schema", "json_object", "none"]

# Уровень рассуждений. ``off`` — выключить thinking, ``low``/``medium`` — reasoning.effort,
# ``none`` — параметр модели не знаком, не слать. На OCR рассуждения не помогают, а стоят втрое.
Reasoning = Literal["off", "low", "medium", "none"]

DEFAULT_MODEL = "deepseek-v41-flash"


@dataclass(frozen=True)
class ModelSpec:
    name: str  # короткое имя в CLI и в meta
    openrouter_id: str
    price_in_per_m: float
    price_out_per_m: float
    json_mode: JsonMode = "json_schema"
    reasoning: Reasoning = "off"
    # ``detail`` в image_url: DeepSeek по ``low`` ужимает картинку до 512×512.
    image_detail: str | None = "high"
    # Предпочтительные провайдеры OpenRouter по убыванию; ``provider_ignore`` — кого не брать никогда.
    provider_order: tuple[str, ...] = ()
    provider_ignore: tuple[str, ...] = ()
    # Потолок токенов рассуждений там, где thinking не выключается (Gemini 3.x).
    reasoning_max_tokens: int | None = None
    notes: str = ""


# fmt: off
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "deepseek-v41-flash", "deepseek/deepseek-v4.1-flash", 0.15, 0.60,
        # Родной эндпоинт умеет response_format, но не structured_outputs — отсюда json_object. Он же
        # первый по порядку: самый дешёвый и без квантования. Relace отдаёт fp4-копию — не берём.
        json_mode="json_object", provider_order=("deepseek",), provider_ignore=("relace",),
        notes="нативное зрение, потолок 1024 токенов на картинку ≈ 1300 px: полоса идёт тайлами",
    ),
    ModelSpec("gemini-31-flash-lite", "google/gemini-3.1-flash-lite", 0.25, 1.50, reasoning="low", reasoning_max_tokens=1024, notes="плитки 768 px; thinking не выключается, только low + потолок"),
    ModelSpec("qwen38-flash", "qwen/qwen3.8-flash", 0.15, 0.47, notes="MoE-flash Qwen3.8, thinking выключаем"),
)
# fmt: on

REGISTRY: dict[str, ModelSpec] = {spec.name: spec for spec in MODELS}


def resolve(name: str) -> ModelSpec:
    """Спецификация по короткому имени; неизвестное имя — ошибка со списком известных."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"неизвестная модель {name!r}; известны: {', '.join(sorted(REGISTRY))}") from None
