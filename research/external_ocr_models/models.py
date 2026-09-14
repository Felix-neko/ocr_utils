"""Реестр моделей: что гоняем, под каким именем и с какими особенностями запроса.

Цены здесь справочные (каталог OpenRouter на 2026-09-13, $ за миллион токенов) — они нужны
только для прикидки до прогона. Реальная стоимость каждой полосы берётся из ``usage.cost``
ответа и попадает в .meta.json; отчёт считает по ней, а не по этой таблице.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Как просить структурированный ответ. ``json_schema`` — строгая схема через
# response_format (поддерживают не все провайдеры одной и той же модели, поэтому запрос
# идёт с require_parameters); ``json_object`` — «верни валидный JSON», схема описана в
# промпте; ``none`` — только промпт, JSON вытаскивается из текста.
JsonMode = Literal["json_schema", "json_object", "none"]

# Уровень рассуждений. ``off`` — попросить провайдера выключить thinking (там, где это
# возможно), ``low``/``medium`` — reasoning.effort. Замер на DeepSeek V4.1 Flash: с
# ``low`` полоса стоила втрое дороже (4238 токенов рассуждений) при побайтно том же тексте,
# поэтому по умолчанию рассуждения выключены.
Reasoning = Literal["off", "low", "medium", "none"]


@dataclass(frozen=True)
class ModelSpec:
    name: str  # короткое имя в CLI и в имени папки выхода
    openrouter_id: str
    price_in_per_m: float
    price_out_per_m: float
    json_mode: JsonMode = "json_schema"
    # ``none`` — параметр reasoning модели не знаком (instruct-модели без thinking), не слать.
    reasoning: Reasoning = "off"
    # ``detail`` в image_url: DeepSeek по ``low`` ужимает картинку до 512×512, остальные
    # провайдеры поле либо игнорируют, либо трактуют как у OpenAI.
    image_detail: str | None = "high"
    # Предпочтительные провайдеры OpenRouter (slug) в порядке убывания; пусто — на
    # усмотрение маршрутизатора (он берёт самый дешёвый живой). ``provider_ignore`` —
    # кого не брать никогда (например, fp4-квантованные копии).
    provider_order: tuple[str, ...] = ()
    provider_ignore: tuple[str, ...] = ()
    # Потолок токенов рассуждений там, где thinking не выключается (Gemini 3.x): без него
    # модель изредка уходит в спираль на 15 тыс. токенов и обрезает сам ответ.
    reasoning_max_tokens: int | None = None
    notes: str = ""
    local: bool = False  # локальный движок из research/external_ocr_models/local


# fmt: off
OPENROUTER_MODELS: tuple[ModelSpec, ...] = (
    # --- дешёвые ($0.15-0.25 за M входа) ---
    ModelSpec(
        "deepseek-v41-flash", "deepseek/deepseek-v4.1-flash", 0.15, 0.60,
        # Родной эндпоинт DeepSeek умеет response_format, но не structured_outputs — отсюда
        # json_object. Он же первый в порядке: самый дешёвый и без квантования. Его отсекает
        # настройка приватности OpenRouter «провайдеры, обучающиеся на запросах» — если она
        # включена, маршрутизатор молча уходит на Fireworks/Morph/DeepInfra (fp8, та же цена).
        # Relace отдаёт fp4-квантованную копию — не берём никогда.
        json_mode="json_object", provider_order=("deepseek",), provider_ignore=("relace",),
        notes="нативное зрение (DeepSeek-ViT), потолок 1024 токенов на картинку ≈ 1300 px; для плотной полосы нужны --strips 2-3",
    ),
    ModelSpec("gemini-31-flash-lite", "google/gemini-3.1-flash-lite", 0.25, 1.50, reasoning="low", reasoning_max_tokens=1024, notes="плитки 768 px по 258 токенов; thinking не выключается, только low + потолок"),
    ModelSpec("qwen38-flash", "qwen/qwen3.8-flash", 0.15, 0.47, notes="MoE-flash Qwen3.8, thinking выключаем"),
    ModelSpec("gpt-56-luna", "openai/gpt-5.6-luna", 0.20, 1.20, notes="самый дешёвый OpenAI с картинками"),
    ModelSpec("mistral-small-4", "mistralai/mistral-small-2603", 0.15, 0.60),
    ModelSpec("glm-53-flash", "z-ai/glm-5.3-flash", 0.15, 0.50),
    # --- средние ---
    ModelSpec("gemini-38-flash", "google/gemini-3.8-flash", 0.75, 3.75, reasoning="low", reasoning_max_tokens=1024, notes="лидер OCR Arena среди flash-моделей; thinking не выключается, только low + потолок"),
    ModelSpec(
        "qwen3-vl-235b", "qwen/qwen3-vl-235b-a22b-instruct", 0.21, 1.90, reasoning="none",
        notes="instruct без thinking; заявлен document parsing",
    ),
    ModelSpec("claude-haiku-45", "anthropic/claude-haiku-4.5", 1.00, 5.00),
    # --- резерв: в реестре есть, в пробник по умолчанию не входят ---
    ModelSpec("gemini-31-pro", "google/gemini-3.1-pro-preview", 2.00, 12.00, reasoning="low", reasoning_max_tokens=1024, notes="эталон качества, дорого"),
    ModelSpec("claude-sonnet-5", "anthropic/claude-sonnet-5", 2.00, 10.00, notes="эталон качества, дорого"),
    ModelSpec("gpt-54-mini", "openai/gpt-5.4-mini", 0.75, 4.50),
    ModelSpec("qwen37-flash", "qwen/qwen3.7-flash", 0.03, 0.13, json_mode="json_object", notes="самая дешёвая с картинками"),
    ModelSpec("gemma-4-31b", "google/gemma-4-31b-it", 0.09, 0.34, notes="открытые веса, max_out 16k"),
    ModelSpec("seed-20-mini", "bytedance-seed/seed-2.0-mini", 0.10, 0.40),
)
# fmt: on

# Локальные движки: имя → подпапка в research/external_ocr_models/local с pyproject и worker.py.
LOCAL_ENGINES: tuple[ModelSpec, ...] = (
    ModelSpec(
        "local-paddleocr-vl", "PaddlePaddle/PaddleOCR-VL-1.6", 0, 0, json_mode="none", reasoning="none", local=True
    ),
    ModelSpec("local-glm-ocr", "zai-org/GLM-OCR", 0, 0, json_mode="none", reasoning="none", local=True),
    ModelSpec(
        "local-deepseek-ocr2", "deepseek-ai/DeepSeek-OCR-2", 0, 0, json_mode="none", reasoning="none", local=True
    ),
    ModelSpec("local-dots-ocr", "rednote-hilab/dots.ocr", 0, 0, json_mode="none", reasoning="none", local=True),
    ModelSpec("local-marker", "datalab-to/marker", 0, 0, json_mode="none", reasoning="none", local=True),
)

# Состав пробника по умолчанию: дешёвые + средние, без эталонов.
PROBE_MODELS: tuple[str, ...] = (
    "deepseek-v41-flash",
    "gemini-31-flash-lite",
    "qwen38-flash",
    "gpt-56-luna",
    "mistral-small-4",
    "glm-53-flash",
    "gemini-38-flash",
    "qwen3-vl-235b",
    "claude-haiku-45",
)

REGISTRY: dict[str, ModelSpec] = {spec.name: spec for spec in OPENROUTER_MODELS + LOCAL_ENGINES}


def resolve(name: str) -> ModelSpec:
    """Спецификация по короткому имени; неизвестное имя — ошибка со списком известных."""
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"неизвестная модель {name!r}; известны: {known}") from None


def local_engine_dir(spec: ModelSpec) -> str:
    """Имя подпапки локального движка: ``local-paddleocr-vl`` → ``paddleocr_vl``."""
    return spec.name.removeprefix("local-").replace("-", "_")
