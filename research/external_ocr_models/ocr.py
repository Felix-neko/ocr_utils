"""Одна полоса → один запрос к модели → PageResult и .meta.json.

Здесь собирается тело запроса под особенности модели (режим JSON, рассуждения, порядок
провайдеров) и здесь же — запасные ходы: если провайдер не принял строгую схему, тот же
запрос уходит в режиме json_object, потом вообще без response_format.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.external_ocr_models import PROMPT_VERSION
from research.external_ocr_models.client import ChatResponse, OpenRouterClient, OpenRouterError
from research.external_ocr_models.imaging import DEFAULT_MAX_SIDE, DEFAULT_QUALITY, PreparedImage, prepare
from research.external_ocr_models.models import JsonMode, ModelSpec
from research.external_ocr_models.prompts import system_prompt, user_prompt
from research.external_ocr_models.render import to_markdown
from research.external_ocr_models.schema import (
    PageResult,
    ParseError,
    json_schema,
    parse_json_text,
    parse_markdown_text,
    tag_counts,
    tags_from_edge_words,
    unbalanced_tags,
)

logger = logging.getLogger(__name__)

# Потолок выходных токенов. Полоса — 3-6 тыс. знаков ≈ 2-4 тыс. токенов, таблица в HTML
# вдвое больше; 16k оставляет запас, но не даёт зациклившейся модели наговорить на доллар.
DEFAULT_MAX_TOKENS = 16000

# Статусы, после которых имеет смысл повторить запрос в более простом режиме JSON: провайдер
# не понял response_format или маршрутизатор не нашёл провайдера с нужными параметрами.
FALLBACK_STATUSES = frozenset({400, 404, 422})


@dataclass
class RunOptions:
    output_mode: str = "json"  # json | markdown
    strips: int = 1
    max_side: int = DEFAULT_MAX_SIDE
    quality: int = DEFAULT_QUALITY
    reasoning: str | None = None  # переопределение уровня из реестра
    max_tokens: int = DEFAULT_MAX_TOKENS
    write_json: bool = True
    write_md: bool = True
    damage: bool = False  # правило про повреждённые буквы: <restored>, <fuzzy>, <unknown/>
    source: str = ""  # описание издания для промпта («журнал «…», 1966»); пусто — пресса вообще
    hint: str = ""  # подсказка модели про полосу (например, «левый край срезан корешком»)
    hints: dict[str, str] = field(default_factory=dict)  # подсказки по полосам: относительный путь → текст
    damage_side: str = "none"  # auto — по суффиксу _L/_R угадать, какой край у корешка


def reasoning_field(spec: ModelSpec, override: str | None) -> dict | None:
    level = override or spec.reasoning
    if level == "none" or spec.reasoning == "none":
        return None
    if level == "off":
        return {"enabled": False}
    if spec.reasoning_max_tokens:
        # У OpenRouter effort и max_tokens взаимоисключающие; потолок важнее уровня.
        return {"max_tokens": spec.reasoning_max_tokens}
    return {"effort": level}


SIDE_HINTS = {
    "L": "This is the LEFT page of a spread photographed in a tightly bound volume: the RIGHT ends of the lines run "
    "into the binding gutter, so the last letters of many lines may be cut off, squashed or blurred.",
    "R": "This is the RIGHT page of a spread photographed in a tightly bound volume: the LEFT beginnings of the lines run "
    "into the binding gutter, so the first letters of many lines may be cut off, squashed or blurred.",
}


def page_hint(options: RunOptions, rel: Path | None) -> str:
    """Подсказка для полосы: общая ``--hint`` + строка из ``--hints`` + автоподсказка по стороне."""
    parts = [options.hint]
    if rel is not None:
        parts.append(options.hints.get(rel.as_posix(), ""))
        side = rel.stem[-1] if rel.stem[-2:-1] == "_" else ""
        if options.damage_side == "auto":
            parts.append(SIDE_HINTS.get(side, ""))
        elif options.damage_side in ("L", "R"):
            parts.append(SIDE_HINTS[options.damage_side])
    return " ".join(part.strip() for part in parts if part and part.strip())


def build_payload(
    spec: ModelSpec,
    images: list[PreparedImage],
    options: RunOptions,
    json_mode: JsonMode,
    skip_reasoning: bool = False,
    rel: Path | None = None,
) -> dict:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": user_prompt(len(images), page_hint(options, rel), options.damage)}
    ]
    for image in images:
        part: dict[str, Any] = {"url": image.data_url()}
        if spec.image_detail:
            part["detail"] = spec.image_detail
        content.append({"type": "image_url", "image_url": part})
    payload: dict[str, Any] = {
        "model": spec.openrouter_id,
        "messages": [
            {"role": "system", "content": system_prompt(options.output_mode, options.damage, options.source)},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": options.max_tokens,
    }
    provider: dict[str, Any] = {}
    if spec.provider_order:
        provider["order"] = list(spec.provider_order)
        provider["allow_fallbacks"] = True
    if spec.provider_ignore:
        provider["ignore"] = list(spec.provider_ignore)
    if options.output_mode == "json":
        if json_mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "page_transcription", "strict": True, "schema": json_schema(options.damage)},
            }
            provider["require_parameters"] = True
        elif json_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
    reasoning = reasoning_field(spec, options.reasoning) if not skip_reasoning else None
    if reasoning is not None:
        payload["reasoning"] = reasoning
    if provider:
        payload["provider"] = provider
    return payload


def _fallback_chain(spec: ModelSpec, options: RunOptions) -> list[JsonMode]:
    if options.output_mode != "json":
        return ["none"]
    chain: list[JsonMode] = ["json_schema", "json_object", "none"]
    return chain[chain.index(spec.json_mode) :]


def parse_response(text: str, output_mode: str) -> PageResult:
    return parse_json_text(text) if output_mode == "json" else parse_markdown_text(text)


def output_paths(out_dir: Path, rel: Path) -> dict[str, Path]:
    base = out_dir / rel.with_suffix("")
    return {
        "json": base.with_suffix(".json"),
        "md": base.with_suffix(".md"),
        "meta": base.with_suffix(".meta.json"),
        "raw": base.with_suffix(".raw.txt"),
    }


def is_done(out_dir: Path, rel: Path) -> bool:
    """Сделано = есть .meta.json без ошибки. Сбойные страницы при --skip-done идут заново."""
    meta_path = output_paths(out_dir, rel)["meta"]
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return not meta.get("error") and not meta.get("parse_error")


def recognize_page(
    client: OpenRouterClient, spec: ModelSpec, in_path: Path, rel: Path, out_dir: Path, options: RunOptions
) -> dict:
    """Распознать одну полосу и записать выходы. Возвращает meta (он же строка сводки)."""
    paths = output_paths(out_dir, rel)
    paths["meta"].parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "page": rel.as_posix(),
        "model": spec.name,
        "openrouter_id": spec.openrouter_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION,
        "output_mode": options.output_mode,
        "strips": options.strips,
        "max_side": options.max_side,
        "reasoning": options.reasoning or spec.reasoning,
        "error": None,
        "parse_error": None,
    }
    started = time.monotonic()
    try:
        images = prepare(in_path, options.max_side, options.quality, strips=options.strips)
    except Exception as error:  # битый файл — не повод ронять прогон
        meta["error"] = f"картинка: {error}"
        _write_meta(paths["meta"], meta)
        return meta
    meta["image_px"] = [[image.width, image.height] for image in images]
    meta["image_bytes"] = sum(len(image.data) for image in images)

    response: ChatResponse | None = None
    skip_reasoning = False
    for json_mode in _fallback_chain(spec, options):
        payload = build_payload(spec, images, options, json_mode, skip_reasoning, rel)
        try:
            response = client.chat(payload)
        except OpenRouterError as error:
            meta.setdefault("fallbacks", []).append(
                {"json_mode": json_mode, "error": str(error), "body": error.body[:300]}
            )
            if error.status in FALLBACK_STATUSES and "reasoning" in payload and "reasoning" in error.body.lower():
                # Провайдер не понял параметр рассуждений — повторяем тот же режим без него.
                logger.warning("%s %s: параметр reasoning отвергнут (%s), повторяю без него", spec.name, rel, error)
                skip_reasoning = True
                try:
                    response = client.chat(build_payload(spec, images, options, json_mode, True, rel))
                except OpenRouterError as again:
                    meta["fallbacks"].append({"json_mode": json_mode, "error": str(again), "body": again.body[:300]})
                    error = again
                else:
                    meta["json_mode_used"] = json_mode
                    break
            if error.status in FALLBACK_STATUSES and json_mode != "none":
                logger.warning("%s %s: режим %s отвергнут (%s), пробую проще", spec.name, rel, json_mode, error)
                continue
            meta["error"] = f"{error} {error.body[:300]}".strip()
            break
        meta["json_mode_used"] = json_mode
        break
    meta["reasoning_sent"] = not skip_reasoning

    meta["seconds"] = round(time.monotonic() - started, 2)
    if response is None:
        meta.setdefault("error", "запрос не удался")
        _write_meta(paths["meta"], meta)
        return meta

    meta.update(
        provider=response.provider,
        served_model=response.model,
        request_id=response.request_id,
        finish_reason=response.finish_reason,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        reasoning_tokens=response.reasoning_tokens,
        cost_usd=response.cost_usd,
        latency_s=round(response.latency_s, 2),
        attempts=response.attempts,
        response_chars=len(response.text),
    )
    try:
        result = parse_response(response.text, options.output_mode)
    except ParseError as error:
        meta["parse_error"] = str(error)
        paths["raw"].write_text(response.text, encoding="utf-8")
        _write_meta(paths["meta"], meta)
        return meta
    if options.damage and result.edge_words:
        # Модель охотнее заполняет список повреждённых строк, чем ставит теги в тексте:
        # доставляем пометки из списка туда, где их нет.
        result.content_markdown, inserted = tags_from_edge_words(result.content_markdown, result.edge_words)
        meta["tags_from_edge_words"] = inserted
    if options.write_json:
        paths["json"].write_text(result.to_json(), encoding="utf-8")
    if options.write_md:
        paths["md"].write_text(to_markdown(result), encoding="utf-8")
    meta.update(
        page_number=result.page_number,
        is_toc=result.is_toc,
        content_chars=len(result.content_markdown),
        has_header=result.running_header is not None,
    )
    if options.damage:
        meta["tags"] = tag_counts(result.content_markdown)
        meta["damage_seen"] = result.damage
        broken = unbalanced_tags(result.content_markdown)
        if broken:
            meta["tag_warning"] = "непарные теги: " + ", ".join(broken)
    _write_meta(paths["meta"], meta)
    return meta


def _write_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
