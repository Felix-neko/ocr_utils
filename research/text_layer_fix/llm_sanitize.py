"""Сравнение: языковая модель вычёркивает мусор в строках слоя против геометрического вердикта.

Идея пользователя как одна из альтернатив «санации»: отдать строку текстового слоя
(строка дерева структуры FineReader) модели и попросить назвать бессмысленные токены.
Здесь она проверяется как ЭТАЛОН СРАВНЕНИЯ, не как рабочий путь: считается согласие
модели с геометрией по токенам, цена и время. Модель и клиент — из
``external_ocr_services`` (DeepSeek V4.1 Flash через OpenRouter), текст без картинок.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from ocr_utils.external_ocr_services.client import OpenRouterClient
from ocr_utils.external_ocr_services.models import ModelSpec

from research.text_layer_fix.classify import Verdict

SYSTEM_PROMPT = (
    "You review lines of OCR output from a Russian technical journal of the 1960s-1970s. "
    "Some tokens are garbage: the OCR engine read rotated text (table headers printed sideways, "
    "labels on drawings) as random horizontal characters, for example 'X', '^', 'goo', 'i\"o=', 'Е Е я'. "
    "Genuine tokens are Russian or Latin words, abbreviations (шт., т/сут, кг), numbers, punctuation and formula fragments. "
    'Answer with JSON only: {"garbage": [zero-based indices of garbage tokens]}. When unsure, keep the token.'
)

# Токенов в одном запросе, не больше: строки дерева FineReader короткие, но сгруппированные запросы дешевле.
MAX_TOKENS_PER_REQUEST = 60


@dataclass
class LineCase:
    """Одна строка слоя: токены и геометрический вердикт по каждому."""

    pdf: str
    page: int
    struct_line: str
    tokens: list[str]
    geometric_garbage: list[bool]  # DELETE/SANITIZE/SUSPECT → True
    verdicts: list[str]
    llm_garbage: list[bool] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str = ""


def lines_from_payload(payload: dict, only_flagged: bool = True) -> list[LineCase]:
    """Строки дерева структуры страницы с токенами по словам слоя (в порядке слева направо).

    Args:
        payload: JSON страницы из кэша прогона.
        only_flagged: Брать только строки, где есть хотя бы одно слово не KEEP.

    Returns:
        Список строк.
    """
    groups: dict[str, list[dict]] = {}
    for word in payload.get("words", []):
        line = word.get("struct_line")
        if not line:
            continue
        groups.setdefault(line, []).append(word)
    cases: list[LineCase] = []
    for line, words in groups.items():
        words = sorted(words, key=lambda w: (w["bbox_px"][0], w["bbox_px"][1]))
        flagged = [w["verdict"] in (Verdict.DELETE.value, Verdict.SANITIZE.value, Verdict.SUSPECT.value) for w in words]
        if only_flagged and not any(flagged):
            continue
        cases.append(
            LineCase(
                payload["pdf"],
                payload["page"],
                line,
                [w["text"] for w in words],
                flagged,
                [w["verdict"] for w in words],
            )
        )
    return cases


def build_payload(spec: ModelSpec, tokens: list[str], max_tokens: int = 300) -> dict:
    """Тело запроса: система + пронумерованные токены строки.

    Args:
        spec: Модель из реестра ``external_ocr_services``.
        tokens: Токены строки.
        max_tokens: Потолок ответа.

    Returns:
        Payload в форме HTTP API OpenRouter.
    """
    numbered = "\n".join(f"{i}: {token}" for i, token in enumerate(tokens))
    payload: dict = {
        "model": spec.openrouter_id,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Tokens:\n{numbered}"}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": "none"},
    }
    provider: dict = {}
    if spec.provider_order:
        provider["order"], provider["allow_fallbacks"] = list(spec.provider_order), True
    if spec.provider_ignore:
        provider["ignore"] = list(spec.provider_ignore)
    if provider:
        payload["provider"] = provider
    return payload


def parse_answer(text: str, count: int) -> list[bool]:
    """Индексы мусора из ответа модели → маска по токенам (кривой ответ — всё «не мусор»)."""
    try:
        data = json.loads(text)
        indices = {int(i) for i in data.get("garbage", []) if 0 <= int(i) < count}
    except (ValueError, TypeError, AttributeError):
        indices = set()
    return [i in indices for i in range(count)]


def ask(client: OpenRouterClient, spec: ModelSpec, case: LineCase) -> LineCase:
    """Спросить модель про одну строку и записать ответ в неё.

    Args:
        client: Клиент OpenRouter.
        spec: Модель.
        case: Строка.

    Returns:
        Та же строка с заполненными ``llm_garbage``, ценой и временем.
    """
    started = time.time()
    try:
        response = client.chat(build_payload(spec, case.tokens[:MAX_TOKENS_PER_REQUEST]))
        case.llm_garbage = parse_answer(response.text, len(case.tokens))
        case.cost_usd = float(response.cost_usd or 0.0)
    except Exception as error:  # noqa: BLE001 — одна строка не должна валить сравнение
        case.error = f"{type(error).__name__}: {error}"
        case.llm_garbage = [False] * len(case.tokens)
    case.latency_s = time.time() - started
    return case


@dataclass
class Agreement:
    """Согласие модели с геометрией по токенам."""

    both_garbage: int = 0
    geometry_only: int = 0
    llm_only: int = 0
    both_clean: int = 0
    lines: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    errors: int = 0

    def add(self, case: LineCase) -> None:
        self.lines += 1
        self.cost_usd += case.cost_usd
        self.seconds += case.latency_s
        if case.error:
            self.errors += 1
            return
        for geometric, llm in zip(case.geometric_garbage, case.llm_garbage):
            if geometric and llm:
                self.both_garbage += 1
            elif geometric:
                self.geometry_only += 1
            elif llm:
                self.llm_only += 1
            else:
                self.both_clean += 1

    def to_json(self) -> dict:
        total = self.both_garbage + self.geometry_only + self.llm_only + self.both_clean
        return {
            "lines": self.lines,
            "tokens": total,
            "both_garbage": self.both_garbage,
            "geometry_only": self.geometry_only,
            "llm_only": self.llm_only,
            "both_clean": self.both_clean,
            "agreement": round((self.both_garbage + self.both_clean) / total, 3) if total else 0.0,
            "cost_usd": round(self.cost_usd, 4),
            "seconds_per_line": round(self.seconds / self.lines, 2) if self.lines else 0.0,
            "errors": self.errors,
        }
