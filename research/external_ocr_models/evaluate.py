"""Метрики по выходам моделей: буквы, структура, надёжность, цена.

Истинной разметки полос нет, поэтому буквы меряются двумя прокси: расстоянием до
текстового слоя FineReader (он хорош на буквах, плох на структуре — как раз то, что нам
нужно) и согласием с остальными моделями (ошибка одной модели редко совпадает с ошибкой
другой). Структура меряется счётом разметки и поиском известных фраз из повёрнутых ячеек.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz

_FRONT_MATTER = re.compile(r"^\s*---\s*\n.*?\n---\s*\n", re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_MD_MARKS = re.compile(r"[#*_`>|]+")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$", re.MULTILINE)
_HYPHEN_BREAK = re.compile(r"(\w)[-­]\s*\n\s*(\w)")
_SOFT_HYPHEN = "­"
_SPACES = re.compile(r"\s+")
_NOTE = re.compile(r"\[(картинка|блок-схема|неразборчиво)[^\]]*\]")
_DAMAGE_TAG = re.compile(r"</?(restored|fuzzy|rubric|author|position)>|<unknown\s*/>")


def normalize(text: str) -> str:
    """Свести markdown или текстовый слой PDF к «голому» тексту для сравнения букв."""
    text = _FRONT_MATTER.sub("", text)
    text = _TABLE_RULE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _NOTE.sub(" ", text)
    text = _DAMAGE_TAG.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = text.replace(_SOFT_HYPHEN, "")
    text = _MD_MARKS.sub(" ", text)
    text = text.replace("ё", "е").replace("Ё", "Е")
    return _SPACES.sub(" ", text).strip().lower()


def cer(hypothesis: str, reference: str) -> float | None:
    """Расстояние Левенштейна к длине эталона; None, если эталон пуст."""
    if not reference:
        return None
    return Levenshtein.distance(hypothesis, reference) / len(reference)


@dataclass
class Structure:
    h1: int = 0
    h2: int = 0
    h3: int = 0
    authors: int = 0  # <author> (или абзацы только из **жирного** в старых выходах)
    positions: int = 0  # <position> (или абзацы только из *курсива*)
    rubrics: int = 0  # <rubric>
    tables: int = 0  # GFM + <table>
    html_tables: int = 0
    pictures: int = 0
    schemas: int = 0
    footnotes: int = 0
    unreadable: int = 0
    lists: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


_BOLD_PARA = re.compile(r"^\*\*[^*\n]+\*\*[,.;:]?$")
_ITALIC_PARA = re.compile(r"^\*[^*\n]+\*[,.;:]?$|^_[^_\n]+_[,.;:]?$")


def structure(markdown: str) -> Structure:
    body = _FRONT_MATTER.sub("", markdown)
    counts = Structure()
    # Авторы, должности и рубрики — по тегам (промпт v13+); старые выходы без тегов
    # считаются по абзацам из одного жирного/курсивного фрагмента.
    tagged = {
        name: len(re.findall(rf"<{name}>.*?</{name}>", body, re.DOTALL)) for name in ("rubric", "author", "position")
    }
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("### "):
            counts.h3 += 1
        elif line.startswith("## "):
            counts.h2 += 1
        elif line.startswith("# "):
            counts.h1 += 1
        elif not any(tagged.values()) and _BOLD_PARA.match(line):
            counts.authors += 1
        elif not any(tagged.values()) and _ITALIC_PARA.match(line):
            counts.positions += 1
        elif re.match(r"^\s*([-*+]|\d+[.)])\s+", raw):
            counts.lists += 1
    if any(tagged.values()):
        counts.authors, counts.positions, counts.rubrics = tagged["author"], tagged["position"], tagged["rubric"]
    counts.html_tables = len(re.findall(r"<table\b", body, re.IGNORECASE))
    counts.tables = counts.html_tables + len(_TABLE_RULE.findall(body))
    counts.pictures = len(re.findall(r"\[картинка", body))
    counts.schemas = len(re.findall(r"\[блок-схема", body))
    counts.footnotes = len(re.findall(r"^\[\^[^\]]+\]:", body, re.MULTILINE))
    counts.unreadable = body.count("[неразборчиво]")
    return counts


def repetition_score(text: str, window: int = 40) -> float:
    """Доля повторных окон текста: зациклившаяся модель гонит одни и те же строки."""
    text = _SPACES.sub(" ", text)
    if len(text) < window * 4:
        return 0.0
    chunks = [text[i : i + window] for i in range(0, len(text) - window, window)]
    return 1.0 - len(set(chunks)) / len(chunks)


def phrase_recall(markdown: str, phrases: list[str], threshold: int = 80) -> tuple[int, int]:
    """Сколько эталонных фраз (например, из повёрнутых ячеек) нашлось в тексте — нечётко."""
    haystack = normalize(markdown)
    found = 0
    for phrase in phrases:
        needle = normalize(phrase)
        if needle and fuzz.partial_ratio(needle, haystack) >= threshold:
            found += 1
    return found, len(phrases)


# --- эталоны ---


def finereader_pages(pdf_path: Path) -> list[str]:
    """Текстовый слой каждой страницы PDF FineReader в порядке страниц."""
    import fitz

    with fitz.open(pdf_path) as document:
        return [page.get_text() for page in document]


def rotated_phrases(info_dir: Path, issue_prefix: str, min_conf: float = 0.8) -> dict[str, list[str]]:
    """Фразы уверенно прочитанных повёрнутых ячеек по полосам: ``IMG_0114_2R`` → [...]."""
    phrases: dict[str, list[str]] = {}
    for path in sorted(info_dir.glob(f"{issue_prefix}_*.json")):
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        stem = path.stem[len(issue_prefix) + 1 :]
        page = re.sub(r"_t\d+$", "", stem)
        for cell in info.get("cells") or []:
            text = (cell.get("text") or "").strip()
            if cell.get("rotate_cw") in (90, 270) and text and float(cell.get("conf_tesseract") or 0) >= min_conf:
                phrases.setdefault(page, []).append(text)
    return phrases


# --- сбор по папкам выходов ---


@dataclass
class PageOutput:
    model: str
    page: str  # относительный путь полосы, например 1966/03/IMG_0104_2R.jpg
    meta: dict
    markdown: str | None  # None — сбой
    result: dict | None = None

    @property
    def ok(self) -> bool:
        """Пустое тело — тоже сбой: dots.ocr иногда отдаёт одни рамки без текста."""
        return (
            bool(self.markdown and self.markdown.strip())
            and not self.meta.get("error")
            and not self.meta.get("parse_error")
        )


def load_outputs(model_dir: Path) -> list[PageOutput]:
    outputs: list[PageOutput] = []
    for meta_path in sorted(model_dir.rglob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        base = meta_path.with_name(meta_path.name[: -len(".meta.json")])
        markdown = None
        result = None
        json_path = base.with_suffix(".json")
        md_path = base.with_suffix(".md")
        if json_path.is_file():
            result = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = result.get("content_markdown")
        elif md_path.is_file():
            markdown = _FRONT_MATTER.sub("", md_path.read_text(encoding="utf-8"))
        outputs.append(
            PageOutput(
                model_dir.name, meta.get("page") or base.relative_to(model_dir).as_posix(), meta, markdown, result
            )
        )
    return outputs


@dataclass
class PageScore:
    model: str
    page: str
    ok: bool
    error: str | None
    cer_finereader: float | None
    reference_chars: int  # длина эталона FineReader; 0 — эталона нет
    agreement: float | None  # средний CER до других моделей на той же полосе
    structure: Structure
    rotated_found: int
    rotated_total: int
    repetition: float
    page_number: str | None
    page_number_expected: str | None
    is_toc: bool | None
    truncated: bool
    cost_usd: float | None

    @property
    def page_number_ok(self) -> bool | None:
        """Совпал ли распознанный номер с ожидаемым; None — ожидаемого нет (обложка)."""
        if self.page_number_expected is None:
            return None
        return (self.page_number or "").strip() == self.page_number_expected

    latency_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    chars: int
    tags: dict = field(default_factory=dict)  # режим damage: restored/fuzzy/unknown
    tags_from_edge_words: int = 0
    damage_seen: str = ""


def score_outputs(
    by_model: dict[str, list[PageOutput]],
    reference_text: dict[str, str],
    rotated: dict[str, list[str]],
    expected_numbers: dict[str, str] | None = None,
) -> list[PageScore]:
    """Оценить все выходы.

    ``reference_text`` — нормализованный текст FineReader, ``rotated`` — фразы повёрнутых
    ячеек, ``expected_numbers`` — ожидаемый номер страницы; всё по имени полосы без
    расширения (``IMG_0104_2R``), чтобы пробник (пути с выпуском) и полный прогон (пути
    относительно выпуска) считались одинаково.
    """
    expected_numbers = expected_numbers or {}
    normalized: dict[tuple[str, str], str] = {}
    for model, outputs in by_model.items():
        for output in outputs:
            if output.ok:
                normalized[(model, output.page)] = normalize(output.markdown or "")
    scores: list[PageScore] = []
    for model, outputs in by_model.items():
        for output in outputs:
            page = output.page
            stem = Path(page).stem
            own = normalized.get((model, page))
            others = [text for (other, other_page), text in normalized.items() if other_page == page and other != model]
            agreement = None
            if own is not None and others:
                values = [value for value in (cer(own, other) for other in others) if value is not None]
                agreement = sum(values) / len(values) if values else None
            counts = structure(output.markdown or "") if output.ok else Structure()
            found, total = (
                phrase_recall(output.markdown or "", rotated.get(stem, []))
                if output.ok
                else (0, len(rotated.get(stem, [])))
            )
            meta = output.meta
            scores.append(
                PageScore(
                    model=model,
                    page=page,
                    ok=output.ok,
                    error=meta.get("error") or meta.get("parse_error") or (None if output.ok else "пустой ответ"),
                    cer_finereader=cer(own, reference_text.get(stem, "")) if own is not None else None,
                    reference_chars=len(reference_text.get(stem, "")),
                    agreement=agreement,
                    structure=counts,
                    rotated_found=found,
                    rotated_total=total,
                    repetition=repetition_score(output.markdown or "") if output.ok else 0.0,
                    page_number=(output.result or {}).get("page_number") if output.result else None,
                    page_number_expected=expected_numbers.get(stem),
                    is_toc=(output.result or {}).get("is_toc") if output.result else None,
                    truncated=meta.get("finish_reason") == "length",
                    cost_usd=meta.get("cost_usd"),
                    latency_s=meta.get("latency_s") or meta.get("seconds"),
                    prompt_tokens=meta.get("prompt_tokens"),
                    completion_tokens=meta.get("completion_tokens"),
                    chars=len(output.markdown or ""),
                    tags=meta.get("tags") or {},
                    tags_from_edge_words=int(meta.get("tags_from_edge_words") or 0),
                    damage_seen=meta.get("damage_seen") or "",
                )
            )
    return scores
