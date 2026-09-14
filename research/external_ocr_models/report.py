"""Сводные таблицы по оценкам: по моделям и по полосам. Только читает готовые выходы."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Sequence

from research.external_ocr_models.evaluate import PageScore

# Число полос в выпуске и в паке — для экстраполяции стоимости.
PAGES_PER_ISSUE = 97
PAGES_PER_PACK = 12135


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Простая markdown-таблица без выравнивания по ширине: её читают в готовом виде."""
    lines = ["| " + " | ".join(str(cell) for cell in header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _fmt(value: float | None, digits: int = 3, scale: float = 1.0) -> str:
    return "—" if value is None else f"{value * scale:.{digits}f}"


def _avg(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def _ratio(flags: list[bool | None]) -> str:
    known = [flag for flag in flags if flag is not None]
    return f"{sum(known)}/{len(known)}" if known else "—"


def _median(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return median(clean) if clean else None


# Полоса считается текстовой, если у FineReader на ней не меньше стольких знаков: на
# обложке и рекламных полосах эталона почти нет, и CER там говорит об эталоне, а не о модели.
TEXT_PAGE_MIN_CHARS = 300


MODEL_HEADER = (
    "модель",
    "полос",
    "сбоев",
    "обрезано",
    "CER к FR (медиана)",
    "CER к FR (текст)",
    "согласие",
    "h1/h2/h3",
    "авторов",
    "должн.",
    "рубрик",
    "таблиц",
    "повёрн.",
    "схем",
    "карт.",
    "зацикл.",
    "№ стр. верно",
    "с/полоса",
    "¢/полоса",
    "¢/выпуск",
    "$/пак",
)


def model_rows(scores: list[PageScore]) -> list[list[object]]:
    by_model: dict[str, list[PageScore]] = defaultdict(list)
    for score in scores:
        by_model[score.model].append(score)
    rows: list[list[object]] = []
    for model, items in sorted(by_model.items()):
        ok = [item for item in items if item.ok]
        costs = [item.cost_usd for item in items if item.cost_usd is not None]
        cost_page = mean(costs) if costs else None
        rotated_found = sum(item.rotated_found for item in ok)
        rotated_total = sum(item.rotated_total for item in items)
        rows.append(
            [
                model,
                len(items),
                len(items) - len(ok),
                sum(item.truncated for item in items),
                _fmt(_median([item.cer_finereader for item in ok])),
                _fmt(_avg([item.cer_finereader for item in ok if item.reference_chars >= TEXT_PAGE_MIN_CHARS])),
                _fmt(_avg([item.agreement for item in ok])),
                f"{sum(i.structure.h1 for i in ok)}/{sum(i.structure.h2 for i in ok)}/{sum(i.structure.h3 for i in ok)}",
                sum(item.structure.authors for item in ok),
                sum(item.structure.positions for item in ok),
                sum(item.structure.rubrics for item in ok),
                sum(item.structure.tables for item in ok),
                f"{rotated_found}/{rotated_total}" if rotated_total else "—",
                sum(item.structure.schemas for item in ok),
                sum(item.structure.pictures for item in ok),
                sum(item.repetition > 0.3 for item in ok),
                _ratio([item.page_number_ok for item in ok]),
                _fmt(_avg([item.latency_s for item in ok]), 0),
                _fmt(cost_page, 3, 100),
                _fmt(cost_page, 1, 100 * PAGES_PER_ISSUE),
                _fmt(cost_page, 2, PAGES_PER_PACK),
            ]
        )
    return rows


PAGE_HEADER = (
    "полоса",
    "модель",
    "ок",
    "CER к FR",
    "согласие",
    "h1/h2/h3",
    "авт/должн",
    "таблиц",
    "повёрн.",
    "схем",
    "карт.",
    "№ стр.",
    "оглавл.",
    "знаков",
    "¢",
    "с",
)


def _page_number_cell(score: PageScore) -> str:
    if not score.page_number:
        return "—"
    if score.page_number_ok is None:
        return score.page_number
    return (
        score.page_number if score.page_number_ok else f"**{score.page_number}** (ожид. {score.page_number_expected})"
    )


def page_rows(scores: list[PageScore]) -> list[list[object]]:
    rows: list[list[object]] = []
    for score in sorted(scores, key=lambda item: (item.page, item.model)):
        counts = score.structure
        rows.append(
            [
                Path(score.page).stem,
                score.model,
                "ок" if score.ok else f"сбой: {(score.error or '')[:60]}",
                _fmt(score.cer_finereader),
                _fmt(score.agreement),
                f"{counts.h1}/{counts.h2}/{counts.h3}",
                f"{counts.authors}/{counts.positions}",
                counts.tables,
                f"{score.rotated_found}/{score.rotated_total}" if score.rotated_total else "—",
                counts.schemas,
                counts.pictures,
                _page_number_cell(score),
                "да" if score.is_toc else "",
                score.chars,
                _fmt(score.cost_usd, 3, 100),
                _fmt(score.latency_s, 0),
            ]
        )
    return rows


def write_scores_csv(scores: list[PageScore], path: Path) -> None:
    fields = [
        "page", "model", "ok", "error", "cer_finereader", "agreement", "h1", "h2", "h3", "authors", "positions",
        "rubrics", "tables", "html_tables", "rotated_found", "rotated_total", "schemas", "pictures", "footnotes", "unreadable", "lists",
        "repetition", "page_number", "page_number_expected", "is_toc", "truncated", "cost_usd", "latency_s", "prompt_tokens",
        "completion_tokens", "chars",
    ]  # fmt: skip
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for score in scores:
            row = {name: getattr(score, name) for name in fields if hasattr(score, name)}
            row.update(score.structure.as_dict())
            writer.writerow(row)


DAMAGE_HEADER = ("полоса", "модель", "restored", "fuzzy", "unknown", "из списка", "непарных", "что видит модель")


def damage_rows(scores: list[PageScore]) -> list[list[object]]:
    rows: list[list[object]] = []
    for score in sorted(scores, key=lambda item: (item.page, item.model)):
        if not score.tags:
            continue
        tags = score.tags
        rows.append(
            [
                Path(score.page).stem,
                score.model,
                tags.get("restored", 0),
                tags.get("fuzzy", 0),
                tags.get("unknown", 0),
                score.tags_from_edge_words,
                "да" if score.error and "непарные" in score.error else "",
                score.damage_seen.replace("|", "/")[:160],
            ]
        )
    return rows


def build_report(scores: list[PageScore], title: str) -> str:
    parts = [f"# {title}", "", "## По моделям", "", markdown_table(MODEL_HEADER, model_rows(scores)), ""]
    damage = damage_rows(scores)
    if damage:
        parts += ["## Повреждения: пометки модели", "", markdown_table(DAMAGE_HEADER, damage), ""]
    parts += [
        "CER к FR — расстояние Левенштейна от текста модели до текстового слоя FineReader, делённое на длину последнего "
        "(оба нормализованы: без разметки, переносов, регистра); медиана по всем полосам и среднее по полосам, где у "
        f"FineReader не меньше {TEXT_PAGE_MIN_CHARS} знаков. Согласие — средний CER до остальных моделей на той же полосе. "
        f"¢/выпуск — из расчёта {PAGES_PER_ISSUE} полос, $/пак — {PAGES_PER_PACK} полос.",
        "",
        "## По полосам",
        "",
        markdown_table(PAGE_HEADER, page_rows(scores)),
        "",
    ]
    return "\n".join(parts)
