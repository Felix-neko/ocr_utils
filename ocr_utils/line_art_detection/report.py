"""Отчёты детектора: CSV для калибровки, markdown и консольная таблица для чтения.

CSV ЗДЕСЬ — НЕ ОТЧЁТ, А ПРОМЕЖУТОЧНЫЙ ФОРМАТ. В нём лежат ВСЕ страницы со всеми
признаками и с уже найденными прямоугольниками, поэтому команда ``export`` не считает
ничего заново: порог применяется к готовому столбцу. Ради этого прогон и разбит на две
команды — пороги заказчик хочет пробовать разные, а тяжёлый проход по девяти тысячам
страниц ради каждого повторять незачем.
"""

import csv
from pathlib import Path

from ocr_utils.line_art_detection.analysis import PageResult

FIELDS = (
    "pdf",
    "issue",
    "page",
    "dpi",
    "width",
    "height",
    "coverage",
    "n_candidates",
    "n_components",
    "max_cc_area",
    "full_page",
    "sources",
    "boxes",
    "dropped",
    "status",
)


def _pack_boxes(boxes) -> str:
    """Прямоугольники -> одна строка ``x1 y1 x2 y2;x1 y1 x2 y2``."""
    return ";".join(" ".join(str(int(v)) for v in box) for box in boxes)


def _unpack_boxes(text: str) -> list:
    """Обратно к списку четвёрок; пустая строка — пустой список."""
    return [tuple(int(v) for v in chunk.split()) for chunk in text.split(";") if chunk.strip()]


def _pack_dropped(dropped: dict) -> str:
    return ";".join(f"{name}={count}" for name, count in sorted(dropped.items()))


def write_csv(path: Path, results: list[PageResult]) -> None:
    """Полный CSV по всем страницам."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for r in results:
            writer.writerow(
                [
                    r.pdf,
                    r.issue,
                    r.page_no,
                    r.dpi,
                    r.width,
                    r.height,
                    f"{r.coverage:.6f}",
                    r.n_candidates,
                    r.n_components,
                    r.max_cc_area,
                    int(r.full_page),
                    r.sources,
                    _pack_boxes(r.boxes),
                    _pack_dropped(r.dropped),
                    r.status,
                ]
            )


def read_csv(path: Path) -> list[PageResult]:
    """Читает CSV обратно в :class:`PageResult` — этим живёт команда ``export``."""
    results = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            results.append(
                PageResult(
                    pdf=row["pdf"],
                    page_no=int(row["page"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    dpi=int(row["dpi"]),
                    coverage=float(row["coverage"]),
                    n_candidates=int(row["n_candidates"]),
                    n_components=int(row["n_components"]),
                    max_cc_area=int(row["max_cc_area"]),
                    full_page=bool(int(row["full_page"])),
                    sources=row["sources"],
                    boxes=_unpack_boxes(row["boxes"]),
                    status=row["status"],
                )
            )
    return results


def console_table(results: list[PageResult], limit: int = 40) -> str:
    """Худшие (самые «картиночные») страницы — списком."""
    shown = [r for r in results if r.status == "ok"]
    shown.sort(key=lambda r: -r.coverage)
    lines = [f"{'выпуск':<18} {'стр':>5} {'покрытие':>9} {'канд':>5}  источники"]
    lines.append("-" * 62)
    for r in shown[:limit]:
        lines.append(f"{r.issue:<18} {r.page_no:>5} {r.coverage:>9.4f} {r.n_candidates:>5}  {r.sources}")
    if len(shown) > limit:
        lines.append(f"... ещё {len(shown) - limit}, полный список в CSV")
    return "\n".join(lines)


def summary(results: list[PageResult], thresholds=(0.02, 0.05, 0.15)) -> str:
    """Сколько страниц перешагивает каждый из порогов — по этому выбирается рабочий."""
    ok = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status != "ok"]
    lines = [f"Проанализировано страниц: {len(ok)} (пропущено: {len(skipped)})"]
    for name, count in sorted({r.status: 0 for r in skipped}.items()):
        lines.append(f"  пропущено «{name}»: {sum(1 for r in skipped if r.status == name)}")
    for t in thresholds:
        hit = [r for r in ok if r.coverage >= t]
        share = 100.0 * len(hit) / len(ok) if ok else 0.0
        lines.append(f"  покрытие >= {t:.0%}: {len(hit)} страниц ({share:.1f}%)")
    return "\n".join(lines)


def markdown_report(results: list[PageResult], threshold: float, limit: int = 200) -> str:
    """Markdown-отчёт: сводка по порогам и таблица найденных страниц по выпускам."""
    ok = [r for r in results if r.status == "ok"]
    hit = sorted((r for r in ok if r.coverage >= threshold), key=lambda r: (-r.coverage, r.issue, r.page_no))

    lines = ["# Страницы с крупным line art", ""]
    lines.append(f"Порог покрытия — **{threshold:.0%}** площади полосы.")
    lines.append("")
    lines.append("```")
    lines.append(summary(results))
    lines.append("```")
    lines.append("")
    lines.append(f"## Страницы выше порога ({len(hit)})")
    lines.append("")
    lines.append("| выпуск | стр. | покрытие | кандидатов | источники |")
    lines.append("|---|---:|---:|---:|---|")
    for r in hit[:limit]:
        lines.append(f"| {r.issue} | {r.page_no} | {r.coverage:.1%} | {r.n_candidates} | {r.sources} |")
    if len(hit) > limit:
        lines.append(f"\nПоказаны первые {limit} из {len(hit)}; полный список — в CSV.")

    by_issue: dict[str, int] = {}
    for r in hit:
        by_issue[r.issue] = by_issue.get(r.issue, 0) + 1
    lines.append("")
    lines.append("## Сколько страниц выше порога в каждом выпуске")
    lines.append("")
    lines.append("| выпуск | страниц |")
    lines.append("|---|---:|")
    for issue in sorted(by_issue):
        lines.append(f"| {issue} | {by_issue[issue]} |")
    return "\n".join(lines) + "\n"
