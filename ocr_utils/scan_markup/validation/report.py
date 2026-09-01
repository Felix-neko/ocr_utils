"""Отчёт по валидационному прогону: консоль, markdown и CSV.

CSV здесь — самое ценное. В нём по строке на область со всеми измерениями, и следующую
итерацию порогов можно откатать вообще без чтения оригиналов: сорок мегабайт таблицы против
четырёх гигабайт TIFF.
"""

import csv
from pathlib import Path

from ocr_utils.scan_markup.validation.cases import DEFECTS
from ocr_utils.scan_markup.validation.run import ValidateReport

CSV_COLUMNS = (
    "defect",
    "rel_path",
    "fixed",
    "regions",
    "x1",
    "y1",
    "x2",
    "y2",
    "kind",
    "full_page",
    "chroma_frac",
    "chroma_spread",
    "chroma_self_frac",
    "dot_frac",
    "mid_frac",
    "tone_entropy",
    "screen_peak",
    "ink_contrast",
)


def console_lines(report: ValidateReport) -> list[str]:
    """Таблица «дефект | всего | починено | осталось» плюс перечень непочиненных."""
    grouped = report.by_defect()
    lines = [f"{'дефект':16s} {'всего':>6s} {'починено':>9s} {'осталось':>9s}   ожидание"]
    scored_total = scored_fixed = 0
    for defect in DEFECTS:
        outcomes = grouped.get(defect.key, [])
        if not outcomes:
            continue
        fixed = sum(1 for outcome in outcomes if outcome.fixed)
        if defect.scored:
            scored_total += len(outcomes)
            scored_fixed += fixed
        mark = "" if defect.scored else "  (вне зачёта)"
        lines.append(
            f"{defect.key:16s} {len(outcomes):6d} {fixed:9d} {len(outcomes) - fixed:9d}   {defect.expectation}{mark}"
        )
    lines.append(f"{'ИТОГО':16s} {scored_total:6d} {scored_fixed:9d} {scored_total - scored_fixed:9d}")

    remaining = [o for o in report.outcomes if not o.fixed and o.case.defect.scored]
    if remaining:
        lines.append("")
        lines.append("Осталось не починено:")
        for outcome in remaining:
            lines.append(f"  {outcome.case.defect.key:16s} {outcome.case.rel_path:30s} {outcome.note}")

    if report.control:
        lost = [c for c in report.control if c.now == 0]
        lines.append("")
        lines.append(
            f"Контроль: полос {len(report.control)}, область пропала у {len(lost)}. "
            "Пропажу надо смотреть глазами: часть контрольных срабатываний была ошибками прошлого детектора."
        )
        for item in lost:
            lines.append(f"  {item.rel_path:30s} было областей {item.was}, стало 0")

    if report.notes:
        lines.append("")
        lines.append("Замечания по выборке:")
        lines.extend(f"  {note}" for note in report.notes)
    return lines


def write_csv(report: ValidateReport, path: Path) -> None:
    """По строке на область; полоса без областей даёт одну строку с пустыми координатами."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for outcome in report.outcomes:
            base = {
                "defect": outcome.case.defect.key,
                "rel_path": outcome.case.rel_path,
                "fixed": int(outcome.fixed),
                "regions": len(outcome.result.regions),
            }
            if not outcome.result.regions:
                writer.writerow(base)
                continue
            for region in outcome.result.regions:
                row = dict(base)
                row.update(
                    x1=region.box[0],
                    y1=region.box[1],
                    x2=region.box[2],
                    y2=region.box[3],
                    kind=region.kind,
                    full_page=int(region.full_page),
                    chroma_frac=region.chroma_frac,
                    chroma_spread=region.chroma_spread,
                    chroma_self_frac=region.chroma_self_frac,
                    dot_frac=region.dot_frac,
                    mid_frac=region.mid_frac,
                    tone_entropy=region.tone_entropy,
                    screen_peak=region.screen_peak,
                    ink_contrast=region.ink_contrast,
                )
                writer.writerow(row)


def write_markdown(report: ValidateReport, path: Path, out_dir: Path | None) -> None:
    """Отчёт со ссылками на оверлеи — чтобы спорные случаи открывались одним щелчком."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Валидация детекции растровых областей", "", "```"]
    lines.extend(console_lines(report))
    lines.append("```")

    remaining = [o for o in report.outcomes if not o.fixed and o.case.defect.scored]
    if remaining and out_dir is not None:
        lines += ["", "## Непочиненные полосы", ""]
        for outcome in remaining:
            overlay = out_dir / outcome.case.defect.key / "осталось" / f"{outcome.case.rel_path.replace('/', '__')}.jpg"
            lines.append(f"* [{outcome.case.rel_path}]({overlay.as_posix()}) — {outcome.note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
