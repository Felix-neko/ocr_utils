"""Безэталонная оценка выпрямления: метрики кривизны до и после.

Эталона «как должно быть» нет, зато есть детекторы кривизны из
``scan_markup.curved_lines``: карта локальных углов (``skew_map``) и аппроксимации строк
(``line_fit``). Если движок выпрямил полосу, разброс углов и прогибы строк на результате
должны упасть; если сломал — вырасти. Это ровно те величины, которыми в литературе по
dewarp оценивают результат без эталона (прямизна и параллельность строк).

Оговорка: метрики считаются той же классикой, что и детектор, и у них те же слепые пятна
(таблицы, заголовки вразрядку). Сравнивать надо «до/после» на одной полосе, а не
абсолютные значения между полосами.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ocr_utils.scan_markup.curved_lines.detectors import line_fit, skew_map
from ocr_utils.scan_markup.curved_lines.detectors.base import Frame

# (детектор, метрика) — что сравнивается до/после.
METRICS: tuple[tuple[str, str], ...] = (
    ("skew_map", "max_dev_deg"),
    ("skew_map", "resid_deg"),
    ("line_fit", "sagitta_rel_p90"),
    ("line_fit", "sagitta_rel_max3"),
    ("line_fit", "slope_spread_deg"),
    ("line_fit", "slope_resid_deg"),
)


def frame_from_bgr(img: np.ndarray, dpi: int, rel_path: str = "") -> Frame:
    """Кадр детекторов из BGR-массива известного разрешения."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    height, width = gray.shape

    def scaled(target: int) -> np.ndarray:
        factor = target / dpi
        size = (max(1, round(width * factor)), max(1, round(height * factor)))
        return cv2.resize(gray, size, interpolation=cv2.INTER_AREA) if size != (width, height) else gray

    gray300 = scaled(300)
    gray150 = cv2.resize(
        gray300, (max(1, gray300.shape[1] // 2), max(1, gray300.shape[0] // 2)), interpolation=cv2.INTER_AREA
    )
    return Frame(
        rel_path=rel_path, path=Path(rel_path), width=width, height=height, dpi=dpi, gray150=gray150, gray300=gray300
    )


def measure(img: np.ndarray, dpi: int) -> dict[str, float]:
    """Метрики кривизны одного кадра, ключи ``детектор.метрика``."""
    frame = frame_from_bgr(img, dpi)
    out: dict[str, float] = {}
    for module, name in ((skew_map, "skew_map"), (line_fit, "line_fit")):
        result = module.measure(frame, False)
        for key, value in result.metrics.items():
            out[f"{name}.{key}"] = float(value)
    return out


@dataclass
class QualityRow:
    page: str
    engine: str
    ok: bool
    seconds: float
    note: str = ""
    before: dict[str, float] = field(default_factory=dict)
    after: dict[str, float] = field(default_factory=dict)


def header() -> list[str]:
    columns = ["page", "engine", "ok", "seconds", "note"]
    for detector, metric in METRICS:
        columns += [f"{detector}.{metric}_before", f"{detector}.{metric}_after"]
    return columns


def write_csv(path: Path, rows: Sequence[QualityRow]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header())
        for row in rows:
            record = [row.page, row.engine, "1" if row.ok else "0", f"{row.seconds:.2f}", row.note]
            for detector, metric in METRICS:
                key = f"{detector}.{metric}"
                record += [
                    f"{row.before[key]:.3f}" if key in row.before else "",
                    f"{row.after[key]:.3f}" if key in row.after else "",
                ]
            writer.writerow(record)


def markdown_table(rows: Sequence[QualityRow]) -> str:
    """Таблица «до → после» по полосам и движкам плюс сводка по движкам (медиана изменения)."""
    lines = ["| полоса | движок | с | " + " | ".join(f"{d}.{m}" for d, m in METRICS) + " |"]
    lines.append("|---|---|---|" + "---|" * len(METRICS))
    for row in sorted(rows, key=lambda r: (r.page, r.engine)):
        cells = []
        for detector, metric in METRICS:
            key = f"{detector}.{metric}"
            if key in row.before and key in row.after:
                cells.append(f"{row.before[key]:.2f} → {row.after[key]:.2f}")
            elif not row.ok:
                cells.append("сбой")
            else:
                cells.append("—")
        lines.append(f"| {row.page} | {row.engine} | {row.seconds:.1f} | " + " | ".join(cells) + " |")

    lines += ["", "Сводка по движкам: медиана отношения «после / до» (меньше 1 — стало ровнее).", ""]
    lines.append("| движок | полос | сбоев | медиана с | " + " | ".join(f"{d}.{m}" for d, m in METRICS) + " |")
    lines.append("|---|---|---|---|" + "---|" * len(METRICS))
    engines = sorted({row.engine for row in rows})
    for engine in engines:
        mine = [row for row in rows if row.engine == engine]
        failed = sum(1 for row in mine if not row.ok)
        seconds = float(np.median([row.seconds for row in mine])) if mine else 0.0
        cells = []
        for detector, metric in METRICS:
            key = f"{detector}.{metric}"
            ratios = [
                row.after[key] / row.before[key]
                for row in mine
                if key in row.before and key in row.after and row.before[key] > 1e-6
            ]
            cells.append(f"{np.median(ratios):.2f}" if ratios else "—")
        lines.append(f"| {engine} | {len(mine)} | {failed} | {seconds:.1f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
