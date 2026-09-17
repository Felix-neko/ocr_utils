"""Строки текста попарно: наклон и волнистость каждой строки «было | стало».

Сводки кривизны по странице (``fitting.page_stats``) FineReader улучшает почти всегда —
это его прямая работа. Порча видна на ОТДЕЛЬНЫХ строках: крупный заголовок, повёрнутый
не в ту сторону (1967/03 с.36), заголовок в разрядку, у которого буквы запрыгали
(1967/03 с.11). Поэтому строки сопоставляются одна к одной через поле смещений, и берётся
худшая разность по странице; сводки считаются тоже — как контекст «что выиграли».
"""

from __future__ import annotations

import numpy as np

from ocr_utils.scan_markup.curved_lines.fitting import page_stats
from research.geometry_regression import WORK_DPI, mm_to_px, px_to_mm
from research.geometry_regression.field import Field
from research.geometry_regression.regions import TextLine

# Сопоставление: допуск по центру строки в долях её высоты (не меньше 0.7 мм) и минимальное
# перекрытие по x — В ДОЛЯХ ДЛИННОЙ строки: иначе строке, распавшейся в A на куски по широким
# пробелам, сопоставляется короткий кусок, и его наклон — шум (1967/01 с.4).
MATCH_DY_HEIGHTS = 0.7
MATCH_DY_MIN_MM = 0.7
MIN_OVERLAP = 0.75


def match_lines(
    before: list[TextLine], after: list[TextLine], field: Field | None, dpi: float = WORK_DPI
) -> list[tuple[TextLine, TextLine]]:
    """Пары «строка B — та же строка в A»: центр переносится полем, ищется ближайшая по y с перекрытием по x."""
    if not before or not after:
        return []
    centres = np.array([[line.cx, line.cy] for line in before])
    moved = field.transform(centres) if field is not None else centres
    shift_x = moved[:, 0] - centres[:, 0]
    pairs: list[tuple[TextLine, TextLine]] = []
    taken: set[int] = set()
    dy_min = MATCH_DY_MIN_MM * dpi / 25.4
    for line, (_, my), dx in zip(before, moved, shift_x):
        tolerance = max(dy_min, MATCH_DY_HEIGHTS * line.height)
        best, best_dy = None, tolerance
        for j, other in enumerate(after):
            if j in taken:
                continue
            dy = abs(other.cy - my)
            if dy >= best_dy:
                continue
            overlap = min(line.x1 + dx, other.x1) - max(line.x0 + dx, other.x0)
            if overlap < MIN_OVERLAP * max(line.length, other.length):
                continue
            best, best_dy = j, dy
        if best is not None:
            taken.add(best)
            pairs.append((line, after[best]))
    return pairs


def line_metrics(
    before: list[TextLine],
    after: list[TextLine],
    pairs: list[tuple[TextLine, TextLine]],
    min_len_mm: float,
    dpi: float = WORK_DPI,
) -> tuple[dict[str, float], dict]:
    """Худшие попарные разности по длинным строкам плюс сводки кривизны страницы."""
    min_len_px = mm_to_px(min_len_mm, dpi)
    metrics: dict[str, float] = {"lines_b": float(len(before)), "lines_a": float(len(after))}
    stats_b = page_stats([l.fit for l in before], [l.fit_scale * l.height for l in before])
    stats_a = page_stats([l.fit for l in after], [l.fit_scale * l.height for l in after])
    for key in ("sagitta_rel_p90", "slope_spread_deg"):
        metrics[f"text_{key}_b"] = stats_b.get(key, 0.0)
        metrics[f"text_{key}_a"] = stats_a.get(key, 0.0)
        metrics[f"text_{key}_delta"] = stats_a.get(key, 0.0) - stats_b.get(key, 0.0)
    long = [(b, a) for b, a in pairs if b.length >= min_len_px and a.length >= min_len_px]
    metrics["lines_matched"] = float(len(pairs))
    metrics["lines_long_matched"] = float(len(long))
    culprits: dict = {}
    if not long:
        metrics["line_dev_max_delta_mm"] = 0.0
        metrics["line_wobble_delta_max"] = 0.0
        return metrics, culprits
    # Наклон — в мм ухода конца строки от горизонтали (длина × sin): короткий кусок строки под
    # 1.6° — это 0.7 мм и шум сегментации, заголовок в 100 мм под 0.8° — 1.4 мм и видно глазом.
    dev = [
        px_to_mm(b.length, dpi) * (np.sin(np.radians(abs(a.slope_deg))) - np.sin(np.radians(abs(b.slope_deg))))
        for b, a in long
    ]
    wobble = [a.wobble_rel - b.wobble_rel for b, a in long]
    for name, values in (("line_dev_max_delta_mm", dev), ("line_wobble_delta_max", wobble)):
        worst = int(np.argmax(values))
        metrics[name] = float(values[worst])
        culprits[name] = {"b": long[worst][0].box, "a": long[worst][1].box}
    return metrics, culprits
