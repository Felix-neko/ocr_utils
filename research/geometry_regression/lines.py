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
from research.geometry_regression import WORK_DPI, px_to_mm
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
    before: list[TextLine], after: list[TextLine], pairs: list[tuple[TextLine, TextLine]], dpi: float = WORK_DPI
) -> dict[str, float]:
    """Сводки кривизны страницы по подтверждённым парам строк: контекст и ВЫИГРЫШ.

    Наклон и волну отдельных строк здесь не мерим — это делает ``stretch.glyph_line_metrics``
    по тем же глифам; сводки же (прогиб p90, разброс наклонов) — по аппроксимациям центр-линий
    ``line_fit``, как в ``curved_lines``. Выигрыш в мм: прогиб в долях высоты × медианная высота.
    """
    metrics: dict[str, float] = {
        "lines_b": float(len(before)),
        "lines_a": float(len(after)),
        "lines_matched": float(len(pairs)),
    }
    lines_b = [b for b, _ in pairs]
    lines_a = [a for _, a in pairs]
    stats_b = page_stats([l.fit for l in lines_b], [l.fit_scale * l.height for l in lines_b])
    stats_a = page_stats([l.fit for l in lines_a], [l.fit_scale * l.height for l in lines_a])
    for key in ("sagitta_rel_p90", "slope_spread_deg"):
        metrics[f"text_{key}_b"] = stats_b.get(key, 0.0)
        metrics[f"text_{key}_a"] = stats_a.get(key, 0.0)
        metrics[f"text_{key}_delta"] = stats_a.get(key, 0.0) - stats_b.get(key, 0.0)
    height_mm = px_to_mm(float(np.median([l.height for l in lines_b])), dpi) if lines_b else 0.0
    metrics["text_sag_gain_mm"] = max(0.0, -metrics["text_sagitta_rel_p90_delta"]) * height_mm
    metrics["text_spread_gain_deg"] = max(0.0, -metrics["text_slope_spread_deg_delta"])
    return metrics
