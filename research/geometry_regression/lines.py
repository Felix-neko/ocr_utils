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
# Меньше стольких пар в колонке — выигрыш по ней не считается (сводка по 3 строкам — шум).
MIN_COLUMN_LINES = 8
# Строка, бокс которой заходит на соседнюю строку той же колонки глубже этой доли высоты
# соседней, — две слипшиеся строки (перенос с хвостом соседней, 1976/02 с.84); в пары не идёт.
# По одной высоте судить нельзя: крупный заголовок втрое выше корпуса и ни с кем не слипся.
MAX_NEIGHBOUR_OVERLAP = 0.25
# Наклон слипшейся строки отличается от наклона перекрытого соседа не меньше чем на это.
MERGED_SLOPE_DIFF_DEG = 1.2


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
    merged = _merged_lines(before)
    for line, (mx, my), dx in zip(before, moved, shift_x):
        if id(line) in merged:
            continue
        tolerance = max(dy_min, MATCH_DY_HEIGHTS * line.height)
        best, best_dy = None, tolerance
        for j, other in enumerate(after):
            # Только внутри одной колонки — по перекрытию боксов по x после переноса полем, а не
            # по номерам: межколонники в A и B могут разойтись (трапеция убрана, врезка курсивом).
            # Строка через межколонник (column −1) спаривается только с такой же.
            if j in taken or (line.column < 0) != (other.column < 0):
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


def _merged_lines(lines: list[TextLine]) -> set[int]:
    """Строки, слипшиеся с соседней по вертикали в той же колонке.

    Слипшаяся строка (перенос с хвостом соседней, 1976/02 с.84) выше соседей и перекрывает
    их боксом, а её центр-линия наклонена НЕ так, как у перекрытого соседа: хвост тянет её
    в сторону. Косые строки блока (1968/05 с.55, которые FineReader потом выпрямил) тоже
    высокие и перекрывают соседей, но наклонены с ними одинаково.
    """
    merged: set[int] = set()
    for line in lines:
        for other in lines:
            if other is line or other.column != line.column:
                continue
            if other.x0 >= line.x1 or other.x1 <= line.x0:
                continue
            overlap = min(line.y1, other.y1) - max(line.y0, other.y0)
            if overlap <= MAX_NEIGHBOUR_OVERLAP * (other.y1 - other.y0) or (line.y1 - line.y0) <= (other.y1 - other.y0):
                continue
            if abs(line.slope_deg - other.slope_deg) > MERGED_SLOPE_DIFF_DEG:
                merged.add(id(line))
                break
    return merged


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
    # Сводки по странице целиком — контекст; выигрыш — по колонкам, берётся лучшая: ровная
    # левая колонка не должна размывать распрямлённую правую (1976/02 с.84).
    page = _stats_pair(pairs, dpi)
    for key in ("sagitta_rel_p90", "slope_spread_deg"):
        metrics[f"text_{key}_b"] = page[f"{key}_b"]
        metrics[f"text_{key}_a"] = page[f"{key}_a"]
        metrics[f"text_{key}_delta"] = page[f"{key}_a"] - page[f"{key}_b"]
    columns = sorted({b.column for b, _ in pairs})
    best_sag, best_spread = 0.0, 0.0
    for column in columns:
        own = [(b, a) for b, a in pairs if b.column == column]
        if len(own) < MIN_COLUMN_LINES:
            continue
        stats = _stats_pair(own, dpi)
        best_sag = max(best_sag, (stats["sagitta_rel_p90_b"] - stats["sagitta_rel_p90_a"]) * stats["height_mm"])
        best_spread = max(best_spread, stats["slope_spread_deg_b"] - stats["slope_spread_deg_a"])
    metrics["text_sag_gain_mm"] = best_sag
    metrics["text_spread_gain_deg"] = best_spread
    metrics["text_columns"] = float(len(columns))
    return metrics


def _stats_pair(pairs: list[tuple[TextLine, TextLine]], dpi: float) -> dict[str, float]:
    """Прогиб p90 и разброс наклонов по парам строк, для B и A, плюс медианная высота строки в мм."""
    lines_b = [b for b, _ in pairs]
    lines_a = [a for _, a in pairs]
    stats_b = page_stats([l.fit for l in lines_b], [l.fit_scale * l.height for l in lines_b])
    stats_a = page_stats([l.fit for l in lines_a], [l.fit_scale * l.height for l in lines_a])
    return {
        "sagitta_rel_p90_b": stats_b.get("sagitta_rel_p90", 0.0),
        "sagitta_rel_p90_a": stats_a.get("sagitta_rel_p90", 0.0),
        "slope_spread_deg_b": stats_b.get("slope_spread_deg", 0.0),
        "slope_spread_deg_a": stats_a.get("slope_spread_deg", 0.0),
        "height_mm": px_to_mm(float(np.median([l.height for l in lines_b])), dpi) if lines_b else 0.0,
    }
