"""Профиль сжатия у корешка по отдельным буквам: ширина каждой компоненты высотой со строчную к медиане таких же компонент той же строки вдали от корешка, как функция расстояния от края текста — с разрешением в букву, а не в окно 5 мм; тот же замер от внешнего края текста — ноль (место в строке без корешка)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.gutter_compression.measure import MeasureParams, Side, TextLine, comp_widths, ink_mask, page_lines

# Опора строки — буквы дальше этого от края текста у корешка, мм (как ``MeasureParams.far_mm``).
FAR_MM = 25.0
# Опорных букв в строке не меньше этого, иначе строка не в счёт.
MIN_FAR_LETTERS = 12


def line_letters(ink: np.ndarray, line: TextLine, spine_x: float, px_per_mm: float) -> pd.DataFrame:
    """Буквы одной строки: положение от края текста у корешка и от внешнего края, ширина к опоре строки.

    Буква — связная компонента пояса строки высотой 0.7–1.25 строчной (``measure.comp_widths``): знаки препинания
    и дефисы не проходят по высоте. Опора — медиана ширины букв этой строки дальше :data:`FAR_MM` от края у корешка:
    кегль, гарнитура и печать строки сокращаются.

    Args:
        ink: Маска краски холста.
        line: Строка.
        spine_x: X линии корешка на холсте, px.
        px_per_mm: Плотность, px/мм.

    Returns:
        Таблица букв: ``from_edge_mm`` (центр буквы от края текста у корешка), ``from_outer_mm`` (от внешнего края
        строки), ``width_px``, ``ratio`` (к опоре строки); пустая — опорных букв мало.
    """
    xh = float(line.core_height)
    pad = int(round(0.5 * xh))
    top, bottom = max(0, line.core_top - pad), min(ink.shape[0], line.core_bottom + pad + 1)
    band = ink[top:bottom, line.x_lo : line.x_hi + 1]
    centres, widths = comp_widths(band, xh)
    if len(widths) == 0:
        return pd.DataFrame()
    xs = centres + line.x_lo
    # Край строки у корешка — конец краски, ближний к корешку; внешний — дальний.
    toward = line.side is Side.LEFT
    near_end, far_end = (line.x_hi, line.x_lo) if toward else (line.x_lo, line.x_hi)
    from_edge = np.abs(xs - near_end) / px_per_mm
    from_outer = np.abs(xs - far_end) / px_per_mm
    far = from_edge >= FAR_MM
    if far.sum() < MIN_FAR_LETTERS:
        return pd.DataFrame()
    reference = float(np.median(widths[far]))
    return pd.DataFrame({"from_edge_mm": from_edge, "from_outer_mm": from_outer, "width_px": widths,
                         "ratio": widths / reference,
                         "edge_mm": abs(spine_x - near_end) / px_per_mm})  # fmt: skip


def spread_letters(gray: np.ndarray, rects: tuple, spine_x: float, px_per_mm: float,
                   params: MeasureParams = MeasureParams()) -> pd.DataFrame:  # fmt: skip
    """Буквы обеих страниц выпрямленного разворота (строки — как у ``measure.spread_windows``).

    Args:
        gray: Выпрямленный холст uint8.
        rects: Прямоугольники левой и правой страницы.
        spine_x: X линии корешка, px.
        px_per_mm: Плотность, px/мм.
        params: Параметры меры (краска, строки).

    Returns:
        Таблица букв с ``side``, ``line`` (номер строки страницы), ``y`` (центр пояса строчных, px холста).
    """
    ink = ink_mask(gray, px_per_mm, params)
    parts = []
    for side, rect in zip(Side, rects):
        x_lo, x_hi = max(0, int(rect.x0)), min(gray.shape[1], int(np.ceil(rect.x1)))
        if x_hi - x_lo < 10:
            continue
        lines, _ = page_lines(ink, x_lo, x_hi, px_per_mm, side, params)
        for index, line in enumerate(lines):
            table = line_letters(ink, line, spine_x, px_per_mm)
            if len(table):
                table.insert(0, "side", side.value)
                table.insert(1, "line", index)
                table.insert(2, "y", (line.core_top + line.core_bottom) / 2)
                parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def binned_profile(letters: pd.DataFrame, column: str, edges_mm: np.ndarray) -> pd.DataFrame:
    """Медиана отношения ширины букв по поясам расстояния (от края у корешка или от внешнего края).

    Args:
        letters: Буквы стороны или группы сторон.
        column: ``from_edge_mm`` или ``from_outer_mm``.
        edges_mm: Границы поясов, мм.

    Returns:
        Таблица: ``lo``, ``hi`` (пояс, мм), ``median``, ``n`` (букв).
    """
    bins = pd.cut(letters[column], edges_mm, right=False)
    grouped = letters.groupby(bins, observed=True).ratio
    out = grouped.agg(["median", "size"]).reset_index()
    out["lo"] = [iv.left for iv in out[column]]
    out["hi"] = [iv.right for iv in out[column]]
    return out.rename(columns={"size": "n"})[["lo", "hi", "median", "n"]]


__all__ = ["binned_profile", "line_letters", "spread_letters"]
