"""Аномалии выровненных сторон текстовых блоков: «пупырышки» (выступ одной строки) и «ступеньки» (выступ подряд в несколько строк) огибающей наружу."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ocr_utils.page_layout import px_to_mm
from ocr_utils.page_layout.text_blocks.alignment import AlignKind, Alignment
from ocr_utils.page_layout.text_blocks.blocks import TextBlock
from ocr_utils.page_layout.text_blocks.page import PageAnalysis

# Выступ стороны наружу не меньше этого — аномалия, мм. Калибровка (research/edge_marks, reports/edge_guard.md): при
# 0.6 мм ловится 71 из 75 кусков сора с выступом, аномальны 10 % случайных текстовых полос пака, зря помечено 12 из
# 82 законных выступов (висячие дефисы, широкие буквы); 0.5 мм — +2 куска сора ценой +2 полос и +3 знаков.
BUMP_MM = 0.6
# Блоки короче стольких строк не смотрятся: подписи, выходные данные, заголовки, оглавления дают выступы по 2–70 мм
# (8 ложных аномальных полос из 200), а сора с выступом на них нет.
MIN_ROWS = 6
# Окно строки: ± столько шагов строк вокруг её середины — там ищется наибольший выступ.
WINDOW_PITCHES = 0.5
# Кольцо сравнения: строки на расстоянии от ... до ... шагов — по ним медиана «нормальной» стороны.
RING_PITCHES = (1.5, 4.0)
# Выровненные виды выключки и их выровненные стороны.
ALIGNED_SIDES = {AlignKind.BOTH: ("left", "right"), AlignKind.LEFT: ("left",), AlignKind.RIGHT: ("right",)}


class BumpKind(str, Enum):
    """Вид выступа стороны."""

    BUMP = "bump"  # пупырышек: выступает одна строка
    STEP = "step"  # ступенька: выступ держится две строки и больше подряд


@dataclass(frozen=True)
class SideBump:
    """Выступ выровненной стороны блока наружу.

    Attributes:
        block: Номер блока в ``PageAnalysis.blocks``.
        side: ``left`` или ``right``.
        y0, y1: Участок стороны по высоте, пиксели рабочей копии (полшага строк вокруг крайних строк выступа).
        mm: Наибольший местный выступ на участке, мм.
        rows: Сколько строк подряд выступает.
        kind: Пупырышек или ступенька.
    """

    block: int
    side: str
    y0: float
    y1: float
    mm: float
    rows: int
    kind: BumpKind


def robust_line(points: np.ndarray) -> tuple[float, float]:
    """Прямая ``x = a·y + b`` по точкам стороны с отсевом выбросов (три прохода, > 2.5 MAD).

    Наклон колонки снимается прямой, а выступы в неё почти не влияют: отсев выбрасывает их.

    Args:
        points: Точки стороны ``(N, 2)`` — ``x, y``.

    Returns:
        ``(a, b)``.
    """
    ys, xs = points[:, 1], points[:, 0]
    keep = np.ones(len(ys), dtype=bool)
    a, b = 0.0, float(np.median(xs))
    for _ in range(3):
        if keep.sum() < 2:
            break
        a, b = np.polyfit(ys[keep], xs[keep], 1)
        residual = xs - (a * ys + b)
        mad = float(np.median(np.abs(residual[keep]))) + 0.5
        keep = np.abs(residual) <= 2.5 * mad
    return float(a), float(b)


def row_excess(block: TextBlock, side: str) -> np.ndarray:
    """Местный выступ огибающей стороны наружу у каждой строки блока, мм.

    Отклонение огибающей от устойчивой прямой стороны (наружу — плюс): наибольшее в окне ±``WINDOW_PITCHES``
    шага вокруг строки минус медиана в кольце ``RING_PITCHES`` шагов. Кольцо снимает плавный изгиб стороны
    и её волны, остаётся местный выступ.

    Args:
        block: Блок.
        side: ``left`` или ``right``.

    Returns:
        Массив по строкам блока (порядок ``block.rows``); где сторона пуста — нули.
    """
    curve = np.asarray(getattr(block.envelope, side), dtype=np.float64)
    if len(curve) < 3 or not block.rows:
        return np.zeros(len(block.rows))
    a, b = robust_line(curve)
    # Наружу: у левой стороны — влево (меньше x), у правой — вправо.
    sign = -1.0 if side == "left" else 1.0
    deviation = sign * (curve[:, 0] - (a * curve[:, 1] + b))
    pitch = block.pitch_px if block.pitch_px > 0 else float(np.median([row.height for row in block.rows]))
    out = np.zeros(len(block.rows))
    for index, row in enumerate(block.rows):
        distance = np.abs(curve[:, 1] - row.y)
        window = deviation[distance <= WINDOW_PITCHES * pitch]
        ring = deviation[(distance >= RING_PITCHES[0] * pitch) & (distance <= RING_PITCHES[1] * pitch)]
        if len(window) == 0:
            continue
        base = float(np.median(ring)) if len(ring) else 0.0
        out[index] = px_to_mm(float(window.max()) - base, block.dpi)
    return out


def side_bumps(block: TextBlock, alignment: Alignment, number: int, bump_mm: float = BUMP_MM) -> list[SideBump]:
    """Выступы выровненных сторон блока.

    Смотрятся только стороны, по которым блок выровнен (``both`` — обе, ``left``/``right`` — одна) и которые
    сами признаны выровненными, в блоке не меньше ``MIN_ROWS`` строк. Строки с выступом не меньше ``bump_mm`` сливаются в участки подряд идущих
    строк: одна строка — пупырышек, больше — ступенька.

    Args:
        block: Блок.
        alignment: Его выключка.
        number: Номер блока на странице.
        bump_mm: Порог выступа, мм.

    Returns:
        Выступы сверху вниз, по сторонам.
    """
    out: list[SideBump] = []
    if len(block.rows) < MIN_ROWS:
        return out
    pitch = block.pitch_px if block.pitch_px > 0 else 0.0
    for side in ALIGNED_SIDES.get(alignment.kind, ()):
        if not getattr(alignment, side).aligned:
            continue
        excess = row_excess(block, side)
        flagged = excess >= bump_mm
        index = 0
        while index < len(flagged):
            if not flagged[index]:
                index += 1
                continue
            # Участок подряд идущих строк с выступом.
            end = index
            while end + 1 < len(flagged) and flagged[end + 1]:
                end += 1
            rows = end - index + 1
            y0 = block.rows[index].y - 0.5 * max(pitch, block.rows[index].height)
            y1 = block.rows[end].y + 0.5 * max(pitch, block.rows[end].height)
            out.append(SideBump(number, side, float(y0), float(y1), float(excess[index : end + 1].max()), rows,
                                BumpKind.BUMP if rows == 1 else BumpKind.STEP))  # fmt: skip
            index = end + 1
    return out


def page_bumps(analysis: PageAnalysis, bump_mm: float = BUMP_MM) -> list[SideBump]:
    """Все выступы выровненных сторон блоков страницы; страница аномальна, если список не пуст.

    Args:
        analysis: Разбор страницы.
        bump_mm: Порог выступа, мм.

    Returns:
        Выступы по блокам.
    """
    out: list[SideBump] = []
    for number, (block, alignment) in enumerate(zip(analysis.blocks, analysis.alignments)):
        out += side_bumps(block, alignment, number, bump_mm)
    return out


__all__ = ["BUMP_MM", "MIN_ROWS", "BumpKind", "SideBump", "page_bumps", "robust_line", "row_excess", "side_bumps"]
