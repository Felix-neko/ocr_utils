"""Колонка номеров страниц оглавления: номера справа от отточий, выключенные по правому краю, — отдельный блок с выровненной правой стороной.

Номер из одной-трёх цифр (2–4 мм) оси не получает — :func:`lines.axis_of` выбрасывает оси короче
``MIN_LENGTH_MM`` = 4 мм, — поэтому в блоки он не попадал, и правой стороны по колонке номеров не было (1968/04
IMG_0052_1L, ПХ 1975/01 IMG_0085: 10 номеров из 15 без оси). Здесь номер ищется по краске справа от конца каждого
отточия; номера, чьи правые края лежат на одной вертикали, становятся блоком «колонка номеров» сквозь заголовки
разделов. Все размеры — в мм бумаги, координаты — пиксели рабочей копии (``WORK_DPI``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.page_layout.text_blocks.blocks import BlockEnvelope, Row, TextBlock
from ocr_utils.page_layout.text_blocks.lines import LineAxis
from ocr_utils.units import mm_to_px_float, px_to_mm

# Окно поиска номера за концом отточия: по высоте ± NUMBER_HALF_MM от отточия, по ширине от NUMBER_GAP_MM до
# NUMBER_REACH_MM за концом точек; номер — первый сгусток краски, кончающийся просветом шире NUMBER_STOP_MM, шириной
# NUMBER_MIN_W_MM–NUMBER_MAX_W_MM и высотой NUMBER_MIN_H_MM–NUMBER_MAX_H_MM.
NUMBER_HALF_MM = 1.5
NUMBER_GAP_MM = 0.5
NUMBER_REACH_MM = 15.0
NUMBER_STOP_MM = 3.0
NUMBER_MIN_W_MM = 1.0
NUMBER_MAX_W_MM = 12.0
NUMBER_MIN_H_MM = 1.2
NUMBER_MAX_H_MM = 4.0
# Колонка — правые края в допуске NUMBER_TOL_MM от медианы (замер ПХ 1975/01 IMG_0085: 14 номеров на 117.74–117.97 мм,
# размах 0.23 мм) и не меньше NUMBER_MIN номеров.
NUMBER_TOL_MM = 0.6
NUMBER_MIN = 4


@dataclass(frozen=True)
class NumberMark:
    """Номер страницы за отточием: рамка краски ``(x0, y0, x1, y1)`` и высота отточия ``y`` (px рабочей копии)."""

    x0: float
    y0: float
    x1: float
    y1: float
    y: float


def leader_numbers(ink: np.ndarray, leaders: list, k: float) -> list[NumberMark]:
    """Номера страниц справа от отточий по краске текста.

    Args:
        ink: Краска текста рендера (``bool``, :func:`page.text_ink`).
        leaders: Отточия страницы (``leaders.Leader``, px рабочей копии).
        k: Во сколько раз рендер крупнее рабочей копии.

    Returns:
        Номера (px рабочей копии); отточие без номера ничего не даёт.
    """
    dpi = 150.0 * k  # разрешение рендера (рабочая копия — ``WORK_DPI`` = 150)
    half, gap = mm_to_px_float(NUMBER_HALF_MM, dpi), mm_to_px_float(NUMBER_GAP_MM, dpi)
    reach, stop = mm_to_px_float(NUMBER_REACH_MM, dpi), mm_to_px_float(NUMBER_STOP_MM, dpi)
    out = []
    for leader in leaders:
        y, x_end = leader.y * k, leader.x1 * k
        r0, r1 = max(int(y - half), 0), min(int(y + half) + 1, ink.shape[0])
        c0, c1 = min(int(x_end + gap), ink.shape[1]), min(int(x_end + reach), ink.shape[1])
        window = ink[r0:r1, c0:c1]
        cols = np.flatnonzero(window.any(axis=0))
        if not cols.size:
            continue
        # Первый сгусток: столбцы с краской до первого просвета шире ``stop``.
        breaks = np.flatnonzero(np.diff(cols) > stop)
        first = cols[: breaks[0] + 1] if breaks.size else cols
        rows = np.flatnonzero(window[:, first[0] : first[-1] + 1].any(axis=1))
        width = px_to_mm(first[-1] - first[0] + 1, dpi)
        height = px_to_mm(rows[-1] - rows[0] + 1, dpi)
        if NUMBER_MIN_W_MM <= width <= NUMBER_MAX_W_MM and NUMBER_MIN_H_MM <= height <= NUMBER_MAX_H_MM:
            out.append(
                NumberMark(
                    x0=(c0 + first[0]) / k,
                    y0=(r0 + rows[0]) / k,
                    x1=(c0 + first[-1] + 1) / k,
                    y1=(r0 + rows[-1] + 1) / k,
                    y=float(leader.y),
                )
            )
    return out


def number_columns(marks: list[NumberMark], dpi: float) -> list[list[NumberMark]]:
    """Колонки номеров: группы по правому краю (соседи по x ближе допуска), члены у медианы, от :data:`NUMBER_MIN`.

    Args:
        marks: Номера страницы.
        dpi: Разрешение рабочей копии.

    Returns:
        Колонки, номера в каждой — сверху вниз.
    """
    if len(marks) < NUMBER_MIN:
        return []
    tol = mm_to_px_float(NUMBER_TOL_MM, dpi)
    order = sorted(marks, key=lambda m: m.x1)
    groups, current = [], [order[0]]
    for prev, mark in zip(order, order[1:]):
        if mark.x1 - prev.x1 <= tol:
            current.append(mark)
        else:
            groups.append(current)
            current = [mark]
    groups.append(current)
    out = []
    for group in groups:
        middle = float(np.median([m.x1 for m in group]))
        own = [m for m in group if abs(m.x1 - middle) <= tol]
        if len(own) >= NUMBER_MIN:
            out.append(sorted(own, key=lambda m: m.y))
    return out


def inside_numbers(axis: LineAxis, marks: list[NumberMark], pad: float) -> bool:
    """Лежит ли ось целиком на номере (3-значный номер длиннее 4 мм оси получает и в общий разбор не должен идти)."""
    return any(
        mark.x0 - pad <= axis.x0 and axis.x1 <= mark.x1 + pad and mark.y0 - pad <= axis.cy <= mark.y1 + pad
        for mark in marks
    )


def column_block(column: list[NumberMark], number: int, dpi: float) -> TextBlock:
    """Блок «колонка номеров»: ряд на номер, ось — отрезок на высоте отточия, стороны — по краям номеров.

    Правая сторона (``core_right``) — точки правых краёв номеров сверху вниз: по ней стенд выпрямления ставит
    вертикаль. Левая — края номеров разной длины (рваная по определению).

    Args:
        column: Номера колонки сверху вниз.
        number: Номер блока (поле ``column``).
        dpi: Разрешение рабочей копии.

    Returns:
        Блок с флагом ``number_column``.
    """
    rows = []
    for mark in column:
        height = mark.y1 - mark.y0
        points = np.array([[mark.x0, mark.y], [mark.x1, mark.y]], np.float64)
        axis = LineAxis(
            points=points, height=float(height), column=0, dpi=float(dpi), cross=False, sagitta_mm=0.0,
            slope_deg=0.0, bend_mm=0.0, resid_parabola_mm=0.0,
        )  # fmt: skip
        rows.append(Row(y=mark.y, height=float(height), x0=mark.x0, x1=mark.x1, axes=(axis,)))
    left = np.array([[m.x0, m.y] for m in column], np.float64)
    right = np.array([[m.x1, m.y] for m in column], np.float64)
    top = np.array([[column[0].x0, column[0].y0], [column[0].x1, column[0].y0]], np.float64)
    bottom = np.array([[column[-1].x0, column[-1].y1], [column[-1].x1, column[-1].y1]], np.float64)
    polygon = np.vstack(
        [[[m.x0, m.y0], [m.x1, m.y0]] for m in column[:1]]
        + [[[m.x1, m.y0], [m.x1, m.y1]] for m in column]
        + [[[m.x1, m.y1], [m.x0, m.y1]] for m in column[-1:]]
        + [[[m.x0, m.y1], [m.x0, m.y0]] for m in reversed(column)]
    )
    envelope = BlockEnvelope(
        left=left, right=right, top=top, bottom=bottom, polygon=polygon, smooth_pitches=0.0,
        core_left=left, core_right=right,
    )  # fmt: skip
    pitch = float(np.median(np.diff([m.y for m in column]))) if len(column) > 1 else 0.0
    span = (int(min(m.x0 for m in column)), int(np.ceil(max(m.x1 for m in column))))
    return TextBlock(
        column=number, index=0, span=span, rows=tuple(rows), pitch_px=pitch, dpi=float(dpi),
        envelope=envelope, envelope_coarse=envelope, number_column=True,
    )  # fmt: skip


__all__ = ["NumberMark", "column_block", "inside_numbers", "leader_numbers", "number_columns"]
