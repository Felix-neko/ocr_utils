"""Выключка блока: выровнен ли он по левому краю, по правому, по обоим — и насколько изогнут.

Всё меряется ОТНОСИТЕЛЬНО гладкой огибающей, а не относительно вертикали: блок на кривой
бумаге наклонён и изогнут сам по себе, и вертикаль тут ничего не говорит. Выключенная сторона
— та, у которой края рядов плотно лежат на своей огибающей; рваная — та, где они разбросаны.
Абзацные отступы (слева) и короткие последние строки абзацев (справа) — односторонние выбросы:
они считаются отдельно и в меру плотности не входят.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ocr_utils.curved_layout.blocks import BlockEnvelope, Row, TextBlock
from ocr_utils.page_layout import px_to_mm

# Ряд считается лежащим на огибающей, если он отстоит от неё не больше чем на столько мм.
ALIGN_TOL_MM = 0.8
# Сторона выключена, если на огибающей лежит не меньше этой доли рядов (без учёта исключений).
CORE_SHARE = 0.8
# Ряд, ушедший внутрь блока больше чем на столько мм, — исключение (абзацный отступ слева,
# конец абзаца справа), а не рваный край: такие ряды не портят вердикт, но считаются отдельно.
INDENT_MIN_MM = 2.5
# Исключений больше этой доли рядов — сторона всё-таки рваная (в рваном наборе «отступ» у каждой второй).
INDENT_MAX_SHARE = 0.45


class Side(str, Enum):
    """Сторона блока."""

    LEFT = "left"
    RIGHT = "right"


class AlignKind(str, Enum):
    """Вид выключки блока."""

    RAGGED = "ragged"  # ни одна сторона не выровнена
    LEFT = "left"  # выключка влево (правый край рваный)
    RIGHT = "right"  # выключка вправо
    BOTH = "both"  # выключка по формату


@dataclass(frozen=True)
class SideStats:
    """Меры одной стороны блока (всё в мм, кроме долей и счётчиков)."""

    resid_mad_mm: float  # медианное |отклонение| краёв рядов от огибающей
    core_share: float  # доля рядов на огибающей (в допуске ALIGN_TOL_MM)
    indent_rows: int  # ряды, ушедшие внутрь блока (отступ абзаца, конец абзаца)
    indent_mm: float  # медианный размер такого захода внутрь
    envelope_dev_mm: float  # расхождение основной огибающей с крупной: max |E_w − E_W|
    bend_mm: float  # изгиб самой огибающей: размах её остатка от прямой по концам
    aligned: bool


@dataclass(frozen=True)
class Alignment:
    """Вердикт по блоку: выключка и меры по обеим сторонам."""

    kind: AlignKind
    left: SideStats
    right: SideStats


def _core_curve(envelope: BlockEnvelope, side: Side) -> np.ndarray:
    """Кривая по телу блока с нужной стороны: ``core_*``, а если её нет — внешний контур."""
    if side is Side.LEFT:
        return envelope.core_left if envelope.core_left is not None else envelope.left
    return envelope.core_right if envelope.core_right is not None else envelope.right


def _residuals(rows: list[Row], envelope: BlockEnvelope, side: Side) -> np.ndarray:
    """Отклонения краёв рядов от огибающей, со знаком «внутрь блока — плюс» (пиксели)."""
    # Меры выключки считаются от тренда по телу блока (``core_*``), а не от внешнего контура:
    # контур отодвинут наружу до самых дальних строк и за одиночным выносом прыгает весь.
    curve = _core_curve(envelope, side)
    xs = np.array([row.x0 if side is Side.LEFT else row.x1 for row in rows], dtype=np.float64)
    ys = np.array([row.y for row in rows], dtype=np.float64)
    fitted = np.interp(ys, curve[:, 1], curve[:, 0])
    return (xs - fitted) if side is Side.LEFT else (fitted - xs)


def _bend(curve: np.ndarray) -> float:
    """Изгиб кривой: размах её остатка от прямой, проведённой по концам (пиксели)."""
    xs, ys = curve[:, 0], curve[:, 1]
    if ys[-1] - ys[0] <= 0:
        return 0.0
    chord = xs[0] + (xs[-1] - xs[0]) * (ys - ys[0]) / (ys[-1] - ys[0])
    resid = xs - chord
    return float(resid.max() - resid.min())


def side_stats(block: TextBlock, side: Side) -> SideStats:
    """Меры одной стороны блока: плотность прилегания к огибающей, отступы, изгиб, девиация."""
    rows = list(block.rows)
    dpi = block.dpi
    resid = _residuals(rows, block.envelope, side)
    tol = ALIGN_TOL_MM
    resid_mm = np.array([px_to_mm(value, dpi) for value in resid], dtype=np.float64)
    indent = resid_mm >= INDENT_MIN_MM  # ушли внутрь блока — исключения
    core = np.abs(resid_mm) <= tol
    considered = ~indent
    share = float(core[considered].mean()) if considered.any() else 0.0
    indent_share = float(indent.mean()) if indent.size else 0.0
    # Девиация масштабов тоже считается по тренду тела: внешний контур обоих масштабов прижат
    # к одним и тем же выносам и различался бы меньше, чем сами блоки.
    fine = _core_curve(block.envelope, side)
    coarse = _core_curve(block.envelope_coarse, side)
    dev = float(np.abs(fine[:, 0] - np.interp(fine[:, 1], coarse[:, 1], coarse[:, 0])).max())
    aligned = share >= CORE_SHARE and indent_share <= INDENT_MAX_SHARE
    return SideStats(
        resid_mad_mm=float(np.median(np.abs(resid_mm[considered]))) if considered.any() else 0.0,
        core_share=share,
        indent_rows=int(indent.sum()),
        indent_mm=float(np.median(resid_mm[indent])) if indent.any() else 0.0,
        envelope_dev_mm=px_to_mm(dev, dpi),
        bend_mm=px_to_mm(_bend(fine), dpi),
        aligned=aligned,
    )


def alignment_of(block: TextBlock) -> Alignment:
    """Выключка блока по обеим сторонам."""
    left = side_stats(block, Side.LEFT)
    right = side_stats(block, Side.RIGHT)
    if left.aligned and right.aligned:
        kind = AlignKind.BOTH
    elif left.aligned:
        kind = AlignKind.LEFT
    elif right.aligned:
        kind = AlignKind.RIGHT
    else:
        kind = AlignKind.RAGGED
    return Alignment(kind=kind, left=left, right=right)


__all__ = ["AlignKind", "Alignment", "Side", "SideStats", "alignment_of", "side_stats"]
