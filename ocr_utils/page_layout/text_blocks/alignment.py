"""Выключка блока: выровнен ли он по левому краю, по правому, по обоим, по центру — и насколько изогнут.

Всё меряется ОТНОСИТЕЛЬНО гладкой огибающей, а не относительно вертикали: блок на кривой
бумаге наклонён и изогнут сам по себе, и вертикаль тут ничего не говорит. Выключенная сторона
— та, у которой края рядов плотно лежат на своей огибающей; рваная — та, где они разбросаны.
Абзацные отступы (слева) и короткие последние строки абзацев (справа) — односторонние выбросы:
они считаются отдельно и в меру плотности не входят.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from ocr_utils.page_layout.text_blocks.blocks import BlockEnvelope, Row, TextBlock
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
# Набор по центру (подпись автора, эпиграф, строки заголовка): середины рядов вдоль строк не дальше
# ``CENTER_TOL_MM`` от их медианы у не меньше ``CENTER_SHARE`` рядов, при этом длины рядов заметно
# различаются (размах больше ``CENTER_MIN_SPREAD_MM``: иначе это колонка по формату, у которой
# середины совпадают тоже). Нужно от ``CENTER_MIN_ROWS`` рядов. Допуск 1.5 мм — по курсивной подписи
# «В. ГУЛЕНКО, учёный секретарь экспертной комиссии ВАК» (1971/10 с.87): середины 771, 774 и 782 px,
# третья строка отходит на 1.35 мм. Проверяется центр, только если ни одна сторона не выровнена.
CENTER_TOL_MM = 1.5
CENTER_SHARE = 0.8
CENTER_MIN_SPREAD_MM = 3.0
CENTER_MIN_ROWS = 2


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
    CENTER = "center"  # по центру: середины строк на одной вертикали, края рваные


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
    center_share: float = 0.0  # доля рядов, чья середина на общей вертикали (``CENTER_TOL_MM``)


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


def block_frame(rows: list[Row]) -> float:
    """Медианный наклон строк блока (радианы) — поворот его системы координат.

    Args:
        rows: Ряды блока.

    Returns:
        Угол от горизонтали кадра; 0.0, если у рядов нет осей.
    """
    angles = []
    for row in rows:
        for axis in row.axes:
            points = np.asarray(axis.points, dtype=np.float64)
            if points.shape[0] >= 2 and points[-1, 0] - points[0, 0] > 0:
                angles.append(math.atan2(points[-1, 1] - points[0, 1], points[-1, 0] - points[0, 0]))
    return float(np.median(angles)) if angles else 0.0


def center_share(rows: list[Row], dpi: float) -> float:
    """Доля рядов, чья середина лежит на общей вертикали блока (в его системе координат).

    Середина ряда — ``(x0 + x1) / 2`` на его ординате; точки поворачиваются на наклон строк блока
    (:func:`block_frame`), и берётся координата вдоль строк — так наклон страницы не выдаёт центр за
    рваный край. Отклонение меряется от медианы.

    Args:
        rows: Ряды блока.
        dpi: Разрешение рабочей копии.

    Returns:
        Доля от 0 до 1; 0.0 — рядов меньше ``CENTER_MIN_ROWS``.
    """
    if len(rows) < CENTER_MIN_ROWS:
        return 0.0
    theta = block_frame(rows)
    along = np.array([(row.x0 + row.x1) / 2.0 * math.cos(theta) + row.y * math.sin(theta) for row in rows])
    deviation = np.array([px_to_mm(float(value), dpi) for value in np.abs(along - np.median(along))])
    return float((deviation <= CENTER_TOL_MM).mean())


def is_centered(rows: list[Row], dpi: float) -> bool:
    """Набран ли блок по центру: середины рядов на одной вертикали, а длины рядов различаются.

    Args:
        rows: Ряды блока.
        dpi: Разрешение рабочей копии.

    Returns:
        ``True`` — набор по центру.
    """
    if len(rows) < CENTER_MIN_ROWS:
        return False
    widths = np.array([row.x1 - row.x0 for row in rows])
    if px_to_mm(float(np.ptp(widths)), dpi) <= CENTER_MIN_SPREAD_MM:
        return False  # строки одной длины: середины совпадают и у колонки по формату
    return center_share(rows, dpi) >= CENTER_SHARE


def verdict(left: bool, right: bool, centered: bool) -> AlignKind:
    """Вердикт по выровненности сторон и центра: ``both`` → ``left`` → ``right`` → ``center`` → ``ragged``."""
    if left and right:
        return AlignKind.BOTH
    if left:
        return AlignKind.LEFT
    if right:
        return AlignKind.RIGHT
    return AlignKind.CENTER if centered else AlignKind.RAGGED


def alignment_of(block: TextBlock) -> Alignment:
    """Выключка блока по обеим сторонам и по центру.

    Однострочный блок выключки не имеет: оба края одной строки совпадают со своим трендом по
    определению, и прежде такой блок получал ``both``. Теперь — ``ragged`` (на оверлее ``none``).
    """
    left = side_stats(block, Side.LEFT)
    right = side_stats(block, Side.RIGHT)
    if len(block.rows) < 2:
        left, right = replace(left, aligned=False), replace(right, aligned=False)
        return Alignment(kind=AlignKind.RAGGED, left=left, right=right)
    share = center_share(list(block.rows), block.dpi)
    kind = verdict(left.aligned, right.aligned, is_centered(list(block.rows), block.dpi))
    return Alignment(kind=kind, left=left, right=right, center_share=share)


__all__ = [
    "AlignKind",
    "Alignment",
    "Side",
    "SideStats",
    "alignment_of",
    "block_frame",
    "center_share",
    "is_centered",
    "side_stats",
    "verdict",
]
