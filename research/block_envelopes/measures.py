"""Меры качества блоков полосы без эталона: выход границы за краску строк, «пила» на крышках, клинья, подозрения на ложный разрез и слияние, потерянные строки."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import median_filter

from ocr_utils.page_layout import mm_to_px, px_to_mm

# Допуск выхода границы за краску строк: поле, которое ставят все алгоритмы, и полпикселя округления.
OVERSHOOT_TOL_MM = 1.0
# Окно медианы, от которой меряется «пила» крышки: зубья короче окна, изгиб строки — длиннее.
SAW_WINDOW_MM = 8.0
# Ложный разрез: соседние по вертикали блоки ближе стольких шагов строки...
SPLIT_GAP_PITCHES = 1.8
# ...перекрытые по x на такую долю более узкого...
SPLIT_OVERLAP = 0.7
# ...и одного кегля (отношение высот рядов на стыке меньше этого).
SPLIT_HEIGHT_RATIO = 1.3
# Слияние по горизонтали: ряды одного блока на одной высоте через пустоту шире этого.
HMERGE_GAP_MM = 3.5
# Строка потеряна, если середина её оси дальше этого от всех границ блоков.
LOST_TOL_MM = 1.0


@dataclass(frozen=True)
class RowBox:
    """Ряд блока для мер: середина по высоте, края краски, высота (пиксели рабочей копии)."""

    y: float
    x0: float
    x1: float
    height: float


@dataclass(frozen=True)
class BlockShape:
    """Блок для мер: контур границы и ряды."""

    polygon: np.ndarray
    rows: tuple[RowBox, ...]
    pitch: float  # шаг строк блока, пиксели


@dataclass(frozen=True)
class PageMeasures:
    """Меры полосы по всем её блокам.

    Attributes:
        blocks: Число блоков.
        overshoot_mm: Наибольший выход границы за краску строк блока (сверх допуска), мм.
        overshoot_blocks: Блоков с выходом больше допуска.
        saw_mm: Наибольшая «пила» крышки блока — сумма колебаний вокруг медианы на 100 мм длины, мм.
        wedge_share: Наибольшая доля площади блока вне «ступенчатого» объединения его рядов,
            кроме заливки коротких строк до выровненной стороны.
        split_pairs: Пар соседних блоков, похожих на один разрезанный.
        hmerge_blocks: Блоков с рядами бок о бок через пустоту.
        lost_axes: Осей строк вне всех блоков.
    """

    blocks: int
    overshoot_mm: float
    overshoot_blocks: int
    saw_mm: float
    wedge_share: float
    split_pairs: int
    hmerge_blocks: int
    lost_axes: int


def _mask(polygon: np.ndarray, origin: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Растр контура в рамке ``origin``/``size`` (высота, ширина)."""
    mask = np.zeros(size, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon - origin).astype(np.int32)], 1)
    return mask.astype(bool)


def overshoot_px(block: BlockShape, dpi: float) -> float:
    """Насколько граница выходит за рамку краски строк блока: по x — за крайние края рядов, по y — за крайние ряды.

    Заливка до выровненной стороны (короткая строка конца абзаца) выходом не считается: она лежит
    внутри рамки. Выход — это шип или хвост за краем текста (1966/02 IMG_0072_2R) и зубья над
    заголовком.

    Args:
        block: Блок.
        dpi: Разрешение рабочей копии.

    Returns:
        Наибольший выход в пикселях (0 — не выходит).
    """
    if block.polygon is None or len(block.polygon) < 3 or not block.rows:
        return 0.0
    points = np.asarray(block.polygon, dtype=np.float64)
    x0, x1, _, _ = _extent(block)
    top = min(row.y - row.height / 2.0 for row in block.rows)
    bottom = max(row.y + row.height / 2.0 for row in block.rows)
    horizontal = np.maximum(x0 - points[:, 0], points[:, 0] - x1)
    vertical = np.maximum(top - points[:, 1], points[:, 1] - bottom)
    return float(max(0.0, horizontal.max(initial=0.0), vertical.max(initial=0.0)))


def saw_px(block: BlockShape, dpi: float) -> float:
    """«Пила» верхней и нижней крышки: колебания профиля вокруг его медианы, на 100 мм длины.

    Профиль — самая верхняя (нижняя) точка растра блока в каждом столбце. Зубья короче окна
    медианы ``SAW_WINDOW_MM`` дают вклад, плавный изгиб строки — нет.

    Args:
        block: Блок.
        dpi: Разрешение рабочей копии.

    Returns:
        Наибольшая из двух крышек сумма колебаний, пиксели на 100 мм.
    """
    if block.polygon is None or len(block.polygon) < 3:
        return 0.0
    points = np.asarray(block.polygon, dtype=np.float64)
    origin = np.floor(points.min(axis=0)) - 1
    size = tuple((np.ceil(points.max(axis=0) - origin) + 2).astype(int)[::-1])
    mask = _mask(points, origin, size)
    columns = np.flatnonzero(mask.any(axis=0))
    if columns.size < 3:
        return 0.0
    window = max(3, int(mm_to_px(SAW_WINDOW_MM, dpi)) | 1)
    # Крайние столбцы — это боковые кромки, а не крышка: отступаем от них на полокна.
    inner = columns[window // 2 : -(window // 2) or None] if columns.size > window else columns
    worst = 0.0
    for profile in (np.argmax(mask[:, columns], axis=0), mask.shape[0] - 1 - np.argmax(mask[::-1, columns], axis=0)):
        profile = profile.astype(np.float64)
        residual = profile - median_filter(profile, size=window, mode="nearest")
        residual = residual[np.isin(columns, inner)]
        if residual.size < 2:
            continue
        length_mm = px_to_mm(residual.size, dpi)
        worst = max(worst, float(np.abs(np.diff(residual)).sum()) * 100.0 / max(length_mm, 1.0))
    return worst


def stepped_union(block: BlockShape, origin: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """«Ступенчатое» объединение рядов: прямоугольник каждого ряда до середин промежутков к соседям.

    Плюс заливка до ВЫРОВНЕННОЙ стороны: короткая строка конца абзаца и абзацный отступ добиваются
    до края соседних рядов, если оба соседа по эту сторону стоят на одной вертикали. Всё, что вне
    этого, — клин или выступ.

    Args:
        block: Блок.
        origin: Левый верхний угол растра.
        size: Размер растра (высота, ширина).

    Returns:
        Булев растр.
    """
    mask = np.zeros(size, dtype=np.uint8)
    rows = sorted(block.rows, key=lambda row: row.y)
    tol = max(block.pitch, 1.0) * 0.3
    for index, row in enumerate(rows):
        above = rows[index - 1] if index > 0 else None
        below = rows[index + 1] if index + 1 < len(rows) else None
        top = (above.y + row.y) / 2.0 if above else row.y - row.height / 2.0
        bottom = (row.y + below.y) / 2.0 if below else row.y + row.height / 2.0
        x0, x1 = row.x0, row.x1
        neighbours = [item for item in (above, below) if item is not None]
        # Левая сторона выровнена у соседей — ряд добивается до неё (абзацный отступ).
        if neighbours and max(item.x0 for item in neighbours) - min(item.x0 for item in neighbours) <= tol:
            x0 = min(x0, min(item.x0 for item in neighbours))
        # То же справа (короткая строка конца абзаца).
        if neighbours and max(item.x1 for item in neighbours) - min(item.x1 for item in neighbours) <= tol:
            x1 = max(x1, max(item.x1 for item in neighbours))
        p0 = np.round(np.array([x0, top]) - origin).astype(int)
        p1 = np.round(np.array([x1, bottom]) - origin).astype(int)
        cv2.rectangle(mask, tuple(p0), tuple(p1), 1, -1)
    return mask.astype(bool)


def wedge_share(block: BlockShape, dpi: float) -> float:
    """Доля площади блока вне ступенчатого объединения рядов (с полем ``OVERSHOOT_TOL_MM``).

    Args:
        block: Блок.
        dpi: Разрешение рабочей копии.

    Returns:
        Доля от 0 до 1.
    """
    if block.polygon is None or len(block.polygon) < 3 or not block.rows:
        return 0.0
    points = np.asarray(block.polygon, dtype=np.float64)
    pad = mm_to_px(3.0, dpi)
    lo = np.minimum(points.min(axis=0), [min(r.x0 for r in block.rows), min(r.y for r in block.rows)]) - pad
    hi = np.maximum(points.max(axis=0), [max(r.x1 for r in block.rows), max(r.y for r in block.rows)]) + pad
    origin = np.floor(lo)
    size = tuple((np.ceil(hi - origin) + 1).astype(int)[::-1])
    polygon = _mask(points, origin, size)
    area = int(polygon.sum())
    if area == 0:
        return 0.0
    union = stepped_union(block, origin, size).astype(np.uint8)
    grow = max(1, int(round(mm_to_px(OVERSHOOT_TOL_MM, dpi))))
    union = cv2.dilate(union, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)).astype(bool)
    return float((polygon & ~union).sum()) / area


def _extent(block: BlockShape) -> tuple[float, float, float, float]:
    """Крайние края рядов блока: ``x0, x1, верх, низ``."""
    return (
        min(row.x0 for row in block.rows),
        max(row.x1 for row in block.rows),
        min(row.y for row in block.rows),
        max(row.y for row in block.rows),
    )


def split_pairs(blocks: list[BlockShape]) -> int:
    """Пары соседних блоков, похожие на один разрезанный: близко по вертикали, друг под другом, одного кегля.

    Args:
        blocks: Блоки полосы.

    Returns:
        Число таких пар.
    """
    count = 0
    for upper in blocks:
        if not upper.rows:
            continue
        ux0, ux1, _, _ = _extent(upper)
        last = max(upper.rows, key=lambda row: row.y)
        for lower in blocks:
            if lower is upper or not lower.rows:
                continue
            lx0, lx1, _, _ = _extent(lower)
            first = min(lower.rows, key=lambda row: row.y)
            gap = first.y - last.y
            pitch = max(min(upper.pitch, lower.pitch), max(first.height, last.height))
            if not 0 < gap <= SPLIT_GAP_PITCHES * pitch:
                continue
            overlap = min(ux1, lx1) - max(ux0, lx0)
            if overlap < SPLIT_OVERLAP * min(ux1 - ux0, lx1 - lx0):
                continue
            if max(first.height, last.height) > SPLIT_HEIGHT_RATIO * min(first.height, last.height):
                continue
            count += 1
    return count


def hmerged(block: BlockShape, dpi: float) -> bool:
    """Есть ли в блоке ряды бок о бок: на одной высоте, через пустоту шире ``HMERGE_GAP_MM``.

    Args:
        block: Блок.
        dpi: Разрешение рабочей копии.

    Returns:
        ``True`` — блок похож на слитые по горизонтали.
    """
    limit = mm_to_px(HMERGE_GAP_MM, dpi)
    rows = block.rows
    for index, first in enumerate(rows):
        for second in rows[index + 1 :]:
            if abs(first.y - second.y) >= (first.height + second.height) / 2.0:
                continue
            if max(first.x0, second.x0) - min(first.x1, second.x1) > limit:
                return True
    return False


def lost_axes(blocks: list[BlockShape], axes_mid: list[tuple[float, float]], dpi: float) -> int:
    """Сколько осей строк лежат серединой вне всех границ блоков.

    Args:
        blocks: Блоки полосы.
        axes_mid: Середины осей ``(x, y)``.
        dpi: Разрешение рабочей копии.

    Returns:
        Число потерянных осей.
    """
    tol = mm_to_px(LOST_TOL_MM, dpi)
    contours = [np.asarray(b.polygon, np.float32).reshape(-1, 1, 2) for b in blocks if b.polygon is not None]
    lost = 0
    for x, y in axes_mid:
        if not any(cv2.pointPolygonTest(c, (float(x), float(y)), True) >= -tol for c in contours):
            lost += 1
    return lost


def page_measures(
    blocks: list[BlockShape], axes_mid: list[tuple[float, float]], dpi: float, lost: int | None = None
) -> PageMeasures:
    """Все меры полосы.

    Args:
        blocks: Блоки полосы.
        axes_mid: Середины осей строк ``(x, y)``.
        dpi: Разрешение рабочей копии.
        lost: Число осей вне рядов всех блоков, если оно известно по членству (прогон по кэшу);
            ``None`` — считать геометрически: ось вне всех границ (JSON без связи осей с рядами).
            Геометрия врёт в пользу раздутых границ: выступ боевой огибающей случайно накрывает
            ось, не вошедшую ни в один блок.

    Returns:
        :class:`PageMeasures`.
    """
    tol = mm_to_px(OVERSHOOT_TOL_MM, dpi)
    overs = [overshoot_px(block, dpi) for block in blocks]
    return PageMeasures(
        blocks=len(blocks),
        overshoot_mm=round(px_to_mm(max([0.0, *[max(0.0, o - tol) for o in overs]]), dpi), 2),
        overshoot_blocks=sum(1 for o in overs if o > tol),
        saw_mm=round(px_to_mm(max([0.0, *[saw_px(block, dpi) for block in blocks]]), dpi), 2),
        wedge_share=round(max([0.0, *[wedge_share(block, dpi) for block in blocks]]), 3),
        split_pairs=split_pairs(blocks),
        hmerge_blocks=sum(1 for block in blocks if hmerged(block, dpi)),
        lost_axes=lost if lost is not None else lost_axes(blocks, axes_mid, dpi),
    )


def shapes_from_json(payload: dict) -> tuple[list[BlockShape], list[tuple[float, float]]]:
    """Блоки и середины осей из JSON разбора (:func:`text_blocks.report.page_json`).

    Args:
        payload: JSON полосы.

    Returns:
        ``(блоки, середины осей)``.
    """
    dpi = float(payload["dpi"])
    blocks = []
    for block in payload["blocks"]:
        rows = tuple(RowBox(r["y"], r["x0"], r["x1"], r["height"]) for r in block["rows"])
        blocks.append(
            BlockShape(
                polygon=np.asarray(block["envelope"]["polygon"], dtype=np.float64),
                rows=rows,
                pitch=mm_to_px(block["pitch_mm"], dpi),
            )
        )
    axes_mid = []
    for axis in payload["axes"]:
        points = np.asarray(axis["points"], dtype=np.float64)
        middle = points[len(points) // 2]
        axes_mid.append((float(middle[0]), float(middle[1])))
    return blocks, axes_mid


def _axis_key(axis) -> tuple[int, int, int]:
    """Ключ оси для сравнения копий: округлённые середина по высоте и концы (как ``blocks._axis_key``)."""
    return (round(axis.cy), round(axis.x0), round(axis.x1))


def shapes_from_blocks(blocks: list, axes: list) -> tuple[list[BlockShape], list[tuple[float, float]], int]:
    """Блоки, середины осей и число осей вне рядов всех блоков из объектов разбора (``TextBlock``, ``LineAxis``).

    Args:
        blocks: Блоки ``TextBlock``.
        axes: Оси ``LineAxis``.

    Returns:
        ``(блоки, середины осей, потерянных осей по членству)``.
    """
    shapes = [
        BlockShape(
            polygon=np.asarray(block.envelope.polygon, dtype=np.float64),
            rows=tuple(RowBox(row.y, row.x0, row.x1, row.height) for row in block.rows),
            pitch=float(block.pitch_px),
        )
        for block in blocks
    ]
    axes_mid = []
    for axis in axes:
        points = np.asarray(axis.points, dtype=np.float64)
        middle = points[len(points) // 2]
        axes_mid.append((float(middle[0]), float(middle[1])))
    # Оси в рядах — копии (с номером колонки), поэтому сравниваются по ключу: середина и концы.
    members = {_axis_key(axis) for block in blocks for row in block.rows for axis in row.axes}
    lost = sum(1 for axis in axes if _axis_key(axis) not in members)
    return shapes, axes_mid, lost


__all__ = ["BlockShape", "PageMeasures", "RowBox", "page_measures", "shapes_from_blocks", "shapes_from_json"]
