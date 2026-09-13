"""Ось и сторона поворота ячейки: смыкание глифов и выносные элементы.

ТО ЖЕ, ЧТО ``orientation.detectors.ink_axis``, НО НА ЯЧЕЙКЕ. Отличий два, и оба вынуждены
масштабом. Первое: константы даны для копии 300 dpi, а не 150 — в ячейке всего несколько
слов, и на 150 dpi кегль петита падает до 7 px, где отбор компонент по размеру перестаёт
отличать букву от точки. Второе: порог длины «строки» вчетверо меньше — строка в ячейке
это одно-два слова, а не полоса набора.

ЗАМЕР, НА КОТОРОМ СТОЯТ ПОРОГИ. Таблица 1 полосы 76 выпуска 1966/01 (25 ячеек, из них 10
повёрнутых, размечено глазами): при зазоре смыкания 8 px счёт оси у всех повёрнутых
от −0.86 до −1.00, у всех ячеек с прямым текстом от +0.92 до +1.00, у числовых ячеек
ровно 0.00 — числа короткие, «строкой» ни в одну сторону не становятся, и детектор о них
честно молчит. Порог 0.25 стоит посреди пустого места.

СТОРОНА берётся тем же признаком, что и на полосе: у кириллицы почти нет верхних выносных
(одна «б») и много нижних (р, у, ф, д, ц, щ), поэтому масса краски в строке смещена ВЫШЕ
геометрического центра. Константа знака ``UPRIGHT_ASYMMETRY`` калибрована на 400 полосах
пака и переиспользуется отсюда: она безразмерная, это доля высоты строки.
"""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.scan_markup.orientation.detectors.ink_axis import ASYMMETRY_THR, UPRIGHT_ASYMMETRY

from research.legacy.table_processing.detection.ruling import mm_to_px
from ocr_utils.scan_markup.table_detection.verify import (  # noqa: F401 — реэкспорт для стенда
    GLYPH_MIN_MM,
    GLYPH_MAX_MM,
    GLYPH_MAX_LONG_FACTOR,
    GLYPH_MIN_FILL,
    RLSA_GAP_MM,
    LINE_ASPECT,
    LINE_MIN_LENGTH_MM,
    MIN_LINE_INK_FRAC,
    AXIS_MARGIN_THR,
    MIN_LINES_FOR_SIGN,
    glyph_mask,
)
from research.legacy.table_processing.rotation.base import CellCrop, CellDetector, Verdict, unknown

# Рабочее разрешение детектора.
WORK_DPI = 300


def line_boxes(mask: np.ndarray, horizontal: bool, dpi: int) -> tuple[np.ndarray, float]:
    """Габариты «строк» вдоль оси и площадь краски в них."""
    gap = mm_to_px(RLSA_GAP_MM, dpi)
    kernel = np.ones((1, gap), np.uint8) if horizontal else np.ones((gap, 1), np.uint8)
    smeared = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(smeared, 8)
    if count <= 1:
        return np.empty((0, 4), int), 0.0
    stats = stats[1:]
    width = stats[:, cv2.CC_STAT_WIDTH].astype(float)
    height = stats[:, cv2.CC_STAT_HEIGHT].astype(float)
    long_side, short_side = (width, height) if horizontal else (height, width)
    lineish = (long_side >= LINE_ASPECT * np.maximum(short_side, 1.0)) & (
        long_side >= mm_to_px(LINE_MIN_LENGTH_MM, dpi)
    )
    boxes = stats[lineish][:, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
    return boxes, float(stats[lineish][:, cv2.CC_STAT_AREA].sum())


def asymmetry(mask: np.ndarray, dpi: int) -> tuple[float, int]:
    """Асимметрия краски поперёк ГОРИЗОНТАЛЬНЫХ строк и число учтённых строк."""
    boxes, _ = line_boxes(mask, horizontal=True, dpi=dpi)
    minimum = mm_to_px(GLYPH_MIN_MM, dpi) * 2
    values: list[float] = []
    for left, top, width, height in boxes:
        if height < minimum:
            continue
        profile = mask[top : top + height, left : left + width].sum(axis=1).astype(float)
        weight = profile.sum()
        if weight <= 0:
            continue
        centre = float((np.arange(height) * profile).sum() / weight)
        values.append((centre - (height - 1) / 2.0) / height)
    if len(values) < MIN_LINES_FOR_SIGN:
        return 0.0, len(values)
    return float(np.median(values)), len(values)


def detect(crop: CellCrop) -> Verdict:
    mask = glyph_mask(crop.gray, crop.dpi)
    _, horizontal_ink = line_boxes(mask, True, crop.dpi)
    _, vertical_ink = line_boxes(mask, False, crop.dpi)

    total = horizontal_ink + vertical_ink
    if total < MIN_LINE_INK_FRAC * mask.size:
        return unknown("в ячейке нечего мерить")

    axis_score = (horizontal_ink - vertical_ink) / total
    metrics = {"axis_score": axis_score, "line_ink": total / mask.size}
    if abs(axis_score) < AXIS_MARGIN_THR:
        return Verdict(0, 0.0, metrics=metrics, note="ось не различается")

    if axis_score > 0:
        # Ось книжная: в таблице это значит «поворот не нужен». 180 не рассматриваем.
        return Verdict(0, min(1.0, abs(axis_score)), metrics=metrics)

    candidates = tuple(rotation for rotation in (90, 270) if rotation in crop.allowed)
    if not candidates:
        return Verdict(0, 0.0, metrics=metrics, note="боковая ось вне набора допустимых углов")
    if len(candidates) == 1:
        return Verdict(candidates[0], min(1.0, abs(axis_score)), metrics=metrics)

    turned = np.ascontiguousarray(np.rot90(mask, k=-(candidates[0] // 90)))
    value, lines = asymmetry(turned, crop.dpi)
    metrics["asymmetry"] = value
    metrics["lines"] = float(lines)
    axis_confidence = min(1.0, abs(axis_score))
    if abs(value) < ASYMMETRY_THR:
        return Verdict(90, axis_confidence, axis_only=True, metrics=metrics, note="сторона не различается")
    matches = (value > 0) == (UPRIGHT_ASYMMETRY > 0)
    rotation = candidates[0] if matches else candidates[1]
    sign_confidence = min(1.0, abs(value) / (ASYMMETRY_THR * 4)) * min(1.0, lines / 4.0)
    return Verdict(rotation, min(axis_confidence, sign_confidence), metrics=metrics)


ALGORITHM = CellDetector(
    name="ink_axis",
    summary="своя мера: ось по смыканию глифов (RLSA), сторона по выносным элементам",
    stage="cpu",
    gives_sign=True,
    run=detect,
)
