"""Ось текста в ячейке по форме самих букв.

САМАЯ ПРЯМАЯ МЕРА ИЗ ВСЕХ. Кириллическая буква в наборе выше, чем шире: у неё есть
строчная высота и выносные, а ширина ограничена кеглем. Поверните её на 90 градусов —
и отношение перевернётся. Значит, чтобы понять, повёрнут ли текст, достаточно посмотреть
на медианное отношение ширины компоненты к её высоте, и никакой морфологии не нужно.

ЗАМЕР (56 ячеек четырёх таблиц 1966/01, разметка глазами; отброшены ячейки, где меньше
трёх компонент):

    повёрнутые (19): 1.07 ... 1.53, медиана 1.13
    прямые     (37): 0.58 ... 0.94, медиана 0.64

Порог 1.0 стоит в пустом промежутке с запасом 0.13 в обе стороны.

ПОЧЕМУ ЭТО ВАЖНЕЕ, ЧЕМ RLSA. Смыкание глифов (``ink_axis``) на тех же ячейках ошибалось
пять раз из 56, и все пять ошибок — одного рода: КОЛОНКА ЧИСЕЛ. Числа стоят друг под
другом с малым просветом, вертикальное смыкание собирает их в один столбец, и мера
говорит «текст боковой». Форма букв на это не покупается: цифра остаётся выше, чем шире,
как её ни складывай.

ЧЕГО ЭТА МЕРА НЕ ДАЁТ — СТОРОНЫ. Буква, повёрнутая влево, и буква, повёрнутая вправо,
одинаково широки. Поэтому детектор всегда ``axis_only``, а сторону называют другие.
"""

from __future__ import annotations

import cv2
import numpy as np

from research.legacy.table_processing.detection.ruling import mm_to_px
from research.legacy.table_processing.rotation.base import CellCrop, CellDetector, Verdict, unknown
from research.legacy.table_processing.rotation.ink_axis_cell import GLYPH_MIN_MM, glyph_mask

# Меньше этого числа компонент — медиана считается по шуму. Три: ячейка «93» даёт две
# цифры, и по двум цифрам судить нельзя.
MIN_COMPONENTS = 3

# Компонента мельче этой площади (в квадратных миллиметрах) — пыль, точка над «й» или
# обрывок засечки. 0.15 мм² при 300 dpi это 21 px².
MIN_COMPONENT_MM2 = 0.15

# Граница между «выше, чем шире» и «шире, чем выше».
ASPECT_THRESHOLD = 1.0

# Отступ от границы, при котором уверенность считается полной: 0.13 — это ровно тот зазор,
# который остался между двумя облаками в замере.
ASPECT_FULL_MARGIN = 0.13


def median_aspect(mask: np.ndarray, dpi: int) -> tuple[float, int]:
    """Медианное отношение ширины компоненты к высоте и число учтённых компонент."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return 0.0, 0
    width = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    height = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    minimal = MIN_COMPONENT_MM2 * (dpi / 25.4) ** 2
    keep = (width * height) >= minimal
    if int(keep.sum()) < MIN_COMPONENTS:
        return 0.0, int(keep.sum())
    return float(np.median(width[keep] / np.maximum(height[keep], 1.0))), int(keep.sum())


def detect(crop: CellCrop) -> Verdict:
    mask = glyph_mask(crop.gray, crop.dpi)
    aspect, components = median_aspect(mask, crop.dpi)
    if components < MIN_COMPONENTS:
        return unknown(f"в ячейке {components} компонент, судить не по чему")

    metrics = {"aspect": aspect, "components": float(components)}
    confidence = min(1.0, abs(aspect - ASPECT_THRESHOLD) / ASPECT_FULL_MARGIN)
    if aspect <= ASPECT_THRESHOLD:
        return Verdict(0, confidence, metrics=metrics)
    # Ось боковая; сторону эта мера не различает принципиально.
    return Verdict(90, confidence, axis_only=True, metrics=metrics)


# Порог глифа переиспользуется из соседнего детектора, чтобы обе меры смотрели на одну и ту
# же маску: расхождение мер должно означать расхождение МЕР, а не разную предобработку.
assert GLYPH_MIN_MM > 0

ALGORITHM = CellDetector(
    name="glyph_aspect",
    summary="ось по медианному отношению ширины буквы к высоте",
    stage="cpu",
    gives_sign=False,
    run=detect,
)
