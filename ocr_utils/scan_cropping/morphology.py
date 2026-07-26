"""Морфология бинарных масок ДИСКОМ — через distance transform, а не ядром.

Эрозия маски диском радиуса ``r`` — это ровно «пиксели, удалённые от фона больше
чем на ``r``», а дилатация — «пиксели, удалённые от маски не больше чем на ``r``».
То есть обе операции получаются одним ``cv2.distanceTransform`` и порогом, без
структурного элемента вообще.

Зачем так. ``cv2.erode``/``cv2.dilate`` с ``MORPH_ELLIPSE`` стоят
O(пиксели × площадь ядра): круг неразделим, и на радиусах в десятки пикселей
(``EXTRA_EROSION_PX`` = 80, ``BG_FILL_EROSION_PX`` = 100) это секунды на кадр.
Distance transform стоит O(пикселей) и от радиуса не зависит вовсе. Замер на
маске разворота 2048×1715, радиус 51 px:

    cv2.erode(MORPH_ELLIPSE)     372 мс
    distanceTransform + порог      8 мс     ← в 46 раз быстрее

Причём результат не «примерно такой же», а ТОЧНЕЕ: ``getStructuringElement``
растеризует круг в многоугольник, и по контуру расходится с настоящим кругом
(на том же замере — 923 px, 0.026% кадра). ``DIST_MASK_PRECISE`` даёт истинную
евклидову метрику — побитово то же, что эталонный ``scipy.ndimage.
distance_transform_edt``, но быстрее его в 48 раз. Приближённые маски (``DIST_MASK_3``/
``DIST_MASK_5``) не берём: они не быстрее precise, а ошибаются сильнее.

ГРАНИЦА КАДРА. Как и ``cv2.erode`` с настройками по умолчанию
(``morphologyDefaultBorderValue()`` = +inf), всё за пределами массива считается
ФОНОМ МАСКИ, то есть эрозия не подъедает маску со стороны рамки кадра. Если нужно
обратное поведение, вызывающий добавляет нулевую кайму (см. ``trim_cover_fragments``
в ``geometry``) — с distance transform для этого хватило бы и одного пикселя, но
кайма там нужна ещё и чтобы уместить промежуточный результат дилатации.

Здесь только ИЗОТРОПНЫЕ (дисковые) операции на всей маске. Анизотропная дилатация
эллипсом и покомпонентно живёт в ``finger_removal.asymmetric_dilation`` —
она построена на том же приёме, но со своим ``sampling`` по осям.
"""

import cv2
import numpy as np


def erode_disk(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Эрозия маски диском радиуса ``radius_px`` (uint8 0/255 → uint8 0/255).

    Остаются пиксели, у которых ВЕСЬ диск радиуса ``radius_px`` лежит внутри маски,
    то есть расстояние до ближайшего фонового пикселя строго больше радиуса.
    ``radius_px <= 0`` — маска возвращается как есть.
    """
    if radius_px <= 0:
        return mask
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (dist > radius_px).astype(np.uint8) * 255


def dilate_disk(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Дилатация маски диском радиуса ``radius_px`` (uint8 0/255 → uint8 0/255).

    Добавляются пиксели, удалённые от маски не больше чем на ``radius_px``; сама
    маска сохраняется (у её пикселей расстояние 0). ``radius_px <= 0`` — маска
    возвращается как есть.
    """
    if radius_px <= 0:
        return mask
    dist = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (dist <= radius_px).astype(np.uint8) * 255
