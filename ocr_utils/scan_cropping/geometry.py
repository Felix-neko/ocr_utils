"""Геометрия crop-зоны: правильный поворот, повёрнутый bbox, подготовка масок.

«Правильный поворот» ищется перебором углов вокруг центра тяжести маски разворота:
берётся угол, при котором осевой bounding box повёрнутого силуэта минимален по
площади. Всё остальное здесь считается в ЭТОЙ повёрнутой системе координат —
величина ``ext`` (minx, miny, maxx, maxy) везде задана относительно центра тяжести
и повёрнута на найденный угол.
"""

from typing import Optional

import cv2
import numpy as np

from ocr_utils.scan_cropping.morphology import dilate_disk, erode_disk
from ocr_utils.scan_cropping.page_detection import WORK_SIDE

# Поиск правильного поворота разворота: перебор углов ± предела с шагом (градусы)
ROT_RANGE_DEG = 35
ROT_STEP_DEG = 1


# Доп. «обрезка» краёв силуэта книги перед копированием, пикс. Маска страницы на
# тёмном фоне захватывает не только светлые страницы, но и куски сравнительно
# тёмной обложки подшивки у краёв/углов. Просто взять min-area bbox и отступить
# внутрь мало: книга не прямая, и в углах B2 всё равно остаются тёмные фрагменты
# обложки. Поэтому область КОПИРОВАНИЯ (E2) получаем из маски (E1) морфологией
# «диляция на extra + эрозия на 2*extra» — это закрытие мелких вырезов + чистый
# сдвиг края внутрь на extra: периферийные слои обложки срезаются, а то, что в B2
# вне E2, заливается усреднённым светлым цветом страницы. 0 — выключить.
EXTRA_EROSION_PX = 80


def rotation_matrix(angle_deg: float) -> np.ndarray:
    """Матрица поворота 2×2 на ``angle_deg`` градусов."""
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def min_area_rotated_bbox(mask: np.ndarray) -> Optional[tuple]:
    """Возвращает (cx, cy, angle, (minx, miny, maxx, maxy)) или None.

    Центр тяжести — среднее X и Y по всем пикселям маски. Перебираем углы поворота
    вокруг центра и берём тот, у которого осевой bbox повёрнутых точек минимален по
    площади. ``ext`` — в повёрнутой системе координат (относительно центра).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float64)  # (N, 2) в (x, y)
    rel = pts - np.array([cx, cy])

    best = None
    for ang in range(-ROT_RANGE_DEG, ROT_RANGE_DEG + 1, ROT_STEP_DEG):
        rot = rel @ rotation_matrix(ang).T
        mn = rot.min(axis=0)
        mx = rot.max(axis=0)
        area = (mx[0] - mn[0]) * (mx[1] - mn[1])
        if best is None or area < best[0]:
            best = (area, ang, (mn[0], mn[1], mx[0], mx[1]))

    _, angle, ext = best
    return cx, cy, angle, ext


def ext_with_margins(ext: tuple, margins: "tuple[int, int, int, int]") -> tuple:
    """Применяет припуски к ext (minx, miny, maxx, maxy): >0 расширяет наружу, <0 сжимает внутрь.

    ``margins`` = (left, top, right, bottom) — своя величина на каждую сторону
    crop-зоны (левая двигает minx, верхняя — miny, правая — maxx, нижняя — maxy).
    """
    minx, miny, maxx, maxy = ext
    left, top, right, bottom = margins
    return (minx - left, miny - top, maxx + right, maxy + bottom)


def bbox_corners(cx: float, cy: float, angle: float, ext: tuple) -> np.ndarray:
    """4 угла повёрнутого bbox в координатах исходного кадра (порядок TL,TR,BR,BL)."""
    minx, miny, maxx, maxy = ext
    corners = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]], dtype=np.float64)
    # Обратно в исходный кадр: rel = rot @ R(angle), затем + центр
    return (corners @ rotation_matrix(angle) + np.array([cx, cy])).astype(np.float32)


def ext_to_mask(shape: "tuple[int, int]", cx: float, cy: float, angle: float, ext: tuple) -> np.ndarray:
    """Бинарная маска (uint8 0/255) залитого повёрнутого bbox ``ext`` в координатах кадра."""
    m = np.zeros(shape[:2], dtype=np.uint8)
    corners = bbox_corners(cx, cy, angle, ext).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [corners], 255)
    return m


def layout_ext_bounds(
    cx: float, cy: float, angle: float, layout_mask: Optional[np.ndarray]
) -> Optional["tuple[float, float, float, float]"]:
    """Габариты блоков layout в осях crop-зоны (та же повёрнутая система, что и ``ext``).

    ``layout_mask`` — бинарная маска блоков УЖЕ с padding'ом (см. ``polygons_to_mask``).
    Контурные точки маски переводятся в повёрнутую вокруг ``(cx, cy)`` систему
    координат (как в ``min_area_rotated_bbox``) и по ним берётся осевой bbox.
    Возвращает (minx, miny, maxx, maxy) относительно центра либо None, если маска пуста.
    """
    if layout_mask is None:
        return None
    contours, _ = cv2.findContours(layout_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float64)
    rot = (pts - np.array([cx, cy])) @ rotation_matrix(angle).T
    mn = rot.min(axis=0)
    mx = rot.max(axis=0)
    return (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))


def crop_ext_with_layout(
    ext: tuple, margins: "tuple[int, int, int, int]", layout_bounds: Optional["tuple[float, float, float, float]"]
) -> tuple:
    """Финальный ext crop-зоны: ext с припусками, дополнительно расширенный под layout.

    Сначала применяются ``margins`` (могут быть и отрицательными — отступ внутрь),
    затем зона расширяется НАРУЖУ ровно настолько, чтобы целиком вместить габариты
    блоков layout (``layout_bounds`` в тех же осях). Если блоки и так внутри —
    ничего не меняется. Так отрицательные припуски не срезают часть обложки/текста,
    которую Surya распознала как контент (см. IMG_0003 с завышенными припусками).
    """
    minx, miny, maxx, maxy = ext_with_margins(ext, margins)
    if layout_bounds is not None:
        lminx, lminy, lmaxx, lmaxy = layout_bounds
        minx, miny = min(minx, lminx), min(miny, lminy)
        maxx, maxy = max(maxx, lmaxx), max(maxy, lmaxy)
    return (minx, miny, maxx, maxy)


def trim_cover_fragments(
    mask: np.ndarray, extra_erosion_px: int = EXTRA_EROSION_PX, work_side: int = WORK_SIDE
) -> np.ndarray:
    """E2 из E1: срезает периферийные фрагменты обложки, оставшиеся в маске страницы.

    К маске страницы (``mask`` = E1) применяется диляция на ``extra_erosion_px`` и
    затем эрозия на ``2 * extra_erosion_px``. Это закрытие мелких вырезов/зазубрин
    (диляция+эрозия на ту же величину) плюс чистый сдвиг края внутрь на
    ``extra_erosion_px`` (остаток эрозии): криволинейный край книги отступает
    внутрь, и тонкие слои тёмной обложки у краёв/углов (которые детектор включил в
    маску) отсекаются. Возвращает уменьшенную маску E2 (uint8 0/255) в разрешении
    исходной ``mask``.

    Морфология считается на копии, уменьшенной до ``work_side``: для «обрезки»
    краёв точность полного разрешения не нужна — граница потом всё равно у
    бумажных полей, не у текста, — а работы на 30-48 Мп кадре в разы больше.

    Сами дилатация и эрозия идут диском через distance transform
    (``morphology.dilate_disk`` / ``erode_disk``), а не ядром ``MORPH_ELLIPSE``:
    круг неразделим, и ядро радиусом ``2*extra_erosion_px`` (диаметр ~321 px при
    80) стоило секунды на кадр. Замер на этой маске: 2.7 с ядром против ~30 мс
    через distance transform, причём диск получается настоящий, а не
    растеризованный многоугольник (подробности — в докстринге ``morphology``).

    Морфология идёт на холсте, добитом нулями на ``2*d`` с каждой стороны, и
    результат обрезается обратно. Без этого маска, подходящая к рамке кадра
    ближе ``extra_erosion_px``, после диляции упирается в границу кадра, а
    эрозия (как и ``cv2.erode`` по умолчанию) считает всё за пределами холста
    ФОНОМ МАСКИ и с этой стороны маску не подъедает — оставался прилипший к
    рамке «язык», сточенный только с боков (см. IMG_0011 из ve_80s: у корешка
    верх E1 в 74 px от края кадра, и E2 вместо отступа 110 px дотягивался до
    y=0; «язык» был на всех 15 кадрах партии). Нулевой холст даёт одинаковый
    отступ по всему контуру, в том числе от рамки кадра.

    Альтернатива «считать без каймы» ПРОВЕРЕНА И ОТВЕРГНУТА: «языки» она тоже
    убирает, но диляция при этом обрезается рамкой кадра, и у самой рамки эрозия
    съедает уже не ``d``, а ``2*d`` — вместо контура книги вдоль края кадра идёт
    горизонтальная «полка» с отступом 2*``extra_erosion_px`` (на проверенных кадрах
    на 25-35 px глубже нужного, до 1.6% пикселей выходного кадра).
    """
    if extra_erosion_px <= 0:
        return mask
    h, w = mask.shape[:2]
    scale = work_side / max(h, w) if max(h, w) > work_side else 1.0
    small = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else mask
    d = max(1, int(round(extra_erosion_px * scale)))
    # Запаса 2*d хватает: диляция выносит маску за рамку максимум на d, а эрозия
    # смотрит на 2*d вокруг каждого пикселя исходной области.
    pad = 2 * d
    padded = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    out = erode_disk(dilate_disk(padded, d), 2 * d)
    out = out[pad : pad + small.shape[0], pad : pad + small.shape[1]]
    if scale < 1.0:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
    return out
