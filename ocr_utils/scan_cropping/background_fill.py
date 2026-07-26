"""Заливка областей кадра, где нет содержимого книги.

Два разных «снаружи» и потому два семейства функций:

* ``fill_outside_mask`` — всё вне силуэта книги в ИСХОДНОМ кадре, перед
  rotated-crop. Криволинейная маска страницы не совпадает с осевым min-area bbox,
  и в углы повёрнутого кропа иначе попадает чёрный фон.
* ``replicate_edge_fill`` / ``voronoi_fill`` — «уши» между наклонённым crop-bbox и
  осевым холстом при ``--crop-mode=pixel-exact`` (см. ``cropping.crop_pixel_exact``).

Общий приём всех локальных методов: цвет берётся у самой границы известной области
и продлевается наружу — так воспроизводится и неравномерный свет, и цветная
обложка. Считается на уменьшенной до ``WORK_SIDE`` копии: заливка гладкая по
построению, а distance transform и эрозия на кадрах 30-48 Мп заметно тормозят.
"""

import cv2
import numpy as np
from typing import Optional

from ocr_utils.scan_cropping.geometry import rotation_matrix
from ocr_utils.scan_cropping.page_detection import WORK_SIDE

# Заливка фона за пределами силуэта книги (перед rotated-crop): эрозия маски
# книги перед расчётом цвета заливки, пикс. — чтобы источник цвета не захватывал
# шумную/смазанную границу силуэта (там же соседствует фон).
BG_FILL_EROSION_PX = 100

# Способы заливки внешней зоны (значения --bg-fill-method). Все, кроме average,
# берут цвет из приграничной полосы страницы и локально продлевают его наружу —
# так воспроизводится и неравномерный свет, и цветная обложка (см.
# background_fill_extrapolation_report.md). Считаем, что на краю страницы текста
# нет, поэтому источник цвета чистый и подавление чернил не нужно.
BG_FILL_AVERAGE = "average"  # один усреднённый цвет по всей странице (старый способ)
BG_FILL_NEAREST = "nearest"  # цвет ближайшего пикселя границы E2 (Вороной, distance transform)
BG_FILL_METHODS = (BG_FILL_AVERAGE, BG_FILL_NEAREST)


# Заполнение «ушей» между наклонённым crop-bbox и осевым холстом (--crop-mode=pixel-exact).
# replicate — продление краевых пикселей bbox наружу по нормали к его сторонам
# (clamp-to-edge/BORDER_REPLICATE в осях bbox): линии, выходящие из crop-зоны (корешок
# разворота), продолжаются прямо. voronoi — цвет ближайшей точки границы bbox: у углов
# bbox расходится веером и загибает такие линии, ломая разбивку разворота в ScanTailor.
CROP_FILL_REPLICATE = "replicate"
CROP_FILL_VORONOI = "voronoi"
CROP_FILL_METHODS = (CROP_FILL_REPLICATE, CROP_FILL_VORONOI)


def _eroded_mean_color(bgr: np.ndarray, mask: np.ndarray, erosion_px: int) -> np.ndarray:
    """Средний цвет ``bgr`` внутри ``mask``, эрозированной на ``erosion_px`` (BGR float (3,)).

    Эрозия — чтобы в среднее не попал шумный край силуэта и подтёкший из-за края
    тёмный фон. Если эрозия съела маску целиком (узкая область), берётся исходная
    маска. ``erosion_px`` — в том же разрешении, в котором переданы ``bgr``/``mask``.
    """
    sample_sel = mask
    if erosion_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px * 2 + 1, erosion_px * 2 + 1))
        eroded = cv2.erode(mask, k)
        if np.any(eroded > 0):
            sample_sel = eroded
    return bgr[sample_sel > 0].mean(axis=0)


def book_mean_color(
    bgr: np.ndarray, mask: np.ndarray, erosion_px: int = BG_FILL_EROSION_PX, work_side: int = WORK_SIDE
) -> Optional[np.ndarray]:
    """Средний цвет области книги (``mask``, сильно эрозированной) — BGR float (3,) или None.

    Тот же способ, что даёт цвет заливки в ``fill_outside_mask(method='average')``:
    сильная эрозия отсекает край силуэта, и остаётся «чистая бумага/обложка».
    Считается на копии, уменьшенной до ``work_side`` (среднее по 30-48 Мп маске —
    заметная и лишняя трата). ``None``, если маска пуста.
    """
    if not np.any(mask > 0):
        return None
    h, w = mask.shape[:2]
    scale = work_side / max(h, w) if max(h, w) > work_side else 1.0
    if scale < 1.0:
        small_mask = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
        small_bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        erosion_px = max(1, int(round(erosion_px * scale)))
    else:
        small_mask, small_bgr = mask, bgr
    if not np.any(small_mask > 0):  # маска исчезла при уменьшении
        return None
    return _eroded_mean_color(small_bgr, small_mask, erosion_px)


def nearest_edge_fill(small_bgr: np.ndarray, e2_mask: np.ndarray) -> np.ndarray:
    """Заливка «по Вороному»: каждому пикселю — цвет ближайшего пикселя границы E2.

    Ближайший пиксель E2 для точки снаружи всегда лежит на границе E2, поэтому это и
    есть «цвет ближайшей точки границы зоны копирования». ``distance_transform_edt``
    с ``return_indices`` даёт индексы ближайшего известного (нулевого) пикселя — O(N),
    без перебора границы. ``e2_mask`` — маска E2 (0 вне). Возвращает BGR uint8.
    Минус метода — ступеньки на медиальной оси (где ближайшая точка границы
    переключается); лечится размытием заливки (см. ``distance_weighted_blur``).
    """
    from scipy.ndimage import distance_transform_edt

    iy, ix = distance_transform_edt(e2_mask == 0, return_indices=True, return_distances=False)
    return small_bgr[iy, ix]


def distance_weighted_blur(img: np.ndarray, e2_mask: np.ndarray, max_sigma: float) -> np.ndarray:
    """Размывает ``img`` тем сильнее, чем дальше пиксель от зоны копирования ``e2_mask``.

    У самой границы E2 размытия нет (вес 0) — так шов остаётся непрерывным, а ядро
    приграничного пикселя почти не залезает в E2 (нет ореола от контента у края).
    Вдали вес растёт до 1 (полное размытие ``max_sigma``) за ``~4·max_sigma`` пикселей
    от границы. Заливка — гладкий цвет без структуры, поэтому линейного бленда
    «резкая ⊕ сильно размытая» достаточно (двоения не даёт), собран на встроенных
    ``cv2.distanceTransform`` + ``cv2.blendLinear``. ``img`` BGR uint8, ``e2_mask`` —
    маска E2 (0 вне). Пиксели E2 вызывающий НЕ переписывает, поэтому E2 нетронута.
    """
    if max_sigma < 0.5:
        return img
    # Расстояние до ближайшего пикселя E2 (0 внутри E2, растёт наружу) → вес размытия.
    d = cv2.distanceTransform((e2_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(d / (4.0 * max_sigma), 0.0, 1.0).astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), float(max_sigma))
    # blendLinear: (w1·s1 + w2·s2)/(w1+w2); w1+w2=1 → попиксельный лерп по alpha.
    return cv2.blendLinear(img, blurred, 1.0 - alpha, alpha)


def fill_outside_mask(
    bgr: np.ndarray,
    mask: np.ndarray,
    erosion_px: int = BG_FILL_EROSION_PX,
    work_side: int = WORK_SIDE,
    method: str = BG_FILL_AVERAGE,
    blur_px: float = 0.0,
) -> np.ndarray:
    """Закрашивает всё вне ``mask`` цветом бумаги/обложки внутри неё.

    Криволинейная маска страницы не идеально совпадает с осевым min-area bbox
    (неровные/загнутые края) — в углы повёрнутого кропа может попасть кусок
    чёрного фона. Заранее закрасив фон, получаем ровный угол вместо чёрного пятна,
    даже если crop-зона чуть шире силуэта.

    ``method`` — способ заливки (см. ``BG_FILL_METHODS``):

    - ``average`` — один усреднённый цвет по всей странице (старый способ): дёшево,
      но не учитывает ни неравномерный свет, ни цветную обложку. Цвет усредняется по
      маске, эрозированной на ``erosion_px`` — чтобы шумный край не сдвигал среднее.
    - ``nearest`` — цвет ближайшей точки границы E2 (см. ``nearest_edge_fill``): без
      эрозии (цвет нужен у самой границы), учитывает неравномерный свет и цветную
      обложку. Опционально сглаживается размытием (``blur_px``).

    ``blur_px`` (>0, только для локальных методов) — переменное размытие заливки: у
    границы E2 нуль, вдали до σ=``blur_px`` (см. ``distance_weighted_blur``). Зона
    копирования при этом остаётся нетронутой (переписываем только пиксели вне E2).

    Всё считается на копии, уменьшенной до ``work_side`` (эрозия и экстраполяция на
    кадрах 30-48 Мп заметно тормозят), к полному разрешению применяется только
    подстановка заполненного цвета во внешнюю зону.
    """
    sel = mask > 0
    if not np.any(sel):
        return bgr

    h, w = mask.shape[:2]
    scale = work_side / max(h, w) if max(h, w) > work_side else 1.0
    if scale < 1.0:
        small_mask = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
        small_bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_mask, small_bgr = mask, bgr

    if method == BG_FILL_AVERAGE:
        # Источник среднего цвета — маска, эрозированная на erosion_px (без шумного
        # края/подтёкшего фона). Локальным методам эрозия не нужна: они берут цвет у
        # самой границы E2 (уже обрезанной trim_cover_fragments).
        small_erosion_px = max(1, int(round(erosion_px * scale))) if scale < 1.0 else erosion_px
        avg_color = _eroded_mean_color(small_bgr, small_mask, small_erosion_px)
        out = bgr.copy()
        out[~sel] = avg_color.astype(np.uint8)
        return out

    filled_small = nearest_edge_fill(small_bgr, small_mask)
    if blur_px > 0:
        # Размытие тем сильнее, чем дальше от границы E2 (на downscale).
        filled_small = distance_weighted_blur(filled_small, small_mask, blur_px * scale)
    # Карта заливки гладкая (без деталей) — обычного билинейного апскейла достаточно.
    filled = cv2.resize(filled_small, (w, h), interpolation=cv2.INTER_LINEAR) if scale < 1.0 else filled_small
    out = bgr.copy()
    out[~sel] = filled[~sel]
    return out


def voronoi_fill(canvas: np.ndarray, known: np.ndarray, work_side: int) -> np.ndarray:
    """Заливка холста «по Вороному» от области ``known`` (см. ``nearest_edge_fill``).

    Считается на копии, уменьшенной до ``work_side``, и растягивается обратно:
    заливка гладкая по построению, а ``distance_transform_edt`` на 30-48 Мп заметно
    тормозит. Возвращает BGR uint8 в размер ``canvas``.
    """
    h, w = canvas.shape[:2]
    scale = work_side / max(h, w) if max(h, w) > work_side else 1.0
    if scale < 1.0:
        size = (int(w * scale), int(h * scale))
        small_known = cv2.resize(known, size, interpolation=cv2.INTER_NEAREST)
        if np.any(small_known):  # если «уши» тоньше шага уменьшения — считаем в полный размер
            small_canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
            return cv2.resize(nearest_edge_fill(small_canvas, small_known), (w, h), interpolation=cv2.INTER_LINEAR)
    return nearest_edge_fill(canvas, known)


def clamp_to_edge(img: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Двумерный clamp-to-edge: краевые пиксели ``known`` продлеваются наружу по осям.

    Для каждого неизвестного пикселя берётся ближайший известный ПО ОСИ — по столбцу
    (продление вверх/вниз) либо по строке (влево/вправо), смотря что ближе:
      * индекс строки зажимается между первой и последней известной строкой ЕГО столбца,
        индекс колонки — между первой и последней известной колонкой ЕГО строки;
      * из двух вариантов берётся тот, где идти ближе (если один невозможен — другой);
      * если ни в строке, ни в столбце известного нет (углы) — строка добирается
        вертикальным продлением от ближайшей строки-донора.

    Оба «зажима» обязаны считаться по известным пикселям именно своей строки/своего
    столбца, и выбор между ними — по расстоянию. Граница книги (E2) криволинейна: у
    нижних строк она уходит правее края crop-зоны, и столбец там известен только сверху.
    Если в таком столбце всё равно продлевать вертикально, цвет берётся с далёкого
    верхнего пикселя — в выходном кадре это давало резкую светлую полосу вдоль левого
    «уха» (IMG_0042), хотя настоящий край книги был в паре пикселей сбоку.

    ``img`` BGR, ``known`` — маска известного (uint8 0/255) того же размера.
    """
    known_b = known > 0
    h, w = known_b.shape
    rows = np.arange(h, dtype=np.int32)[:, None]
    cols = np.arange(w, dtype=np.int32)[None, :]

    has_col = known_b.any(axis=0)
    first_r = np.argmax(known_b, axis=0).astype(np.int32)
    last_r = (h - 1 - np.argmax(known_b[::-1], axis=0)).astype(np.int32)
    src_r = np.clip(rows, first_r[None, :], last_r[None, :])
    vert = np.take_along_axis(img, src_r[..., None].astype(np.intp), axis=0)
    if has_col.all():
        return vert

    has_row = known_b.any(axis=1)
    first_c = np.argmax(known_b, axis=1).astype(np.int32)
    last_c = (w - 1 - np.argmax(known_b[:, ::-1], axis=1)).astype(np.int32)
    src_c = np.clip(cols, first_c[:, None], last_c[:, None])
    horz = np.take_along_axis(img, src_c[..., None].astype(np.intp), axis=1)

    # Кому идти ближе: вверх/вниз по столбцу или вбок по строке.
    dist_v = np.abs(src_r - rows)
    dist_h = np.abs(src_c - cols)
    use_v = has_col[None, :] & (~has_row[:, None] | (dist_v <= dist_h))
    out = np.where(use_v[..., None], vert, horz)

    if not has_row.all():
        donor = np.broadcast_to(has_row[:, None], (h, w))
        first_d = np.argmax(donor, axis=0)
        last_d = h - 1 - np.argmax(donor[::-1], axis=0)
        out = np.take_along_axis(
            out, np.clip(rows, first_d[None, :], last_d[None, :])[..., None].astype(np.intp), axis=0
        )
    return out


def replicate_edge_fill(canvas: np.ndarray, known: np.ndarray, angle: float) -> np.ndarray:
    """Продление краевых пикселей ``known`` НАРУЖУ ПО ОСЯМ CROP-ЗОНЫ (clamp-to-edge).

    Это обычная replicate-экстраполяция края (``cv2.BORDER_REPLICATE``, np.pad(mode=
    'edge')), только выполненная не в осях кадра, а в осях повёрнутой crop-зоны: холст
    поворачивается на ``-angle``, там край продлевается по столбцам/строкам
    (``clamp_to_edge``), и результат поворачивается обратно. Для верхней/нижней
    стороны это в точности «краевой пиксель поднимается перпендикулярно стороне».

    Зачем это вместо ``voronoi_fill``. Вороной тянет цвет ближайшей точки границы, и у
    выпуклых углов границы (угол crop-зоны, край страницы у корешка) ближайшей для целой
    области оказывается ОДНА точка — заливка расходится оттуда веером. Тёмная линия
    корешка, выходящая из зоны, в таком веере загибается, и ScanTailor перестаёт
    находить по ней разрез разворота (см. IMG_0004/IMG_0034 из ve_80s). Clamp-to-edge
    продолжает её прямо — по нормали к стороне crop-зоны, т.е. в выходном кадре под тем
    же наклоном, под которым лежит книга.

    Поворот считается интерполяцией, но результат берётся ТОЛЬКО вне ``known``, поэтому
    исходные пиксели crop-зоны это не затрагивает.
    """
    h, w = canvas.shape[:2]
    r = rotation_matrix(angle)
    cos_a, sin_a = abs(float(r[0, 0])), abs(float(r[0, 1]))
    rw = int(np.ceil(w * cos_a + h * sin_a))
    rh = int(np.ceil(w * sin_a + h * cos_a))
    # Аффинное преобразование холст → оси crop-зоны (local = R @ rel), с центрированием.
    m = np.zeros((2, 3), dtype=np.float64)
    m[:, :2] = r
    m[:, 2] = np.array([rw / 2.0, rh / 2.0]) - r @ np.array([w / 2.0, h / 2.0])

    # Перед поворотом убираем неизвестные (чёрные) пиксели тем же продлением в осях
    # холста: иначе билинейная интерполяция поворота размажет их внутрь известной зоны,
    # и продление вынесет эту грязь наружу.
    base = clamp_to_edge(canvas, known)
    rot_img = cv2.warpAffine(base, m, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rot_known = cv2.warpAffine(known, m, (rw, rh), flags=cv2.INTER_NEAREST)
    if not np.any(rot_known):  # поворот «потерял» тонкую маску — продлеваем без него
        return base
    rot_filled = clamp_to_edge(rot_img, rot_known)
    return cv2.warpAffine(
        rot_filled, cv2.invertAffineTransform(m), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def blur_downscaled(img: np.ndarray, sigma: float, work_side: int) -> np.ndarray:
    """Гауссово размытие через уменьшенную копию (σ на 30-48 Мп стоит секунды).

    Результат размытия гладкий, поэтому уменьшение/растяжение на нём не сказывается.
    """
    h, w = img.shape[:2]
    scale = work_side / max(h, w) if max(h, w) > work_side else 1.0
    if scale < 1.0:
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), max(float(sigma * scale), 0.5))
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (0, 0), max(float(sigma), 0.5))
