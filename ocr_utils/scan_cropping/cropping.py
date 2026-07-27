"""Вырезка crop-зоны из кадра двумя способами (см. ``--crop-mode``).

``crop_rotated`` поворачивает кадр на найденный угол и вырезает выпрямленный
прямоугольник; ``crop_pixel_exact`` копирует ту же зону пиксель-в-пиксель в
минимальный осевой холст, не трогая исходные пиксели (книга остаётся наклонённой,
выпрямление — снаружи, например в ScanTailor).
"""

import cv2
import numpy as np
from typing import Optional

from ocr_utils.scan_cropping.background_fill import (
    CROP_FILL_REPLICATE,
    blur_downscaled,
    replicate_edge_fill,
    voronoi_fill,
)
from ocr_utils.scan_cropping.geometry import bbox_corners
from ocr_utils.scan_cropping.page_detection import WORK_SIDE

# Способы вырезки crop-зоны (значения --crop-mode), см. crop_rotated / crop_pixel_exact.
CROP_MODE_ROTATE = "rotate"  # повернуть кадр на найденный угол и вырезать выпрямленный прямоугольник
CROP_MODE_PIXEL_EXACT = "pixel-exact"  # скопировать пиксель-в-пиксель в осевой холст, книга остаётся наклонённой
CROP_MODES = (CROP_MODE_ROTATE, CROP_MODE_PIXEL_EXACT)

# Параметры заливки «ушей» при --crop-mode=pixel-exact (сами способы заливки — в
# ``background_fill``, значения --crop-fill-method).
CROP_FILL_BLUR_PX = 48.0  # макс. σ размытия заливки (растёт с расстоянием от crop-bbox)
CROP_FILL_FADE = 1.0  # доля выцветания к среднему цвету книги на самом дальнем пикселе (0 — не выцветать)


def crop_rotated(
    bgr: np.ndarray, cx: float, cy: float, angle: float, crop_ext: tuple, upscale: Optional[float] = None
) -> np.ndarray:
    """Поворот вокруг центра тяжести + вырез crop-зоны → выпрямленный прямоугольник.

    ``crop_ext`` — финальный ext crop-зоны (уже с припусками и расширением под
    layout, см. ``crop_ext_with_layout``). Берём 4 угла crop-зоны в исходном кадре
    и перспективным преобразованием отображаем их в осевой прямоугольник нужного
    размера (это и есть поворот кадра на найденный угол с одновременным вырезом
    области). ``upscale`` увеличивает только выходной холст (источник сэмплирования —
    всегда исходный полноразмерный кадр), поэтому апскейл получается за один
    интерполяционный проход, без потерь от промежуточного ресайза целого кадра.
    ``None`` — апскейл вообще не считается (экономит время: без умножения размеров и
    без INTER_CUBIC).
    """
    minx, miny, maxx, maxy = crop_ext
    if upscale is None:
        out_w = max(1, int(round(maxx - minx)))
        out_h = max(1, int(round(maxy - miny)))
        flags = cv2.INTER_LINEAR
    else:
        out_w = max(1, int(round((maxx - minx) * upscale)))
        out_h = max(1, int(round((maxy - miny) * upscale)))
        flags = cv2.INTER_CUBIC
    src = bbox_corners(cx, cy, angle, (minx, miny, maxx, maxy))
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, m, (out_w, out_h), flags=flags)


def crop_pixel_exact(
    bgr: np.ndarray,
    cx: float,
    cy: float,
    angle: float,
    crop_ext: tuple,
    fade_color: Optional[np.ndarray] = None,
    blur_px: float = CROP_FILL_BLUR_PX,
    fade_strength: float = CROP_FILL_FADE,
    fill_method: str = CROP_FILL_REPLICATE,
    content_mask: Optional[np.ndarray] = None,
    work_side: int = WORK_SIDE,
) -> np.ndarray:
    """Вырез crop-зоны БЕЗ поворота: пиксель-в-пиксель, книга остаётся наклонённой.

    Альтернатива ``crop_rotated`` (см. ``--crop-mode``). ``crop_rotated`` пересэмплирует
    ВЕСЬ кадр интерполяцией — на скромном разрешении и заметном угле это слегка мылит
    текст, а лечится только апскейлом (и раздутым файлом). Здесь исходные пиксели не
    трогаются вовсе: берётся минимальный ОСЕВОЙ bbox, в который вписан повёрнутый
    crop-bbox, и содержимое копируется из кадра как есть. Выпрямлять разворот в этом
    режиме предполагается снаружи (ScanTailor), уже по неиспорченным пикселям.

    Цена — «уши» между наклонённым crop-bbox и осевым холстом (тем больше, чем больше
    угол). Они не обрезаются, а заполняются так, чтобы не мозолить глаз и не сбивать
    последующую обработку:
      1. базовая заливка, способ ``fill_method`` (см. ``CROP_FILL_METHODS``):
         ``replicate`` — краевые пиксели продлеваются наружу по осям crop-зоны, т.е. по
         НОРМАЛИ к её сторонам (``replicate_edge_fill``); ``voronoi`` — цвет ближайшей
         точки границы (``voronoi_fill``). Разница важна для линий, выходящих из зоны
         (корешок разворота): replicate продолжает их прямо, Вороной у выпуклых углов
         границы расходится веером и загибает их — см. ``replicate_edge_fill``;
      2. размытие, растущее с расстоянием от crop-bbox (σ до ``blur_px``): у шва резко,
         вдали — гладко. ВНИМАНИЕ: размытие смазывает и продолженную линию корешка,
         поэтому под разбивку в ScanTailor его лучше держать в нуле;
      3. выцветание к ``fade_color`` (средний цвет книги, см. ``book_mean_color``) —
         линейно по расстоянию, на самом дальнем пикселе доля ``fade_strength``
         (1.0 — уходит в средний цвет полностью, 0 — не выцветать).
    Расстояние нормируется на максимальное в самих «ушах», поэтому и размытие, и
    выцветание доходят до конца при любом угле и размере кадра.

    Часть осевого холста может выйти за границы исходного кадра (при положительных
    припусках) — эти пиксели считаются неизвестными наравне с «ушами» и заполняются
    так же, а не остаются чёрными.

    ``content_mask`` (в координатах КАДРА) — область настоящего контента, обычно E2
    (силуэт книги после ``trim_cover_fragments``). Если задана, «известным» считается
    её пересечение с crop-bbox, и заливка идёт от края КНИГИ, а не от края bbox. Это
    важно: между краем книги и краем crop-зоны обычно лежит полоса в десятки пикселей
    (припуски меньше ``--extra-erosion-px``), и без ``content_mask`` её пришлось бы
    заполнять отдельно — в ``fill_outside_mask``, где осей crop-зоны нет и заливка
    Вороного веером загибает линию корешка (см. ``replicate_edge_fill``).
    """
    corners = bbox_corners(cx, cy, angle, crop_ext)
    x0, y0 = int(np.floor(corners[:, 0].min())), int(np.floor(corners[:, 1].min()))
    x1, y1 = int(np.ceil(corners[:, 0].max())), int(np.ceil(corners[:, 1].max()))
    out_w, out_h = max(1, x1 - x0), max(1, y1 - y0)

    h, w = bgr.shape[:2]
    out = np.zeros((out_h, out_w, 3), dtype=bgr.dtype)
    # Пиксель-в-пиксель: пересечение осевого bbox с кадром копируется срезом, без ресэмплинга.
    valid = np.zeros((out_h, out_w), dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = bgr[sy0:sy1, sx0:sx1]
        valid[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = 255

    box = np.zeros((out_h, out_w), dtype=np.uint8)
    cv2.fillPoly(box, [np.round(corners - np.array([x0, y0], dtype=np.float32)).astype(np.int32)], 255)
    known = cv2.bitwise_and(box, valid)
    if content_mask is not None:
        content = np.zeros((out_h, out_w), dtype=np.uint8)
        if sx1 > sx0 and sy1 > sy0:
            content[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = content_mask[sy0:sy1, sx0:sx1]
        if np.any(cv2.bitwise_and(known, content)):  # пустое пересечение — заливать не от чего
            known = cv2.bitwise_and(known, content)
    outside = known == 0
    if not np.any(known) or not np.any(outside):
        return out

    if fill_method == CROP_FILL_REPLICATE:
        filled = replicate_edge_fill(out, known, angle)
    else:
        filled = voronoi_fill(out, known, work_side)

    # Одна карта расстояний до crop-bbox на оба эффекта. Нормируем её на максимум
    # внутри «ушей»: их глубина зависит от угла и размера кадра, и без нормировки
    # (как в distance_weighted_blur, где вес насыщается только к 4σ) на мелких
    # «ушах» и размытие, и выцветание не успевали бы набрать силу.
    dist = cv2.distanceTransform((known == 0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_max = float(dist.max())
    if dist_max > 0 and (blur_px > 0 or (fade_color is not None and fade_strength > 0)):
        # Размытие и выцветание считаем ТОЛЬКО в заполняемых пикселях: наружу
        # уходит ровно ``filled[outside]``, а его обычно 0.3-0.5% холста. Раньше
        # весь холст разворачивался во float32 (30 МБ → 120 МБ) и по нему шло
        # несколько проходов, из которых 99.5% результата выбрасывалось.
        # Арифметика поэлементная и на тех же типах, так что значения те же.
        vals = filled[outside].astype(np.float32)
        norm = (dist[outside] / dist_max).astype(np.float32)[:, None]
        if blur_px > 0:
            # Размытая версия нужна целиком (ядро тянет соседей из известной зоны),
            # поэтому её единственную считаем по всему холсту; выбираем из неё опять
            # же только наши пиксели. У шва резко (вес 0 — заливка стыкуется с краем
            # страницы без ореола), на самом дальнем пикселе — полное размытие σ=blur_px.
            #
            # ВАЖНО: на вход размытию идёт float32, а не uint8. Внутри
            # blur_downscaled кадр уменьшается, размывается и растягивается обратно;
            # на uint8 каждый из этих шагов округляется, и цвет заливки расходится с
            # прежним (проверено: при --crop-fill-method=voronoi расхождение видно
            # на глаз в diff). Приведение типа стоит один проход по холсту — это
            # всё ещё в разы дешевле прежнего варианта, который гонял во float32
            # весь холст и делал по нему несколько проходов смешивания.
            blurred = blur_downscaled(filled.astype(np.float32), blur_px, work_side)
            vals = vals * (1.0 - norm) + blurred[outside] * norm
        if fade_color is not None and fade_strength > 0:
            alpha = float(fade_strength) * norm
            target = np.asarray(fade_color, dtype=np.float32).reshape(1, 3)
            vals = vals * (1.0 - alpha) + target * alpha
        out[outside] = np.clip(vals, 0, 255).astype(np.uint8)
    else:
        out[outside] = filled[outside]
    return out
