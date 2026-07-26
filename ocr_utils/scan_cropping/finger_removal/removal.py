"""Удаление придерживающего страницу пальца из кадра: детекция → закраска.

Верхний уровень пакета ``finger_removal``: собирает вместе построение маски
(``masking``), защиту контента от закраски (``text_protection``) и сам инпейнтинг
(``GpuModels.inpaint``). Вызывается из ``scan_cropping.pipeline`` ДО детекции
разворота — палец искажает силуэт книги и попадает в кроп, поэтому убирать его
надо раньше всего остального.
"""

import cv2
import numpy as np
from typing import Optional

from ocr_utils.scan_cropping.finger_removal.asymmetric_dilation import DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO
from ocr_utils.scan_cropping.finger_removal.inpaint_roi import roi_bounds_list
from ocr_utils.scan_cropping.finger_removal.masking import (
    build_finger_mask,
    drop_fingers_on_content,
    keep_border_components,
)
from ocr_utils.scan_cropping.finger_removal.text_protection import (
    DEFAULT_LAYOUT_PAD_PX,
    PROTECT_COPY_BACK,
    PROTECT_LIMIT_LAMA,
    copy_back_layout,
    layout_polygons,
    limit_paint_zone,
    polygons_to_mask,
)
from ocr_utils.timing import log_timing


# Низкий порог нужен для recall (слабые боксы на смазанных/неярких пальцах, см.
# IMG_0028.jpg — лучший бокс conf=0.046, ниже стандартного 0.05); раздутая маска
# была из-за скин-добора и невложенных дублей боксов — то и другое уже устранено
# (skin-добор убран, _suppress_nested_boxes в neural_hand_mask), так что низкий
# conf теперь безопасен.
FINGER_CONF = 0.01
# Дилатация маски пальца (build_finger_mask default=12) — тонкая мягкая тень по
# краю силуэта (полутона на стыке кожа/бумага) иначе не докрашивается.
FINGER_DILATE_PX = 40
# Доля кадра для проверки контакта с рамкой в keep_border_components. Настоящий
# палец физически ОБРЕЗАН рамкой кадра (рука уходит за границу снимка), поэтому
# его маска доходит почти до самого края (~0 px). Узкая полоса надёжнее широкой:
# при 0.12 на 36-Мп сканах полоса ~430 px, и в неё попадают внутренние тёмные
# иллюстрации/фото у верхнего/бокового поля, ошибочно принятые YOLO-World за руку
# (см. IMG_0109.jpg: карта СССР в эмблеме «50 ЛЕТ СОЮЗА ССР» — 408 px от верха,
# 11.4 % высоты — пролезала впритык под 12 %). Настоящий палец здесь на 0 %,
# так что зазор огромный, 4 % чисто разделяет случаи.
FINGER_EDGE_FRAC = 0.04
FINGER_PADDING = 64  # контекст вокруг маски пальца для LaMa, пикс. (см. inpaint_roi.py)
# ROI для LaMa увеличивается в FINGER_ROI_SCALE раз от центра (после padding) —
# без этого LaMa не видит достаточно кромки/фона и заливает дыру доминирующим
# цветом (см. inpaint_roi.py, коммит "Сделали хороший закрас с помощью lama").
FINGER_ROI_SCALE = 1.5
# LaMa заливает область пальца заметно ТЕМНЕЕ окружающей бумаги (проверено на
# нескольких кадрах: разница ~25-35 отн. ед. яркости у самой маски). Поэтому
# перед закраской осветляем зону пальца — плавно, чтобы не было резкой границы:
# полный инкремент внутри самой маски (она уже включает дилатацию на
# FINGER_DILATE_PX), спад до нуля к границе маски + ещё 2×FINGER_DILATE_PX наружу
# (эта кайма — как раз тот контекст, по которому LaMa восстанавливает цвет дыры).
# Значение 20 подобрано по серии кадров из /mnt/system/raw/mts/cropped/1972 —
# заметно снижает остаточное потемнение, не давая цветового ухода в оранжевый
# (при 25-30 на тонированной («состаренной») бумаге появляется через чур тёплый оттенок).
FINGER_ZONE_LIGHT_INCREMENT = 20


def brighten_finger_zone(
    rgb: np.ndarray, mask: np.ndarray, increment: "float | tuple[float, float]", falloff_px: int
) -> np.ndarray:
    """Осветляет зону пальца перед закраской (см. ``FINGER_ZONE_LIGHT_INCREMENT``).

    Внутри ``mask`` — полный ``increment``; далее вес плавно (линейно по
    расстоянию) спадает до 0 на удалении ``falloff_px`` от границы маски.
    Прибавляется поровну ко всем каналам — контраст-нейтрально (не искажает
    цветовой баланс сам по себе), итоговый цвет заливки всё равно определяет LaMa.

    ``increment`` — одно число (одинаково для всего кадра) либо пара
    ``(слева, справа)``: свет в кадре может падать не симметрично, и тогда
    правая и левая половины разворота требуют разной компенсации (см.
    ``--finger-zone-light-increment``). Компонента маски относится к той
    половине, где лежит центр её масс.
    """
    if int(np.count_nonzero(mask)) == 0:
        return rgb
    left_inc, right_inc = increment if isinstance(increment, tuple) else (increment, increment)
    if left_inc <= 0 and right_inc <= 0:
        return rgb
    h, w = mask.shape[:2]
    num, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    out = rgb.astype(np.float32)
    for i in range(1, num):
        inside = labels == i
        _, xs = np.where(inside)
        inc = left_inc if xs.mean() < w / 2 else right_inc
        if inc <= 0:
            continue
        if falloff_px > 0:
            dist = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 5)
            weight = np.clip(1.0 - dist / falloff_px, 0.0, 1.0)
            weight[inside] = 1.0
        else:
            weight = inside.astype(np.float32)
        out += weight[..., None] * inc
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_fingers(
    bgr: np.ndarray,
    models,
    conf: float = FINGER_CONF,
    want_boxes: bool = False,
    dilate_px: int = FINGER_DILATE_PX,
    light_increment: "float | tuple[float, float]" = FINGER_ZONE_LIGHT_INCREMENT,
    asymmetric_dilation_ratio: float = DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO,
    protect_text: bool = False,
    protect_mode: str = PROTECT_LIMIT_LAMA,
    layout_pad_px: "int | tuple[int, int]" = DEFAULT_LAYOUT_PAD_PX,
    log_name: str = "",
) -> tuple[np.ndarray, np.ndarray, Optional[list], Optional[np.ndarray], str, Optional[np.ndarray], Optional[list]]:
    """Детектирует и закрашивает пальцы (finger_removal.masking + GpuModels.inpaint) в BGR-кадре.

    Возвращает (bgr, finger_mask, lama_roi_bboxes, yolo_boxes, info, finger_mask_predilate,
    layout_polys) — маска,
    список ROI-боксов LaMa (по одному на компоненту маски) и боксы YOLO-World
    нужны только для debug-оверлея, на итоговый bgr не влияют. ``yolo_boxes``
    берётся из ``build_finger_mask(..., return_boxes=True)`` — та же самая
    детекция, что уже нужна для маски, без повторного прогона YOLO-World
    (раньше эти боксы для debug-оверлея считались отдельным, дублирующим
    вызовом ``finger_yolo_boxes``). Возвращается только при ``want_boxes=True``
    (т.е. когда включён ``--debug-dir``), а не всегда, просто чтобы не тащить
    в debug-неактуальные боксы через весь пайплайн.

    При ``protect_text=True`` кадр (ДО закраски) прогоняется через Surya layout, и
    найденные блоки защищаются от закраски способом ``protect_mode``: урезанием
    зоны закраски (``limit-lama-zone``) либо копированием блоков обратно с
    оригинала уже ПОСЛЕ закраски (``copy-back-layout-zones``) — см.
    ``finger_removal.text_protection``. Возвращаемые ``layout_polys`` нужны только
    для debug-оверлея.

    Палец может исказить детекцию разворота и итоговый кроп, поэтому закраска
    выполняется до ``page_mask``/``crop_rotated``. ``build_finger_mask("auto", ...)``
    не проверяет контакт нейромаски с рамкой кадра — из-за этого крупные ФОТО
    людей/рук на самой странице (в глубине кадра, не с края) иногда ложно
    принимаются за палец. Настоящий палец всегда входит С КРАЯ кадра, поэтому
    дополнительно отсекаем компоненты, не касающиеся рамки, через
    ``keep_border_components``. Перед самой закраской зона пальца осветляется
    (``brighten_finger_zone``) — LaMa иначе заливает дыру заметно темнее
    окружающей бумаги. Если палец не найден — кадр возвращается без
    изменений.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with log_timing("build_finger_mask", log_name):
        mask, info, raw_boxes, mask_predilate = build_finger_mask(
            rgb,
            models,
            method="auto",
            conf=conf,
            dilate_px=dilate_px,
            return_boxes=True,
            return_predilate=True,
            asymmetric_dilation_ratio=asymmetric_dilation_ratio,
            log_name=log_name,
        )
    yolo_boxes = raw_boxes if want_boxes else None
    predilate = mask_predilate if want_boxes else None
    if int(np.count_nonzero(mask)) > 0:
        mask = keep_border_components(mask, edge_frac=FINGER_EDGE_FRAC)
        if int(np.count_nonzero(mask)) == 0:
            info = "auto(отсеяно: не у края)"
    if int(np.count_nonzero(mask)) == 0:
        return bgr, mask, None, yolo_boxes, info, predilate, None

    layout_polys: Optional[list] = None
    if protect_text:
        with log_timing("layout_polygons", log_name):
            layout_polys = layout_polygons(rgb, models)
        info = f"{info}, layout: блоков={len(layout_polys)}"
        if layout_polys:
            layout_mask = polygons_to_mask(mask.shape, layout_polys, layout_pad_px)
            # Ложные «пальцы» на печатном контенте (лица на портретах) убираем из
            # маски ещё ДО закраски — иначе LaMa затрёт сам контент. Работает в
            # обоих режимах защиты (и copy-back, и limit-lama).
            with log_timing("drop_fingers_on_content", log_name):
                mask, mask_predilate, dropped = drop_fingers_on_content(
                    mask, mask_predilate, layout_mask, dilate_px, asymmetric_dilation_ratio, FINGER_EDGE_FRAC
                )
            if dropped:
                info = f"{info}, ложных пальцев на контенте убрано={dropped}"
                if want_boxes:
                    predilate = mask_predilate
                if int(np.count_nonzero(mask)) == 0:
                    return bgr, mask, None, yolo_boxes, info, predilate, layout_polys
            if protect_mode == PROTECT_LIMIT_LAMA:
                before = int(np.count_nonzero(mask))
                mask = limit_paint_zone(mask, mask_predilate, layout_mask)
                after = int(np.count_nonzero(mask))
                info = f"{info}, зона закраски {before}→{after} px"
                if after == 0:
                    return bgr, mask, None, yolo_boxes, info, predilate, layout_polys

    roi_bboxes = roi_bounds_list(mask, padding=FINGER_PADDING, roi_scale=FINGER_ROI_SCALE)
    with log_timing("brighten_finger_zone", log_name):
        rgb_bright = brighten_finger_zone(rgb, mask, light_increment, 2 * dilate_px)
    with log_timing("lama_inpaint", log_name):
        rgb_clean = models.inpaint(rgb_bright, mask, padding=FINGER_PADDING, roi_scale=FINGER_ROI_SCALE)

    # Копирование блоков обратно — строго ПОСЛЕ закраски и с ИСХОДНОГО (неосветлённого)
    # кадра: rgb_bright уже подкрашен под LaMa и вернул бы контент со сдвигом яркости.
    if protect_text and protect_mode == PROTECT_COPY_BACK and layout_polys:
        with log_timing("copy_back_layout", log_name):
            rgb_clean, restored = copy_back_layout(rgb, rgb_clean, layout_polys, mask, layout_pad_px)
        info = f"{info}, восстановлено блоков={restored}"

    return cv2.cvtColor(rgb_clean, cv2.COLOR_RGB2BGR), mask, roi_bboxes, yolo_boxes, info, predilate, layout_polys
