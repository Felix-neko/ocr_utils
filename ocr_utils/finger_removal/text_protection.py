"""Защита контентных зон (page layout через Surya) от закраски пальца.

ЗАЧЕМ ЭТО НУЖНО
---------------
Дилатация маски пальца — это вилка. Большая дилатация надёжно накрывает тень
вокруг пальца, но заползает на текст и портит его (LaMa дорисовывает вместо
строк «чистую бумагу»). Маленькая дилатация текст не трогает, но оставляет
вокруг пальца тень, и LaMa, видя тень в контексте, дорисовывает тень же.

Выход — прогнать кадр через Surya layout ДО закраски и защитить найденные блоки.
Защищать можно двумя способами (``--text-protect-mode``):

``limit-lama-zone``
    Зона закраски урезается: из маски вычитаются блоки layout. LaMa вообще не
    видит контент, но и тень, попавшую на блок, не убирает.

``copy-back-layout-zones``
    Зона закраски НЕ урезается — LaMa работает во всю ширину дилатации (и потому
    лучше вычищает тень), а после закраски блоки layout, пересекающиеся с зоной
    закраски, копируются обратно с исходного кадра. Контент восстанавливается
    ровно там, где он есть, а всё между блоками (поля, межколоночные пробелы,
    край страницы) остаётся вычищенным.

В ОБОИХ режимах берутся ВСЕ блоки layout, включая картинки/таблицы: растровая
иллюстрация — такой же контент, который нельзя ни затирать, ни дорисовывать.

Сам палец (первичная маска до дилатации) закрашивается всегда и никогда не
восстанавливается: под ним контента всё равно не видно, и оставлять там кожу
нельзя — поэтому ядро исключается и из вычитания, и из копирования обратно.
"""

from typing import Optional

import cv2
import numpy as np

# Режимы защиты (значения --text-protect-mode)
PROTECT_LIMIT_LAMA = "limit-lama-zone"
PROTECT_COPY_BACK = "copy-back-layout-zones"
PROTECT_MODES = (PROTECT_LIMIT_LAMA, PROTECT_COPY_BACK)

# Запас вокруг блока layout ПО УМОЛЧАНИЮ, пикс. (переопределяется из CLI —
# --layout-pad-px). Surya обводит блок впритык, а закраска у самой границы глифов
# даёт заметный «съеденный» край строки. Этот же запас берётся при исключении ядра
# пальца из защищаемой области. Запас работает в пользу сохранности контента и
# против вычистки тени: в limit-lama-zone кайма блока не закрашивается вовсе, в
# copy-back-layout-zones — копируется обратно вместе с блоком (т.е. с тенью).
DEFAULT_LAYOUT_PAD_PX = 12

def _pad_xy(pad_px: "int | tuple[int, int]") -> "tuple[int, int]":
    """Нормализует запас к паре (по X, по Y): скаляр → (N, N).

    Скаляр — одинаково по обеим осям; пара — раздельно. Строки текста тянутся по
    горизонтали, а тень от пальца часто приходит сверху/снизу, поэтому иногда
    полезно задать разный запас по осям.
    """
    if isinstance(pad_px, (tuple, list)):
        px, py = pad_px
        return int(px), int(py)
    return int(pad_px), int(pad_px)


# Край восстановленной области НЕ размывается намеренно. Блоки layout стоят
# вплотную друг к другу и нередко пересекаются, поэтому плавный переход на шве
# неизбежно залез бы в соседний блок и подмешал туда заливку LaMa, т.е. испортил
# бы контент, который мы и защищаем. Ступенька яркости на шве — меньшее зло.

# Сторона, до которой уменьшается кадр перед прогоном layout. Surya всё равно
# ресайзит вход под свой размер, а на 36-Мп сканах предварительное уменьшение
# экономит секунды на одном только декодировании/конвертации.
LAYOUT_WORK_SIDE = 2048

_LAYOUT_PREDICTOR = None


def load_layout_predictor():
    """Ленивая загрузка Surya LayoutPredictor (один раз на процесс)."""
    global _LAYOUT_PREDICTOR
    if _LAYOUT_PREDICTOR is None:
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.settings import settings

        predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
        predictor.disable_tqdm = True
        _LAYOUT_PREDICTOR = predictor
    return _LAYOUT_PREDICTOR


def layout_polygons(rgb: np.ndarray) -> "list[np.ndarray]":
    """Полигоны ВСЕХ блоков layout (текст, заголовки, картинки, таблицы...).

    Кадр уменьшается до ``LAYOUT_WORK_SIDE`` по длинной стороне, полигоны
    возвращаются пересчитанными обратно в координаты полного кадра.
    """
    from PIL import Image as PILImage

    h, w = rgb.shape[:2]
    scale = min(1.0, LAYOUT_WORK_SIDE / max(h, w))
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb

    result = load_layout_predictor()([PILImage.fromarray(small)])[0]
    return [np.asarray(box.polygon, dtype=np.float32) / scale for box in result.bboxes]


def polygons_to_mask(
    shape: "tuple[int, int]", polygons: "list[np.ndarray]", pad_px: "int | tuple[int, int]" = DEFAULT_LAYOUT_PAD_PX
):
    """Бинарная маска (uint8 0/255) объединения полигонов, расширенных на ``pad_px``.

    ``pad_px`` — скаляр или пара (по X, по Y).
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if not polygons:
        return mask
    cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in polygons], 255)
    return _dilated(mask, pad_px)


def _dilated(mask: np.ndarray, pad_px: "int | tuple[int, int]") -> np.ndarray:
    """Маска, расширенная на ``pad_px`` (скаляр или пара по X/Y; без изменений при <=0)."""
    px, py = _pad_xy(pad_px)
    if px <= 0 and py <= 0:
        return mask
    px, py = max(px, 0), max(py, 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * py + 1))
    return cv2.dilate(mask, k)


def intersecting_polygons(
    polygons: "list[np.ndarray]", paint_mask: np.ndarray, pad_px: "int | tuple[int, int]" = DEFAULT_LAYOUT_PAD_PX
) -> "list[np.ndarray]":
    """Блоки layout, хоть сколько-нибудь пересекающиеся с зоной закраски.

    ``pad_px`` — скаляр или пара (по X, по Y). Проверка идёт внутри bbox блока
    (полномасштабная маска на каждый из десятков блоков 36-Мп кадра — заметная и
    совершенно лишняя трата).
    """
    px, py = _pad_xy(pad_px)
    px, py = max(px, 0), max(py, 0)
    h, w = paint_mask.shape[:2]
    hits: list[np.ndarray] = []
    for poly in polygons:
        pts = np.round(poly).astype(np.int32)
        x1 = max(0, int(pts[:, 0].min()) - px)
        y1 = max(0, int(pts[:, 1].min()) - py)
        x2 = min(w, int(pts[:, 0].max()) + px + 1)
        y2 = min(h, int(pts[:, 1].max()) + py + 1)
        if x2 <= x1 or y2 <= y1 or not paint_mask[y1:y2, x1:x2].any():
            continue
        sub = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillPoly(sub, [pts - (x1, y1)], 255)
        sub = _dilated(sub, pad_px)
        if cv2.bitwise_and(sub, paint_mask[y1:y2, x1:x2]).any():
            hits.append(poly)
    return hits


def limit_paint_zone(mask: np.ndarray, core: Optional[np.ndarray], layout_mask: np.ndarray) -> np.ndarray:
    """Режим ``limit-lama-zone``: вырезает блоки layout из маски закраски.

    ``mask`` — маска пальца после дилатации (то, что уйдёт в LaMa), ``core`` — она
    же до дилатации. Всё, что входит в ``core``, возвращается назад: сам палец
    закрашивается при любом раскладе.
    """
    out = cv2.bitwise_and(mask, cv2.bitwise_not(layout_mask))
    if core is not None:
        out = cv2.bitwise_or(out, cv2.bitwise_and(mask, core))
    return out


def copy_back_layout(
    rgb_orig: np.ndarray,
    rgb_clean: np.ndarray,
    polygons: "list[np.ndarray]",
    paint_mask: np.ndarray,
    core: Optional[np.ndarray],
    pad_px: "int | tuple[int, int]" = DEFAULT_LAYOUT_PAD_PX,
) -> "tuple[np.ndarray, int]":
    """Режим ``copy-back-layout-zones``: возвращает контент блоков layout из оригинала.

    ``pad_px`` — скаляр или пара (по X, по Y). Копируются целиком те блоки, что
    пересекаются с зоной закраски, за вычетом ядра пальца (``core``, с тем же
    запасом ``pad_px``) — иначе вместе с контентом вернулся бы и сам палец. Копирование жёсткое, без размытия шва (почему — см.
    комментарий про шов среди констант модуля).

    Возвращает (кадр, число восстановленных блоков).
    """
    hits = intersecting_polygons(polygons, paint_mask, pad_px)
    if not hits:
        return rgb_clean, 0

    region = polygons_to_mask(paint_mask.shape, hits, pad_px)
    if core is not None:
        region = cv2.bitwise_and(region, cv2.bitwise_not(_dilated(core, pad_px)))
    if not region.any():
        return rgb_clean, 0

    out = rgb_clean.copy()
    out[region > 0] = rgb_orig[region > 0]
    return out, len(hits)
