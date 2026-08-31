"""Защита растровых иллюстраций по разметке страницы (Surya layout).

ЗАЧЕМ. Основной алгоритм подпакета защищает то, что темнее порога, — то есть текст
и line art. Растровая (полутоновая) фотография под это правило не подходит: её
светлые участки от фона по яркости неотличимы, и размытие выело бы в ней острова.
Поэтому кадры с крупным растром целиком отбраковывались (``has_halftone``) и
копировались как есть — вместе с «перцем» на текстовых колонках вокруг фотографии.

Флаг ``--use-surya-layout`` разрывает этот компромисс: страница прогоняется через
Surya layout, найденные блоки-иллюстрации добавляются в защитную маску с тем же
припуском, что и текст, а всё остальное поле сглаживается как обычно. Ценой
примерно 0.7 с на кадр (GPU) страницы «текст + фотография» перестают быть
неприкасаемыми.

Модель грузится лениво и ровно одна — LayoutPredictor. ``scan_cropping.gpu_models``
сюда не годится: его конструктор тянет ещё YOLO-World x2, SAM и LaMa (вплоть до
скачивания весов), а здесь не нужно ничего, кроме разметки.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ocr_utils.background_smoothing.processing import (
    HALFTONE_DOWNSCALE,
    HALFTONE_HI,
    HALFTONE_LO,
    HALFTONE_OPEN_PX,
    has_halftone,
)

logger = logging.getLogger(__name__)

# Классы блоков Surya, которые считаем кандидатами в растровую иллюстрацию. Берём
# только ``Picture`` (фотография): ``Figure`` — это графики и диаграммы, то есть тот
# самый line art, который основной алгоритм защищает по яркости точнее, чем
# прямоугольником блока.
PICTURE_LABELS = ("Picture",)

# Сторона, до которой уменьшается кадр перед прогоном layout. Surya всё равно
# ресайзит вход под свой размер, а на 21-Мп сканах предварительное уменьшение
# экономит секунды на одной только конвертации (то же значение и по той же причине,
# что в ``scan_cropping.finger_removal.text_protection``).
LAYOUT_WORK_SIDE = 2048

# --- Притяжение блоков к реальному растру ----------------------------------
# Surya размечает не «растровое изображение», а ВИЗУАЛЬНЫЙ БЛОК, и на смонтированной
# полосе может обвести всю дизайнерскую плашку целиком, прихватив пустую бумагу и
# срезав край самой фотографии. Пример: 1967/08 IMG_0094_1L, блок conf=0.62 —
# заголовок вместе с портретом, причём верх портрета (y 424..708) остался снаружи.
#
# Поэтому к блокам добавляются СВЯЗНЫЕ РАСТРОВЫЕ ОБЛАСТИ, которые с ними граничат:
# та же маска средних тонов, что и в ``has_halftone``, но со смыканием и разбором на
# связные компоненты. Защищается ОБЪЕДИНЕНИЕ прямоугольника Surya и охватывающих
# прямоугольников таких компонент.
#
# Именно объединение, а не замена: компонента обводит лишь ту часть фотографии, что
# попала в средние тона, и на снимке со светлым небом она заметно уже самого снимка
# (замер, 1968/01 IMG_0015_2R: блок y 1089..2926, компонента y 1376..2908). Замена
# срезала бы 287 px неба, и по фотографии прошёл бы шов сглаживания — это хуже, чем
# оставить лишнюю бумагу несглаженной. Объединение не сжимает блок никогда.
# Ядро смыкания на копии 1/4: сводит зерно одной фотографии в одну компоненту.
# Побочный эффект у самой рамки кадра: за пределами массива морфология считает фон
# «своим», поэтому область, подошедшая к краю ближе чем на половину ядра, дотягивается
# до края. Для фотографии, свёрстанной в обрез, это как раз верно, а лишний защищённый
# поясок в 60 px по краю скана ничего не стоит.
RASTER_CLOSE_PX = 31
RASTER_MIN_AREA_PX = 900  # площадь на копии 1/4 (~120x120 в полном кадре): мельче — крапина, не иллюстрация


class LayoutDetector:
    """Surya LayoutPredictor: блоки-иллюстрации на странице.

    Создаётся один раз на прогон (в ``pipeline.run_batch``), модель грузится при
    первом обращении — чтобы пустой список файлов не стоил загрузки весов.
    """

    def __init__(self, labels: "tuple[str, ...]" = PICTURE_LABELS) -> None:
        self._labels = labels
        self._predictor = None

    def _load(self):
        """Ленивая загрузка предиктора (импорт surya тоже ленивый — он не быстрый)."""
        if self._predictor is None:
            from surya.foundation import FoundationPredictor
            from surya.layout import LayoutPredictor
            from surya.settings import settings

            logger.info("Загружаю Surya layout (--use-surya-layout)")
            predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
            predictor.disable_tqdm = True  # иначе на каждый кадр рвётся прогресс-бар пачки
            self._predictor = predictor
        return self._predictor

    def picture_polygons(
        self, bgr: np.ndarray, gray: "np.ndarray | None" = None, filter_raster: bool = True
    ) -> "list[np.ndarray]":
        """Полигоны РАСТРОВЫХ иллюстраций (4 точки) в координатах ПОДАННОГО кадра.

        Кадр уменьшается до ``LAYOUT_WORK_SIDE`` по длинной стороне, координаты
        возвращаются пересчитанными обратно в полное разрешение. Блоки прочих
        классов (Text, Caption, PageFooter...) отбрасываются: их содержимое и так
        темнее порога, отдельная защита прямоугольником им не нужна.

        Класс ``Picture`` Surya ставит и настоящим фотографиям, и крупным чертежам,
        поэтому каждый блок ещё и ПРОВЕРЯЕТСЯ детектором растра — см.
        :func:`is_raster_block`. Чертежу сплошная защита не нужна и вредна: под ней
        останется невычищенным весь фон внутри рамки блока, а это бывает больше
        половины страницы.

        ``filter_raster=False`` отдаёт СЫРОЙ список блоков ``Picture``, без этой
        проверки. Нужен ``scan_markup``: там растр от штриха отличают по статистике
        размеров пятен краски на полном кадре, а :func:`is_raster_block` меряет
        ``has_halftone`` по копии 1/4 — на ней плотная штриховка неотличима от
        полутоновой печати по яркости, и именно на этом признаке из пака-1 в растр
        уехала 31 полоса со штриховыми виньетками. Своей задаче — защите от
        сглаживания фона — проверка по-прежнему годится, поэтому умолчание прежнее.

        ``gray`` — серая версия кадра, если она уже посчитана вызывающим.
        """
        from PIL import Image as PILImage

        h, w = bgr.shape[:2]
        scale = min(1.0, LAYOUT_WORK_SIDE / max(h, w))
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        blocks = self._load()([PILImage.fromarray(rgb)])[0].bboxes
        candidates = [np.asarray(b.polygon, dtype=np.float32) / scale for b in blocks if b.label in self._labels]
        if not candidates or not filter_raster:
            return candidates

        if gray is None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return [poly for poly in candidates if is_raster_block(gray, poly)]


def is_raster_block(gray: np.ndarray, polygon: np.ndarray) -> bool:
    """Есть ли внутри блока настоящий растр (полутоновая печать), а не line art.

    Проверяется тем же детектором, что решает судьбу всего кадра, — только по
    прямоугольнику блока. Замер по 1968/01, доля растра внутри блоков ``Picture``:

        фотографии (0015_1L, 0015_2R, 0016_1L, 0030_2R, 0050_2R)   0.32 - 0.69
        чертежи    (0029_1L, 0029_2R)                              0.0000

    Порог ``HALFTONE_MIN_FRAC`` = 0.01 лежит посреди пустого промежутка, так что
    разделение устойчивое. Блок вне кадра или вырожденный — не растр.
    """
    h, w = gray.shape[:2]
    pts = polygon.reshape(-1, 2)
    x1, y1 = np.clip(np.floor(pts.min(axis=0)), 0, None).astype(int)
    x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
    x2, y2 = min(int(x2), w), min(int(y2), h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return False
    return has_halftone(gray[y1:y2, x1:x2])


def _rect(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Прямоугольник как полигон из 4 точек — в том же виде, в каком их отдаёт Surya."""
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def raster_regions(
    gray: np.ndarray, polygons: "list[np.ndarray]", downscale: int = HALFTONE_DOWNSCALE
) -> "list[np.ndarray]":
    """Связные растровые области, граничащие с блоками ``polygons`` (их bbox-полигоны).

    Считается на копии 1/``downscale``: маска средних тонов → размыкание (убирает
    каймы букв, как в :func:`has_halftone`) → смыкание (сводит зерно одной
    фотографии в одну компоненту) → связные компоненты. Возвращаются охватывающие
    прямоугольники тех компонент, которые пересекают хоть один блок и не меньше
    ``RASTER_MIN_AREA_PX``.

    Это ДОБАВКА к блокам, а не замена — мотивировка у констант ``RASTER_*`` выше.
    Компонента может выходить за блок (ровно так возвращается срезанный край
    фотографии) и может целиком лежать внутри него (тогда объединение ничего не
    меняет — так на всех корректных блоках 1968/01).

    Растровые области, не граничащие ни с одним блоком, СЮДА НЕ ПОПАДАЮТ: они
    остаются в области анализа, и кадр с ними задержит ``has_halftone`` — страховка
    на промах разметки от этого не слабеет.
    """
    if not polygons:
        return []

    h, w = gray.shape[:2]
    size = (max(1, w // downscale), max(1, h // downscale))
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    mid = ((small > HALFTONE_LO) & (small < HALFTONE_HI)).astype(np.uint8)
    mid = cv2.morphologyEx(mid, cv2.MORPH_OPEN, np.ones((HALFTONE_OPEN_PX,) * 2, np.uint8))
    mid = cv2.morphologyEx(mid, cv2.MORPH_CLOSE, np.ones((RASTER_CLOSE_PX,) * 2, np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mid, 8)
    if count <= 1:  # растра на кадре нет вовсе
        return []

    regions = []
    for poly in polygons:
        pts = poly.reshape(-1, 2) / downscale
        x1, y1 = np.clip(np.floor(pts.min(axis=0)), 0, None).astype(int)
        x2 = min(int(np.ceil(pts[:, 0].max())), labels.shape[1])
        y2 = min(int(np.ceil(pts[:, 1].max())), labels.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue

        # Все компоненты блока сводятся в ОДИН прямоугольник: одна фотография
        # распадается на несколько пятен (светлое небо, засвеченный лоб портрета
        # в средние тона не попадают), и по отдельности они оставили бы между
        # собой незащищённые прорехи прямо внутри снимка.
        boxes = [
            stats[label, :4]
            for label in np.unique(labels[y1:y2, x1:x2])
            if label and stats[label, cv2.CC_STAT_AREA] >= RASTER_MIN_AREA_PX
        ]
        if not boxes:
            continue
        boxes = np.asarray(boxes, dtype=int)
        left, top = boxes[:, 0].min(), boxes[:, 1].min()
        right, bottom = (boxes[:, 0] + boxes[:, 2]).max(), (boxes[:, 1] + boxes[:, 3]).max()
        regions.append(_rect(left * downscale, top * downscale, right * downscale, bottom * downscale))
    return regions


def polygons_mask(shape: "tuple[int, ...]", polygons: "list[np.ndarray]") -> np.ndarray:
    """Бинарная маска (uint8 0/255) объединения полигонов; пустой список — нулевая маска.

    Припуск здесь не добавляется: маска идёт в общую дилатацию защитной маски и
    получает тот же припуск, что и текст (см. ``pipeline.process_frame``).

    Полигоны заливаются ПО ОДНОМУ намеренно. ``cv2.fillPoly`` со списком контуров
    считает их одной фигурой и заливает по правилу чётности, поэтому вложенные
    прямоугольники взаимно уничтожаются: прямоугольник растра внутри блока Surya
    вычитал бы блок (замер: покрытие кадра падало с 48.6% до 3.8%).
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
    return mask


def analysis_roi(picture_mask: np.ndarray, polygons: "list[np.ndarray]") -> Optional[np.ndarray]:
    """Область, по которой считать пороги и искать растр: весь кадр МИНУС ``picture_mask``.

    Фотография портит обе статистики кадра. Гистограмма: её средние тона тянут порог
    Оцу вверх, и часть бумаги вокруг текста уезжает под маску. Детектор растра:
    он сработал бы ровно на ней и забраковал бы всю страницу — а ведь ради того,
    чтобы этого НЕ случилось, layout и запускался. Поэтому при ``--use-surya-layout``
    и то и другое считается по остатку кадра.

    ``None`` (весь кадр) — когда иллюстраций не нашлось: это дешевле любой маски и
    даёт ровно то же поведение, что без флага.
    """
    return None if not polygons else cv2.bitwise_not(picture_mask)


def make_detector(enabled: bool) -> Optional[LayoutDetector]:
    """``LayoutDetector`` при включённом флаге, иначе ``None`` (модель не грузится)."""
    return LayoutDetector() if enabled else None
