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

from ocr_utils.background_smoothing.processing import has_halftone

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

    def picture_polygons(self, bgr: np.ndarray, gray: "np.ndarray | None" = None) -> "list[np.ndarray]":
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

        ``gray`` — серая версия кадра, если она уже посчитана вызывающим.
        """
        from PIL import Image as PILImage

        h, w = bgr.shape[:2]
        scale = min(1.0, LAYOUT_WORK_SIDE / max(h, w))
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        blocks = self._load()([PILImage.fromarray(rgb)])[0].bboxes
        candidates = [np.asarray(b.polygon, dtype=np.float32) / scale for b in blocks if b.label in self._labels]
        if not candidates:
            return []

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


def polygons_mask(shape: "tuple[int, ...]", polygons: "list[np.ndarray]") -> np.ndarray:
    """Бинарная маска (uint8 0/255) объединения полигонов; пустой список — нулевая маска.

    Припуск здесь не добавляется: маска идёт в общую дилатацию защитной маски и
    получает тот же припуск, что и текст (см. ``pipeline.process_frame``).
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in polygons], 255)
    return mask


def analysis_roi(picture_mask: np.ndarray, polygons: "list[np.ndarray]") -> Optional[np.ndarray]:
    """Область, по которой считать пороги и искать растр: весь кадр МИНУС иллюстрации.

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
