"""Поиск областей строк текста через surya-ocr (``DetectionPredictor``).

ПОЧЕМУ ТАЙЛАМИ, А НЕ КАДРОМ ЦЕЛИКОМ. ``DetectionPredictor.prepare_image`` ужимает вход до
размера своего процессора (порядка 1024 px по стороне), а ``split_image`` режет кадр
ТОЛЬКО по высоте. Ширина превью (2944 px у портретного скана) давится втрое, и корпусная
строка газеты высотой ~17 px приходит в сеть высотой в пять пикселей.

Замер на пяти полосах «Социалистической индустрии» 1985 г. (``scripts/check_surya_line_detection.py``):

    режим   строк     высота p10/p50/p90      строк в самом бедном тайле 3x3
    page      761        15 / 17 / 26 px                  14
    tiles    1904        15 / 16 / 28 px                  36

Строк втрое больше — но важнее не это, а то, КАКИХ строк. У режима ``page`` заметно выше
p90 высоты (на других полосах 46-49 px против 30-31 у тайлов): целую полосу он размечает
преимущественно по заголовкам, а корпус пропускает — то есть ровно тот текст, ради
которого всё и затевается. И покрытие у него вдвое-втрое беднее в самом бедном тайле, а
зональная карта строится именно на противопоставлении частей кадра друг другу.

Цена — примерно 4 с на кадр против 1.3 с. Поэтому здесь есть дисковый кэш: он нужен не
для красоты, а потому что метрики калибруются итеративно, и повторный прогон той же папки
с другой агрегацией не должен трогать GPU вообще.

Мельчить тайлы дальше вредно: при стороне 800 px найденных строк стало МЕНЬШЕ (1554
против 1904), потому что строка всё чаще перерезается границей тайла, а контекста сети
не хватает.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.defocus_detection.lines.regions import LineRegion

logger = logging.getLogger(__name__)

DETECT_MODES = ("page", "tiles")
DEFAULT_MODE = "tiles"
# Сторона тайла. 1100 px на превью 2944x4416 даёт сетку 3x5: ужатие под процессор сети
# всего в полтора раза вместо трёх, при этом строка редко перерезается границей.
DEFAULT_TILE_SIDE = 1100
# Перекрытие соседей. Строка корпуса длиной в колонку — это сотни пикселей, и попавшая
# на стык должна целиком уместиться хотя бы в одном тайле.
DEFAULT_TILE_OVERLAP = 300
DEFAULT_MIN_CONF = 0.5
# Версия схемы кэша: меняется, когда меняется смысл сохранённых полей, чтобы старые
# файлы кэша не подхватились молча как валидные.
CACHE_VERSION = 1


@dataclass(frozen=True)
class DetectParams:
    """Параметры детекции — они же часть ключа дискового кэша.

    Attributes:
        mode: "page" — кадр целиком, "tiles" — перекрывающимися тайлами.
        tile_side: Сторона тайла в режиме "tiles".
        tile_overlap: Перекрытие соседних тайлов.
        page_max_side: До какой длинной стороны уменьшать кадр в режиме "page";
            None — подавать как есть.
        min_conf: Порог уверенности блока.
        batch_size: Размер батча для surya; None — её собственный выбор.
    """

    mode: str = DEFAULT_MODE
    tile_side: int = DEFAULT_TILE_SIDE
    tile_overlap: int = DEFAULT_TILE_OVERLAP
    page_max_side: int | None = None
    min_conf: float = DEFAULT_MIN_CONF
    batch_size: int | None = None

    def key(self) -> str:
        """Короткий отпечаток параметров для имени файла кэша.

        ``batch_size`` в отпечаток НЕ входит: он влияет на скорость, но не на результат,
        и включать его значило бы терять кэш при каждой смене размера батча.

        Returns:
            Шестнадцатеричная строка.
        """
        payload = (
            f"{CACHE_VERSION}|{self.mode}|{self.tile_side}|{self.tile_overlap}|{self.page_max_side}|{self.min_conf}"
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


class DetectCache:
    """Дисковый кэш результатов детекции: один JSON на кадр.

    Ключ — путь к файлу, его размер и mtime плюс отпечаток параметров детекции. Правка
    файла или смена параметров делают старую запись невидимой, поэтому чистить кэш руками
    не нужно.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _path(self, image: Path, params: DetectParams) -> Path:
        """Путь к файлу кэша для кадра.

        Args:
            image: Путь к изображению.
            params: Параметры детекции.

        Returns:
            Путь к JSON-файлу кэша.
        """
        try:
            stat = image.stat()
            stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            stamp = "?"
        digest = hashlib.sha1(f"{image.resolve()}|{stamp}".encode("utf-8")).hexdigest()
        # Двухсимвольный подкаталог: в одной папке иначе копятся десятки тысяч файлов.
        return self._root / params.key() / digest[:2] / f"{digest}.json"

    def load(self, image: Path, params: DetectParams) -> list[LineRegion] | None:
        """Читает сохранённые области строк.

        Args:
            image: Путь к изображению.
            params: Параметры детекции.

        Returns:
            Список областей либо None, если записи нет или она битая.
        """
        path = self._path(image, params)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return [
                LineRegion(polygon=np.asarray(item["polygon"], dtype=np.float64), confidence=item["confidence"])
                for item in payload["lines"]
            ]
        except (OSError, ValueError, KeyError, TypeError):
            # Битая или отсутствующая запись — не повод падать, просто посчитаем заново.
            return None

    def store(self, image: Path, params: DetectParams, regions: list[LineRegion]) -> None:
        """Сохраняет области строк.

        Args:
            image: Путь к изображению.
            params: Параметры детекции.
            regions: Найденные области.
        """
        path = self._path(image, params)
        payload = {
            "source": str(image),
            "lines": [
                {"polygon": np.asarray(r.polygon, dtype=np.float64).reshape(4, 2).tolist(), "confidence": r.confidence}
                for r in regions
            ],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Пишем через временный файл: прерванный прогон не должен оставить обрубок,
            # который потом прочитается как валидный кэш.
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(path)
        except OSError as error:
            logger.warning("Не удалось записать кэш детекции %s: %s", path, error)


def tile_origins(size: int, side: int, overlap: int) -> list[int]:
    """Начала тайлов вдоль одной оси.

    Последний тайл прижимается к дальнему краю, поэтому его перекрытие с предыдущим
    может оказаться больше запрошенного — это лучше узкой полоски в конце.

    Args:
        size: Длина оси в пикселях.
        side: Сторона тайла.
        overlap: Желаемое перекрытие соседей.

    Returns:
        Список координат начала тайлов.
    """
    if size <= side:
        return [0]
    step = max(1, side - overlap)
    origins = list(range(0, size - side, step))
    origins.append(size - side)
    return origins


class LineDetector:
    """Области строк текста на кадре. Модель грузится лениво и ровно одна.

    Ленивая загрузка — не микрооптимизация: импорт surya и веса детектора стоят секунд,
    а пустой список файлов или попадание всей папки в кэш не должны их стоить. Тот же
    приём и по той же причине применён в ``background_smoothing.layout.LayoutDetector``.
    """

    def __init__(self, params: DetectParams | None = None, cache: DetectCache | None = None) -> None:
        self._params = params or DetectParams()
        self._cache = cache
        self._predictor = None

    @property
    def params(self) -> DetectParams:
        """Параметры детекции.

        Returns:
            Текущие ``DetectParams``.
        """
        return self._params

    def _load(self):
        """Ленивая загрузка предиктора.

        Returns:
            Готовый ``DetectionPredictor``.
        """
        if self._predictor is None:
            from surya.detection import DetectionPredictor

            logger.info("Загружаю Surya detection (--use-surya-lines)")
            predictor = DetectionPredictor()
            predictor.disable_tqdm = True  # иначе на каждый кадр рвётся полоса прогресса пачки
            self._predictor = predictor
        return self._predictor

    def detect(self, path: Path, gray: np.ndarray) -> list[LineRegion]:
        """Находит области строк на кадре, по возможности взяв их из кэша.

        Args:
            path: Путь к изображению — нужен как ключ кэша.
            gray: Полутоновый кадр ПОЛНОГО разрешения.

        Returns:
            Список областей в координатах полного кадра.
        """
        if self._cache is not None:
            cached = self._cache.load(path, self._params)
            if cached is not None:
                return cached

        if self._params.mode == "page":
            regions = self._detect_page(gray)
        else:
            regions = self._detect_tiles(gray)

        if self._cache is not None:
            self._cache.store(path, self._params, regions)
        return regions

    def _to_pil(self, gray: np.ndarray):
        """Полутоновый массив -> RGB-картинка Pillow (surya принимает только PIL).

        Args:
            gray: Полутоновый кадр uint8.

        Returns:
            Изображение Pillow в режиме RGB.
        """
        from PIL import Image as PILImage

        return PILImage.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

    def _keep(self, box) -> bool:
        """Проходит ли блок порог уверенности.

        Args:
            box: ``PolygonBox`` от surya.

        Returns:
            True, если блок берём.
        """
        return box.confidence is None or box.confidence >= self._params.min_conf

    def _detect_page(self, gray: np.ndarray) -> list[LineRegion]:
        """Детекция по кадру целиком.

        Args:
            gray: Полутоновый кадр полного разрешения.

        Returns:
            Список областей в координатах полного кадра.
        """
        h, w = gray.shape
        scale, small = 1.0, gray
        if self._params.page_max_side:
            scale = min(1.0, self._params.page_max_side / max(h, w))
            if scale < 1.0:
                small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        result = self._load()([self._to_pil(small)], batch_size=self._params.batch_size)[0]
        return [
            LineRegion(
                polygon=np.asarray(box.polygon, dtype=np.float64).reshape(4, 2) / scale,
                confidence=float(box.confidence) if box.confidence is not None else float("nan"),
            )
            for box in result.bboxes
            if self._keep(box)
        ]

    def _detect_tiles(self, gray: np.ndarray) -> list[LineRegion]:
        """Детекция по перекрывающимся тайлам в нативном разрешении.

        Все тайлы кадра уходят в сеть ОДНИМ вызовом: surya сама режет их на батчи по
        ``batch_size`` (по умолчанию 36 для cuda), и отдавать их по одному значило бы
        гонять GPU вхолостую.

        Строки в зоне перекрытия нашлись бы дважды. Дубли снимаются не порогом IoU, а
        геометрически: тайл принимает только строки, ЦЕНТР которых лежит в его «ядре» —
        тайле, урезанном на половину перекрытия с тех сторон, где есть сосед. Ядра
        соседей не пересекаются и вместе покрывают кадр целиком, поэтому каждая строка
        достаётся ровно одному тайлу, и подбирать пороги не приходится.

        Args:
            gray: Полутоновый кадр полного разрешения.

        Returns:
            Список областей в координатах полного кадра.
        """
        h, w = gray.shape
        side = min(self._params.tile_side, h, w)
        xs = tile_origins(w, side, self._params.tile_overlap)
        ys = tile_origins(h, side, self._params.tile_overlap)
        half = self._params.tile_overlap // 2

        crops, offsets, cores = [], [], []
        for y0 in ys:
            for x0 in xs:
                crops.append(self._to_pil(gray[y0 : y0 + side, x0 : x0 + side]))
                offsets.append((x0, y0))
                # Ядро урезается только со стороны реального соседа: у крайних тайлов
                # внешняя граница остаётся на месте, иначе край полосы выпал бы из покрытия.
                cores.append(
                    (
                        x0 + (half if x0 != xs[0] else 0),
                        y0 + (half if y0 != ys[0] else 0),
                        x0 + side - (half if x0 != xs[-1] else 0),
                        y0 + side - (half if y0 != ys[-1] else 0),
                    )
                )

        results = self._load()(crops, batch_size=self._params.batch_size)

        regions: list[LineRegion] = []
        for result, (x0, y0), (cx1, cy1, cx2, cy2) in zip(results, offsets, cores):
            for box in result.bboxes:
                if not self._keep(box):
                    continue
                polygon = np.asarray(box.polygon, dtype=np.float64).reshape(4, 2) + np.array([x0, y0], dtype=np.float64)
                cx, cy = polygon.mean(axis=0)
                if cx1 <= cx < cx2 and cy1 <= cy < cy2:
                    regions.append(
                        LineRegion(
                            polygon=polygon,
                            confidence=float(box.confidence) if box.confidence is not None else float("nan"),
                        )
                    )
        return regions
