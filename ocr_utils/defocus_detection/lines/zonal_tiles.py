"""Зональный расфокус по сетке тайлов: где на кадре мелкий текст поплыл.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ``defocus_detection.zonal``. Там кадр режется на мелкую сетку, из
тайлов каскадом фильтров однородности отбираются те, что похожи на корпусный набор, и по
ним строятся профили резкости вдоль четырёх направлений. Весь этот каскад существует
ровно потому, что тайл — это не текст: в него попадает вперемешку заголовок, поле,
полутоновое фото и корпус, и отличать их приходится косвенно, по числу краёв и длине
штриха.

Здесь границы текста известны точно — их дал детектор. Поэтому фильтры не нужны: тайл
получает медиану по замерам строк, которые в него попали. Медиана, а не среднее:
один жирный курсивный лозунг, проскочивший коридор кегля, не должен утащить весь тайл.

Тайлов немного (3x3 или 4x4), и это осознанно. Вопрос, на который отвечает эта метрика, —
«какую часть кадра надо переснять», и ответ «правый нижний угол» полезнее, чем карта
резкости из сотни ячеек. Оптический завал плоскости — это плавный градиент через
полкадра, а не пятна размером с колонку.

Метрика кадра — перепад между самым резким и самым мягким тайлом, в той же шкале и по
той же формуле, что ``zonal.profile_drop``: ``σ_мягкого / σ_резкого − 1``. Одинаковая
шкала здесь не косметика — обе колонки печатаются в одном отчёте рядом, и читать их
приходится одним и тем же глазом.
"""

from dataclasses import dataclass

import numpy as np

from ocr_utils.defocus_detection.lines.measure import LineMeasurements
from ocr_utils.defocus_detection.zonal import zone_name

DEFAULT_TILE_SIDE = 3
# Минимум измеренных строк в тайле. Меньше — тайл не участвует ни в перепаде, ни в
# отчёте: медиана по горстке строк шумит сильнее, чем сам искомый зональный эффект.
# На замеренных полосах в самый бедный тайл сетки 3x3 попадало 36-94 строки, так что
# порог отсекает не нормальные тайлы, а поля, обрез страницы и фотографию во всю ячейку.
DEFAULT_MIN_LINES = 6
# Сколько тайлов должно набраться, чтобы перепад вообще имел смысл.
MIN_TILES = 4


@dataclass
class TileZonalResult:
    """Зональный расфокус кадра по сетке тайлов.

    Attributes:
        drop: Относительный перепад: 0.0 — кадр ровный, 0.3 — в мягком тайле край
            на 30 % шире, чем в резком.
        sharpness: Карта резкости по тайлам (n, n), больше = резче; NaN — тайл пуст.
        counts: Сколько измеренных строк пришлось на каждый тайл.
        worst: Индексы (iy, ix) самого мягкого тайла.
        best: Индексы (iy, ix) самого резкого тайла.
        n: Сторона сетки.
    """

    drop: float
    sharpness: np.ndarray
    counts: np.ndarray
    worst: tuple[int, int]
    best: tuple[int, int]
    n: int

    def where(self) -> str:
        """Человекочитаемое описание, какая часть кадра поплыла.

        Зона называется по положению худшего тайла, пересчитанному в смещение от центра
        кадра в его долях, — той же функцией и с тем же порогом, что и в направленном
        зональном отчёте, чтобы формулировки в двух колонках совпадали дословно.

        Returns:
            Строка вида «низ (тайл 3/3 по вертикали, резче 1)».
        """
        iy, ix = self.worst
        # Центр тайла в долях кадра, отсчитанный от его середины: [-0.5, +0.5].
        dx = (ix + 0.5) / self.n - 0.5
        dy = (iy + 0.5) / self.n - 0.5
        by, bx = self.best
        return f"{zone_name(dx, dy)} (тайл r{iy + 1}c{ix + 1} из {self.n}x{self.n}, резче r{by + 1}c{bx + 1})"


def tile_zonal(
    measurements: LineMeasurements, n: int = DEFAULT_TILE_SIDE, min_lines: int = DEFAULT_MIN_LINES
) -> TileZonalResult | None:
    """Строит зональную карту по замерам строк.

    Args:
        measurements: Замеры по строкам одного кадра.
        n: Сторона сетки тайлов.
        min_lines: Минимум измеренных строк в тайле, иначе тайл не участвует.

    Returns:
        ``TileZonalResult`` либо None, если заполненных тайлов набралось меньше
        ``MIN_TILES`` — судить о перепаде тогда не по чему.
    """
    sharpness = np.full((n, n), np.nan, dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.int64)

    valid = measurements.valid
    values = measurements.sharpness[valid]
    tiles = measurements.tile_index[valid]

    for tile in range(n * n):
        here = values[tiles == tile]
        iy, ix = divmod(tile, n)
        counts[iy, ix] = here.size
        if here.size >= min_lines:
            sharpness[iy, ix] = float(np.median(here))

    filled = np.isfinite(sharpness)
    if int(filled.sum()) < MIN_TILES:
        return None

    best_flat = int(np.nanargmax(sharpness))
    worst_flat = int(np.nanargmin(sharpness))
    best = divmod(best_flat, n)
    worst = divmod(worst_flat, n)

    # Резкость обратна ширине края, поэтому отношение резкостей — это и есть отношение
    # ширин, только перевёрнутое: σ_мягкого / σ_резкого = s_резкого / s_мягкого.
    soft = sharpness[worst]
    sharp = sharpness[best]
    drop = float(sharp / soft - 1.0) if soft > 0 else float("nan")

    return TileZonalResult(drop=drop, sharpness=sharpness, counts=counts, worst=worst, best=best, n=n)
