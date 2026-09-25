"""Сетка строк страницы: поле их НАКЛОНА и назначение кусков строкам.

Жадная сборка строки «ближайший подходящий кусок справа» на сильно искажённой бумаге уводит
цепочку на соседнюю строку: сгустки двух строк ЧЕРЕДУЮТСЯ по x (замер на 1973/08 с.85 дал ряд
ординат 1200, 1226, 1204, 1232, 1209, 1234), и каждый шаг выглядит законным — изломы внутри
цепочек не превышают 10°, наклоны звеньев 11°. Поэтому сначала оценивается ход строк по всей
странице, а потом каждый кусок отдаётся своей строке.

Поле хранит НАКЛОН набора, а не смещение. Смещение «ордината минус середина своей строки»
непригодно: на двухколоннике левые якоря (середина около x = 350) и правые (x = 950) при наклоне
в три градуса дают в межколоннике рассогласование под тридцать пикселей — больше межстрочного
шага, и сплайн размазывает этот скачок по всей странице (проверено: покрытие краски падало до
17 %). Наклон же — величина локальная, одинаково пригодная и от длинного якоря, и от короткого, и
сшивается через межколонник без разрывов. Выпрямленная ордината получается интегрированием:
``y' = y - ∫ tanθ dx``.

Почему вообще нужно поле: ход одной строки слева направо (медиана 4.5–6.8 px, p90 9–16 px)
сопоставим с межстрочным шагом (15–21 px), то есть в сырых координатах соседние строки
перекрываются, а после выпрямления разброс внутри строки — единицы пикселей.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates

from ocr_utils.curved_layout import WORK_DPI
from ocr_utils.page_layout import mm_to_px

# Якорная строка — длиннее этой доли от p90 длин (тот же приём, что у ``fitting.long_mask``).
ANCHOR_SHARE = 0.6
# Меньше стольких якорей — поле не строим: на странице-схеме подписи разбросаны, и единый ход
# строк для них бессмыслен (замер: схема 1975/05 с.99 — 6 якорей, чертёж 1969/02 с.43 — 11,
# блок-схема 1973/01 с.84 — 20, а сплошной текст и таблицы дают 48–125).
MIN_ANCHORS = 12
# Окно, по которому меряется наклон вдоль якорной строки (мм бумаги): около четырёх высот
# корпуса — короче окно, и шум центра строки даёт лишний градус.
SLOPE_WINDOW_MM = 20.0
# Сколько отсчётов наклона берётся с одной якорной строки и сколько их всего (плотная система
# сплайна решается за куб числа узлов).
SAMPLES_PER_LINE = 8
MAX_SAMPLES = 600
# Сетка поля и сглаживание сплайна (координаты нормируются на размер страницы).
GRID_COLS = 32
GRID_ROWS = 48
FIELD_SMOOTHING = 0.01
# Наклон строк на скане не превышает этого; больше — поле себе что-то придумало.
SLOPE_LIMIT_DEG = 5.0
# Накопленный подъём не должен превышать столько межстрочных шагов.
MAX_RISE_PITCHES = 4.0
# Разумные пределы межстрочного шага (мм бумаги): ими проверяется оценка по якорям.
PITCH_MIN_MM = 2.0
PITCH_MAX_MM = 12.0


@dataclass(frozen=True)
class LineField:
    """Ход строк страницы: наклон набора на сетке и накопленный по x подъём.

    Args:
        rise: Накопленный подъём ``∫ tanθ dx`` в узлах сетки (пиксели рабочей копии).
        x0, x1, y0, y1: Охват сетки в пикселях рабочей копии.
        pitch: Межстрочный шаг в тех же пикселях.
        anchors: Сколько якорных строк участвовало в оценке.
    """

    rise: np.ndarray
    x0: float
    x1: float
    y0: float
    y1: float
    pitch: float
    anchors: int

    def straighten(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Выпрямленные ординаты точек: у кусков одной строки они совпадают в пределах пары пикселей."""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        if xs.size == 0:
            return np.zeros(0, dtype=np.float64)
        rows, cols = self.rise.shape
        # Координаты точки в узлах сетки; за её пределами поле продолжается постоянным.
        col = np.clip((xs - self.x0) / max(self.x1 - self.x0, 1e-6) * (cols - 1), 0, cols - 1)
        row = np.clip((ys - self.y0) / max(self.y1 - self.y0, 1e-6) * (rows - 1), 0, rows - 1)
        lift = map_coordinates(self.rise, [row, col], order=1, mode="nearest")
        return ys - lift


def anchors_of(segments: list, share: float = ANCHOR_SHARE) -> list:
    """Якорные строки — заметно длиннее остальных: только по ним оценивается ход строк.

    Args:
        segments: Строки первого прохода (``segment.Segment``).
        share: Доля от p90 длин, ниже которой строка якорем не считается.

    Returns:
        Подсписок ``segments``.
    """
    if not segments:
        return []
    lengths = np.array([item.x1 - item.x0 for item in segments], dtype=np.float64)
    limit = share * float(np.percentile(lengths, 90))
    return [item for item, length in zip(segments, lengths) if length >= limit]


def slope_samples(anchors: list, k: float, dpi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Отсчёты наклона вдоль якорных строк: ``(x, y, tanθ)`` в пикселях рабочей копии.

    Наклон меряется скользящим окном ``SLOPE_WINDOW_MM`` по точкам оси: прямая, подогнанная по
    окну, даёт локальный наклон строки, а отсчёт относится к середине окна.

    Args:
        anchors: Якорные строки.
        k: Во сколько раз рендер крупнее рабочей копии.
        dpi: Разрешение рабочей копии.

    Returns:
        Три массива одинаковой длины.
    """
    window = mm_to_px(SLOPE_WINDOW_MM, dpi)
    xs_out: list[float] = []
    ys_out: list[float] = []
    slopes: list[float] = []
    for item in anchors:
        xs = np.asarray(item.xs, dtype=np.float64) / k
        ys = np.asarray(item.ys, dtype=np.float64) / k
        if xs.size < 6 or xs[-1] - xs[0] < window:
            continue
        edges = np.linspace(xs[0], xs[-1] - window, SAMPLES_PER_LINE)
        for left in edges:
            own = (xs >= left) & (xs <= left + window)
            if own.sum() < 4:
                continue
            slope, _ = np.polyfit(xs[own], ys[own], 1)
            xs_out.append(float(np.mean(xs[own])))
            ys_out.append(float(np.mean(ys[own])))
            slopes.append(float(slope))
    return np.array(xs_out), np.array(ys_out), np.array(slopes)


def field_of(segments: list, shape: tuple[int, int], k: float, dpi: float = WORK_DPI) -> LineField | None:
    """Оценить ход строк по якорным строкам первого прохода.

    Args:
        segments: Строки первого прохода; их ``xs``/``ys`` заданы в пикселях рендера.
        shape: Размер рабочей копии ``(высота, ширина)``.
        k: Во сколько раз рендер крупнее рабочей копии.
        dpi: Разрешение рабочей копии.

    Returns:
        :class:`LineField` или ``None``, если якорей мало, наклон неправдоподобен или шаг строк
        не оценивается.
    """
    chosen = anchors_of(segments)
    if len(chosen) < MIN_ANCHORS:
        return None
    xs, ys, slopes = slope_samples(chosen, k, dpi)
    if xs.size < MIN_ANCHORS * 2:
        return None
    limit = np.tan(np.radians(SLOPE_LIMIT_DEG))
    keep = np.abs(slopes) <= limit
    xs, ys, slopes = xs[keep], ys[keep], slopes[keep]
    if xs.size < MIN_ANCHORS * 2:
        return None
    if xs.size > MAX_SAMPLES:
        take = np.linspace(0, xs.size - 1, MAX_SAMPLES).astype(int)
        xs, ys, slopes = xs[take], ys[take], slopes[take]
    height, width = shape
    nodes = np.column_stack([xs / max(width, 1), ys / max(height, 1)])
    model = RBFInterpolator(nodes, slopes, kernel="thin_plate_spline", smoothing=FIELD_SMOOTHING)
    grid_x = np.linspace(float(xs.min()), float(xs.max()), GRID_COLS)
    grid_y = np.linspace(float(ys.min()), float(ys.max()), GRID_ROWS)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    flat = np.column_stack([mesh_x.ravel() / max(width, 1), mesh_y.ravel() / max(height, 1)])
    slope_grid = np.asarray(model(flat)).reshape(GRID_ROWS, GRID_COLS)
    slope_grid = np.clip(slope_grid, -limit, limit)
    step = (grid_x[-1] - grid_x[0]) / max(GRID_COLS - 1, 1)
    rise = np.cumsum(slope_grid, axis=1) * step
    rise -= rise.mean(axis=1, keepdims=True)  # отсчёт от середины строки: поле не сдвигает страницу
    field = LineField(
        rise=rise,
        x0=float(grid_x[0]),
        x1=float(grid_x[-1]),
        y0=float(grid_y[0]),
        y1=float(grid_y[-1]),
        pitch=0.0,
        anchors=len(chosen),
    )
    middles_x = np.array([float(np.median(np.asarray(item.xs) / k)) for item in chosen])
    middles_y = np.array([float(np.median(np.asarray(item.ys) / k)) for item in chosen])
    pitch = _pitch_of(field.straighten(middles_x, middles_y), dpi)
    if pitch is None or float(np.abs(rise).max()) > MAX_RISE_PITCHES * pitch:
        return None
    return LineField(rise=rise, x0=field.x0, x1=field.x1, y0=field.y0, y1=field.y1, pitch=pitch, anchors=len(chosen))


def _pitch_of(centres: np.ndarray, dpi: float) -> float | None:
    """Межстрочный шаг по выпрямленным ординатам якорей; ``None`` — оценка неправдоподобна."""
    if centres.size < 3:
        return None
    steps = np.diff(np.sort(centres))
    low, high = mm_to_px(PITCH_MIN_MM, dpi), mm_to_px(PITCH_MAX_MM, dpi)
    # Строки соседних колонок стоят на одной высоте и дают почти нулевые разности — их
    # отбрасываем, иначе медиана уезжает; всё, что выше PITCH_MAX_MM, — разрыв между блоками.
    steps = steps[(steps >= low) & (steps <= high)]
    if steps.size < 2:
        return None
    return float(np.median(steps))


__all__ = ["LineField", "anchors_of", "field_of", "slope_samples"]
