"""Оценка пространственно-переменного размытия по спектру мощности.

Эталон берётся из самого же скана: самые резкие его участки. Для каждой ячейки
сетки считается отношение усреднённых спектров «ячейка / эталон», и в это
отношение вписывается анизотропный гауссиан. Получается не абсолютная PSF, а
*относительная* — то ядро, которым резкая часть кадра превращается в размытую.
Именно её и надо развернуть, чтобы выровнять резкость по кадру.

В модели есть множитель усиления: текст в разных местах страницы разной плотности,
и без него разница в содержимом притворилась бы размытием. Но отпускать его на
волю нельзя — он вырожден с ослаблением на низких частотах и, если позволить,
забирает размытие себе, занижая оценку почти вдвое. Поэтому отношение сперва
приводится к единице на низких частотах (у любого нормированного ядра K(0) = 1),
а множителю оставляется лишь узкий допуск на остаточное рассогласование.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

# Размер окна БПФ. 256 px при 300 dpi — это ~2 см, несколько строк текста:
# достаточно для устойчивого спектра и достаточно мало, чтобы размытие внутри
# окна можно было считать постоянным.
WINDOW = 256

# Полоса частот для вписывания. Ниже — общий наклон спектра документа и неровности
# фона, выше — шум сканера; и то и другое к размытию отношения не имеет.
FREQ_LO = 0.02
FREQ_HI = 0.22

# Кольцо низких частот, по которому отношение приводится к единице. Любое
# нормированное ядро размытия имеет K(0) = 1, поэтому весь сдвиг уровня на этих
# частотах — разница в содержимом, а не размытие.
ANCHOR_LO = 0.02
ANCHOR_HI = 0.05

# Насколько множителю усиления позволено отходить от единицы после привязки, в
# логарифме. Держать его в узде обязательно: множитель вырожден с ослаблением на
# низких частотах, и, отпущенный на волю, он забирает размытие себе. При свободном
# множителе оценка занижалась почти вдвое.
GAIN_BOUND = 0.06

# Окно считается «текстовым», если его контраст выше этого порога. Отсекает поля,
# корешок и пустые участки, по которым спектр оценивать бессмысленно.
MIN_STD = 0.05

# Доля самых резких окон кадра, принимаемая за эталон.
REFERENCE_QUANTILE = 0.85

# Потолок для оценки: сигма больше этой считается срывом подгонки, а не размытием.
MAX_SIGMA = 6.0


@dataclass(frozen=True)
class BlurCell:
    """Оценка размытия в одной ячейке сетки.

    Attributes:
        row: Номер строки сетки.
        col: Номер столбца сетки.
        sigma_major: Сигма гауссиана вдоль направления смаза, px.
        sigma_minor: Сигма поперёк смаза, px.
        angle_deg: Направление смаза в градусах, 0 — вправо по оси X, ось Y вниз.
        windows: Сколько текстовых окон попало в ячейку.
        cost: Невязка подгонки; большая означает, что модель ячейке не подошла.
    """

    row: int
    col: int
    sigma_major: float
    sigma_minor: float
    angle_deg: float
    windows: int
    cost: float

    @property
    def covariance(self) -> np.ndarray:
        """Ковариационная матрица гауссова ядра.

        Returns:
            Матрица 2x2 в порядке осей (x, y).
        """
        angle = np.deg2rad(self.angle_deg)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return rot @ np.diag([self.sigma_major**2, self.sigma_minor**2]) @ rot.T


@dataclass(frozen=True)
class BlurField:
    """Сетка оценок размытия по всему кадру.

    Attributes:
        rows: Число строк сетки.
        cols: Число столбцов сетки.
        cells: Ячейки, по одной на каждую позицию сетки.
        reference_windows: Сколько окон вошло в эталон.
    """

    rows: int
    cols: int
    cells: list[BlurCell]
    reference_windows: int

    def cell(self, row: int, col: int) -> BlurCell:
        """Возвращает ячейку по координатам сетки.

        Args:
            row: Номер строки.
            col: Номер столбца.

        Returns:
            Соответствующая ячейка.
        """
        return self.cells[row * self.cols + col]


def _hann2d(size: int) -> np.ndarray:
    """Двумерное окно Ханна.

    Args:
        size: Сторона окна.

    Returns:
        Массив size x size.
    """
    line = np.hanning(size)
    return np.outer(line, line).astype(np.float32)


def _window_origins(x0: int, x1: int, y0: int, y1: int, size: int, step: int) -> list[tuple[int, int]]:
    """Перечисляет левые верхние углы окон, укладывающихся в прямоугольник.

    Args:
        x0: Левая граница.
        x1: Правая граница.
        y0: Верхняя граница.
        y1: Нижняя граница.
        size: Сторона окна.
        step: Шаг сетки окон.

    Returns:
        Список координат (x, y).
    """
    return [(x, y) for y in range(y0, y1 - size + 1, step) for x in range(x0, x1 - size + 1, step)]


def _power_spectrum(
    gray: np.ndarray, origins: list[tuple[int, int]], size: int, min_std: float
) -> tuple[np.ndarray, int]:
    """Усредняет спектр мощности по текстовым окнам.

    Args:
        gray: Полутоновый кадр в диапазоне 0..1.
        origins: Углы окон.
        size: Сторона окна.
        min_std: Порог контраста, ниже которого окно пропускается.

    Returns:
        Пара (усреднённый спектр со сдвинутым нулём, число учтённых окон).
    """
    hann = _hann2d(size)
    acc = np.zeros((size, size), np.float64)
    used = 0
    for x, y in origins:
        tile = gray[y : y + size, x : x + size]
        if tile.std() < min_std:
            continue
        spectrum = np.fft.fftshift(np.fft.fft2((tile - tile.mean()) * hann))
        acc += np.abs(spectrum) ** 2
        used += 1
    if used == 0:
        return acc, 0
    return acc / used, used


def _frequency_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Сетка пространственных частот со сдвинутым нулём.

    Args:
        size: Сторона окна БПФ.

    Returns:
        Пара массивов (fx, fy) в циклах на пиксель.
    """
    axis = np.fft.fftshift(np.fft.fftfreq(size))
    fy, fx = np.meshgrid(axis, axis, indexing="ij")
    return fx, fy


def fit_anisotropic_gaussian(
    ratio: np.ndarray,
    freq_lo: float = FREQ_LO,
    freq_hi: float = FREQ_HI,
    max_sigma: float = MAX_SIGMA,
    gain_bound: float = GAIN_BOUND,
) -> tuple[float, float, float, float]:
    """Вписывает анизотропный гауссиан в измеренное отношение амплитуд спектра.

    Модель: |K(f)| = gain * exp(-2 pi^2 (sa^2 fa^2 + sb^2 fb^2)), где fa и fb —
    проекции частоты на оси эллипса. Подгонка идёт в логарифме и с мягкой
    L1-функцией потерь: отдельные частоты, где содержимое ячейки и эталона
    разошлось, не должны утаскивать оценку за собой.

    Перед подгонкой отношение приводится к единице на низких частотах. Без этого
    множитель усиления и размытие вырождены друг с другом, и оценка занижается.

    Args:
        ratio: Отношение амплитуд |K| со сдвинутым нулём, сторона равна размеру окна.
        freq_lo: Нижняя граница рабочей полосы, циклы на пиксель.
        freq_hi: Верхняя граница рабочей полосы, циклы на пиксель.
        max_sigma: Верхний предел для сигм.
        gain_bound: Допуск множителя усиления в логарифме.

    Returns:
        Кортеж (sigma_major, sigma_minor, angle_deg, cost).
    """
    size = ratio.shape[0]
    fx, fy = _frequency_grid(size)
    radius = np.hypot(fx, fy)
    log_ratio = np.log(np.maximum(ratio, 1e-6))
    anchor = (radius > ANCHOR_LO) & (radius < ANCHOR_HI)
    if anchor.any():
        log_ratio = log_ratio - log_ratio[anchor].mean()

    mask = (radius > freq_lo) & (radius < freq_hi)
    target = log_ratio[mask]
    fx_m, fy_m = fx[mask], fy[mask]

    def residual(params: np.ndarray) -> np.ndarray:
        log_gain, sigma_a, sigma_b, angle = params
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        proj_a = fx_m * cos_a + fy_m * sin_a
        proj_b = -fx_m * sin_a + fy_m * cos_a
        model = log_gain - 2 * np.pi**2 * (sigma_a**2 * proj_a**2 + sigma_b**2 * proj_b**2)
        return model - target

    bounds = ([-gain_bound, 0.0, 0.0, -np.pi], [gain_bound, max_sigma, max_sigma, 2 * np.pi])
    best = None
    # Угол входит в модель нелинейно и даёт локальные минимумы, поэтому стартуем
    # с нескольких ориентаций и берём лучшую.
    for start_angle in np.linspace(0.0, np.pi, 7)[:-1]:
        guess = [0.0, 1.0, 0.5, float(start_angle)]
        fit = least_squares(residual, guess, bounds=bounds, loss="soft_l1", f_scale=0.3)
        if best is None or fit.cost < best.cost:
            best = fit

    _, sigma_a, sigma_b, angle = best.x
    if sigma_a < sigma_b:
        sigma_a, sigma_b = sigma_b, sigma_a
        angle += np.pi / 2
    return float(sigma_a), float(sigma_b), float(np.degrees(angle) % 180.0), float(best.cost)


def _reference_spectrum(
    gray: np.ndarray, size: int, step: int, min_std: float, quantile: float
) -> tuple[np.ndarray, int]:
    """Строит эталонный спектр по самым резким окнам кадра.

    Args:
        gray: Полутоновый кадр в диапазоне 0..1.
        size: Сторона окна.
        step: Шаг сетки окон.
        min_std: Порог контраста.
        quantile: Доля отсечения; берутся окна резче этого квантиля.

    Returns:
        Пара (эталонный спектр, число окон в эталоне).

    Raises:
        ValueError: Если в кадре не нашлось ни одного текстового окна.
    """
    height, width = gray.shape
    origins = _window_origins(0, width, 0, height, size, step)
    scored: list[tuple[float, tuple[int, int]]] = []
    for x, y in origins:
        tile = gray[y : y + size, x : x + size]
        if tile.std() < min_std:
            continue
        scored.append((float(cv2.Laplacian(tile, cv2.CV_32F).var()), (x, y)))
    if not scored:
        raise ValueError("в кадре не нашлось текстовых участков — нечего брать за эталон")

    threshold = np.quantile([s for s, _ in scored], quantile)
    sharpest = [origin for score, origin in scored if score >= threshold]
    return _power_spectrum(gray, sharpest, size, min_std)


def estimate_blur_field(
    gray: np.ndarray,
    rows: int = 3,
    cols: int = 4,
    window: int = WINDOW,
    step: int | None = None,
    min_std: float = MIN_STD,
    quantile: float = REFERENCE_QUANTILE,
    max_sigma: float = MAX_SIGMA,
    min_windows: int = 12,
) -> BlurField:
    """Оценивает размытие в каждой ячейке сетки относительно резких зон кадра.

    Args:
        gray: Полутоновый кадр в диапазоне 0..1.
        rows: Число строк сетки.
        cols: Число столбцов сетки.
        window: Сторона окна БПФ.
        step: Шаг сетки окон; по умолчанию треть окна.
        min_std: Порог контраста для окна.
        quantile: Квантиль отсечения при выборе эталона.
        max_sigma: Верхний предел оценки сигмы.
        min_windows: Минимум окон в ячейке; при нехватке ячейка считается резкой.

    Returns:
        Сетка оценок размытия.
    """
    step = step or max(window // 3, 1)
    height, width = gray.shape
    reference, reference_windows = _reference_spectrum(gray, window, step, min_std, quantile)

    cells: list[BlurCell] = []
    for row in range(rows):
        y0, y1 = row * height // rows, (row + 1) * height // rows
        for col in range(cols):
            x0, x1 = col * width // cols, (col + 1) * width // cols
            origins = _window_origins(x0, x1, y0, y1, window, step)
            spectrum, used = _power_spectrum(gray, origins, window, min_std)
            if used < min_windows:
                # Пустая или почти пустая ячейка: оценивать нечего, оставляем как есть.
                cells.append(BlurCell(row, col, 0.0, 0.0, 0.0, used, 0.0))
                continue
            ratio = np.sqrt(spectrum / (reference + 1e-12))
            sigma_a, sigma_b, angle, cost = fit_anisotropic_gaussian(ratio, max_sigma=max_sigma)
            cells.append(BlurCell(row, col, sigma_a, sigma_b, angle, used, cost))

    return BlurField(rows=rows, cols=cols, cells=cells, reference_windows=reference_windows)


def smooth_field(field: BlurField, strength: float = 0.5) -> BlurField:
    """Сглаживает оценки по соседним ячейкам.

    Размытие по кадру меняется плавно, а отдельная ячейка может дать выброс из-за
    содержимого. Усреднение идёт по ковариационным матрицам, а не по углам: угол
    задан с точностью до 180 градусов, и усреднять его напрямую нельзя.

    Args:
        field: Исходная сетка оценок.
        strength: Доля, забираемая у соседей, от 0 (без сглаживания) до 1.

    Returns:
        Новая сетка со сглаженными оценками.
    """
    if strength <= 0:
        return field

    covariances = np.zeros((field.rows, field.cols, 2, 2))
    for cell in field.cells:
        covariances[cell.row, cell.col] = cell.covariance

    smoothed: list[BlurCell] = []
    for cell in field.cells:
        neighbours = []
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                row, col = cell.row + d_row, cell.col + d_col
                if 0 <= row < field.rows and 0 <= col < field.cols and (d_row, d_col) != (0, 0):
                    neighbours.append(covariances[row, col])
        if not neighbours:
            smoothed.append(cell)
            continue
        mixed = (1 - strength) * cell.covariance + strength * np.mean(neighbours, axis=0)
        # Выпуклая смесь неотрицательно определённых матриц остаётся таковой,
        # поэтому собственные числа гарантированно неотрицательны.
        values, vectors = np.linalg.eigh(mixed)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        sigma_a, sigma_b = np.sqrt(np.maximum(values, 0.0))
        angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])) % 180.0
        smoothed.append(
            BlurCell(cell.row, cell.col, float(sigma_a), float(sigma_b), float(angle), cell.windows, cell.cost)
        )

    return BlurField(rows=field.rows, cols=field.cols, cells=smoothed, reference_windows=field.reference_windows)
