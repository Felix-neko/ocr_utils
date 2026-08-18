"""Деконволюция с ядром, плавно меняющимся по кадру.

Ядро своё в каждой ячейке сетки, поэтому кадр разворачивается столько раз,
сколько ячеек, и результаты складываются с весами-«домиками». Веса в сумме дают
единицу в каждой точке, так что швов между зонами не возникает — в отличие от
поблочной обработки, где границы блоков видно.

Такой способ дороже поблочного, но на нескольких десятках ячеек это секунды, а
артефактов он не даёт вовсе.
"""

import numpy as np

# Запас по краям под свёртку: БПФ заворачивает изображение по кругу, и без запаса
# верх кадра подмешался бы в низ.
PAD = 48


def gaussian_kernel(covariance: np.ndarray, size: int | None = None) -> np.ndarray:
    """Строит гауссово ядро с заданной ковариацией в пространственной области.

    Нужно, чтобы размывать кадр в тестах. Для деконволюции берите gaussian_otf:
    выборка на целочисленной сетке при малой поперечной сигме сужает ядро против
    заказанного, и разворачивалось бы меньше, чем измерено.

    Args:
        covariance: Матрица 2x2 в порядке осей (x, y).
        size: Сторона ядра; по умолчанию подбирается по сигмам.

    Returns:
        Нормированное ядро с суммой 1.
    """
    sigma_max = float(np.sqrt(max(np.linalg.eigvalsh(covariance).max(), 0.0)))
    if size is None:
        size = max(3, int(2 * np.ceil(3 * sigma_max) + 1))
    if size % 2 == 0:
        size += 1

    half = size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(coords, coords)
    stack = np.stack([grid_x.ravel(), grid_y.ravel()])

    # Регуляризация: при sigma_minor = 0 матрица вырождена и не обращается. Это
    # физически нормально (идеально линейный смаз), но численно требует пола.
    regularized = covariance + np.eye(2) * 1e-3
    quadratic = np.einsum("ij,jk,ik->i", stack.T, np.linalg.inv(regularized), stack.T)
    kernel = np.exp(-0.5 * quadratic).reshape(size, size)
    return (kernel / kernel.sum()).astype(np.float32)


def gaussian_otf(covariance: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Строит передаточную функцию гауссова размытия аналитически.

    Считать её надо именно так, а не через БПФ выборки ядра в пространстве.
    Гауссиан с поперечной сигмой заметно меньше пикселя ложится на целочисленную
    сетку с искажением, и фактическое ядро выходит уже заказанного — в опытах
    заказ (1.0, 0.15) давал реальные 0.87. Разворачивалось бы тогда меньше, чем
    измерено, и часть смаза оставалась бы на месте.

    Формула та же, что в модели оценщика, так что инвертируется ровно то, что
    было измерено.

    Args:
        covariance: Матрица 2x2 в порядке осей (x, y).
        shape: Размер кадра (высота, ширина).

    Returns:
        Вещественная передаточная функция в раскладке rfft2.
    """
    height, width = shape
    freq_y = np.fft.fftfreq(height).reshape(-1, 1)
    freq_x = np.fft.rfftfreq(width).reshape(1, -1)
    quadratic = covariance[0, 0] * freq_x**2 + 2 * covariance[0, 1] * freq_x * freq_y + covariance[1, 1] * freq_y**2
    return np.exp(-2 * np.pi**2 * quadratic)


def wiener_deconvolve(plane: np.ndarray, otf: np.ndarray, nsr: float) -> np.ndarray:
    """Разворачивает свёртку фильтром Винера.

    Args:
        plane: Канал изображения.
        otf: Передаточная функция размытия в раскладке rfft2.
        nsr: Отношение шум/сигнал; чем больше, тем осторожнее фильтр.

    Returns:
        Восстановленный канал.
    """
    inverse = np.conj(otf) / (np.abs(otf) ** 2 + nsr)
    return np.fft.irfft2(np.fft.rfft2(plane) * inverse, s=plane.shape)


def richardson_lucy(plane: np.ndarray, otf: np.ndarray, iterations: int) -> np.ndarray:
    """Разворачивает свёртку итерациями Ричардсона—Люси.

    В отличие от Винера сохраняет неотрицательность, но на гауссовом ядре сходится
    медленно: сотня итераций даёт примерно то же, что Винер за одно БПФ, а после
    двух-трёх сотен начинает раскачивать зерно и результат снова портится.

    Args:
        plane: Канал изображения в диапазоне 0..1.
        otf: Передаточная функция размытия в раскладке rfft2.
        iterations: Число итераций.

    Returns:
        Восстановленный канал.
    """
    otf_conj = np.conj(otf)
    observed = np.clip(plane, 1e-4, None)
    estimate = observed.copy()
    for _ in range(iterations):
        blurred = np.fft.irfft2(np.fft.rfft2(estimate) * otf, s=plane.shape)
        ratio = observed / np.clip(blurred, 1e-4, None)
        correction = np.fft.irfft2(np.fft.rfft2(ratio) * otf_conj, s=plane.shape)
        estimate = np.clip(estimate * correction, 0.0, None)
    return estimate


def axis_weights(length: int, count: int) -> np.ndarray:
    """Строит по одной оси кусочно-линейные веса, дающие в сумме единицу.

    Веса привязаны к центрам ячеек; за крайними центрами вес крайней ячейки
    держится равным единице, иначе у границ кадра сумма провалилась бы.

    Args:
        length: Длина оси в пикселях.
        count: Число ячеек вдоль оси.

    Returns:
        Массив формы (count, length).
    """
    weights = np.zeros((count, length), np.float32)
    if count == 1:
        weights[0] = 1.0
        return weights

    centers = (np.arange(count) + 0.5) * length / count
    positions = np.arange(length) + 0.5
    left = np.clip(np.searchsorted(centers, positions) - 1, 0, count - 2)
    span = centers[left + 1] - centers[left]
    fraction = np.clip((positions - centers[left]) / span, 0.0, 1.0)
    columns = np.arange(length)
    weights[left, columns] += 1.0 - fraction
    weights[left + 1, columns] += fraction
    return weights


def deblur_plane(
    plane: np.ndarray,
    field,
    method: str = "wiener",
    nsr: float = 0.01,
    iterations: int = 120,
    extra_sigma: float = 0.0,
    min_sigma: float = 0.12,
) -> np.ndarray:
    """Разворачивает размытие канала по сетке оценок.

    Args:
        plane: Канал изображения в диапазоне 0..1.
        field: Сетка оценок размытия, объект BlurField.
        method: "wiener" или "rl".
        nsr: Отношение шум/сигнал для фильтра Винера.
        iterations: Число итераций для Ричардсона—Люси.
        extra_sigma: Добавка изотропной резкости ко всем ячейкам, включая резкие.
        min_sigma: Ниже этой сигмы ячейка считается резкой и не обрабатывается.

    Returns:
        Восстановленный канал в диапазоне 0..1.

    Raises:
        ValueError: Если метод неизвестен.
    """
    if method not in ("wiener", "rl"):
        raise ValueError(f"неизвестный метод деконволюции: {method}")

    height, width = plane.shape
    padded = np.pad(plane, PAD, mode="reflect")
    weights_y = axis_weights(height, field.rows)
    weights_x = axis_weights(width, field.cols)

    result = np.zeros((height, width), np.float32)
    for cell in field.cells:
        weight = weights_y[cell.row][:, None] * weights_x[cell.col][None, :]
        if weight.max() <= 1e-6:
            continue

        covariance = cell.covariance + np.eye(2) * extra_sigma**2
        if np.sqrt(np.linalg.eigvalsh(covariance).max()) < min_sigma:
            # Ячейка и так резкая: разворачивать нечего, только шум поднимать.
            result += weight * plane
            continue

        otf = gaussian_otf(covariance, padded.shape)
        if method == "wiener":
            restored = wiener_deconvolve(padded, otf, nsr)
        else:
            restored = richardson_lucy(padded, otf, iterations)
        result += weight * restored[PAD : PAD + height, PAD : PAD + width].astype(np.float32)

    return np.clip(result, 0.0, 1.0)
