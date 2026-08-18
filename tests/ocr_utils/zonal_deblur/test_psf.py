"""Проверка оценки размытия на синтетике с заранее известным ядром."""

import cv2
import numpy as np
import pytest

from ocr_utils.zonal_deblur.deconv import axis_weights, gaussian_kernel
from ocr_utils.zonal_deblur.psf import estimate_blur_field, smooth_field


def make_page(height: int = 768, width: int = 1024, seed: int = 0) -> np.ndarray:
    """Рисует страницу «текста»: тёмные штрихи на светлой бумаге.

    Спектр такой картинки похож на спектр набора: широкий, без выделенных частот.

    Args:
        height: Высота кадра.
        width: Ширина кадра.
        seed: Зерно генератора.

    Returns:
        Кадр float32 в диапазоне 0..1.
    """
    rng = np.random.default_rng(seed)
    page = np.ones((height, width), np.float32)
    for y in range(40, height - 40, 28):
        x = 40
        while x < width - 60:
            w, h = int(rng.integers(4, 12)), int(rng.integers(12, 20))
            page[y : y + h, x : x + w] = rng.uniform(0.05, 0.25)
            x += w + int(rng.integers(3, 10))
    return cv2.GaussianBlur(page, (0, 0), 0.5)


def effective_sigma_major(kernel: np.ndarray) -> float:
    """Считает фактическую сигму ядра вдоль главной оси по второму моменту.

    Сравнивать оценку надо именно с этой величиной, а не с запрошенной сигмой:
    ядро живёт на целочисленной сетке, и при малой поперечной сигме дискретизация
    заметно сужает его. Запрошенная 1.0 может обернуться фактической 0.87.

    Args:
        kernel: Нормированное ядро.

    Returns:
        Корень из большего собственного числа матрицы вторых моментов.
    """
    half = kernel.shape[0] // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(coords, coords)
    moments = np.array(
        [
            [(kernel * grid_x * grid_x).sum(), (kernel * grid_x * grid_y).sum()],
            [(kernel * grid_x * grid_y).sum(), (kernel * grid_y * grid_y).sum()],
        ]
    )
    return float(np.sqrt(np.linalg.eigvalsh(moments).max()))


def blur_region(
    page: np.ndarray, x1: int, sigma_major: float, sigma_minor: float, angle_deg: float
) -> tuple[np.ndarray, float]:
    """Смазывает левую часть кадра заданным анизотропным ядром.

    Args:
        page: Исходный кадр.
        x1: Правая граница размываемой области.
        sigma_major: Сигма вдоль смаза.
        sigma_minor: Сигма поперёк смаза.
        angle_deg: Направление смаза в градусах.

    Returns:
        Пара (кадр с размытой левой частью, фактическая сигма ядра).
    """
    angle = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    covariance = rot @ np.diag([sigma_major**2, sigma_minor**2]) @ rot.T
    kernel = gaussian_kernel(covariance, size=31)

    out = page.copy()
    # Свёртка с запасом по краю, чтобы на стыке зон не было ступеньки от границы.
    margin = 40
    patch = page[:, : x1 + margin]
    out[:, :x1] = cv2.filter2D(patch, -1, kernel, borderType=cv2.BORDER_REFLECT)[:, :x1]
    return out, effective_sigma_major(kernel)


@pytest.mark.parametrize("sigma_major,angle_deg", [(1.0, 0.0), (1.2, 45.0), (1.5, 90.0), (1.2, 135.0)])
def test_recovers_known_kernel(sigma_major: float, angle_deg: float) -> None:
    """Оценщик находит сигму и направление искусственного смаза."""
    page = make_page()
    blurred, expected_sigma = blur_region(page, x1=512, sigma_major=sigma_major, sigma_minor=0.2, angle_deg=angle_deg)

    field = estimate_blur_field(blurred, rows=1, cols=4, min_windows=4)
    left = field.cell(0, 0)
    right = field.cell(0, 3)

    assert left.sigma_major == pytest.approx(expected_sigma, abs=0.2)
    # Угол задан с точностью до 180 градусов, поэтому сравниваем по кругу.
    delta = abs(left.angle_deg - angle_deg) % 180.0
    assert min(delta, 180.0 - delta) < 20.0
    # Резкая половина не должна выглядеть размытой.
    assert right.sigma_major < 0.4
    assert left.sigma_major > right.sigma_major + 0.4


def test_sharp_page_reports_no_blur() -> None:
    """На равномерно резкой странице оценщик не выдумывает размытие."""
    field = estimate_blur_field(make_page(), rows=1, cols=4, min_windows=4)
    assert all(cell.sigma_major < 0.4 for cell in field.cells)


def test_smoothing_keeps_estimates_positive() -> None:
    """Сглаживание не портит знак и порядок оценок."""
    page, _ = blur_region(make_page(), x1=512, sigma_major=1.4, sigma_minor=0.2, angle_deg=30.0)
    field = smooth_field(estimate_blur_field(page, rows=1, cols=4, min_windows=4), strength=0.5)
    assert all(cell.sigma_major >= cell.sigma_minor >= 0.0 for cell in field.cells)
    assert field.cell(0, 0).sigma_major > field.cell(0, 3).sigma_major


def test_axis_weights_form_partition_of_unity() -> None:
    """Веса ячеек в каждой точке дают в сумме единицу — иначе будут швы."""
    for count in (1, 2, 4, 7):
        weights = axis_weights(500, count)
        assert weights.sum(axis=0) == pytest.approx(np.ones(500), abs=1e-5)


def test_gaussian_kernel_matches_requested_covariance() -> None:
    """Построенное ядро имеет ту ковариацию, которую у него просили."""
    angle = np.deg2rad(30.0)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    covariance = rot @ np.diag([4.0, 1.0]) @ rot.T
    kernel = gaussian_kernel(covariance, size=41)

    half = kernel.shape[0] // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(coords, coords)
    measured = np.array(
        [
            [(kernel * grid_x * grid_x).sum(), (kernel * grid_x * grid_y).sum()],
            [(kernel * grid_x * grid_y).sum(), (kernel * grid_y * grid_y).sum()],
        ]
    )
    assert measured == pytest.approx(covariance, abs=0.15)
