"""Проверка деконволюции и сквозного прогона на синтетике."""

import numpy as np
import pytest

from ocr_utils.zonal_deblur.deconv import deblur_plane, gaussian_otf, richardson_lucy, wiener_deconvolve
from ocr_utils.zonal_deblur.psf import estimate_blur_field, smooth_field
from tests.ocr_utils.zonal_deblur.test_psf import make_page


def covariance_of(sigma_major: float, sigma_minor: float, angle_deg: float) -> np.ndarray:
    """Собирает ковариацию гауссова ядра из сигм и угла.

    Args:
        sigma_major: Сигма вдоль главной оси.
        sigma_minor: Сигма поперёк.
        angle_deg: Направление главной оси в градусах.

    Returns:
        Матрица 2x2 в порядке осей (x, y).
    """
    angle = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rot @ np.diag([sigma_major**2, sigma_minor**2]) @ rot.T


def blur_with_otf(plane: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Размывает кадр аналитической передаточной функцией.

    Args:
        plane: Исходный кадр.
        covariance: Ковариация размытия.

    Returns:
        Размытый кадр.
    """
    otf = gaussian_otf(covariance, plane.shape)
    return np.fft.irfft2(np.fft.rfft2(plane) * otf, s=plane.shape)


def test_gaussian_otf_has_unit_gain_at_zero() -> None:
    """Ядро нормировано: на нулевой частоте передача равна единице."""
    otf = gaussian_otf(covariance_of(2.0, 0.5, 30.0), (128, 128))
    assert otf[0, 0] == pytest.approx(1.0)
    assert otf.min() >= 0.0


def test_wiener_undoes_known_blur() -> None:
    """Винер возвращает кадр, размытый известным ядром."""
    page = make_page(256, 256)
    covariance = covariance_of(1.2, 0.2, 40.0)
    blurred = blur_with_otf(page, covariance)
    restored = wiener_deconvolve(blurred, gaussian_otf(covariance, page.shape), nsr=1e-6)

    inner = (slice(32, -32), slice(32, -32))
    assert np.abs(restored[inner] - page[inner]).mean() < np.abs(blurred[inner] - page[inner]).mean() / 5


def test_richardson_lucy_undoes_known_blur() -> None:
    """Ричардсон—Люси тоже сходится к исходному кадру."""
    page = make_page(256, 256)
    covariance = covariance_of(1.0, 0.3, 0.0)
    blurred = blur_with_otf(page, covariance)
    restored = richardson_lucy(blurred, gaussian_otf(covariance, page.shape), iterations=120)

    # Сходится он медленно: на гауссовом ядре даже сотня итераций даёт всего
    # двукратный выигрыш, тогда как Винер закрывает то же самое за одно БПФ.
    inner = (slice(32, -32), slice(32, -32))
    assert np.abs(restored[inner] - page[inner]).mean() < np.abs(blurred[inner] - page[inner]).mean() / 2


# Во сколько раз каждый метод обязан ужать измеренный смаз. У Винера порог выше:
# на таком мягком размытии он попросту точнее, а RL к этому моменту ещё ползёт.
MIN_IMPROVEMENT = {"wiener": 2.0, "rl": 1.8}


@pytest.mark.parametrize("method", ["wiener", "rl"])
def test_pipeline_equalises_sharpness(method: str) -> None:
    """Сквозной прогон: оценка плюс деконволюция снимают зональный смаз.

    Проверяется не картинка, а измеренный после обработки смаз: он должен упасть
    заметно, а резкая половина — не поехать.
    """
    page = make_page(768, 1024)
    covariance = covariance_of(1.3, 0.25, 45.0)
    blurred = page.copy()
    blurred[:, :512] = blur_with_otf(page, covariance)[:, :512]

    field = smooth_field(estimate_blur_field(blurred, rows=1, cols=4, min_windows=4), strength=0.35)
    restored = deblur_plane(blurred, field, method=method, nsr=0.003, iterations=120)

    after = estimate_blur_field(restored, rows=1, cols=4, min_windows=4)
    assert after.cell(0, 0).sigma_major < field.cell(0, 0).sigma_major / MIN_IMPROVEMENT[method]

    # Резкую половину сверяем прямо с исходными пикселями. Мерить её тем же
    # оценщиком нельзя: эталон берётся из обработанного кадра и после исправления
    # левой половины сам уезжает, отчего нетронутая правая «слепнет» на ровном месте.
    untouched = (slice(None), slice(800, 1000))
    assert np.abs(restored[untouched] - page[untouched]).mean() < 0.03


def test_sharp_cells_are_left_alone() -> None:
    """Ячейки без размытия проходят насквозь без изменений."""
    page = make_page(384, 384)
    field = smooth_field(estimate_blur_field(page, rows=1, cols=2, min_windows=4), strength=0.0)
    restored = deblur_plane(page, field, method="wiener", nsr=0.01)
    assert np.abs(restored - page).max() < 0.02
