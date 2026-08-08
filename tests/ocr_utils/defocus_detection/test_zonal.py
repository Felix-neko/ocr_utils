"""Поиск зонального расфокуса: что детектор обязан видеть, а что — игнорировать."""

import cv2
import numpy as np
import pytest
from tests.ocr_utils.defocus_detection.pages import blur, draw_page

from ocr_utils.defocus_detection.tiles import make_grid
from ocr_utils.defocus_detection.zonal import zonal_defocus

SIZE = 1536
# Лёгкая база размытия: у реального снимка край всегда шире пикселя, а идеальная
# синтетическая ступенька даёт σ = 0, и относительный перепад теряет смысл.
BASE_BLUR = 0.7


def page(**kwargs) -> np.ndarray:
    """Синтетическая полоса покрупнее — чтобы в сетке хватило полос.

    Args:
        **kwargs: Параметры ``draw_page``.

    Returns:
        Полутоновый кадр с базовым размытием, как у настоящей фотографии.
    """
    return blur(draw_page(height=SIZE, width=SIZE, **kwargs), BASE_BLUR)


def blur_band(image: np.ndarray, start: float, end: float, sigma: float, axis: int = 0) -> np.ndarray:
    """Размывает полосу кадра, плавно наращивая размытие к краю.

    Args:
        image: Полутоновый кадр.
        start: Начало полосы в долях стороны.
        end: Конец полосы в долях стороны.
        sigma: Максимальная сигма размытия на дальнем крае полосы.
        axis: 0 — горизонтальная полоса (по высоте), 1 — вертикальная.

    Returns:
        Кадр с размытой полосой.
    """
    out = image.copy()
    length = image.shape[axis]
    lo, hi = int(length * start), int(length * end)
    steps = 6
    for k in range(steps):
        a = lo + (hi - lo) * k // steps
        b = lo + (hi - lo) * (k + 1) // steps
        strength = sigma * (k + 1) / steps
        piece = out[a:b, :] if axis == 0 else out[:, a:b]
        blurred = cv2.GaussianBlur(piece, (0, 0), strength)
        if axis == 0:
            out[a:b, :] = blurred
        else:
            out[:, a:b] = blurred
    return out


def drop(image: np.ndarray, axis: str = "rows") -> float:
    """Перепад резкости внутри кадра, посчитанный боевыми параметрами.

    Args:
        image: Полутоновый кадр.
        axis: Ось профиля.

    Returns:
        Относительный перепад; 0.0, если оценить не удалось.
    """
    result = zonal_defocus(image, make_grid(image.shape), axis=axis)
    return result.drop if result else 0.0


def test_soft_bottom_is_detected() -> None:
    """Размытая нижняя треть при резком верхе должна давать заметный перепад."""
    flat = page()
    tilted = blur_band(page(), 0.6, 1.0, sigma=1.6)
    assert drop(tilted) > drop(flat) + 0.08, f"ровный={drop(flat):.3f} с мягким низом={drop(tilted):.3f}"


def test_soft_zone_is_located() -> None:
    """Детектор должен показывать, какая часть кадра поплыла."""
    result = zonal_defocus(blur_band(page(), 0.6, 1.0, sigma=1.6), make_grid((SIZE, SIZE)))
    assert result is not None
    assert "низ" in result.where(), result.where()
    assert result.worst > result.best


def test_uniform_blur_is_not_zonal() -> None:
    """Кадр, промазанный целиком, — это не зональный расфокус.

    Именно это разделение и оправдывает два отдельных отчёта: такой кадр обязан
    всплыть в первом (общее качество) и не всплыть во втором.
    """
    assert drop(blur(page(), 1.2)) < 0.08


def test_layout_change_is_not_mistaken_for_a_zone() -> None:
    """Смена кегля в нижней трети — вёрстка, а не расфокус.

    Ровно на этом ломается наивная версия метрики: у крупного шрифта другая ширина
    края, и без коридора по длине штриха полоса выглядела бы мягкой.
    """
    mixed = page()
    mixed[int(SIZE * 0.6) :, :] = page(stroke=6, line_height=36)[int(SIZE * 0.6) :, :]
    assert drop(mixed) < 0.10, f"перепад на смене кегля {drop(mixed):.3f}"


def test_photo_band_is_not_mistaken_for_a_zone() -> None:
    """Полутоновая иллюстрация во всю ширину — не расфокус.

    У фото нет белого фона, и по этому признаку такие тайлы выбрасываются.
    """
    rng = np.random.default_rng(3)
    with_photo = page()
    noise = rng.integers(30, 150, size=(int(SIZE * 0.25), SIZE)).astype(np.uint8)
    with_photo[int(SIZE * 0.65) : int(SIZE * 0.65) + noise.shape[0], :] = cv2.GaussianBlur(noise, (0, 0), 1.0)
    assert drop(with_photo) < 0.10, f"перепад на фото {drop(with_photo):.3f}"


def test_columns_axis_finds_a_left_right_gradient() -> None:
    """Ось cols нужна для материала, где мягкой оказывается боковая часть кадра."""
    tilted = blur_band(page(), 0.6, 1.0, sigma=1.6, axis=1)
    assert drop(tilted, axis="cols") > drop(page(), axis="cols") + 0.08
    # По горизонтальным полосам этот же дефект почти не виден — оси не взаимозаменяемы.
    assert drop(tilted, axis="rows") < drop(tilted, axis="cols")


def test_blank_page_has_no_zonal_estimate() -> None:
    """На кадре без текста судить о зоне не по чему — должно вернуться None."""
    blank = np.full((SIZE, SIZE), 235, dtype=np.uint8)
    assert zonal_defocus(blank, make_grid(blank.shape)) is None


@pytest.mark.parametrize("axis", ["rows", "cols"])
def test_result_reports_profile_and_bands(axis: str) -> None:
    """Результат должен нести профиль и индексы полос — они идут в отчёт."""
    result = zonal_defocus(page(), make_grid((SIZE, SIZE)), axis=axis)
    assert result is not None
    assert result.axis == axis
    assert len(result.profile) == result.n_bands
    assert 0 <= result.best < result.n_bands and 0 <= result.worst < result.n_bands
