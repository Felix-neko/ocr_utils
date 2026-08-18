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


def blur_corner(image: np.ndarray, sigma: float, right: bool = True, bottom: bool = True) -> np.ndarray:
    """Размывает один угол кадра, плавно наращивая размытие к самому углу.

    Именно этот случай не ловится ни горизонтальными, ни вертикальными полосами:
    завал угла размазывается по всей длине полосы любой из двух осей.

    Args:
        image: Полутоновый кадр.
        sigma: Максимальная сигма размытия в углу.
        right: True — угол правый, False — левый.
        bottom: True — угол нижний, False — верхний.

    Returns:
        Кадр с размытым углом.
    """
    height, width = image.shape
    ys = np.linspace(0.0, 1.0, height)[:, None]
    xs = np.linspace(0.0, 1.0, width)[None, :]
    # Расстояние до угла в долях диагонали: 0 в самом углу, 1 в противоположном.
    # xs растёт вправо, ys — вниз, поэтому до ПРАВОГО края расстояние 1 - xs.
    distance = np.sqrt((1.0 - xs if right else xs) ** 2 + (1.0 - ys if bottom else ys) ** 2) / np.sqrt(2.0)
    weight = np.clip(1.0 - distance / 0.5, 0.0, 1.0)

    out = image.astype(np.float32)
    blurred = cv2.GaussianBlur(out, (0, 0), sigma)
    return (out * (1.0 - weight) + blurred * weight).astype(np.uint8)


def drop(image: np.ndarray, axis: str = "rows") -> float:
    """Перепад резкости внутри кадра, посчитанный боевыми параметрами.

    Args:
        image: Полутоновый кадр.
        axis: Направление профиля.

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


@pytest.mark.parametrize(
    ("right", "bottom", "expected"),
    [
        (True, True, "правый нижний угол"),
        (False, True, "левый нижний угол"),
        (True, False, "правый верхний угол"),
        (False, False, "левый верхний угол"),
    ],
)
def test_soft_corner_is_detected_and_named(right: bool, bottom: bool, expected: str) -> None:
    """Завал любого из четырёх углов обязан находиться и называться правильно.

    Ради этого случая профиль и обобщён на произвольное направление: провал в углу
    может не дотянуть до верха списка ни по горизонтальным полосам, ни по вертикальным.
    """
    corner = blur_corner(page(), sigma=1.8, right=right, bottom=bottom)
    result = zonal_defocus(corner, make_grid(corner.shape), axis="all")
    assert result is not None
    assert result.drop > drop(page(), axis="all") + 0.10, f"перепад в углу всего {result.drop:.3f}"
    assert expected in result.where(), f"ждали «{expected}», получили «{result.where()}»"


def test_diagonal_direction_earns_its_place() -> None:
    """Есть углы, которые ни одна из осей сетки не видит так же хорошо, как диагональ.

    Правый верхний угол — как раз такой: по диагонали ``anti`` перепад в полтора раза
    больше, чем по любой из осей. Без диагональных направлений такой кадр остался бы
    в середине списка.
    """
    corner = blur_corner(page(), sigma=1.8, right=True, bottom=False)
    assert drop(corner, axis="anti") > 1.5 * max(drop(corner, axis="rows"), drop(corner, axis="cols"))


def test_all_axes_picks_the_worst_direction() -> None:
    """Режим all обязан сам находить нужное направление во всех трёх типах зон."""
    cases = {
        "низ": blur_band(page(), 0.6, 1.0, sigma=1.6),
        "право": blur_band(page(), 0.6, 1.0, sigma=1.6, axis=1),
        "правый верхний угол": blur_corner(page(), sigma=1.8, right=True, bottom=False),
    }
    for expected, image in cases.items():
        result = zonal_defocus(image, make_grid(image.shape), axis="all")
        assert result is not None, expected
        assert expected in result.where(), f"ждали «{expected}», получили «{result.where()}»"
        # Худшее направление и попадает в drop — остальные посчитаны, но не выбраны.
        assert result.drop == max(result.drops.values())


def test_flat_page_stays_flat_in_all_axes_mode() -> None:
    """Максимум по четырём направлениям не должен превращать ровный кадр в зональный.

    Максимум коррелированных величин смещён вверх, и это неизбежно; проверяем, что
    смещение осталось в пределах порога, ниже которого кадр считается ровным.
    """
    assert drop(page(), axis="all") < 0.08


def test_blank_page_has_no_zonal_estimate() -> None:
    """На кадре без текста судить о зоне не по чему — должно вернуться None."""
    blank = np.full((SIZE, SIZE), 235, dtype=np.uint8)
    assert zonal_defocus(blank, make_grid(blank.shape)) is None


@pytest.mark.parametrize("axis", ["rows", "cols", "diag", "anti"])
def test_result_reports_profile_and_bands(axis: str) -> None:
    """Результат должен нести профиль и индексы полос — они идут в отчёт."""
    result = zonal_defocus(page(), make_grid((SIZE, SIZE)), axis=axis)
    assert result is not None
    assert result.axis == axis
    assert len(result.profile) == result.n_bands
    assert 0 <= result.best < result.n_bands and 0 <= result.worst < result.n_bands
