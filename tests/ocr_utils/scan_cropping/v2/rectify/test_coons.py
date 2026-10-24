"""Тесты карты выпрямления по поверхности Кунса на синтетике: тождество на прямоугольнике, выпрямление дуг, совпадение сетки с поточечной формулой, продолжение за область."""

import numpy as np

from ocr_utils.scan_cropping.v2.page_frames import PageFrame
from research.page_rectify.coons import ArcCurve, page_patch, plan_layout, rectify_maps

TANGENT_PX = 20.0


def line(a: tuple[float, float], b: tuple[float, float], n: int = 50) -> np.ndarray:
    """Отрезок ``a → b`` из ``n`` точек."""
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) * np.array(a) + t * np.array(b)


def arc(x0: float, x1: float, y: float, bulge: float, n: int = 200) -> np.ndarray:
    """Дуга-парабола от ``x0`` до ``x1`` на высоте ``y`` со стрелкой ``bulge`` px (вниз — положительно)."""
    xs = np.linspace(x0, x1, n)
    s = (xs - x0) / (x1 - x0)
    return np.stack([xs, y + 4 * bulge * s * (1 - s)], axis=1)


def spread(bulge: float) -> list[PageFrame]:
    """Разворот 2×(200×300) с корешком на x = 250: верх и низ страниц — дуги со стрелкой ``bulge``."""
    frames = []
    for x_far, x_spine in ((50.0, 250.0), (450.0, 250.0)):
        lo, hi = min(x_far, x_spine), max(x_far, x_spine)
        top, bottom = arc(lo, hi, 50.0, bulge), arc(lo, hi, 350.0, bulge)
        if x_far > x_spine:
            top, bottom = top[::-1], bottom[::-1]
        far = line((x_far, top[0, 1]), (x_far, bottom[0, 1]))
        spine = np.array([top[-1], bottom[-1]])
        frames.append(PageFrame(top, bottom, far, spine, np.zeros((0, 2))))
    return frames


def identity(points: np.ndarray) -> np.ndarray:
    """Перевод координат без сдвига (области уже в px итога)."""
    return np.asarray(points, np.float64)


def patches_of(frames: list[PageFrame]):
    """Области обеих страниц."""
    return page_patch(frames[0], True, identity, TANGENT_PX), page_patch(frames[1], False, identity, TANGENT_PX)


def test_arc_curve_extends_by_tangent():
    """За концами кривая продолжается по касательной: прямая остаётся прямой."""
    curve = ArcCurve.from_points(line((0, 0), (100, 0)), TANGENT_PX)
    pts = curve.at(np.array([-0.5, 0.5, 1.5]))
    np.testing.assert_allclose(pts, [[-50, 0], [50, 0], [150, 0]], atol=1e-9)


def test_rectangle_maps_to_itself():
    """Прямоугольные страницы: карта — тождество (холст совпадает с итогом)."""
    patches = patches_of(spread(0.0))
    layout = plan_layout(patches, (400, 500))
    assert (layout.width, layout.height) == (500, 400)
    map_x, map_y = rectify_maps(patches, layout)
    rows, cols = np.mgrid[0:400, 0:500]
    np.testing.assert_allclose(map_x, cols, atol=1e-3)
    np.testing.assert_allclose(map_y, rows, atol=1e-3)


def test_curved_sides_land_on_rectangle_edges():
    """Дуги верха и низа (стрелка 20 px) ложатся на стороны прямоугольника: верхняя строка прямоугольника берётся с дуги."""
    frames = spread(20.0)
    patches = patches_of(frames)
    layout = plan_layout(patches, (400, 500))
    map_x, map_y = rectify_maps(patches, layout)
    rect = layout.rects[0]
    row = int(round(rect.y0))
    cols = np.arange(int(np.ceil(rect.x0)) + 1, int(rect.x1) - 1)
    # Строка верха прямоугольника → точки на дуге верха левой страницы.
    xs, ys = map_x[row, cols], map_y[row, cols]
    s = (xs - 50.0) / 200.0
    expected = 50.0 + 4 * 20.0 * s * (1 - s)
    # Прямоугольник стоит на целой строке, а его верх — на дробной высоте угла: допуск — доля пикселя.
    assert np.abs(ys - expected - (row - rect.y0)).max() < 0.6


def test_grid_matches_pointwise():
    """Сетка (раздельная свёртка) совпадает с поточечной формулой Кунса, в том числе вне области."""
    patch = patches_of(spread(15.0))[0]
    u = np.linspace(-0.3, 1.2, 17)
    v = np.linspace(-0.2, 1.3, 13)
    gx, gy = patch.grid(u, v)
    uu, vv = np.meshgrid(u, v)
    pts = patch.points(uu.ravel(), vv.ravel())
    np.testing.assert_allclose(gx.ravel(), pts[:, 0], atol=1e-2)
    np.testing.assert_allclose(gy.ravel(), pts[:, 1], atol=1e-2)


def test_extrapolation_monotone():
    """За областью карта без разрывов и складок: X растёт по столбцам, Y — по строкам на всём холсте."""
    patches = patches_of(spread(20.0))
    layout = plan_layout(patches, (400, 500))
    map_x, map_y = rectify_maps(patches, layout)
    assert (np.diff(map_x, axis=1) > 0).all()
    assert (np.diff(map_y, axis=0) > 0).all()
    # Шов на корешке: соседние столбцы по обе стороны отличаются не больше чем на пару пикселей.
    split = int(np.ceil(layout.spine_x))
    assert np.abs(map_y[:, split] - map_y[:, split - 1]).max() < 2.0
