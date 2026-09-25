"""Тесты адаптеров чужих движков: ось по полигону, разбор единого JSON воркера, меры стенда сравнения."""

import numpy as np

from ocr_utils.page_layout.text_blocks.engines.external import centre_from_polygon, polygon_height
from ocr_utils.page_layout.text_blocks.engines.generic import fill_heights, line_of
from research.line_axis_models.measures import Axis, _wobble


def _arc_polygon(vertices: int, thickness: float = 20.0) -> np.ndarray:
    """Полигон дуги-строки с редкими вершинами: верхняя кромка слева направо, нижняя — обратно."""
    xs = np.linspace(0.0, 400.0, vertices)
    top = np.column_stack([xs, 50.0 + 0.001 * (xs - 200.0) ** 2])
    bottom = np.column_stack([xs[::-1], 50.0 + thickness + 0.001 * (xs[::-1] - 200.0) ** 2])
    return np.vstack([top, bottom])


def test_centre_from_polygon_follows_middle_with_sparse_vertices():
    # Вершин мало, и они не совпадают по x у кромок: прежний расчёт по вершинам давал пилу.
    polygon = _arc_polygon(9)
    centre = centre_from_polygon(polygon)
    expected = 60.0 + 0.001 * (centre[:, 0] - 200.0) ** 2
    assert np.abs(centre[:, 1] - expected).max() < 2.0
    assert abs(polygon_height(polygon) - 21.0) <= 1.0


def test_line_of_prefers_centre_then_baseline_then_polygon():
    polygon = _arc_polygon(9).tolist()
    # Есть готовая центр-линия — берётся она, не базовая линия.
    line = line_of({"centre": [[0, 60], [400, 60]], "baseline": [[0, 70], [400, 70]], "boundary": polygon}, 0.5)
    assert not line.baseline and np.allclose(line.points[:, 1], 30.0)
    # Только базовая линия с x-height (Laypa): высота — x-height, ось поднимется на её половину.
    line = line_of({"baseline": [[0, 70], [400, 70]], "height": 0, "x_height": 16}, 0.5)
    assert line.baseline and line.height == 8.0
    # Только контур — середина залитого полигона.
    line = line_of({"boundary": polygon}, 1.0)
    assert not line.baseline and line.points.shape[0] > 10
    assert line_of({"baseline": [[0, 1]]}, 1.0) is None


def test_fill_heights_uses_page_median():
    lines = [
        line_of({"baseline": [[0, 70], [400, 70]], "height": 20}, 1.0),
        line_of({"baseline": [[0, 90], [400, 90]], "height": 30}, 1.0),
        line_of({"baseline": [[0, 110], [400, 110]]}, 1.0),
    ]
    assert [line.height for line in fill_heights(lines)] == [20.0, 30.0, 25.0]


def test_wobble_separates_saw_from_smooth_curve():
    xs = np.arange(0.0, 400.0, 2.0)
    smooth = Axis(np.column_stack([xs, 60.0 + 0.001 * (xs - 200.0) ** 2]), 0.0)
    saw = Axis(np.column_stack([xs, 60.0 + 5.0 * ((xs // 6) % 2)]), 0.0)
    assert _wobble(smooth, 150.0) < 0.05 < _wobble(saw, 150.0)
