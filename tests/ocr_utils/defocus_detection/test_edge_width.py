"""Проверка оценщика ширины края: он должен мерить именно размытие, в пикселях."""

import numpy as np
from tests.ocr_utils.defocus_detection.pages import blur, draw_page

from ocr_utils.defocus_detection.metrics.edge_width import edge_maps
from ocr_utils.defocus_detection.tiles import make_grid


def mean_sigma(image: np.ndarray) -> float:
    """Средняя по кадру субпиксельная ширина края.

    Args:
        image: Полутоновый кадр.

    Returns:
        Средняя σ края в пикселях.
    """
    grid = make_grid(image.shape)
    sigma, _ = edge_maps(image, grid)
    return float(np.nanmean(sigma))


def test_sharp_edges_measure_near_zero() -> None:
    """У идеально резкой ступеньки после вычета пиксельной апертуры остаётся ~0."""
    assert mean_sigma(draw_page()) < 0.1


def test_sigma_grows_with_blur() -> None:
    """С ростом размытия оценка ширины края растёт монотонно."""
    page = draw_page()
    values = [mean_sigma(blur(page, s)) for s in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert values == sorted(values), values
    assert values[-1] > values[0] + 0.5, values


def test_isolated_step_recovers_true_sigma() -> None:
    """На одиночной широкой ступеньке оценка совпадает с настоящей σ размытия.

    На тексте оценка занижена — соседние штрихи обрезают хвосты профиля, — поэтому
    точность проверяется там, где краю ничего не мешает: на широких полосах.
    """
    page = np.full((512, 512), 235.0)
    page[:, ::64] = 35.0
    for column in range(0, 512, 64):
        page[:, column : column + 24] = 35.0
    page = page.astype(np.uint8)
    grid = make_grid(page.shape)

    for true_sigma in (0.8, 1.2, 1.6):
        sigma, _ = edge_maps(blur(page, true_sigma), grid, min_edges=10)
        measured = float(np.nanmean(sigma))
        assert abs(measured - true_sigma) < 0.2, f"σ={true_sigma}: измерено {measured:.3f}"


def test_text_density_does_not_change_edge_width() -> None:
    """Количество текста меняет число замеров, но не их среднее."""
    dense = mean_sigma(blur(draw_page(fill=1.0), 1.0))
    sparse = mean_sigma(blur(draw_page(fill=0.25), 1.0))
    assert abs(dense - sparse) < 0.1, f"плотная={dense:.3f} разреженная={sparse:.3f}"
