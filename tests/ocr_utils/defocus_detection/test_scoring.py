"""Агрегация тайлов в балл файла и сведение метрик в общий ранг."""

import numpy as np

from ocr_utils.defocus_detection.scoring import aggregate, rank_combine


def test_best_mode_ignores_soft_layout_tiles() -> None:
    """Режим best смотрит на лучший контент полосы, а не на среднее по вёрстке.

    Полоса, где шесть тайлов из десяти заняты крупным заголовком (низкая резкость
    по причинам вёрстки), не должна проигрывать полосе из сплошного тела-текста:
    хватает и оставшихся четырёх тайлов тела-текста, чтобы квантиль 0.8 их достал.
    """
    headline_page = np.array([[0.2] * 6 + [0.9] * 4])
    body_page = np.full((1, 10), 0.9)
    printed = np.ones((1, 10), dtype=bool)
    assert aggregate(headline_page, printed, mode="best") == aggregate(body_page, printed, mode="best")
    assert aggregate(headline_page, printed, mode="median") < aggregate(body_page, printed, mode="median")


def test_worst_mode_catches_a_soft_corner() -> None:
    """Режим worst, наоборот, реагирует на локальный провал резкости."""
    zonal = np.array([[0.2, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]])
    clean = np.full((1, 10), 0.9)
    printed = np.ones((1, 10), dtype=bool)
    assert aggregate(zonal, printed, mode="worst") < aggregate(clean, printed, mode="worst")


def test_empty_tiles_are_excluded() -> None:
    """Пустые поля не участвуют в балле, даже если их большинство."""
    tile_map = np.array([[0.0, 0.0, 0.0, 0.0, 0.8, 0.8]])
    printed = np.array([[False, False, False, False, True, True]])
    assert aggregate(tile_map, printed, mode="median") == 0.8


def test_page_without_ink_still_gets_a_score() -> None:
    """Обложка без краски не должна выпадать из отчёта с NaN."""
    tile_map = np.array([[0.3, 0.4, 0.5]])
    printed = np.zeros((1, 3), dtype=bool)
    assert np.isfinite(aggregate(tile_map, printed))


def test_rank_combine_averages_disagreeing_metrics() -> None:
    """Сводный балл — средний нормированный ранг, а не среднее несопоставимых шкал."""
    combined = rank_combine({"a": [1.0, 2.0, 3.0], "b": [300.0, 200.0, 100.0]})
    assert combined == [0.5, 0.5, 0.5]

    agreeing = rank_combine({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    assert agreeing == [0.0, 0.5, 1.0]


def test_rank_combine_survives_a_metric_that_failed() -> None:
    """Если одна метрика вернула NaN, файл ранжируется по остальным."""
    combined = rank_combine({"a": [1.0, 2.0, 3.0], "b": [float("nan"), 20.0, 30.0]})
    assert all(np.isfinite(combined))
    assert combined[0] < combined[1] < combined[2]
