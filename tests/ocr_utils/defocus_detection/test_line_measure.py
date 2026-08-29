"""Замер резкости по областям строк: реакция на размытие, наклон и кегль."""

import numpy as np
import pytest

from ocr_utils.defocus_detection.lines.measure import LineMeasurements, measure_lines, small_text_mask, tile_of
from ocr_utils.defocus_detection.lines.regions import LineRegion
from ocr_utils.defocus_detection.metrics import ALGORITHMS

from .pages import blur, draw_text_lines

EDGE_WIDTH = ALGORITHMS["edge_width"]


def measure(image, polygons, algorithm=EDGE_WIDTH, n_tiles=3, corridor=(0.0, 100.0)) -> LineMeasurements:
    """Считает замеры по готовым полигонам строк.

    Args:
        image: Полутоновый кадр.
        polygons: Полигоны строк.
        algorithm: Алгоритм оценки резкости.
        n_tiles: Сторона зональной сетки.
        corridor: Коридор перцентилей высоты строки.

    Returns:
        Замеры по строкам.
    """
    regions = [LineRegion(polygon=p) for p in polygons]
    return measure_lines(image, regions, algorithm, n_tiles=n_tiles, corridor=corridor)


def mean_sigma(measurements: LineMeasurements) -> float:
    """Средняя ширина края по измеренным строкам.

    Args:
        measurements: Замеры по строкам.

    Returns:
        σ в пикселях.
    """
    values = measurements.sharpness[measurements.valid]
    return float(1.0 / np.mean(values))


# Метрики, которым нужен кусок кадра приличного размера (спектральные), по кускам строк
# работать не могут и объявляют это флагом supports_regions. Проверять на них замер по
# строкам нечего — режим с ними запрещён в CLI.
REGION_ALGORITHMS = [a for a in ALGORITHMS.values() if a.supports_regions]


@pytest.mark.parametrize("algorithm", REGION_ALGORITHMS, ids=lambda a: a.name)
def test_sharpness_of_chunks_falls_with_blur(algorithm):
    """Любая метрика по областям строк должна монотонно падать с размытием."""
    page, polygons = draw_text_lines(height=800, width=900, stroke=4, line_height=32, seed=1)
    scores = []
    for sigma in (0.0, 0.8, 1.6):
        measurements = measure(blur(page, sigma), polygons, algorithm=algorithm)
        values = measurements.sharpness[measurements.valid]
        assert values.size > 0, "строки должны измеряться"
        scores.append(float(np.mean(values)))
    assert scores[0] > scores[1] > scores[2]


@pytest.mark.parametrize("slant", [0.02, 0.05, 0.10])
def test_slanted_lines_measure_the_same_as_straight_ones(slant):
    """ГЛАВНЫЙ ТЕСТ ПРО ТРАПЕЦИЮ: наклон строк не должен менять измеренную σ.

    Наклонная и прямая полосы нарисованы из одних и тех же штрихов, сдвинутых на целое
    число пикселей, — рисунок каждого штриха дословно совпадает. Поэтому расхождение в σ
    означало бы, что нарезка кусков либо прихватывает бумагу, либо (хуже) где-то
    пересчитывает пиксели.
    """
    straight, straight_polys = draw_text_lines(height=800, width=900, stroke=4, line_height=32, slant=0.0, seed=7)
    slanted, slanted_polys = draw_text_lines(height=800, width=900, stroke=4, line_height=32, slant=slant, seed=7)

    reference = mean_sigma(measure(straight, straight_polys))
    tilted = mean_sigma(measure(slanted, slanted_polys))
    assert tilted == pytest.approx(reference, rel=0.05)


def test_bounding_box_would_be_worse_than_chunks():
    """Тест-документация: почему куски, а не охватывающий прямоугольник строки.

    У наклонной строки bbox прихватывает бумагу сверху и снизу. Бумага — это гладкий
    фон без краёв, и σ по ней завышается: доля «мягких» замеров растёт вместе с наклоном.
    """
    page, polygons = draw_text_lines(height=800, width=900, stroke=4, line_height=32, slant=0.08, seed=7)
    # Кадр размывается: на идеально резкой синтетике σ обеих нарезок упирается в
    # нижнюю отсечку 0.001 px, и сравнивать было бы нечего.
    slanted = blur(page, 1.0)

    by_chunks = mean_sigma(measure(slanted, polygons))

    boxes = [
        np.array(
            [
                [p[:, 0].min(), p[:, 1].min()],
                [p[:, 0].max(), p[:, 1].min()],
                [p[:, 0].max(), p[:, 1].max()],
                [p[:, 0].min(), p[:, 1].max()],
            ]
        )
        for p in polygons
    ]
    by_boxes = mean_sigma(measure(slanted, boxes))
    assert by_boxes > by_chunks, "охватывающий прямоугольник обязан быть хуже — в нём бумага"


def test_normalized_score_is_about_readability_not_optics():
    """σ/высота строки должна ухудшаться при уменьшении кегля, сырая σ — нет.

    Это и есть разница между «оптика попала в фокус» и «мелкий текст читается»: одно и
    то же размытие не мешает крупному набору и убивает петит.
    """
    coarse, coarse_polys = draw_text_lines(height=1100, width=900, stroke=8, line_height=56, seed=2)
    fine, fine_polys = draw_text_lines(height=1100, width=900, stroke=4, line_height=32, seed=2)

    coarse_m = measure(blur(coarse, 1.2), coarse_polys)
    fine_m = measure(blur(fine, 1.2), fine_polys)

    raw = (float(np.nanmean(coarse_m.sharpness)), float(np.nanmean(fine_m.sharpness)))
    assert raw[1] == pytest.approx(raw[0], rel=0.35), "сырая σ мерит оптику и от кегля зависеть почти не должна"

    normalized = (float(np.nanmean(coarse_m.normalized())), float(np.nanmean(fine_m.normalized())))
    assert normalized[1] < normalized[0], "нормированный балл обязан упасть на мелком кегле"


def test_normalized_score_is_not_computed_for_dimensionless_metrics():
    """У безразмерных метрик деление на высоту строки смысла не имеет и не делается."""
    assert EDGE_WIDTH.length_scaled is True
    assert ALGORITHMS["reblur"].length_scaled is False
    assert ALGORITHMS["hf_mid"].length_scaled is False


def test_no_lines_gives_empty_measurements():
    """Кадр без текста (обложка подшивки) не должен ронять замер."""
    page, _ = draw_text_lines(height=400, width=400)
    measurements = measure_lines(page, [], EDGE_WIDTH, n_tiles=3)
    assert measurements.chunks == []
    assert measurements.valid.size == 0


def test_tile_of_puts_centres_in_the_right_cell():
    """Привязка к тайлу идёт по центру тяжести, сетка нумеруется построчно."""
    shape = (900, 900)
    assert tile_of((10.0, 10.0), shape, 3) == 0
    assert tile_of((890.0, 890.0), shape, 3) == 8
    assert tile_of((450.0, 450.0), shape, 3) == 4
    # Точка ровно на границе кадра не должна выпадать за сетку.
    assert tile_of((900.0, 900.0), shape, 3) == 8


def test_small_text_corridor_is_computed_per_tile():
    """Коридор кегля считается ВНУТРИ тайла, а не по всему кадру.

    При трапеции ближний край кадра снят крупнее, и общий на весь кадр коридор выбросил
    бы его целиком — то есть выкосил бы из зональной карты одну сторону, причём ту, где
    искажения сильнее всего.
    """
    # Тайл 0 набран мелко, тайл 1 — вдвое крупнее (имитация трапеции).
    heights = np.array([10.0] * 10 + [11.0] * 10 + [20.0] * 10 + [22.0] * 10)
    tiles = np.array([0] * 20 + [1] * 20)

    keep = small_text_mask(heights, tiles, (0.0, 60.0), n_tiles=2)
    # В каждом тайле осталась его собственная мелкая половина, а не «мелкие по кадру».
    assert keep[:20].sum() > 0 and keep[20:].sum() > 0, "ни один тайл не должен выпасть целиком"
    assert set(heights[keep[:20].nonzero()[0]]) <= {10.0, 11.0}
    assert 20.0 in set(heights[20:][keep[20:]]), "в крупном тайле мелким считается его собственный низ"


def test_tiny_tiles_keep_all_their_lines():
    """Перцентили по горстке строк — лотерея, поэтому бедный тайл берётся целиком."""
    heights = np.array([10.0, 40.0, 12.0])
    tiles = np.zeros(3, dtype=np.int64)
    assert small_text_mask(heights, tiles, (0.0, 60.0), n_tiles=1).all()
