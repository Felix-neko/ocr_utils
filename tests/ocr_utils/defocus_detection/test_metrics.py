"""Свойства метрик резкости, ради которых подпакет и написан.

Главные охраняемые инварианты:
  * все метрики монотонно падают с ростом размытия;
  * ``edge_width`` почти не реагирует на вёрстку — количество текста и кегль, —
    тогда как наивный ``laplacian`` реагирует сильно (это и есть причина, по которой
    его нельзя брать для ранжирования);
  * ``edge_width`` не реагирует на экспозицию.
"""

import numpy as np
import pytest
from tests.ocr_utils.defocus_detection.pages import blur, draw_page, expose

from ocr_utils.defocus_detection.metrics import ALGORITHMS
from ocr_utils.defocus_detection.scoring import DEFAULT_AGGREGATION, aggregate
from ocr_utils.defocus_detection.tiles import detail_rms_map, make_grid, printed_mask

# Метрики, которые обязаны быть устойчивы к количеству текста и к экспозиции.
CONTENT_ROBUST = ("edge_width", "reblur", "hf_mid")
# А вот к кеглю устойчивы не все: hf_mid нормируется на средние частоты, и крупный шрифт
# сдвигает вниз обе полосы спектра сразу, так что отношение всё-таки падает.
FONT_SIZE_ROBUST = ("edge_width", "reblur")


def score(image: np.ndarray, algorithm: str, mode: str = DEFAULT_AGGREGATION) -> float:
    """Считает балл резкости кадра выбранным алгоритмом.

    Args:
        image: Полутоновый кадр.
        algorithm: Имя алгоритма из реестра.
        mode: Режим агрегации тайлов (по умолчанию — боевой).

    Returns:
        Балл резкости (больше = резче).
    """
    grid = make_grid(image.shape)
    printed = printed_mask(detail_rms_map(image, grid))
    tile_map = ALGORITHMS[algorithm].tile_sharpness(image, grid)
    return aggregate(tile_map, printed, mode=mode)


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
def test_sharpness_falls_monotonically_with_blur(algorithm: str) -> None:
    """Балл любой метрики должен монотонно падать с ростом размытия."""
    page = draw_page()
    scores = [score(blur(page, sigma), algorithm) for sigma in (0.0, 0.6, 1.2, 2.0)]
    assert all(np.isfinite(s) for s in scores), scores
    assert scores == sorted(scores, reverse=True), f"{algorithm}: {scores}"


@pytest.mark.parametrize("algorithm", CONTENT_ROBUST)
def test_text_amount_barely_moves_the_score(algorithm: str) -> None:
    """Плотная и разреженная полосы в одном фокусе должны получить близкий балл.

    Это ровно тот случай, на котором ломаются энергетические метрики: у полосы,
    заполненной текстом на четверть, суммарной «резкости» вчетверо меньше, хотя
    оптика та же самая.
    """
    dense = score(draw_page(fill=1.0), algorithm)
    sparse = score(draw_page(fill=0.25), algorithm)
    assert abs(dense - sparse) / dense < 0.15, f"{algorithm}: плотная={dense:.4f} разреженная={sparse:.4f}"


@pytest.mark.parametrize("algorithm", FONT_SIZE_ROBUST)
def test_font_size_barely_moves_the_score(algorithm: str) -> None:
    """Полоса крупным кеглем не должна выглядеть расфокусной.

    У крупных букв переходов меньше, но каждый переход такой же крутой — метрика,
    меряющая геометрию края, обязана это видеть.
    """
    small = score(draw_page(stroke=2, line_height=12), algorithm)
    large = score(draw_page(stroke=6, line_height=36), algorithm)
    assert abs(small - large) / small < 0.25, f"{algorithm}: мелкий={small:.4f} крупный={large:.4f}"


@pytest.mark.parametrize("algorithm", ["laplacian", "hf_mid"])
def test_known_metrics_are_fooled_by_font_size(algorithm: str) -> None:
    """Известные дефекты метрик: крупный кегль они принимают за расфокус.

    Тест фиксирует не желаемое поведение, а границы применимости, описанные в README.
    Если он перестанет воспроизводиться — значит метрика изменилась и README устарел.
    """
    small = score(draw_page(stroke=2, line_height=12), algorithm)
    large = score(draw_page(stroke=6, line_height=36), algorithm)
    assert abs(small - large) / small > 0.25, f"{algorithm}: мелкий={small:.4f} крупный={large:.4f}"


@pytest.mark.parametrize("algorithm", CONTENT_ROBUST)
def test_exposure_barely_moves_the_score(algorithm: str) -> None:
    """Разная экспозиция и контраст не должны менять оценку фокуса."""
    page = draw_page()
    normal = score(page, algorithm)
    dim = score(expose(page, gain=0.6, offset=-20), algorithm)
    assert abs(normal - dim) / normal < 0.20, f"{algorithm}: обычный={normal:.4f} тёмный={dim:.4f}"


@pytest.mark.parametrize("algorithm", CONTENT_ROBUST)
def test_blur_beats_layout_and_exposure(algorithm: str) -> None:
    """Лёгкое размытие должно сдвигать балл сильнее, чем вёрстка и экспозиция вместе.

    Это и есть рабочее требование: в отчёте расфокусный кадр обязан оказаться ниже
    любого резкого, как бы тот ни был свёрстан и снят.
    """
    reference = score(draw_page(), algorithm)
    softened = score(blur(draw_page(), 0.8), algorithm)
    distractors = [score(draw_page(fill=0.25), algorithm), score(expose(draw_page(), gain=0.6, offset=-20), algorithm)]
    if algorithm in FONT_SIZE_ROBUST:
        distractors.append(score(draw_page(stroke=6, line_height=36), algorithm))
    assert softened < min(distractors), f"{algorithm}: размытый={softened:.4f} помехи={distractors}"
    assert softened < reference
