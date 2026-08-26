"""Классификация color / grayscale под сепийным налётом бумаги."""

import numpy as np
import pytest

from ocr_utils.scan_markup.detection.color_kind import chroma_fraction, classify, paper_color
from ocr_utils.scan_markup.db.models import KIND_COLOR, KIND_GRAYSCALE

# Цвет пожелтевшей бумаги в BGR: синего заметно меньше красного.
PAPER_BGR = (200, 231, 255)


def _sepia_halftone(seed: int = 0, size: int = 200) -> np.ndarray:
    """Серая типографская печать, окрашенная налётом бумаги: краска нейтральная."""
    rng = np.random.default_rng(seed)
    gray = rng.integers(60, 230, (size, size)).astype(np.float32)
    tint = np.array(PAPER_BGR, np.float32) / 255.0
    return np.clip(gray[..., None] * tint, 0, 255).astype(np.uint8)


def _page_with(region: np.ndarray, size: int = 400) -> np.ndarray:
    page = np.full((size, size, 3), PAPER_BGR, np.uint8)
    page[100 : 100 + region.shape[0], 100 : 100 + region.shape[1]] = region
    return page


def test_paper_color_finds_the_tint() -> None:
    """Цвет бумаги оценивается по светлым пикселям полосы, а не по её среднему."""
    page = _page_with(_sepia_halftone())
    assert np.allclose(paper_color(page), PAPER_BGR, atol=6)


def test_sepia_grayscale_is_not_called_color() -> None:
    """Серый растр на пожелтевшей бумаге — grayscale, хотя насыщенность у него есть.

    Это главный случай, ради которого налёт вообще снимается: без снятия хроматичность
    такой области выше порога на всей площади.
    """
    region = _sepia_halftone()
    page = _page_with(region)
    kind, fraction = classify(region, paper_color(page))
    assert kind == KIND_GRAYSCALE
    assert fraction < 0.02


def test_sepia_without_white_balance_would_look_colored() -> None:
    """Контроль: та же область относительно НЕЙТРАЛЬНОЙ бумаги читается как цветная."""
    assert chroma_fraction(_sepia_halftone(), np.array([128.0, 128.0, 128.0])) > 0.5


@pytest.mark.parametrize("rows,expected", [(10, KIND_GRAYSCALE), (40, KIND_COLOR)])
def test_color_patch_size_decides(rows: int, expected: str) -> None:
    """Цветная плашка переводит область в color, когда занимает заметную долю площади.

    Порог откалиброван по 1966 (см. COLOR_FRAC_THR): у ч/б фотографий остаточная
    хроматичность доходит до 0.052, у настоящих цветных обложек начинается с 0.177.
    Плашка на 5% площади в этот зазор попадает и цветной область не делает, на 20% —
    делает.
    """
    region = _sepia_halftone()
    region[:rows, :, 0] = 40  # синий вниз
    region[:rows, :, 2] = 220  # красный вверх
    page = _page_with(region)
    kind, fraction = classify(region, paper_color(page))
    assert kind == expected
    assert fraction == pytest.approx(rows / 200, abs=0.01)


def test_residual_paper_tint_stays_grayscale() -> None:
    """Ч/б фотография с остаточным налётом (доля до 0.05) в color не уезжает.

    Это главная ошибка, которую ловит порог: балансом белого по всей полосе налёт до конца
    не снимается, у самой фотографии оттенок свой, отличный от полей.
    """
    region = _sepia_halftone()
    region[:8, :, 2] = 210  # слабый тёплый остаток на 4% площади
    assert classify(region, paper_color(_page_with(region)))[0] == KIND_GRAYSCALE


def test_empty_region_is_grayscale() -> None:
    """Вырожденная область не должна ронять прогон."""
    assert classify(np.zeros((0, 0, 3), np.uint8), np.array(PAPER_BGR, np.float32))[0] == KIND_GRAYSCALE
