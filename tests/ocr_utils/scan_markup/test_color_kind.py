"""Классификация color / grayscale под сепийным налётом бумаги."""

import numpy as np
import pytest

from ocr_utils.scan_markup.db.models import KIND_COLOR, KIND_GRAYSCALE
from ocr_utils.scan_markup.detection.color_kind import chroma_fraction, classify, paper_color

# Цвет пожелтевшей бумаги в BGR: синего заметно меньше красного.
PAPER_BGR = (200, 231, 255)


def _sepia_halftone(seed: int = 0, size: int = 200) -> np.ndarray:
    """Серая типографская печать, окрашенная налётом бумаги: краска нейтральная."""
    rng = np.random.default_rng(seed)
    gray = rng.integers(60, 230, (size, size)).astype(np.float32)
    tint = np.array(PAPER_BGR, np.float32) / 255.0
    return np.clip(gray[..., None] * tint, 0, 255).astype(np.uint8)


def _shifted_halftone(shift=(-25.0, 0.0, 25.0), size: int = 200) -> np.ndarray:
    """Ч/б фотография с ЧУЖИМ оттенком: тени ушли в синь сильнее, чем поля полосы.

    Ровно это ломало прошлую метрику. Баланс белого по бумаге выправляет один уровень
    яркости, а у фотографии оттенок свой, и остаточная хроматичность получается высокой на
    всей её площади. Оттенок при этом ОДИН на всю область — на том новая метрика и стоит.
    """
    region = _sepia_halftone(size=size).astype(np.float32)
    return np.clip(region + np.array(shift, np.float32), 0, 255).astype(np.uint8)


def _page_with(region: np.ndarray, size: int = 400) -> np.ndarray:
    page = np.full((size, size, 3), PAPER_BGR, np.uint8)
    page[100 : 100 + region.shape[0], 100 : 100 + region.shape[1]] = region
    return page


def test_paper_color_finds_the_tint() -> None:
    """Цвет бумаги оценивается по светлым пикселям полосы, а не по её среднему."""
    page = _page_with(_sepia_halftone())
    assert np.allclose(paper_color(page), PAPER_BGR, atol=6)


def test_sepia_grayscale_is_not_called_color() -> None:
    """Серый растр на пожелтевшей бумаге — grayscale, хотя насыщенность у него есть."""
    region = _sepia_halftone()
    result = classify(region, paper_color(_page_with(region)))
    assert result.kind == KIND_GRAYSCALE
    assert result.chroma_spread < 8.0


def test_sepia_without_white_balance_would_look_colored() -> None:
    """Контроль: та же область относительно НЕЙТРАЛЬНОЙ бумаги читается как цветная."""
    assert chroma_fraction(_sepia_halftone(), np.array([128.0, 128.0, 128.0])) > 0.5


def test_own_tint_no_longer_makes_a_photo_color() -> None:
    """Ч/б фотография со своим оттенком остаётся grayscale, хотя доля хроматичных высока.

    Это главный дефект, ради которого метрика переписана: на паке-1 таких полос набралось
    63 штуки, и у всех абсолютная хроматичность была выше прежнего порога 0.10.
    """
    region = _shifted_halftone()
    result = classify(region, paper_color(_page_with(region)))
    assert result.chroma_frac > 0.10, "иначе случай не тот — прошлая метрика тут не ошибалась"
    assert result.kind == KIND_GRAYSCALE


def test_many_hues_make_a_region_color() -> None:
    """Область с РАЗНЫМИ оттенками — цветная: у неё большой разброс хроматичности."""
    region = _sepia_halftone()
    region[:100, :, 2] = 230  # верх в красное
    region[100:, :, 0] = 230  # низ в синее
    result = classify(region, paper_color(_page_with(region)))
    assert result.kind == KIND_COLOR
    assert result.chroma_spread > 8.0


def test_self_frac_rule_is_off_unless_asked() -> None:
    """Второе условие по умолчанию выключено, но включается порогом.

    Порога, который делит ч/б и цветные по этой метрике, на паке-1 не нашлось, поэтому в
    решении она не участвует. Само число считается всегда — оно пишется в базу.
    """
    region = _shifted_halftone()
    paper = paper_color(_page_with(region))
    assert classify(region, paper).chroma_self_frac >= 0.0
    assert classify(region, paper, chroma_self_frac_thr=-1.0).kind == KIND_COLOR


def test_empty_region_is_grayscale() -> None:
    """Вырожденная область не должна ронять прогон."""
    assert classify(np.zeros((0, 0, 3), np.uint8), np.array(PAPER_BGR, np.float32)).kind == KIND_GRAYSCALE
