"""Тесты конфигурации."""

from __future__ import annotations

import pytest

from ocr_utils.config import OCR_LANGUAGE, OCR_OVERSAMPLE_DPI, SPREAD_ASPECT_THRESHOLD


def test_spread_threshold_value() -> None:
    """Порог aspect ratio должен быть ≈ 1.152."""
    assert SPREAD_ASPECT_THRESHOLD == pytest.approx(288.0 / 250.0)
    assert 1.1 < SPREAD_ASPECT_THRESHOLD < 1.2


def test_ocr_language() -> None:
    """Язык OCR по умолчанию — русский."""
    assert OCR_LANGUAGE == "rus"


def test_oversample_dpi() -> None:
    """DPI оверсемплинга по умолчанию — 900."""
    assert OCR_OVERSAMPLE_DPI == 900
