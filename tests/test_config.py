"""Тесты конфигурации."""

from __future__ import annotations

from ocr_utils.config import OCR_LANGUAGE, OCR_UPSCALE_RATIO


def test_ocr_language() -> None:
    """Язык OCR по умолчанию — русский."""
    assert OCR_LANGUAGE == "rus"


def test_ocr_upscale_ratio() -> None:
    """Коэффициент увеличения перед OCR по умолчанию — 2.0."""
    assert OCR_UPSCALE_RATIO == 2.0
