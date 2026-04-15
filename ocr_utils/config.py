"""Общие константы и конфигурация ocr_utils."""

from __future__ import annotations

# Порог соотношения сторон: если ширина/высота >= этого значения, считаем страницу разворотом.
# Получено из типичных размеров журнального разворота: 288мм в ширину / 250мм в высоту ≈ 1.152.
SPREAD_ASPECT_THRESHOLD: float = 288.0 / 250.0  # ≈ 1.152

# Параметры OCR
OCR_LANGUAGE: str = "rus"
OCR_UPSCALE_RATIO: float = 2.0  # Коэффициент увеличения изображений перед OCR

# Имя бинарника jpegtran
JPEGTRAN_BIN: str = "jpegtran"
