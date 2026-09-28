"""Гладкие боковые стороны блока — перенесены в боевой код (``ocr_utils.page_layout.text_blocks.smooth_sides``); здесь — переэкспорт для стенда."""

from ocr_utils.page_layout.text_blocks.smooth_sides import *  # noqa: F401,F403
from ocr_utils.page_layout.text_blocks.smooth_sides import EdgeKind, SideCurve, side_curve  # noqa: F401
from ocr_utils.page_layout.text_blocks.smooth_sides import PUNCT_MM  # noqa: F401
