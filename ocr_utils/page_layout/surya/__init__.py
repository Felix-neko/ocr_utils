"""Surya layout одним местом: типы блоков, единственная модель, кэш ответов и их источник для детекторов."""

from ocr_utils.page_layout.surya.blocks import (
    EQUATION_LABELS,
    FIGURE_LABELS,
    FORM_LABELS,
    MIN_CONFIDENCE,
    PICTURE_LABELS,
    TABLE_LABELS,
    TEXT_LABELS,
    Block,
    LayoutBlocks,
)
from ocr_utils.page_layout.surya.cache import CACHE_VERSION, SuryaCache
from ocr_utils.page_layout.surya.source import OnMiss, SuryaMissing, SuryaSource

__all__ = [
    "Block",
    "CACHE_VERSION",
    "EQUATION_LABELS",
    "FIGURE_LABELS",
    "FORM_LABELS",
    "LayoutBlocks",
    "MIN_CONFIDENCE",
    "OnMiss",
    "PICTURE_LABELS",
    "SuryaCache",
    "SuryaMissing",
    "SuryaSource",
    "TABLE_LABELS",
    "TEXT_LABELS",
]
