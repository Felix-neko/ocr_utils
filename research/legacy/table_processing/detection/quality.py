"""Меры качества рамки и компоненты краски под границей.

ЖИВОЙ КОД ПЕРЕЕХАЛ в ``ocr_utils.page_layout.tables.quality``: здесь остался реэкспорт, чтобы стенд исследований
(сравнение версий, добыча, отчёты) мерил тот же детектор, что стоит в конвейере.
"""

from ocr_utils.page_layout.tables.quality import (  # noqa: F401 — реэкспорт для стенда
    SIDES,
    CLEAN_SHARE,
    SEARCH_MM,
    HALO_MM,
    BoxQuality,
    HEADER,
    _profiles,
    edge_ink,
    clearance,
    _first_clean,
    outside_rules,
    outside_rules_page,
    _overlap,
    foreign_text,
    measure,
    MIN_GLYPH_AREA_PX,
    glyph_components,
    straddling,
    crossed_glyphs,
)

__all__ = [
    "SIDES",
    "CLEAN_SHARE",
    "SEARCH_MM",
    "HALO_MM",
    "BoxQuality",
    "HEADER",
    "_profiles",
    "edge_ink",
    "clearance",
    "_first_clean",
    "outside_rules",
    "outside_rules_page",
    "_overlap",
    "foreign_text",
    "measure",
    "MIN_GLYPH_AREA_PX",
    "glyph_components",
    "straddling",
    "crossed_glyphs",
]
