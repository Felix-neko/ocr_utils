"""Детектор четвёртой версии (v4.2): связные рёбра, вид объекта, граница без разрезанных букв.

ЖИВОЙ КОД ПЕРЕЕХАЛ в ``ocr_utils.scan_markup.table_detection.detector``: здесь остался реэкспорт, чтобы стенд исследований
(сравнение версий, добыча, отчёты) мерил тот же детектор, что стоит в конвейере.
Политика ``POLICY_V4`` конвейера зовётся ``POLICY``; здесь оба имени.
"""

from ocr_utils.scan_markup.table_detection.detector import (  # noqa: F401 — реэкспорт для стенда
    TABLE_POLICY,
    POLICY,
    MIN_LONG_RULES_FOR_TABLE,
    HALO_MM,
    CONTINUATION_GAP_MM,
    GROW_CAP_MM,
    GROW_PASSES,
    MAX_FOREIGN_GAIN,
    DIAGRAM_GLUE_MM,
    DIAGRAM_CAP_MM,
    TEXT_MAX_HEIGHT_MM,
    TEXT_MAX_WIDTH_MM,
    PAGE_RULE_SPAN,
    FIGURE_SLACK_MM,
    BANNER_GLYPH_P90_MM,
    BANNER_MIN_GLYPHS,
    BANNER_MAX_HEIGHT_MM,
    FRAME_MAX_RULES,
    glyph_height_p90,
    LAYOUT_COVER,
    KIND_FORM,
    _extent,
    _inside,
    _overlaps,
    _centre_inside,
    _cap,
    _stop_before,
    _text_ink,
    grow_table,
    line_art_mask,
    grow_diagram,
    _layout_kind,
    detect,
    _merge_diagrams,
    _union,
    _with,
    _deduplicate,
)
from research.legacy.table_processing.detection.base import TableDetector

POLICY_V4 = POLICY

ALGORITHM = TableDetector(
    name="ruling_v4",
    summary="связные рёбра, вид объекта (таблица/схема/рисунок), граница без разрезанных букв (CPU)",
    stage="cpu",
    run=detect,
)
