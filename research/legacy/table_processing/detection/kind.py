"""Вид объекта по линейкам: таблица, схема или рисунок.

ЖИВОЙ КОД ПЕРЕЕХАЛ в ``ocr_utils.scan_markup.table_detection.kind``: здесь остался реэкспорт, чтобы стенд исследований
(сравнение версий, добыча, отчёты) мерил тот же детектор, что стоит в конвейере.
"""

from ocr_utils.scan_markup.table_detection.kind import (  # noqa: F401 — реэкспорт для стенда
    ALIGN_TOL_MM,
    CROSS_TOL_MM,
    LONG_SPAN,
    DIAGRAM_MIN_RULES,
    DIAGRAM_MIN_VERTICALS,
    DIAGRAM_MAX_CROSS_DENSITY,
    DIAGRAM_MAX_LONG_SHARE,
    DIAGRAM_MIN_SIDE_MM,
    TALL_SPAN,
    DIAGRAM_MAX_TALL_SHARE,
    DIAGRAM_MAX_LONG_RULES,
    KindFeatures,
    KIND_HEADER,
    _distinct,
    features_of,
    is_diagram,
    classify,
)

__all__ = [
    "ALIGN_TOL_MM",
    "CROSS_TOL_MM",
    "LONG_SPAN",
    "DIAGRAM_MIN_RULES",
    "DIAGRAM_MIN_VERTICALS",
    "DIAGRAM_MAX_CROSS_DENSITY",
    "DIAGRAM_MAX_LONG_SHARE",
    "DIAGRAM_MIN_SIDE_MM",
    "TALL_SPAN",
    "DIAGRAM_MAX_TALL_SHARE",
    "DIAGRAM_MAX_LONG_RULES",
    "KindFeatures",
    "KIND_HEADER",
    "_distinct",
    "features_of",
    "is_diagram",
    "classify",
]
