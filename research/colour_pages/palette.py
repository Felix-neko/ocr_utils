"""Палитра страницы и рост бумаги у края по ней — перенесены в пакет (``scan_cropping.v2.colour_pages``); здесь — реэкспорт для стенда."""

from ocr_utils.scan_cropping.v2.colour_pages import (  # noqa: F401
    CENTRE_SHARE,
    LATERAL_MARGIN_MM,
    PALETTE_L_LIGHT,
    PALETTE_L_SHADE,
    PALETTE_MIN_SHARE,
    QUAD_GROW_MM,
    REACH_MM,
    between_sides,
    centre_quad,
    edge_barrier,
    edge_light,
    grow_by_palette,
    in_palette,
    palette_of,
)
