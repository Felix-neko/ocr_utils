"""Кандидаты в псевдоглифы и их признаки стенда — те же, что в ядре (``text_blocks.pseudo_glyphs``): крайние компоненты рядов и компоненты у края вырезки."""

from __future__ import annotations

from ocr_utils.page_layout.text_blocks.pseudo_glyphs import (
    EDGE_MM,
    OUTWARD_MM,
    CandidateKind,
    Components,
    EdgeSide,
    components_of,
    page_candidates,
    row_members,
    stroke_mm,
    word_spaces_mm,
)

__all__ = ["EDGE_MM", "OUTWARD_MM", "CandidateKind", "Components", "EdgeSide", "components_of", "page_candidates",
           "row_members", "stroke_mm", "word_spaces_mm"]  # fmt: skip
