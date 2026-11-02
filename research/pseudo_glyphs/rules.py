"""Правила «псевдоглиф или знак» стенда — те же, что в ядре (``text_blocks.pseudo_glyphs``); ``RuleParams`` — прежнее имя ``PseudoParams``."""

from __future__ import annotations

from ocr_utils.page_layout.text_blocks.pseudo_glyphs import PseudoParams as RuleParams
from ocr_utils.page_layout.text_blocks.pseudo_glyphs import PseudoReason, reason_of, reasons, relative_height

__all__ = ["PseudoReason", "RuleParams", "reason_of", "reasons", "relative_height"]
