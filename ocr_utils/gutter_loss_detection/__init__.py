"""Поиск разворотов, у которых текст ушёл под переплёт при тугой подшивке.

Подпакет ранжирует кадры по тому, насколько внутреннее поле полосы съедено сгибом, и
отделяет текстовые полосы (восстановимы по контексту) от табличных (только
пересканировать). Подробности и валидация — в README.md рядом.
"""

from ocr_utils.gutter_loss_detection.analysis import (
    FileResult,
    SideResult,
    analyze_file,
    analyze_folder,
    sort_worst_first,
)
from ocr_utils.gutter_loss_detection.geometry import SideGeometry, SpreadGeometry, analyze_spread, read_work_gray
from ocr_utils.gutter_loss_detection.metrics import THRESHOLD, Verdict, is_tabular, side_bite, spread_bite, verdict

__all__ = [
    "FileResult",
    "SideGeometry",
    "SideResult",
    "SpreadGeometry",
    "THRESHOLD",
    "Verdict",
    "analyze_file",
    "analyze_folder",
    "analyze_spread",
    "is_tabular",
    "read_work_gray",
    "side_bite",
    "sort_worst_first",
    "spread_bite",
    "verdict",
]
