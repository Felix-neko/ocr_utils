"""Поиск полос, на которых просвечивает текст с оборота листа (show-through).

Подпакет ранжирует страницы папки со сканами по силе просвета и выдаёт список кадров,
которые стоит пересканировать с другого экземпляра. Подробности и валидация — в
README.md рядом.
"""

from ocr_utils.show_through_detection.analysis import (
    FileResult,
    HalfResult,
    all_halves,
    analyze_file,
    analyze_folder,
    sort_files_worst_first,
    sort_worst_first,
)
from ocr_utils.show_through_detection.metrics import ALGORITHMS, CHOICES, COMBO_NAME
from ocr_utils.show_through_detection.zones import Zones, build_zones

__all__ = [
    "ALGORITHMS",
    "CHOICES",
    "COMBO_NAME",
    "FileResult",
    "HalfResult",
    "Zones",
    "all_halves",
    "analyze_file",
    "analyze_folder",
    "build_zones",
    "sort_files_worst_first",
    "sort_worst_first",
]
