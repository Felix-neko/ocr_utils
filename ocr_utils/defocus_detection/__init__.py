"""Поиск расфокусированных кадров в папке со сканами.

Подпакет ранжирует все изображения папки по качеству фокуса и показывает самые
подозрительные — те, что стоит переснять. Подробности и валидация — в README.md рядом.
"""

from ocr_utils.defocus_detection.analysis import FileResult, analyze_file, analyze_folder, sort_worst_first
from ocr_utils.defocus_detection.metrics import ALGORITHMS, CHOICES, COMBO_NAME

__all__ = ["ALGORITHMS", "CHOICES", "COMBO_NAME", "FileResult", "analyze_file", "analyze_folder", "sort_worst_first"]
