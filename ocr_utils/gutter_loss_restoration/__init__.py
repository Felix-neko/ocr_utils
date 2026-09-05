"""Восстановление текста, срезанного корешком при тугой подшивке.

Читает полосы распознавателем, вычисляет утраченное слово по переносу и словарю
выпуска, набирает его литерами с той же бумаги. Подробности — в README.md рядом.
"""

from ocr_utils.gutter_loss_restoration.pipeline import build_shared, restore_many
from ocr_utils.gutter_loss_restoration.restore import LineReport, restore_spread

__all__ = ["LineReport", "build_shared", "restore_many", "restore_spread"]
