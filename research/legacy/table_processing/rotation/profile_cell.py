"""Ось текста в ячейке по периодичности проекционного профиля.

КЛАССИКА (Постль, семейство «projection profile / Radon»): строки текста дают правильное
чередование краски и бумаги ПОПЕРЁК себя, и вдоль верной оси профиль проекции периодичен.
Периодичность меряется автокорреляцией профиля с вычтенным скользящим средним — иначе
корреляцию забивает огибающая (плато на тексте, ноль на полях).

ЗАЧЕМ ТРЕТЬЯ МЕРА ОСИ, если есть форма буквы и смыкание. Затем, что она ошибается в другом
месте. Форма буквы слепа к ячейке с одной короткой надписью в две-три буквы; смыкание
покупается на колонку чисел; периодичности нужно не меньше трёх-четырёх строк, зато ей
всё равно, что за глифы и как они стоят. Совпадение трёх независимых мер — повод не
смотреть на ячейку глазами, расхождение — повод посмотреть.

Реализация меры взята из ``orientation.detectors.profile`` (та же функция ``periodicity``),
но диапазон межстрочного расстояния задан в миллиметрах и пересчитан под 300 dpi: на
полосе он был 12-47 px при 150 dpi, здесь это 2-8 мм.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from research.legacy.table_processing.detection.ruling import mm_to_px
from research.legacy.table_processing.rotation.base import CellCrop, CellDetector, Verdict, unknown
from research.legacy.table_processing.rotation.ink_axis_cell import glyph_mask

# Межстрочное расстояние: от 2 мм (плотный петит таблицы) до 8 мм (разрежённый набор шапки).
PITCH_MIN_MM = 2.0
PITCH_MAX_MM = 8.0

# Окно скользящего среднего в долях максимального межстрочного: то же, что на полосе.
DETREND_FACTOR = 2.0

# Ниже этого перевеса ось считается неразличимой.
AXIS_MARGIN_THR = 0.15


def periodicity(mask: np.ndarray, horizontal: bool, dpi: int) -> float:
    """Высота пика автокорреляции профиля проекции."""
    pitch_min = mm_to_px(PITCH_MIN_MM, dpi)
    pitch_max = mm_to_px(PITCH_MAX_MM, dpi)
    profile = mask.sum(axis=1 if horizontal else 0).astype(np.float64)
    if profile.size <= pitch_max * 2:
        return 0.0
    window = int(pitch_max * DETREND_FACTOR) | 1
    profile -= uniform_filter1d(profile, size=window, mode="nearest")
    energy = float((profile * profile).sum())
    if energy <= 0.0:
        return 0.0
    correlation = np.correlate(profile, profile, mode="full")[profile.size - 1 :] / energy
    window_slice = correlation[pitch_min : pitch_max + 1]
    return float(window_slice.max()) if window_slice.size else 0.0


def detect(crop: CellCrop) -> Verdict:
    mask = glyph_mask(crop.gray, crop.dpi)
    if not mask.any():
        return unknown("в ячейке нет краски")

    horizontal = max(0.0, periodicity(mask, True, crop.dpi))
    vertical = max(0.0, periodicity(mask, False, crop.dpi))
    total = horizontal + vertical
    if total <= 0.0:
        return unknown("профиль без периодичности")

    axis_score = (horizontal - vertical) / total
    metrics = {"axis_score": axis_score, "peak_h": horizontal, "peak_v": vertical}
    if abs(axis_score) < AXIS_MARGIN_THR:
        return Verdict(0, 0.0, metrics=metrics, note="ось не различается")
    # Периодичность симметрична относительно переворота: строки остаются строками.
    return Verdict(0 if axis_score > 0 else 90, min(1.0, abs(axis_score)), axis_only=True, metrics=metrics)


ALGORITHM = CellDetector(
    name="profile",
    summary="классика: периодичность проекционного профиля (автокорреляция)",
    stage="cpu",
    gives_sign=False,
    run=detect,
)
