"""Минимальный читаемый кегль и требуемый DPI страницы.

КОНСТАНТА В ПИКСЕЛЯХ, А НЕ В ПУНКТАХ. Распознавателю важен размер знака на растре, а не
на бумаге: 8 pt при 600 dpi читается лучше, чем 12 pt при 150. Поэтому порог задан как
em-размер шрифта в пикселях и не зависит от dpi пака, который плавает от пака к паку.

ОТКУДА 40. Рекомендация ABBYY для FineReader: 300 dpi достаточно для кегля от 10 pt, для
более мелкого — 400–600 dpi; 10 pt при 300 dpi — это em в 41.7 px, 8 pt при 400 dpi —
44 px, 6 pt при 600 — 50 px. У tesseract известный порог — x-height не ниже 20 px, что при
x-height ≈0.5 em даёт те же 40. Набор здесь чистый (шрифт, а не скан), так что 40 — с запасом.

ТРЕБУЕМЫЙ DPI. Если самый мелкий вписанный кегль ``f`` px меньше порога, страницу надо
увеличить в ``MIN_FONT_EM_PX / f`` раз — при сохранении геометрии таблицы кегль вырастет
ровно во столько же. Потолок 1350 dpi задан условием задачи; выше него страницу не растим,
а меняем геометрию таблицы (``reshape``).
"""

from __future__ import annotations

import math

# Минимальный em-размер шрифта в пикселях, при котором распознаватель ещё читает уверенно.
MIN_FONT_EM_PX = 40

# Потолок DPI страницы после увеличения.
MAX_DPI = 1350

# Кегль замены не крупнее этого — «не более 10 кегля» по условию задачи.
MAX_FONT_PT = 10


def pt_to_px(points: float, dpi: float) -> float:
    return points * dpi / 72.0


def font_cap_px(dpi: float, native_font_px: float, min_font_em_px: int = MIN_FONT_EM_PX) -> int:
    """Верхняя граница кегля замены: не крупнее 10 pt и не крупнее собственного кегля
    таблицы — но собственный кегль ниже порога читаемости не ограничивает, иначе на паке
    в 300 dpi замена была бы мельче порога только потому, что мелок оригинал."""
    cap = pt_to_px(MAX_FONT_PT, dpi)
    if native_font_px > 0:
        cap = min(cap, max(native_font_px, float(min_font_em_px)))
    return max(1, int(math.floor(cap)))


def required_dpi(dpi: int, min_font_px: float, min_font_em_px: int = MIN_FONT_EM_PX) -> "int | None":
    """DPI страницы, при котором самый мелкий кегль дорастает до порога. None — кегль
    неизвестен (текст не влез вовсе), и увеличением это не лечится."""
    if min_font_px <= 0:
        return None
    if min_font_px >= min_font_em_px:
        return dpi
    return int(math.ceil(dpi * min_font_em_px / min_font_px))


def font_px_for_dpi(dpi: int, target_dpi: int, min_font_em_px: int = MIN_FONT_EM_PX) -> int:
    """Какой кегль в исходных пикселях дорастёт до порога при увеличении до ``target_dpi``."""
    return max(1, int(math.ceil(min_font_em_px * dpi / target_dpi)))
