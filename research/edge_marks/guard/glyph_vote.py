"""Голосование детекторов «символ или сор» по картам полосы: CRAFT (центры символов), pero ParseNet (базовая линия), docTR DBNet (текст) — компонента сор, если так говорит хотя бы один."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.page_layout.text_blocks.page import PageAnalysis


class Engine(str, Enum):
    """Детектор, чья карта полосы голосует (имя — папка карт ``<maps>/<имя>/<ключ>.png``)."""

    CRAFT = "craft"  # карта «регион» CRAFT: центры символов
    PERO = "pero"  # вероятность базовой линии pero ParseNet
    DOCTR = "doctr"  # вероятность текста docTR DBNet


# Пороги голосования: оценка детектора ниже порога — «сор». Подбор на 188 размеченных кандидатах research/edge_marks
# (reports/edge_guard.md): у каждого порог с запасом от знаков; голосование CRAFT + pero отсекает 54 из 75 кусков
# сора с выступом при 1 знаке из 80, CRAFT в одиночку — 43 при 0; docTR добавляет 1 кусок сора из 106.
# pero — нижний край плато 0.40–0.49 (одни и те же исходы на кандидатах): при 0.45 выброшена «7» номера в оглавлении
# (1972/03 IMG_0105_1L, оценка pero 0.42 при CRAFT 0.72).
THRESHOLDS = {Engine.CRAFT: 0.35, Engine.PERO: 0.40, Engine.DOCTR: 0.07}
# Окно базовой линии pero: база ниже середины строки на столько x-высот, окно по высоте — ± столько x-высот.
BASELINE_BELOW_XH, BASELINE_WINDOW_XH = 0.5, 0.6
# pero голосует, только если середина компоненты не дальше стольких x-высот от середины её строки: вне полосы строки
# окно базовой линии пусто и pero зовёт сором всё подряд (выбросил «7» номера страницы «67» под последней строкой,
# 1974/05). Сор вне полосы строки в строку не затягивается и выступа не даёт.
PERO_ROW_BAND_XH = 1.0


def load_maps(maps_dir: Path, key: str, engines: tuple[Engine, ...]) -> dict[Engine, np.ndarray]:
    """Карты детекторов полосы, 0…1, в пикселях рендера 300 dpi.

    Args:
        maps_dir: Корень карт (``<maps_dir>/<детектор>/<ключ>.png``).
        key: Полоса.
        engines: Какие детекторы.

    Returns:
        ``детектор → карта``.
    """
    return {e: cv2.imread(str(maps_dir / e.value / f"{key}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            for e in engines}  # fmt: skip


def row_geometry(analysis: PageAnalysis, blocks: set[int], y300: float) -> tuple[float, float]:
    """Середина ближайшей по высоте строки аномальных блоков и её x-высота, пиксели рендера.

    Args:
        analysis: Разбор (пиксели рабочей копии).
        blocks: Номера блоков с выступами.
        y300: Ордината компоненты, пиксели рендера.

    Returns:
        ``(середина строки, x-высота)``.
    """
    k = RENDER_DPI / analysis.dpi
    rows = [row for number in blocks for row in analysis.blocks[number].rows]
    row = min(rows, key=lambda r: abs(r.y * k - y300))
    x_h = row.glyph_h if row.glyph_h > 0 else 0.5 * row.height
    return row.y * k, x_h * k


def component_scores(labels: np.ndarray, stats: np.ndarray, index: int, maps: dict[Engine, np.ndarray],
                     row: tuple[float, float]) -> dict[Engine, float]:  # fmt: skip
    """Оценки «это символ» одной компоненты каждым детектором.

    CRAFT и docTR — максимум карты по пикселям компоненты. pero — максимум вероятности базовой линии в столбцах
    компоненты в окне у базовой линии её строки: знак в конце строки продолжает базовую линию, сор — нет. Вне
    полосы строки (``PERO_ROW_BAND_XH``) pero воздерживается — его оценки в ответе нет.

    Args:
        labels: Метки компонент полосы.
        stats: Их статистика ``connectedComponentsWithStats``.
        index: Номер компоненты.
        maps: Карты детекторов.
        row: Середина строки и x-высота (:func:`row_geometry`).

    Returns:
        ``детектор → оценка``.
    """
    x, y, w, h = (int(v) for v in stats[index, :4])
    pixels = labels[y : y + h, x : x + w] == index
    out = {}
    for engine, heat in maps.items():
        if engine is Engine.PERO:
            row_y, x_h = row
            if abs(y + h / 2.0 - row_y) > PERO_ROW_BAND_XH * x_h:
                continue  # вне полосы строки pero воздерживается
            base = row_y + BASELINE_BELOW_XH * x_h
            window = heat[max(0, int(base - BASELINE_WINDOW_XH * x_h)) : int(base + BASELINE_WINDOW_XH * x_h) + 1,
                          x : x + w + 1]  # fmt: skip
            out[engine] = float(window.max()) if window.size else 0.0
        else:
            out[engine] = float(heat[y : y + h, x : x + w][pixels].max())
    return out


def is_junk(scores: dict[Engine, float]) -> bool:
    """Голосование: сор, если хотя бы один детектор дал оценку ниже своего порога ``THRESHOLDS``."""
    return any(value < THRESHOLDS[engine] for engine, value in scores.items())


__all__ = ["Engine", "PERO_ROW_BAND_XH", "THRESHOLDS", "component_scores", "is_junk", "load_maps", "row_geometry"]
