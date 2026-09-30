"""Мозаики кандидатов для разметки глазами: в клетке — вырезка первого прохода и залитая (что видел второй проход), подпись с признаками."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import header_strip
from research.margin_marks.sources import read_gray

# Клетка: две вырезки рядом, каждая вписывается в эти размеры (пиксели), и подпись над ними.
PANEL_W, PANEL_H = 330, 250
COLUMNS = 3
GAP = 10


def _fit(gray: np.ndarray | None) -> np.ndarray:
    """Вырезка, вписанная в панель (белое поле вокруг); нет вырезки — серая заглушка."""
    canvas = np.full((PANEL_H, PANEL_W), 255, np.uint8)
    if gray is None:
        canvas[:] = 200
        return canvas
    scale = min(PANEL_W / gray.shape[1], PANEL_H / gray.shape[0])
    small = cv2.resize(
        gray, (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA
    )
    canvas[: small.shape[0], : small.shape[1]] = small
    return canvas


def cell(root: Path, row: dict, number: int, words: list | None = None, inner=None) -> np.ndarray:
    """Клетка мозаики: подпись (номер, id, признаки) и две вырезки «первый проход | залитая».

    Args:
        root: Корень варианта разбора.
        row: Строка ``features.csv``.
        number: Номер клетки на листе.
        words: Слова первого прохода — чтобы залить вырезку заново, если второго прохода не было.
        inner: Рамка кандидата в вырезке (для заливки заново).
    """
    pass1 = read_gray(root, row["id"], "pass1")
    filled = read_gray(root, row["id"], "pass2")
    if filled is None and pass1 is not None and inner is not None:
        from ocr_utils.page_layout.line_art.deepseek.pass2 import fill_words

        filled, _ = fill_words(pass1, inner, words or [])
    pair = np.hstack([_fit(pass1), np.full((PANEL_H, 6), 160, np.uint8), _fit(filled)])
    pair = cv2.cvtColor(pair, cv2.COLOR_GRAY2BGR)
    width = pair.shape[1]
    lines = [
        f"{number}. {row['id']}  {row['classes'] or row['outcome']}  {row['box_w_mm']}×{row['box_h_mm']} мм",
        f"компонент {row['components']}, краска {float(row['ink_share']) * 100:.1f} %, линейной {float(row['linear_share']):.2f}, "
        f"толщина {row['thickness_max_mm']} мм, серость {row['gray_median']}",
        f"первый проход: {row['pass1_nontext'] or 'только текст'}; залитая: "
        f"{'второго прохода' if row['filled_from'] == 'pass2' else 'залита заново (второго прохода не было)'}",
    ]
    return np.vstack([header_strip(lines, width), pair])


def contact_sheet(root: Path, rows: list[dict], title: str, words: dict | None = None, inners: dict | None = None) -> np.ndarray:
    """Мозаика из клеток по ``COLUMNS`` в ряд, с заголовком листа в поле.

    Args:
        root: Корень варианта разбора.
        rows: Строки ``features.csv``.
        title: Заголовок листа.
        words: Слова первого прохода по id — для заливки заново.
        inners: Рамки кандидатов в вырезке по id.
    """
    words, inners = words or {}, inners or {}
    cells = [cell(root, row, number, words.get(row["id"]), inners.get(row["id"])) for number, row in enumerate(rows)]
    height = max(c.shape[0] for c in cells)
    width = cells[0].shape[1]
    cells = [np.vstack([c, np.full((height - c.shape[0], width, 3), 255, np.uint8)]) for c in cells]
    while len(cells) % COLUMNS:
        cells.append(np.full((height, width, 3), 255, np.uint8))
    rows_img = []
    for start in range(0, len(cells), COLUMNS):
        parts = []
        for item in cells[start : start + COLUMNS]:
            parts += [item, np.full((height, GAP, 3), 255, np.uint8)]
        rows_img.append(np.hstack(parts[:-1]))
        rows_img.append(np.full((GAP, rows_img[-1].shape[1], 3), 255, np.uint8))
    body = np.vstack(rows_img[:-1])
    legend = "в клетке: слева — вырезка первого прохода (серая, как есть), справа — залитая: слова закрашены, это видел второй проход DeepSeek"
    return np.vstack([header_strip([title, legend], body.shape[1]), body])


__all__ = ["cell", "contact_sheet"]
