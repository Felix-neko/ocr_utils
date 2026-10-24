"""Сверка меры по словам tesseract: ширина слова к ожидаемой по его буквам (ширины букв подогнаны по словам той же страницы вдали от корешка) — сжатие с поправкой на состав слова, независимая от формы штрихов проверка мер окна."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ocr_utils.scan_markup.tesseract import words
from research.gutter_compression.measure import Side

# Слово в счёт: только строчная кириллица от трёх букв (без знаков: дефис и запятая меняют ширину рамки).
WORD = re.compile(r"^[а-яё]{3,}$")
ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
# Уверенность tesseract в слове не ниже этой.
MIN_CONFIDENCE = 75.0
# Опорные слова (для ширин букв) — ближний к корешку край слова дальше этого от края текста, мм.
FAR_MM = 25.0
# Опорных слов не меньше этого, иначе страница без сверки.
MIN_FAR_WORDS = 80
# Гребневая добавка к нормальным уравнениям ширин букв (px²): редкие буквы не разлетаются.
RIDGE = 5.0


def page_words(gray: np.ndarray, x_lo: int, x_hi: int, spine_x: float, side: Side, edge_mm: float,
               px_per_mm: float) -> pd.DataFrame:  # fmt: skip
    """Слова страницы с отношением ширины к ожидаемой по буквам.

    Args:
        gray: Выпрямленный холст uint8.
        x_lo: Левый край страницы на холсте, px.
        x_hi: Правый край (исключительно), px.
        spine_x: X линии корешка, px.
        side: Страница.
        edge_mm: Отступ края текста стороны от корешка (медиана строк), мм.
        px_per_mm: Плотность, px/мм.

    Returns:
        Таблица слов: text, x0, x1, cy, near_mm и far_mm (концы слова от края текста стороны), width, expected,
        ratio. Пустая — опорных слов мало.
    """
    page = gray[:, x_lo:x_hi]
    rows = []
    for word in words(page, psm=4):
        text = word.text.lower()
        if word.confidence < MIN_CONFIDENCE or not WORD.match(text):
            continue
        x0, x1 = word.left + x_lo, word.right + x_lo
        ends = sorted([abs(x0 - spine_x), abs(x1 - spine_x)])
        rows.append({"text": text, "x0": x0, "x1": x1, "cy": word.top + word.height / 2, "width": word.width,
                     "near_mm": ends[0] / px_per_mm - edge_mm, "far_mm": ends[1] / px_per_mm - edge_mm})  # fmt: skip
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    counts = np.array([[t.count(c) for c in ALPHABET] + [1] for t in table.text], np.float64)
    far = (table.near_mm >= FAR_MM).to_numpy()
    if far.sum() < MIN_FAR_WORDS:
        return pd.DataFrame()
    a, b = counts[far], table.width.to_numpy(np.float64)[far]
    reg = RIDGE * np.eye(a.shape[1])
    reg[-1, -1] = 0.0  # постоянная (боковые просветы рамки) без гребня
    widths = np.linalg.solve(a.T @ a + reg, a.T @ b)
    table["expected"] = counts @ widths
    table["ratio"] = table.width / table.expected
    table.insert(0, "side", side.value)
    return table


def paired_windows(words_table: pd.DataFrame, windows: pd.DataFrame, column: str, line_tol_px: float) -> np.ndarray:
    """Средняя мера окна (столбец ``column``) под каждым словом: окна той же стороны и строки, центр внутри слова.

    Args:
        words_table: Слова стороны (``page_words``).
        windows: Окна той же стороны (``measure.page_windows``) с колонкой ``x_centre``.
        column: Столбец отношения меры.
        line_tol_px: Допуск по высоте между центром слова и строкой окна, px.

    Returns:
        float[n слов] — среднее отношение окон под словом или NaN.
    """
    out = np.full(len(words_table), np.nan)
    xs, ys, vals = windows.x_centre.to_numpy(), windows.y.to_numpy(), windows[column].to_numpy()
    for i, w in enumerate(words_table.itertuples()):
        sel = (np.abs(ys - w.cy) < line_tol_px) & (xs >= w.x0) & (xs <= w.x1) & np.isfinite(vals)
        if sel.any():
            out[i] = float(vals[sel].mean())
    return out
