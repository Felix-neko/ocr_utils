"""Дефисы переноса и составных слов: низкие широкие чёрточки у буквы, которые маска глифов выбрасывает.

Краска текста (``page.text_ink``) строится маской глифов (``orientation...ink_axis.glyph_mask``), а та
пропускает компоненты не ниже ``GLYPH_MIN_PX`` = 5 px рабочей копии. Дефис при 150 dpi — примерно
5 × 3 px: широкий, но низкий, и он молча выпадал. Строка с переносом кончалась на дефис, а край её
ряда стоял на 1 мм левее (1976/09 с.92: 49 строк с переносом из 104, дефис за краем ряда на
1.02–1.10 мм). Правый край колонки, выключенной по формату, от этого выглядел рваным: выключка
выходила ``left`` вместо ``both``, дефис оставался вне границы блока по краске.

Дефис узнаётся по двум признакам сразу:

* **форма** — компонента, не прошедшая маску глифов, низкая (до ``HYPHEN_MAX_HEIGHT_MM``) и широкая
  (``HYPHEN_MIN_WIDTH_MM``…``HYPHEN_MAX_WIDTH_MM``, шире своей средней толщины в ``HYPHEN_MIN_ASPECT`` раза),
  плотно залитая (``HYPHEN_MIN_FILL``): точка и запятая не шире своей высоты и сюда не попадают;
* **место** — вплотную справа от буквы (зазор до ``HYPHEN_MAX_GAP_MM``) на уровне её середины: буква
  слева заходит и выше, и ниже дефиса на ``HYPHEN_LETTER_SPAN_MM``. Пыль и обрывки черт так не стоят.

Замер на 1976/09 с.92: 66 дефисов, у правых краёв колонок найдено 40 из 42; всё найденное не у края —
тоже дефисы (переносы в концах абзацев, составные слова «…но-т…»).
"""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout import mm_to_px

HYPHEN_MAX_HEIGHT_MM = 0.75
HYPHEN_MIN_WIDTH_MM = 0.65
HYPHEN_MAX_WIDTH_MM = 2.7
HYPHEN_MIN_ASPECT = 1.3
HYPHEN_MIN_FILL = 0.6
HYPHEN_MAX_GAP_MM = 1.0
HYPHEN_LETTER_SPAN_MM = 0.5


def hyphens_mask(work: np.ndarray, glyphs: np.ndarray, dpi: float) -> np.ndarray:
    """Маска дефисов рабочей копии: низкие широкие чёрточки вплотную справа от буквы.

    Args:
        work: Серая рабочая копия страницы.
        glyphs: Маска глифов той же копии (``glyph_mask``) — по ней ищется буква слева от дефиса,
            а её компоненты сами дефисами не считаются.
        dpi: Разрешение рабочей копии (пороги заданы в мм бумаги).

    Returns:
        Маска ``uint8`` размера ``work``: 255 — пиксель дефиса.
    """
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    if count <= 1:
        return out
    # Компоненты, уже взятые маской глифов, — буквы, а не дефисы.
    taken = np.zeros(count, dtype=bool)
    taken[np.unique(labels[glyphs > 0])] = True
    max_height = mm_to_px(HYPHEN_MAX_HEIGHT_MM, dpi)
    min_width, max_width = mm_to_px(HYPHEN_MIN_WIDTH_MM, dpi), mm_to_px(HYPHEN_MAX_WIDTH_MM, dpi)
    gap = max(1, int(round(mm_to_px(HYPHEN_MAX_GAP_MM, dpi))))
    span = max(1, int(round(mm_to_px(HYPHEN_LETTER_SPAN_MM, dpi))))
    rows, _ = glyphs.shape
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if taken[index] or height > max_height or not min_width <= width <= max_width:
            continue
        # Отношение сторон — по СРЕДНЕЙ толщине (площадь / ширина), а не по высоте бокса: к чёрточке
        # при бинаризации прилипает бледный край, и бокс дефиса 5 × 2.5 px выходил 4 × 4
        # (1973/07 с.88, «организа-»). Точка и запятая (3 × 3, 3 × 4) не проходят и так.
        thickness = area / max(width, 1)
        if width < HYPHEN_MIN_ASPECT * thickness or area < HYPHEN_MIN_FILL * width * height:
            continue
        # Буква слева: в окне шириной ``gap`` перед дефисом краска глифа есть и выше, и ниже его середины.
        middle = int(y + height / 2.0)
        window = glyphs[max(0, middle - span) : min(rows, middle + span + 1), max(0, x - gap) : x] > 0
        if window.shape[0] < 2 * span + 1 or not window[:span].any() or not window[span + 1 :].any():
            continue
        out[labels == index] = 255
    return out


__all__ = ["hyphens_mask"]
