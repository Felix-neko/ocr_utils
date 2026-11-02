"""Тесты выпрямления вырезки находки вдоль изогнутых горизонтальных линеек (``tables.curve``, TABLES_VERSION 8)."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.tables.curve import MIN_SAG_MM, straighten, straightened

DPI = 150


def _curved_table(sag_px: float) -> np.ndarray:
    """Серая вырезка 150 dpi: три горизонтальные линейки «улыбкой» с прогибом ``sag_px`` и две вертикали."""
    gray = np.full((200, 600), 255, np.uint8)
    xs = np.arange(20, 580)
    bow = sag_px * (1 - ((xs - 300) / 280.0) ** 2)  # прогиб посередине
    for base in (30, 60, 170):
        ys = np.round(base + bow).astype(int)
        for dy in (0, 1):
            gray[ys + dy, xs] = 0
    gray[30:172, 400:402] = 0
    gray[30:172, 500:502] = 0
    return gray


def test_straighten_flattens_bowed_rules():
    """Линейки с прогибом 12 px (2 мм) после выпрямления лежат в одной строке (размах оси ≤ 2 px)."""
    gray = _curved_table(12.0)
    flat, sag = straighten(gray, DPI)
    assert sag >= MIN_SAG_MM
    for base in (30, 60, 170):
        band = flat[base - 6 : base + 20, 40:560] < 128
        rows = [np.flatnonzero(band[:, c]).mean() for c in range(0, 340, 20) if band[:, c].any()]  # левее вертикалей
        assert np.ptp(rows) <= 2.0


def test_straight_table_untouched():
    """Без изгиба вырезка возвращается как есть."""
    gray = _curved_table(0.0)
    flat, sag = straighten(gray, DPI)
    assert sag < MIN_SAG_MM and flat is gray


def test_straightened_returns_flat_only_when_bowed():
    """``straightened`` отдаёт выпрямленную вырезку при изгибе и исходную без него."""
    bowed, plain = _curved_table(12.0), _curved_table(0.0)
    assert straightened(bowed, DPI) is not bowed
    assert straightened(plain, DPI) is plain
