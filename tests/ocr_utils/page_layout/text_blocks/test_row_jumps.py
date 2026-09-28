"""Перескок оси на соседнюю строку: мера ``metrics.row_jumps_of`` и резка двухрядных сгустков ``segment._split_two_rows``."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks.metrics import ROW_JUMP_PITCH_SHARE, row_jumps_of
from ocr_utils.page_layout.text_blocks.segment import SCALES, _smeared

# Синтетические оси (пиксели рабочей копии): межстрочный шаг, длина строк и их число.
PITCH = 23.0
LINE_X = (50.0, 850.0)
LINES = 8

# Маска двух рядов букв (пиксели рабочей копии): высота и ширина буквы, просвет, шаг рядов.
LETTER_H = 12
LETTER_W = 8
LETTER_GAP = 3
ROW_STEP = 23


def _flat_axes(bow: float = 0.0) -> list[np.ndarray]:
    """Оси ``LINES`` строк с шагом ``PITCH``; ``bow`` — прогиб середины всех строк (изгиб бумаги)."""
    xs = np.linspace(*LINE_X, 200)
    sag = bow * (1.0 - ((xs - xs.mean()) / (xs.max() - xs.mean())) ** 2)
    return [np.column_stack([xs, 100.0 + index * PITCH + sag]) for index in range(LINES)]


def test_measure_is_quiet_on_bowed_lines():
    """Изогнутые строки: соседи повторяют изгиб, расстояние между осями постоянно — перескоков нет."""
    assert row_jumps_of(_flat_axes(bow=15.0), PITCH) == []


def test_measure_catches_oblique_jump():
    """Ось, косо уходящая на строку ниже (переход длиной в несколько шагов), — перескок."""
    axes = _flat_axes()
    xs = axes[3][:, 0]
    # На участке x 200..300 ось плавно опускается на шаг и дальше идёт по следующей строке.
    drop = np.clip((xs - 200.0) / 100.0, 0.0, 1.0) * PITCH
    axes[3] = np.column_stack([xs, axes[3][:, 1] + drop])
    found = row_jumps_of(axes, PITCH)
    assert [jump.axis for jump in found] == [3]
    assert found[0].drift > ROW_JUMP_PITCH_SHARE * PITCH


def _row(mask: np.ndarray, top: int, x0: int, x1: int) -> None:
    """Нарисовать ряд букв-прямоугольников от ``x0`` до ``x1`` с верхом ``top``."""
    for x in range(x0, x1 - LETTER_W, LETTER_W + LETTER_GAP):
        mask[top : top + LETTER_H, x : x + LETTER_W] = 255


def _two_rows_with_bridge() -> np.ndarray:
    """Маска: верхний ряд «ярмарка», нижний «реализовано» и отдельный глиф-черта, связывающий их смыканием.

    Как на 1966/01 IMG_0011_2R: черта букв не касается, но её верх стоит на высоте верхнего ряда в
    4 px справа от его конца, а низ — на высоте нижнего ряда в 4 px слева от его начала: смыкание RLSA
    (ядро 8 px) сливает оба ряда в один сгусток через неё.
    """
    mask = np.zeros((80, 300), dtype=np.uint8)
    top, bottom = 20, 20 + ROW_STEP
    _row(mask, top, 10, 120)  # верхний ряд кончается на x ≈ 116
    _row(mask, bottom, 140, 290)  # нижний начинается с x 140
    cv2.line(mask, (122, top + 2), (135, bottom + LETTER_H - 2), 255, 2)
    return mask


def test_bridge_merges_rows_without_split():
    """Без резки оба ряда — один сгусток (проверка, что синтетика воспроизводит мост)."""
    count, _, _ = _smeared(_two_rows_with_bridge(), SCALES[0], None, split_rows=False)
    assert count - 1 == 1


def test_split_separates_rows():
    """С резкой ряды расходятся по разным сгусткам, и ни один их сгусток не выше полутора букв."""
    count, labels, stats = _smeared(_two_rows_with_bridge(), SCALES[0], None, split_rows=True)
    upper = set(np.unique(labels[20 : 20 + LETTER_H, 10:110])) - {0}
    lower = set(np.unique(labels[43 : 43 + LETTER_H, 150:280])) - {0}
    assert upper and lower and not upper & lower
    assert all(stats[index, cv2.CC_STAT_HEIGHT] <= 1.5 * LETTER_H for index in upper | lower)


def test_split_leaves_single_row_alone():
    """Один ряд букв (с прописной повыше) не режется: карта и статистика те же, что без резки."""
    mask = np.zeros((60, 300), dtype=np.uint8)
    _row(mask, 20, 10, 290)
    mask[14:20, 10:18] = 255  # прописная: буква выше строчных
    plain = _smeared(mask, SCALES[0], None, split_rows=False)
    split = _smeared(mask, SCALES[0], None, split_rows=True)
    assert plain[0] == split[0]
    assert np.array_equal(plain[1], split[1])
