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


def test_split_three_rows_bridged_by_one_stroke():
    """Карандашная кривая через три строки: резка повторяется, и каждая строка — свой сгусток."""
    mask = np.zeros((110, 300), dtype=np.uint8)
    tops = (15, 15 + ROW_STEP, 15 + 2 * ROW_STEP)
    for top in tops:
        _row(mask, top, 10, 290)
    # Пробелы под кривую и сама кривая — от первой строки до третьей.
    for top in tops:
        mask[top : top + LETTER_H, 140:160] = 0
    cv2.line(mask, (157, tops[0] + 2), (143, tops[2] + LETTER_H - 2), 255, 2)
    _, labels, stats = _smeared(mask, SCALES[0], None, split_rows=True)
    rows = [set(np.unique(labels[top : top + LETTER_H, 20:120])) - {0} for top in tops]
    assert all(rows) and not (rows[0] & rows[1]) and not (rows[1] & rows[2])
    assert all(stats[i, cv2.CC_STAT_HEIGHT] <= 1.5 * LETTER_H for row in rows for i in row)


def test_percent_upper_ring_sits_on_baseline():
    """Верхний кружок «%» (глиф со строчную над чертой) — знак над буквой: базовая линия с прямой строки."""
    from ocr_utils.page_layout.text_blocks.pieces import baselines_of

    x_h = 12.0
    letters = [[10 + 12 * i, 100 - x_h / 2, 9, x_h] for i in range(6)]  # cx, cy, w, h: низ на 100
    ring = [86, 87, 6, 12]  # низ на 93 — на 7 px выше строки (больше полувысоты строчной)
    slash = [88, 98, 10, 14]  # черта с нижним кружком: центр масс внизу (кружок тяжелее черты), низ на 105
    base = baselines_of(np.array(letters + [ring, slash], dtype=np.float64), x_h)
    assert abs(base[6] - 100.0) < 1.5


def _piece(x0: float, y: float, letters: int, slope: float = 0.0):
    """Кусок строки из ``letters`` букв шириной 9 px с шагом 12 px, якоря на ``y`` (с наклоном ``slope``)."""
    from ocr_utils.page_layout.text_blocks.pieces import Piece

    xs = x0 + 12.0 * np.arange(letters) + 4.5
    anchors = np.column_stack([xs, y + slope * (xs - xs[0])])
    return Piece(
        blobs=(int(x0),),
        anchors=anchors,
        sizes=np.tile([9.0, 12.0], (letters, 1)),
        marks=np.zeros(letters, dtype=bool),
        x0=float(x0),
        y0=float(anchors[:, 1].min() - 6),
        x1=float(xs[-1] + 4.5),
        y1=float(anchors[:, 1].max() + 6),
        x_h=12.0,
        letter_w=9.0,
        leader_dots=0,
    )


def test_joint_step_blocks_link_to_next_line():
    """Конец абзаца и слово следующей строки (ступенька в шаг) не сливаются; соседние слова одной строки — сливаются."""
    from ocr_utils.page_layout.text_blocks.zones import _guard, joint_step

    left = _piece(100.0, 200.0, 6)
    same = _piece(185.0, 200.5, 5)
    below = _piece(185.0, 200.0 + PITCH, 5)
    assert abs(joint_step(left, same)[0]) < 2.0
    assert _guard(left, same, PITCH)
    assert joint_step(left, below)[0] > 0.5 * PITCH
    assert not _guard(left, below, PITCH)


def test_joint_step_follows_tilted_line():
    """Строка с наклоном 4°: продолжение по касательной ступенькой не считается."""
    from ocr_utils.page_layout.text_blocks.zones import _guard

    tilt = np.tan(np.radians(4.0))
    left = _piece(100.0, 200.0, 6, tilt)
    right = _piece(200.0, 200.0 + tilt * (200.0 - 104.5), 6, tilt)
    assert _guard(left, right, PITCH)


def test_joint_step_short_piece_below_is_blocked():
    """Кусок из двух знаков («* *») на полстроки ниже конца строки не сцепляется; тот же кусок на строке — сцепляется."""
    from ocr_utils.page_layout.text_blocks.zones import _guard

    left = _piece(100.0, 200.0, 8)
    stars_low = _piece(210.0, 200.0 + 0.66 * PITCH, 2)
    stars_same = _piece(210.0, 201.0, 2)
    assert not _guard(left, stars_low, PITCH)
    assert _guard(left, stars_same, PITCH)


def test_joint_step_ignores_marks_only_piece():
    """Кусок из одной точки или дефиса ступенькой не меряется (точка сидит на базовой линии, а не на оси)."""
    from ocr_utils.page_layout.text_blocks.zones import joint_step

    left = _piece(100.0, 200.0, 8)
    dot = _piece(200.0, 206.0, 1)
    dot = dot.__class__(**{**dot.__dict__, "marks": np.ones(1, dtype=bool)})
    assert joint_step(left, dot) is None


def _page_rows(tops: tuple[int, ...]) -> np.ndarray:
    """Маска с рядами букв на высотах ``tops`` — фон для проверки резки глифов (нужна медиана букв страницы)."""
    mask = np.zeros((140, 400), dtype=np.uint8)
    for top in tops:
        _row(mask, top, 10, 390)
    return mask


def test_cut_glyph_grown_from_two_rows():
    """Выносной элемент буквы верхнего ряда касается буквы нижнего: глиф режется по перемычке."""
    from ocr_utils.page_layout.text_blocks.segment import _cut_tall_glyphs

    mask = _page_rows((20, 20 + ROW_STEP))
    # «у» верхнего ряда (x 150..158) хвостом тонкой перемычкой достаёт до «Ш» нижнего.
    mask[20 + LETTER_H : 20 + ROW_STEP, 153:155] = 255
    before = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)[0]
    cut = _cut_tall_glyphs(mask, SCALES[0])
    after = cv2.connectedComponentsWithStats((cut > 0).astype(np.uint8), 8)[0]
    assert after == before + 1


def test_tall_single_glyph_is_not_cut():
    """Одиночная высокая буква без перемычки (скобка во всю высоту строки с выносными) не режется."""
    from ocr_utils.page_layout.text_blocks.segment import _cut_tall_glyphs

    mask = _page_rows((20, 20 + ROW_STEP, 20 + 2 * ROW_STEP))
    mask[40:70, 200:206] = 255  # толстая вертикаль в два ряда: перемычки нет
    cut = _cut_tall_glyphs(mask, SCALES[0])
    assert np.array_equal(cut, mask)
