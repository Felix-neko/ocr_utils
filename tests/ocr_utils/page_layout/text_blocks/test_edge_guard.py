"""Защита выровненных сторон блоков (``edge_guard``): выступы, пометка недостоверных участков, голосование CRAFT + pero, зона и фильтр, второй проход в ``analyse_gray``."""

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks.alignment import AlignKind, Alignment, SideStats
from ocr_utils.page_layout.text_blocks.blocks import BlockEnvelope, Row, TextBlock
from ocr_utils.page_layout.text_blocks.edge_guard import (
    BumpKind,
    GlyphEngine,
    component_scores,
    is_junk,
    keep_gray,
    mark_unreliable,
    page_bumps,
    side_zones,
)
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.page import analyse_gray
from ocr_utils.page_layout.text_blocks.smooth_envelope import SmoothEnvelope
from tests.ocr_utils.page_layout.orientation.synthetic import GLYPH_H, INK, LINE_STEP, MARGIN
from tests.ocr_utils.page_layout.text_blocks.synthetic import column_page

DPI = 150.0
PITCH = 24.0  # шаг строк, пиксели рабочей копии (~4 мм)
ROWS = 20
LEFT_X, RIGHT_X = 100.0, 700.0
ONE_MM = DPI / 25.4


def _block(right_bumps: dict[int, float]) -> TextBlock:
    """Блок с прямыми сторонами; у правой стороны выступы наружу ``{строка: пикселей}``."""
    rows = tuple(Row(y=50 + i * PITCH, height=18.0, x0=LEFT_X, x1=RIGHT_X, axes=()) for i in range(ROWS))
    ys = np.arange(40.0, 50 + ROWS * PITCH, 1.0)
    right_x = np.full_like(ys, RIGHT_X)
    for row, shift in right_bumps.items():
        right_x[np.abs(ys - (50 + row * PITCH)) <= 9] += shift
    left = np.column_stack([np.full_like(ys, LEFT_X), ys])
    right = np.column_stack([right_x, ys])
    envelope = BlockEnvelope(left, right, left[:2], right[:2], np.vstack([left, right[::-1]]), 2.5)
    return TextBlock(0, 0, (0, 800), rows, PITCH, DPI, envelope, envelope)


def _alignment(kind: AlignKind = AlignKind.BOTH) -> Alignment:
    stats = SideStats(0.1, 0.95, 0, 0.0, 0.1, 0.1, True)
    return Alignment(kind, stats, stats)


def test_bump_and_step_on_aligned_side_straight_side_clean():
    """Выступ одной строки на 1.2 мм — пупырышек, три строки подряд — ступенька; прямая левая сторона чиста."""
    block = _block({5: 1.2 * ONE_MM, 12: 1.0 * ONE_MM, 13: 1.0 * ONE_MM, 14: 1.0 * ONE_MM})
    bumps = page_bumps((block,), (_alignment(),))
    assert [(b.side, b.kind, b.rows) for b in bumps] == [("right", BumpKind.BUMP, 1), ("right", BumpKind.STEP, 3)]
    assert abs(bumps[0].mm - 1.2) < 0.15


def test_small_bump_ragged_side_and_short_block_ignored():
    """Выступ 0.3 мм ниже порога; невыровненная сторона и короткий блок не смотрятся."""
    assert page_bumps((_block({5: 0.3 * ONE_MM}),), (_alignment(),)) == []
    assert page_bumps((_block({5: 2.0 * ONE_MM}),), (_alignment(AlignKind.LEFT),)) == []
    short = _block({2: 2.0 * ONE_MM})
    short = TextBlock(0, 0, (0, 800), short.rows[:4], PITCH, DPI, short.envelope, short.envelope)
    assert page_bumps((short,), (_alignment(),)) == []


def test_mark_unreliable_adds_span():
    """Пометка переводит простую огибающую в гладкую и добавляет участок выступа."""
    block = _block({5: 1.5 * ONE_MM})
    bumps = page_bumps((block,), (_alignment(),))
    envelope = mark_unreliable((block,), bumps)[0].envelope
    assert isinstance(envelope, SmoothEnvelope)
    assert envelope.unreliable_right == ((bumps[0].y0, bumps[0].y1),) and envelope.unreliable_left == ()


def test_zone_starts_outside_the_side():
    """Зона фильтра начинается снаружи от стороны: краска внутри колонки у стороны не проверяется."""
    block = _block({5: 1.5 * ONE_MM})
    zone = side_zones((block,), page_bumps((block,), (_alignment(),)), DPI, (1400, 1800))
    row, side = int((50 + 5 * PITCH) * 2), int(RIGHT_X * 2)
    assert not zone[row, side - 4] and not zone[row, side + 2] and zone[row, side + 30]


def test_vote_any_engine_and_pero_abstains_off_row():
    """Голосование «хотя бы один»; pero воздерживается вне полосы строки."""
    assert not is_junk({GlyphEngine.CRAFT: 0.9, GlyphEngine.PERO: 0.9})
    assert is_junk({GlyphEngine.CRAFT: 0.9, GlyphEngine.PERO: 0.1})
    ink = np.zeros((200, 100), dtype=np.uint8)
    ink[45:55, 10:16] = 1
    ink[150:160, 10:16] = 1
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    maps = {GlyphEngine.PERO: np.zeros(ink.shape, dtype=np.float32)}
    assert component_scores(labels, stats, int(labels[50, 12]), maps, (50.0, 10.0)) == {GlyphEngine.PERO: 0.0}
    assert component_scores(labels, stats, int(labels[155, 12]), maps, (50.0, 10.0)) == {}


def test_keep_gray_drops_only_junk_in_zone_not_bars():
    """В зоне закрашивается пятно без символа на карте; буква, пятно вне зоны и тире остаются."""
    gray = np.full((100, 300), 255, dtype=np.uint8)
    gray[40:50, 20:30] = 0  # буква в зоне
    gray[40:46, 60:66] = 0  # сор в зоне
    gray[40:46, 200:206] = 0  # сор вне зоны
    gray[70:74, 40:60] = 0  # тире в зоне
    craft = np.zeros(gray.shape, dtype=np.float32)
    craft[40:50, 20:30] = 0.9
    zone = np.zeros(gray.shape, dtype=bool)
    zone[:, :100] = True
    out, dropped = keep_gray(gray, zone, {GlyphEngine.CRAFT: craft}, lambda y: (45.0, 10.0))
    assert dropped == [(60, 40, 66, 46)]
    assert (out[40:50, 20:30] == 0).all() and (out[40:46, 200:206] == 0).all() and (out[70:74, 40:60] == 0).all()


def _page_with_blob() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Колонка с выключкой по формату и клякса, прилипшая к концу десятой строки за правой стороной."""
    page = column_page(columns=1, justify="both")
    right = int(np.nonzero((page[MARGIN : MARGIN + GLYPH_H] < 128).any(axis=0))[0].max())
    y = MARGIN + 10 * LINE_STEP
    box = (right + 3, y, right + 24, y + GLYPH_H)
    page[box[1] : box[3], box[0] : box[2]] = INK
    return page, box


def test_clean_column_not_anomalous():
    """Чистая колонка: отчёт защиты есть, выступов нет, второго прохода нет."""
    analysis = analyse_gray(column_page(columns=1, justify="both"), InkEngine())
    assert analysis.edge_guard is not None and not analysis.edge_guard.anomalous


def test_blob_marked_without_maps_and_removed_with_maps():
    """Клякса у конца строки: без карт — выступ помечен недостоверным; с картами, где клякса не символ, — выброшена
    вторым проходом, выступов по итогу нет."""
    page, box = _page_with_blob()
    first = analyse_gray(page, InkEngine())
    assert first.edge_guard.anomalous and first.edge_guard.bumps_final
    assert any(block.envelope.unreliable_right for block in first.blocks)
    craft = np.ones(page.shape, dtype=np.float32)
    craft[box[1] : box[3], box[0] : box[2]] = 0.0
    maps = {GlyphEngine.CRAFT: craft, GlyphEngine.PERO: np.ones(page.shape, dtype=np.float32)}
    second = analyse_gray(page, InkEngine(), glyph_maps=maps)
    assert second.edge_guard.second_pass and second.edge_guard.dropped
    assert not second.edge_guard.bumps_final
