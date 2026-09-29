"""Тесты защиты сторон блоков на синтетике: выступы выровненных сторон, зона и фильтр по карте CRAFT, пометка недостоверных участков."""

import numpy as np

from ocr_utils.page_layout.text_blocks.alignment import AlignKind, Alignment, SideStats
from ocr_utils.page_layout.text_blocks.blocks import BlockEnvelope, Row, TextBlock
from ocr_utils.page_layout.text_blocks.page import PageAnalysis
from ocr_utils.page_layout.text_blocks.smooth_envelope import SmoothEnvelope
from research.edge_marks.guard.anomaly import BumpKind, page_bumps
from research.edge_marks.guard.craft_filter import keep_gray
from research.edge_marks.guard.guard import mark_unreliable

DPI = 150.0
PITCH = 24.0  # шаг строк, пиксели рабочей копии (~4 мм)
ROWS = 20
LEFT_X, RIGHT_X = 100.0, 700.0


def _block(right_bumps: dict[int, float]) -> TextBlock:
    """Блок с прямыми сторонами; у правой стороны выступы наружу ``{строка: пикселей}``."""
    rows = tuple(Row(y=50 + i * PITCH, height=18.0, x0=LEFT_X, x1=RIGHT_X, axes=()) for i in range(ROWS))
    ys = np.arange(40.0, 50 + ROWS * PITCH, 1.0)
    right_x = np.full_like(ys, RIGHT_X)
    for row, shift in right_bumps.items():
        centre = 50 + row * PITCH
        right_x[np.abs(ys - centre) <= 9] += shift
    left = np.column_stack([np.full_like(ys, LEFT_X), ys])
    right = np.column_stack([right_x, ys])
    polygon = np.vstack([left, right[::-1]])
    envelope = BlockEnvelope(left, right, left[:2], right[:2], polygon, 2.5)
    return TextBlock(0, 0, (0, 800), rows, PITCH, DPI, envelope, envelope)


def _analysis(block: TextBlock, kind: AlignKind = AlignKind.BOTH) -> PageAnalysis:
    stats = SideStats(0.1, 0.95, 0, 0.0, 0.1, 0.1, True)
    return PageAnalysis("p", 1, "nogeo", "ink", 800, 700, DPI, (), (block,), (Alignment(kind, stats, stats),), (), (),
                        0.0)  # fmt: skip


def test_bump_on_aligned_side_found_and_straight_side_clean():
    """Выступ одной строки на 1.2 мм — пупырышек, три строки подряд — ступенька; прямая левая сторона чиста."""
    one_mm = DPI / 25.4
    bumps = page_bumps(_analysis(_block({5: 1.2 * one_mm, 12: 1.0 * one_mm, 13: 1.0 * one_mm, 14: 1.0 * one_mm})))
    assert [(b.side, b.kind, b.rows) for b in bumps] == [("right", BumpKind.BUMP, 1), ("right", BumpKind.STEP, 3)]
    assert abs(bumps[0].mm - 1.2) < 0.15


def test_small_bump_and_ragged_side_ignored():
    """Выступ 0.3 мм (висячая точка) ниже порога; сторона, по которой блок не выровнен, не смотрится."""
    one_mm = DPI / 25.4
    assert page_bumps(_analysis(_block({5: 0.3 * one_mm}))) == []
    assert page_bumps(_analysis(_block({5: 2.0 * one_mm}), AlignKind.LEFT)) == []


def test_mark_unreliable_adds_span_to_stepped_envelope():
    """Пометка переводит простую огибающую в гладкую и добавляет участок выступа."""
    one_mm = DPI / 25.4
    analysis = _analysis(_block({5: 1.5 * one_mm}))
    bumps = page_bumps(analysis)
    marked = mark_unreliable(analysis, bumps)
    envelope = marked.blocks[0].envelope
    assert isinstance(envelope, SmoothEnvelope)
    assert envelope.unreliable_right == ((bumps[0].y0, bumps[0].y1),)
    assert envelope.unreliable_left == ()


def test_keep_gray_drops_only_low_map_components_in_zone():
    """В зоне закрашивается пятно без символа на карте; буква с картой и пятно вне зоны остаются."""
    gray = np.full((100, 300), 255, dtype=np.uint8)
    gray[40:50, 20:30] = 0  # буква в зоне, карта высокая
    gray[40:46, 60:66] = 0  # сор в зоне, карты нет
    gray[40:46, 200:206] = 0  # сор вне зоны
    gray[70:74, 40:60] = 0  # тире в зоне, карты нет — горизонтальный штрих не трогается
    region = np.zeros(gray.shape, dtype=np.float32)
    region[40:50, 20:30] = 0.9
    zone = np.zeros(gray.shape, dtype=bool)
    zone[:, :100] = True
    from research.edge_marks.guard.glyph_vote import Engine

    out, dropped = keep_gray(gray, zone, {Engine.CRAFT: region}, lambda y: (45.0, 10.0))
    assert dropped == [(60, 40, 66, 46)]
    assert (out[40:46, 60:66] == 255).all()
    assert (out[40:50, 20:30] == 0).all() and (out[40:46, 200:206] == 0).all()
    assert (out[70:74, 40:60] == 0).all()


def test_side_zone_starts_outside_the_side():
    """Зона фильтра начинается снаружи от стороны: компонента внутри колонки у стороны не проверяется."""
    from research.edge_marks.guard.craft_filter import side_zones

    one_mm = DPI / 25.4
    analysis = _analysis(_block({5: 1.5 * one_mm}))
    zone = side_zones(analysis, page_bumps(analysis), (1400, 1800))
    row = int((50 + 5 * PITCH) * 2)
    side = int(RIGHT_X * 2)
    assert not zone[row, side - 4] and not zone[row, side + 2]
    assert zone[row, side + 30]


def test_vote_junk_if_any_engine_says_so():
    """Голосование: сор, если хотя бы один детектор ниже порога; знак — только если все выше."""
    from research.edge_marks.guard.glyph_vote import Engine, is_junk

    assert not is_junk({Engine.CRAFT: 0.9, Engine.PERO: 0.9})
    assert is_junk({Engine.CRAFT: 0.9, Engine.PERO: 0.1})
    assert is_junk({Engine.CRAFT: 0.1, Engine.PERO: 0.9})


def test_pero_abstains_off_row():
    """pero голосует у строки (пустая карта — сор) и воздерживается под строкой (номер страницы)."""
    import cv2

    from research.edge_marks.guard.glyph_vote import Engine, component_scores

    ink = np.zeros((200, 100), dtype=np.uint8)
    ink[45:55, 10:16] = 1  # на строке (середина 50, x-высота 10)
    ink[150:160, 10:16] = 1  # под строкой
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    maps = {Engine.PERO: np.zeros(ink.shape, dtype=np.float32)}
    on_row = component_scores(labels, stats, int(labels[50, 12]), maps, (50.0, 10.0))
    off_row = component_scores(labels, stats, int(labels[155, 12]), maps, (50.0, 10.0))
    assert on_row == {Engine.PERO: 0.0} and off_row == {}
