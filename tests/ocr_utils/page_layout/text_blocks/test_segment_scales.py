"""Два масштаба сегментации: крупный не видит корпусных строк, перескок через ряд отбрасывается."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.text_blocks import segment as seg
from ocr_utils.page_layout.text_blocks import zones as zn
from ocr_utils.page_layout.text_blocks.columns import separators_for_segmentation
from tests.ocr_utils.page_layout.text_blocks.synthetic import column_page
from tests.ocr_utils.page_layout.text_blocks.test_zones import _piece

# Шаг строк и высота строки корпуса (пиксели рабочей копии), как на корпусе пака-1.
PITCH = 23.0
HEIGHT = 16.0


def _segment(xs: np.ndarray, ys: np.ndarray, height: float = HEIGHT, scale: str = "корпус") -> seg.Segment:
    """Строка с осью по точкам ``xs, ys``; бокс — охват оси плюс полвысоты сверху и снизу."""
    return seg.Segment(
        x0=int(xs.min()),
        y0=int(ys.min() - height / 2),
        x1=int(xs.max()),
        y1=int(ys.max() + height / 2),
        height=height,
        xs=xs.astype(np.float64),
        ys=ys.astype(np.float64),
        weights=np.ones(xs.size),
        scale=scale,
    )


def _row(y: float, x0: float = 0.0, x1: float = 400.0) -> seg.Segment:
    """Прямая корпусная строка на уровне ``y``."""
    xs = np.arange(x0, x1 + 1.0, 2.0)
    return _segment(xs, np.full(xs.size, y))


def _diagonal() -> seg.Segment:
    """Крупная «строка» с перескоком: левая часть на ряду y = 100, правая — на ряду ниже, между ними косой спуск."""
    xs = np.arange(0.0, 401.0, 2.0)
    ys = np.interp(xs, [0.0, 160.0, 240.0, 400.0], [100.0, 100.0, 100.0 + PITCH, 100.0 + PITCH])
    return _segment(xs, ys, height=HEIGHT, scale="крупный")


def test_diagonal_is_covered_by_the_axes_of_two_rows():
    """Диагональ, лежащая большей частью на осях двух рядов, накрыта ими — хотя по боксу нет."""
    body = [_row(100.0), _row(100.0 + PITCH)]
    assert seg._covered(_diagonal(), body)


def test_headline_between_rows_is_not_covered():
    """Строка посередине между рядами (свой заголовок) корпусом не накрыта."""
    headline = _segment(np.arange(0.0, 401.0, 2.0), np.full(201, 60.0), height=30.0, scale="крупный")
    assert not seg._covered(headline, [_row(100.0), _row(100.0 + PITCH)])


def test_jump_through_a_row_crosses_it():
    """Ось, спустившаяся сквозь корпусный ряд, его скрещивает; ось вдоль ряда — нет."""
    xs = np.arange(0.0, 401.0, 2.0)
    through = _segment(xs, np.interp(xs, [0.0, 400.0], [100.0, 100.0 + 2 * PITCH]), scale="крупный")
    middle = _row(100.0 + PITCH)
    assert seg._crosses_body(through, [middle])
    assert not seg._crosses_body(_row(100.0 + PITCH, 50.0, 300.0), [middle])


def test_text_rows_need_enough_rows_and_a_neighbour():
    """Строки абзаца — с рядом сверху или снизу; мало строк — абзацев нет; одиночное слово не в счёт."""
    assert seg._text_rows([_row(100.0)]) == []
    rows = [_row(40.0 + PITCH * index) for index in range(seg.BODY_MIN_SEGMENTS)]
    lonely = _row(40.0 + PITCH * 20, 100.0, 160.0)  # слово заголовка кеглем корпуса далеко от абзаца
    found = seg._text_rows(rows + [lonely])
    assert len(found) == len(rows)
    assert lonely not in found


def test_claimed_mask_covers_the_band_of_each_row():
    """Маска забранной краски: полоса вокруг оси каждого ряда, между рядами пусто."""
    shape = (400, 420)
    assert seg._claimed_mask([], shape) is None
    rows = [_row(40.0 + PITCH * index) for index in range(seg.BODY_MIN_SEGMENTS)]
    claimed = seg._claimed_mask(rows, shape)
    assert claimed is not None
    assert claimed[40, 200] and claimed[40 + int(HEIGHT * 0.4), 200]
    # Между рядами (полшага от оси при полосе в полвысоты) краска не забрана.
    assert not claimed[40 + int(PITCH / 2), 200]


def test_claimed_components_are_hidden_from_the_large_scale():
    """Компонента в забранной полосе выбрасывается из маски, компонента вне неё остаётся."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # буква корпусной строки
    mask[60:80, 10:40] = 1  # буква заголовка
    claimed = np.zeros((100, 100), dtype=bool)
    claimed[8:22, :] = True
    out = seg._without_claimed(mask, claimed)
    assert out[15, 15] == 0
    assert out[70, 20] == 1


def test_large_scale_finds_nothing_in_plain_body_text():
    """На сплошном корпусе крупный масштаб не даёт ни одной строки: высокие буквы — забранная краска."""
    gray = column_page(columns=2)
    segments, _ = seg.segments_of(gray, separators_for_segmentation([]))
    assert segments
    assert not [segment for segment in segments if segment.scale == "крупный"]


def test_link_verdicts_name_the_reason():
    """Встреча зон двух слов одной строки принимается; куски друг над другом отбиваются с причиной."""
    left = _piece(0.0, 100.0, letters=5)
    right = _piece(80.0, 100.0, letters=5)
    pieces = [left, right]
    zones = zn.zones_of(pieces, PITCH, False)
    ink = np.zeros((10, 10), dtype=bool)
    verdicts = zn.link_verdicts(zones, pieces, seg.SCALES[0], [], None, ink, 2.0)
    assert any(verdict is zn.LinkVerdict.ACCEPTED for _, verdict in verdicts)
    assert zn._verdict_of(left, _piece(10.0, 104.0, letters=5), seg.SCALES[0], [], None, ink, 2.0, seg._crosses) is (
        zn.LinkVerdict.OVERLAP
    )
