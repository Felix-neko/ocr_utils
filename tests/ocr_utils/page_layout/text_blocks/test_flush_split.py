"""Тесты разделения блоков по смене выключки (``flush_split``): подпись авторов вправо и «г. Город» влево под абзацем по формату."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ocr_utils.page_layout.text_blocks import flush_split as F
from ocr_utils.page_layout.text_blocks.blocks import Row

DPI = 150.0
# Колонка шириной ~100 мм рабочей копии 150 dpi: левый край 36 px, правый 632 px (как ХиП 1991/01 IMG_1008, с. 59).
LEFT, RIGHT = 36.0, 632.0


def _row(y: float, x0: float, x1: float, glyph_h: float = 10.0, weight: float = 2.7, axes: tuple = ()) -> Row:
    """Ряд с заданными краями, высотой глифа и жирностью (оси — по желанию, для резки далёких кусков)."""
    return Row(y=y, height=13.0, x0=x0, x1=x1, axes=axes, glyph_h=glyph_h, glyph_w=8.0, weight=weight)


def _axis(x0: float, x1: float) -> SimpleNamespace:
    """Заглушка оси: резке рядов нужны только края по x."""
    return SimpleNamespace(x0=x0, x1=x1)


def _paragraph(y0: float = 60.0, rows: int = 6, pitch: float = 22.0) -> list[Row]:
    """Абзац по формату: первая строка с отступом, последняя — короткая концовка у левого края."""
    out = [_row(y0, LEFT + 25.0, RIGHT)]
    out += [_row(y0 + pitch * i, LEFT, RIGHT) for i in range(1, rows - 1)]
    out.append(_row(y0 + pitch * (rows - 1), LEFT, 122.0))
    return out


def _signature(y0: float) -> list[Row]:
    """Подпись авторов вправо: фамилия, регалии, фамилия, регалия — мельче и светлее корпуса."""
    spec = ((536.0, 2.5, 0.0), (369.0, 1.95, 8.5), (448.0, 1.92, 8.5), (526.0, 2.44, 12.0), (528.0, 1.9, 8.5))
    return [_row(y0 + 18.0 * i, x0, RIGHT, glyph_h=gh, weight=w) for i, (x0, w, gh) in enumerate(spec)]


def test_default_mode_is_align_right_gap() -> None:
    """Способ по умолчанию — объединение ``ALIGN_RIGHT`` и ``ALIGN_GAP``."""
    assert F.DEFAULT_FLUSH_SPLIT is F.FlushSplit.ALIGN_RIGHT_GAP


def test_paragraph_then_signature_is_side_change() -> None:
    """Абзац по формату и подпись вправо под ним — смена стороны выключки; склейка запрещена при любом способе, кроме OFF."""
    upper, lower = _paragraph(), _signature(60.0 + 22.0 * 5 + 27.0)
    assert F.side_change(upper, lower, DPI)
    assert F.flush_change(upper, lower, 27.0, 22.0, DPI, F.FlushSplit.ALIGN_GAP)
    assert F.flush_change(upper, lower, 27.0, 22.0, DPI, F.FlushSplit.ALIGN)
    assert not F.flush_change(upper, lower, 27.0, 22.0, DPI, F.FlushSplit.OFF)


def test_align_gap_needs_gap_or_type_change() -> None:
    """У ``ALIGN_GAP`` одной смены стороны мало: при обычном шаге и том же наборе склейка разрешена."""
    upper = _paragraph()
    lower = [_row(200.0, 500.0, RIGHT)]  # строка вправо того же набора на обычном шаге
    assert F.side_change(upper, lower, DPI)
    assert not F.flush_change(upper, lower, 22.0, 22.0, DPI, F.FlushSplit.ALIGN_GAP)
    # Увеличенный промежуток (1.23 шага) — склейка запрещена.
    assert F.flush_change(upper, lower, 27.0, 22.0, DPI, F.FlushSplit.ALIGN_GAP)
    # Другой набор (глиф мельче в 1.18 раза) — тоже.
    lighter = [_row(200.0, 500.0, RIGHT, glyph_h=8.5)]
    assert F.flush_change(upper, lighter, 22.0, 22.0, DPI, F.FlushSplit.ALIGN_GAP)


def test_surname_stays_with_its_titles() -> None:
    """Фамилия над регалиями (обе вправо) — не смена выключки: подпись не рвётся на фамилию и регалии."""
    signature = _signature(352.0)
    assert not F.side_change(signature[:1], signature[1:], DPI)


def test_signature_then_city_is_side_change() -> None:
    """Подпись вправо и «г. Ереван» влево под ней — смена стороны."""
    signature = _signature(352.0)
    city = [_row(445.0, LEFT, 102.0)]
    assert F.side_change(signature, city, DPI)


@pytest.mark.parametrize(
    "lower",
    [
        # Следующий абзац с отступом идёт до правого края.
        [_row(214.0, LEFT + 25.0, RIGHT), _row(236.0, LEFT, RIGHT)],
        # Пункты списка у левого края.
        [_row(214.0, LEFT, 400.0), _row(236.0, LEFT, 380.0)],
        # Строка по центру.
        [_row(214.0, 250.0, 420.0)],
    ],
)
def test_ordinary_continuations_are_not_side_change(lower: list[Row]) -> None:
    """Новый абзац, список, центрированная строка под концовкой абзаца — не смена выключки."""
    assert not F.side_change(_paragraph(), lower, DPI)


def test_far_pieces_split_when_neighbours_reach_one_side() -> None:
    """Ряд «г. Ереван … юрисконсульт» режется: строка над ним дотягивается только до правого куска."""
    joined = _row(426.0, LEFT, RIGHT, axes=(_axis(36.0, 102.0), _axis(529.0, 631.0)))
    above = _row(405.0, 526.0, RIGHT, axes=(_axis(527.0, 631.0),))
    left, right, gap = F._pieces(joined)
    assert gap > 400.0
    assert F._one_sided(joined, left, right, [above, joined])


def test_far_pieces_kept_when_neighbour_bridges() -> None:
    """Строка таблицы или колонки, над которой есть строка через оба куска, не режется."""
    joined = _row(426.0, LEFT, RIGHT, axes=(_axis(36.0, 102.0), _axis(529.0, 631.0)))
    above = _row(404.0, LEFT, RIGHT, axes=(_axis(36.0, 631.0),))
    left, right, _ = F._pieces(joined)
    assert not F._one_sided(joined, left, right, [above, joined])


def test_far_pair_off_mode() -> None:
    """``far_pair`` у способа ``OFF`` и без краски молчит, у основного — узнаёт далёкие куски с соседом над одним; отточие
    в просвете (строка оглавления «Название ...... 28») пустоту отменяет."""
    import numpy as np

    city = _row(428.0, LEFT, 102.0, axes=(_axis(36.0, 102.0),))
    tail = _row(425.0, 528.0, RIGHT, axes=(_axis(529.0, 631.0),))
    above = _row(405.0, 526.0, RIGHT, axes=(_axis(527.0, 631.0),))
    rows = [above, tail, city]
    blank = np.zeros((1000, 1400), np.uint8)  # рендер 300 dpi: вдвое больше рабочей копии
    assert F.far_pair(city, tail, rows, DPI, blank)
    assert not F.far_pair(city, tail, rows, DPI, None)
    assert not F.far_pair(city, tail, rows, DPI, blank, F.FlushSplit.OFF)
    dots = blank.copy()
    dots[850:858, 210:1050:12] = 1  # точки отточия через 6 px рабочей копии в полосе ряда
    assert F.far_pair(city, tail, rows, DPI, dots) is False


def test_signature_page_blocks() -> None:
    """Синтетика вёрстки ХиП 1991/01 IMG_1008: абзац по формату, подпись вправо и город влево — разные блоки; у ``OFF``
    подпись и город срастаются с абзацем, как было."""
    from ocr_utils.page_layout.text_blocks.alignment import AlignKind
    from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
    from ocr_utils.page_layout.text_blocks.page import analyse_gray
    from tests.ocr_utils.page_layout.text_blocks.synthetic import signature_page

    page = signature_page()
    analysis = analyse_gray(page, InkEngine())
    kinds = [(len(block.rows), al.kind) for block, al in zip(analysis.blocks, analysis.alignments)]
    assert (10, AlignKind.BOTH) in kinds
    assert (5, AlignKind.RIGHT) in kinds
    assert len(analysis.blocks) == 4
    old = F.DEFAULT_FLUSH_SPLIT
    try:
        F.DEFAULT_FLUSH_SPLIT = F.FlushSplit.OFF
        legacy = analyse_gray(page, InkEngine())
    finally:
        F.DEFAULT_FLUSH_SPLIT = old
    assert len(legacy.blocks) < len(analysis.blocks)


def test_dialogue_indent_is_not_flush_right() -> None:
    """Первая строка реплики с тире и отступом 8.5 мм до правого края — не строка «вправо» (ХиП 1991/01 IMG_0992)."""
    upper = [_row(60.0, LEFT + 25.0, RIGHT), _row(82.0, LEFT, 300.0)]
    dialogue = [_row(104.0, LEFT + 50.0, RIGHT)]
    assert F.row_side(dialogue[0], LEFT, RIGHT, DPI) is None
    assert not F.side_change(upper, dialogue, DPI)


def test_toc_page_numbers_are_not_flush_right_group() -> None:
    """Колонка номеров страниц оглавления (цифры по 4 мм у правого края) — не подпись: смены выключки нет."""
    toc = [_row(60.0 + 22.0 * i, LEFT, 560.0) for i in range(6)]
    numbers = [_row(200.0 + 44.0 * i, 610.0, RIGHT) for i in range(4)]  # 22 px ≈ 3.7 мм при 150 dpi
    assert not F.side_change(toc, numbers, DPI)


def _flat(row: Row) -> Row:
    """Ряд с плоской осью-заглушкой (ордината ряда по всей длине) — для мер шага и разрыва."""
    axis = SimpleNamespace(x0=row.x0, x1=row.x1, y_at=lambda x, y=row.y: y)
    return Row(
        y=row.y,
        height=row.height,
        x0=row.x0,
        x1=row.x1,
        axes=(axis,),
        glyph_h=row.glyph_h,
        glyph_w=8.0,
        weight=row.weight,
    )


def test_split_by_flush_cuts_signature_reached_by_paragraph() -> None:
    """Подпись, перекрытая по x с концовкой абзаца (одна группа с самого начала), отрезается внутри группы; абзац цел."""
    paragraph = _paragraph()
    paragraph[-1] = _row(paragraph[-1].y, LEFT + 100.0, 560.0)  # концовка с середины строки, до x 560
    rows = [_flat(row) for row in paragraph + _signature(paragraph[-1].y + 27.0)]
    parts = F.split_by_flush(rows, DPI)
    assert [len(part) for part in parts] == [6, 5]
    assert F.split_by_flush(rows, DPI, F.FlushSplit.OFF) == [rows]


def test_split_by_flush_keeps_plain_paragraphs() -> None:
    """Абзацы с отступами и концовками и пункт списка — без разрезов."""
    rows = [_flat(row) for row in _paragraph() + _paragraph(y0=60.0 + 22.0 * 6, rows=5)]
    assert F.split_by_flush(rows, DPI) == [rows]


def test_two_authors_of_one_signature_are_not_side_change() -> None:
    """Фамилия и регалия первого автора над фамилией и регалией второго — одна подпись (ХиП 1991/01 IMG_1042_R)."""
    first = [_row(352.0, 536.0, RIGHT, weight=2.6), _row(370.0, 360.0, RIGHT, weight=1.9)]
    second = [_row(392.0, 520.0, RIGHT, weight=2.6), _row(410.0, 590.0, RIGHT, weight=1.9)]
    assert not F.side_change(first, second, DPI)


# --- Способ ALIGN_RIGHT: смену выключки решает первая строка, без подкреплений -----------------------------------


def _staggered_signature(y0: float) -> list[Row]:
    """Подпись с регалиями лесенкой: фамилия жирным вправо, самая длинная регалия задаёт левый край охвата подписи,
    строки под ней отступают на 3–6 мм (ХиП 1991/01 IMG_0982_L «М. ГАЙНЕТДИНОВ»)."""
    spec = ((470.0, 2.6), (430.0, 1.9), (300.0, 1.9), (340.0, 1.9))
    return [_flat(_row(y0 + 20.0 * i, x0, RIGHT, glyph_h=8.5, weight=w)) for i, (x0, w) in enumerate(spec)]


def test_align_right_single_right_line_needs_no_reinforcement() -> None:
    """Одна строка вправо того же набора на обычном шаге под абзацем — новый блок у ``ALIGN_RIGHT``, не у ``ALIGN_GAP``."""
    upper = _paragraph()
    lower = [_row(192.0, 500.0, RIGHT)]
    assert F.flush_change(upper, lower, 22.0, 22.0, DPI, F.FlushSplit.ALIGN_RIGHT)
    assert not F.flush_change(upper, lower, 22.0, 22.0, DPI, F.FlushSplit.ALIGN_GAP)


def test_align_right_cuts_signature_with_small_gap() -> None:
    """Подпись под абзацем с промежутком 1.12 шага и почти тем же набором (ХиП 1991/04 IMG_0198 L) режется."""
    paragraph = [_flat(row) for row in _paragraph()]
    signature = [_flat(_row(paragraph[-1].y + 24.6 + 18.0 * i, x0, RIGHT, glyph_h=9.5, weight=w))
                 for i, (x0, w) in enumerate(((520.0, 3.0), (380.0, 2.6), (500.0, 3.0), (400.0, 2.6)))]  # fmt: skip
    rows = paragraph + signature
    assert [len(part) for part in F.split_by_flush(rows, DPI, F.FlushSplit.ALIGN_RIGHT)] == [6, 4]
    assert F.split_by_flush(rows, DPI, F.FlushSplit.ALIGN_GAP) == [rows]


def test_align_right_keeps_paragraph_tail() -> None:
    """Концовка абзаца у левого края и абзацный отступ следующего абзаца — без разрезов."""
    rows = [_flat(row) for row in _paragraph() + _paragraph(y0=60.0 + 22.0 * 6, rows=4)]
    assert F.split_by_flush(rows, DPI, F.FlushSplit.ALIGN_RIGHT) == [rows]


def test_align_right_staggered_regalia_stay_in_signature() -> None:
    """Регалии лесенкой под фамилией — одна подпись: самая длинная регалия у левого края охвата подписи — не абзац."""
    signature = _staggered_signature(200.0)
    assert F.split_by_flush(signature, DPI, F.FlushSplit.ALIGN_RIGHT) == [signature]
    # Склейка фамилии с регалиями (разный набор) разрешена.
    assert F.signature_continues(signature[:1], signature[1:], DPI, F.FlushSplit.ALIGN_RIGHT)
    assert not F.signature_continues(signature[:1], signature[1:], DPI, F.FlushSplit.ALIGN_GAP)


def test_align_right_short_quote_and_its_author() -> None:
    """Цитата в две строки (полная и концовка у левого края), автор вправо, следующая цитата — три блока
    (ХиП 1991/05 IMG_0332_R)."""
    quote = [_flat(_row(60.0, LEFT, RIGHT)), _flat(_row(82.0, LEFT, 150.0))]
    author = [_flat(_row(104.0, 540.0, RIGHT, weight=2.9))]
    nxt = [_flat(_row(134.0, LEFT, RIGHT)), _flat(_row(156.0, LEFT, 140.0))]
    rows = quote + author + nxt
    assert [len(part) for part in F.split_by_flush(rows, DPI, F.FlushSplit.ALIGN_RIGHT)] == [2, 1, 2]


def test_align_right_city_under_signature() -> None:
    """«г. Город» влево под подписью вправо — отдельный блок без подкреплений."""
    signature = _signature(352.0)
    city = [_row(352.0 + 18.0 * 5, LEFT, 102.0, glyph_h=8.5, weight=1.9)]
    assert F.flush_change(signature, city, 18.0, 18.0, DPI, F.FlushSplit.ALIGN_RIGHT)


def test_align_right_line_piece_is_not_right_line() -> None:
    """Правый кусок строки, разорванной широким пробелом (выше остатка на 0.8 высоты на изогнутой странице), — не
    строка вправо: абзац не режется (ХиП 1991/03 IMG_0082_L «…(ч. 1 ст. 214»)."""
    paragraph = [_flat(row) for row in _paragraph()[:-1]]
    rest = _flat(_row(paragraph[-1].y + 22.0, LEFT, 520.0))
    piece = _flat(_row(rest.y - 10.0, 575.0, RIGHT))
    tail = _flat(_row(rest.y + 22.0, LEFT, 300.0))
    # Кусок в той же группе.
    rows = paragraph + [piece, rest, tail]
    assert F.split_by_flush(rows, DPI, F.FlushSplit.ALIGN_RIGHT) == [rows]
    # Кусок отдельной группой: склейку разрешает поиск по рядам страницы.
    assert not F.flush_change(paragraph, [piece], 22.0, 22.0, DPI, F.FlushSplit.ALIGN_RIGHT, page_rows=rows)
    assert F.flush_change(paragraph, [piece], 22.0, 22.0, DPI, F.FlushSplit.ALIGN_RIGHT)


# --- Способ ALIGN_RIGHT_GAP: разрез любого из двух способов, но не внутри подписи -----------------------------------


def test_align_right_gap_cuts_where_either_mode_cuts() -> None:
    """Строку вправо без подкреплений режет ``ALIGN_RIGHT``, подпись под абзацем с промежутком — оба; у объединения —
    оба разреза."""
    upper = _paragraph()
    assert F.flush_change(upper, [_row(192.0, 500.0, RIGHT)], 22.0, 22.0, DPI, F.FlushSplit.ALIGN_RIGHT_GAP)
    signature = _signature(60.0 + 22.0 * 5 + 27.0)
    assert F.flush_change(upper, signature, 27.0, 22.0, DPI, F.FlushSplit.ALIGN_RIGHT_GAP)
    paragraph = [_flat(row) for row in _paragraph()]
    lead = [_flat(_row(paragraph[-1].y + 24.6 + 18.0 * i, x0, RIGHT, glyph_h=9.5, weight=w))
            for i, (x0, w) in enumerate(((520.0, 3.0), (380.0, 2.6), (500.0, 3.0), (400.0, 2.6)))]  # fmt: skip
    rows = paragraph + lead
    assert [len(part) for part in F.split_by_flush(rows, DPI, F.FlushSplit.ALIGN_RIGHT_GAP)] == [6, 4]


def test_align_right_gap_keeps_signature_whole() -> None:
    """Регалии лесенкой под жирной фамилией не отрываются: разрез ``ALIGN_GAP`` по смене набора внутри подписи не
    ставится, склейка разрешена."""
    signature = _staggered_signature(200.0)
    assert F.split_by_flush(signature, DPI, F.FlushSplit.ALIGN_RIGHT_GAP) == [signature]
    assert F.signature_continues(signature[:1], signature[1:], DPI, F.FlushSplit.ALIGN_RIGHT_GAP)
    assert not F.flush_change(signature[:1], signature[1:], 20.0, 20.0, DPI, F.FlushSplit.ALIGN_RIGHT_GAP)


def _centred_signature(y0: float) -> list[Row]:
    """Подпись по центру своей оси в правой части колонки: фамилия и регалии с общим центром x ≈ 500, до правого края
    колонки 30–80 px (1973/10 IMG_0173_2R «Ю. ДОЛГОРУКОВ,»)."""
    spec = ((420.0, 580.0), (385.0, 615.0), (415.0, 585.0), (375.0, 625.0))
    return [_flat(_row(y0 + 20.0 * i, x0, x1, glyph_h=9.5, weight=2.4)) for i, (x0, x1) in enumerate(spec)]


def test_centred_signature_under_paragraph_is_own_block() -> None:
    """Абзац с короткой концовкой и подпись на общем центре на обычном шаге — два блока; подпись цела."""
    paragraph = [_flat(row) for row in _paragraph()]
    signature = _centred_signature(paragraph[-1].y + 22.0)
    parts = F.split_by_flush(paragraph + signature, DPI)
    assert [len(part) for part in parts] == [len(paragraph), len(signature)]
    assert F.change_kind(paragraph, signature, DPI) is F.FlushChange.TO_CENTRE
    assert F.flush_change(paragraph, signature, 22.0, 22.0, DPI)


def test_right_flush_rows_are_not_centred() -> None:
    """Строки у правого края (подпись вправо) — не группа на общем центре: правые края совпадают."""
    signature = _staggered_signature(200.0)[:2]
    cx0, cx1 = LEFT, RIGHT
    assert not F._centred(signature, cx0, cx1, DPI)
