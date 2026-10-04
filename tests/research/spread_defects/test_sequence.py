"""Тесты ряда номеров страниц: пропуск, повтор, ошибки чтения и мусор на краях выпуска не дают ложных событий."""

from research.spread_defects.sequence import Event, offset_candidates, sequence_events


def _candidates(pairs: list[tuple[int | None, int | None]]) -> list[set[int]]:
    """Кандидаты смещения по парам (левая, правая) номеров кадров выпуска."""
    return [offset_candidates(k, left, right) for k, (left, right) in enumerate(pairs)]


def _spreads(first: int, count: int) -> list[tuple[int, int]]:
    """Ряд разворотов подряд: (first, first+1), (first+2, first+3), …"""
    return [(first + 2 * n, first + 2 * n + 1) for n in range(count)]


def test_complete_issue_has_no_events():
    """Полный выпуск с обложкой без номеров и мусорными числами на краях — без событий."""
    pairs = [(None, None), (70, None)] + _spreads(2, 30) + [(244, 127), (None, None)]
    assert sequence_events(_candidates(pairs)) == []


def test_single_misread_side_is_tolerated():
    """Одна сторона прочитана неверно (14/73) — вторая и соседи вытягивают кадр, событий нет."""
    pairs = _spreads(2, 10)
    pairs[5] = (pairs[5][0], 73)
    pairs[7] = (3, pairs[7][1])
    assert sequence_events(_candidates(pairs)) == []


def test_missing_spread_is_gap():
    """Пропущен разворот 12–13 — событие GAP на один разворот между кадрами с 10 и 14."""
    pairs = _spreads(2, 5) + _spreads(14, 6)
    events = sequence_events(_candidates(pairs))
    assert len(events) == 1
    assert events[0].kind == Event.GAP
    assert (events[0].number_before, events[0].number_after, events[0].missing_spreads) == (10, 14, 1)


def test_duplicate_spread_is_repeat():
    """Разворот 10–11 снят дважды — событие REPEAT."""
    pairs = _spreads(2, 5) + _spreads(10, 6)
    events = sequence_events(_candidates(pairs))
    assert [e.kind for e in events] == [Event.REPEAT]
    assert events[0].number_after == 10


def test_unnumbered_frames_do_not_break_run():
    """Кадры без номера внутри выпуска (вкладка) не рвут ряд, если номера по обе стороны сходятся."""
    pairs = _spreads(2, 5)
    pairs[2] = (None, None)
    assert sequence_events(_candidates(pairs)) == []


def test_unnumbered_insert_is_extra_not_repeat():
    """Лишний кадр без номера между разворотами (вкладка) — EXTRA, а не повтор."""
    pairs = _spreads(2, 5) + [(None, None)] + _spreads(12, 5)
    events = sequence_events(_candidates(pairs))
    assert [e.kind for e in events] == [Event.EXTRA]
    assert events[0].missing_spreads == -1


def test_gap_hidden_behind_unnumbered_frame_is_not_reported():
    """Разворот без прочитанного номера на месте «пропуска» — не пропуск: кадров ровно столько, сколько разворотов."""
    pairs = _spreads(2, 5) + [(None, None)] + _spreads(14, 5)
    assert sequence_events(_candidates(pairs)) == []


def test_trailing_foreign_numbers_do_not_make_gap():
    """Задняя обложка и реклама со своей нумерацией в конце выпуска — FOREIGN, без пропусков и «назад»."""
    pairs = _spreads(2, 20) + [(2, 3), (4, 5)]
    events = sequence_events(_candidates(pairs))
    assert [e.kind for e in events] == [Event.FOREIGN]


def test_misread_segment_is_foreign_not_gap():
    """Отрезок, где «7» прочитана как «1» (70→10, 72→12, 74→14), — FOREIGN, ряд вокруг цел."""
    pairs = _spreads(60, 5) + [(10, 11), (12, 13), (14, 15)] + _spreads(76, 5)
    events = sequence_events(_candidates(pairs))
    assert [e.kind for e in events] == [Event.FOREIGN]
