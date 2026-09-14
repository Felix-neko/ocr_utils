"""Правила решения по выпуску: сильные и слабые признаки, пара 3+4, продолжение вперёд, виды."""

from __future__ import annotations

from ocr_utils.scan_markup.toc import KIND_CONTENTS, KIND_INDEX
from ocr_utils.scan_markup.toc.decide import Thresholds, decide_issue, in_window
from ocr_utils.scan_markup.toc.features import PageFeatures

N = 100  # полос в выпуске


def page(index: int, **kwargs) -> PageFeatures:
    return PageFeatures(f"1975/12/p{index:03d}.tif", index, N - 1 - index, **kwargs)


def kinds(decisions) -> dict[int, str | None]:
    return {d.order_index: d.kind for d in decisions}


def test_window_bounds() -> None:
    thr = Thresholds()
    assert in_window(0, N - 1, thr) and in_window(4, N - 5, thr) and not in_window(5, N - 6, thr)
    assert in_window(N - 12, 11, thr) and not in_window(N - 13, 12, thr)


def test_strong_signals_mark_alone() -> None:
    got = kinds(decide_issue([page(2, surya_toc_conf=0.75), page(3), page(4)], 5))
    assert got[2] == KIND_CONTENTS and got[4] is None


def test_pair_pulls_second_page_with_a_single_numbered_line() -> None:
    """1972/07: хвост оглавления на выходных данных — две строки с номером, surya молчит."""
    got = kinds(
        decide_issue(
            [
                page(2, kw_contents=True, num_tail_lines=7, num_tail_ratio=0.29),
                page(3, num_tail_lines=2, num_tail_ratio=0.06, kw_imprint=True),
            ],
            7,
        )
    )
    assert got == {2: KIND_CONTENTS, 3: KIND_CONTENTS}
    # И в обратную сторону: surya нашла только выходные данные.
    got = kinds(decide_issue([page(2, num_tail_lines=9, num_tail_ratio=0.27), page(3, surya_toc_conf=0.44)], 10))
    assert got == {2: KIND_CONTENTS, 3: KIND_CONTENTS}
    # Без единой строки с номером пара не срабатывает.
    got = kinds(decide_issue([page(2, kw_contents=True, num_tail_lines=7, num_tail_ratio=0.29), page(3)], 7))
    assert got == {2: KIND_CONTENTS, 3: None}
    # Вторая обложка с подписями к фотографиям даёт 2-3 строки с числом — это не пара.
    got = kinds(decide_issue([page(1, num_tail_lines=3, num_tail_ratio=0.14), page(2, surya_toc_conf=0.7)], 2))
    assert got == {1: None, 2: KIND_CONTENTS}
    # 1970/10: второй обложки нет, оглавление на полосах 1-2, а полоса 3 — статья с одним
    # случайным числом в конце строки.
    got = kinds(decide_issue([page(1, surya_toc_conf=0.3), page(2, surya_toc_conf=0.4), page(3, num_tail_lines=1)], 10))
    assert got == {1: KIND_CONTENTS, 2: KIND_CONTENTS, 3: None}
    # 1973/11: хвост в одну строку не прочитался, но на полосе есть «РЕДКОЛЛЕГИЯ».
    got = kinds(decide_issue([page(2, surya_toc_conf=0.7), page(3, kw_imprint=True), page(4, kw_imprint=True)], 11))
    assert got == {2: KIND_CONTENTS, 3: KIND_CONTENTS, 4: None}
    # В конце выпуска пары нет: реклама рядом с содержанием 1966 года остаётся рекламой.
    got = kinds(decide_issue([page(N - 3, surya_toc_conf=0.8), page(N - 2, num_tail_lines=2, kw_imprint=True)], 1))
    assert got == {N - 3: KIND_CONTENTS, N - 2: None}


def test_weak_page_alone_counts_unless_it_looks_like_a_table() -> None:
    got = kinds(decide_issue([page(N - 2, num_tail_lines=12, num_tail_ratio=0.3)], 5))
    assert got[N - 2] == KIND_CONTENTS
    got = kinds(decide_issue([page(N - 2, num_tail_lines=30, num_tail_ratio=0.6, surya_table_area=0.7)], 5))
    assert got[N - 2] is None, "таблица норм отгрузки с числами справа — не оглавление"
    got = kinds(decide_issue([page(N - 2, num_tail_lines=3, num_tail_ratio=0.2)], 5))
    assert got[N - 2] is None, "три строки с ценами на рекламной полосе (1975/03) — мало"
    got = kinds(decide_issue([page(N - 2, num_tail_lines=13, num_tail_ratio=0.46, surya_table_area=0.23)], 4))
    assert got[N - 2] is None, "таблица приложения (1972/04): блок Table хоть и небольшой"


def test_keyword_needs_numbered_lines() -> None:
    """«СОДЕРЖАНИЕ» капителью бывает подзаголовком в тексте; у оглавления есть строки с номером."""
    got = kinds(decide_issue([page(N - 5, kw_contents=True, num_tail_lines=0)], 4))
    assert got[N - 5] is None
    got = kinds(decide_issue([page(N - 5, kw_contents=True, num_tail_lines=2, num_tail_ratio=0.05)], 4))
    assert got[N - 5] == KIND_CONTENTS


def test_index_continues_forward_through_table_like_pages_but_not_backward() -> None:
    """1967/12: surya пометила 3 полосы указателя из 6, остальные назвала таблицей; 1970/12:
    таблица норм ПЕРЕД указателем продолжением не является."""
    pages = [
        page(N - 11, num_tail_lines=14, num_tail_ratio=0.32, surya_table_area=0.57),  # таблица до указателя
        page(N - 10, surya_toc_conf=0.79, kw_index=True, num_tail_lines=33, num_tail_ratio=0.57),
        page(N - 9, surya_toc_conf=0.74, num_tail_lines=33, num_tail_ratio=0.55),
        page(N - 8, surya_table_area=0.72, num_tail_lines=32, num_tail_ratio=0.59),
        page(N - 7, surya_toc_conf=0.74, num_tail_lines=49, num_tail_ratio=0.78),
        page(N - 6, surya_table_area=0.78, num_tail_lines=37, num_tail_ratio=0.63),
        page(N - 5, surya_table_area=0.81, num_tail_lines=50, num_tail_ratio=0.76),
        page(N - 4),  # некролог с фотографией
        page(N - 3, surya_toc_conf=1.0, kw_contents=True, num_tail_lines=16, num_tail_ratio=0.42),
        page(N - 2),
        page(N - 1),
    ]
    got = kinds(decide_issue(pages, 12))
    assert got[N - 11] is None
    assert all(got[i] == KIND_INDEX for i in range(N - 10, N - 4))
    assert got[N - 4] is None
    assert got[N - 3] == KIND_CONTENTS and got[N - 2] is None


def test_december_run_without_heading_is_index_and_splits_at_contents() -> None:
    """1968/12: указатель и содержание подряд, заголовок указателя tesseract не прочёл."""
    pages = [
        page(N - 7, surya_toc_conf=0.75, num_tail_lines=40, num_tail_ratio=0.7),
        page(N - 6, surya_toc_conf=0.72, num_tail_lines=40, num_tail_ratio=0.7),
        page(N - 5, surya_toc_conf=0.77, num_tail_lines=40, num_tail_ratio=0.7),
        page(N - 4, surya_toc_conf=0.75, kw_contents=True, num_tail_lines=12, num_tail_ratio=0.4),
        page(N - 3, num_tail_lines=6, num_tail_ratio=0.25),
    ]
    got = kinds(decide_issue(pages, 12))
    assert got == {N - 7: KIND_INDEX, N - 6: KIND_INDEX, N - 5: KIND_INDEX, N - 4: KIND_CONTENTS, N - 3: KIND_CONTENTS}


def test_non_december_end_run_is_contents() -> None:
    """1966: одна полоса содержания в конце; две подряд — тоже содержание, не указатель."""
    got = kinds(decide_issue([page(N - 3, surya_toc_conf=0.8), page(N - 2, num_tail_lines=6, num_tail_ratio=0.3)], 1))
    assert got == {N - 3: KIND_CONTENTS, N - 2: KIND_CONTENTS}


def test_start_run_with_contents_and_index_reference_is_contents() -> None:
    """1975/12: полоса 4 ссылается на указатель («Указатель статей ... 90»), вид — содержание."""
    got = kinds(
        decide_issue([page(2, surya_toc_conf=0.75, kw_contents=True), page(3, surya_toc_conf=0.9, kw_index=True)], 12)
    )
    assert got == {2: KIND_CONTENTS, 3: KIND_CONTENTS}


def test_thresholds_parse() -> None:
    thr = Thresholds.parse(["weak_min_lines=5,weak_min_ratio=0.5", "window_end=9"])
    assert thr.weak_min_lines == 5 and thr.weak_min_ratio == 0.5 and thr.window_end == 9
    try:
        Thresholds.parse(["nope=1"])
    except ValueError as error:
        assert "nope" in str(error)
    else:
        raise AssertionError("неизвестный порог должен быть ошибкой")
