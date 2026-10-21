"""Тесты ухода текста под корешок на постраничной съёмке: сторона по имени, сгиб, отступ, потерянный дефис, решение."""

import numpy as np
import pytest

from research.spread_defects.single_page_gutter import (
    Side,
    FrameSigns,
    Verdict,
    decide,
    find_fold,
    gutter_strip,
    line_margin,
    page_side,
    gutter_features,
)
from research.spread_defects.truncated_words import check_lines


def test_page_side_by_suffix():
    """``_1`` — правая полоса (сгиб у начала строк), ``_2`` — левая; иное имя — ошибка."""
    assert page_side("0030_1") is Side.RIGHT
    assert page_side("0030_2") is Side.LEFT
    with pytest.raises(ValueError):
        page_side("DSCF0001")


def _synthetic_frame(fold_row: int = 200) -> np.ndarray:
    """Кадр бумаги 180 с тёмной складкой-тенью на строке ``fold_row`` и светлее ниже неё."""
    gray = np.full((1200, 2000), 180.0, dtype=np.float32)
    gray[:fold_row] = 150.0
    gray[fold_row - 4 : fold_row + 4] = 90.0
    return gray


def test_find_fold_on_synthetic_crease():
    """Сгиб на синтетической складке находится на её строке по всей ширине."""
    fold, _, _, _ = find_fold(_synthetic_frame(200))
    assert fold is not None
    assert np.allclose(np.polyval(fold, [100, 1000, 1900]), 200, atol=4)


def test_find_fold_absent_on_blank():
    """На ровной бумаге без складки сгиб не находится."""
    fold, _, _, _ = find_fold(np.full((1200, 2000), 180.0, dtype=np.float32))
    assert fold is None


def test_gutter_strip_orientation():
    """Правая полоса: сгиб уходит влево ленты; левая — вправо; за сгибом закрашено бумагой."""
    rgb = np.repeat(_synthetic_frame(200).astype(np.uint8)[..., None], 3, axis=2)
    fold = np.full(rgb.shape[1], 200.0)
    right = gutter_strip(rgb, fold, Side.RIGHT, depth=300)
    left = gutter_strip(rgb, fold, Side.LEFT, depth=300)
    assert right.image.shape[:2] == (2000, 500) and left.image.shape[:2] == (2000, 500)
    assert np.allclose(right.fold_col, 200) and np.allclose(left.fold_col, 299)
    # Тёмная до сгиба зона (150) закрашена светлым уровнем бумаги.
    assert right.image[1000, 50, 0] >= 170 and left.image[1000, 450, 0] >= 170


def test_line_margin_sign():
    """Отступ положителен, когда между строкой и сгибом бумага, и отрицателен, когда строка заходит за сгиб."""
    fold_col = np.full(100, 50.0)
    assert line_margin({"bbox": [80, 10, 400, 30]}, fold_col, Side.RIGHT) == 30
    assert line_margin({"bbox": [40, 10, 400, 30]}, fold_col, Side.RIGHT) == -10
    assert line_margin({"bbox": [0, 10, 20, 30]}, fold_col, Side.LEFT) == 30


def test_lost_hyphen_merges_only_with_option():
    """Конец строки без прочитанного дефиса склеивается с началом следующей только с ``lost_hyphen``."""
    texts = [["рост", "экономи"], ["ческого", "потенциала"]]
    assert check_lines(texts, "L").fold_unknown == 1
    assert check_lines(texts, "L", lost_hyphen=True).fold_unknown == 0
    # Обрубок с утраченными буквами не склеивается и с опцией.
    assert check_lines([["рост", "экон"], ["ческого", "потенциала"]], "L", lost_hyphen=True).fold_unknown == 1


def test_decide_bands():
    """Словарь без геометрии и пикселей: потеря, норма, между порогами, мало слов, шумный OCR."""
    assert decide(FrameSigns(100, 0.40, 0.01))[0] is Verdict.LOSS
    assert decide(FrameSigns(100, 0.03, 0.01))[0] is Verdict.NORMAL
    assert decide(FrameSigns(100, 0.09, 0.01))[0] is Verdict.UNCLEAR
    assert decide(FrameSigns(5, 0.90, 0.01))[0] is Verdict.UNCLEAR
    assert decide(FrameSigns(100, 0.40, 0.20))[0] is Verdict.UNCLEAR


def test_decide_geometry_normal_beats_few_words_and_noise():
    """Строки не касаются складки — норма, даже при мало слов, шумном OCR и спорной доле."""
    assert decide(FrameSigns(3, float("nan"), float("nan"), touch_share=0.0, margin_p10=1.2))[0] is Verdict.NORMAL
    assert decide(FrameSigns(70, 0.16, 0.17, touch_share=0.0, margin_p10=1.1))[0] is Verdict.NORMAL


def test_decide_pixels():
    """Буквы тонут в тени — потеря при малой доле обрубков; целы — норма; спорный пояс решает словарь."""
    touching = dict(touch_share=0.9, margin_p10=-0.3, touching=80)
    assert decide(FrameSigns(80, 0.05, 0.01, gap_median=-0.18, **touching))[0] is Verdict.LOSS
    assert decide(FrameSigns(80, 0.06, 0.01, gap_median=0.0, **touching))[0] is Verdict.NORMAL
    assert decide(FrameSigns(80, 0.13, 0.01, gap_median=-0.07, **touching))[0] is Verdict.LOSS
    assert decide(FrameSigns(80, 0.02, 0.01, gap_median=-0.07, **touching))[0] is Verdict.NORMAL
    assert decide(FrameSigns(80, 0.07, 0.01, gap_median=-0.07, **touching))[0] is Verdict.UNCLEAR
    # Шумный OCR не мешает потере по пикселям (реклама с обрезанными строками, 1510_1).
    assert decide(FrameSigns(80, 0.64, 0.12, gap_median=-0.13, **touching))[0] is Verdict.LOSS


def test_hyphenated_compound_not_truncated():
    """Слово с дефисом, разорванное переносом («стран-» + «членов»), склеивается с дефисом и обрубком не считается."""
    texts = [["рост", "стран-"], ["членов", "СЭВ"]]
    assert check_lines(texts, "L", lost_hyphen=True).fold_unknown == 0


def test_skip_capitalized_drops_names():
    """Имя собственное у сгиба словарь не знает; с ``skip_capitalized`` оно не проверяется, обрубок — проверяется."""
    texts = [["сказал", "Асаульченко"], ["рост", "сооруже"]]
    assert check_lines(texts, "L").fold_unknown == 2
    check = check_lines(texts, "L", skip_capitalized=True)
    assert check.fold_checked == 1 and check.fold_unknown == 1


def test_second_column_not_mixed_into_fold_column():
    """Правая полоса: строки второй колонки (13 высот от сгиба) не перемежаются с крайней — хвосты переносов
    крайней колонки склеиваются со своими строками и обрубками не считаются."""
    fold_col = np.zeros(2000)
    lines = []
    for n in range(20):
        y = 100 + n * 30
        # Крайняя колонка: каждая строка кончается переносом, следующая начинается хвостом.
        lines.append({"bbox": [5, y, 400, y + 25], "text": "работа органи-" if n % 2 == 0 else "зации завода и"})
        # Вторая колонка на той же высоте.
        lines.append({"bbox": [330 + 5, y + 3, 700, y + 28], "text": "ной продукции цеха"})
    features = gutter_features(lines, fold_col, Side.RIGHT)
    assert features.fold_checked >= 8 and features.fold_unknown == 0


def _line_strip(letters_end: int, shadow_from: int) -> np.ndarray:
    """Синтетическая строка левой полосы: бумага 200, буквы-палочки до ``letters_end``, тень складки с
    ``shadow_from`` (бумага 120) до правого края."""
    gray = np.full((60, 400), 200, np.uint8)
    for x in range(20, letters_end, 24):
        gray[15:45, x : x + 6] = 40
    gray[:, shadow_from:] = (gray[:, shadow_from:] * 0.6).astype(np.uint8)
    return gray


def test_line_edge_gap_sign():
    """Буквы, заходящие в тень, дают отрицательный просвет; буквы, кончающиеся до тени, — положительный."""
    from research.spread_defects.single_page_gutter import line_edge

    line = {"bbox": [0, 10, 390, 50]}
    fold_col = np.full(60, 399.0)
    # Буква 308–314 начинается до тени (311) и заходит в неё — срезанная.
    into = line_edge(_line_strip(390, 311), line, fold_col, Side.LEFT, 40.0, keep=0.0)
    # Последняя буква 284–290 кончается до тени (330): просвет есть.
    clear = line_edge(_line_strip(300, 330), line, fold_col, Side.LEFT, 40.0, keep=0.0)
    assert into.gap < 0 < clear.gap
