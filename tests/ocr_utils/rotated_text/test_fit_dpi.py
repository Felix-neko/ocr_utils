"""Подбор кегля, требуемая ширина и требуемый DPI."""

from ocr_utils.rotated_text.tables.dpi import MAX_DPI, MIN_FONT_EM_PX, font_cap_px, font_px_for_dpi, required_dpi
from ocr_utils.rotated_text.tables.fit import FIT_MIN_PX, fit_text, load_font, required_width, text_width, wrap


def test_fit_prefers_whole_words_over_larger_broken_font():
    text = "фактический отпуск леса"
    fit = fit_text(text, 130, 400, max_px=60)
    assert fit is not None
    assert all("-" not in line for line in fit.lines), fit


def test_fit_is_none_when_box_is_hopeless():
    assert fit_text("длинное слово", 4, 4, max_px=60) is None
    assert fit_text("", 100, 100, max_px=60) is None


def test_fit_respects_box():
    fit = fit_text("расчетная лесосека", 200, 120, max_px=60)
    assert fit is not None
    assert fit.width <= 200 and fit.height <= 120 and FIT_MIN_PX <= fit.font_px <= 60


def test_wrap_breaks_by_hyphen_then_letters():
    font = load_font(20)
    narrow = text_width(font, "сутко-") + 2
    lines = wrap("сутко-комплекте", font, narrow)
    assert lines[0] == "сутко-"
    lines = wrap("Специфицированная", font, text_width(font, "Специфи-") + 1)
    assert all(text_width(font, line) <= text_width(font, "Специфи-") + 1 for line in lines)


def test_required_width_makes_text_fit():
    text = "1966—1970 гг. в % к 1961—1965 гг."
    width = required_width(text, 40, 300)
    fit = fit_text(text, width, 300, max_px=40, min_px=40)
    assert fit is not None and fit.font_px == 40


def test_required_dpi():
    assert required_dpi(600, 60) == 600  # уже крупнее порога
    assert required_dpi(600, MIN_FONT_EM_PX) == 600
    assert required_dpi(600, 20) == 1200
    assert required_dpi(600, 15) == 1600  # выше потолка — решает лестница, не эта функция
    assert required_dpi(600, 0) is None
    assert MAX_DPI == 1350


def test_font_cap_and_font_for_dpi():
    assert font_cap_px(600, 67) == 67  # собственный кегль таблицы мельче 10 pt
    assert font_cap_px(600, 0) == 83  # 10 pt при 600 dpi
    assert font_cap_px(300, 33) == MIN_FONT_EM_PX  # мелкий оригинал не опускает ниже порога
    # Кегль в исходных пикселях, который дорастёт до порога при увеличении до потолка.
    assert font_px_for_dpi(600, 1350) * 1350 / 600 >= MIN_FONT_EM_PX
