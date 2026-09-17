"""Пересчёт разметки скана в координаты страницы бинаризованного PDF."""

from ocr_utils.line_art_detection.markup import PdfPageMarkup


def page(**kwargs) -> PdfPageMarkup:
    defaults = dict(
        scan_width=3589,
        scan_height=6445,
        scan_dpi=600,
        margin_left_px=288.0,
        margin_top_px=144.0,
        regions=((0, 0, 3589, 6445),),
        full_page=False,
    )
    return PdfPageMarkup(**{**defaults, **kwargs})


def test_поля_сдвигают_прямоугольник_в_родном_разрешении():
    """Замер на full_1966_01.pdf стр.5: скан 3589x6445 + поля 288/144 = страница 4165x6733."""
    assert page().boxes_at(600) == [(288, 144, 3877, 6589)]


def test_другое_разрешение_рендера_масштабирует_и_поля_тоже():
    assert page().boxes_at(300) == [(144, 72, 1938, 3294)]


def test_прямоугольник_внутри_полосы_едет_вместе_с_полями():
    inner = page(regions=((2908 - 288, 2064 - 144, 3752 - 288, 2652 - 144),))
    assert inner.boxes_at(600) == [(2908, 2064, 3752, 2652)]


def test_без_полей_координаты_не_меняются():
    assert page(margin_left_px=0.0, margin_top_px=0.0).boxes_at(600) == [(0, 0, 3589, 6445)]
