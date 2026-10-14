"""Тесты стенда закраса пограничных областей на синтетике: мера шлейфа, маска подозрительного у края, pull-push, раздельный закрас страниц."""

import numpy as np

from research.border_fill.measure import trail_score
from research.border_fill.variants import FillInput, Variant, fill_variant, periphery, page_map, pull_push

PX_PER_MM = 4.0
PAPER = (200, 225, 235)
COVER = (90, 40, 30)


def page_with_stamp() -> FillInput:
    """Страница 300×300 на поле 400×400 (вырезка — вся картинка): у нижнего края — тёмная печать, под ней — фон."""
    img = np.full((400, 400, 3), COVER, np.uint8)
    img[:300, :] = PAPER
    img[270:300, 150:250] = (120, 60, 120)  # печать у края, внутри бумаги
    paper = np.zeros((400, 400), bool)
    paper[:300, :] = True
    return FillInput(img, img, paper, np.zeros_like(paper), PX_PER_MM)


def test_stamp_is_suspect() -> None:
    """Печать у края — в маске подозрительного, чистая бумага — нет."""
    inp = page_with_stamp()
    small = inp.base[::4, ::4]
    per = periphery(small, inp.paper[::4, ::4], page_map((100, 100), None), PX_PER_MM / 4)
    assert per.suspect[70:74, 40:60].mean() > 0.9
    assert per.suspect[60:70, 5:30].mean() < 0.05


def test_clean_variants_no_trail() -> None:
    """Telea по бумаге тянет печать вниз; варианты с маской подозрительного — нет."""
    inp = page_with_stamp()
    plain = trail_score(fill_variant(inp, Variant.TELEA_PAPER), inp.paper, inp.keep, PX_PER_MM)
    assert plain.area_mm2 > 0
    for v in (Variant.TELEA_CLEAN, Variant.PULL_PUSH, Variant.REGION, Variant.HARMONIC):
        out = fill_variant(inp, v)
        score = trail_score(out, inp.paper, inp.keep, PX_PER_MM)
        assert score.area_mm2 < plain.area_mm2 / 4, v
        assert (out[:300] == inp.top[:300]).all()  # бумага не тронута


def test_pull_push_constant() -> None:
    """Pull-push по известной половине ровного цвета заливает вторую половину тем же цветом."""
    img = np.zeros((64, 64, 3), np.float32)
    img[:, :32] = 100
    weight = np.zeros((64, 64), np.float32)
    weight[:, :32] = 1
    out = pull_push(img, weight)
    assert np.abs(out[:, 32:] - 100).max() < 1


def test_two_pages_colours_kept() -> None:
    """Голубая и белая страницы: закрас под каждой — цвета своей страницы."""
    img = np.full((400, 400, 3), COVER, np.uint8)
    img[:300, :200] = (230, 200, 60)  # голубая
    img[:300, 200:] = (235, 240, 240)  # белая
    paper = np.zeros((400, 400), bool)
    paper[:300] = True
    inp = FillInput(img, img, paper, np.zeros_like(paper), PX_PER_MM, spine_x=200)
    out = fill_variant(inp, Variant.PULL_PUSH)
    assert np.abs(out[350, 60].astype(int) - (230, 200, 60)).max() < 12
    assert np.abs(out[350, 340].astype(int) - (235, 240, 240)).max() < 12
