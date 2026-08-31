"""Гашение следов шариковой ручки: линейная комбинация каналов вокруг самой кляксы."""

import numpy as np
from PIL import Image

from ocr_utils.scan_markup.pen_marks import DEFAULT_WEIGHTS, fix_page, pen_mask, recombine, remove_pen_marks

# Средние BGR по замеру 1968/01 IMG_0045_2R — те же числа, что в докстринге модуля.
PAPER = (253, 247, 250)
TEXT = (43, 34, 37)
PEN = (162, 87, 59)
TEXT_UNDER_PEN = (91, 24, 13)

BLOT = (400, 300, 700, 800)  # x1, y1, x2, y2


def _page(size=(1200, 1000)) -> np.ndarray:
    """Полоса: бумага, строки печатного текста и синяя клякса поверх части из них."""
    page = np.full((*size, 3), PAPER, np.uint8)
    for y in range(100, size[0] - 100, 60):
        page[y : y + 26, 100 : size[1] - 100] = TEXT

    x1, y1, x2, y2 = BLOT
    blot = page[y1:y2, x1:x2]
    is_text = (blot == np.array(TEXT, np.uint8)).all(axis=-1)
    blot[is_text] = TEXT_UNDER_PEN
    blot[~is_text] = PEN
    return page


def test_recombine_makes_the_blot_pale_and_keeps_text_dark() -> None:
    """Главное свойство комбинации: клякса уходит к бумаге, текст под ней остаётся тёмным.

    Уровни — из замера в докстринге модуля: клякса 217 при бумаге 245, текст под кляксой 118
    при печатном тексте 30. Проверяется порядок величин, а не точное совпадение: приведение
    считается по перцентилям самой полосы.
    """
    page = _page()
    result = recombine(page, DEFAULT_WEIGHTS)[..., 0].astype(float)

    x1, y1, x2, y2 = BLOT
    source = page[y1:y2, x1:x2]
    is_text_under = (source == np.array(TEXT_UNDER_PEN, np.uint8)).all(axis=-1)
    inside = result[y1:y2, x1:x2]

    paper_level = result[10:60, 10:60].mean()
    blot_level = inside[~is_text_under].mean()
    text_under_level = inside[is_text_under].mean()

    assert blot_level > paper_level - 45, "клякса обязана уйти почти к уровню бумаги"
    assert text_under_level < blot_level - 60, "текст под кляксой обязан остаться читаемым"


def test_single_channel_cannot_do_both() -> None:
    """Контроль: одним каналом обе стороны сразу не берутся — потому комбинация и нужна.

    Синий гасит текст под кляксой вместе с ней, красный делает кляксу чернее текста.
    """
    page = _page()
    x1, y1, x2, y2 = BLOT
    source = page[y1:y2, x1:x2]
    is_text_under = (source == np.array(TEXT_UNDER_PEN, np.uint8)).all(axis=-1)

    def levels(weights):
        result = recombine(page, weights)[..., 0].astype(float)
        inside = result[y1:y2, x1:x2]
        return result[10:60, 10:60].mean(), inside[~is_text_under].mean(), inside[is_text_under].mean()

    _paper, blue_blot, _blue_text = levels((1, 0, 0))
    _paper, red_blot, _red_text = levels((0, 0, 1))
    _paper, combo_blot, _combo_text = levels(DEFAULT_WEIGHTS)

    assert blue_blot < combo_blot, "у синего канала клякса заметно темнее"
    assert red_blot < blue_blot, "у красного клякса самая тёмная — паста поглощает красный"


def test_only_the_blot_neighbourhood_changes() -> None:
    """Комбинация подмешивается только вокруг кляксы, остальной кадр остаётся прежним."""
    page = _page()
    fixed, area = remove_pen_marks(page, dpi=600)
    assert area > 0

    changed = np.abs(fixed.astype(int) - page.astype(int)).max(axis=2) > 2
    assert changed.any()
    # Далёкий от кляксы угол не тронут вовсе.
    assert not changed[:100, :100].any()
    # Изменения не расползлись по полосе: маска раздувается всего на пару миллиметров.
    rows, cols = np.nonzero(changed)
    assert cols.min() > BLOT[0] - 120 and cols.max() < BLOT[2] + 120
    assert rows.min() > BLOT[1] - 120 and rows.max() < BLOT[3] + 120


def test_page_without_pen_is_left_alone() -> None:
    """Полоса без пасты возвращается как есть, и файл не пишется."""
    page = np.full((600, 500, 3), PAPER, np.uint8)
    page[100:130, 50:450] = TEXT
    fixed, area = remove_pen_marks(page, dpi=600)
    assert area == 0
    assert fixed is page


def test_pen_mask_ignores_a_speck() -> None:
    """Крапина цветного шума в маску не попадает: иначе она раздулась бы на два миллиметра."""
    page = np.full((600, 500, 3), PAPER, np.uint8)
    page[300:310, 300:310] = PEN  # 100 px при пороге 4000
    assert not pen_mask(page, dpi=600).any()


def test_fix_page_keeps_dpi_and_writes_beside(tmp_path) -> None:
    """Починенный файл ложится в отдельную папку с тем же относительным путём и тем же DPI."""
    pack = tmp_path / "пак"
    (pack / "1968" / "01").mkdir(parents=True)
    src = pack / "1968" / "01" / "a.tif"
    Image.fromarray(_page()[..., ::-1]).save(src, dpi=(600, 600), compression="tiff_lzw")

    out = tmp_path / "починенные"
    result = fix_page(pack, "1968/01/a.tif", out)
    assert result.error == "" and result.dst == out / "1968" / "01" / "a.tif"
    assert result.pen_area_px > 0

    with Image.open(result.dst) as written:
        assert written.info.get("dpi") == (600, 600)
        assert written.mode == "RGB"
        assert written.size == (1000, 1200)
