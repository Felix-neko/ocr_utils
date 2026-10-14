"""Тесты шагов масок scan_cropping v2 на синтетике: copy back в пределах книги, склейка, доводка края, закрас до рамки, масштабы."""

import numpy as np
import pytest

from ocr_utils.scan_cropping.v2.copy_back import copy_back_blocks
from ocr_utils.scan_cropping.v2.edge import snap_outward
from ocr_utils.scan_cropping.v2.fill import FillMethod, fill_then_overlay, fill_to_rectangle, paper_bbox
from ocr_utils.scan_cropping.v2.glue import glue_hull_bridge
from ocr_utils.scan_cropping.v2.scale import REF_PX_PER_MM, Scale


def spread(gap: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Две страницы 300×280 со щелью ``gap`` px и книга вокруг с полями 30 px.

    Args:
        gap: Ширина щели, px.

    Returns:
        ``(маска бумаги, маска книги)``.
    """
    paper = np.zeros((400, 700), bool)
    paper[50:350, 50:330] = True
    paper[50:350, 330 + gap : 610 + gap] = True
    book = np.zeros_like(paper)
    book[20:380, 20 : 640 + gap] = True
    return paper, book


def test_copy_back_only_inside_book() -> None:
    """Блок, задетый закрасом и вылезающий за книгу, возвращается с оригинала только внутри книги."""
    original = np.full((100, 200, 3), 10, np.uint8)
    clean = np.full_like(original, 200)
    paint = np.zeros((100, 200), np.uint8)
    paint[40:60, 50:150] = 255
    book = np.zeros((100, 200), bool)
    book[:, :100] = True
    block = np.array([[40, 30], [160, 30], [160, 70], [40, 70]], np.float32)
    out, returned, blocks = copy_back_blocks(original, clean, [block], paint, book, (0, 0))
    assert blocks == 1
    # Внутри книги — оригинал, за книгой — закрас.
    assert (out[50, 60] == 10).all()
    assert (out[50, 140] == 200).all()
    # «Возвращено» — только зона закраса внутри книги.
    assert returned[50, 60] and not returned[50, 140] and not returned[35, 60]


def test_copy_back_untouched_block_ignored() -> None:
    """Блок, не задетый закрасом, не возвращается."""
    original = np.zeros((50, 50, 3), np.uint8)
    clean = np.full_like(original, 255)
    paint = np.zeros((50, 50), np.uint8)
    paint[0:5, 0:5] = 255
    block = np.array([[30, 30], [45, 30], [45, 45], [30, 45]], np.float32)
    out, returned, blocks = copy_back_blocks(original, clean, [block], paint, np.ones((50, 50), bool), (0, 0))
    assert blocks == 0 and not returned.any() and (out == 255).all()


def test_glue_closes_gap_inside_book() -> None:
    """Щель 20 px между страницами склеивается, а за край книги склейка не выходит."""
    paper, book = spread(20)
    result = glue_hull_bridge(paper, book)
    assert result.paper[200, 340]
    assert not (result.paper & ~book).any()
    assert not result.paper[200, 30]


def test_glue_wide_gap_not_glued() -> None:
    """Щель шире предела (W_MAX) не склеивается."""
    paper, book = spread(150)
    result = glue_hull_bridge(paper, book)
    assert not result.paper[200, 400]


def test_snap_bg_moves_edge_only_near_background() -> None:
    """``snap_bg``: край доводится до светлой кромки там, где за ней фон, и стоит на месте у торца без фона."""
    h, w = 200, 300
    bgr = np.full((h, w, 3), 30, np.uint8)
    # Бумага 40..260 по x; маска SAM отстаёт от кромки на 8 px с обеих сторон.
    bgr[20:180, 40:260] = 200
    paper = np.zeros((h, w), bool)
    paper[28:172, 48:252] = True
    book = np.zeros((h, w), bool)
    # Слева фон начинается в 12 px от края маски (ближе SNAP_BG_REACH), справа книга тянется до 290.
    book[10:190, 36:290] = True
    bg = ~book
    # Справа за кромкой — светлый «торец» до края книги, фон начинается дальше 15 px.
    bgr[20:180, 260:290] = 190
    snapped = snap_outward(bgr, paper, book, bg, bg_reach=15)
    assert snapped[100, 42]  # слева — фон рядом, край ушёл к кромке
    assert not snapped[100, 254]  # справа фона рядом нет — край на месте


def test_fill_keeps_paper_and_crops_bbox() -> None:
    """Закрас до прямоугольника: размер — осевая рамка бумаги, пиксели бумаги не тронуты, вне бумаги — не чёрное."""
    bgr = np.zeros((120, 160, 3), np.uint8)
    paper = np.zeros((120, 160), bool)
    paper[20:100, 30:130] = True
    paper[20:40, 110:130] = False  # выемка в углу
    bgr[paper] = (180, 190, 200)
    assert paper_bbox(paper) == (30, 20, 130, 100)
    for method in FillMethod:
        out = fill_to_rectangle(bgr, paper, method)
        assert out.shape == (80, 100, 3)
        sub = paper[20:100, 30:130]
        assert (out[sub] == (180, 190, 200)).all()
        assert out[~sub].mean() > 100


def test_fill_empty_paper() -> None:
    """Пустая маска бумаги — None."""
    assert fill_to_rectangle(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), bool), FillMethod.TELEA) is None


@pytest.mark.parametrize("ratio", [1.0, 0.5, 2.0])
def test_scale(ratio: float) -> None:
    """Масштаб сегментации и копии глифов обратны плотности; на калибровочной плотности — ½."""
    s = Scale(REF_PX_PER_MM * ratio)
    assert s.seg == pytest.approx(0.5 / ratio)
    assert s.glyph == pytest.approx(0.5 / ratio)
    assert s.frame_px(40) == round(40 * ratio)


def test_replicate_fills_enclosed_notch() -> None:
    """replicate: выемка не-бумаги между кусками бумаги того же столбца (палец на странице) заливается, а не остаётся."""
    bgr = np.full((120, 160, 3), (180, 190, 200), np.uint8)
    paper = np.zeros((120, 160), bool)
    paper[20:100, 30:130] = True
    # «Палец» — тёмная выемка у левого края страницы: сверху и снизу в её столбцах есть бумага.
    paper[50:70, 30:60] = False
    bgr[50:70, 30:60] = (60, 80, 160)
    out = fill_to_rectangle(bgr, paper, FillMethod.REPLICATE)
    notch = out[30:50, 0:30]
    assert np.abs(notch.astype(int) - (180, 190, 200)).max() < 20


def test_visible_frames_skip_hidden(tmp_path) -> None:
    """Кадры в скрытых папках (``.covers``, ``.дубликаты``) и скрытые файлы отбрасываются."""
    from ocr_utils.scan_cropping.v2.pipeline import visible_frames

    files = [tmp_path / "a" / "IMG_1.jpg", tmp_path / "a" / ".covers" / "IMG_2.jpg", tmp_path / ".x" / "IMG_3.jpg",
             tmp_path / "b" / ".IMG_4.jpg"]  # fmt: skip
    assert visible_frames(files, tmp_path) == [tmp_path / "a" / "IMG_1.jpg"]


def test_thin_bg_line_keeps_margin():
    """Тонкая линия фона поперёк страницы (линейка печати, куда дотекла заливка) не отрезает поле от книги."""
    from ocr_utils.scan_cropping.v2.book import book_close_open

    bg = np.ones((600, 800), bool)
    bg[100:550, 100:700] = False  # страница
    bg[190:196, 100:700] = True  # линия рамки 6 px, связанная с фоном по краям страницы
    hulls = np.zeros_like(bg)
    hulls[250:500, 150:650] = True
    book = book_close_open(bg, hulls)
    assert book[105:185, 150:650].all()  # поле над линией осталось книгой



def test_fill_then_overlay_no_trail() -> None:
    """Край обложки в защите у края бумаги не тянется закрасом в соседнюю не-бумагу; защита и copy back — из ``top``."""
    base = np.full((200, 200, 3), 30, np.uint8)
    base[20:150, 20:180] = 220  # бумага
    base[150:160, 20:180] = 40  # тёмный край обложки под бумагой
    paper = np.zeros((200, 200), bool)
    paper[20:150, 20:180] = True
    keep = np.zeros_like(paper)
    keep[150:160, 60:120] = True  # кусок защиты: блок surya с припуском заходит на обложку
    top = base.copy()
    top[60:70, 60:70] = 0  # copy back внутри бумаги
    out = fill_then_overlay(base, top, paper, keep, FillMethod.TELEA)
    old = fill_to_rectangle(base, paper | keep, FillMethod.TELEA)
    # Рамка прежняя: бумага ∪ защита (строки 20..160, столбцы 20..180).
    assert out.shape == old.shape == (140, 160, 3)
    # Не-бумага справа от защиты (кадр: строки 150..160, столбцы 122..130): закрас светлый, как бумага;
    # прежний порядок тянет туда тёмную обложку из защиты.
    near = (slice(130, 140), slice(102, 110))
    assert out[near].mean() > 180
    assert old[near].mean() < out[near].mean() - 10
    # Защита и copy back — точно из ``top``.
    assert (out[130:140, 40:100] == top[150:160, 60:120]).all()
    assert (out[40:50, 40:50] == 0).all()


def test_block_labels_semi_transparent():
    """Подпись типа блока ложится у верхнего левого угла блока и полупрозрачна: тёмный фон под ней виден."""
    from ocr_utils.scan_cropping.v2.overlay import block_labels

    picture = np.full((200, 300, 3), 255, np.uint8)
    poly = np.array([[40, 50], [250, 60], [240, 180], [30, 170]], np.float32)
    out = block_labels(picture, [poly], ["SectionHeader"])
    changed = np.nonzero((out != 255).any(axis=2))
    assert changed[0].min() >= 50 and changed[1].min() >= 40  # внутри блока от угла
    assert out.min() > 40  # буквы не сплошные: смешаны с бумагой
    assert (block_labels(picture, [poly], None) == picture).all()
