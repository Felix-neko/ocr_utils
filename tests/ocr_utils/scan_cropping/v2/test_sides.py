"""Тесты выравнивания боковых сторон scan_cropping v2 на синтетике: выброс, яма, яма над фоном, подтяжка к краю, приподнятая страница, дыры, замкнутые заливкой, кромка листа за светлым полем цветной страницы."""

import numpy as np

from ocr_utils.scan_cropping.v2.page_pose import PageFeatures, PagePose, PageQuad
from ocr_utils.scan_cropping.v2.side_fit import mm, regularize_sides, snap_sheet_edges

# Цвета синтетики (BGR): жёлтая бумага, бумага в тени у кромки (того же оттенка), белая стопка, чёрный коврик.
PAPER = (170, 225, 235)
SHADE = (150, 205, 215)
STACK = (240, 240, 240)
MAT = (20, 20, 20)


def single_page(paper: np.ndarray, pose: PagePose = PagePose.FLAT) -> list[PageQuad]:
    """Одиночная страница с трапецией по рамке маски.

    Args:
        paper: Маска бумаги.
        pose: Класс положения.

    Returns:
        Список из одной страницы.
    """
    ys, xs = np.nonzero(paper)
    quad = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], float)
    f = PageFeatures(fill=1.0, keystone=0.0, width_share=1.0, light_share=1.0, finger=0.0, curl=0.0)
    return [PageQuad(paper, quad, f, pose)]


def frame(shape=(700, 600)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Кадр: страница 100..600 × 100..500 на белой стопке, вокруг коврик.

    Returns:
        ``(картинка, бумага, фон)``.
    """
    img = np.zeros(shape + (3,), np.uint8)
    img[:] = MAT
    img[60:640, 60:560] = STACK
    img[100:600, 100:500] = PAPER
    paper = np.zeros(shape, bool)
    paper[100:600, 100:500] = True
    bg = np.all(img == MAT, axis=2)
    return img, paper, bg


def test_pit_filled_lobe_cut():
    """Яма 3 мм на правой стороне залита, выброс 5 мм на левой срезан, верх и низ не тронуты."""
    img, paper, bg = frame()
    sam = paper.copy()
    sam[300:360, 500 - mm(3) : 500] = False  # яма
    sam[400:450, 100 - mm(5) : 100] = True  # выброс (на стопке, не на фоне)
    r = regularize_sides(sam, single_page(sam), bg, img)
    assert r.paper[300:360, 500 - mm(3) : 499].all()
    assert not r.paper[400:450, 100 - mm(5) : 100 - mm(2)].any()
    assert (r.paper[100] == paper[100]).all() and (r.paper[599] == paper[599]).all()


def test_pit_over_background_not_filled():
    """Яма, под которой чёрный фон, не заливается."""
    img, paper, bg = frame()
    sam = paper.copy()
    sam[300:360, 480:500] = False
    img[300:360, 480:500] = MAT
    bg = np.all(img == MAT, axis=2)
    r = regularize_sides(sam, single_page(sam), bg, img)
    assert not r.paper[300:360, 482:500].any()


def test_snap_to_edge_behind_shade():
    """SAM 3 колеблется в тени у кромки: сторона подтягивается к переходу «жёлтое → белое»."""
    img, paper, bg = frame()
    img[100:600, 500 - mm(6) : 500] = SHADE
    sam = paper.copy()
    rng = np.random.default_rng(0)
    for y in range(100, 600, 10):
        sam[y : y + 10, 500 - rng.integers(mm(1), mm(6)) : 500] = False
    r = regularize_sides(sam, single_page(sam), bg, img)
    right = [c for c in r.curves if c.side.value == "right"][0]
    assert right.shift > 0
    assert abs(right.at(np.array([350]))[0] - 499) <= 3


def test_raised_page_untouched():
    """Приподнятая страница не выравнивается."""
    img, paper, bg = frame()
    sam = paper.copy()
    sam[300:360, 500 - mm(3) : 500] = False
    r = regularize_sides(sam, single_page(sam, PagePose.RAISED), bg, img)
    assert (r.paper == sam).all() and not r.curves


def test_wavy_side_kept():
    """Волнистая сторона (выгиб крупнее окна медианы) сохраняется: ничего не срезается и не заливается сверх ямы."""
    img, paper, bg = frame((900, 700))
    img[:] = MAT
    img[40:860, 40:660] = STACK
    sam = np.zeros((900, 700), bool)
    for y in range(100, 800):
        right = 500 + int(30 * np.sin((y - 100) / 700 * 2 * np.pi))
        sam[y, 100:right] = True
        img[y, 100:right] = PAPER
    bg = np.all(img == MAT, axis=2)
    r = regularize_sides(sam, single_page(sam), bg, img, snap=False)
    assert r.cut.sum() < 0.002 * sam.sum() and r.added.sum() < 0.002 * sam.sum()


def finger_pit(sam: np.ndarray) -> np.ndarray:
    """Яма от пальца на правой стороне с язычком бумаги под кончиком (как у МТС 1991/02 IMG_0634).

    Args:
        sam: Маска бумаги, меняется копия.

    Returns:
        Маска с ямой 5 мм глубиной на строках 300..380 и язычком бумаги у кромки на строках 360..380.
    """
    sam = sam.copy()
    sam[300:380, 500 - mm(5) : 500] = False
    sam[360:380, 500 - mm(1.5) : 500] = True  # язычок у кромки: заливка в этих строках упирается в него
    return sam


def test_hole_closed_by_fill_filled():
    """Дыра за язычком, которую замкнула заливка ямы, доливается: внутри страницы не остаётся не-бумаги."""
    img, paper, bg = frame()
    sam = finger_pit(paper)
    r = regularize_sides(sam, single_page(sam), bg, img)
    # Последний столбец страницы кривая отсекает долями пикселя (как и без язычка) — проверка до него.
    assert r.paper[300:380, 500 - mm(5) : 499].all()
    assert r.added[360:380, 500 - mm(5) : 500 - mm(1.5)].any()


def test_dark_hole_closed_by_fill_kept():
    """Тёмная дыра (печать, которую SAM 3 не взял), замкнутая заливкой, не доливается: она темнее бумаги на 25+."""
    img, paper, bg = frame()
    sam = finger_pit(paper)
    img[360:380, 500 - mm(5) : 500 - mm(1.5)] = (40, 40, 40)
    r = regularize_sides(sam, single_page(sam), bg, img)
    assert not r.paper[362:378, 500 - mm(4.5) : 500 - mm(2)].any()


def test_hole_closed_before_fill_untouched():
    """Дыра внутри страницы, замкнутая ещё до выравнивания (полоса у корешка, печать), не трогается."""
    img, paper, bg = frame()
    sam = paper.copy()
    sam[300:340, 300:340] = False
    r = regularize_sides(sam, single_page(sam), bg, img)
    assert not r.paper[300:340, 300:340].any()



def colour_page() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Цветная страница: оранжевая плашка, кремовое поле 15 мм слева, тень кромки листа, за ней белый картон.

    Returns:
        ``(картинка, рваная маска по плашке, книга, фон)``.
    """
    img = np.zeros((700, 600, 3), np.uint8)
    img[:] = MAT
    img[60:640, 40:560] = (240, 240, 240)  # белый картон переплёта
    img[100:600, 150:500] = (200, 228, 238)  # кремовая бумага листа
    img[100:600, 150:152] = (150, 170, 180)  # тень кромки листа
    edge = 150 + mm(15)
    img[110:590, edge:490] = (40, 140, 240)  # оранжевая плашка
    paper = np.zeros((700, 600), bool)
    paper[110:590, edge:490] = True
    # Рваные языки SAM 3 в поле: случайной глубины 0–8 мм полосами по 10 строк (как у IMG_1113, ст. откл. края 23 px).
    rng = np.random.default_rng(1)
    for y in range(110, 590, 10):
        paper[y : y + 10, edge - mm(rng.uniform(0, 8)) : edge] = True
    paper[100:600, 490:500] = True
    book = np.zeros((700, 600), bool)
    book[60:640, 40:560] = True
    bg = np.all(img == MAT, axis=2)
    return img, paper, book, bg


def test_sheet_edge_behind_margin():
    """Поле цветной страницы до кромки листа заливается; картон за кромкой — нет."""
    img, paper, book, bg = colour_page()
    added = snap_sheet_edges(paper, single_page(paper), bg, book, img)
    assert (added | paper)[200:500, 155 : 150 + mm(14)].mean() > 0.95
    assert not added[:, :148].any()


def test_sheet_edge_smooth_mask_untouched():
    """Ровный край маски (SAM 3 уверен) — кромку не ищем, ничего не добавляется."""
    img, paper, book, bg = colour_page()
    paper[110:590, 150 + mm(10) : 150 + mm(15)] = False
    paper = paper.copy()
    paper[:, : 150 + mm(15)] = False
    added = snap_sheet_edges(paper, single_page(paper), bg, book, img)
    assert not added.any()
