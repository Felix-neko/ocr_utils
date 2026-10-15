"""Тесты тени пальца (синтетика): тень на бумаге с буквами снимается, край зоны поиска без ступеньки, без тени — без поправки, кожа на белой и жёлтой бумаге, оверлей тени."""

import cv2
import numpy as np
import pytest

from ocr_utils.scan_cropping.v2.finger_shadow import estimate_finger_shadow, lighten_finger_shadow, skin_mask
from ocr_utils.scan_cropping.v2.overlay import draw_finger_shadow

PX_PER_MM = 8.0
PAPER = np.array([215, 230, 238], np.float32)
H, W = 1600, 1200


def page(shadow: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Страница — бумага со строками букв, справа коврик; палец у правого края, гауссова тень вокруг.

    Returns:
        ``(итог BGR, зона закраса, ядро, книга)``.
    """
    img = np.tile(PAPER, (H, W, 1))
    if shadow:
        yy, xx = np.mgrid[0:H, 0:W]
        g = np.exp(
            -(((xx - 1050) / 80.0) ** 2 + ((yy - 800) / 110.0) ** 2)
        )  # σ 10 и 14 мм: тень в пределах зоны поиска
        img *= (1 - 0.25 * g)[..., None]
    img = img.astype(np.uint8)
    for y in range(120, H - 80, 40):
        cv2.putText(img, "TEXT TEXT TEXT TEXT", (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
    book = np.zeros((H, W), bool)
    book[:, :1100] = True
    # За краем книги в итоге к этому шагу — закрас pull-push цвета бумаги, а не коврик.
    img[~book] = PAPER.astype(np.uint8)
    core = np.zeros((H, W), np.uint8)
    cv2.ellipse(core, (1100, 800), (90, 50), 0, 0, 360, 1, -1)
    paint = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0
    return img, paint, core > 0, book


def paper_drop(img: np.ndarray, book: np.ndarray) -> float:
    """Наибольший провал серого бумаги (без букв) у пальца против бумаги вдали, уровни."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    near = gray[700:900, 900:1090]
    far = np.median(gray[100:300, 20:50])
    return float(far - np.percentile(near[near > 150], 5))


def test_shadow_removed() -> None:
    """Тень вокруг пальца после осветления почти уходит, буквы остаются тёмными."""
    img, paint, core, book = page(shadow=True)
    shadow = estimate_finger_shadow(img, paint, core, book, ~paint, PX_PER_MM)
    out = lighten_finger_shadow(img, shadow)
    assert paper_drop(img, book) > 30
    assert paper_drop(out, book) < 8
    letters = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[book] < 80
    assert letters.sum() > 0.5 * (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[book] < 80).sum()


def test_no_shadow_no_change() -> None:
    """Без тени поправка почти нулевая."""
    img, paint, core, book = page(shadow=False)
    out = lighten_finger_shadow(img, estimate_finger_shadow(img, paint, core, book, ~paint, PX_PER_MM))
    assert np.abs(out.astype(int) - img.astype(int))[book].mean() < 1.0


def test_edge_starts_from_zero() -> None:
    """На краю зоны поиска, где снаружи бумага, поправка нулевая, внутри — заметная."""
    img, paint, core, book = page(shadow=True)
    shadow = estimate_finger_shadow(img, paint, core, book, ~paint, PX_PER_MM)
    level = shadow.delta.max(axis=2)
    book_s = cv2.resize(book.astype(np.uint8), level.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    paper_out = book_s & ~shadow.area
    edge = shadow.area & (cv2.dilate(paper_out.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    # Меньше уровня: опора — за тенью (кольцо 25–35 мм), на краю зоны 15 мм тень ещё есть, и рампа гасит не ноль.
    assert level[edge].max() < 1.0
    assert level.max() > 20


def lab_to_bgr(light: float, a: float, b: float) -> tuple[int, int, int]:
    """Цвет BGR uint8 по Lab (L 0..100, a*, b* как в CIE)."""
    lab = np.array([[[light * 255 / 100, a + 128, b + 128]]], np.uint8)
    return tuple(int(v) for v in cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0])


@pytest.mark.parametrize("paper_ab", [(0, 2), (3, 18)])
def test_skin_relative_to_paper(paper_ab: tuple[float, float]) -> None:
    """Белая и пожелтевшая бумага (b* +18) — не кожа; кожа — кожа; фиолетовая печать и тень — нет; в закрасе — нет."""
    img = np.full((400, 400, 3), lab_to_bgr(88, *paper_ab), np.uint8)
    img[100:200, 100:200] = lab_to_bgr(65, 13, 20)  # кожа
    img[100:200, 250:350] = lab_to_bgr(65, 13, 20)  # тот же цвет, но в закрасе LaMa (не оригинал)
    img[250:350, 40:180] = lab_to_bgr(40, 20, -40)  # фиолетовая печать
    img[250:350, 220:380] = lab_to_bgr(66, paper_ab[0] * 0.8, paper_ab[1] * 0.8)  # тень
    allowed = np.ones((400, 400), bool)
    allowed[90:210, 240:360] = False
    everywhere = np.ones((400, 400), bool)
    skin = skin_mask(img, 4.0, everywhere, everywhere, allowed)
    assert skin[120:180, 120:180].all()
    assert not skin[120:180, 270:330].any()
    assert not skin[260:340, 60:160].any()
    assert not skin[260:340, 240:360].any()


def test_shadow_picture() -> None:
    """Картинка тени: есть зона — есть картинка с двумя панелями; нет зоны — None."""
    img, paint, core, book = page(shadow=True)
    shadow = estimate_finger_shadow(img, paint, core, book, ~paint, PX_PER_MM)
    picture = draw_finger_shadow(img, shadow, paint, core, PX_PER_MM, "тест")
    assert picture is not None and picture.shape[1] > picture.shape[0] * 0.5
    empty = estimate_finger_shadow(img, np.zeros_like(paint), np.zeros_like(core), book, ~paint, PX_PER_MM)
    assert draw_finger_shadow(img, empty, paint, core, PX_PER_MM, "тест") is None


def test_black_print_on_cyan_paper_not_skin() -> None:
    """Чёрная печать и полоса на голубой обложке — не кожа (относительно голубой бумаги чёрный «краснее»)."""
    img = np.full((400, 400, 3), lab_to_bgr(75, -25, -15), np.uint8)
    img[100:300, 150:230] = lab_to_bgr(15, 0, 0)  # чёрная полоса
    img[50:90, 40:360] = lab_to_bgr(30, 1, 1)  # крупная чёрная печать
    everywhere = np.ones((400, 400), bool)
    assert not skin_mask(img, 4.0, everywhere, everywhere, everywhere).any()


def test_light_fill_beyond_edge_not_overlit() -> None:
    """Тёмный призрак у края листа, а закрас за краем уже цвета бумаги: закрас не осветляется сверх бумаги (IMG_0972)."""
    img = np.tile(PAPER, (H, W, 1))
    yy, xx = np.mgrid[0:H, 0:W]
    ghost = np.exp(-(((xx - 1080) / 40.0) ** 2 + ((yy - 800) / 60.0) ** 2))
    img = (img * (1 - 0.3 * ghost * (xx < 1100))[..., None]).astype(np.uint8)  # призрак только на листе
    book = np.zeros((H, W), bool)
    book[:, :1100] = True
    core = np.zeros((H, W), np.uint8)
    cv2.ellipse(core, (1150, 800), (60, 40), 0, 0, 360, 1, -1)
    paint = (cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0) & book
    out = lighten_finger_shadow(img, estimate_finger_shadow(img, paint, core > 0, book, ~paint, PX_PER_MM))
    gray_in, gray_out = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(int) for x in (img, out))
    paper = int(np.median(gray_in[100:300, 100:300]))
    assert gray_out[700:900, 1100:1180].max() <= paper + 3  # закрас за краем не светлее бумаги
    assert gray_out[790:810, 1060:1095].min() >= paper - 12  # призрак на листе осветлён


def test_ring_on_other_surface_not_lightened() -> None:
    """Палец на голубой обложке у края, кольцо опоры — на белой бумаге: зона не осветляется (МТС 1990/04 IMG_1112)."""
    cyan = np.array(lab_to_bgr(72, -25, -30), np.float32)  # светлее порога «крупной печати» (25 L), как у IMG_1112
    img = np.tile(PAPER, (H, W, 1))
    img[:, 1060:] = cyan  # полоса обложки у правого края; опора 25–35 мм от пальца — на белой бумаге слева
    img = img.astype(np.uint8)
    book = np.ones((H, W), bool)
    core = np.zeros((H, W), np.uint8)
    cv2.ellipse(core, (1150, 800), (60, 40), 0, 0, 360, 1, -1)
    paint = (cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0) & book
    shadow = estimate_finger_shadow(img, paint, core > 0, book, ~paint, PX_PER_MM)
    assert shadow.gaps and max(shadow.gaps) > 20
    out = lighten_finger_shadow(img, shadow)
    assert np.abs(out.astype(int) - img.astype(int))[:, 1060:].max() <= 1
