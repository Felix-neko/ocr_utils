"""Тесты тени пальца (синтетика): тень на бумаге с буквами снимается, край зоны поиска без ступеньки, без тени — без поправки, кожа на белой и жёлтой бумаге, оверлей тени."""

import cv2
import numpy as np
import pytest

from ocr_utils.scan_cropping.v2.finger_shadow import MAX_ADD_L, estimate_finger_shadow, lighten_finger_shadow, skin_mask
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
    """Палец на голубой обложке у края, кольцо опоры большей частью на белой бумаге: опора — голубая часть кольца,
    обложка не высветляется (МТС 1990/04 IMG_1112)."""
    cyan = np.array(lab_to_bgr(72, -25, -30), np.float32)  # светлее порога «крупной печати» (25 L), как у IMG_1112
    img = np.tile(PAPER, (H, W, 1))
    img[:, 1060:] = cyan  # полоса обложки у правого края; опора 25–35 мм от пальца — на белой бумаге слева
    img = img.astype(np.uint8)
    book = np.ones((H, W), bool)
    core = np.zeros((H, W), np.uint8)
    cv2.ellipse(core, (1150, 800), (60, 40), 0, 0, 360, 1, -1)
    paint = (cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0) & book
    shadow = estimate_finger_shadow(img, paint, core > 0, book, ~paint, PX_PER_MM)
    out = lighten_finger_shadow(img, shadow)
    assert np.abs(out.astype(int) - img.astype(int))[:, 1060:].max() <= 1


def edge_finger() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Книга до x = 1100, палец у её правого края.

    Returns:
        ``(зона закраса, ядро, книга)``.
    """
    book = np.zeros((H, W), bool)
    book[:, :1100] = True
    core = np.zeros((H, W), np.uint8)
    cv2.ellipse(core, (1130, 800), (60, 40), 0, 0, 360, 1, -1)
    paint = (cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0) & book
    return paint, core > 0, book


def test_sharp_blot_no_rim() -> None:
    """Пятно LaMa с резким краем в зоне закраса снимается без тёмной дуги по краю (МТС 1991/02 IMG_0622)."""
    paint, core, book = edge_finger()
    img = np.tile(PAPER, (H, W, 1))
    blot = np.zeros((H, W), np.uint8)
    cv2.circle(blot, (1090, 800), 45, 1, -1)
    # Тёмное пятно с резким краем: переход ~2 px, как у пятна LaMa (без однопиксельных выступов растровой окружности);
    # краем листа и зоной закраса пятно срезано жёстко.
    soft = cv2.GaussianBlur(blot.astype(np.float32), (0, 0), 1.0) * paint
    blot = (blot > 0) & paint
    img = (img * (1 - 0.45 * soft)[..., None]).astype(np.uint8)
    out = lighten_finger_shadow(img, estimate_finger_shadow(img, paint, core, book, ~paint, PX_PER_MM))
    gray_in, gray_out = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(int) for x in (img, out))
    paper = int(np.median(gray_in[100:300, 100:300]))
    # Полоса ±1 мм вокруг края пятна на листе.
    edge = blot ^ (cv2.erode(blot.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    edge &= book
    assert np.percentile(gray_in[edge], 5) < paper - 40
    # Нет тёмной дуги: темнее бумаги на 15 — не больше 5 % полосы (прежняя карта на ¼ с гладкой B — 59 %; остаток —
    # 1–3 px у верха круга, где пятно срезано краем листа и медиана 1 мм берёт бумагу).
    assert (gray_out[edge] < paper - 15).mean() <= 0.05
    assert np.percentile(gray_out[edge], 98) <= paper + 6  # нет светлого ореола


def edge_band(img: np.ndarray) -> np.ndarray:
    """Серая полоса у края листа (торец, завал): к краю x = 1100 темнее на 30 уровней, ширина ~8 мм.

    Args:
        img: Картинка float32 (H, W, 3), меняется копия.

    Returns:
        Картинка с полосой, float32.
    """
    xx = np.arange(W, dtype=np.float32)
    drop = 30 * np.clip((xx - 1036) / 64, 0, 1) * (xx < 1100)
    return img - drop[None, :, None]


def test_edge_band_not_lightened() -> None:
    """Серая полоса у края листа без тени у пальца не высветляется (МТС 1991/02 IMG_0669)."""
    paint, core, book = edge_finger()
    img = edge_band(np.tile(PAPER, (H, W, 1))).astype(np.uint8)
    out = lighten_finger_shadow(img, estimate_finger_shadow(img, paint, core, book, ~paint, PX_PER_MM))
    band = np.zeros((H, W), bool)
    band[650:950, 1060:1100] = True
    diff = out.astype(int) - img.astype(int)
    assert diff[band].mean() <= 2.0


def test_shadow_in_edge_band_to_band_level() -> None:
    """Тень пальца в серой полосе у края снимается до уровня полосы, а не до белого."""
    paint, core, book = edge_finger()
    clean = edge_band(np.tile(PAPER, (H, W, 1)))
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.exp(-(((xx - 1090) / 40.0) ** 2 + ((yy - 800) / 60.0) ** 2))
    img = (clean * (1 - 0.3 * g)[..., None]).astype(np.uint8)
    out = lighten_finger_shadow(img, estimate_finger_shadow(img, paint, core, book, ~paint, PX_PER_MM))
    gray_clean, gray_in, gray_out = (
        cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(int) for x in (clean, img, out)
    )
    spot = (slice(780, 820), slice(1075, 1095))
    assert (gray_clean[spot] - gray_in[spot]).mean() > 30  # тень есть
    # Снята почти вся, и не светлее полосы без тени больше чем на 5.
    assert np.abs(gray_out[spot] - gray_clean[spot]).mean() <= 6
    assert (gray_out[spot] - gray_clean[spot]).max() <= 5


def test_cover_with_white_edge_strip() -> None:
    """Голубая обложка с белой полосой обреза у края и тёмным пятном LaMa у пальца (МТС 1990/04 IMG_1112): пятно
    снимается до голубого, полоса обреза не меняет цвет, голубое не белеет."""
    cyan = np.array(lab_to_bgr(55, -25, -30), np.float32)
    white = np.array(lab_to_bgr(90, 1, 4), np.float32)
    img = np.tile(PAPER, (H, W, 1))
    img[:, 500:1088] = cyan
    img[:, 1088:1100] = white  # обрез 1.5 мм у края листа
    img[:, 1100:] = cyan * 0.9  # закрас за краем — тон обложки
    paint, core, book = edge_finger()
    # Второй палец — на белой странице слева: порог «печати» по кольцам всего кадра брался бы с белой бумаги и забирал
    # голубую обложку целиком (так было у IMG_1112).
    left = np.zeros((H, W), np.uint8)
    cv2.ellipse(left, (0, 800), (60, 40), 0, 0, 360, 1, -1)
    core = core | (left > 0)
    paint = paint | (cv2.dilate(left, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0)
    blot = np.zeros((H, W), np.uint8)
    cv2.circle(blot, (1060, 800), 35, 1, -1)
    blot = (blot > 0) & paint & (np.arange(W)[None, :] < 1088)
    img[blot] = cyan * 0.45  # тёмно-синее пятно LaMa
    img = img.astype(np.uint8)
    out = lighten_finger_shadow(img, estimate_finger_shadow(img, paint, core, book, ~paint, PX_PER_MM))
    lab_in, lab_out = (cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(int) for x in (img, out))
    strip = np.zeros((H, W), bool)
    strip[600:1000, 1089:1099] = True
    strip &= ~paint
    assert np.abs(lab_out[strip] - lab_in[strip]).max() <= 3  # обрез не меняет цвет
    spot = blot & (cv2.erode(blot.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    target = cv2.cvtColor(cyan.astype(np.uint8)[None, None], cv2.COLOR_BGR2LAB)[0, 0].astype(int)
    assert np.abs(np.median(lab_out[spot], axis=0) - target).max() <= 8  # пятно — до голубого, не до белого


def test_black_spot_lightness_capped() -> None:
    """Почти чёрное пятно в зоне закраса на белой бумаге: светлота растёт не больше потолка MAX_ADD_L — чёрное не
    уходит в белое."""
    paint, core, book = edge_finger()
    img = np.tile(PAPER, (H, W, 1))
    blot = np.zeros((H, W), np.uint8)
    cv2.circle(blot, (1070, 800), 35, 1, -1)
    blot = (blot > 0) & paint
    img[blot] = (12, 12, 12)
    img = img.astype(np.uint8)
    out = lighten_finger_shadow(img, estimate_finger_shadow(img, paint, core, book, ~paint, PX_PER_MM))
    light_in, light_out = (cv2.cvtColor(x, cv2.COLOR_BGR2LAB)[..., 0].astype(float) * 100 / 255 for x in (img, out))
    inner = blot & (cv2.erode(blot.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    assert (light_out[inner] - light_in[inner]).max() <= MAX_ADD_L + 3  # + округление uint8 и Lab
    assert light_out[inner].max() < 80
