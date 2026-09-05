"""Проверки вынесенного расчёта кадра (``processing.smooth_frame``) и развязки радиусов.

Расчёт вынут из ``pipeline.process_frame``, чтобы его мог делить закрас разметки
(``scan_cleanup``). Здесь охраняется то, ради чего вынос делался:

* радиус размытия задаётся независимо от радиуса припуска, но БЕЗ явных опций
  повторяет прежнее ``dilate_px * blur_mult`` число в число;
* область, поданная как защищаемая целиком, остаётся побитово исходной;
* кадр, который трогать нельзя, возвращается сам собой и с названной причиной.
"""

import cv2
import numpy as np
import pytest

from ocr_utils.background_smoothing.processing import (
    BLUR_MODE_PLAIN,
    DEFAULT_BLUR_MULT,
    METHOD_SAUVOLA,
    PROTECT_DILATE_FRAC,
    blur_radius,
    dilate_radius,
    smooth_frame,
)

PAPER = 250
INK = 40
SHAPE = (400, 600)


def text_page(h: int = SHAPE[0], w: int = SHAPE[1], line_step: int = 40, seed: int = 0) -> np.ndarray:
    """Бумага с «строками текста»: тёмные горизонтальные полосы через ``line_step``."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), PAPER, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + rng.integers(-2, 3, size=img.shape), 0, 255).astype(np.uint8)
    for y in range(line_step, h - line_step, line_step):
        img[y : y + 6, 40 : w - 40] = INK
    return img


# ----------------------------------------------------------------------
# Разрешение радиусов
# ----------------------------------------------------------------------


def test_blur_radius_fallback_reproduces_old_behaviour():
    """Без явных опций число в точности прежнее — иначе прошлые прогоны сдвинулись бы."""
    radius = dilate_radius((6100, 3500))
    assert blur_radius((6100, 3500), dilate_px=radius) == pytest.approx(radius * DEFAULT_BLUR_MULT)
    assert radius == pytest.approx(PROTECT_DILATE_FRAC * 6100)


def test_blur_px_wins_over_everything():
    assert blur_radius((6100, 3500), blur_px=120, blur_frac=0.5, dilate_px=25, blur_mult=4.0) == 120


def test_blur_frac_beats_mult_but_yields_to_px():
    assert blur_radius((6100, 3500), blur_frac=0.02, dilate_px=25) == pytest.approx(122.0)


def test_blur_is_independent_of_dilate():
    """Смысл развязки: припуск можно менять, не трогая силу размытия."""
    a = blur_radius((6100, 3500), blur_px=90, dilate_px=15)
    b = blur_radius((6100, 3500), blur_px=90, dilate_px=45)
    assert a == b == 90


def test_dilate_frac_is_used_when_px_is_absent():
    assert dilate_radius((6000, 3000), dilate_frac=0.01) == pytest.approx(60.0)
    assert dilate_radius((6000, 3000), dilate_px=25, dilate_frac=0.01) == 25.0


# ----------------------------------------------------------------------
# Расчёт кадра
# ----------------------------------------------------------------------


def test_content_under_the_mask_is_bit_identical():
    src = text_page()
    res = smooth_frame(src, src, dilate_px=6, blur_px=24)
    assert np.array_equal(res.image[res.m_dilated > 0], src[res.m_dilated > 0])
    assert not res.skip_reason


def test_protect_mask_area_is_bit_identical():
    """Область, защищённая целиком (иллюстрация из базы), не меняется вовсе."""
    src = text_page()
    protect = np.zeros(SHAPE, np.uint8)
    protect[50:150, 300:500] = 255

    res = smooth_frame(src, src, protect_mask=protect, dilate_px=6, blur_px=24)

    assert np.array_equal(res.image[protect > 0], src[protect > 0])
    assert (res.m_dilated[protect > 0] > 0).all()
    # Первичная маска про иллюстрацию ничего не знает — та присоединяется отдельно.
    assert res.m_primary[60, 400] == 0


def test_reported_radii_match_the_request():
    res = smooth_frame(text_page(), text_page(), dilate_px=7, blur_px=21)
    assert (res.dilate_px, res.blur_px) == (7.0, 21.0)


def test_blank_page_is_returned_untouched_with_a_reason():
    blank = np.full(SHAPE, PAPER, np.uint8)
    res = smooth_frame(blank, blank, dilate_px=6, blur_px=24)
    assert res.skip_reason
    assert res.image is blank
    assert not res.m_dilated.any()


def test_halftone_guard_can_be_switched_off():
    """С выключённым предохранителем растровый кадр обрабатывается, а не копируется.

    Это нужно закрасу разметки: там про иллюстрации известно точно, из базы, и
    гадать по гистограмме незачем.
    """
    rng = np.random.default_rng(3)
    src = text_page()
    src[100:300, 100:400] = rng.integers(120, 200, size=(200, 300), dtype=np.uint8)  # «растр»

    assert smooth_frame(src, src, dilate_px=6, blur_px=24).skip_reason
    assert not smooth_frame(src, src, dilate_px=6, blur_px=24, check_halftone=False).skip_reason


def test_plain_mode_blurs_everything():
    src = text_page()
    masked = smooth_frame(src, src, dilate_px=6, blur_px=24)
    plain = smooth_frame(src, src, dilate_px=6, blur_px=24, blur_mode=BLUR_MODE_PLAIN)
    # В режиме plain чернила затягиваются в фон, и фон вне маски проседает.
    background = masked.m_dilated == 0
    assert plain.image[background].mean() < masked.image[background].mean()


def test_colour_frame_keeps_shape():
    gray = text_page()
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    res = smooth_frame(bgr, gray, method=METHOD_SAUVOLA, dilate_px=6, blur_px=24)
    assert res.image.shape == bgr.shape
    assert np.array_equal(res.image[res.m_dilated > 0], bgr[res.m_dilated > 0])


# ----------------------------------------------------------------------
# Чистка мелких областей (despeckle)
# ----------------------------------------------------------------------


def fine_text(h: int = 300, w: int = 400) -> np.ndarray:
    """Набор из тонких штрихов — окрестность, похожая на настоящий текст.

    Важно именно так: у сплошной жирной строки локальное среднее в окне проваливается
    к чернилам, и порог Саволы уходит НИЖЕ бледных пикселей. У настоящего набора
    краски в окне около десятой части, среднее держится у 230, и локальный порог
    садится чуть ниже него — там-то Савола и добирает бледное.
    """
    img = np.full((h, w), PAPER, np.uint8)
    for x in range(20, w - 20, 10):
        img[20 : h - 20, x : x + 1] = INK
    return img


def test_isolated_speck_is_dropped():
    from ocr_utils.background_smoothing.processing import METHOD_OTSU, primary_mask

    img = text_page()
    img[:60, :] = PAPER  # чистое поле сверху
    img[20:24, 300:304] = 210  # пылинка: 16 px, ни к чему не примыкает

    assert primary_mask(img, METHOD_OTSU)[22, 302] == 0, "глобальный порог её не берёт"
    assert primary_mask(img, METHOD_SAUVOLA, min_glyph_area=0)[22, 302] > 0, "без чистки локальный берёт"
    assert primary_mask(img, METHOD_SAUVOLA)[22, 302] == 0, "с чисткой — отбрасывается"


def test_pale_stroke_touching_ink_survives():
    """Бледное, примыкающее к найденной краске, остаётся: ради этого локальный порог и нужен."""
    from ocr_utils.background_smoothing.processing import METHOD_OTSU, primary_mask

    img = fine_text()
    img[100:140, 100:101] = 212  # бледный участок ШТРИХА, продолжает его сверху и снизу

    assert primary_mask(img, METHOD_OTSU, window=41)[120, 100] == 0
    assert primary_mask(img, METHOD_SAUVOLA, window=41)[120, 100] > 0


# ----------------------------------------------------------------------
# despeckle — чистая функция, проверяется напрямую
# ----------------------------------------------------------------------


def speckle_case():
    """Маска: крупная область, мелкая рядом с ней, мелкая одинокая и пара мелких вместе."""
    mask = np.zeros((200, 400), bool)
    strong = np.zeros((200, 400), bool)

    mask[20:40, 20:40] = True  # крупная (400 px), она же прошла глобальный порог
    strong[20:40, 20:40] = True

    mask[25:28, 60:63] = True  # мелкая (9 px) в 20 px от крупной — обломок буквы рядом с буквой
    mask[150:153, 300:303] = True  # мелкая одинокая — пылинка
    mask[150:153, 320:323] = True  # и вторая рядом с ней: скопление крапин
    return mask, strong


def test_big_component_is_kept():
    from ocr_utils.background_smoothing.processing import despeckle

    mask, strong = speckle_case()
    assert despeckle(mask, strong, min_area=48, support_px=25.0)[30, 30]


def test_small_fragment_near_confirmed_is_rescued():
    """Ваш случай: обломок пересвеченной буквы рядом с нормальной буквой."""
    from ocr_utils.background_smoothing.processing import despeckle

    mask, strong = speckle_case()
    assert despeckle(mask, strong, min_area=48, support_px=25.0)[26, 61]


def test_lonely_speck_is_removed():
    from ocr_utils.background_smoothing.processing import despeckle

    mask, strong = speckle_case()
    assert not despeckle(mask, strong, min_area=48, support_px=25.0)[151, 301]


def test_specks_cannot_support_each_other():
    """Скопление крапин не вытягивает себя само.

    Просвет с оборота — призрак СТРОК, крапины в нём стоят кучно. Разреши им
    поддерживать друг друга, и на четырёх полосах выживает 34/31/12/206 крапин
    вместо 12/1/0/28.
    """
    from ocr_utils.background_smoothing.processing import despeckle

    mask, strong = speckle_case()
    kept = despeckle(mask, strong, min_area=48, support_px=25.0)
    assert not kept[151, 301] and not kept[151, 321], "обе крапины рядом друг с другом, но не с краской"


def test_support_distance_is_respected():
    from ocr_utils.background_smoothing.processing import despeckle

    mask, strong = speckle_case()
    # 20 px до крупной области: при допуске 10 обломок уже не спасти.
    assert not despeckle(mask, strong, min_area=48, support_px=10.0)[26, 61]


def test_despeckle_is_identity_for_a_globally_thresholded_mask():
    """Ветка Оцу не затрагивается по построению: там strong совпадает с самой маской."""
    from ocr_utils.background_smoothing.processing import despeckle

    mask, _strong = speckle_case()
    assert np.array_equal(despeckle(mask, mask, min_area=48, support_px=25.0), mask)


def test_zero_min_area_disables_the_cleanup():
    from ocr_utils.background_smoothing.processing import METHOD_OTSU, primary_mask

    img = text_page()
    img[:60, :] = PAPER
    img[20:24, 300:304] = 210
    assert primary_mask(img, METHOD_SAUVOLA, min_glyph_area=0)[22, 302] > 0
