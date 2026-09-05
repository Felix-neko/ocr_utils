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


def test_trust_strong_restores_the_old_behaviour():
    """При ``trust_strong`` однопороговая маска проходит чистку без изменений.

    Это прежнее поведение и страховка: раньше ветка Оцу отсевом не затрагивалась вовсе.
    """
    from ocr_utils.background_smoothing.processing import despeckle

    mask, _strong = speckle_case()
    assert np.array_equal(despeckle(mask, mask, min_area=48, support_px=25.0, trust_strong=True), mask)


def test_strong_mask_is_no_longer_exempt_by_default():
    """Суть изменения: прошедшее глобальный порог больше не спасается автоматически.

    Крапина просвета проходит и сильный порог тоже — на 1976/01 IMG_0052_1L таких
    56.6% пикселей мусора, — поэтому поблажка снята.
    """
    from ocr_utils.background_smoothing.processing import despeckle

    mask, _strong = speckle_case()
    assert not despeckle(mask, mask, min_area=48, support_px=25.0)[71, 151]


def test_zero_min_area_disables_the_cleanup():
    from ocr_utils.background_smoothing.processing import METHOD_OTSU, primary_mask

    img = text_page()
    img[:60, :] = PAPER
    img[20:24, 300:304] = 210
    assert primary_mask(img, METHOD_SAUVOLA, min_glyph_area=0)[22, 302] > 0


# ----------------------------------------------------------------------
# Отражение вместо яркости
# ----------------------------------------------------------------------


def refl_case():
    """Тонкий штрих на ПЕРЕСВЕЧЕННОЙ бумаге и крапина просвета на обычной.

    Кадр 600 px высотой не случайно: константы ``paper.paper_level`` масштабируются по
    высоте, и на кадре в 200 px окно размытия схлопывается до пяти пикселей — уже
    самого штриха, — после чего оценка бумаги садится на штрих и отражение вырождается.
    Штрихи по три пикселя шириной по той же причине: на широкой плашке «raise above
    background» честно считает её саму бумагой.

    Абсолютная яркость их не разделит (180 против 210 при бумаге 255 и 252), отражение
    разделяет: 0.71 против 0.83.
    """
    gray = np.full((600, 400), 252, np.uint8)
    gray[:, :200] = 255  # пересвеченная половина
    gray[100:140, 40:43] = 180  # настоящий, но бледный штрих на пересвете
    gray[100:140, 240:243] = 210  # крапина просвета на обычной бумаге
    mask = np.zeros(gray.shape, bool)
    mask[100:140, 40:43] = True
    mask[100:140, 240:243] = True
    return gray, mask


def test_component_reflectance_matches_the_ratio_to_paper():
    from ocr_utils.background_smoothing.processing import component_reflectance

    labels = np.zeros((10, 10), np.int32)
    labels[0:2, 0:2] = 1
    labels[5:8, 5:9] = 2
    refl = np.ones((10, 10), np.float32)
    refl[0:2, 0:2] = 0.4
    refl[5:8, 5:9] = 0.9

    out = component_reflectance(refl, labels, 3)

    assert out[0] == pytest.approx(1.0), "фон обязан выглядеть чистой бумагой"
    assert out[1] == pytest.approx(0.4)
    assert out[2] == pytest.approx(0.9)


def test_pale_stroke_on_overexposed_paper_is_confirmed():
    """Контрпример из пака: пересвеченная таблица 1966/01 IMG_0047_2R.

    Там уровень бумаги 255, а «краска» 123-193 — светлее крапин просвета на другой
    полосе. Порог по яркости срезал бы её; порог по отражению — нет.
    """
    from ocr_utils.background_smoothing.processing import despeckle
    from ocr_utils.paper import reflectance

    gray, mask = refl_case()
    kept = despeckle(mask, np.zeros_like(mask), reflectance(gray), min_area=34, ink_level=0.75, support_px=0.0)

    assert kept[120, 41], "бледный штрих на пересвете должен остаться"
    assert not kept[120, 241], "крапина просвета — нет"


def test_ink_level_none_falls_back_to_area_only():
    from ocr_utils.background_smoothing.processing import despeckle
    from ocr_utils.paper import reflectance

    gray, mask = refl_case()
    kept = despeckle(mask, np.zeros_like(mask), reflectance(gray), min_area=34, ink_level=None, support_px=0.0)

    assert kept[120, 41] and kept[120, 241], "без отражения обе области крупные и потому подтверждены"


def test_support_works_on_top_of_reflectance():
    """Мелкий бледный обломок рядом с подтверждённой областью остаётся."""
    from ocr_utils.background_smoothing.processing import despeckle
    from ocr_utils.paper import reflectance

    gray, mask = refl_case()
    gray[150:154, 40:43] = 215  # обломок 12 px в десяти пикселях под штрихом
    mask[150:154, 40:43] = True

    kept = despeckle(mask, np.zeros_like(mask), reflectance(gray), min_area=34, ink_level=0.75, support_px=25.0)

    assert kept[152, 41]


def test_auto_ink_level_is_computed_over_the_text_block():
    """Порог по Оцу считается по наборной полосе — краске ВМЕСТЕ с бумагой между строк.

    По одним пикселям краски Оцу поделил бы пополам саму краску: на боевых полосах это
    давало 0.37 вместо 0.59.
    """
    from ocr_utils.background_smoothing.processing import auto_ink_level
    from ocr_utils.paper import reflectance
    from ocr_utils.scan_cropping.morphology import dilate_disk

    # Зерно и мягкие края обязательны: на двухуровневой картинке Оцу вырожден —
    # между двумя пиками любой порог даёт одну и ту же дисперсию, и OpenCV возвращает
    # нижний. У настоящего скана гистограмма непрерывная.
    rng = np.random.default_rng(0)
    gray = np.full((600, 400), 250, np.int16)
    for y in range(100, 500, 40):
        gray[y : y + 4, 40:360] = 40
    gray = cv2.GaussianBlur(gray.astype(np.uint8), (5, 5), 1.2)
    gray = np.clip(gray.astype(np.int16) + rng.integers(-6, 7, size=gray.shape), 0, 255).astype(np.uint8)
    refl = reflectance(gray)
    mask = gray <= 128

    only_ink = auto_ink_level(refl, mask)
    block = dilate_disk(mask.astype(np.uint8) * 255, 25.0) > 0
    with_paper = auto_ink_level(refl, block)

    assert with_paper > only_ink, "по наборной полосе порог обязан быть выше, чем по одной краске"
    assert 0.3 < with_paper < 0.95, "порог обязан сесть между краской и бумагой"


def test_auto_ink_level_falls_back_on_an_empty_block():
    from ocr_utils.background_smoothing.processing import auto_ink_level

    refl = np.ones((50, 50), np.float32)
    assert auto_ink_level(refl, np.zeros((50, 50), bool), fallback=0.65) == 0.65


def test_large_pale_structure_is_confirmed_by_size_alone():
    """Бледная линейка таблицы не должна удаляться за одну лишь бледность.

    Без подтверждения по размеру правило вырождается в «крупная И тёмная», и на
    пересвеченной таблице 1966/01 IMG_0047_2R при строгом пороге удалялась линейка в
    14 731 px — очевидное содержимое.
    """
    from ocr_utils.background_smoothing.processing import despeckle
    from ocr_utils.paper import reflectance

    gray = np.full((600, 400), 255, np.uint8)
    gray[300:303, 20:380] = 200  # длинная бледная линейка, 1080 px
    mask = gray < 255

    refl = reflectance(gray)
    strict = dict(min_area=34, ink_level=0.55, support_px=0.0)

    assert not despeckle(mask, np.zeros_like(mask), refl, sure_area=10**9, **strict)[301, 200]
    assert despeckle(mask, np.zeros_like(mask), refl, sure_area=500, **strict)[301, 200]


def test_sure_area_does_not_rescue_ghost_specks():
    """Крапины просвета столько не набирают: самая крупная удаляемая — 243 px."""
    from ocr_utils.background_smoothing.processing import despeckle
    from ocr_utils.paper import reflectance

    gray = np.full((600, 400), 252, np.uint8)
    gray[300:310, 200:220] = 215  # крапина 200 px, бледная
    mask = gray < 252

    kept = despeckle(
        mask, np.zeros_like(mask), reflectance(gray), min_area=34, ink_level=0.65, sure_area=500, support_px=0.0
    )
    assert not kept[305, 210]
