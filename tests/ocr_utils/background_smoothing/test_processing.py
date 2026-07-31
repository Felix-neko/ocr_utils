"""Проверки расчёта по кадру: защитная маска и нормированное размытие фона.

Главное, что здесь охраняется:

* контент под защитной маской не меняется ВООБЩЕ (побитово);
* нормированное размытие не затягивает чернила в фон, а обычное — затягивает
  (это ключевое решение подпакета, и без теста оно легко откатывается назад);
* щедрость маски растёт монотонно с ``--threshold-bias``.

Изображения синтетические, на numpy/cv2: «светлая бумага + тёмные штрихи».
"""

import cv2
import numpy as np
import pytest

from ocr_utils.background_smoothing.processing import (
    METHOD_OTSU,
    METHOD_SAUVOLA,
    compose,
    dilate_radius,
    global_threshold,
    has_content,
    has_halftone,
    normalized_blur,
    odd,
    primary_mask,
)
from ocr_utils.scan_cropping.morphology import dilate_disk

PAPER = 250
INK = 40


def _text_page(h: int = 400, w: int = 600, line_step: int = 40, seed: int = 0) -> np.ndarray:
    """Бумага с «строками текста»: тёмные горизонтальные полосы через ``line_step``."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), PAPER, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + rng.integers(-2, 3, size=img.shape), 0, 255).astype(np.uint8)
    for y in range(line_step, h - line_step, line_step):
        img[y : y + 12, 50 : w - 50] = INK
    return img


class TestMask:
    """Первичная маска: порог считается из картинки и лежит между чернилами и бумагой."""

    def test_global_threshold_between_ink_and_paper(self):
        """Порог отделяет чернила от бумаги при любом разумном bias."""
        img = _text_page()
        for bias in (0.0, 0.5, 0.9):
            t = global_threshold(img, bias)
            # На строго бимодальном кадре Оцу садится ровно на уровень чернил, поэтому
            # нижняя граница нестрогая — как и само сравнение в primary_mask.
            assert INK <= t < PAPER

    def test_bias_pushes_threshold_towards_paper(self):
        """Рост bias двигает порог к бумаге и монотонно расширяет маску."""
        img = _text_page()
        thresholds = [global_threshold(img, b) for b in (0.0, 0.3, 0.6, 0.9)]
        assert thresholds == sorted(thresholds)

        areas = [int(np.count_nonzero(primary_mask(img, bias=b))) for b in (0.0, 0.3, 0.6, 0.9)]
        assert areas == sorted(areas)

    def test_mask_covers_ink_and_not_whole_frame(self):
        """Маска накрывает штрихи, но не съедает весь кадр."""
        img = _text_page()
        mask = primary_mask(img, method=METHOD_OTSU)
        assert np.all(mask[img == INK] > 0)
        assert 0 < np.count_nonzero(mask) < mask.size

    def test_blank_page_yields_empty_mask(self):
        """Чистый лист не даёт маски ни одним методом: Оцу иначе разрезал бы зерно пополам.

        Без проверки контраста пустая страница вышла бы пятнистой — половина кадра
        «защищена», половина сглажена.
        """
        rng = np.random.default_rng(1)
        flat = np.clip(np.full((300, 300), PAPER, np.int16) + rng.integers(-3, 4, (300, 300)), 0, 255).astype(np.uint8)
        assert not has_content(flat)
        for method in (METHOD_OTSU, METHOD_SAUVOLA):
            assert np.count_nonzero(primary_mask(flat, method=method)) == 0

    def test_text_page_has_content(self):
        """Страница с текстом контрастна — проверка контраста её не отбрасывает."""
        assert has_content(_text_page())

    def test_halftone_block_is_detected(self):
        """Крупное пятно средних тонов опознаётся как растр — такой кадр не сглаживают."""
        img = _text_page()
        img[100:300, 100:400] = 170
        assert has_halftone(img)

    def test_plain_text_has_no_halftone(self):
        """У текста серое сидит тонкой каймой по краям букв — размыкание её убирает.

        Замер по 1966/03: у всех 96 текстовых страниц доля растра ровно 0.0,
        у двух обложек — 0.05 и 0.10 при пороге 0.01.
        """
        assert not has_halftone(_text_page())

    def test_low_contrast_cover_has_no_content(self):
        """Пёстрая малоконтрастная обложка не считается текстовой страницей.

        Такие кадры идут отдельным трактом, и трогать их этот подпакет не должен.
        """
        h, w = 300, 300
        yy, xx = np.mgrid[0:h, 0:w]
        cover = np.clip(190 + 20 * np.sin(xx / 30.0) + 15 * np.cos(yy / 25.0), 0, 255).astype(np.uint8)
        assert not has_content(cover)

    def test_sauvola_covers_thick_strokes_without_holes(self):
        """Толстые штрихи накрыты целиком, без дыр в середине.

        Внутри крупного тёмного пятна локальное СКО около нуля и порог Саволы уходит
        ниже самих чернил. Пересечение с глобальной маской выгрызло бы там дыры —
        объединение сохраняет штрих сплошным.
        """
        img = _text_page()
        mask = primary_mask(img, method=METHOD_SAUVOLA, window=11)
        assert np.all(mask[img == INK] > 0)

    def test_sauvola_mask_is_superset_of_otsu(self):
        """Локальный порог только добирает контент, ничего не отнимая у глобального."""
        img = _text_page()
        otsu = primary_mask(img, method=METHOD_OTSU) > 0
        sauvola = primary_mask(img, method=METHOD_SAUVOLA) > 0
        assert np.all(sauvola[otsu])

    def test_unknown_method_rejected(self):
        """Неизвестный метод — явная ошибка, а не молчаливый фолбэк."""
        with pytest.raises(ValueError):
            primary_mask(_text_page(), method="niblack")

    def test_dilate_radius_scales_with_frame(self):
        """Радиус припуска берётся из длинной стороны кадра, но явное значение важнее."""
        assert dilate_radius((6096, 3452)) == pytest.approx(15.0, abs=0.5)
        assert dilate_radius((6096, 3452), dilate_px=7) == 7.0

    def test_odd_rounds_up(self):
        assert (odd(4), odd(5), odd(0)) == (5, 5, 1)


class TestSmoothing:
    """Размытие фона: нормировка обязана держать фон на уровне бумаги."""

    def _masks(self, img, radius=6.0):
        m_primary = primary_mask(img)
        return m_primary, dilate_disk(m_primary, radius)

    def test_masked_blur_keeps_background_at_paper_level(self):
        """Нормированное размытие не тянет чернила в фон, обычное — тянет.

        Регрессия на ключевое решение подпакета: на реальном скане (IMG_0130_1L,
        1966/03) обычное размытие проседало до 168-190 при бумаге 253.
        """
        img = _text_page()
        _, m_dilated = self._masks(img)
        bg = m_dilated == 0
        weight = (m_dilated == 0).astype(np.uint8)

        masked = normalized_blur(img, weight, radius_px=24.0)
        plain = normalized_blur(img, None, radius_px=24.0)

        assert np.median(masked[bg]) > PAPER - 5
        assert np.median(plain[bg]) < np.median(masked[bg]) - 10

    def test_masked_blur_has_smaller_seam_jump(self):
        """На шве защитной маски нормированное размытие даёт куда меньший скачок яркости."""
        img = _text_page()
        _, m_dilated = self._masks(img)
        bg = m_dilated == 0
        weight = (m_dilated == 0).astype(np.uint8)

        src = img[bg].astype(np.float32)
        jump_masked = np.abs(src - normalized_blur(img, weight, 24.0)[bg]).mean()
        jump_plain = np.abs(src - normalized_blur(img, None, 24.0)[bg]).mean()
        assert jump_masked < jump_plain

    def test_content_under_mask_is_bit_exact(self):
        """Под защитной маской результат побитово равен исходнику — контент не трогаем вообще."""
        img = _text_page()
        _, m_dilated = self._masks(img)
        blurred = normalized_blur(img, (m_dilated == 0).astype(np.uint8), 24.0)
        final = compose(img, blurred, m_dilated)
        assert np.array_equal(final[m_dilated > 0], img[m_dilated > 0])

    def test_background_ripple_is_suppressed(self):
        """Волнистость фона вне маски подавляется в разы — это и есть цель обработки."""
        h, w = 300, 300
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.clip(PAPER - 12 * np.sin(xx / 4.0) * np.sin(yy / 4.0), 0, 255).astype(np.uint8)
        m_dilated = np.zeros((h, w), np.uint8)  # контента нет — размывается весь кадр

        blurred = normalized_blur(img, (m_dilated == 0).astype(np.uint8), 24.0)
        assert blurred.std() < img.std() / 5

    def test_no_nan_when_support_is_missing(self):
        """Пустой вес не даёт NaN/inf: такие пиксели лежат внутри маски и в результат не идут."""
        img = _text_page()
        blurred = normalized_blur(img, np.zeros(img.shape, np.uint8), 24.0)
        assert np.isfinite(blurred).all()
        assert np.array_equal(blurred, np.zeros_like(blurred))

    def test_color_input_keeps_shape_and_channels(self):
        """Цветной кадр обрабатывается поканально и сохраняет форму."""
        gray = _text_page()
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        _, m_dilated = self._masks(gray)
        blurred = normalized_blur(bgr, (m_dilated == 0).astype(np.uint8), 24.0)
        final = compose(bgr, blurred, m_dilated)
        assert final.shape == bgr.shape
        assert np.array_equal(final[m_dilated > 0], bgr[m_dilated > 0])
