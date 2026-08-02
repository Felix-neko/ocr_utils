"""Проверки защиты растровых иллюстраций по разметке страницы (``--use-surya-layout``).

Сама сеть здесь не запускается: ``LayoutDetector`` подменяется заглушкой, которая
отдаёт заранее известные полигоны. Проверяется то, что от неё зависит, — область
анализа, отбор растровых блоков и поведение пайплайна, а не качество Surya.

Главное, что здесь охраняется: разметка НЕ отменяет детектор растра. Если растр
нашёлся ещё и вне размеченных блоков (обложка, которую сеть не разметила), кадр
обязан дойти до выхода нетронутым.
"""

import cv2
import numpy as np

from ocr_utils.background_smoothing.layout import analysis_roi, is_raster_block, polygons_mask
from ocr_utils.background_smoothing.pipeline import COLOR_PICTURE, SmoothParams, draw_overlay, process_frame
from ocr_utils.background_smoothing.processing import global_threshold, has_halftone

PAPER = 250
INK = 40


class _StubDetector:
    """Заглушка ``LayoutDetector``: отдаёт заданные полигоны, ничего не считая."""

    def __init__(self, *boxes: "tuple[int, int, int, int]") -> None:
        self.polys = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32) for x1, y1, x2, y2 in boxes]

    def picture_polygons(self, bgr, gray=None) -> "list[np.ndarray]":
        return list(self.polys)


def _text_page(h: int = 400, w: int = 600) -> np.ndarray:
    """Страница с текстовыми строками: чернила и бумага, ничего среднего."""
    img = np.full((h, w, 3), PAPER, dtype=np.uint8)
    for y in range(40, h - 40, 40):
        img[y : y + 12, 50 : w - 50] = INK
    return img


def _halftone(shape: "tuple[int, int]", seed: int = 0) -> np.ndarray:
    """Полутоновая фотография: сплошное поле средних тонов с зерном."""
    rng = np.random.default_rng(seed)
    return np.clip(160 + rng.normal(0, 12, shape), 0, 255).astype(np.uint8)


def _write(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return img


def _params(tmp_path, **kw) -> SmoothParams:
    return SmoothParams(input_dir=tmp_path / "in", output_dir=tmp_path / "out", **kw)


class TestRoi:
    """Область анализа: кадр минус иллюстрации."""

    def test_no_polygons_means_whole_frame(self):
        """Иллюстраций не нашлось — ``None``, то есть ровно прежнее поведение без флага."""
        assert analysis_roi(np.zeros((10, 10), np.uint8), []) is None

    def test_polygons_are_cut_out(self):
        """Найденные блоки исключаются из области анализа."""
        polys = [np.array([[2, 2], [6, 2], [6, 6], [2, 6]], np.float32)]
        mask = polygons_mask((10, 10), polys)
        roi = analysis_roi(mask, polys)
        assert roi[4, 4] == 0 and roi[0, 0] == 255

    def test_threshold_ignores_photo(self):
        """Средние тона фотографии не тянут порог Оцу: с ``roi`` он остаётся «текстовым».

        Без исключения блока порог считается по смеси «чернила + бумага + фотография»
        и уезжает вверх — под маску начинает попадать бумага вокруг текста.
        """
        page = cv2.cvtColor(_text_page(), cv2.COLOR_BGR2GRAY)
        page[20:380, 300:580] = _halftone((360, 280))
        polys = [np.array([[300, 20], [580, 20], [580, 380], [300, 380]], np.float32)]
        roi = analysis_roi(polygons_mask(page.shape, polys), polys)

        clean = cv2.cvtColor(_text_page(), cv2.COLOR_BGR2GRAY)
        assert abs(global_threshold(page, roi=roi) - global_threshold(clean)) < 5
        assert global_threshold(page) > global_threshold(page, roi=roi)

    def test_halftone_measured_only_outside(self):
        """Детектор растра, получив ``roi``, не видит уже опознанную фотографию."""
        page = cv2.cvtColor(_text_page(), cv2.COLOR_BGR2GRAY)
        page[20:380, 300:580] = _halftone((360, 280))
        polys = [np.array([[300, 20], [580, 20], [580, 380], [300, 380]], np.float32)]
        roi = analysis_roi(polygons_mask(page.shape, polys), polys)

        assert has_halftone(page) is True
        assert has_halftone(page, roi=roi) is False


class TestRasterBlock:
    """Отбор блоков: класс ``Picture`` Surya ставит и фотографиям, и чертежам."""

    def test_photo_block_is_raster(self):
        page = cv2.cvtColor(_text_page(), cv2.COLOR_BGR2GRAY)
        page[20:380, 300:580] = _halftone((360, 280))
        poly = np.array([[300, 20], [580, 20], [580, 380], [300, 380]], np.float32)
        assert is_raster_block(page, poly) is True

    def test_line_art_block_is_not_raster(self):
        """Чертёж защищать не надо: под сплошной защитой фон внутри рамки остался бы грязным."""
        page = cv2.cvtColor(_text_page(), cv2.COLOR_BGR2GRAY)
        poly = np.array([[50, 50], [550, 50], [550, 350], [50, 350]], np.float32)
        assert is_raster_block(page, poly) is False

    def test_degenerate_block_is_not_raster(self):
        page = cv2.cvtColor(_text_page(), cv2.COLOR_BGR2GRAY)
        assert is_raster_block(page, np.array([[10, 10], [11, 10], [11, 11], [10, 11]], np.float32)) is False


class TestPipelineWithLayout:
    """Поведение кадра целиком: что защищено, что сглажено, что скопировано."""

    def _page_with_photo(self) -> np.ndarray:
        page = _text_page()
        page[20:380, 300:580] = _halftone((360, 280))[:, :, None]
        return page

    def test_photo_page_is_processed_and_photo_survives(self, tmp_path):
        """Со флагом страница «текст + фото» обрабатывается, а фото доходит побитово.

        Без разметки этот же кадр целиком копируется (см. ``test_pipeline``), и «перец»
        вокруг текстовых колонок остаётся — ради этого случая флаг и заводился.
        """
        src = _write(tmp_path / "in" / "page.png", self._page_with_photo())
        detector = _StubDetector((300, 20, 580, 380))
        process_frame(tmp_path / "in" / "page.png", _params(tmp_path), detector)

        out = cv2.imread(str(tmp_path / "out" / "page.png"))
        assert not np.array_equal(out, src), "кадр должен быть обработан, а не скопирован"
        assert np.array_equal(out[20:380, 300:580], src[20:380, 300:580]), "фотография изменилась"

    def test_raster_outside_blocks_still_stops_processing(self, tmp_path):
        """Растр вне размеченных блоков задерживает кадр: страховка на промах разметки.

        Ровно случай непокрытой обложки: сеть разметила одну фотографию, а вторая
        (или пёстрый фон обложки) осталась вне блоков — кадр обязан уйти нетронутым.
        """
        page = self._page_with_photo()
        page[20:380, 20:280] = _halftone((360, 260), seed=1)[:, :, None]
        src = _write(tmp_path / "in" / "page.png", page)
        detector = _StubDetector((300, 20, 580, 380))  # размечена только правая фотография
        process_frame(tmp_path / "in" / "page.png", _params(tmp_path), detector)

        assert np.array_equal(cv2.imread(str(tmp_path / "out" / "page.png")), src)

    def test_empty_detection_matches_run_without_flag(self, tmp_path):
        """Сеть не нашла ничего — результат ровно такой же, как без флага."""
        _write(tmp_path / "in" / "page.png", _text_page())
        process_frame(tmp_path / "in" / "page.png", _params(tmp_path), _StubDetector())
        with_flag = cv2.imread(str(tmp_path / "out" / "page.png"))

        _write(tmp_path / "in2" / "page.png", _text_page())
        params = SmoothParams(input_dir=tmp_path / "in2", output_dir=tmp_path / "out2")
        process_frame(tmp_path / "in2" / "page.png", params)
        assert np.array_equal(with_flag, cv2.imread(str(tmp_path / "out2" / "page.png")))

    def test_overlay_marks_pictures(self, tmp_path):
        """На оверлее блок обведён ярко-сиреневой рамкой — по контуру и только по нему.

        Сравнение не на точный цвет: рамка рисуется со сглаживанием (``LINE_AA``),
        и краевые пиксели подмешаны к фону. Признак — сильный перекос B над G,
        какого нет ни у бумаги, ни у чернил, ни у растра (все они серые).
        """
        page = self._page_with_photo()
        zeros = np.zeros(page.shape[:2], np.uint8)
        overlay = draw_overlay(page, zeros, zeros, _StubDetector((300, 20, 580, 380)).polys)

        violet = (overlay[:, :, 0].astype(int) - overlay[:, :, 1]) > 100
        assert violet.any(), "рамка не нарисована"
        assert not violet[100:300, 350:550].any(), "заливки внутри блока быть не должно"
        assert COLOR_PICTURE[0] > COLOR_PICTURE[1]
