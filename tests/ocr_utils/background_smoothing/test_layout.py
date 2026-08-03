"""Проверки защиты растровых иллюстраций по разметке страницы (``--use-surya-layout``).

Сама сеть здесь не запускается: ``LayoutDetector`` подменяется заглушкой, которая
отдаёт заранее известные полигоны. Проверяется то, что от неё зависит, — область
анализа, отбор растровых блоков, притяжение к реальному растру и поведение
пайплайна, а не качество Surya.

Главное, что здесь охраняется: разметка НЕ отменяет детектор растра. Если растр
нашёлся ещё и вне размеченных блоков (обложка, которую сеть не разметила), кадр
обязан дойти до выхода нетронутым.

Кадры взяты крупными (1200x1600) намеренно. Морфологические ядра подпакета заданы
в пикселях копии 1/4 и рассчитаны на скан 600 dpi; на игрушечном кадре 400x600
ядро смыкания заняло бы треть ширины и склеило бы в одну компоненту всё подряд.
"""

import cv2
import numpy as np

from ocr_utils.background_smoothing.layout import analysis_roi, is_raster_block, polygons_mask, raster_regions
from ocr_utils.background_smoothing.pipeline import (
    COLOR_PICTURE,
    COLOR_RASTER,
    SmoothParams,
    draw_overlay,
    process_frame,
)
from ocr_utils.background_smoothing.processing import global_threshold, has_halftone

PAPER = 250
INK = 40

# Фотография на странице: (y1, y2, x1, x2). Отстоит от края кадра дальше, чем
# половина ядра смыкания (15 px копии 1/4, то есть 60 px кадра), иначе область
# растра расплылась бы до самой рамки — см. ``RASTER_CLOSE_PX`` в ``layout``.
PHOTO = (200, 1000, 800, 1400)


class _StubDetector:
    """Заглушка ``LayoutDetector``: отдаёт заданные полигоны, ничего не считая."""

    def __init__(self, *boxes: "tuple[int, int, int, int]") -> None:
        self.polys = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32) for x1, y1, x2, y2 in boxes]

    def picture_polygons(self, bgr, gray=None) -> "list[np.ndarray]":
        return list(self.polys)


def _poly(y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
    """Полигон-прямоугольник из тех же координат, что и ``PHOTO``."""
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _text_page(h: int = 1200, w: int = 1600) -> np.ndarray:
    """Страница с текстовыми строками: чернила и бумага, ничего среднего."""
    img = np.full((h, w, 3), PAPER, dtype=np.uint8)
    for y in range(120, h - 120, 120):
        img[y : y + 36, 150 : w - 150] = INK
    return img


def _halftone(shape: "tuple[int, int]", seed: int = 0) -> np.ndarray:
    """Полутоновая фотография: сплошное поле средних тонов с зерном."""
    rng = np.random.default_rng(seed)
    return np.clip(160 + rng.normal(0, 12, shape), 0, 255).astype(np.uint8)


def _page_with_photo(rect: "tuple[int, int, int, int]" = PHOTO, seed: int = 0) -> np.ndarray:
    """Текстовая страница с врезанной фотографией."""
    page = _text_page()
    y1, y2, x1, x2 = rect
    page[y1:y2, x1:x2] = _halftone((y2 - y1, x2 - x1), seed)[:, :, None]
    return page


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
        polys = [_poly(2, 6, 2, 6)]
        roi = analysis_roi(polygons_mask((10, 10), polys), polys)
        assert roi[4, 4] == 0 and roi[0, 0] == 255

    def test_threshold_ignores_photo(self):
        """Средние тона фотографии не тянут порог Оцу: с ``roi`` он остаётся «текстовым».

        Без исключения блока порог считается по смеси «чернила + бумага + фотография»
        и уезжает вверх — под маску начинает попадать бумага вокруг текста.
        """
        page = cv2.cvtColor(_page_with_photo(), cv2.COLOR_BGR2GRAY)
        polys = [_poly(*PHOTO)]
        roi = analysis_roi(polygons_mask(page.shape, polys), polys)

        clean = cv2.cvtColor(_text_page(), cv2.COLOR_BGR2GRAY)
        assert abs(global_threshold(page, roi=roi) - global_threshold(clean)) < 5
        assert global_threshold(page) > global_threshold(page, roi=roi)

    def test_halftone_measured_only_outside(self):
        """Детектор растра, получив ``roi``, не видит уже опознанную фотографию."""
        page = cv2.cvtColor(_page_with_photo(), cv2.COLOR_BGR2GRAY)
        polys = [_poly(*PHOTO)]
        roi = analysis_roi(polygons_mask(page.shape, polys), polys)

        assert has_halftone(page) is True
        assert has_halftone(page, roi=roi) is False


class TestRasterBlock:
    """Отбор блоков: класс ``Picture`` Surya ставит и фотографиям, и чертежам."""

    def test_photo_block_is_raster(self):
        page = cv2.cvtColor(_page_with_photo(), cv2.COLOR_BGR2GRAY)
        assert is_raster_block(page, _poly(*PHOTO)) is True

    def test_line_art_block_is_not_raster(self):
        """Чертёж защищать не надо: под сплошной защитой фон внутри рамки остался бы грязным."""
        page = cv2.cvtColor(_text_page(), cv2.COLOR_BGR2GRAY)
        assert is_raster_block(page, _poly(150, 1050, 200, 1400)) is False

    def test_degenerate_block_is_not_raster(self):
        page = cv2.cvtColor(_text_page(), cv2.COLOR_BGR2GRAY)
        assert is_raster_block(page, _poly(10, 11, 10, 11)) is False


class TestPolygonsMask:
    """Заливка полигонов в маску."""

    def test_nested_polygons_are_united_not_xored(self):
        """Вложенный полигон НЕ вычитает объемлющий.

        Регрессия: ``cv2.fillPoly`` со списком контуров считает их одной фигурой и
        заливает по правилу чётности, поэтому прямоугольник растра внутри блока
        Surya взаимно уничтожался с ним — покрытие кадра падало с 48.6% до 3.8%,
        и страница уходила в копирование как «растровая».
        """
        outer, inner = _poly(10, 90, 10, 90), _poly(30, 70, 30, 70)
        mask = polygons_mask((100, 100), [outer, inner])
        assert mask[50, 50] == 255
        assert np.count_nonzero(mask) == np.count_nonzero(polygons_mask((100, 100), [outer]))


class TestRasterRegions:
    """Связные растровые области, которыми дополняются блоки Surya."""

    def test_clipped_block_is_extended_to_the_photo(self):
        """Блок, срезавший верх фотографии, дополняется до её настоящих границ.

        Ровно случай 1967/08 IMG_0094_1L: Surya обвела баннер с заголовком, и верх
        портрета остался снаружи.
        """
        page = cv2.cvtColor(_page_with_photo(), cv2.COLOR_BGR2GRAY)
        y1, y2, x1, x2 = PHOTO
        regions = raster_regions(page, [_poly(y1 + 500, y2, x1, x2)])  # блок накрывает лишь низ
        assert len(regions) == 1
        assert regions[0][:, 1].min() < y1 + 100, "область должна дотянуться до верха фотографии"

    def test_region_stays_inside_a_correct_block(self):
        """У корректного блока добавка лежит внутри него, то есть ничего не меняет."""
        page = cv2.cvtColor(_page_with_photo(), cv2.COLOR_BGR2GRAY)
        y1, y2, x1, x2 = PHOTO
        block = _poly(y1 - 40, y2 + 40, x1 - 40, x2 + 40)
        regions = raster_regions(page, [block])
        assert len(regions) == 1
        pts = regions[0].reshape(-1, 2)
        assert pts[:, 0].min() >= x1 - 40 and pts[:, 1].min() >= y1 - 40
        assert pts[:, 0].max() <= x2 + 40 and pts[:, 1].max() <= y2 + 40

    def test_components_of_one_block_are_merged(self):
        """Разорванная фотография (светлое небо внутри) отдаётся одним прямоугольником.

        По отдельности пятна оставили бы незащищённые прорехи прямо внутри снимка.
        """
        page = _page_with_photo()
        y1, y2, x1, x2 = PHOTO
        page[y1 + 300 : y1 + 500, x1:x2] = PAPER  # светлая полоса поперёк снимка
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        regions = raster_regions(gray, [_poly(*PHOTO)])
        assert len(regions) == 1
        pts = regions[0].reshape(-1, 2)
        assert pts[:, 1].min() < y1 + 300 and pts[:, 1].max() > y1 + 500

    def test_line_art_block_yields_nothing(self):
        """У чертежа растровых компонент нет — добавлять нечего."""
        page = cv2.cvtColor(_text_page(), cv2.COLOR_BGR2GRAY)
        assert raster_regions(page, [_poly(150, 1050, 200, 1400)]) == []

    def test_no_blocks_means_no_work(self):
        assert raster_regions(cv2.cvtColor(_page_with_photo(), cv2.COLOR_BGR2GRAY), []) == []

    def test_raster_far_from_blocks_is_not_taken(self):
        """Растр, не граничащий ни с одним блоком, не защищается: он остаётся страховке.

        Иначе разметка одной фотографии молча узаконила бы вторую, непомеченную, —
        и обложка, которую Surya не разметила, прошла бы обработку.
        """
        page = _page_with_photo()
        page[200:1000, 100:600] = _halftone((800, 500), seed=1)[:, :, None]  # вторая фотография, слева
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        regions = raster_regions(gray, [_poly(*PHOTO)])
        assert len(regions) == 1
        assert regions[0][:, 0].min() > 600, "левая фотография не должна попасть в защиту"


class TestPipelineWithLayout:
    """Поведение кадра целиком: что защищено, что сглажено, что скопировано."""

    def test_photo_page_is_processed_and_photo_survives(self, tmp_path):
        """Со флагом страница «текст + фото» обрабатывается, а фото доходит побитово.

        Без разметки этот же кадр целиком копируется (см. ``test_pipeline``), и «перец»
        вокруг текстовых колонок остаётся — ради этого случая флаг и заводился.
        """
        y1, y2, x1, x2 = PHOTO
        src = _write(tmp_path / "in" / "page.png", _page_with_photo())
        process_frame(tmp_path / "in" / "page.png", _params(tmp_path), _StubDetector((x1, y1, x2, y2)))

        out = cv2.imread(str(tmp_path / "out" / "page.png"))
        assert not np.array_equal(out, src), "кадр должен быть обработан, а не скопирован"
        assert np.array_equal(out[y1:y2, x1:x2], src[y1:y2, x1:x2]), "фотография изменилась"

    def test_photo_outside_a_clipped_block_survives(self, tmp_path):
        """Часть фотографии вне блока Surya всё равно доходит побитово.

        Блок-заглушка срезает верх снимка; его возвращает добавка растровых областей,
        и сглаживание туда не заходит.
        """
        y1, y2, x1, x2 = PHOTO
        src = _write(tmp_path / "in" / "page.png", _page_with_photo())
        process_frame(tmp_path / "in" / "page.png", _params(tmp_path), _StubDetector((x1, y1 + 500, x2, y2)))

        out = cv2.imread(str(tmp_path / "out" / "page.png"))
        assert not np.array_equal(out, src), "кадр должен быть обработан"
        assert np.array_equal(out[y1:y2, x1:x2], src[y1:y2, x1:x2]), "фотография изменилась"

    def test_raster_outside_blocks_still_stops_processing(self, tmp_path):
        """Растр вне размеченных блоков задерживает кадр: страховка на промах разметки.

        Ровно случай непокрытой обложки: сеть разметила одну фотографию, а вторая
        (или пёстрый фон обложки) осталась вне блоков — кадр обязан уйти нетронутым.
        """
        page = _page_with_photo()
        page[200:1000, 100:600] = _halftone((800, 500), seed=1)[:, :, None]
        src = _write(tmp_path / "in" / "page.png", page)
        y1, y2, x1, x2 = PHOTO
        process_frame(tmp_path / "in" / "page.png", _params(tmp_path), _StubDetector((x1, y1, x2, y2)))

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

    def test_overlay_marks_pictures(self):
        """На оверлее блок обведён ярко-сиреневой рамкой — по контуру и только по нему.

        Сравнение не на точный цвет: рамка рисуется со сглаживанием (``LINE_AA``),
        и краевые пиксели подмешаны к фону. Признак — сильный перекос B над G,
        какого нет ни у бумаги, ни у чернил, ни у растра (все они серые).
        """
        page = _page_with_photo()
        y1, y2, x1, x2 = PHOTO
        zeros = np.zeros(page.shape[:2], np.uint8)
        overlay = draw_overlay(page, zeros, zeros, _StubDetector((x1, y1, x2, y2)).polys)

        violet = (overlay[:, :, 0].astype(int) - overlay[:, :, 1]) > 100
        assert violet.any(), "рамка не нарисована"
        assert not violet[y1 + 50 : y2 - 50, x1 + 50 : x2 - 50].any(), "заливки внутри блока быть не должно"
        assert COLOR_PICTURE[0] > COLOR_PICTURE[1]

    def test_overlay_marks_raster_regions_in_green(self):
        """Добавленные растровые области обводятся ярко-зелёным — отдельно от блоков Surya.

        Смысл двух цветов: по расхождению рамок видно, где сеть промахнулась и
        насколько её поправил детектор растра.
        """
        page = _page_with_photo()
        y1, y2, x1, x2 = PHOTO
        zeros = np.zeros(page.shape[:2], np.uint8)
        block = _StubDetector((x1, y1 + 500, x2, y2)).polys  # блок срезал верх снимка
        overlay = draw_overlay(page, zeros, zeros, block, raster_regions(cv2.cvtColor(page, cv2.COLOR_BGR2GRAY), block))

        green = (overlay[:, :, 1].astype(int) - overlay[:, :, 0] > 100) & (
            overlay[:, :, 1].astype(int) - overlay[:, :, 2] > 100
        )
        assert green.any(), "зелёная рамка не нарисована"
        assert green[: y1 + 500].any(), "рамка должна выйти выше блока Surya — туда, где растр"
        assert COLOR_RASTER == (0, 255, 0)
