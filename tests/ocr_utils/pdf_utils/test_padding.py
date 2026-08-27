"""Тесты добавления полей к картинкам, экспортируемым из PDF.

Главные свойства, которые проверяем:

* поля к JPEG добавляются **без перекодирования** — вся картинка, кроме внешней кромки
  шириной в один DCT-блок, совпадает с оригиналом пиксель в пиксель (о причине кромки
  см. докстринг :mod:`ocr_utils.pdf_utils.padding`);
* ширина поля округляется вверх до кратной размеру MCU;
* цвет заливки — цвет бумаги, а не среднее по картинке (текст и иллюстрации не должны
  его утягивать), и осветление работает;
* разрешение проставляется только по запросу (--dpi) и только как тег — размер
  картинки в пикселях от него не меняется;
* экспорт идёт пиксель-в-пиксель: размеры не уменьшаются.
"""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image

from ocr_utils.pdf_utils.extract_images import extract_images_from_pdf
from ocr_utils.pdf_utils.padding import (
    JPEG_DCT_BLOCK_SIZE,
    JPEG_MAX_MCU_SIZE,
    TARGET_DPI,
    align_padding_up,
    brighten_color,
    estimate_paper_color,
    jpeg_mcu_size,
    jpeg_padding_step,
    pad_image_array,
    pad_jpeg_lossless,
    read_jpeg_layout,
    set_jpeg_dpi,
)

PAPER_BGR = (205, 220, 235)  # желтоватая бумага в BGR


def _page(width: int = 300, height: int = 420, seed: int = 0) -> np.ndarray:
    """Синтетическая «страница»: поля, наборная полоса со строками и тёмная иллюстрация.

    Поля вокруг набора важны: именно по ним и оценивается цвет бумаги на уменьшенной
    копии, где сами строки текста сливаются в серую массу.
    """
    rng = np.random.default_rng(seed)
    page = np.full((height, width, 3), PAPER_BGR, dtype=np.uint8)
    # лёгкая зернистость бумаги
    noise = rng.integers(-3, 4, size=(height, width, 1), dtype=np.int16)
    page = np.clip(page.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    margin_x, margin_y = width // 8, height // 8
    line_step = max(6, height // 30)
    text_bottom = height - margin_y - height // 4
    for y in range(margin_y, text_bottom, line_step):
        cv2.line(page, (margin_x, y), (width - margin_x, y), (30, 35, 40), max(1, line_step // 4))
    # тёмная иллюстрация внизу — она обязана НЕ утянуть цвет бумаги
    page[text_bottom + line_step : height - margin_y, margin_x : width - margin_x] = (70, 65, 60)
    return page


def _noise(width: int, height: int, seed: int = 3) -> np.ndarray:
    """Некоррелированный шум: любая подмена или пересжатие сразу видны как расхождение."""
    return np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _encode_jpeg(image_bgr: np.ndarray, subsampling: int = 2, quality: int = 95) -> bytes:
    """Закодировать BGR-картинку в JPEG с заданной субдискретизацией."""
    buffer = BytesIO()
    Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).save(
        buffer, format="JPEG", quality=quality, subsampling=subsampling
    )
    return buffer.getvalue()


def _dpi(image: Image.Image) -> tuple[int, int]:
    """Разрешение картинки, округлённое до целого.

    PNG хранит разрешение в точках на метр, поэтому 300 DPI читаются обратно как
    299.9994 — сравнивать надо округлённые значения.
    """
    x, y = image.info["dpi"]
    return round(x), round(y)


def _decode(data: bytes) -> np.ndarray:
    """Декодировать JPEG/PNG в BGR."""
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


class TestPaperColor:
    """Оценка цвета бумаги."""

    def test_ignores_ink_and_illustration(self) -> None:
        """Цвет бумаги, а не среднее по странице: текст и тёмный блок не должны его тянуть."""
        page = _page()
        estimated = estimate_paper_color(page)
        assert np.allclose(estimated, PAPER_BGR, atol=6), estimated

        plain_mean = page.reshape(-1, 3).mean(axis=0)
        assert plain_mean.mean() < np.mean(PAPER_BGR) - 15, "тест бессмысленен: среднее и так равно бумаге"

    def test_dark_illustration_does_not_win(self) -> None:
        """Тёмная иллюстрация в шестую часть листа не должна перебивать бумагу."""
        page = _page()
        page[: page.shape[0] // 6] = (60, 60, 60)
        assert np.allclose(estimate_paper_color(page), PAPER_BGR, atol=8)

    def test_scanner_background_does_not_win(self) -> None:
        """Белый фон сканера по краю не должен выигрывать у бумаги.

        У сканов разворотов вдоль края почти всегда видна подложка сканера: она ярче
        бумаги и идеально гладкая. Пока брался самый светлый тон, заливка уходила
        в холодный почти-белый вместо тёплой бумаги — теперь берётся самый массивный.

        Кайма здесь — 13% площади листа; на реальных сканах пака выходило меньше (около
        10% гладких пикселей против 24% у бумаги).
        """
        page = _page(600, 840)
        border = 24
        page[:border] = (252, 253, 255)
        page[-border:] = (252, 253, 255)
        page[:, :border] = (252, 253, 255)
        page[:, -border:] = (252, 253, 255)

        estimated = estimate_paper_color(page)
        assert np.allclose(estimated, PAPER_BGR, atol=6), estimated

    def test_brightest_tone_loses_to_the_most_common(self) -> None:
        """Из двух гладких тонов побеждает тот, что занимает больше площади, а не тот, что светлее."""
        page = np.full((400, 400, 3), PAPER_BGR, dtype=np.uint8)
        page[:120] = (250, 250, 252)  # светлее, но меньше по площади
        assert np.allclose(estimate_paper_color(page), PAPER_BGR, atol=3)

    def test_grainy_paper_beats_a_smaller_uniform_patch(self) -> None:
        """Зернистая бумага должна побеждать ровную плашку меньшей площади.

        Тон бумаги размазан зернистостью по десятку столбиков гистограммы, а ровная
        заливка стоит в одном — по высоте столбика выигрывала бы плашка. Считаем массу
        в окне, поэтому выигрывает площадь.
        """
        rng = np.random.default_rng(0)
        page = np.full((400, 400, 3), PAPER_BGR, dtype=np.uint8)
        grain = rng.integers(-6, 7, size=(400, 400, 1), dtype=np.int16)
        page = np.clip(page.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        page[:150] = (120, 120, 120)  # идеально ровная плашка на 37% листа

        assert np.allclose(estimate_paper_color(page), PAPER_BGR, atol=4)

    def test_uniform_image(self) -> None:
        """Однотонная картинка: цвет бумаги — она сама."""
        flat = np.full((64, 64, 3), (120, 130, 140), dtype=np.uint8)
        assert np.allclose(estimate_paper_color(flat), (120, 130, 140), atol=2)

    def test_grayscale_input(self) -> None:
        """Картинка в градациях серого не должна ронять оценку."""
        gray = np.full((64, 64), 200, dtype=np.uint8)
        assert np.allclose(estimate_paper_color(gray), (200, 200, 200), atol=2)


class TestBrighten:
    """Осветление цвета заливки."""

    def test_adds_tones(self) -> None:
        assert brighten_color((100, 110, 120), 10) == (110, 120, 130)

    def test_none_and_zero_keep_color(self) -> None:
        assert brighten_color((100, 110, 120), None) == (100, 110, 120)
        assert brighten_color((100, 110, 120), 0) == (100, 110, 120)

    def test_clips_at_255(self) -> None:
        assert brighten_color((250, 200, 10), 10) == (255, 210, 20)


class TestAlignment:
    """Выравнивание ширины поля по сетке MCU."""

    @pytest.mark.parametrize("subsampling, expected", [(0, JPEG_DCT_BLOCK_SIZE), (2, JPEG_MAX_MCU_SIZE)])
    def test_step_matches_subsampling(self, subsampling: int, expected: int) -> None:
        """Без субдискретизации шаг 8 px, при 4:2:0 — 16 px."""
        assert jpeg_padding_step(_encode_jpeg(_page(), subsampling=subsampling)) == expected

    def test_mcu_size(self) -> None:
        assert jpeg_mcu_size(((1, 1), (1, 1), (1, 1))) == (8, 8)
        assert jpeg_mcu_size(((2, 2), (1, 1), (1, 1))) == (16, 16)

    @pytest.mark.parametrize(
        "padding, step, expected", [(64, 16, 64), (65, 16, 80), (1, 16, 16), (0, 16, 0), (9, 8, 16), (8, 8, 8)]
    )
    def test_rounds_up(self, padding: int, step: int, expected: int) -> None:
        assert align_padding_up(padding, step) == expected

    def test_non_multiple_padding_is_rounded_in_result(self) -> None:
        """Некратная ширина округляется вверх, а не отбрасывается."""
        data = _encode_jpeg(_page(), subsampling=2)
        padded, actual = pad_jpeg_lossless(data, 65, PAPER_BGR)
        assert actual == 80
        layout = read_jpeg_layout(padded)
        assert (layout["width"], layout["height"]) == (300 + 160, 420 + 160)


class TestLosslessJpegPadding:
    """Поля к JPEG без перекодирования."""

    @pytest.mark.parametrize("width, height", [(300, 420), (301, 421), (320, 416)])
    def test_bit_exact_without_subsampling(self, width: int, height: int) -> None:
        """Без субдискретизации цветности результат совпадает с оригиналом побитово.

        Размеры взяты и кратные размеру MCU, и некратные: у некратных последний ряд MCU
        неполный, и его легко потерять — ``jpegtran -drop`` переносит только целые MCU.
        """
        data = _encode_jpeg(_noise(width, height), subsampling=0)
        padding = 64
        padded, actual = pad_jpeg_lossless(data, padding, PAPER_BGR)
        assert actual == padding

        original = _decode(data)
        result = _decode(padded)
        assert result.shape[:2] == (height + 2 * padding, width + 2 * padding)
        assert np.array_equal(result[padding : padding + height, padding : padding + width], original)

    @pytest.mark.parametrize("subsampling", [1, 2])
    @pytest.mark.parametrize("width, height", [(300, 420), (301, 421), (320, 416)])
    def test_bit_exact_inside_one_pixel_edge(self, subsampling: int, width: int, height: int) -> None:
        """При 4:2:2/4:2:0 отличается только внешний ряд пикселей — интерполяция цветности."""
        data = _encode_jpeg(_noise(width, height), subsampling=subsampling)
        padding = 64
        padded, _ = pad_jpeg_lossless(data, padding, PAPER_BGR)

        original = _decode(data)
        inserted = _decode(padded)[padding : padding + height, padding : padding + width]
        assert np.array_equal(inserted[1:-1, 1:-1], original[1:-1, 1:-1])

    def test_trailing_partial_mcu_is_not_lost(self) -> None:
        """Правый и нижний край не должны подменяться заливкой.

        Ширина 300 при MCU 16 px даёт неполный последний ряд MCU. Если бы он терялся,
        последние 12 столбцов оказались бы цветом поля.
        """
        width, height = 300, 420
        noise = _noise(width, height)
        data = _encode_jpeg(noise, subsampling=2)
        padded, padding = pad_jpeg_lossless(data, 64, (255, 0, 255))

        original = _decode(data)
        inserted = _decode(padded)[padding : padding + height, padding : padding + width]
        assert np.array_equal(inserted[1:-1, -20:-1], original[1:-1, -20:-1])
        assert np.array_equal(inserted[-20:-1, 1:-1], original[-20:-1, 1:-1])

    def test_padding_is_filled_with_requested_color(self) -> None:
        """Поля залиты заказанным цветом (с точностью до квантования JPEG)."""
        padded, _ = pad_jpeg_lossless(_encode_jpeg(_page()), 64, PAPER_BGR)
        result = _decode(padded)
        assert np.allclose(result[4, 4], PAPER_BGR, atol=4)
        assert np.allclose(result[-4, -4], PAPER_BGR, atol=4)

    def test_grayscale_jpeg(self) -> None:
        """Градации серого: субдискретизации нет, шаг 8 px."""
        buffer = BytesIO()
        Image.fromarray(cv2.cvtColor(_page(), cv2.COLOR_BGR2GRAY)).save(buffer, format="JPEG", quality=90)
        data = buffer.getvalue()
        assert jpeg_padding_step(data) == JPEG_DCT_BLOCK_SIZE
        padded, actual = pad_jpeg_lossless(data, 24, (200, 200, 200))
        assert actual == 24
        assert read_jpeg_layout(padded)["width"] == 300 + 48

    def test_result_carries_requested_dpi(self) -> None:
        padded, _ = pad_jpeg_lossless(_encode_jpeg(_page()), 64, PAPER_BGR, dpi=TARGET_DPI)
        with Image.open(BytesIO(padded)) as image:
            assert _dpi(image) == (TARGET_DPI, TARGET_DPI)

    def test_dpi_is_left_alone_by_default(self) -> None:
        """Без запроса разрешение не трогаем."""
        buffer = BytesIO()
        Image.fromarray(cv2.cvtColor(_page(), cv2.COLOR_BGR2RGB)).save(buffer, format="JPEG", dpi=(72, 72))
        padded, _ = pad_jpeg_lossless(buffer.getvalue(), 64, PAPER_BGR)
        with Image.open(BytesIO(padded)) as image:
            assert _dpi(image) == (72, 72)


class TestQuantTableLayout:
    """Раскладка таблиц квантования по сегментам DQT не должна ломать беспотерьный путь."""

    @staticmethod
    def _merge_dqt(data: bytes) -> bytes:
        """Слепить все сегменты DQT в один — так пишет, например, Photoshop.

        libjpeg кладёт по таблице в отдельный сегмент, Photoshop — все разом в один.
        Картинка от этого не меняется, а вот наивное сравнение сегментов байт в байт
        считает такие файлы несовместимыми.
        """
        from ocr_utils.pdf_utils.padding import _iter_segments

        tables = b""
        out = bytearray(data[:2])
        for marker, start, length in _iter_segments(data):
            segment = data[start - 4 : start + length]
            if marker == 0xDB:
                tables += data[start : start + length]
                continue
            if marker == 0xDA:
                merged = b"\xff\xdb" + (len(tables) + 2).to_bytes(2, "big") + tables
                return bytes(out) + merged + data[start - 4 :]
            out += segment
        raise AssertionError("в JPEG не найден SOS")

    def test_tables_packed_into_one_segment(self) -> None:
        """Все таблицы в одном сегменте DQT — вставка всё равно беспотерьная."""
        data = self._merge_dqt(_encode_jpeg(_noise(320, 416), subsampling=0))
        assert len(read_jpeg_layout(data)["quant_tables"]) == 2

        padded, padding = pad_jpeg_lossless(data, 64, PAPER_BGR)
        original = _decode(data)
        inserted = _decode(padded)[padding : padding + 416, padding : padding + 320]
        assert np.array_equal(inserted, original)

    def test_separate_table_per_component(self) -> None:
        """Своя таблица на каждую компоненту (три вместо двух) — тоже беспотерьно."""
        source = Image.fromarray(cv2.cvtColor(_noise(320, 416), cv2.COLOR_BGR2RGB))
        base = source.copy()
        buffer = BytesIO()
        base.save(buffer, format="JPEG", quality=90, subsampling=0)
        with Image.open(BytesIO(buffer.getvalue())) as probe:
            luma, chroma = probe.quantization[0], probe.quantization[1]

        buffer = BytesIO()
        source.save(buffer, format="JPEG", qtables={0: luma, 1: chroma, 2: chroma}, subsampling=0)
        data = buffer.getvalue()
        layout = read_jpeg_layout(data)
        assert sorted(layout["quant_tables"]) == [0, 1, 2]
        assert layout["component_tables"] == (0, 1, 2)

        padded, padding = pad_jpeg_lossless(data, 64, PAPER_BGR)
        original = _decode(data)
        inserted = _decode(padded)[padding : padding + 416, padding : padding + 320]
        assert np.array_equal(inserted, original)


class TestJpegDpi:
    """Проставление разрешения без перекодирования."""

    def test_none_leaves_bytes_untouched(self) -> None:
        """dpi=None — файл возвращается байт в байт."""
        data = _encode_jpeg(_page())
        assert set_jpeg_dpi(data, None) is data

    def test_overwrites_existing_density(self) -> None:
        buffer = BytesIO()
        Image.fromarray(cv2.cvtColor(_page(), cv2.COLOR_BGR2RGB)).save(buffer, format="JPEG", dpi=(72, 72))
        patched = set_jpeg_dpi(buffer.getvalue(), TARGET_DPI)
        with Image.open(BytesIO(patched)) as image:
            assert _dpi(image) == (TARGET_DPI, TARGET_DPI)

    def test_inserts_jfif_when_missing(self) -> None:
        """У JPEG из PDF сегмента JFIF может не быть — тогда он вставляется."""
        data = _encode_jpeg(_page())
        assert data[2:4] == b"\xff\xe0"
        length = int.from_bytes(data[4:6], "big")
        without_jfif = data[:2] + data[4 + length - 2 :]

        patched = set_jpeg_dpi(without_jfif, TARGET_DPI)
        with Image.open(BytesIO(patched)) as image:
            assert _dpi(image) == (TARGET_DPI, TARGET_DPI)

    def test_entropy_data_untouched(self) -> None:
        """Правка разрешения не трогает сжатые данные."""
        data = _encode_jpeg(_page())
        patched = set_jpeg_dpi(data, TARGET_DPI)
        assert _scan_data(patched) == _scan_data(data)

    def test_patches_exif_resolution(self) -> None:
        """Разрешение в Exif тоже переписывается — иначе его прочитают вместо JFIF."""
        pytest.importorskip("PIL.ExifTags")
        image = Image.fromarray(cv2.cvtColor(_page(), cv2.COLOR_BGR2RGB))
        exif = image.getexif()
        exif[0x011A] = 72.0
        exif[0x011B] = 72.0
        exif[0x0128] = 2
        buffer = BytesIO()
        image.save(buffer, format="JPEG", exif=exif, dpi=(72, 72))

        patched = set_jpeg_dpi(buffer.getvalue(), TARGET_DPI)
        with Image.open(BytesIO(patched)) as result:
            assert result.getexif()[0x011A] == TARGET_DPI
            assert result.getexif()[0x011B] == TARGET_DPI


def _scan_data(jpeg: bytes) -> bytes:
    """Сжатые данные JPEG — всё после маркера SOS."""
    from ocr_utils.pdf_utils.padding import _iter_segments

    for marker, start, length in _iter_segments(jpeg):
        if marker == 0xDA:
            return jpeg[start + length :]
    raise AssertionError("в JPEG не найден SOS")


class TestRasterPadding:
    """Поля на растре (PNG-ветка)."""

    def test_geometry_and_color(self) -> None:
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        padded = pad_image_array(image, 5, (1, 2, 3))
        assert padded.shape == (20, 30, 3)
        assert tuple(padded[0, 0]) == (1, 2, 3)
        assert tuple(padded[10, 10]) == (0, 0, 0)

    def test_zero_padding_is_noop(self) -> None:
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        assert pad_image_array(image, 0, (1, 2, 3)) is image


def _make_pdf(path, images, page_size=(200.0, 280.0)):
    """Собрать PDF, где на каждой странице лежит по одной картинке.

    Страница делается заметно меньше картинки в пунктах (200x280 pt ≈ 72 DPI), чтобы
    поймать экспорт, который рендерит страницу вместо того, чтобы вынуть картинку.

    Args:
        path: Путь к создаваемому PDF
        images: Список байтов картинок (по одной на страницу)
        page_size: Размер страницы в пунктах

    Returns:
        Путь к созданному PDF
    """
    import fitz

    doc = fitz.open()
    for image in images:
        page = doc.new_page(width=page_size[0], height=page_size[1])
        page.insert_image(fitz.Rect(0, 0, *page_size), stream=image)
    doc.save(str(path))
    doc.close()
    return path


class TestExtractPixelExact:
    """Экспорт из PDF: размеры и разрешение."""

    def test_jpeg_is_extracted_pixel_for_pixel(self, tmp_path) -> None:
        """Картинка выходит в своём исходном разрешении, а не в размере страницы."""
        source = _page(600, 840)
        _make_pdf(tmp_path / "in.pdf", [_encode_jpeg(source)])

        assert extract_images_from_pdf(tmp_path / "in.pdf", tmp_path / "out") == 1
        exported = sorted((tmp_path / "out").iterdir())[0]
        result = _decode(exported.read_bytes())
        assert result.shape[:2] == (840, 600)

    def test_jpeg_bytes_are_not_recompressed_without_padding(self, tmp_path) -> None:
        """Без полей сжатые данные JPEG должны остаться прежними."""
        data = _encode_jpeg(_page(600, 840))
        _make_pdf(tmp_path / "in.pdf", [data])

        extract_images_from_pdf(tmp_path / "in.pdf", tmp_path / "out")
        exported = sorted((tmp_path / "out").iterdir())[0]
        assert _scan_data(exported.read_bytes()) == _scan_data(data)

    def test_dpi_option_sets_tag_without_resampling(self, tmp_path) -> None:
        """--dpi меняет только тег: пиксели остаются теми же и того же размера."""
        data = _encode_jpeg(_page(600, 840), subsampling=0)
        _make_pdf(tmp_path / "in.pdf", [data])

        extract_images_from_pdf(tmp_path / "in.pdf", tmp_path / "out", dpi=TARGET_DPI)
        exported = sorted((tmp_path / "out").iterdir())[0]
        with Image.open(exported) as image:
            assert _dpi(image) == (TARGET_DPI, TARGET_DPI)
            assert image.size == (600, 840)
            assert _dpi(image) == (TARGET_DPI, TARGET_DPI)
        assert _scan_data(exported.read_bytes()) == _scan_data(data)

    def test_dpi_is_not_touched_by_default(self, tmp_path) -> None:
        """Без --dpi разрешение исходника сохраняется."""
        buffer = BytesIO()
        Image.fromarray(cv2.cvtColor(_page(600, 840), cv2.COLOR_BGR2RGB)).save(buffer, format="JPEG", dpi=(150, 150))
        _make_pdf(tmp_path / "in.pdf", [buffer.getvalue()])

        extract_images_from_pdf(tmp_path / "in.pdf", tmp_path / "out")
        with Image.open(sorted((tmp_path / "out").iterdir())[0]) as image:
            assert _dpi(image) == (150, 150)

    def test_padding_grows_image_and_keeps_content(self, tmp_path) -> None:
        """С полями картинка растёт ровно на 2*padding, содержимое не трогается."""
        source = _page(600, 840)
        data = _encode_jpeg(source, subsampling=0)
        _make_pdf(tmp_path / "in.pdf", [data])

        extract_images_from_pdf(tmp_path / "in.pdf", tmp_path / "out", padding=64, dpi=TARGET_DPI)
        exported = sorted((tmp_path / "out").iterdir())[0]
        result = _decode(exported.read_bytes())
        assert result.shape[:2] == (840 + 128, 600 + 128)
        assert np.array_equal(result[64:-64, 64:-64], _decode(data))
        with Image.open(exported) as image:
            assert _dpi(image) == (TARGET_DPI, TARGET_DPI)

    def test_padding_is_filled_with_paper_color(self, tmp_path) -> None:
        """Поля залиты цветом бумаги, а не средним по странице."""
        _make_pdf(tmp_path / "in.pdf", [_encode_jpeg(_page(600, 840))])

        extract_images_from_pdf(tmp_path / "in.pdf", tmp_path / "out", padding=64)
        result = _decode(sorted((tmp_path / "out").iterdir())[0].read_bytes())
        assert np.allclose(result[8, 8], PAPER_BGR, atol=6)

    def test_paper_color_survives_draft_decoding(self, tmp_path) -> None:
        """Цвет бумаги оценивается верно и на крупной странице.

        Крупную страницу экспорт распаковывает уменьшенной (draft), и если запросить
        слишком мелкую копию, строки текста сливаются в серую массу — заливка уезжает
        в тень. Страница здесь заведомо больше PAPER_ANALYSIS_MAX_SIDE.
        """
        source = _page(1600, 2200)
        _make_pdf(tmp_path / "in.pdf", [_encode_jpeg(source)])

        extract_images_from_pdf(tmp_path / "in.pdf", tmp_path / "out", padding=64)
        result = _decode(sorted((tmp_path / "out").iterdir())[0].read_bytes())
        assert np.allclose(result[8, 8], PAPER_BGR, atol=6), result[8, 8]

    def test_brighten_lifts_fill_color(self, tmp_path) -> None:
        """Осветление поднимает цвет заливки, не трогая саму картинку."""
        _make_pdf(tmp_path / "in.pdf", [_encode_jpeg(_page(600, 840))])

        extract_images_from_pdf(tmp_path / "in.pdf", tmp_path / "plain", padding=64)
        extract_images_from_pdf(tmp_path / "in.pdf", tmp_path / "bright", padding=64, brighten=10)

        plain = _decode(sorted((tmp_path / "plain").iterdir())[0].read_bytes())
        bright = _decode(sorted((tmp_path / "bright").iterdir())[0].read_bytes())
        assert np.allclose(bright[8, 8].astype(int) - plain[8, 8].astype(int), 10, atol=3)


class TestMrcPageRendering:
    """Страницы с несколькими картинками (MRC) рендерятся целиком."""

    def _mrc_pdf(self, path, size=(600, 840), page_size=(200.0, 280.0)):
        """PDF с одной страницей, на которой лежат две картинки разного разрешения."""
        import fitz

        doc = fitz.open()
        page = doc.new_page(width=page_size[0], height=page_size[1])
        rect = fitz.Rect(0, 0, *page_size)
        page.insert_image(rect, stream=_encode_jpeg(_page(*size)))
        page.insert_image(rect, stream=_encode_jpeg(_page(size[0] // 2, size[1] // 2), quality=60))
        doc.save(str(path))
        doc.close()
        return path

    def test_render_is_not_downscaled(self, tmp_path) -> None:
        """Страница рендерится в разрешение самой крупной из вложенных картинок.

        Раньше DPI рендера округлялся вниз до целого — и страница выходила на пару
        пикселей меньше оригинала.
        """
        self._mrc_pdf(tmp_path / "in.pdf")
        assert extract_images_from_pdf(tmp_path / "in.pdf", tmp_path / "out", dpi=TARGET_DPI) == 1

        exported = sorted((tmp_path / "out").iterdir())[0]
        assert exported.suffix == ".png"
        with Image.open(exported) as image:
            assert image.size == (600, 840)

    def test_render_with_padding(self, tmp_path) -> None:
        """Поля добавляются и к отрендеренной странице."""
        self._mrc_pdf(tmp_path / "in.pdf")
        extract_images_from_pdf(tmp_path / "in.pdf", tmp_path / "out", padding=64, brighten=10, dpi=TARGET_DPI)

        exported = sorted((tmp_path / "out").iterdir())[0]
        with Image.open(exported) as image:
            assert image.size == (600 + 128, 840 + 128)
            assert _dpi(image) == (TARGET_DPI, TARGET_DPI)
        result = _decode(exported.read_bytes())
        assert np.allclose(result[8, 8], brighten_color(PAPER_BGR, 10), atol=8)
