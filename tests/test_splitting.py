"""Тесты модуля splitting: детекция разворотов, разбивка на страницы, lossless crop."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from ocr_utils.config import SPREAD_ASPECT_THRESHOLD
from ocr_utils.splitting import PageBox, build_split_pdf, compute_page_boxes, split_pdf_pages


class TestComputePageBoxes:
    """Тесты для compute_page_boxes."""

    def test_portrait_page_not_split(self, portrait_pdf: Path) -> None:
        """Портретная страница не должна разбиваться."""
        doc = fitz.open(str(portrait_pdf))
        boxes = compute_page_boxes(doc)
        doc.close()
        assert len(boxes) == 1
        box = boxes[0]
        assert box.src_page_idx == 0
        assert box.width == pytest.approx(595, abs=1)
        assert box.height == pytest.approx(842, abs=1)

    def test_landscape_spread_split_in_two(self, landscape_spread_pdf: Path) -> None:
        """Альбомный разворот должен быть разбит на 2 страницы."""
        doc = fitz.open(str(landscape_spread_pdf))
        boxes = compute_page_boxes(doc)
        doc.close()
        assert len(boxes) == 2
        # Обе половины ссылаются на страницу 0
        assert boxes[0].src_page_idx == 0
        assert boxes[1].src_page_idx == 0
        # Левая половина
        assert boxes[0].width == pytest.approx(595, abs=1)
        # Правая половина
        assert boxes[1].width == pytest.approx(595, abs=1)
        # Правая начинается там, где кончается левая
        assert boxes[1].x0 == pytest.approx(boxes[0].x1, abs=1)

    def test_mixed_pdf_correct_count(self, mixed_pdf: Path) -> None:
        """Смешанный PDF (портрет + разворот + портрет) → 4 box'а."""
        doc = fitz.open(str(mixed_pdf))
        boxes = compute_page_boxes(doc)
        doc.close()
        assert len(boxes) == 4  # 1 + 2 + 1

    def test_aspect_threshold(self) -> None:
        """Проверить, что порог aspect ratio разумный."""
        assert SPREAD_ASPECT_THRESHOLD == pytest.approx(1.152, abs=0.001)


class TestSplitPdfPages:
    """Тесты для split_pdf_pages."""

    def test_portrait_passthrough(self, portrait_pdf: Path, tmp_dir: Path) -> None:
        """Портретный PDF должен пройти без разбивки (1 стр. → 1 стр.)."""
        result = split_pdf_pages(portrait_pdf, tmp_dir)
        doc = fitz.open(str(result))
        assert len(doc) == 1
        page = doc[0]
        assert page.rect.width == pytest.approx(595, abs=2)
        assert page.rect.height == pytest.approx(842, abs=2)
        doc.close()

    def test_spread_becomes_two_pages(self, landscape_spread_pdf: Path, tmp_dir: Path) -> None:
        """Разворот должен стать двумя страницами."""
        result = split_pdf_pages(landscape_spread_pdf, tmp_dir)
        doc = fitz.open(str(result))
        assert len(doc) == 2
        for i in range(2):
            page = doc[i]
            # Каждая половина — примерно портретная
            assert page.rect.width < page.rect.height or page.rect.width == pytest.approx(page.rect.height, abs=100)
        doc.close()

    def test_mixed_pdf_four_pages(self, mixed_pdf: Path, tmp_dir: Path) -> None:
        """Смешанный PDF → 4 страницы."""
        result = split_pdf_pages(mixed_pdf, tmp_dir)
        doc = fitz.open(str(result))
        assert len(doc) == 4
        doc.close()

    def test_page_selection_with_list(self, mixed_pdf: Path, tmp_dir: Path) -> None:
        """Выбор конкретных страниц списком."""
        # Обрабатываем только страницу 0 (портрет)
        result = split_pdf_pages(mixed_pdf, tmp_dir, pages=[0])
        doc = fitz.open(str(result))
        assert len(doc) == 1
        doc.close()

    def test_page_selection_with_slice(self, mixed_pdf: Path, tmp_dir: Path) -> None:
        """Выбор страниц слайсом."""
        # Обрабатываем страницы 0 и 1 (портрет + разворот = 3 результата)
        result = split_pdf_pages(mixed_pdf, tmp_dir, pages=slice(0, 2))
        doc = fitz.open(str(result))
        assert len(doc) == 3  # 1 (портрет) + 2 (разворот)
        doc.close()

    def test_output_exists(self, portrait_pdf: Path, tmp_dir: Path) -> None:
        """Результат должен существовать на диске."""
        result = split_pdf_pages(portrait_pdf, tmp_dir)
        assert result.exists()
        assert result.stat().st_size > 0


class TestJpegSpread:
    """Тесты для разворотов с JPEG-изображениями."""

    def test_jpeg_spread_split(self, jpeg_spread_pdf: Path, tmp_dir: Path) -> None:
        """Разворот с JPEG должен корректно разбиться на 2 страницы."""
        result = split_pdf_pages(jpeg_spread_pdf, tmp_dir)
        doc = fitz.open(str(result))
        assert len(doc) == 2
        # Каждая страница должна содержать изображение
        for i in range(2):
            images = doc[i].get_images(full=True)
            assert len(images) >= 1
        doc.close()
