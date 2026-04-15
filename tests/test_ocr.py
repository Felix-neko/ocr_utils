"""Тесты OCR-функциональности."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from ocr_utils.ocr import run_ocr


class TestOcrBasic:
    """Базовые тесты OCR с реальным Tesseract."""

    def test_ocr_simple_text_page(self, tmp_dir: Path) -> None:
        """OCR должен распознать простой текст на странице."""
        from PIL import Image, ImageDraw, ImageFont
        import io

        img = Image.new("RGB", (800, 600), color="white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except OSError:
            font = ImageFont.load_default()
        draw.text((100, 100), "Привет, мир!", fill="black", font=font)

        jpeg_buf = io.BytesIO()
        img.save(jpeg_buf, format="JPEG", quality=95)
        jpeg_bytes = jpeg_buf.getvalue()

        doc = fitz.open()
        page = doc.new_page(width=800, height=600)
        page.insert_image(fitz.Rect(0, 0, 800, 600), stream=jpeg_bytes)
        input_pdf = tmp_dir / "input.pdf"
        output_pdf = tmp_dir / "output.pdf"
        doc.save(str(input_pdf))
        doc.close()

        result = run_ocr(input_pdf=input_pdf, output_pdf=output_pdf, language="rus", oversample_dpi=300)

        assert result == output_pdf
        assert output_pdf.exists()

        result_doc = fitz.open(str(output_pdf))
        text = result_doc[0].get_text()
        result_doc.close()

        assert "Привет" in text or "мир" in text or "Ilpueer" in text

    def test_ocr_multiple_pages(self, tmp_dir: Path) -> None:
        """OCR должен обработать несколько страниц."""
        doc = fitz.open()
        page1 = doc.new_page(width=595, height=842)
        page1.insert_text((100, 100), "Страница 1", fontsize=20)
        page2 = doc.new_page(width=595, height=842)
        page2.insert_text((100, 100), "Страница 2", fontsize=20)
        page3 = doc.new_page(width=595, height=842)
        page3.insert_text((100, 100), "Страница 3", fontsize=20)

        input_pdf = tmp_dir / "multi.pdf"
        output_pdf = tmp_dir / "multi_ocr.pdf"
        doc.save(str(input_pdf))
        doc.close()

        result = run_ocr(input_pdf=input_pdf, output_pdf=output_pdf, language="rus", oversample_dpi=300)

        assert result == output_pdf
        assert output_pdf.exists()

        result_doc = fitz.open(str(output_pdf))
        assert len(result_doc) == 3
        result_doc.close()

    def test_ocr_preserves_images(self, tmp_dir: Path) -> None:
        """OCR не должен пережимать изображения (optimize=0)."""
        from PIL import Image
        import io

        img = Image.new("RGB", (800, 600), color=(255, 200, 150))
        jpeg_buf = io.BytesIO()
        img.save(jpeg_buf, format="JPEG", quality=95)
        jpeg_bytes = jpeg_buf.getvalue()

        doc = fitz.open()
        page = doc.new_page(width=800, height=600)
        page.insert_image(fitz.Rect(0, 0, 800, 600), stream=jpeg_bytes)

        input_pdf = tmp_dir / "image.pdf"
        output_pdf = tmp_dir / "image_ocr.pdf"
        doc.save(str(input_pdf))
        doc.close()

        run_ocr(input_pdf=input_pdf, output_pdf=output_pdf, language="rus", oversample_dpi=300)

        assert output_pdf.exists()
        input_size = input_pdf.stat().st_size
        output_size = output_pdf.stat().st_size
        assert output_size > input_size * 0.8


class TestOcrOptions:
    """Тесты различных опций OCR."""

    def test_ocr_with_deskew_disabled(self, tmp_dir: Path) -> None:
        """OCR с отключённым выравниванием должен работать."""
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((100, 100), "Тест без выравнивания", fontsize=20)
        input_pdf = tmp_dir / "input.pdf"
        output_pdf = tmp_dir / "output.pdf"
        doc.save(str(input_pdf))
        doc.close()

        result = run_ocr(
            input_pdf=input_pdf, output_pdf=output_pdf, language="rus", oversample_dpi=300, deskew=False
        )

        assert result == output_pdf
        assert output_pdf.exists()

    def test_ocr_with_clean_disabled(self, tmp_dir: Path) -> None:
        """OCR с отключённой очисткой шума должен работать."""
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((100, 100), "Тест без очистки", fontsize=20)
        input_pdf = tmp_dir / "input.pdf"
        output_pdf = tmp_dir / "output.pdf"
        doc.save(str(input_pdf))
        doc.close()

        result = run_ocr(input_pdf=input_pdf, output_pdf=output_pdf, language="rus", oversample_dpi=300, clean=False)

        assert result == output_pdf
        assert output_pdf.exists()

    def test_ocr_with_rotate_disabled(self, tmp_dir: Path) -> None:
        """OCR с отключённым автоповоротом должен работать."""
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((100, 100), "Тест без поворота", fontsize=20)
        input_pdf = tmp_dir / "input.pdf"
        output_pdf = tmp_dir / "output.pdf"
        doc.save(str(input_pdf))
        doc.close()

        result = run_ocr(
            input_pdf=input_pdf, output_pdf=output_pdf, language="rus", oversample_dpi=300, rotate_pages=False
        )

        assert result == output_pdf
        assert output_pdf.exists()

    def test_ocr_with_different_dpi(self, tmp_dir: Path) -> None:
        """OCR с разными DPI должен работать."""
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((100, 100), "Тест DPI", fontsize=20)
        input_pdf = tmp_dir / "input.pdf"
        doc.save(str(input_pdf))
        doc.close()

        for dpi in [150, 300, 600]:
            output_pdf = tmp_dir / f"output_{dpi}.pdf"
            result = run_ocr(input_pdf=input_pdf, output_pdf=output_pdf, language="rus", oversample_dpi=dpi)
            assert result == output_pdf
            assert output_pdf.exists()
