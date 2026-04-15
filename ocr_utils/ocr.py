"""OCR-слой: запуск ocrmypdf на промежуточном PDF для получения финального результата."""

from __future__ import annotations

import logging
from pathlib import Path

import ocrmypdf

from ocr_utils.config import OCR_LANGUAGE, OCR_OVERSAMPLE_DPI

logger = logging.getLogger(__name__)


def run_ocr(
    input_pdf: Path,
    output_pdf: Path,
    language: str = OCR_LANGUAGE,
    oversample_dpi: int = OCR_OVERSAMPLE_DPI,
    deskew: bool = True,
    clean: bool = True,
    rotate_pages: bool = True,
) -> Path:
    """Запустить ocrmypdf на входном PDF для добавления текстового OCR-слоя.

    Аргументы:
        input_pdf: путь к промежуточному (разбитому) PDF.
        output_pdf: путь для сохранения финального PDF с OCR.
        language: код(ы) языка Tesseract.
        oversample_dpi: DPI оверсемплинга перед OCR (выше = лучше качество, но медленнее).
        deskew: выравнивать ли страницы перед OCR.
        clean: очищать ли шум на страницах перед OCR.
        rotate_pages: автоповорот страниц по ориентации текста.

    Возвращает:
        Path к выходному PDF.
    """
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Запуск OCR: %s → %s (lang=%s, dpi=%d)", input_pdf, output_pdf, language, oversample_dpi)

    ocrmypdf.ocr(
        input_file=str(input_pdf),
        output_file=str(output_pdf),
        language=language,
        oversample=oversample_dpi,
        deskew=deskew,
        clean=clean,
        rotate_pages=rotate_pages,
        skip_text=True,  # Не перераспознавать страницы, где уже есть текст
        optimize=0,  # Не пережимать картинки — сохранить оригиналы
        progress_bar=False,
    )

    logger.info("OCR завершён: %s", output_pdf)
    return output_pdf
