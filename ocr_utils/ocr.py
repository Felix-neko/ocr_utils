"""OCR-слой: двухэтапный подход для максимального качества распознавания без изменения изображений."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import fitz
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
    intermediate_pdf_path: Path | str | None = None,
) -> Path:
    """Запустить OCR с максимальным качеством распознавания, сохраняя исходные изображения.

    Алгоритм:
    1. Создать промежуточный PDF #2 с deskew/clean/rotate для качественного OCR
    2. Извлечь текстовый слой из PDF #2
    3. Наложить текстовый слой на исходные изображения из input_pdf → output_pdf

    Аргументы:
        input_pdf: путь к промежуточному PDF #1 (разбитые страницы, исходные изображения).
        output_pdf: путь для сохранения финального PDF с OCR.
        language: код(ы) языка Tesseract.
        oversample_dpi: DPI оверсемплинга перед OCR (выше = лучше качество).
        deskew: выравнивать ли страницы при OCR (применяется только к промежуточному PDF #2).
        clean: очищать ли шум при OCR (применяется только к промежуточному PDF #2).
        rotate_pages: автоповорот страниц при OCR (применяется только к промежуточному PDF #2).
        intermediate_pdf_path: путь для сохранения промежуточного PDF #2 (после OCR с обработкой).

    Возвращает:
        Path к выходному PDF.
    """
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Запуск двухэтапного OCR: %s → %s (lang=%s, dpi=%d, deskew=%s, clean=%s, rotate=%s)",
        input_pdf,
        output_pdf,
        language,
        oversample_dpi,
        deskew,
        clean,
        rotate_pages,
    )

    if intermediate_pdf_path:
        tmp_ocr_path = Path(intermediate_pdf_path)
        tmp_ocr_path.parent.mkdir(parents=True, exist_ok=True)
        delete_tmp = False
    else:
        tmp_ocr_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_ocr_path = Path(tmp_ocr_file.name)
        tmp_ocr_file.close()
        delete_tmp = True

    try:
        logger.info("Этап 1: OCR с обработкой для качественного распознавания → %s", tmp_ocr_path)
        ocrmypdf.ocr(
            str(input_pdf),
            str(tmp_ocr_path),
            language=language,
            oversample=oversample_dpi,
            deskew=deskew,
            clean=clean,
            rotate_pages=rotate_pages,
            skip_text=True,
            optimize=0,
            progress_bar=False,
        )

        logger.info("Этап 2: Перенос текстового слоя на исходные изображения")
        _transfer_text_layer(input_pdf, tmp_ocr_path, output_pdf)

        logger.info("OCR завершён: %s", output_pdf)
        return output_pdf
    finally:
        if delete_tmp and tmp_ocr_path.exists():
            tmp_ocr_path.unlink()


def _transfer_text_layer(source_images_pdf: Path, ocr_pdf: Path, output_pdf: Path) -> None:
    """Перенести текстовый слой из ocr_pdf на страницы из source_images_pdf.

    Извлекаем текстовые операторы из OCR XObject'ов и встраиваем их напрямую,
    избегая копирования ссылок на отсутствующие изображения.

    Аргументы:
        source_images_pdf: PDF с исходными изображениями (без OCR).
        ocr_pdf: PDF с OCR-слоем (изображения могут быть изменены).
        output_pdf: выходной PDF (изображения из source_images_pdf + текст из ocr_pdf).
    """
    import pikepdf
    from pikepdf import Array, Dictionary, Stream

    src_pdf = pikepdf.open(str(source_images_pdf))
    ocr_pdf_obj = pikepdf.open(str(ocr_pdf))

    if len(src_pdf.pages) != len(ocr_pdf_obj.pages):
        logger.warning(
            "Количество страниц не совпадает: source=%d, ocr=%d. Используем минимум.",
            len(src_pdf.pages),
            len(ocr_pdf_obj.pages),
        )

    for page_num in range(min(len(src_pdf.pages), len(ocr_pdf_obj.pages))):
        src_page = src_pdf.pages[page_num]
        ocr_page = ocr_pdf_obj.pages[page_num]

        text_content_parts = []
        fonts_to_copy = {}

        ocr_contents = ocr_page.get("/Contents")
        if ocr_contents:
            if isinstance(ocr_contents, Stream):
                streams = [ocr_contents]
            elif isinstance(ocr_contents, Array):
                streams = list(ocr_contents)
            else:
                streams = []

            ocr_resources = ocr_page.get("/Resources", Dictionary())

            for stream in streams:
                if not isinstance(stream, Stream):
                    continue

                stream_data = stream.read_bytes()
                stream_text = stream_data.decode("latin-1", errors="ignore")

                if "/OCR-" in stream_text and "Do" in stream_text:
                    xobjects = ocr_resources.get("/XObject", Dictionary())
                    for xobj_name in xobjects.keys():
                        if str(xobj_name).startswith("/OCR-"):
                            xobj = xobjects[xobj_name]
                            if isinstance(xobj, Stream):
                                xobj_data = xobj.read_bytes()
                                xobj_text = xobj_data.decode("latin-1", errors="ignore")

                                xobj_text_filtered = "\n".join(
                                    line
                                    for line in xobj_text.split("\n")
                                    if not ("/Im" in line and "Do" in line)
                                )

                                text_content_parts.append(xobj_text_filtered)

                                xobj_resources = xobj.get("/Resources", Dictionary())
                                if "/Font" in xobj_resources:
                                    for font_name, font_obj in xobj_resources["/Font"].items():
                                        fonts_to_copy[str(font_name)] = font_obj

        if text_content_parts:
            combined_text = "\n".join(text_content_parts)
            text_stream = Stream(src_pdf, combined_text.encode("latin-1"))

            src_contents = src_page.get("/Contents")
            if isinstance(src_contents, Stream):
                src_streams = [src_contents]
            elif isinstance(src_contents, Array):
                src_streams = list(src_contents)
            else:
                src_streams = []

            src_page["/Contents"] = Array(src_streams + [text_stream])

        if fonts_to_copy:
            src_resources = src_page.get("/Resources", Dictionary())
            if "/Font" not in src_resources:
                src_resources["/Font"] = Dictionary()

            for font_name, font_obj in fonts_to_copy.items():
                if font_name not in src_resources["/Font"]:
                    src_resources["/Font"][font_name] = src_pdf.copy_foreign(font_obj)

            src_page["/Resources"] = src_resources

    src_pdf.save(str(output_pdf))
    src_pdf.close()
    ocr_pdf_obj.close()
