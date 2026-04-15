"""OCR-слой: двухэтапный подход для максимального качества распознавания без изменения изображений."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import fitz
import ocrmypdf
from PIL import Image
from deskew import determine_skew

from ocr_utils.config import OCR_LANGUAGE, OCR_OVERSAMPLE_DPI

logger = logging.getLogger(__name__)


def prepare_images_for_ocr(input_pdf: Path, output_pdf: Path, target_dpi: int = 600, deskew: bool = False) -> Path:
    """Подготовить изображения для OCR: определить угол наклона, повернуть, сделать oversample.

    Аргументы:
        input_pdf: путь к промежуточному PDF #1 (с исходными изображениями).
        output_pdf: путь для сохранения промежуточного PDF #2 (с подготовленными изображениями).
        target_dpi: целевой DPI для оверсемплинга.
        deskew: определять ли угол наклона и поворачивать изображения.

    Возвращает:
        Path к выходному PDF.
    """
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Подготовка изображений для OCR: %s → %s (target_dpi=%d, deskew=%s)", input_pdf, output_pdf, target_dpi, deskew
    )

    src_doc = fitz.open(str(input_pdf))
    out_doc = fitz.open()

    for page_num in range(len(src_doc)):
        src_page = src_doc[page_num]
        page_rect = src_page.rect

        # Получаем текущий DPI страницы (предполагаем, что изображение занимает всю страницу)
        images = src_page.get_images(full=True)
        if not images:
            logger.warning("Страница %d не содержит изображений, пропускаем", page_num)
            out_doc.insert_pdf(src_doc, from_page=page_num, to_page=page_num)
            continue

        # Берём первое изображение
        xref = images[0][0]
        img_dict = src_doc.extract_image(xref)
        img_data = img_dict["image"]
        img_width = img_dict["width"]
        img_height = img_dict["height"]

        # Вычисляем текущий DPI
        current_dpi_x = img_width / (page_rect.width / 72.0)
        current_dpi_y = img_height / (page_rect.height / 72.0)
        current_dpi = (current_dpi_x + current_dpi_y) / 2.0

        logger.debug("Страница %d: текущий DPI=%.1f, целевой DPI=%d", page_num, current_dpi, target_dpi)

        # Открываем изображение через PIL
        import io

        img = Image.open(io.BytesIO(img_data))

        # Конвертируем в RGB если нужно
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Определяем угол наклона если включен deskew
        angle = 0.0
        if deskew:
            try:
                # Конвертируем в grayscale для определения угла
                img_gray = img.convert("L")
                detected_angle = determine_skew(img_gray)

                if detected_angle is not None:
                    # Ограничиваем угол разумными пределами (±40°)
                    # Углы близкие к ±90° - ошибка определения
                    if abs(detected_angle) > 40.0:
                        logger.warning(
                            "Страница %d: определён подозрительно большой угол %.2f°, игнорируем",
                            page_num,
                            detected_angle,
                        )
                        angle = 0.0
                    elif abs(detected_angle) > 0.1:
                        angle = detected_angle
                        logger.debug("Страница %d: определён угол наклона %.2f°", page_num, angle)
                    else:
                        angle = 0.0
                else:
                    angle = 0.0
            except Exception as e:
                logger.debug("Страница %d: не удалось определить угол наклона: %s", page_num, e)
                angle = 0.0

        # Вычисляем масштаб для оверсемплинга
        scale = target_dpi / current_dpi if current_dpi > 0 else 1.0

        # Применяем трансформации если нужно
        if abs(angle) > 0.1 or abs(scale - 1.0) > 0.01:
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)

            # Сначала масштабируем (bicubic)
            if abs(scale - 1.0) > 0.01:
                img = img.resize((new_width, new_height), Image.BICUBIC)
                logger.debug(
                    "Страница %d: масштабирование %dx%d → %dx%d (scale=%.2f)",
                    page_num,
                    img_width,
                    img_height,
                    new_width,
                    new_height,
                    scale,
                )

            # Затем поворачиваем
            if abs(angle) > 0.1:
                img = img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
                logger.debug("Страница %d: поворот на %.2f°", page_num, angle)

        # Сохраняем изображение в PNG с правильными DPI метаданными
        img_bytes = io.BytesIO()
        # Устанавливаем DPI в метаданных PNG
        img.save(img_bytes, format="PNG", dpi=(target_dpi, target_dpi))
        img_bytes.seek(0)

        # Создаём новую страницу с правильным размером в пунктах
        new_page_width = img.width * 72.0 / target_dpi
        new_page_height = img.height * 72.0 / target_dpi
        new_page = out_doc.new_page(width=new_page_width, height=new_page_height)

        # Вставляем изображение
        new_page.insert_image(new_page.rect, stream=img_bytes.getvalue())

    # Сохраняем PDF
    out_doc.save(str(output_pdf), garbage=4, deflate=True)
    out_doc.close()
    src_doc.close()

    logger.info("Подготовка изображений завершена: %s", output_pdf)
    return output_pdf


def run_ocr(
    input_pdf: Path,
    output_pdf: Path,
    language: str = OCR_LANGUAGE,
    oversample_dpi: int = OCR_OVERSAMPLE_DPI,
    deskew: bool = False,
    clean: bool = True,
    rotate_pages: bool = True,
    intermediate_pdf_path: Path | str | None = None,
    second_intermediate_pdf_path: Path | str | None = None,
) -> Path:
    """Запустить OCR с максимальным качеством распознавания, сохраняя исходные изображения.

    Алгоритм:
    1. Подготовить изображения: deskew (вне ocrmypdf) + oversample → промежуточный PDF #2
    2. Запустить ocrmypdf на PDF #2 с clean/rotate → промежуточный PDF #3
    3. Извлечь текстовый слой из PDF #3
    4. Наложить текстовый слой на исходные изображения из input_pdf → output_pdf

    Аргументы:
        input_pdf: путь к промежуточному PDF #1 (разбитые страницы, исходные изображения).
        output_pdf: путь для сохранения финального PDF с OCR.
        language: код(ы) языка Tesseract.
        oversample_dpi: DPI оверсемплинга перед OCR (выше = лучше качество).
        deskew: определять ли угол наклона и поворачивать изображения (вне ocrmypdf) + делать ли deskew в самом ocrmypdf.
        clean: очищать ли шум при OCR (применяется в ocrmypdf).
        rotate_pages: автоповорот страниц при OCR (применяется в ocrmypdf).
        intermediate_pdf_path: путь для сохранения промежуточного PDF #3 (после OCR с обработкой).
        second_intermediate_pdf_path: путь для сохранения промежуточного PDF #2 (после deskew+oversample).

    Возвращает:
        Path к выходному PDF.
    """
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Запуск трёхэтапного OCR: %s → %s (lang=%s, dpi=%d, deskew=%s, clean=%s, rotate=%s)",
        input_pdf,
        output_pdf,
        language,
        oversample_dpi,
        deskew,
        clean,
        rotate_pages,
    )

    # Подготовка промежуточного PDF #2 (deskew + oversample)
    if second_intermediate_pdf_path:
        tmp_prepared_path = Path(second_intermediate_pdf_path)
        tmp_prepared_path.parent.mkdir(parents=True, exist_ok=True)
        delete_tmp_prepared = False
    else:
        tmp_prepared_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_prepared_path = Path(tmp_prepared_file.name)
        tmp_prepared_file.close()
        delete_tmp_prepared = True

    # Подготовка промежуточного PDF #3 (после ocrmypdf)
    if intermediate_pdf_path:
        tmp_ocr_path = Path(intermediate_pdf_path)
        tmp_ocr_path.parent.mkdir(parents=True, exist_ok=True)
        delete_tmp_ocr = False
    else:
        tmp_ocr_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_ocr_path = Path(tmp_ocr_file.name)
        tmp_ocr_file.close()
        delete_tmp_ocr = True

    try:
        # Этап 1: Подготовка изображений (deskew + oversample)
        logger.info("Этап 1: Подготовка изображений (deskew + oversample) → %s", tmp_prepared_path)
        prepare_images_for_ocr(input_pdf, tmp_prepared_path, target_dpi=oversample_dpi, deskew=deskew)

        # Этап 2: OCR через ocrmypdf (без deskew и oversample)
        logger.info("Этап 2: OCR через ocrmypdf (clean=%s, rotate=%s) → %s", clean, rotate_pages, tmp_ocr_path)
        ocrmypdf.ocr(
            str(tmp_prepared_path),
            str(tmp_ocr_path),
            language=language,
            deskew=deskew,  # Не знаю, почему, но так работает лучше (если и внешнее вращение, и здесь стоит deskew)
            clean=clean,
            rotate_pages=rotate_pages,
            skip_text=True,
            optimize=0,
            progress_bar=False,
        )

        # Этап 3: Перенос текстового слоя на исходные изображения
        logger.info("Этап 3: Перенос текстового слоя на исходные изображения")
        _transfer_text_layer(input_pdf, tmp_ocr_path, output_pdf)

        logger.info("OCR завершён: %s", output_pdf)
        return output_pdf
    finally:
        if delete_tmp_prepared and tmp_prepared_path.exists():
            tmp_prepared_path.unlink()
        if delete_tmp_ocr and tmp_ocr_path.exists():
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
                                    line for line in xobj_text.split("\n") if not ("/Im" in line and "Do" in line)
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
