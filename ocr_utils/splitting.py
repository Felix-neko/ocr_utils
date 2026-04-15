"""Логика разбивки страниц: детекция разворотов, bounding boxes, lossless JPEG crop, сборка промежуточного PDF."""

from __future__ import annotations

import logging
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import fitz  # PyMuPDF

from ocr_utils.config import JPEGTRAN_BIN, SPREAD_ASPECT_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class PageBox:
    """Прямоугольная область внутри исходной страницы PDF (координаты в PDF-пунктах)."""

    src_page_idx: int
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def compute_page_boxes(doc: fitz.Document) -> list[PageBox]:
    """Проанализировать каждую страницу: портретные — оставить, альбомные развороты — разрезать пополам."""
    boxes: list[PageBox] = []
    for idx in range(len(doc)):
        page = doc[idx]
        rect = page.rect  # fitz.Rect(x0, y0, x1, y1)
        w, h = rect.width, rect.height
        aspect = w / h if h > 0 else 0.0
        if aspect >= SPREAD_ASPECT_THRESHOLD:
            mid_x = rect.x0 + w / 2.0
            boxes.append(PageBox(idx, rect.x0, rect.y0, mid_x, rect.y1))
            boxes.append(PageBox(idx, mid_x, rect.y0, rect.x1, rect.y1))
            logger.debug("Стр. %d (%.0f×%.0f, aspect=%.3f) → разбита на 2 половины", idx, w, h, aspect)
        else:
            boxes.append(PageBox(idx, rect.x0, rect.y0, rect.x1, rect.y1))
            logger.debug("Стр. %d (%.0f×%.0f, aspect=%.3f) → оставлена как есть", idx, w, h, aspect)
    return boxes


# ---------------------------------------------------------------------------
# Вспомогательные функции для JPEG
# ---------------------------------------------------------------------------


def _read_jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Прочитать ширину и высоту JPEG, разбирая SOF-маркеры."""
    i = 0
    if data[i : i + 2] != b"\xff\xd8":
        raise ValueError("Не JPEG")
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0x00 or marker == 0xFF:
            i += 1
            continue
        if marker == 0xD9:  # EOI
            break
        i += 2
        if i + 2 > len(data):
            break
        seg_len = struct.unpack(">H", data[i : i + 2])[0]
        # SOF-маркеры: 0xC0..0xCF, кроме 0xC4 (DHT), 0xC8 (JPG), 0xCC (DAC)
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 7 <= len(data):
                height = struct.unpack(">H", data[i + 3 : i + 5])[0]
                width = struct.unpack(">H", data[i + 5 : i + 7])[0]
                return width, height
        i += seg_len
    raise ValueError("Не удалось найти SOF-маркер в JPEG")


def _jpegtran_crop(jpeg_data: bytes, crop_spec: str, tmp_dir: Path) -> bytes:
    """Запустить jpegtran с lossless crop (padding для неполных MCU-блоков на краях).

    Аргументы:
        jpeg_data: сырые байты JPEG.
        crop_spec: строка кропа для jpegtran, например "320x480+0+0".
        tmp_dir: директория для временных файлов.

    Возвращает:
        Обрезанные JPEG-байты.
    """
    in_path = tmp_dir / "jpegtran_in.jpg"
    out_path = tmp_dir / "jpegtran_out.jpg"
    in_path.write_bytes(jpeg_data)
    cmd = [JPEGTRAN_BIN, "-copy", "all", "-crop", crop_spec, "-outfile", str(out_path), str(in_path)]
    logger.debug("jpegtran: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path.read_bytes()


# ---------------------------------------------------------------------------
# Извлечение и обрезка изображений
# ---------------------------------------------------------------------------


def _extract_page_image(doc: fitz.Document, page_idx: int) -> tuple[bytes, str, int, int]:
    """Извлечь полностраничное изображение из отсканированной PDF-страницы.

    Возвращает (байты_изображения, расширение, ширина_px, высота_px).
    Если на странице одно изображение, покрывающее всю страницу — вернуть его как есть.
    Иначе — отрендерить страницу в pixmap.
    """
    page = doc[page_idx]
    images = page.get_images(full=True)
    if len(images) == 1:
        xref = images[0][0]
        img = doc.extract_image(xref)
        if img:
            return img["image"], img["ext"], img["width"], img["height"]
    # Фоллбэк: рендерить в 300 DPI
    pix = page.get_pixmap(dpi=300)
    return pix.tobytes("png"), "png", pix.width, pix.height


def _crop_image_for_box(
    img_data: bytes, img_ext: str, img_w: int, img_h: int, page_rect: fitz.Rect, box: PageBox, tmp_dir: Path
) -> tuple[bytes, str]:
    """Обрезать изображение по заданному PageBox внутри полной страницы.

    Для JPEG — использовать jpegtran (lossless).
    Для остальных форматов — PyMuPDF pixmap crop.
    """
    # Пересчитать bounding box в пиксельные координаты
    scale_x = img_w / page_rect.width
    scale_y = img_h / page_rect.height
    px_x0 = int((box.x0 - page_rect.x0) * scale_x)
    px_y0 = int((box.y0 - page_rect.y0) * scale_y)
    px_x1 = int((box.x1 - page_rect.x0) * scale_x)
    px_y1 = int((box.y1 - page_rect.y0) * scale_y)
    crop_w = px_x1 - px_x0
    crop_h = px_y1 - px_y0

    if img_ext in ("jpeg", "jpg") and shutil.which(JPEGTRAN_BIN):
        crop_spec = f"{crop_w}x{crop_h}+{px_x0}+{px_y0}"
        try:
            cropped = _jpegtran_crop(img_data, crop_spec, tmp_dir)
            return cropped, "jpg"
        except subprocess.CalledProcessError:
            logger.warning("jpegtran не справился с кропом %s, переходим на pixmap crop", crop_spec)

    # Фоллбэк: обрезка через Pillow
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(img_data))
    cropped = img.crop((px_x0, px_y0, px_x1, px_y1))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue(), "png"


# ---------------------------------------------------------------------------
# Сборка промежуточного PDF
# ---------------------------------------------------------------------------


def build_split_pdf(src_pdf: Path, boxes: list[PageBox], tmp_dir: Path) -> Path:
    """Собрать промежуточный PDF #1 из исходного PDF и списка PageBox.

    Страницы, не требующие разбивки (box покрывает всю страницу), копируются напрямую.
    Для разбитых страниц изображения обрезаются (для JPEG — lossless) и вставляются в новые страницы.

    Возвращает путь к промежуточному PDF.
    """
    src_doc = fitz.open(str(src_pdf))
    out_doc = fitz.open()

    # Кэш изображений страниц (чтобы не извлекать дважды для двух половин разворота)
    page_image_cache: dict[int, tuple[bytes, str, int, int]] = {}

    for box_idx, box in enumerate(boxes):
        src_page = src_doc[box.src_page_idx]
        page_rect = src_page.rect

        # Проверяем, покрывает ли box всю страницу (разбивка не нужна)
        full_page = (
            abs(box.x0 - page_rect.x0) < 1
            and abs(box.y0 - page_rect.y0) < 1
            and abs(box.x1 - page_rect.x1) < 1
            and abs(box.y1 - page_rect.y1) < 1
        )

        if full_page:
            # Копируем страницу как есть
            out_doc.insert_pdf(src_doc, from_page=box.src_page_idx, to_page=box.src_page_idx)
            logger.debug("Box %d: стр. %d скопирована как есть", box_idx, box.src_page_idx)
        else:
            new_w = box.width
            new_h = box.height

            # Пробуем извлечь единственное растровое изображение для lossless crop
            if box.src_page_idx not in page_image_cache:
                page_image_cache[box.src_page_idx] = _extract_page_image(src_doc, box.src_page_idx)

            img_data, img_ext, img_w, img_h = page_image_cache[box.src_page_idx]
            cropped_data, cropped_ext = _crop_image_for_box(img_data, img_ext, img_w, img_h, page_rect, box, tmp_dir)

            # Создаём новую страницу с обрезанным изображением
            new_page = out_doc.new_page(width=new_w, height=new_h)
            img_rect = fitz.Rect(0, 0, new_w, new_h)
            new_page.insert_image(img_rect, stream=cropped_data)
            logger.debug(
                "Box %d: обрезана стр. %d (%s), новая стр. %.0f×%.0f",
                box_idx,
                box.src_page_idx,
                cropped_ext,
                new_w,
                new_h,
            )

    out_path = tmp_dir / "split.pdf"
    out_doc.save(str(out_path), garbage=4, deflate=True)
    out_doc.close()
    src_doc.close()
    logger.info("Собран промежуточный PDF: %d стр., %s", len(boxes), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


def split_pdf_pages(src_pdf: Path, tmp_dir: Path, pages: Sequence[int] | slice | None = None) -> Path:
    """Разбить развороты в PDF на отдельные страницы, записать промежуточный PDF #1 в tmp_dir.

    Аргументы:
        src_pdf: путь к исходному PDF.
        tmp_dir: директория для временных файлов.
        pages: какие исходные страницы обрабатывать.
            None → все страницы.
            list[int] → конкретные 0-based индексы.
            slice → Python-слайс страниц.

    Возвращает:
        Path к промежуточному PDF с разбитыми страницами.
    """
    doc = fitz.open(str(src_pdf))
    total = len(doc)

    if pages is None:
        page_indices = list(range(total))
    elif isinstance(pages, slice):
        page_indices = list(range(*pages.indices(total)))
    else:
        page_indices = list(pages)

    # Собираем поддокумент только из выбранных страниц, затем вычисляем boxes
    if set(page_indices) == set(range(total)):
        sub_doc = doc
    else:
        sub_doc = fitz.open()
        for idx in page_indices:
            sub_doc.insert_pdf(doc, from_page=idx, to_page=idx)
        doc.close()

    boxes = compute_page_boxes(sub_doc)

    if sub_doc is not doc:
        # Сохранить выбранные страницы во временный файл для build_split_pdf
        sub_path = tmp_dir / "selected_pages.pdf"
        sub_doc.save(str(sub_path), garbage=4, deflate=True)
        sub_doc.close()
        result = build_split_pdf(sub_path, boxes, tmp_dir)
    else:
        doc.close()
        result = build_split_pdf(src_pdf, boxes, tmp_dir)

    return result
