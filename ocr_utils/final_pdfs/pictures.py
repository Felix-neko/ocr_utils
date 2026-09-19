"""Иллюстрации на финальной странице: вырезка из очищенной полосы, JPEG, место на странице,
снятие образов-фигур FineReader из-под неё.

Источник вырезок — очищенная полоса шага 7 (``blurred/``: печати и надписи закрашены, растр
под защитной маской не размыт), а не заострённая копия: заострение нужно тексту, фото оно
только подчёркивает растровую сетку (решение пользователя 2026-09-19). Файл на диске уже
повёрнут на ``rotate_cw`` полосы, поэтому вырезки берутся по рамкам в координатах файла
(``PagePlan.placed_pictures()``), а место на странице — те же рамки, сдвинутые на поля
промежуточной PDF и переведённые в pt по dpi полосы.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np

from ocr_utils.final_pdfs.plan import PagePlan
from ocr_utils.final_pdfs.sources import Margins
from ocr_utils.pdf_utils.intermediate_pdfs import MM_PER_INCH, PicturePlan
from ocr_utils.pdf_utils.jpeg_pdf import crop, encode_jpeg, load_image

# Образ FineReader считается лежащим под иллюстрацией, если внутри её рамки — от этой доли его площади.
FIGURE_COVER_MIN = 0.8

# Полностраничный растр: иллюстрация занимает от этой доли площади полосы — снимаются все образы страницы.
FULL_PAGE_MIN = 0.97

# Умолчания для пака-1: 300 dpi достаточно для полутонового растра журнальной печати; при
# уменьшении 600 → 300 dpi сетка растра остаётся видимой, Gaussian σ = 0.05 мм (≈1.2 px при
# 600 dpi) гасит её, не размывая деталей (проба на 4 врезках, 2026-09-19); JPEG 75 — решение
# пользователя (сессия 2026-09-06).
DEFAULT_PICTURE_DPI = 300
DEFAULT_JPEG_QUALITY = 75
DEFAULT_DESCREEN_SIGMA_MM = 0.05


@dataclass(frozen=True)
class PictureJpeg:
    """Готовая иллюстрация: байты JPEG, цветность и место в координатах файла полосы (px, без полей)."""

    jpeg: bytes
    gray: bool
    rect_px: tuple[int, int, int, int]
    width: int  # размер JPEG в пикселях
    height: int
    kind: str
    full_page: bool


def resample(array: np.ndarray, src_dpi: int, target_dpi: int, descreen_sigma_mm: float) -> np.ndarray:
    """Уменьшить вырезку до целевого dpi: опциональное Gaussian-предразмытие, затем INTER_AREA.

    Args:
        array: Вырезка (``H×W`` или ``H×W×3``).
        src_dpi: Разрешение вырезки.
        target_dpi: Целевое разрешение; не меньше исходного — вырезка не меняется.
        descreen_sigma_mm: σ Gaussian по исходному растру, мм бумаги; 0 — без размытия.

    Returns:
        Уменьшенная вырезка ``uint8``.
    """
    if target_dpi >= src_dpi or array.size == 0:
        return array
    if descreen_sigma_mm > 0:
        array = cv2.GaussianBlur(array, (0, 0), descreen_sigma_mm / MM_PER_INCH * src_dpi)
    height, width = array.shape[:2]
    scale = target_dpi / src_dpi
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(array, size, interpolation=cv2.INTER_AREA)


def render_pictures(
    source_path: Path,
    page_plan: PagePlan,
    target_dpi: int = DEFAULT_PICTURE_DPI,
    quality: int = DEFAULT_JPEG_QUALITY,
    descreen_sigma_mm: float = DEFAULT_DESCREEN_SIGMA_MM,
) -> list[PictureJpeg]:
    """Все иллюстрации полосы из её очищенной копии.

    Args:
        source_path: Очищенная полоса (TIFF 600 dpi из ``blurred/``, уже повёрнутая на ``rotate_cw``).
        page_plan: Полоса из базы с рамками иллюстраций.
        target_dpi: Разрешение вставляемых JPEG.
        quality: Качество JPEG.
        descreen_sigma_mm: σ предразмытия против растровой сетки, мм; 0 — выключено.

    Returns:
        Список :class:`PictureJpeg` в порядке разметки.

    Raises:
        ValueError: Размер файла не равен размеру полосы (с учётом поворота) из базы.
    """
    if not page_plan.pictures:
        return []
    image = load_image(source_path)
    width, height = page_plan.file_size
    if image.shape[1] != width or image.shape[0] != height:
        raise ValueError(
            f"{source_path}: размер {image.shape[1]}×{image.shape[0]}, а в базе полоса "
            f"{width}×{height} (rotate_cw={page_plan.rotate_cw})"
        )
    result: list[PictureJpeg] = []
    for picture in page_plan.placed_pictures():
        full_page = is_full_page(picture, width, height)
        if full_page:
            # Обложка: страница будет обрезана до полосы, и JPEG должен покрыть её целиком —
            # режем всю полосу, а не рамку региона (та бывает на пару пикселей меньше).
            picture = PicturePlan(0, 0, width, height, picture.kind)
        part = crop(image, picture.rect)
        if picture.gray and part.ndim == 3:
            part = cv2.cvtColor(part, cv2.COLOR_RGB2GRAY)
        part = resample(part, page_plan.dpi, target_dpi, descreen_sigma_mm)
        result.append(
            PictureJpeg(
                jpeg=encode_jpeg(part, picture.gray, quality),
                gray=picture.gray,
                rect_px=picture.rect,
                width=part.shape[1],
                height=part.shape[0],
                kind=picture.kind,
                full_page=full_page,
            )
        )
    return result


def is_full_page(picture: PicturePlan, width: int, height: int) -> bool:
    """Занимает ли иллюстрация (почти) всю полосу — тогда бинарный растр под ней не нужен вовсе."""
    area = max(1, (picture.x2 - picture.x1) * (picture.y2 - picture.y1))
    return area >= FULL_PAGE_MIN * width * height


def placement_rect(rect_px: tuple[int, int, int, int], margins: Margins, dpi: int) -> fitz.Rect:
    """Место иллюстрации на странице в pt: рамка в файле полосы + поля промежуточной PDF.

    Args:
        rect_px: Рамка в пикселях файла полосы (без полей).
        margins: Поля промежуточной PDF в пикселях.
        dpi: Разрешение полосы.

    Returns:
        Прямоугольник PyMuPDF (начало координат — левый верхний угол страницы).
    """
    scale = 72.0 / dpi
    x1, y1, x2, y2 = rect_px
    return fitz.Rect(
        (x1 + margins.x_px) * scale,
        (y1 + margins.y_px) * scale,
        (x2 + margins.x_px) * scale,
        (y2 + margins.y_px) * scale,
    )


def figures_under(page: fitz.Page, rects: list[fitz.Rect], full_page: bool) -> list[int]:
    """xref образов страницы, которые лежат под иллюстрациями и подлежат снятию.

    Args:
        page: Страница (до вставки своих JPEG).
        rects: Места иллюстраций в pt.
        full_page: Есть полностраничная иллюстрация — снимаются все образы.

    Returns:
        Список xref (без повторов, в порядке появления).
    """
    xrefs: list[int] = []
    for info in page.get_image_info(xrefs=True):
        box = fitz.Rect(info["bbox"])
        if box.is_empty:
            continue
        covered = full_page or any((box & rect).get_area() >= FIGURE_COVER_MIN * box.get_area() for rect in rects)
        if covered and info["xref"] not in xrefs:
            xrefs.append(int(info["xref"]))
    return xrefs


def remove_images(doc: fitz.Document, page: fitz.Page, xrefs: list[int]) -> int:
    """Снять образы со страницы: убрать их ``Do`` из потока содержимого и ключ из ресурсов.

    Убирается только оператор ``/Имя Do`` (маркированное содержимое, ``cm`` и клип вокруг
    остаются: они ничего не рисуют, а дерево структуры продолжает ссылаться на свои MCID).
    Сам объект образа остаётся без ссылок и выбрасывается при ``save(garbage=…)``.

    Args:
        doc: Документ.
        page: Страница с одним потоком содержимого.
        xrefs: xref образов.

    Returns:
        Сколько операторов ``Do`` убрано.
    """
    if not xrefs:
        return 0
    names = [name for xref, name in _xobject_names(doc, page) if xref in xrefs]
    if not names:
        return 0
    streams = page.get_contents()
    removed = 0
    for stream_xref in streams:
        raw = doc.xref_stream(stream_xref)
        pattern = re.compile(rb"/(" + b"|".join(re.escape(n.encode()) for n in names) + rb")\s+Do\b")
        edited, count = pattern.subn(b"", raw)
        if count:
            doc.update_stream(stream_xref, edited)
            removed += count
    for name in names:
        doc.xref_set_key(page.xref, f"Resources/XObject/{name}", "null")
    return removed


def _xobject_names(doc: fitz.Document, page: fitz.Page) -> list[tuple[int, str]]:
    """Пары (xref, имя ресурса) образов страницы."""
    return [(int(info[0]), str(info[7])) for info in page.get_images(full=True)]


def insert_picture(doc: fitz.Document, page: fitz.Page, picture: PictureJpeg, rect: fitz.Rect) -> int:
    """Положить JPEG на страницу верхним слоем как есть (DCTDecode, без перекодирования).

    Args:
        doc: Документ.
        page: Страница.
        picture: Готовая иллюстрация.
        rect: Место на странице в pt.

    Returns:
        xref вставленного образа.
    """
    xref = page.insert_image(rect, stream=picture.jpeg, keep_proportion=False)
    # PyMuPDF заворачивает JPEG в ICCBased; устройственное пространство короче и однозначно
    # говорит просмотрщику «один канал» / «три канала».
    doc.xref_set_key(xref, "ColorSpace", "/DeviceGray" if picture.gray else "/DeviceRGB")
    return xref


def crop_page(page: fitz.Page, rect: fitz.Rect) -> None:
    """Обрезать страницу до рамки: MediaBox = рамка (CropBox по умолчанию равен ему — полей нет
    ни в одном просмотрщике).

    Args:
        page: Страница.
        rect: Рамка в координатах fitz страницы (y вниз); ``set_mediabox`` ждёт координаты PDF,
            поэтому рамка переводится обратной матрицей страницы.
    """
    page.set_mediabox(rect * ~page.transformation_matrix)
