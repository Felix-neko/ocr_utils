"""Растр страницы PDF FineReader и перевод координат «pt слоя ↔ пиксели растра».

Страница пака — JBIG2-образ 600 dpi на всю страницу плюс, если FineReader счёл часть
страницы картинкой (чертёж, схема, а иногда и полоска вертикального текста), отдельные
JBIG2-образы этих кусков, положенные поверх. Поэтому страница РЕНДЕРИТСЯ целиком в родном
разрешении основного образа (``page.get_pixmap``), а не декодируется один объект: иначе
чертёж на растре пропадёт. Рамки дополнительных образов сохраняются как ``figures`` —
это готовый признак «FineReader считает это картинкой» для line art.

Рамка основного образа может чуть ВЫХОДИТЬ за CropBox (замер: 802.9×466.1 pt при странице
802.9×460.3), поэтому родное разрешение считается по ней, а пиксели — по ``page.rect``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import fitz
import numpy as np

from ocr_utils.text_layer_fix import PAGE_DPI


@dataclass(frozen=True)
class PageRaster:
    """Растр страницы: разрешение рендера, размер в пикселях и рамки картинок FineReader (pt)."""

    dpi: float
    width: int
    height: int
    page_rect: fitz.Rect
    main_xref: int
    # Дополнительные образы страницы — куски, которые FineReader вырезал как картинки.
    figures: tuple[fitz.Rect, ...] = field(default_factory=tuple)
    figure_xrefs: tuple[int, ...] = field(default_factory=tuple)

    @property
    def zoom(self) -> float:
        return self.dpi / 72.0

    def to_px(self, dpi: "float | None" = None) -> fitz.Matrix:
        """Матрица перевода координат страницы (pt, fitz) в пиксели растра.

        Args:
            dpi: Разрешение целевого растра; None — разрешение рендера.

        Returns:
            Матрица PyMuPDF: ``rect_pt * matrix`` даёт рамку в пикселях.
        """
        zoom = (dpi or self.dpi) / 72.0
        return fitz.Matrix(zoom, zoom)

    def to_pt(self, dpi: "float | None" = None) -> fitz.Matrix:
        """Обратная матрица: пиксели растра → pt страницы."""
        return ~self.to_px(dpi)


def page_raster(page: fitz.Page) -> PageRaster:
    """Описание растра страницы: родное разрешение основного образа и рамки картинок.

    Args:
        page: Страница PyMuPDF.

    Returns:
        Описание; без картинок на странице разрешение — ``PAGE_DPI``.
    """
    infos = page.get_image_info(xrefs=True)
    if not infos:
        zoom = PAGE_DPI / 72.0
        return PageRaster(
            float(PAGE_DPI), int(round(page.rect.width * zoom)), int(round(page.rect.height * zoom)), page.rect, 0
        )
    main = max(infos, key=lambda info: info["width"] * info["height"])
    bbox = fitz.Rect(main["bbox"])
    dpi = main["width"] / (bbox.width / 72.0) if bbox.width else float(PAGE_DPI)
    zoom = dpi / 72.0
    others = [info for info in infos if info["xref"] != main["xref"]]
    return PageRaster(
        dpi,
        int(round(page.rect.width * zoom)),
        int(round(page.rect.height * zoom)),
        page.rect,
        main["xref"],
        tuple(fitz.Rect(info["bbox"]) for info in others),
        tuple(int(info["xref"]) for info in others),
    )


def render_gray(page: fitz.Page, raster: PageRaster) -> np.ndarray:
    """Серый рендер страницы в разрешении растра (0 — краска, 255 — бумага).

    Args:
        page: Страница PyMuPDF.
        raster: Описание растра из :func:`page_raster`.

    Returns:
        Массив ``uint8`` формы ``(height, width)``.
    """
    pixmap = page.get_pixmap(matrix=raster.to_px(), colorspace=fitz.csGRAY, alpha=False)
    array = np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width)
    return np.ascontiguousarray(array)


def downscale(gray: np.ndarray, factor: float) -> np.ndarray:
    """Уменьшить растр в ``factor`` раз с усреднением (для 600 → 150 dpi это 4).

    Args:
        gray: Серый растр.
        factor: Во сколько раз уменьшить (больше 1).

    Returns:
        Уменьшенный растр ``uint8``.
    """
    if factor <= 1.0 or gray.size == 0:
        return gray
    height, width = gray.shape[:2]
    # Крошечная вырезка (уже ячейки в пиксель) после деления даёт нулевой размер — cv2 падает.
    size = (max(1, int(round(width / factor))), max(1, int(round(height / factor))))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def ink_mask(gray: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Маска краски битональной страницы: глобальный порог, как в ``line_art_detection``.

    Args:
        gray: Серый растр.
        threshold: Порог: темнее — краска.

    Returns:
        Маска ``uint8`` со значениями 0/255.
    """
    return ((gray < threshold).astype(np.uint8)) * 255
