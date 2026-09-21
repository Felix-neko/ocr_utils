"""Кромки фотографий: погнутый и перекошенный снимок ловится, целый и снимок с текстом рядом — нет."""

import cv2
import numpy as np

from ocr_utils.geometry_regression.raster import raster_edge_metrics
from tests.ocr_utils.geometry_regression.synthetic import binarize, text_page

DPI = 150
PHOTO = (200, 400, 800, 900)  # рамка снимка на копии 150 dpi: ~100 × 85 мм


def _halftone(page: np.ndarray, box, seed: int = 0) -> np.ndarray:
    """Растровая сетка внутри рамки: точки через 3 px, случайной толщины — как фото в бинарном PDF."""
    out = page.copy()
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = box
    for y in range(y0, y1, 3):
        for x in range(x0, x1, 3):
            if rng.random() < 0.7:
                out[y : y + 2, x : x + 2] = 0
    return out


def _page_with_photo(warp=None) -> np.ndarray:
    """Страница 150 dpi с текстом и снимком; ``warp`` — функция над картинкой снимка (кромка гнётся вместе)."""
    page = cv2.resize(text_page(width=2000, height=3000, lines=30), (1000, 1500), interpolation=cv2.INTER_AREA)
    page = binarize(page)
    x0, y0, x1, y1 = PHOTO
    page[y0 - 40 : y1 + 40, x0 - 40 : x1 + 40] = 255  # бумага вокруг снимка
    page = _halftone(page, PHOTO)
    if warp is not None:
        page = warp(page)
    return page


def _bend_top(page: np.ndarray, amplitude_px: float) -> np.ndarray:
    """Верхняя половина снимка волной: столбцы сдвигаются по y на синусоиду (как FineReader гнёт кромку)."""
    out = np.full_like(page, 255)
    h, w = page.shape
    xs = np.arange(w)
    x0, x1 = PHOTO[0], PHOTO[2]
    shift = np.zeros(w)
    inside = (xs >= x0) & (xs < x1)
    shift[inside] = amplitude_px * np.sin(np.pi * (xs[inside] - x0) / (x1 - x0))
    map_x = np.repeat(xs[None, :], h, axis=0).astype(np.float32)
    map_y = (np.arange(h)[:, None] + shift[None, :]).astype(np.float32)
    return cv2.remap(page, map_x, map_y, cv2.INTER_NEAREST, dst=out, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def test_intact_photo_is_flat():
    page = _page_with_photo()
    metrics, _ = raster_edge_metrics(page, page, [PHOTO], None, DPI)
    assert metrics["raster_edges"] == 4
    assert metrics["raster_edge_bend_mm"] < 0.2
    assert metrics["raster_edge_tilt_mm"] < 0.2


def test_bent_edge_gives_sagitta_in_mm():
    before = _page_with_photo()
    after = _bend_top(before, amplitude_px=12)  # ≈ 2 мм при 150 dpi
    metrics, culprits = raster_edge_metrics(before, after, [PHOTO], None, DPI)
    assert metrics["raster_edges"] >= 2
    assert 1.2 < metrics["raster_edge_bend_mm"] < 2.6
    assert len(culprits["raster_edge_bend_mm"]["segments_a"]) > 5


def test_skewed_photo_gives_tilt_in_mm():
    before = _page_with_photo()
    # Сдвиг: горизонтальные кромки уходят вниз на 15 px (2.5 мм) слева направо — параллелограмм.
    h, w = before.shape
    src = np.float32([[PHOTO[0], PHOTO[1]], [PHOTO[2], PHOTO[1]], [PHOTO[0], PHOTO[3]]])
    dst = np.float32([[PHOTO[0], PHOTO[1]], [PHOTO[2], PHOTO[1] + 15], [PHOTO[0], PHOTO[3]]])
    matrix = cv2.getAffineTransform(src, dst)
    after = cv2.warpAffine(before, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
    metrics, _ = raster_edge_metrics(before, after, [PHOTO], None, DPI)
    assert metrics["raster_edge_tilt_mm"] > 1.5


def test_text_glued_to_photo_does_not_count_as_edge():
    """Строка текста вплотную к верхней кромке: профиль уходит на буквы и режется, кромка ровная."""
    before = _page_with_photo()
    after = before.copy()
    x0, y0, x1, _ = PHOTO
    # «Буквы» — столбики высотой 6 мм, прижатые к правой трети кромки без зазора.
    for x in range(x0 + 400, x1 - 20, 12):
        after[y0 - 36 : y0, x : x + 6] = 0
    metrics, _ = raster_edge_metrics(before, after, [PHOTO], None, DPI)
    assert metrics["raster_edge_bend_mm"] < 0.3
