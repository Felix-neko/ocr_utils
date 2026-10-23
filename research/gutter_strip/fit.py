"""Подгоночные области страниц у полосы-вклейки на стенде: рамки «было» (прежний сплайн v2 по маске до корешка) и «стало» (``scan_cropping.v2.page_frames.spread_frames``: сплайн с узлами на изгибах, у корешка — линия корешка, у полосы — до вспомогательных линий и отрезок через полосу) и расхождение у корешка."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.gutter_strip import Strip
from ocr_utils.scan_cropping.v2.page_boundary import FitKind, PageBoundary, fit_boundary
from ocr_utils.scan_cropping.v2.page_frames import PageFrame, spread_frames
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.pipeline import BEND_CORE_MARGIN_MM
from ocr_utils.scan_cropping.v2.spread_lines import Line
from research.gutter_strip.band import StripFrame

# Зона у корешка, в которой меряется расхождение верха и низа «было | стало», мм от линии корешка.
NEAR_SPINE_MM = 15.0


@dataclass(frozen=True)
class PageFit:
    """Рамки одной страницы.

    Поля:
        before: Прежняя рамка v2 (``fit_boundary`` сплайном по маске, сторона у корешка — по контуру).
        after: Подгоночная область ``spread_frames``.
        spike_mm: Наибольшее расхождение верха и низа «было | стало» по высоте в ``NEAR_SPINE_MM`` у корешка, мм.
    """

    before: PageBoundary
    after: PageFrame
    spike_mm: float


def page_text(sf: StripFrame, mask: np.ndarray) -> np.ndarray:
    """Вершины блоков surya страницы на масштабе сегментации (блок — к странице, в маске которой его центр)."""
    seg = sf.frame.img.shape[0] / sf.full_shape[0]
    h, w = mask.shape
    out = []
    for poly in sf.polys:
        b = np.asarray(poly, np.float64).reshape(-1, 2) * seg
        if mask[min(int(b[:, 1].mean()), h - 1), min(int(b[:, 0].mean()), w - 1)]:
            out.append(b)
    return np.vstack(out) if out else np.zeros((0, 2))


def near_spine_gap(before: PageBoundary, after: PageFrame, spine: Line) -> float:
    """Наибольшее расхождение по высоте верха и низа двух рамок в ``NEAR_SPINE_MM`` у линии корешка, мм."""
    worst = 0.0
    for a, b in ((before.top, after.top), (before.bottom, after.bottom)):
        d = np.abs(spine.x_at(a[:, 1]) - a[:, 0])
        near = a[d <= NEAR_SPINE_MM * SEG_PX_PER_MM]
        if not len(near):
            continue
        order = np.argsort(b[:, 0])
        yb = np.interp(near[:, 0], b[order, 0], b[order, 1])
        worst = max(worst, float(np.abs(near[:, 1] - yb).max()) / SEG_PX_PER_MM)
    return worst


def fit_pages(sf: StripFrame, strip: Strip | None, spine: Line) -> list[PageFit]:
    """Рамки двух страниц кадра «было» и «стало».

    Args:
        sf: Кадр стенда.
        strip: Полоса или None.
        spine: Линия корешка (``gutter_strip.spine_line``).

    Returns:
        По странице слева направо.
    """
    pages = sf.frame.pages
    texts = [page_text(sf, p.mask) for p in pages]
    r = max(int(round(BEND_CORE_MARGIN_MM * SEG_PX_PER_MM)), 1)
    core_zone = (
        cv2.dilate(sf.core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    )
    after = spread_frames(pages, spine, strip, sf.paint, core_zone, texts)
    out = []
    for page, right, text, frame in zip(pages, (True, False), texts, after):
        before = fit_boundary(page.mask, page.quad, right, FitKind.SPLINE, SEG_PX_PER_MM, sf.paint, text)
        out.append(PageFit(before, frame, near_spine_gap(before, frame, spine)))
    return out
