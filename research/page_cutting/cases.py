"""Кадр прогона v2 для стенда: маски страниц, трапеции, зона пальца и текст surya на масштабе сегментации, рамка итогового TIFF в кадре и сам итог."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.fill import paper_bbox
from ocr_utils.scan_cropping.v2.page_pose import frame_pages
from ocr_utils.scan_cropping.v2.overlay import OverlayTag, overlay_path
from ocr_utils.scan_cropping.v2.pipeline import read_layout, read_layout_labels, to_frame
from ocr_utils.scan_cropping.v2.trapezoid import fit_enclosing_quad
from research.page_cutting.fold import Fold, find_fold, page_spine_edge, split_by_fold, trim_at_edge
from research.page_cutting.runs import RunSpec, read_image, rel_of
from research.page_cutting.selection import TEXT_LABELS, page_quads_of


@dataclass(frozen=True)
class PageInput:
    """Страница кадра на масштабе сегментации.

    Поля:
        mask: Маска страницы.
        quad: Трапеция v2 (4, 2) px сегментации.
        spine_right: Корешок справа (левая страница разворота) — True, слева — False, страница одна — None.
        text: Вершины текстовых блоков surya страницы (N, 2), px сегментации.
        fold_mask: Маска страницы, перерезанная по найденному сгибу (или маска v2, если сгиба нет).
        fold_quad: Трапеция маски ``fold_mask``.
        spine_edge: Край страницы у корешка (N, 2) px сегментации — где кончается цвет её бумаги по дороге к сгибу
            (сам сгиб, если смены нет); None — сгиба нет.
    """

    mask: np.ndarray
    quad: np.ndarray
    spine_right: bool | None
    text: np.ndarray
    fold_mask: np.ndarray
    fold_quad: np.ndarray
    spine_edge: np.ndarray | None


@dataclass(frozen=True)
class FrameCase:
    """Кадр стенда.

    Поля:
        run: Прогон.
        key: Ключ кадра.
        pages: Страницы слева направо (без вырожденных трапеций).
        exclude: Зона пальца на масштабе сегментации (точки контура там негодны).
        seg_to_frame: Множитель px сегментации → px кадра.
        frame_shape: Размер кадра ``(H, W)``.
        box: Рамка итогового TIFF в кадре ``(x0, y0, x1, y1)``.
        fold: Линия сгиба разворота (px сегментации) или None (страница одна, сгиб не нашёлся).
    """

    run: RunSpec
    key: str
    pages: list[PageInput]
    exclude: np.ndarray
    seg_to_frame: float
    frame_shape: tuple[int, int]
    box: tuple[int, int, int, int]
    fold: Fold | None

    @property
    def seg_px_per_mm(self) -> float:
        """Плотность масштаба сегментации, px/мм."""
        return self.run.px_per_mm / self.seg_to_frame

    def output(self) -> np.ndarray:
        """Итоговый TIFF прогона v2 (осевая рамка ``box`` кадра; пальцы закрашены, периферия дорисована)."""
        return read_image(self.run.output_dir / rel_of(self.key).with_suffix(".tiff"))

    def debug(self) -> np.ndarray:
        """Отладочный оверлей v2 кадра («кадр с разметкой | итог»)."""
        return read_image(overlay_path(self.run.debug_dir, rel_of(self.key), OverlayTag.MAIN))


# Яма маски меньше этой площади не засыпается (зубцы края), мм².
PIT_MIN_MM2 = 4.0
# Яма засыпается, если её медианная яркость не темнее медианы бумаги страницы больше чем на столько уровней (0–255):
# тёмная выемка — фон или щель у корешка, светлая — бумага, вырезанная SAM 3 вместе со штампом.
PIT_DARK_DELTA = 40


def fill_paper_pits(mask: np.ndarray, gray: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Маска страницы с засыпанными ямами, похожими на бумагу.

    Ямы — куски выпуклой оболочки маски вне маски. Штамп, рукописная пометка у края SAM 3 вырезает из бумаги
    (ХиП 1991/02 IMG_1091: яма ~25×15 мм под штампом библиотеки у верхнего края); медиана яркости такой ямы — как у
    бумаги, у выемки фона — тёмная.

    Args:
        mask: Маска страницы.
        gray: Кадр в сером на масштабе маски.
        px_per_mm: Плотность масштаба маски, px/мм.

    Returns:
        Маска с засыпанными ямами.
    """
    if not mask.any():
        return mask
    pts = cv2.findNonZero(mask.astype(np.uint8))
    hull = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(hull, [cv2.convexHull(pts)], 1)
    deficit = (hull > 0) & ~mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(deficit.astype(np.uint8), connectivity=4)
    paper_level = float(np.median(gray[mask]))
    min_area = PIT_MIN_MM2 * px_per_mm**2
    out = mask.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = labels == i
        if float(np.median(gray[comp])) >= paper_level - PIT_DARK_DELTA:
            out |= comp
    return out


# Край страницы у корешка не ближе к тексту surya этой страницы, мм: перепад освещения у корешка бывает принят за
# чужую бумагу (ХиП 1991/02 IMG_1168: край уходил в текст левой страницы), а текст на чужой бумаге не печатают.
EDGE_TEXT_MARGIN_MM = 1.0


def keep_text_inside(
    edge: np.ndarray, fold: Fold, blocks: list[np.ndarray], page: np.ndarray, spine_right: bool, margin_px: float
) -> np.ndarray:
    """Край у корешка, отодвинутый к сгибу так, чтобы текстовые блоки страницы остались внутри с припуском.

    Args:
        edge: Край (N, 2), сверху вниз.
        fold: Сгиб (край не уходит за него).
        blocks: Многоугольники текстовых блоков кадра, px сегментации.
        page: Маска страницы (блок относится к странице по центру).
        spine_right: Корешок справа от страницы.
        margin_px: Припуск от блока, px.

    Returns:
        Исправленный край.
    """
    h, w = page.shape
    x = edge[:, 0].copy()
    y = edge[:, 1]
    for poly in blocks:
        cx, cy = poly.mean(axis=0)
        if not page[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))]:
            continue
        rows = (y >= poly[:, 1].min()) & (y <= poly[:, 1].max())
        if spine_right:
            x[rows] = np.maximum(x[rows], poly[:, 0].max() + margin_px)
        else:
            x[rows] = np.minimum(x[rows], poly[:, 0].min() - margin_px)
    fx = fold.x_at(y)
    x = np.minimum(x, fx) if spine_right else np.maximum(x, fx)
    return np.stack([x, y], axis=1)


def load_case(run: RunSpec, key: str) -> FrameCase:
    """Кадр из кэша прогона v2: страницы — как в ``finish_frame`` (итоговая бумага, книга, экземпляры SAM 3).

    Args:
        run: Прогон.
        key: Ключ кадра.

    Returns:
        Кадр стенда.
    """
    cache = run.cache_dir / key
    with np.load(cache / "book.npz") as data:
        book, hulls = data["book"], data["hulls"]
    with np.load(cache / "sam3.npz") as data:
        inst = data["inst"]
    with np.load(cache / "paper.npz") as data:
        paper = data["paper"]
        protected = data["protected"] if "protected" in data.files else np.zeros_like(paper)
    with np.load(cache / "fingers.npz") as data:
        paint = data["paint"]
    frame_shape = paint.shape[:2]
    seg_to_frame = frame_shape[0] / paper.shape[0]
    exclude = cv2.resize(paint.astype(np.uint8), paper.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    masks = frame_pages(paper, book, hulls, inst)
    quads = page_quads_of(cache, masks)
    labels = read_layout_labels(cache / "layout.json") or []
    polys = read_layout(cache / "layout.json") or []
    blocks = [p / seg_to_frame for p, lab in zip(polys, labels) if lab in TEXT_LABELS]
    two = len(masks) == 2 and all(q is not None for q in quads)
    seg_px_per_mm = run.px_per_mm / seg_to_frame
    fold, fold_masks, fold_quads = None, list(masks), list(quads)
    if two:
        gray = cv2.cvtColor(read_image(cache / "seg.png"), cv2.COLOR_BGR2GRAY)
        fold = find_fold(gray, masks[0], masks[1], seg_px_per_mm)
        if fold is not None:
            fold_masks = list(split_by_fold(masks[0], masks[1], fold))
            fold_quads = [fit_enclosing_quad(m) for m in fold_masks]
            fold_quads = [fq if fq is not None else q for fq, q in zip(fold_quads, quads)]
    # Край каждой страницы у корешка по цвету её бумаги; маска обрезается по нему (вклейка, подложенный лист).
    edges: list[np.ndarray | None] = [None] * len(masks)
    if fold is not None:
        seg_bgr = read_image(cache / "seg.png")
        for i in range(2):
            edges[i] = page_spine_edge(seg_bgr, fold, fold_masks[i], i == 0, seg_px_per_mm)
            edges[i] = keep_text_inside(
                edges[i], fold, blocks, fold_masks[i], i == 0, EDGE_TEXT_MARGIN_MM * seg_px_per_mm
            )
            fold_masks[i] = trim_at_edge(fold_masks[i], edges[i], i == 0)
    # Ямы маски, похожие на бумагу (штамп, пометка у края — SAM 3 их вырезает), засыпаются до подгонки рамки.
    gray_seg = cv2.cvtColor(read_image(cache / "seg.png"), cv2.COLOR_BGR2GRAY)
    fold_masks = [fill_paper_pits(m, gray_seg, seg_px_per_mm) for m in fold_masks]
    fold_quads = [fit_enclosing_quad(m) if fq is not None else fq for m, fq in zip(fold_masks, fold_quads)]
    fold_quads = [fq if fq is not None else q for fq, q in zip(fold_quads, quads)]
    pages = []
    for i, (mask, quad) in enumerate(zip(masks, quads)):
        if quad is None:
            continue
        h, w = mask.shape
        own = [
            p for p in blocks if mask[int(np.clip(p[:, 1].mean(), 0, h - 1)), int(np.clip(p[:, 0].mean(), 0, w - 1))]
        ]
        text = np.vstack(own) if own else np.zeros((0, 2))
        pages.append(PageInput(mask, quad, (i == 0) if two else None, text, fold_masks[i], fold_quads[i], edges[i]))
    # Рамка итога — как в finish_frame: осевая рамка «бумага ∪ защита», переведённой в размер кадра.
    box = paper_bbox(to_frame(paper | protected, frame_shape))
    return FrameCase(run, key, pages, exclude, seg_to_frame, frame_shape, box, fold)
