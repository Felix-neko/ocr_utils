"""Подгоночные области страниц кадра по кэшу прогона scan_cropping v2 — тем же кодом, что оверлей шага ``07_lines`` в ``pipeline.finish_frame`` (в кэше они не хранятся), и рамка вырезки итога."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.fill import paper_bbox
from ocr_utils.scan_cropping.v2.page_frames import PageFrame
from ocr_utils.scan_cropping.v2.pipeline import (
    PaperMasks,
    WorkPaths,
    book_zone,
    frame_pages,
    frame_spread_lines,
    paint_on_seg,
    read_bgr,
    read_layout,
    read_layout_labels,
    spread_fit,
    to_frame,
)
from ocr_utils.scan_cropping.v2.spread_lines import SpineMethod

# Метка формулы в блоках surya.
EQUATION_LABEL = "Equation"


@dataclass(frozen=True)
class CachedFrames:
    """Области страниц кадра и привязка к итогу.

    Поля:
        frames: Области левой и правой страницы (px сегментации) или None — страниц с трапецией не две.
        box: Рамка вырезки итога ``(x0, y0, x1, y1)`` в px кадра или None — бумага не найдена.
        seg_scale: Масштаб сегментации относительно кадра по X и Y.
        equations: Блоки surya ``Equation`` из кэша v2 (``layout.json``), многоугольники в px кадра.
    """

    frames: list[PageFrame] | None
    box: tuple[int, int, int, int] | None
    seg_scale: tuple[float, float]
    equations: tuple[np.ndarray, ...] = ()

    def frame_to_output(self, points: np.ndarray) -> np.ndarray:
        """Точки (N, 2) px кадра → px итога (минус левый верхний угол рамки вырезки)."""
        return np.asarray(points, np.float64) - np.array([self.box[0], self.box[1]], np.float64)

    def to_output(self, points: np.ndarray) -> np.ndarray:
        """Точки (N, 2) px сегментации → px итога (кадр минус левый верхний угол рамки вырезки)."""
        x0, y0 = self.box[0], self.box[1]
        return np.asarray(points, np.float64) / np.array(self.seg_scale) - np.array([x0, y0], np.float64)


def cached_frames(src: Path, work_dir: Path, key: str, bg_veto: bool, spine_method: SpineMethod) -> CachedFrames:
    """Пересчёт областей страниц и рамки вырезки кадра по кэшу v2.

    Повторяет финал кадра (``pipeline.finish_frame``): страницы по итоговой бумаге, прямые разворота, полоса и линия
    корешка, ``spread_frames``; рамка — ``paper_bbox`` бумаги с защитой контента (``protected`` пишет финал в
    ``paper.npz``).

    Args:
        src: Исходный кадр (нужен для «не фона без рук» — ``book_zone``).
        work_dir: Папка кэша прогона v2.
        key: Ключ кадра (``FrameJob.key``).
        bg_veto: Прогон шёл с вето фона (маски книги — из ``book_veto.npz``).
        spine_method: Способ разделителя страниц прогона (``--spine-method``).

    Returns:
        Области и рамка.

    Raises:
        FileNotFoundError: Нет файла кэша.
    """
    work = WorkPaths(work_dir / key)
    img = read_bgr(work.seg)
    book = dict(np.load(work.book_veto if bg_veto else work.book))
    sam3 = dict(np.load(work.sam3))
    with np.load(work.masks) as data:
        masks = PaperMasks(**{name: data[name] for name in PaperMasks.__dataclass_fields__ if name in data.files})
        protected = data["protected"] if "protected" in data.files else np.zeros_like(masks.paper)
    fingers = np.load(work.fingers)
    polys = read_layout(work.layout) or []
    labels = read_layout_labels(work.layout) or []
    # Формулы — только surya ``Equation`` (точность 0.96, полнота 0.92 — ``research/surya_equations``): пиксельного
    # детектора формул нет.
    equations = tuple(np.asarray(p, np.float64) for p, label in zip(polys, labels) if label == EQUATION_LABEL)
    original = read_bgr(src)
    shape = original.shape[:2]
    seg_scale = (img.shape[1] / shape[1], img.shape[0] / shape[0])
    box = paper_bbox(to_frame(masks.paper | protected, shape))
    paint_seg = paint_on_seg(fingers, img.shape[:2])
    pages = frame_pages(img, book, sam3, paint_seg, masks)
    zone_seg = book_zone(original, fingers["core"], book["book"])
    lines = frame_spread_lines(pages, book["book"], zone_seg, paint_seg, spine_method)
    if lines is None:
        return CachedFrames(None, box, seg_scale, equations)
    core_seg = cv2.resize(fingers["core"].astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    fit = spread_fit(img, sam3["inst"], book["book"], pages, lines, polys, paint_seg, core_seg, seg_scale[1])
    return CachedFrames(None if fit is None else fit.frames, box, seg_scale, equations)
