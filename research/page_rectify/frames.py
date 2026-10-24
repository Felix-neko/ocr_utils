"""Подгоночные области страниц кадра по кэшу прогона scan_cropping v2 — тем же кодом, что оверлей шага ``07_lines`` в ``pipeline.finish_frame`` (в кэше они не хранятся), и рамка вырезки итога."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ocr_utils.frame_flags.flags import read_flags
from ocr_utils.scan_cropping.v2.fill import paper_bbox
from ocr_utils.scan_cropping.v2.rectify.geometry import SpreadGeometry, equations_of
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

# Прежнее имя геометрии разворота в стенде.
CachedFrames = SpreadGeometry


def cached_frames(src: Path, work_dir: Path, key: str, bg_veto: bool, spine_method: SpineMethod) -> SpreadGeometry:
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
    equations = equations_of(polys, labels)
    original = read_bgr(src)
    shape = original.shape[:2]
    seg_scale = (img.shape[1] / shape[1], img.shape[0] / shape[0])
    box = paper_bbox(to_frame(masks.paper | protected, shape))
    paint_seg = paint_on_seg(fingers, img.shape[:2])
    pages = frame_pages(img, book, sam3, paint_seg, masks, read_flags(src))
    zone_seg = book_zone(original, fingers["core"], book["book"])
    lines = frame_spread_lines(pages, book["book"], zone_seg, paint_seg, spine_method)
    if lines is None:
        return SpreadGeometry(None, box, seg_scale, equations)
    core_seg = cv2.resize(fingers["core"].astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    fit = spread_fit(img, sam3["inst"], book["book"], pages, lines, polys, paint_seg, core_seg, seg_scale[1])
    return SpreadGeometry(None if fit is None else fit.frames, box, seg_scale, equations)
