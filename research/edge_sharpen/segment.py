"""Повторная сегментация бумаги v2 по отфильтрованному кадру: SAM 3 (GPU, оба прохода) по кадру после фильтра, затем бумага, страницы и выравнивание сторон как в ``cpu_pages`` — по кадру без фильтра или с ним."""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from enum import Enum
from pathlib import Path

import numpy as np

from ocr_utils.scan_cropping.v2.colour_pages import colour_pages_growth
from ocr_utils.scan_cropping.v2.paper import SAM3_CACHE_VERSION, sam3_pages
from ocr_utils.scan_cropping.v2.pipeline import frame_pages, frame_paper, paint_on_seg, read_bgr, write_pages
from ocr_utils.scan_cropping.v2.scale import Scale
from ocr_utils.scan_cropping.v2.side_fit import regularize_sides, snap_sheet_edges
from research.edge_sharpen.filters import EdgeFilter, apply_filter
from research.page_cutting.runs import RunSpec

log = logging.getLogger(__name__)

# Глубина уточнения склейки, мм — как ``--refine-depth-mm`` боевых прогонов (v2_pack6_args.sh).
REFINE_DEPTH_MM = 5.0725
# Плотность масштаба сегментации v2, px/мм (постоянна: ``REF_PX_PER_MM × BG_SCALE``).
SEG_PX_PER_MM = Scale().seg_px(10.0) / 10.0


class FilterMode(str, Enum):
    """Куда подаётся отфильтрованный кадр."""

    # Только во вход SAM 3 (оба прохода); доводка края (snap, склейка, уточнение, выравнивание сторон) — по исходному.
    SAM = "sam"
    # И в SAM 3, и в доводку края.
    ALL = "all"


def variant_dir(out_root: Path, filt: EdgeFilter, run: RunSpec, key: str) -> Path:
    """Папка кадра варианта: ``<out_root>/<фильтр>/<прогон>/<ключ>``."""
    return out_root / filt.value / run.name / key


def sam_stage(predictor, run: RunSpec, key: str, filt: EdgeFilter, out_root: Path, redo: bool = False) -> None:
    """SAM 3 по отфильтрованному кадру сегментации (и по кадру без крупной краски) → ``sam3.npz`` варианта.

    Args:
        predictor: ``SAM3SemanticPredictor``.
        run: Прогон v2 (кэш кадра — источник ``seg.png``, ``seg_noink.png``, ``book.npz``).
        key: Ключ кадра.
        filt: Фильтр.
        out_root: Корень выхода стенда.
        redo: Считать заново, даже если готово.
    """
    out = variant_dir(out_root, filt, run, key)
    if (out / "sam3.npz").exists() and not redo:
        return
    cache = run.cache_dir / key
    hulls = np.load(cache / "book.npz")["hulls"]
    seg = apply_filter(read_bgr(cache / "seg.png"), filt, SEG_PX_PER_MM)
    raw, inst = sam3_pages(predictor, seg, hulls)
    data = {"raw": raw, "inst": inst, "version": np.int32(SAM3_CACHE_VERSION)}
    if (cache / "seg_noink.png").exists():
        noink = apply_filter(read_bgr(cache / "seg_noink.png"), filt, SEG_PX_PER_MM)
        data["raw_noink"], data["inst_noink"] = sam3_pages(predictor, noink, hulls)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "sam3.npz", **data)


def cpu_stage(run: RunSpec, key: str, filt: EdgeFilter, mode: FilterMode, out_root: Path) -> str:
    """Бумага и страницы кадра по SAM 3 варианта — как ``pipeline.cpu_pages`` с ключами боевых прогонов
    (``--fit-sides``, ``--no-snap-sides``, цветные страницы) → ``paper_<режим>.npz``, ``pages_<режим>.json``.

    Args:
        run: Прогон v2.
        key: Ключ кадра.
        filt: Фильтр.
        mode: Куда подаётся отфильтрованный кадр.
        out_root: Корень выхода стенда.

    Returns:
        Ключ кадра.
    """
    out = variant_dir(out_root, filt, run, key)
    cache = run.cache_dir / key
    seg = read_bgr(cache / "seg.png")
    img = apply_filter(seg, filt, SEG_PX_PER_MM) if mode is FilterMode.ALL else seg
    book = dict(np.load(cache / "book.npz"))
    sam3 = dict(np.load(out / "sam3.npz"))
    masks = frame_paper(img, book, sam3, Scale().seg_px(REFINE_DEPTH_MM))
    paint_seg = paint_on_seg(np.load(cache / "fingers.npz"), img.shape[:2])
    pages = frame_pages(img, book, sam3, paint_seg, masks)
    edge_added = snap_sheet_edges(masks.paper, pages, book["bg"], book["book"], img)
    if edge_added.any():
        masks = replace(masks, paper=masks.paper | edge_added)
        pages = frame_pages(img, book, sam3, paint_seg, masks)
    palette_added = colour_pages_growth(img, pages, masks.paper, book["book"], book["bg"])
    if palette_added.any():
        masks = replace(masks, paper=masks.paper | palette_added, palette_added=palette_added)
        pages = frame_pages(img, book, sam3, paint_seg, masks)
    side = regularize_sides(masks.paper, pages, book["bg"], img, False)
    masks = replace(masks, paper=side.paper, side_added=side.added | edge_added, side_cut=side.cut)
    pages = frame_pages(img, book, sam3, paint_seg, masks)
    np.savez_compressed(out / f"paper_{mode.value}.npz", **{k: v for k, v in asdict(masks).items() if v is not None})
    write_pages(out / f"pages_{mode.value}.json", pages)
    return key
