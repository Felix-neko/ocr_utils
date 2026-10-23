"""Меры кадра прогона v2 для отбора тестовой коллекции: рваность и изгиб сторон страниц, зазор текста у корешка, цветность, число блоков surya — всё по кэшу, без GPU."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.page_pose import frame_pages
from ocr_utils.scan_cropping.v2.pipeline import read_layout, read_layout_labels
from ocr_utils.scan_cropping.v2.trapezoid import fit_enclosing_quad
from ocr_utils.scan_cropping.v2.page_sides import (
    Side,
    SideProfile,
    inward_normal,
    quad_sides,
    side_profiles,
    smooth_profile,
)
from research.page_cutting.runs import RunSpec, read_image

# Окно медианного сглаживания профиля стороны, мм: длиннее зубцов рваного края (2–8 мм), короче изгиба страницы.
SMOOTH_WINDOW_MM = 15.0
# Перцентиль рваности: доля точек стороны, у которых отклонение от сглаженного профиля не больше меры.
RAGGED_PERCENTILE = 90
# Меньше годных точек у стороны — сторону не мерим (палец, рамка кадра).
MIN_SIDE_POINTS = 30
# Отступ от края страницы при мере цветности, доля размера страницы с каждой стороны: центр без кромки и тени.
CHROMA_INSET = 0.2
# Блоки surya, которые считаются текстом при мере зазора у корешка (колонтитулы и номер страницы — тоже текст у корешка).
# Блок, заходящий за сторону у корешка глубже, — блок surya через оба листа (склейка колонок), а не текст у корешка, мм.
GUTTER_CROSS_MM = 5.0
TEXT_LABELS = ("Text", "PageHeader", "PageFooter", "SectionHeader", "ListItem", "Footnote", "Caption", "Table")


@dataclass(frozen=True)
class FrameMeasures:
    """Меры кадра (по худшей из страниц).

    Поля:
        run: Имя прогона.
        key: Ключ кадра в кэше.
        pages: Число страниц.
        poses: Положения страниц через запятую (``flat``, ``raised``).
        keystone: Наибольший сход сторон трапеции по страницам, °.
        ragged_mm: Наибольшая по сторонам рваность края: перцентиль отклонения контура от сглаженного профиля, мм.
        ragged_side: Сторона с наибольшей рваностью (``L:top`` — верх левой страницы).
        bend_tb_mm: Наибольший изгиб верха или низа: размах сглаженного профиля, мм.
        bend_spine_mm: Наибольший изгиб стороны у корешка, мм.
        bend_far_mm: Наибольший изгиб дальней стороны, мм.
        gutter_gap_mm: Наименьший зазор от текста surya до стороны у корешка, мм (минус — текст за стороной); нет
            текста или одна страница — NaN.
        chroma: Наибольшая по страницам медианная цветность центра страницы (Lab, расстояние от серого).
        blocks: Наименьшее по страницам число текстовых блоков surya.
    """

    run: str
    key: str
    pages: int
    poses: str
    keystone: float
    ragged_mm: float
    ragged_side: str
    bend_tb_mm: float
    bend_spine_mm: float
    bend_far_mm: float
    gutter_gap_mm: float
    chroma: float
    blocks: int


def page_quads_of(cache: Path, pages: list[np.ndarray]) -> list[np.ndarray | None]:
    """Трапеции страниц: из ``pages.json``, если их столько же, сколько масок; иначе подгоняются заново.

    Args:
        cache: Папка кадра в кэше v2.
        pages: Маски страниц слева направо.

    Returns:
        Трапеции (4, 2) px сегментации или None для вырожденных.
    """
    rows = json.loads((cache / "pages.json").read_text())
    if len(rows) == len(pages):
        return [np.asarray(r["quad"], np.float64) if r.get("quad") else None for r in rows]
    return [fit_enclosing_quad(p) for p in pages]


def side_stats(profile: SideProfile, seg_px_per_mm: float) -> tuple[float, float] | None:
    """Рваность и изгиб стороны по годным точкам.

    Args:
        profile: Профиль стороны.
        seg_px_per_mm: Плотность масштаба сегментации, px/мм.

    Returns:
        ``(рваность, изгиб)`` в мм или None, если годных точек мало.
    """
    t, d = profile.t[profile.clean], profile.d[profile.clean]
    if len(t) < MIN_SIDE_POINTS:
        return None
    length = float(np.linalg.norm(profile.b - profile.a))
    smooth = smooth_profile(t, d, length, SMOOTH_WINDOW_MM * seg_px_per_mm)
    ragged = float(np.percentile(np.abs(d - smooth), RAGGED_PERCENTILE)) / seg_px_per_mm
    bend = float(smooth.max() - smooth.min()) / seg_px_per_mm
    return ragged, bend


def gutter_gap(
    blocks: list[np.ndarray], page: np.ndarray, quad: np.ndarray, spine_right: bool, cross_px: float
) -> float:
    """Наименьший зазор от вершин текстовых блоков страницы до стороны у корешка, px (плюс — текст внутри).

    Args:
        blocks: Многоугольники блоков surya, px сегментации.
        page: Маска страницы (блок относится к странице по своему центру).
        quad: Трапеция страницы.
        spine_right: Корешок справа от страницы.
        cross_px: Блок, заходящий за сторону глубже, пропускается (блок через оба листа), px.

    Returns:
        Зазор или NaN, если текста на странице нет.
    """
    a, b = quad_sides(quad, spine_right)[Side.SPINE]
    n = inward_normal(a, b, quad.mean(axis=0))
    h, w = page.shape
    gaps = []
    for poly in blocks:
        cx, cy = poly.mean(axis=0)
        if not page[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))]:
            continue
        gap = float(((poly - a) @ n).min())
        if gap >= -cross_px:
            gaps.append(gap)
    return min(gaps) if gaps else float("nan")


def page_chroma(lab: np.ndarray, page: np.ndarray) -> float:
    """Медианная цветность центра страницы: расстояние (a, b) от серого в Lab OpenCV.

    Args:
        lab: Кадр в Lab (uint8, масштаб сегментации).
        page: Маска страницы.

    Returns:
        Цветность (0 — серое; желтоватая бумага ~10–20, цветная обложка — больше).
    """
    ys, xs = np.nonzero(page)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    dy, dx = int((y1 - y0) * CHROMA_INSET), int((x1 - x0) * CHROMA_INSET)
    sub = page[y0 + dy : y1 - dy, x0 + dx : x1 - dx]
    if not sub.any():
        return 0.0
    ab = lab[y0 + dy : y1 - dy, x0 + dx : x1 - dx][sub][:, 1:].astype(np.float64) - 128.0
    return float(np.median(np.hypot(ab[:, 0], ab[:, 1])))


def frame_measures(run: RunSpec, key: str) -> FrameMeasures:
    """Меры кадра по кэшу прогона v2.

    Маски страниц пересобираются из итоговой бумаги (``paper.npz``), грубой маски книги и экземпляров SAM 3, как в
    ``finish_frame``; трапеции — из ``pages.json``; блоки surya — из ``layout.json`` (px кадра → px сегментации).

    Args:
        run: Прогон.
        key: Ключ кадра.

    Returns:
        Меры кадра.
    """
    cache = run.cache_dir / key
    with np.load(cache / "book.npz") as data:
        book, hulls = data["book"], data["hulls"]
    with np.load(cache / "sam3.npz") as data:
        inst = data["inst"]
    with np.load(cache / "paper.npz") as data:
        paper = data["paper"]
    with np.load(cache / "fingers.npz") as data:
        paint = data["paint"]
    seg = read_image(cache / "seg.png")
    frame_h = paint.shape[0]
    seg_to_frame = frame_h / paper.shape[0]
    seg_px_per_mm = run.px_per_mm / seg_to_frame
    # Зона пальца на масштабе сегментации: точки контура в ней негодны.
    paint_seg = cv2.resize(paint.astype(np.uint8), paper.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    pages = frame_pages(paper, book, hulls, inst)
    quads = page_quads_of(cache, pages)
    poses = [r.get("pose", "") for r in json.loads((cache / "pages.json").read_text())]
    labels = read_layout_labels(cache / "layout.json") or []
    polys = read_layout(cache / "layout.json") or []
    text_blocks = [p / seg_to_frame for p, lab in zip(polys, labels) if lab in TEXT_LABELS]
    lab_img = cv2.cvtColor(seg, cv2.COLOR_BGR2LAB)

    keystone, ragged, ragged_side = 0.0, 0.0, ""
    bends = {Side.TOP: 0.0, Side.SPINE: 0.0, Side.FAR: 0.0}
    gap, chroma, blocks = float("nan"), 0.0, 10**6
    two = len(pages) == 2
    for i, (page, quad) in enumerate(zip(pages, quads)):
        chroma = max(chroma, page_chroma(lab_img, page))
        if quad is None:
            continue
        spine_right = (i == 0) if two else None
        for side, profile in side_profiles(page, quad, spine_right, exclude=paint_seg).items():
            stats = side_stats(profile, seg_px_per_mm)
            if stats is None:
                continue
            if stats[0] > ragged:
                ragged, ragged_side = stats[0], f"{'LR'[i] if two else 'S'}:{side.value}"
            side_kind = Side.TOP if side in (Side.TOP, Side.BOTTOM) else side
            bends[side_kind] = max(bends[side_kind], stats[1])
        sides = quad_sides(quad, spine_right)
        keystone = max(keystone, _keystone(sides))
        h, w = page.shape
        own = [
            p
            for p in text_blocks
            if page[int(np.clip(p[:, 1].mean(), 0, h - 1)), int(np.clip(p[:, 0].mean(), 0, w - 1))]
        ]
        blocks = min(blocks, len(own))
        if two:
            g = gutter_gap(own, page, quad, spine_right, GUTTER_CROSS_MM * seg_px_per_mm) / seg_px_per_mm
            gap = g if np.isnan(gap) else min(gap, g)
    return FrameMeasures(
        run=run.name,
        key=key,
        pages=len(pages),
        poses=",".join(poses),
        keystone=round(keystone, 2),
        ragged_mm=round(ragged, 2),
        ragged_side=ragged_side,
        bend_tb_mm=round(bends[Side.TOP], 2),
        bend_spine_mm=round(bends[Side.SPINE], 2),
        bend_far_mm=round(bends[Side.FAR], 2),
        gutter_gap_mm=round(gap, 2),
        chroma=round(chroma, 1),
        blocks=0 if blocks == 10**6 else blocks,
    )


def _keystone(sides: dict[Side, tuple[np.ndarray, np.ndarray]]) -> float:
    """Наибольший угол между противоположными сторонами (верх–низ, корешок–дальняя), °."""
    out = 0.0
    for s1, s2 in ((Side.TOP, Side.BOTTOM), (Side.SPINE, Side.FAR)):
        v1 = sides[s1][1] - sides[s1][0]
        v2 = sides[s2][1] - sides[s2][0]
        cos = abs(v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        out = max(out, float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))))
    return out
