"""Меры кадра «было (keystone) | стало (coons)» по двум кэшам scan_cropping v2: покрытие строк блоками surya, рыхлость блоков Text, блоки через корешок, защита и срезанный контент, число и типы блоков, сопоставление блоков по IoU."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ocr_utils.scan_cropping.v2.page_frames import load_frames
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.pipeline import WorkPaths, layout_data, read_page_quads

# Блок «через корешок»: вершины по обе стороны линии корешка дальше этого, мм бумаги (≈ 21 px сегментации).
SPINE_CROSS_MM = 3.0
# Блок Text «рыхлый» там, где до своей печати дальше этого, мм бумаги.
LOOSE_REACH_MM = 2.0
# Совпадение блоков двух прогонов: IoU не меньше этого (венгерский алгоритм по матрице IoU).
MATCH_IOU = 0.5
# Масштаб масок сопоставления относительно масштаба сегментации (IoU на ¼ — быстро, точности хватает).
MATCH_SCALE = 0.25
# Метка текстового блока surya.
TEXT_LABEL = "Text"
# Строка лога v2 со временем surya: ``<ключ>: surya 1.2 с, блоков 31, ...``.
SURYA_LOG = re.compile(r"INFO \S+: (\S+): surya ([0-9.]+) с, блоков (\d+)")


@dataclass(frozen=True)
class FrameLayout:
    """Блоки surya кадра одного прогона на масштабе сегментации.

    Поля:
        blocks: Многоугольники (N, 2) px сегментации.
        labels: Типы блоков.
        warp: Чем выпрямлялся кадр (``LayoutWarp``) или пусто для старых версий.
        reason: Причина отката на трапеции (``FramesReason``) или пусто.
    """

    blocks: list[np.ndarray]
    labels: list[str]
    warp: str
    reason: str


def frame_layout(work: WorkPaths, seg: tuple[float, float]) -> FrameLayout | None:
    """Блоки ``layout.json`` кадра на масштабе сегментации.

    Args:
        work: Файлы кэша кадра.
        seg: Масштаб сегментации относительно кадра по X и Y.

    Returns:
        Блоки или None (файла нет или формат негоден).
    """
    data = layout_data(work.layout)
    if data is None:
        return None
    blocks = [np.asarray(p, np.float64).reshape(-1, 2) * np.asarray(seg) for p in data["blocks"]]
    labels = [str(v) for v in data.get("labels", [""] * len(blocks))]
    warp = str(data.get("warp", "keystone" if data.get("keystone") else "none"))
    return FrameLayout(blocks, labels, warp, str(data.get("reason", "")))


def fill_mask(shape: tuple[int, int], polys: list[np.ndarray], scale: float = 1.0) -> np.ndarray:
    """Объединение многоугольников маской.

    Args:
        shape: Размер маски ``(H, W)``.
        polys: Многоугольники (N, 2) px сегментации.
        scale: Множитель координат (маска мельче масштаба сегментации).

    Returns:
        Булева маска.
    """
    mask = np.zeros(shape, np.uint8)
    pts = [np.round(np.asarray(p) * scale).astype(np.int32).reshape(-1, 1, 2) for p in polys if len(p) >= 3]
    if pts:
        cv2.fillPoly(mask, pts, 1)
    return mask > 0


def spine_x_of(new: WorkPaths, base: WorkPaths) -> tuple[float, float, float] | None:
    """Линия корешка на масштабе сегментации ``x = x0 + k·(y − y0)``: по областям surya (кэш coons) или по трапециям.

    Args:
        new: Кэш кадра прогона coons (``layout_frames.npz``).
        base: Кэш кадра прогона keystone (``pages.json`` — запасной путь).

    Returns:
        ``(x0, y0, k)`` или None (страниц не две).
    """
    frames, _ = load_frames(new.layout_frames)
    if frames is not None and len(frames) == 1:
        # Одна страница кадра — корешка нет.
        return None
    if frames:
        a, b = frames[0].spine[0], frames[0].spine[-1]
        k = (b[0] - a[0]) / max(b[1] - a[1], 1e-9)
        return float(a[0]), float(a[1]), float(k)
    quads = read_page_quads(base.pages)
    if len(quads) != 2:
        return None
    left = np.sort(quads[0][:, 0])[2:].mean()
    right = np.sort(quads[1][:, 0])[:2].mean()
    return float((left + right) / 2), 0.0, 0.0


def across_spine(blocks: list[np.ndarray], spine: tuple[float, float, float] | None) -> tuple[int, float]:
    """Блоки через корешок: вершины по обе стороны линии дальше ``SPINE_CROSS_MM``.

    Args:
        blocks: Многоугольники px сегментации.
        spine: Линия корешка или None.

    Returns:
        ``(число блоков, их площадь мм²)``.
    """
    if spine is None:
        return 0, 0.0
    x0, y0, k = spine
    reach = SPINE_CROSS_MM * SEG_PX_PER_MM
    count, area = 0, 0.0
    for b in blocks:
        side = b[:, 0] - (x0 + k * (b[:, 1] - y0))
        if side.max() > reach and side.min() < -reach:
            count += 1
            area += abs(float(cv2.contourArea(b.astype(np.float32).reshape(-1, 1, 2)))) / SEG_PX_PER_MM**2
    return count, area


def match_blocks(a: list[np.ndarray], b: list[np.ndarray], shape: tuple[int, int]) -> tuple[int, int, int, float]:
    """Сопоставление блоков двух прогонов по IoU (венгерский алгоритм).

    Args:
        a: Блоки прогона «было» (px сегментации).
        b: Блоки прогона «стало».
        shape: Размер масштаба сегментации ``(H, W)``.

    Returns:
        ``(совпало, исчезло из «было», новых в «стало», средний IoU совпавших)``.
    """
    if not a or not b:
        return 0, len(a), len(b), float("nan")
    small = (max(1, int(shape[0] * MATCH_SCALE)), max(1, int(shape[1] * MATCH_SCALE)))
    ma = np.stack([fill_mask(small, [p], MATCH_SCALE).ravel() for p in a]).astype(np.float32)
    mb = np.stack([fill_mask(small, [p], MATCH_SCALE).ravel() for p in b]).astype(np.float32)
    inter = ma @ mb.T
    union = ma.sum(axis=1)[:, None] + mb.sum(axis=1)[None, :] - inter
    iou = inter / np.maximum(union, 1.0)
    rows, cols = linear_sum_assignment(-iou)
    good = iou[rows, cols] >= MATCH_IOU
    matched = int(good.sum())
    mean = float(iou[rows, cols][good].mean()) if matched else float("nan")
    return matched, len(a) - matched, len(b) - matched, mean


def summary_rows(path: Path) -> dict[str, dict]:
    """Строки ``summary.csv`` прогона по ключу кадра (последняя запись кадра)."""
    if not path.exists():
        return {}
    with path.open() as f:
        return {row["key"]: row for row in csv.DictReader(f)}


def surya_times(log: Path | None) -> dict[str, float]:
    """Время шага surya по кадрам из лога прогона v2, с."""
    if log is None or not log.exists():
        return {}
    out = {}
    for line in log.read_text(errors="replace").splitlines():
        m = SURYA_LOG.search(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def run_measures(work: WorkPaths, lay: FrameLayout, paper: dict, book: dict, spine) -> dict:
    """Меры одного прогона кадра.

    Args:
        work: Файлы кэша кадра.
        lay: Блоки surya на масштабе сегментации.
        paper: Маски ``paper.npz`` (``paper``, ``protected``).
        book: Маски книги (``content``, ``book``).
        spine: Линия корешка (:func:`spine_x_of`).

    Returns:
        Меры: покрытие строк, непокрытые строки мм², рыхлость Text, через корешок, защита, срезанный контент, число и
        типы блоков.
    """
    shape = paper["paper"].shape
    union = fill_mask(shape, lay.blocks)
    protected = paper.get("protected", np.zeros(shape, bool))
    lines = book["content"] & paper["paper"]
    mm2 = SEG_PX_PER_MM**2
    # Рыхлость: доля площади Text дальше LOOSE_REACH_MM от своей печати.
    r = max(1, int(round(LOOSE_REACH_MM * SEG_PX_PER_MM)))
    near = cv2.dilate(book["content"].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    text = fill_mask(shape, [b for b, lab in zip(lay.blocks, lay.labels) if lab == TEXT_LABEL])
    cross_n, cross_mm2 = across_spine(lay.blocks, spine)
    # Срезанный контент: своя печать в «не фоне» вне бумаги и защиты.
    cut = book["content"] & book["book"] & ~(paper["paper"] | protected)
    return {
        "warp": lay.warp,
        "reason": lay.reason,
        "blocks": len(lay.blocks),
        "types": " ".join(f"{k}:{v}" for k, v in sorted(Counter(lay.labels).items())),
        "line_cover": round(float((lines & union).sum() / max(lines.sum(), 1)), 4),
        "lines_out_mm2": round(float((lines & ~union).sum() / mm2), 1),
        "text_loose": round(float((text & (near == 0)).sum() / max(text.sum(), 1)), 4),
        "spine_blocks": cross_n,
        "spine_mm2": round(cross_mm2, 1),
        "protected_mm2": round(float(protected.sum() / mm2), 1),
        "cut_content_mm2": round(float(cut.sum() / mm2), 1),
    }


def frame_measures(key: str, base_root: Path, new_root: Path) -> dict | None:
    """Меры кадра в двух прогонах и их сопоставление.

    Args:
        key: Ключ кадра (папка кэша).
        base_root: Корень кэша прогона «было» (keystone).
        new_root: Корень кэша прогона «стало» (coons).

    Returns:
        Строка CSV (``base_*``, ``new_*``, сопоставление) или None, если у кадра нет ``layout.json`` в обоих.
    """
    base, new = WorkPaths(base_root / key), WorkPaths(new_root / key)
    if not (base.masks.exists() and new.masks.exists()):
        return None
    paint_shape = np.load(new.fingers)["paint"].shape
    with np.load(new.masks) as d:
        seg_shape = d["paper"].shape
    seg = (seg_shape[1] / paint_shape[1], seg_shape[0] / paint_shape[0])
    lay_a, lay_b = frame_layout(base, seg), frame_layout(new, seg)
    if lay_a is None or lay_b is None:
        return None
    spine = spine_x_of(new, base)
    row: dict = {"key": key}
    for prefix, work, lay in (("base", base, lay_a), ("new", new, lay_b)):
        paper = dict(np.load(work.masks))
        book = dict(np.load(work.book_veto if work.book_veto.exists() else work.book))
        for name, value in run_measures(work, lay, paper, book, spine).items():
            row[f"{prefix}_{name}"] = value
    matched, lost, added, mean_iou = match_blocks(lay_a.blocks, lay_b.blocks, seg_shape)
    row.update(matched=matched, lost=lost, added=added, mean_iou=round(mean_iou, 4))
    row["frames_reason"] = load_frames(new.layout_frames)[1]
    return row


def frame_keys(root: Path) -> list[str]:
    """Ключи кадров кэша (папки с ``layout.json``)."""
    return sorted(p.name for p in root.iterdir() if (p / "layout.json").exists())


def layout_canvas(path: Path) -> dict:
    """Параметры холста из ``layout.json`` версии 5 (пусто — нет)."""
    data = json.loads(path.read_text()) if path.exists() else {}
    return data.get("canvas") or {}
