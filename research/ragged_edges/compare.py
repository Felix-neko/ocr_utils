"""Сравнение двух прогонов v2 по кэшам: страницы кадра у дальней стороны и у корешка — площадь расхождения масок
страниц, способ уточнения и доля пальца у края по ``refine.json``, вырезки с контурами обоих прогонов для листов."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.page_sides import Side, quad_sides
from research.ragged_edges.features import load_frame

log = logging.getLogger(__name__)

# Зона стороны: пиксели расхождения не дальше этого от прямой стороны трапеции прежнего прогона, мм.
ZONE_MM = 30.0
# Вырезка вдоль стороны: столько в обе стороны от прямой стороны, мм; запас по высоте, мм.
CROP_OUT_MM, CROP_IN_MM, PAD_MM = 18.0, 25.0, 5.0
TILE_PX_PER_MM = 4.0  # масштаб плитки, px/мм (сегментация — 6.9)
LINE_PX = 2
OLD_COLOUR = (220, 90, 20)  # контур страницы прежнего прогона — синий
NEW_COLOUR = (0, 150, 0)  # нового — зелёный
# Стороны, которые сравниваются.
ZONES = (Side.FAR, Side.SPINE)


def side_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Расстояние точек ``(N, 2)`` ``x, y`` до прямой через ``a`` и ``b``, px."""
    d = b - a
    return np.abs(d[0] * (points[:, 1] - a[1]) - d[1] * (points[:, 0] - a[0])) / max(float(np.hypot(*d)), 1e-9)


def refine_side(work: Path, page: str) -> dict:
    """Запись стороны ``page`` из ``refine.json`` кадра (пусто — сторона не уточнялась)."""
    path = work / "refine.json"
    if not path.exists():
        return {}
    return next((s for s in json.loads(path.read_text())["sides"] if s["page"] == page), {})


def frame_compare(args: tuple[str, Path, Path]) -> list[dict]:
    """Строки сравнения страниц кадра двух прогонов.

    Args:
        args: ``(набор, папка кадра в кэше прежнего прогона, в кэше нового)``.

    Returns:
        По строке на страницу разворота: площадь расхождения масок у дальней стороны и у корешка (мм²), способ
        уточнения и доля пальца у края в обоих прогонах; пусто — кадр не разворот в одном из прогонов или ошибка.
    """
    name, old_work, new_work = args
    try:
        old, new = load_frame(old_work), load_frame(new_work)
    except Exception:  # noqa: BLE001 — кадр с ошибкой не роняет сравнение
        log.exception("%s: ошибка", new_work)
        return []
    if len(old.pages) != 2 or len(new.pages) != 2:
        return []
    mm2 = SEG_PX_PER_MM**2
    rows = []
    for index, page in enumerate("LR"):
        xor = old.pages[index].mask ^ new.pages[index].mask
        ys, xs = np.nonzero(xor)
        points = np.stack([xs, ys], axis=1).astype(np.float64)
        sides = quad_sides(old.pages[index].quad, index == 0)
        row = {"set": name, "cache": str(new_work.parent), "key": new_work.name, "page": page,
               "xor_mm2": round(len(xs) / mm2, 1)}  # fmt: skip
        for zone in ZONES:
            near = side_distance(points, *sides[zone]) <= ZONE_MM * SEG_PX_PER_MM if len(points) else np.zeros(0, bool)
            row[f"{zone.value}_mm2"] = round(int(near.sum()) / mm2, 1)
        for tag, work in (("old", old_work), ("new", new_work)):
            side = refine_side(work, page)
            row[f"{tag}_chosen"] = side.get("chosen") or side.get("status", "")
            row[f"{tag}_finger"] = side.get("finger_share", "")
        rows.append(row)
    return rows


def pair_tile(old_work: Path, new_work: Path, page: str, zone: Side, caption: str) -> np.ndarray:
    """Вырезка вдоль стороны страницы на кадре нового прогона с контурами страницы обоих прогонов.

    Args:
        old_work: Папка кадра в кэше прежнего прогона.
        new_work: То же нового.
        page: ``L`` или ``R``.
        zone: ``Side.FAR`` (дальняя сторона) или ``Side.SPINE`` (у корешка).
        caption: Подпись (ключ, площади).

    Returns:
        Плитка BGR в масштабе :data:`TILE_PX_PER_MM` с шапкой.
    """
    old, new = load_frame(old_work), load_frame(new_work)
    index = "LR".index(page)
    a, b = quad_sides(old.pages[index].quad, index == 0)[zone]
    mm = SEG_PX_PER_MM
    # Наружу от страницы: дальняя сторона левой и сторона у корешка правой — влево.
    outward_left = (zone is Side.FAR) == (index == 0)
    lo, hi = (CROP_OUT_MM, CROP_IN_MM) if outward_left else (CROP_IN_MM, CROP_OUT_MM)
    h, w = new.img.shape[:2]
    x0, x1 = int(max(0, min(a[0], b[0]) - lo * mm)), int(min(w, max(a[0], b[0]) + hi * mm))
    y0, y1 = int(max(0, min(a[1], b[1]) - PAD_MM * mm)), int(min(h, max(a[1], b[1]) + PAD_MM * mm))
    s = TILE_PX_PER_MM / mm
    crop = cv2.resize(new.img[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    for frame, colour in ((old, OLD_COLOUR), (new, NEW_COLOUR)):
        contours, _ = cv2.findContours(frame.pages[index].mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)  # fmt: skip
        moved = [
            np.round((c[:, 0, :].astype(np.float64) - (x0, y0)) * s).astype(np.int32)[:, None, :] for c in contours
        ]
        cv2.drawContours(crop, moved, -1, colour, LINE_PX)
    head = np.full((34, max(crop.shape[1], 240), 3), 255, np.uint8)
    for k, line in enumerate(caption.split("\n")[:2]):
        cv2.putText(head, line, (4, 14 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    crop = np.pad(crop, ((0, 0), (0, head.shape[1] - crop.shape[1]), (0, 0)), constant_values=255)
    return np.vstack([head, crop])
