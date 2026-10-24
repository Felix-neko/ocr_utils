"""Прогон стенда по полосе: затравки растрового детектора → фото (маска, рамка, стороны), сопоставление с эталоном CVAT, строки мер на каждое эталонное фото и JSON найденного."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research.photo_bounds.bounds import (
    CORE_TONE_DELTA,
    SEARCH_MM,
    find_photos,
    light_map,
    object_ink,
    object_mask,
    object_outside,
    page_maps,
)
from research.photo_bounds.cache import GRAY_DPI, load_page
from research.photo_bounds.data import CacheLayout, PageRef, SizeBand
from research.photo_bounds.measure import SIDES, band_cover, best_match, iou, photo_json, scale_box, side_deltas_mm

log = logging.getLogger(__name__)

# Виды растра детектора, которые служат затравками фото.
SEED_KINDS = ("grayscale", "color")


@dataclass(frozen=True)
class RunJob:
    """Задание прогона полосы: полоса, раскладка кэша, папка JSON найденного."""

    page: PageRef
    layout: CacheLayout
    out_dir: Path


def detector_boxes(detector: dict) -> list[tuple[int, int, int, int]]:
    """Рамки растрового детектора, годные в затравки: серые и цветные, не во всю полосу.

    Args:
        detector: ``detector.json`` полосы.

    Returns:
        Рамки в px оригинала.
    """
    return [tuple(r["box"]) for r in detector["raster"] if r["kind"] in SEED_KINDS and not r["full_page"]]


def run_page(job: RunJob) -> list[dict]:
    """Фото полосы двумя способами и меры против эталона.

    Способы: ``det`` — рамка растрового детектора ``page_layout`` как есть; ``new`` — уточнение стенда по затравкам
    детектора. Диагностика ``refseed`` — то же уточнение с затравкой = эталон (меры края на всех эталонных фото, даже
    пропущенных детектором; в сравнение рамок не идёт).

    Args:
        job: Задание.

    Returns:
        Строки мер: по одной на эталонное фото; JSON найденного — в ``out_dir/<полоса>.json``.
    """
    data = load_page(job.layout, job.page)
    f = data.scale
    mm_per_px = 25.4 / GRAY_DPI
    det = [scale_box(b, f) for b in detector_boxes(data.detector)]
    seeds = [tuple(int(round(v)) for v in b) for b in det]
    photos = find_photos(data.gray, seeds, GRAY_DPI)
    new = [p.box for p in photos]
    refs = [scale_box(ph.box, f) for ph in job.page.photos]
    ref_photos = find_photos(data.gray, [tuple(int(round(v)) for v in r) for r in refs], GRAY_DPI)
    rows = []
    for ref_photo, ref in zip(job.page.photos, refs):
        row = {
            "key": ref_photo.key,
            "page": job.page.page_key,
            "kind": ref_photo.kind,
            "frac": ref_photo.frac,
            "band": SizeBand.of(ref_photo.frac).value,
            "surya_used": data.detector["surya_used"],
            "ref_w_mm": round((ref[2] - ref[0]) * mm_per_px, 1),
            "ref_h_mm": round((ref[3] - ref[1]) * mm_per_px, 1),
        }
        # Меры тесноты без эталона: светлый растр у сторон эталона, прежней и новой рамки (окно — по эталону с полем).
        pad = int(round(SEARCH_MM * GRAY_DPI / 25.4))
        window = tuple(int(v) for v in (ref[0] - pad, ref[1] - pad, ref[2] + pad, ref[3] + pad))
        maps = page_maps(data.gray, window, GRAY_DPI, tuple(int(v) for v in ref))
        # Растр снимка для мер тесноты: разброс или тон темнее бумаги на CORE_TONE_DELTA. Гладкий светлый тон (тень у
        # края листа, просвет оборота) снимком не считается — иначе рамка «срезала» тень у левых полос.
        light = light_map(maps) & ((maps.texture > maps.light_texture) | (maps.tone < maps.paper - CORE_TONE_DELTA))
        tight_boxes = {"ref": ref}
        for name, boxes in (("det", det), ("new", new)):
            idx, _ = best_match(ref, boxes)
            if idx >= 0:
                tight_boxes[name] = boxes[idx]
        # Объект за рамкой (вырезки, стрелы, растушёвка): краска и тон, сцепленные с эталонным снимком, вне рамки, мм².
        # Объект этого снимка: ядро — сопоставленная новая рамка (эталон бывает на два снимка сразу), прочие снимки
        # полосы вычитаются.
        ni, _ = best_match(ref, new)
        core = new[ni] if ni >= 0 else ref
        others = tuple(b for j, b in enumerate(new) if j != ni)
        # Окно и бумага — как у поиска фото (окно вокруг рамки, бумага по кольцу вокруг неё), иначе меры расходятся.
        omaps = page_maps(data.gray, tuple(int(v) for v in (core[0] - pad, core[1] - pad, core[2] + pad, core[3] + pad)),
                          GRAY_DPI, tuple(int(v) for v in core))  # fmt: skip
        obj = object_mask(omaps, core, GRAY_DPI, others)
        for name, box in tight_boxes.items():
            out_obj = object_outside(obj, omaps.origin, box, GRAY_DPI, ink=object_ink(omaps))
            row[f"{name}_obj_out_mm2"] = out_obj["total"]
            row[f"{name}_obj_out_sides"] = "/".join(f"{out_obj[s.value]:.0f}" for s in SIDES)
        for name, box in tight_boxes.items():
            for side in SIDES:
                row[f"{name}_{side.value}_in"] = round(band_cover(light, maps.origin, box, side, True, GRAY_DPI), 3)
                row[f"{name}_{side.value}_out"] = round(band_cover(light, maps.origin, box, side, False, GRAY_DPI), 3)
        for name, boxes in (("det", det), ("new", new)):
            idx, value = best_match(ref, boxes)
            row[f"{name}_iou"] = round(value, 4)
            # Сколько эталонов накрывает та же находка (две картинки одной рамкой).
            if idx >= 0:
                shared = sum(1 for r in refs if inside_share(r, boxes[idx]) > 0.5)
                row[f"{name}_shared"] = shared
                for side, d in side_deltas_mm(boxes[idx], ref, mm_per_px).items():
                    row[f"{name}_{side.value}_mm"] = round(d, 2)
        idx, value = best_match(ref, new)
        if idx >= 0:
            photo = photos[idx]
            row["new_class"] = photo.edge_class.value
            row["new_quad_reason"] = photo.quad_reason
            row["new_index"] = idx
            for fit, sharp in zip(photo.sides, photo.sharp_sides):
                s = fit.side.value
                row[f"{s}_cover"] = round(fit.cover, 3)
                row[f"{s}_resid_p90_mm"] = round(fit.resid_p90_mm, 3) if np.isfinite(fit.resid_p90_mm) else ""
                row[f"{s}_contrast"] = round(fit.contrast_median, 1)
                row[f"{s}_width_mm"] = round(fit.width_median_mm, 3) if np.isfinite(fit.width_median_mm) else ""
                row[f"{s}_bow_mm"] = round(fit.bow_mm, 3)
                row[f"{s}_sharp"] = int(sharp)
        # Диагностика: затравка = эталон.
        rp = ref_photos[len(rows)] if len(ref_photos) == len(refs) else None
        if rp is not None:
            row["refseed_iou"] = round(iou(ref, rp.box), 4)
            row["refseed_class"] = rp.edge_class.value
        rows.append(row)
    payload = {
        "page": job.page.page_key,
        "dpi": GRAY_DPI,
        "scale": f,
        "det": [list(b) for b in det],
        "refs": [list(r) for r in refs],
        "photos": [photo_json(p) for p in photos],
    }
    out = job.out_dir / f"{job.page.page_key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False))
    # Находки без эталона рядом (лишние: детектор нашёл не фото, либо фото разбито на части).
    for i, p in enumerate(photos):
        if all(iou(p.box, r) < 0.1 for r in refs):
            rows.append({"key": f"{job.page.page_key}#extra{i}", "page": job.page.page_key, "extra": 1,
                         "new_class": p.edge_class.value})  # fmt: skip
    return rows


def inside_share(ref, box) -> float:
    """Доля площади эталона ``ref`` внутри рамки ``box``."""
    ix = max(0.0, min(ref[2], box[2]) - max(ref[0], box[0]))
    iy = max(0.0, min(ref[3], box[3]) - max(ref[1], box[1]))
    area = (ref[2] - ref[0]) * (ref[3] - ref[1])
    return ix * iy / area if area > 0 else 0.0


def sides_of(row: dict, method: str) -> list[float]:
    """Промахи сторон строки мер одного способа (пустые пропускаются)."""
    out = []
    for side in SIDES:
        v = row.get(f"{method}_{side.value}_mm")
        if v not in (None, ""):
            out.append(float(v))
    return out
