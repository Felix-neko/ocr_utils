"""Строки и выборки выпрямления на полосах пака-1 при отсеве по рамке растра детектора и по области фото стенда (кривой четырёхугольник или уточнённая рамка): сколько строк возвращается и какие полосы с фото почти без текста."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from research.photo_bounds.cache import GRAY_DPI, load_page
from research.photo_bounds.data import CacheLayout, PageRef
from research.photo_bounds.run import detector_boxes
from research.photo_bounds.measure import scale_box

# Полоса «почти без текста»: годных строк (от ``text_field.MIN_RUN_MM``) вне фото меньше этого — столько же, сколько
# нужно выпрямлению для полного режима поля (``text_field.MIN_LINES``).
FEW_LINES = 8


@dataclass(frozen=True)
class TextJob:
    """Задание: полоса, раскладка кэша, папка прогона стенда (``pages/<полоса>.json``)."""

    page: PageRef
    layout: CacheLayout
    run_dir: Path


def regions_for(boxes: list, photos: list[dict] | None):
    """Области страницы для разбора: рамки детектора (``photos is None``) или области фото стенда.

    Args:
        boxes: Рамки детектора px 300 dpi.
        photos: JSON фото прогона стенда или None.

    Returns:
        Кортеж ``PageRegion``/``PhotoRegion``.
    """
    from ocr_utils.scan_cropping.v2.rectify.text_field import PageRegion, RegionKindR
    from research.photo_bounds.supports import PhotoRegion

    if photos is None:
        return tuple(PageRegion(RegionKindR.RASTER, tuple(float(v) for v in b)) for b in boxes)
    out = []
    for p in photos:
        polygon = np.array(p["polygon"], np.float64)
        box = (float(polygon[:, 0].min()), float(polygon[:, 1].min()), float(polygon[:, 0].max()),
               float(polygon[:, 1].max()))  # fmt: skip
        out.append(PhotoRegion(RegionKindR.RASTER, box, polygon, None))
    return tuple(out)


def count_page(job: TextJob) -> dict:
    """Разбор полосы (``text_field.analyze_page``, движок по краске) с областями по рамке детектора и по области фото;
    число строк в выборках (``line_samples``), число выборок, строки, вернувшиеся из зоны отсева.

    Args:
        job: Задание.

    Returns:
        Строка мер полосы.
    """
    from ocr_utils.scan_cropping.v2.rectify.coons import PageRect
    from ocr_utils.scan_cropping.v2.rectify.text_field import analyze_page, line_samples

    data = load_page(job.layout, job.page)
    payload = json.loads((job.run_dir / "pages" / f"{job.page.page_key}.json").read_text())
    det = [scale_box(b, data.scale) for b in detector_boxes(data.detector)]
    canvas = cv2.cvtColor(data.gray, cv2.COLOR_GRAY2BGR)
    h, w = data.gray.shape
    rect = PageRect(0.0, 0.0, float(w), float(h))
    row = {"page": job.page.page_key, "photos": len(payload["photos"])}
    for name, regions in (("rect", regions_for(det, None)), ("quad", regions_for(det, payload["photos"]))):
        page = analyze_page(canvas, (0, 0, w, h), GRAY_DPI, rect, regions)
        samples, count = line_samples(page, GRAY_DPI)
        # Проверка: ни одна выборка строки не внутри области растра (рамки детектора, рамки или многоугольника фото,
        # охвата вырезки) — ложные «строки» внутри снимка (полка с папками, продолжение подписи через фото) отсеяны.
        inside = np.zeros(len(samples), bool)
        for region in regions:
            if len(samples):
                inside |= region.contains(samples.source)
        row[f"{name}_inside_raster"] = int(inside.sum())
        row[f"{name}_lines"] = count
        row[f"{name}_points"] = len(samples)
        row[f"{name}_axes"] = len(page.analysis.axes)
    # Строки текста, задетые областью растра: разбор без преград (оси строк страницы как есть), строка «задета», если
    # хоть одна точка её оси внутри области. Внутри снимка «строки» тоже есть (штриховка) — они задеты обеими областями.
    from ocr_utils.scan_cropping.v2.rectify.text_field import line_curve

    plain = analyze_page(canvas, (0, 0, w, h), GRAY_DPI, rect, ())
    for name, regions in (("rect", regions_for(det, None)), ("quad", regions_for(det, payload["photos"]))):
        hit = 0
        for axis in plain.analysis.axes:
            pts = plain.to_canvas(line_curve(axis))
            if len(pts) and any(region.contains(pts).any() for region in regions):
                hit += 1
        row[f"{name}_lines_hit"] = hit
    row["axes_total"] = len(plain.analysis.axes)
    # Строки страницы без всякого отсева — для «почти без текста».
    row["returned_lines"] = row["quad_lines"] - row["rect_lines"]
    row["returned_points"] = row["quad_points"] - row["rect_points"]
    row["few_text"] = int(row["quad_lines"] < FEW_LINES)
    return row


# Цвета листа «рамка | четырёхугольник» (BGR, палитра draw-overlay): выборка строки принята / отсеяна.
COLOUR_KEPT = (0, 150, 0)
COLOUR_DROPPED = (0, 0, 220)
COLOUR_RECT = (0, 165, 255)
COLOUR_POLY = (200, 60, 200)


def returned_sheet(job: TextJob, out: Path) -> None:
    """Лист «было | стало» полосы: слева отсев по рамке детектора, справа — по области фото; точки выборок строк
    (зелёные — в опорах, красные — отсеяны зоной фото), рамка и многоугольник.

    Args:
        job: Полоса.
        out: Путь JPEG.
    """
    from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
    from ocr_utils.page_layout.text_blocks.sides_overlay import side_by_side
    from ocr_utils.scan_cropping.v2.rectify.coons import PageRect
    from ocr_utils.scan_cropping.v2.rectify.text_field import (
        RULE_ZONE_MARGIN_MM,
        analyze_page,
        line_curve,
        mm_to_px_float,
    )

    data = load_page(job.layout, job.page)
    payload = json.loads((job.run_dir / "pages" / f"{job.page.page_key}.json").read_text())
    det = [scale_box(b, data.scale) for b in detector_boxes(data.detector)]
    canvas = cv2.cvtColor(data.gray, cv2.COLOR_GRAY2BGR)
    h, w = data.gray.shape
    rect = PageRect(0.0, 0.0, float(w), float(h))
    margin = mm_to_px_float(RULE_ZONE_MARGIN_MM, GRAY_DPI)
    views, titles = [], []
    for name, regions in (
        ("рамка детектора", regions_for(det, None)),
        ("область фото", regions_for(det, payload["photos"])),
    ):
        page = analyze_page(canvas, (0, 0, w, h), GRAY_DPI, rect, regions)
        view = canvas.copy()
        for region in regions:
            if getattr(region, "polygon", None) is not None and len(region.polygon) >= 3:
                cv2.polylines(view, [np.round(region.polygon).astype(np.int32)], True, COLOUR_POLY, 3, cv2.LINE_AA)
            else:
                x0, y0, x1, y1 = (int(round(v)) for v in region.box)
                cv2.rectangle(view, (x0, y0), (x1, y1), COLOUR_RECT, 3)
        kept = dropped = 0
        for axis in page.analysis.axes:
            pts = page.to_canvas(line_curve(axis))
            if len(pts) < 2 or np.ptp(pts[:, 0]) < mm_to_px_float(25.0, GRAY_DPI):
                continue
            inside = np.zeros(len(pts), bool)
            for region in regions:
                inside |= region.contains(pts, margin)
            for (x, y), bad in zip(pts[::3], inside[::3]):
                cv2.circle(view, (int(x), int(y)), 4, COLOUR_DROPPED if bad else COLOUR_KEPT, -1)
            kept += int((~inside).sum())
            dropped += int(inside.sum())
        f = 900 / view.shape[1]
        views.append(cv2.resize(view, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))
        titles.append(f"{name}: точек строк в опорах {kept}, отсеяно {dropped}")
    legend = [
        LegendEntry("рамка растра детектора (отсев и преграда — было)", COLOUR_RECT),
        LegendEntry("область фото стенда (отсев и преграда — стало)", COLOUR_POLY),
        LegendEntry("точка оси строки в опорах", COLOUR_KEPT),
        LegendEntry("точка оси строки отсеяна зоной фото", COLOUR_DROPPED),
    ]
    picture = framed(side_by_side(views, titles), [f"{job.page.page_key} — строки выпрямления у фото"], legend)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
