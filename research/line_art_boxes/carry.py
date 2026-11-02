"""Сверка переноса областей второго преобразования за содержимым (``TextParams.carry_regions``): отставание рамок от рисунков на итоге, меры строк и сторон, листы «было | стало»."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.frame_flags.flags import Side, read_flags
from ocr_utils.page_layout.overlay_frame import header_strip
from ocr_utils.scan_cropping.v2.pipeline import read_bgr
from ocr_utils.scan_cropping.v2.rectify.sheets import sheet_result
from ocr_utils.scan_cropping.v2.rectify.text_field import CROP_MARGIN_MM, RegionKindR, crop_box, page_regions
from ocr_utils.scan_cropping.v2.rectify.text_run import PageControl, TextParams, TextResult, text_frame
from ocr_utils.scan_cropping.v2.spread_lines import SpineMethod
from ocr_utils.units import MM_PER_INCH, mm_to_px_float
from research.line_art_boxes.canvas import V2_SUFFIX, RunPaths
from research.page_rectify.frames import cached_frames

# Какие области сверяются: рамки, которые детектор находит заново по итогу (формулы — из кэша surya, их не пересчитать).
CHECKED_KINDS = (RegionKindR.LINE_ART, RegionKindR.RASTER, RegionKindR.TABLE)


def run_frame(paths: RunPaths, issue: str, frame: str, px_per_mm: float, carry: bool) -> TextResult | None:
    """Второе преобразование кадра целиком (как в прогоне v2) с переносом областей или без.

    Args:
        paths: Пути набора.
        issue: Подпапка выпуска.
        frame: Имя кадра без расширения.
        px_per_mm: Плотность бумаги выпуска, px/мм.
        carry: ``TextParams.carry_regions``.

    Returns:
        Итог ``text_frame`` или None — у кадра нет двух областей страниц или нет итога v2.
    """
    src = paths.photos / issue / f"{frame}.jpg"
    v2_path = paths.output / issue / f"{frame}{V2_SUFFIX}"
    if not v2_path.exists() or not src.exists():
        return None
    image = read_bgr(v2_path)
    dpi = px_per_mm * MM_PER_INCH
    geometry = cached_frames(src, paths.cache / issue, frame, True, SpineMethod.AUTO)
    if geometry.box is None or geometry.frames is None or len(geometry.frames) != 2:
        return None
    flags = read_flags(src)
    controls = tuple(
        PageControl(equalize=flags.equalize(s), field=not flags.ignored(s) and flags.text_rectify(s))
        for s in (Side.LEFT, Side.RIGHT)
    )
    return text_frame(image, geometry, dpi, replace(TextParams(), carry_regions=carry), controls)


def lag_mm(result: TextResult, dpi: float) -> list[dict]:
    """Отставание рамок разбора итога от рамок, найденных детектором заново по итогу (по каждой стороне), мм.

    Args:
        result: Итог ``text_frame``.
        dpi: Разрешение.

    Returns:
        По записи на рамку разбора итога: страница, вид, рамка (мм), наибольшее расхождение сторон с парной свежей
        рамкой (мм) или None — пары нет (свежая рамка не пересекается).
    """
    k = dpi / MM_PER_INCH
    margin = mm_to_px_float(CROP_MARGIN_MM, dpi)
    rows = []
    for i, (page, rect) in enumerate(zip(result.analyses[-1], result.canvas.rects)):
        if page is None:
            continue
        box = crop_box(rect, result.final.shape[:2], margin, result.canvas.spine_x, i == 0)
        fresh = [r for r in page_regions(result.final, box, dpi, False) if r.kind in CHECKED_KINDS]
        for region in page.regions:
            if region.kind not in CHECKED_KINDS:
                continue
            u = region.box
            same = [f.box for f in fresh if f.kind is region.kind]
            overlap = [
                max(0, min(u[2], f[2]) - max(u[0], f[0])) * max(0, min(u[3], f[3]) - max(u[1], f[1])) for f in same
            ]
            if not overlap or max(overlap) == 0:
                rows.append(
                    {"page": "LR"[i], "kind": region.kind.value, "box_mm": [round(v / k, 1) for v in u], "lag": None}
                )
                continue
            f = same[int(np.argmax(overlap))]
            rows.append({"page": "LR"[i], "kind": region.kind.value, "box_mm": [round(v / k, 1) for v in u],
                         "lag": round(max(abs(a - b) for a, b in zip(u, f)) / k, 2)})  # fmt: skip
    return rows


def content_lag_mm(result: TextResult, dpi: float) -> list[dict]:
    """Отставание содержимого от рамки на итоге: сдвиг рисунка внутри рамки итога против рамки холста Кунса, мм.

    Для каждой области берутся вырезка холста Кунса по исходной рамке (``analyses[0]``) и вырезка итога по рамке
    разбора итога (``analyses[-1]``: с переносом или без), приведённая к тому же размеру; фазовая корреляция серых
    вырезок даёт остаточный сдвиг содержимого относительно рамки. Мера не зависит от того, как детектор нашёл бы
    рамку заново: 0 — рамка стоит на рисунке так же, как на холсте Кунса.

    Args:
        result: Итог ``text_frame``.
        dpi: Разрешение.

    Returns:
        По записи на область: страница, вид, остаточный сдвиг (мм).
    """
    k = dpi / MM_PER_INCH
    first = cv2.cvtColor(result.first, cv2.COLOR_BGR2GRAY).astype(np.float32)
    final = cv2.cvtColor(result.final, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows = []
    for i, (before, after) in enumerate(zip(result.analyses[0], result.analyses[-1])):
        if before is None or after is None:
            continue
        for a, b in zip(before.regions, after.regions):
            x0, y0, x1, y1 = (int(round(v)) for v in a.box)
            u0, v0, u1, v1 = (int(round(v)) for v in b.box)
            if x1 - x0 < 16 or y1 - y0 < 16 or u1 <= u0 or v1 <= v0:
                continue
            ref = first[max(0, y0) : y1, max(0, x0) : x1]
            cur = final[max(0, v0) : v1, max(0, u0) : u1]
            if ref.size == 0 or cur.size == 0:
                continue
            cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
            (sx, sy), _ = cv2.phaseCorrelate(ref, cur)
            rows.append({"page": "LR"[i], "kind": a.kind.value, "lag": round(float(np.hypot(sx, sy)) / k, 2)})
    return rows


def compare_frame(paths: RunPaths, issue: str, frame: str, px_per_mm: float, out_dir: Path) -> dict | None:
    """Кадр без переноса и с переносом: отставание рамок, меры итога, лист «было | стало».

    Args:
        paths: Пути набора.
        issue: Подпапка выпуска.
        frame: Имя кадра.
        px_per_mm: Плотность выпуска.
        out_dir: Папка листов и JSON.

    Returns:
        Запись сверки или None — кадр не выпрямляется.
    """
    dpi = px_per_mm * MM_PER_INCH
    record = {"frame": f"{issue}/{frame}"}
    sheets = []
    for carry in (False, True):
        result = run_frame(paths, issue, frame, px_per_mm, carry)
        if result is None:
            return None
        record["carry" if carry else "fixed"] = {
            "lag": lag_mm(result, dpi),
            "content": content_lag_mm(result, dpi),
            "measures": result.passes[-1].row(""),
        }
        title = "стало: области за содержимым" if carry else "было: области на месте холста Кунса"
        sheets.append(sheet_result(result, [f"{issue}/{frame} — {title}"]))
    width = max(s.shape[1] for s in sheets)
    sheets = [np.hstack([s, np.full((s.shape[0], width - s.shape[1], 3), 255, np.uint8)]) for s in sheets]
    sheet = np.vstack([header_strip([f"{issue}/{frame}: перенос областей второго преобразования — было | стало"], width)]
                      + sheets)  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{issue[:2]}_{frame}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out_dir / f"{issue[:2]}_{frame}.json").write_text(json.dumps(record, ensure_ascii=False))
    return record
