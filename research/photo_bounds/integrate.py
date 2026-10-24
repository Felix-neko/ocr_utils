"""Встраивание фото в боевое выпрямление: прогон кадра ``text_frame`` с настройками фото (выключено / вес сторон без ограничения / с ограничением у текста), меры текста и прямизны сторон фото на итоге, листы «было | стало» по страницам с заметной разницей."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from research.photo_bounds.rectify import panel, photos_on, straightness

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Setting:
    """Настройка фото во втором преобразовании.

    Поля:
        name: Имя (подпапка и столбец сводки).
        photos: ``TextParams.photos``.
        near_mm: ``TextParams.photo_near_mm``.
        near_weight: ``TextParams.photo_near_weight``.
    """

    name: str
    photos: bool
    near_mm: float = 10.0
    near_weight: float = 0.25


@dataclass(frozen=True)
class IntegrateJob:
    """Кадр и настройки: набор, исходный кадр, вход, кэш v2, итоги v2, плотность бумаги, настройки, папка выхода."""

    label: str
    src: Path
    input_dir: Path
    cache: Path
    v2_output: Path
    px_per_mm: float
    settings: tuple[Setting, ...]
    out_dir: Path


# Параметры поля — как у прогона v4 (run_page_rectify_text.sh): шаг 3 мм, у корешка 3 мм.
FIELD_STEP_MM = 3.0
GUTTER_SMOOTH_MM = 3.0


def photo_seeds(first: np.ndarray, boxes: list, dpi: float) -> list[tuple[int, int, int, int]]:
    """Рамки фото с чётким краем на холсте после Кунса (общем для всех настроек) — затравки мер итога.

    Args:
        first: Холст после Кунса.
        boxes: Вырезки страниц.
        dpi: Разрешение.

    Returns:
        Рамки px холста.
    """
    from ocr_utils.scan_cropping.v2.rectify.text_field import page_regions

    seeds = []
    for box in boxes:
        for region in page_regions(first, box, dpi, True):
            if region.photo is not None:
                seeds.append(tuple(int(round(v)) for v in region.box))
    return seeds


def run_frame(job: IntegrateJob) -> dict:
    """Кадр всеми настройками: итоги (JPEG), меры текста (``TextMeasures``) и прямизна сторон фото на итоге.

    Args:
        job: Кадр.

    Returns:
        Сводка кадра; итоги и ``summary.json`` — в ``out_dir/<набор>/<кадр>/``.
    """
    from ocr_utils.scan_cropping.v2.pipeline import FrameJob
    from ocr_utils.scan_cropping.v2.rectify.coons import TANGENT_MM
    from ocr_utils.scan_cropping.v2.rectify.text_field import (
        ACROSS_SMOOTH_MM,
        CROP_MARGIN_MM,
        FIELD_SMOOTH_MM,
        crop_box,
        mm_to_px_float,
    )
    from ocr_utils.scan_cropping.v2.rectify.text_run import TextParams, text_frame
    from ocr_utils.scan_cropping.v2.spread_lines import SpineMethod
    from research.page_rectify.cli import V2_SUFFIX, RunConfig, Status, load_frame

    frame = FrameJob(job.src, job.src.relative_to(job.input_dir))
    out = job.out_dir / job.label / frame.rel.with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    bg_veto = (job.cache / frame.key / "book_veto.npz").exists()
    cfg = RunConfig(job.input_dir, job.cache, job.v2_output, out / "copy", None, bg_veto, SpineMethod.AUTO, TANGENT_MM,
                    False, job.label, job.px_per_mm)  # fmt: skip
    loaded = load_frame(frame, cfg, job.v2_output / frame.rel.with_suffix(V2_SUFFIX), out / "copy.tiff", {})
    if isinstance(loaded, Status):
        return {"label": job.label, "frame": str(frame.rel), "status": loaded.value}
    image, dpi, geometry = loaded
    summary = {"label": job.label, "frame": str(frame.rel), "dpi": dpi, "status": "ok", "settings": {}}
    seeds = None
    for setting in job.settings:
        params = TextParams(TANGENT_MM, FIELD_STEP_MM, FIELD_SMOOTH_MM, GUTTER_SMOOTH_MM, ACROSS_SMOOTH_MM,
                            photos=setting.photos, photo_near_mm=setting.near_mm,
                            photo_near_weight=setting.near_weight)  # fmt: skip
        result = text_frame(image, geometry, dpi, params)
        if seeds is None:
            margin = mm_to_px_float(CROP_MARGIN_MM, dpi)
            boxes = [crop_box(r, result.first.shape[:2], margin, result.layout.spine_x, i == 0)
                     for i, r in enumerate(result.layout.rects)]  # fmt: skip
            seeds = photo_seeds(result.first, boxes, dpi)
            cv2.imwrite(str(out / "first.jpg"), result.first, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(out / f"{setting.name}.jpg"), result.final, [cv2.IMWRITE_JPEG_QUALITY, 90])
        photos = [[s.__dict__ | {"side": s.side.value} for s in straightness(p)]
                  for p in photos_on(result.final, seeds, dpi)]  # fmt: skip
        summary["settings"][setting.name] = {
            "lines": [int(p.lines) for p in result.pages],
            "modes": [p.mode.value for p in result.pages],
            **result.before.row("before_"),
            **result.after.row("after_"),
            "photo_sides": photos,
        }
    summary["seeds"] = [list(s) for s in seeds or []]
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    return summary


def compare_sheet(frame_dir: Path, before: str, after: str, threshold_mm: float) -> list[Path]:
    """Листы «было | стало» кадра: у каждого фото — итог двух настроек со сторонами фото; страница целиком, если
    мера текста изменилась больше ``threshold_mm``.

    Args:
        frame_dir: Папка кадра (``summary.json``, итоги настроек).
        before, after: Имена настроек.
        threshold_mm: Порог заметного изменения меры текста (p95 размаха строк или сторон), мм.

    Returns:
        Пути листов.
    """
    from ocr_utils.page_layout.overlay_frame import LegendEntry, framed
    from ocr_utils.page_layout.text_blocks.sides_overlay import side_by_side
    from research.photo_bounds.rectify import COLOUR_CHORD, COLOUR_SIDE

    summary = json.loads((frame_dir / "summary.json").read_text())
    dpi = float(summary["dpi"])
    a, b = summary["settings"][before], summary["settings"][after]
    img_a, img_b = cv2.imread(str(frame_dir / f"{before}.jpg")), cv2.imread(str(frame_dir / f"{after}.jpg"))
    legend = [LegendEntry("подогнанная сторона фото (чёткая)", COLOUR_SIDE),
              LegendEntry("прямая через концы стороны", COLOUR_CHORD)]  # fmt: skip
    head = [f"{summary['label']}/{summary['frame']}: «{before}» | «{after}»",
            f"строки p95 {a['after_line_p95_mm']} → {b['after_line_p95_mm']} мм, наклон медиана "
            f"{a['after_slope_med_deg']} → {b['after_slope_med_deg']}°, стороны блоков p95 {a['after_side_p95_mm']} → "
            f"{b['after_side_p95_mm']} мм; режимы {a['modes']} → {b['modes']}"]  # fmt: skip
    paths = []
    for i, seed in enumerate(summary["seeds"]):
        views, titles = [], []
        for name, img in ((before, img_a), (after, img_b)):
            found = photos_on(img, [tuple(seed)], dpi)
            view, _, _ = panel(img, seed, found[0] if found else None, dpi)
            stats = straightness(found[0]) if found else []
            views.append(view)
            titles.append(f"{name}: " + ", ".join(f"{s.side.value[0]} {s.tilt_deg:.2f}/{s.ptp_mm:.2f}" for s in stats))
        path = frame_dir / f"compare_photo{i + 1}.jpg"
        cv2.imwrite(str(path), framed(side_by_side(views, titles), head, legend), [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(path)
    changed = any(abs(float(a[k]) - float(b[k])) > threshold_mm
                  for k in ("after_line_p95_mm", "after_side_p95_mm")
                  if np.isfinite(float(a[k])) and np.isfinite(float(b[k])))  # fmt: skip
    if changed:
        f = 1400 / img_a.shape[1]
        views = [cv2.resize(x, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) for x in (img_a, img_b)]
        path = frame_dir / "compare_page.jpg"
        cv2.imwrite(
            str(path), framed(side_by_side(views, [before, after]), head, legend), [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        paths.append(path)
    return paths


def region_sheet(frame_dir: Path, dpi: float) -> Path:
    """Лист «рамка растра | область фото» на холсте после Кунса (``first.jpg``): прежние рамки детектора (оранжевые) и
    новые области (рамка по снимку или охват объекта — синие, четырёхугольник — фиолетовый).

    Args:
        frame_dir: Папка кадра integrate.
        dpi: Разрешение холста.

    Returns:
        Путь листа ``regions.jpg``.
    """
    from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
    from ocr_utils.page_layout.text_blocks.sides_overlay import side_by_side
    from ocr_utils.scan_cropping.v2.rectify.text_field import RegionKindR, page_regions

    first = cv2.imread(str(frame_dir / "first.jpg"))
    h, w = first.shape[:2]
    views = []
    for photos in (False, True):
        view = first.copy()
        for half in ((0, 0, w // 2, h), (w // 2, 0, w, h)):
            for region in page_regions(first, half, dpi, photos):
                if region.kind is not RegionKindR.RASTER:
                    continue
                if region.polygon is not None:
                    cv2.polylines(view, [np.round(region.polygon).astype(np.int32)], True, (200, 60, 200), 6)
                else:
                    x0, y0, x1, y1 = (int(round(v)) for v in region.box)
                    cv2.rectangle(view, (x0, y0), (x1, y1), (220, 90, 20) if photos else (0, 165, 255), 6)
        f = 1300 / w
        views.append(cv2.resize(view, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))
    legend = [LegendEntry("рамка растра детектора (было)", (0, 165, 255)),
              LegendEntry("область фото: рамка по снимку или охват объекта (стало)", (220, 90, 20)),
              LegendEntry("кривой четырёхугольник (все стороны чёткие)", (200, 60, 200), style=SampleStyle.LINE)]  # fmt: skip
    path = frame_dir / "regions.jpg"
    cv2.imwrite(str(path), framed(side_by_side(views, ["было", "стало"]), [str(frame_dir)], legend),
                [cv2.IMWRITE_JPEG_QUALITY, 85])  # fmt: skip
    return path
