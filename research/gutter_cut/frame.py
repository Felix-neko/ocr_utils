"""Кадр стенда «линия разреза у корешка»: области по кэшу v2 → холст Кунса → разбор текста обеих страниц (как нулевой проход выхода страниц) → мера и полоса холста у корешка с линиями."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from ocr_utils.scan_cropping.image_io import read_dpi
from ocr_utils.scan_cropping.v2.pipeline import FrameJob, read_bgr
from ocr_utils.scan_cropping.v2.rectify.coons import TANGENT_MM, page_patch, plan_layout, rectify_image, rectify_maps
from ocr_utils.scan_cropping.v2.rectify.text_field import (
    CROP_MARGIN_MM,
    PageText,
    analyze_page,
    crop_box,
    equation_regions,
    page_regions,
)
from ocr_utils.scan_cropping.v2.rectify.text_run import equalize_pages, page_mask
from ocr_utils.scan_cropping.v2.spread_lines import SpineMethod
from ocr_utils.units import MM_PER_INCH, mm_to_px_float
from research.gutter_cut.measure import TIGHT_MM, GutterMeasure, Verdict, block_rows_canvas, gutter_measure
from research.page_rectify.frames import cached_frames

# Полоса холста у корешка на листе: по стольку мм в каждую сторону от линии корешка.
STRIP_HALF_MM = 25.0
# Суффикс итогов v2 (``--output-format tiff`` боевых ключей).
V2_SUFFIX = ".tiff"

# Цвета (BGR, палитра /draw-overlay).
COLOUR_SPINE = (0, 0, 220)  # линия корешка сейчас — «отвергнуто» красным пунктиром
COLOUR_CUT = (0, 150, 0)  # уточнённая линия разреза — «принято»
COLOUR_SIDE = (220, 90, 20)  # выровненная сторона блока у корешка
COLOUR_TIGHT = (180, 105, 255)  # край ряда впритык
COLOUR_ROW = (140, 140, 0)  # край прочих рядов у корешка
COLOUR_SPINE_ALPHA = 0.8


class Status(StrEnum):
    """Исход кадра стенда."""

    OK = "ok"
    MISSING = "missing"  # нет итога v2
    NO_FRAMES = "no_frames"  # одна страница или области не построились
    CACHE_MISMATCH = "cache_mismatch"  # кэш перезаписан другим прогоном
    ERROR = "error"


@dataclass(frozen=True)
class Source:
    """Набор прогона v2, по которому меряем.

    Поля:
        label: Имя набора в сводке (``hp_1991/01 оплачено``).
        input_dir: Папка исходных кадров прогона.
        cache: Папка кэша прогона.
        v2_output: Папка итогов прогона (TIFF до выпрямления).
        px_per_mm: Плотность бумаги, px/мм (``--px-per-mm`` прогона).
    """

    label: str
    input_dir: Path
    cache: Path
    v2_output: Path
    px_per_mm: float


def measure_row(m: GutterMeasure) -> dict:
    """Плоская строка сводки по мере разворота."""
    row = {"verdict": m.verdict.value, "gap_mm": m.gap_mm, "overshoot_mm": m.overshoot_mm, "shift_mm": m.shift_mm}
    for name, side in (("l", m.left), ("r", m.right)):
        for key, value in asdict(side).items():
            if key not in ("y0", "y1"):
                row[f"{name}_{key}"] = value
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}


def draw_strip(first: np.ndarray, spine_x: float, pages: list[PageText], m: GutterMeasure, dpi: float,
               header: list[str]) -> np.ndarray:  # fmt: skip
    """Полоса холста Кунса у корешка с линией корешка, уточнённой линией разреза, сторонами и краями рядов.

    Args:
        first: Холст Кунса BGR.
        spine_x: X линии корешка.
        pages: Разборы левой и правой страницы.
        m: Мера разворота.
        dpi: Разрешение холста.
        header: Строки шапки.

    Returns:
        Картинка с шапкой и легендой в полях.
    """
    half = int(mm_to_px_float(STRIP_HALF_MM, dpi))
    x0 = max(int(spine_x) - half, 0)
    x1 = min(int(spine_x) + half, first.shape[1])
    canvas = first[:, x0:x1].copy()
    thick = max(2, int(round(mm_to_px_float(0.25, dpi))))
    tight_px = mm_to_px_float(TIGHT_MM, dpi)
    # Края рядов со стороны корешка: впритык — розовым, прочие у корешка (до 8 мм) — бирюзовым.
    for left_page, page in zip((True, False), pages):
        for block in page.analysis.blocks:
            for y, h, rx0, rx1 in block_rows_canvas(page, block):
                edge = rx1 if left_page else rx0
                near = (spine_x - edge) if left_page else (edge - spine_x)
                if near > mm_to_px_float(8.0, dpi):
                    continue
                colour = COLOUR_TIGHT if near <= tight_px else COLOUR_ROW
                px = int(round(edge - x0))
                cv2.line(canvas, (px, int(y - h / 2)), (px, int(y + h / 2)), colour, thick)
    # Выровненные стороны у корешка, найденные мерой.
    for side, page, left_page in ((m.left, pages[0], True), (m.right, pages[1], False)):
        if side.flush_mm <= 0:
            continue
        x = (spine_x - mm_to_px_float(side.flush_dist_mm, dpi)) if left_page else (
            spine_x + mm_to_px_float(side.flush_dist_mm, dpi))  # fmt: skip
        cv2.line(canvas, (int(round(x - x0)), int(side.y0)), (int(round(x - x0)), int(side.y1)), COLOUR_SIDE, thick)
    # Линия корешка — пунктиром, уточнённая линия разреза — сплошной.
    sx = int(round(spine_x - x0))
    dash = int(mm_to_px_float(3.0, dpi))
    layer = canvas.copy()
    for y in range(0, canvas.shape[0], 2 * dash):
        cv2.line(layer, (sx, y), (sx, min(y + dash, canvas.shape[0] - 1)), COLOUR_SPINE, thick)
    cv2.addWeighted(layer, COLOUR_SPINE_ALPHA, canvas, 1 - COLOUR_SPINE_ALPHA, 0, canvas)
    if m.verdict is Verdict.SHIFT:
        cx = int(round(m.cut_x - x0))
        cv2.line(canvas, (cx, 0), (cx, canvas.shape[0] - 1), COLOUR_CUT, thick)
    return framed(canvas, header, [
        LegendEntry("линия корешка (разрез сейчас)", COLOUR_SPINE, COLOUR_SPINE_ALPHA, SampleStyle.DASHED),
        LegendEntry("уточнённая линия разреза", COLOUR_CUT),
        LegendEntry("выровненная сторона блока у корешка", COLOUR_SIDE),
        LegendEntry(f"край ряда впритык (≤ {TIGHT_MM} мм)", COLOUR_TIGHT),
        LegendEntry("край ряда у корешка (≤ 8 мм)", COLOUR_ROW),
    ])  # fmt: skip


def measure_frame(args: tuple[Path, Source, Path | None]) -> dict:
    """Мера одного кадра (для пула): кэш v2 → холст Кунса → разбор → мера → полоса у корешка.

    Args:
        args: ``(исходный кадр, набор, папка полос или None — без картинок)``.

    Returns:
        Строка сводки.
    """
    src, source, strips = args
    job = FrameJob(src, src.relative_to(source.input_dir))
    row: dict = {"set": source.label, "key": job.key, "src": str(src), "status": Status.OK.value}
    v2_path = source.v2_output / job.rel.with_suffix(V2_SUFFIX)
    if not v2_path.exists():
        return row | {"status": Status.MISSING.value}
    t0 = time.perf_counter()
    try:
        row |= measure_one(job, source, v2_path, strips)
    except Exception as exc:  # кадр с ошибкой не роняет прогон
        row |= {"status": Status.ERROR.value, "message": repr(exc)}
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def measure_one(job: FrameJob, source: Source, v2_path: Path, strips: Path | None) -> dict:
    """Тело :func:`measure_frame`.

    Args:
        job: Кадр.
        source: Набор.
        v2_path: Итог v2.
        strips: Папка полос у корешка или None.

    Returns:
        Поля сводки (исход, меры, путь полосы).
    """
    image = read_bgr(v2_path)
    dpi = source.px_per_mm * MM_PER_INCH if source.px_per_mm else read_dpi(v2_path)
    # Вето фона — по кэшу: старые прогоны (пак-6 pack6_v2) шли без него и book_veto.npz не писали.
    bg_veto = (source.cache / job.key / "book_veto.npz").exists()
    geometry = cached_frames(job.src, source.cache, job.key, bg_veto, SpineMethod.AUTO)
    if geometry.box is None or geometry.frames is None or len(geometry.frames) != 2:
        return {"status": Status.NO_FRAMES.value}
    x0, y0, x1, y1 = geometry.box
    if (y1 - y0, x1 - x0) != image.shape[:2]:
        return {"status": Status.CACHE_MISMATCH.value}
    tangent = mm_to_px_float(TANGENT_MM, dpi)
    patches = tuple(page_patch(f, i == 0, geometry.to_output, tangent) for i, f in enumerate(geometry.frames))
    layout = plan_layout(patches, image.shape[:2])
    masks = [page_mask(p, image.shape[:2]) for p in patches]
    # Освещение — как в выходе страниц (цветные страницы стенд не отделяет: мера у корешка по краске текста).
    equalized, _ = equalize_pages(image, masks, [False, False], dpi)
    maps = rectify_maps(patches, layout)
    first = rectify_image(equalized, maps, dpi)
    margin = mm_to_px_float(CROP_MARGIN_MM, dpi)
    boxes = [crop_box(rect, first.shape[:2], margin, layout.spine_x, i == 0) for i, rect in enumerate(layout.rects)]
    pages = []
    for box, rect in zip(boxes, layout.rects):
        regions = page_regions(first, box, dpi, True) + equation_regions(geometry.equations, geometry.frame_to_output,
                                                                         maps, rect)  # fmt: skip
        pages.append(analyze_page(first, box, dpi, rect, regions))
    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    m = gutter_measure(gray, layout, pages, dpi)
    out = {"status": Status.OK.value, "dpi": round(dpi, 1)} | measure_row(m)
    if strips is not None:
        header = [f"{source.label} {job.rel} — вердикт {m.verdict.value}, сдвиг {m.shift_mm:.1f} мм, зазор "
                  f"{m.gap_mm:.1f} мм, выход краски {m.overshoot_mm:.1f} мм",
                  f"L: сторона {m.left.flush_mm:.0f} мм в {m.left.flush_dist_mm:.1f} мм, впритык {m.left.tight_rows}"
                  f"/{m.left.rows}  |  R: сторона {m.right.flush_mm:.0f} мм в {m.right.flush_dist_mm:.1f} мм, впритык "
                  f"{m.right.tight_rows}/{m.right.rows}"]  # fmt: skip
        path = strips / source.label / f"{job.key}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), draw_strip(first, layout.spine_x, pages, m, dpi, header), [cv2.IMWRITE_JPEG_QUALITY, 88])
        out["strip"] = str(path)
    return out
