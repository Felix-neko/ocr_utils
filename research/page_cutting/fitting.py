"""Прогон рамок и выпрямления по кадру: варианты рамки (трапеция, полиномы, сплайны), выпрямленные страницы, прямизна строк и оверлей рамок на кадре."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, framed, header_strip
from research.page_cutting.boundary import FitKind, PageBoundary, fit_boundary
from research.page_cutting.cases import FrameCase, load_case
from research.page_cutting.geometry import Side
from research.page_cutting.runs import RunSpec, read_image, write_image
from research.page_cutting.straightness import measure_straightness
from research.page_cutting.warp import plan_page_warp, warp_page

# Цвета рамок (BGR, палитра ``/draw-overlay``): трапеция — серый (справочная), полиномы — оранжевый, сплайны — синий;
# маска страницы — зелёный тонко, вершины текста surya — розовый.
KIND_COLOURS = {FitKind.QUAD: (150, 150, 150), FitKind.POLY: (0, 165, 255), FitKind.SPLINE: (220, 90, 20)}
COLOUR_MASK = (0, 150, 0)
COLOUR_TEXT = (180, 105, 255)
# Сгиб: сглаженный — фиолетовый (крупная линия), шов до сглаживания — приглушённый тон.
COLOUR_FOLD = (200, 60, 200)
COLOUR_FOLD_RAW = (200, 170, 200)
# Край страницы у корешка по цвету бумаги (сторона у корешка poly/spline) — красный.
COLOUR_EDGE = (0, 0, 220)
# Ширина оверлея кадра, px.
OVERLAY_WIDTH = 1600
# Высота выпрямленной страницы на листе сравнения вариантов, px.
COMPARE_HEIGHT = 1100
# Сетка на выпрямленной странице листа сравнения: шаг горизонталей, мм (видно, легли ли строки вдоль), цвет и прозрачность.
GRID_STEP_MM = 20.0
COLOUR_GRID = (200, 140, 60)
GRID_ALPHA = 0.5


@dataclass(frozen=True)
class PageRow:
    """Строка таблицы мер: страница кадра в одном варианте рамки.

    Поля:
        run, key: Кадр.
        page: ``L``, ``R`` или ``S`` (одна страница).
        kind: Вариант рамки.
        resid_top_mm, resid_bottom_mm, resid_spine_mm, resid_far_mm: 90-й перцентиль отклонения контура маски от
            кривой стороны, мм.
        shift_max_mm: Наибольший сдвиг стороны наружу (накрыть зубцы и текст), мм.
        text_outside: Доля вершин текста за рамкой до сдвига.
        width, height: Размер выпрямленной страницы, px кадра.
        lines, slope_med_deg, slope_spread_deg, sagitta_p90_mm, bend_p90_mm, resid_p90_mm: Прямизна строк
            выпрямленной страницы (``straightness.Straightness``).
    """

    run: str
    key: str
    page: str
    kind: str
    resid_top_mm: float
    resid_bottom_mm: float
    resid_spine_mm: float
    resid_far_mm: float
    shift_max_mm: float
    text_outside: float
    width: int
    height: int
    lines: int
    slope_med_deg: float
    slope_spread_deg: float
    sagitta_p90_mm: float
    bend_p90_mm: float
    resid_p90_mm: float


def page_name(case: FrameCase, index: int) -> str:
    """Имя страницы кадра: ``L``/``R`` для разворота, ``S`` для одной страницы."""
    return "LR"[index] if len(case.pages) == 2 else "S"


def draw_overlay(case: FrameCase, boundaries: dict[str, dict[FitKind, PageBoundary]]) -> np.ndarray:
    """Оверлей кадра: маски страниц, текст surya и рамки всех вариантов поверх ``seg.png``, с шапкой и легендой.

    Args:
        case: Кадр.
        boundaries: Страница → вариант → рамка (px сегментации).

    Returns:
        Картинка BGR шириной ``OVERLAY_WIDTH`` с полями.
    """
    seg = read_image(case.run.cache_dir / case.key / "seg.png")
    scale = OVERLAY_WIDTH / seg.shape[1]
    canvas = cv2.resize(seg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for page in case.pages:
        contours, _ = cv2.findContours(page.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        scaled = [np.round(c * scale).astype(np.int32) for c in contours]
        cv2.drawContours(canvas, scaled, -1, COLOUR_MASK, 1, cv2.LINE_AA)
        for x, y in page.text * scale:
            cv2.circle(canvas, (int(x), int(y)), 2, COLOUR_TEXT, -1)
    if case.fold is not None:
        cv2.polylines(
            canvas, [np.round(case.fold.raw * scale).astype(np.int32)], False, COLOUR_FOLD_RAW, 1, cv2.LINE_AA
        )
        cv2.polylines(canvas, [np.round(case.fold.points * scale).astype(np.int32)], False, COLOUR_FOLD, 2, cv2.LINE_AA)
    for page in case.pages:
        if page.spine_edge is not None:
            pts = np.round(page.spine_edge * scale).astype(np.int32)
            cv2.polylines(canvas, [pts], False, COLOUR_EDGE, 1, cv2.LINE_AA)
    for by_kind in boundaries.values():
        for kind, b in by_kind.items():
            poly = np.round(b.polygon() * scale).astype(np.int32)
            cv2.polylines(canvas, [poly], True, KIND_COLOURS[kind], 2 if kind is not FitKind.QUAD else 1, cv2.LINE_AA)
    legend = [
        LegendEntry("маска страницы v2", COLOUR_MASK),
        LegendEntry("вершины текста surya", COLOUR_TEXT),
        LegendEntry("трапеция v2 (quad)", KIND_COLOURS[FitKind.QUAD]),
        LegendEntry("рамка: полиномы (poly)", KIND_COLOURS[FitKind.POLY]),
        LegendEntry("рамка: сплайны (spline)", KIND_COLOURS[FitKind.SPLINE]),
        LegendEntry("сгиб: самый тёмный шов", COLOUR_FOLD_RAW),
        LegendEntry("сгиб: сглаженный", COLOUR_FOLD),
        LegendEntry("край страницы у корешка по цвету бумаги (сторона poly/spline)", COLOUR_EDGE),
    ]
    return framed(canvas, [f"{case.run.name} {case.key} — рамки страниц по маске v2"], legend)


def compare_thumb(img: np.ndarray, kind: FitKind, row: PageRow, px_per_mm: float) -> np.ndarray:
    """Выпрямленная страница для листа сравнения: высота ``COMPARE_HEIGHT``, горизонтали сетки, шапка с мерами.

    Args:
        img: Выпрямленная страница.
        kind: Вариант рамки.
        row: Меры страницы в этом варианте.
        px_per_mm: Плотность картинки, px/мм.

    Returns:
        Картинка BGR.
    """
    scale = COMPARE_HEIGHT / img.shape[0]
    thumb = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    layer = thumb.copy()
    step = GRID_STEP_MM * px_per_mm * scale
    for y in np.arange(step, thumb.shape[0], step):
        cv2.line(layer, (0, int(y)), (thumb.shape[1], int(y)), COLOUR_GRID, 1, cv2.LINE_AA)
    cv2.addWeighted(layer, GRID_ALPHA, thumb, 1 - GRID_ALPHA, 0, thumb)
    head = header_strip(
        [
            f"{kind.value}: строк {row.lines}, наклон {row.slope_med_deg:+.2f}°, разброс {row.slope_spread_deg:.2f}°",
            f"прогиб p90 {row.sagitta_p90_mm:.2f} мм, непрям. {row.bend_p90_mm:.2f}, от параболы {row.resid_p90_mm:.2f}",
        ],
        thumb.shape[1],
    )
    return np.vstack([head, thumb])


def side_by_side(images: list[np.ndarray], pad: int = 12) -> np.ndarray:
    """Картинки в ряд с белым зазором, добитые белым снизу до общей высоты."""
    h = max(i.shape[0] for i in images)
    cells = [np.pad(i, ((0, h - i.shape[0]), (0, pad), (0, 0)), constant_values=255) for i in images]
    return np.hstack(cells)


def process_frame(run: RunSpec, key: str, kinds: tuple[FitKind, ...], out_dir: Path, save_pages: bool) -> list[PageRow]:
    """Рамки страниц кадра во всех вариантах, выпрямленные страницы, их прямизна и оверлей.

    Args:
        run: Прогон.
        key: Ключ кадра.
        kinds: Варианты рамки.
        out_dir: Корень выхода (``overlays/``, ``pages/<вариант>/``).
        save_pages: Писать ли выпрямленные страницы (JPEG, полный размер).

    Returns:
        Строки мер по страницам и вариантам.
    """
    case = load_case(run, key)
    src = case.output()
    x0, y0 = case.box[0], case.box[1]
    rows, boundaries = [], {}
    for i, page in enumerate(case.pages):
        name = page_name(case, i)
        boundaries[name] = {}
        thumbs = []
        for kind in kinds:
            if kind is FitKind.QUAD or case.fold is None:
                # Трапеция — базовая линия v2 как есть (маска и трапеция v2).
                b = fit_boundary(
                    page.mask, page.quad, page.spine_right, kind, case.seg_px_per_mm, case.exclude, page.text
                )
            else:
                # Кривые — по маске, перерезанной по сгибу; сторона у корешка — сам сгиб.
                b = fit_boundary(
                    page.fold_mask,
                    page.fold_quad,
                    page.spine_right,
                    kind,
                    case.seg_px_per_mm,
                    case.exclude,
                    page.text,
                    fold_points=page.spine_edge,
                )
            boundaries[name][kind] = b
            warp = plan_page_warp(b, case.seg_to_frame, (x0, y0))
            img = warp_page(src, warp)
            if save_pages:
                write_image(out_dir / "pages" / kind.value / f"{run.name}__{key}_{name}.jpg", img, quality=92)
            s = measure_straightness(img, run.px_per_mm)
            c = b.curves
            rows.append(
                PageRow(
                    run=run.name,
                    key=key,
                    page=name,
                    kind=kind.value,
                    resid_top_mm=round(c[Side.TOP].resid_p90_mm, 2),
                    resid_bottom_mm=round(c[Side.BOTTOM].resid_p90_mm, 2),
                    resid_spine_mm=round(c[Side.SPINE].resid_p90_mm, 2),
                    resid_far_mm=round(c[Side.FAR].resid_p90_mm, 2),
                    shift_max_mm=round(max(cv.shift_mm for cv in c.values()), 2),
                    text_outside=round(b.text_outside, 4),
                    width=warp.size[0],
                    height=warp.size[1],
                    **asdict(s),
                )
            )
            thumbs.append(compare_thumb(img, kind, rows[-1], run.px_per_mm))
        sheet = side_by_side(thumbs)
        sheet = np.vstack(
            [
                header_strip(
                    [
                        f"{run.name} {key} {name} — выпрямленная страница по вариантам рамки; сетка {GRID_STEP_MM:.0f} мм"
                    ],
                    sheet.shape[1],
                ),
                sheet,
            ]
        )
        write_image(out_dir / "compare" / f"{run.name}__{key}_{name}.jpg", sheet)
    write_image(out_dir / "overlays" / f"{run.name}__{key}.jpg", draw_overlay(case, boundaries))
    return rows
