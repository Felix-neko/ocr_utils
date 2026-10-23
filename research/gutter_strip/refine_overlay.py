"""Оверлей уточнения ленты у корешка: лента целиком с краями всех способов (принятые — сплошные, отвергнутые — пунктир, выбранные — толсто, полоса — заливкой) и крупно верх и низ у концов корешка; шапка с мерами способов и легенда — в полях."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from research.gutter_strip.refine import SpineRefine

# Масштаб ленты целиком и вырезок у концов, px/мм.
VIEW_PPM = 4.0
ZOOM_PPM = 8.0
# Вырезка у конца: столько мм от края пролёта страниц внутрь (наружу — вся добавка ленты ``END_PAD_MM``).
ZOOM_IN_MM = 40.0
# Цвета способов (BGR).
METHOD_COLOURS = {
    "sam2_sq4": (0, 150, 0),
    "sam2_sq4_lvL": (0, 165, 255),
    "sam3_sq4": (220, 90, 20),
    "edges": (200, 60, 200),
    "sam2strip_sq4": (140, 140, 0),
    "sam2_sq1": (0, 90, 0),
    "sam3_sq1": (140, 50, 10),
    "sam2_sq4_clahe": (60, 190, 190),
    "sam3_sq4_lvL": (255, 150, 80),
    "sam3strip_sq4": (120, 80, 230),
    "creases": (150, 150, 150),
    "solid": (0, 110, 200),
}
COLOUR_CHOSEN = (0, 0, 220)
COLOUR_COARSE = (150, 170, 120)
COLOUR_STRIP = (0, 165, 255)
STRIP_ALPHA = 0.35
DASH_PX = 6
# Наименьшая ширина картинки, px: шапка с мерами способов не обрезается.
MIN_WIDTH_PX = 1500


def polyline(edge: np.ndarray, scale: float, row0: int = 0) -> list[np.ndarray]:
    """Ломаные края (столбцы по строкам ленты) в масштабе ``scale`` с вырезкой от строки ``row0``; NaN рвут линию."""
    rows = np.arange(len(edge))
    ok = np.isfinite(edge)
    parts, start = [], None
    for i in range(len(edge) + 1):
        if i < len(edge) and ok[i]:
            start = i if start is None else start
            continue
        if start is not None and i - start >= 2:
            seg = np.stack([edge[start:i] * scale, (rows[start:i] - row0) * scale], axis=1)
            parts.append(np.round(seg).astype(np.int32)[:, None, :])
        start = None
    return parts


def draw_edge(canvas: np.ndarray, edge: np.ndarray, scale: float, colour, thick: int, dashed: bool, row0: int = 0) -> None:  # fmt: skip
    """Край на холсте: сплошной или пунктиром (через ``DASH_PX`` строк холста)."""
    for part in polyline(edge, scale, row0):
        if not dashed:
            cv2.polylines(canvas, [part], False, colour, thick, cv2.LINE_AA)
            continue
        for k in range(0, len(part) - 1, 2 * DASH_PX):
            cv2.polylines(canvas, [part[k : k + DASH_PX + 1]], False, colour, thick, cv2.LINE_AA)


def band_view(res: SpineRefine, bgr: np.ndarray, coarse: tuple[np.ndarray, np.ndarray], scale: float,rows: slice) -> np.ndarray:  # fmt: skip
    """Вырезка ленты строк ``rows`` в масштабе ``scale``: грубые страницы контуром, края способов, выбранные, полоса."""
    band = res.band
    row0 = rows.start or 0
    piece = bgr[rows]
    canvas = cv2.resize(piece, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    for mask in coarse:
        m = cv2.resize(mask[rows].astype(np.uint8), canvas.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, cnt, -1, COLOUR_COARSE, 1)
    if res.strip and res.left is not None:
        layer = canvas.copy()
        lo, hi = res.left[rows], res.right[rows]
        poly = np.concatenate([np.stack([lo, np.arange(len(lo))], 1), np.stack([hi, np.arange(len(hi))], 1)[::-1]])
        cv2.fillPoly(layer, [np.round(poly * scale).astype(np.int32)], COLOUR_STRIP)
        cv2.addWeighted(layer, STRIP_ALPHA, canvas, 1 - STRIP_ALPHA, 0, canvas)
    for name, r in res.results.items():
        colour = METHOD_COLOURS.get(name, (90, 90, 90))
        for edge, m in ((r.left, r.left_m), (r.right, r.right_m)):
            draw_edge(canvas, edge[rows], scale, colour, 1, dashed=not m.accepted)
    if res.left is not None:
        for edge in (res.left, res.right):
            draw_edge(canvas, edge[rows], scale, COLOUR_CHOSEN, 2, dashed=False)
    # Разделитель v2 — середина ленты.
    x = int(round(band.half * scale))
    cv2.line(canvas, (x, 0), (x, canvas.shape[0] - 1), (200, 200, 200), 1)
    return canvas


def method_lines(res: SpineRefine) -> list[str]:
    """Строки шапки по способам: принят ли край, рваность, прямизна, положение (мм от разделителя)."""
    out = []
    for name, r in res.results.items():
        cells = []
        for tag, m in (("Л", r.left_m), ("П", r.right_m)):
            mark = "✓" if m.accepted else "✗"
            cells.append(
                f"{tag}{mark} {m.median_mm:+.1f} рв {m.rough_mm:.2f} ст {m.step_mm:.1f} пр {m.straight_mm:.2f}"
                f" на {m.inlier_share:.0%} оп {m.support_l:.0f} кр {m.ink_out_mm2:.0f}"
            )
        order = "" if r.order_ok else " ПОРЯДОК"
        out.append(f"{name:15s} {'  |  '.join(cells)}  промахов {r.missed_chunks}{order}  {r.seconds:.1f} с")
    return out


def refine_overlay(res: SpineRefine, bgr: np.ndarray, coarse: tuple[np.ndarray, np.ndarray], title: str) -> np.ndarray:
    """Оверлей уточнения кадра: лента целиком слева, верх и низ крупно справа, шапка и легенда в полях.

    Args:
        res: Итог уточнения.
        bgr: Лента кадра полного разрешения.
        coarse: Грубые страницы по ленте ``(левая, правая)``.
        title: Первая строка шапки (кадр, метка).

    Returns:
        Картинка BGR.
    """
    band = res.band
    whole = band_view(res, bgr, coarse, VIEW_PPM / band.ppm, slice(0, band.length))
    lo, hi = band.page_rows
    zoom_rows = int(ZOOM_IN_MM * band.ppm)
    z = ZOOM_PPM / band.ppm
    top = band_view(res, bgr, coarse, z, slice(0, min(lo + zoom_rows, band.length)))
    bottom = band_view(res, bgr, coarse, z, slice(max(hi - zoom_rows, 0), band.length))
    gap = np.full((8, top.shape[1], 3), 255, np.uint8)
    right = np.vstack([top, gap, bottom])
    h = max(whole.shape[0], right.shape[0])
    whole, right = (np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)), constant_values=255) for im in (whole, right))
    canvas = np.hstack([whole, np.full((h, 12, 3), 255, np.uint8), right])
    # Шапка с мерами способов длинная: холст не уже MIN_WIDTH_PX (белое поле справа).
    canvas = np.pad(canvas, ((0, 0), (0, max(0, MIN_WIDTH_PX - canvas.shape[1])), (0, 0)), constant_values=255)
    decision = (f"полоса {res.width_mm:.1f} мм" if res.strip else
                (f"полосы нет ({res.width_mm:.1f} мм)" if np.isfinite(res.width_mm) else res.status))  # fmt: skip
    gap_mm = res.coarse.extra.get("gap_mm", float("nan"))
    header = [
        title,
        f"уточнение: {decision}; левый край — {res.left_by or '—'}, правый — {res.right_by or '—'}"
        + ("; полный набор" if res.full_set else ""),
        f"грубый SAM 3: щель {gap_mm:.1f} мм, языки Л {res.coarse.extra.get('tongue_left_share', 0):.2f}"
        f" / П {res.coarse.extra.get('tongue_right_share', 0):.2f}",
        "способ  Л/П: ✓ принят, мм от разделителя, рв — рваность p90, ст — ступень, пр — остаток к прямой p90 мм,"
        " на — доля строк на прямой, оп — опора на снимке (уровни L), кр — краска своей страницы за краем мм²",
        *method_lines(res),
    ]
    legend = [LegendEntry(name, colour) for name, colour in METHOD_COLOURS.items() if name in res.results]
    legend += [
        LegendEntry("отвергнутый край способа", (90, 90, 90), style=SampleStyle.DASHED),
        LegendEntry("выбранные края страниц", COLOUR_CHOSEN),
        LegendEntry("грубые страницы v2 (контур)", COLOUR_COARSE),
        LegendEntry("полоса-вклейка", COLOUR_STRIP, alpha=STRIP_ALPHA, style=SampleStyle.BOX),
        LegendEntry("разделитель v2", (200, 200, 200)),
    ]
    return framed(canvas, header, legend)
