"""Полоса из уточнённых краёв ленты у корешка в подгоночных областях страниц: ``Strip`` по прямым Хубера через выбранные края (``refine.refine_spine``), подгоночные области v2 с ней и с прежней полосой, углы страниц на линии корешка и их расхождение."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ocr_utils.scan_cropping.v2.gutter_strip import Strip, spine_line
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.spread_lines import Line, SpineMethod, fit_line
from research.gutter_strip.band import StripFrame
from research.gutter_strip.fit import PageFit, fit_pages


@dataclass(frozen=True)
class SavedEdges:
    """Выбранные края уточнения кадра из ``npz/<пункт>.npz`` команды ``refine``.

    Поля:
        left: Правый край левой страницы по строкам ленты (столбцы) или None (края не выбраны).
        right: Левый край правой страницы.
        ys_seg: Строки ленты в кадре сегментации.
        half: Столбец разделителя в ленте.
        ppm: px/мм ленты.
    """

    left: np.ndarray | None
    right: np.ndarray | None
    ys_seg: np.ndarray
    half: int
    ppm: float


def load_edges(path: Path) -> SavedEdges:
    """Края уточнения из ``npz`` команды ``refine``."""
    with np.load(path) as d:
        has = "chosen_left" in d.files
        return SavedEdges(
            d["chosen_left"] if has else None,
            d["chosen_right"] if has else None,
            d["ys_seg"],
            int(d["half"]),
            float(d["ppm"]),
        )


def edge_points(edges: SavedEdges, edge: np.ndarray, spine: Line, valid: np.ndarray) -> np.ndarray:
    """Точки края (столбцы ленты по строкам) в кадре сегментации ``(N, 2)`` ``x, y`` по годным строкам."""
    ratio = edges.ppm / SEG_PX_PER_MM
    rows = np.flatnonzero(valid & np.isfinite(edge))
    ys = edges.ys_seg[rows]
    xs = spine.x_at(ys) + (edge[rows] - edges.half) / ratio
    return np.stack([xs, ys], axis=1)


def strip_from_edges(sf: StripFrame, edges: SavedEdges) -> Strip | None:
    """Полоса по выбранным краям уточнения: прямые Хубера через края в средней части пролёта страниц, маска — между
    ними в пределах грубой книги (как ``gutter_strip.segment_strip``).

    Args:
        sf: Кадр стенда.
        edges: Края уточнения.

    Returns:
        Полоса или None (края не выбраны или прямая не построилась).
    """
    if edges.left is None:
        return None
    spine = sf.spine0
    pages = sf.frame.pages
    y_top = float(min(p.quad[:, 1].min() for p in pages))
    y_bot = float(max(p.quad[:, 1].max() for p in pages))
    # Средняя часть пролёта (без 15 мм у концов): у концов полоса задирается и отходит от страниц.
    cut = 15.0 * SEG_PX_PER_MM
    valid = (edges.ys_seg >= y_top + cut) & (edges.ys_seg <= y_bot - cut)
    lines, points = [], []
    for edge in (edges.left, edges.right):
        pts = edge_points(edges, edge, spine, valid)
        line = fit_line(pts[:, 0], pts[:, 1], spine.y_top, spine.y_bot) if len(pts) else None
        if line is None:
            return None
        lines.append(line)
        points.append(pts)
    left, right = lines
    book = sf.frame.book
    h, w = book.shape
    yy = np.arange(h, dtype=np.float64)
    cols = np.arange(w)[None, :]
    mask = (cols >= left.x_at(yy)[:, None]) & (cols <= right.x_at(yy)[:, None]) & book
    mid = (spine.y_top + spine.y_bot) / 2
    width = float((right.x_at(mid) - left.x_at(mid)) / SEG_PX_PER_MM)
    return Strip(left, right, mask, width, points[0], points[1])


@dataclass(frozen=True)
class VariantFit:
    """Подгоночные области кадра одним вариантом.

    Поля:
        name: Имя варианта.
        strip: Полоса варианта или None.
        spine: Линия корешка.
        spine_source: Откуда она (подпись).
        fits: Рамки страниц.
        corners: Углы на линии корешка ``{"Lt", "Lb", "Rt", "Rb"}`` → ``y`` px сегментации.
    """

    name: str
    strip: Strip | None
    spine: Line
    spine_source: str
    fits: list[PageFit]
    corners: dict[str, float]


def variant(sf: StripFrame, name: str, strip: Strip | None) -> VariantFit:
    """Подгоночные области кадра с полосой ``strip`` (или без неё): линия корешка, ``spread_frames``, углы у корешка."""
    lines = sf.frame.lines[SpineMethod.AUTO]
    spine, source = spine_line(strip, sf.spine0, lines.method.value)
    fits = fit_pages(sf, strip, spine)
    corners = {}
    for tag, fit in zip("LR", fits):
        top, bottom = fit.after.top, fit.after.bottom
        corners[f"{tag}t"] = float(top[-1, 1])
        corners[f"{tag}b"] = float(bottom[-1, 1])
    return VariantFit(name, strip, spine, source, fits, corners)


def corner_steps_mm(v: VariantFit) -> tuple[float, float]:
    """Ступенька углов двух страниц на линии корешка по высоте, мм: ``(верх, низ)``."""
    return (abs(v.corners["Lt"] - v.corners["Rt"]) / SEG_PX_PER_MM,
            abs(v.corners["Lb"] - v.corners["Rb"]) / SEG_PX_PER_MM)  # fmt: skip


# Кромка полосы у концов: столбцы полосы без ``END_EDGE_GUARD_MM`` у её краёв делятся на три части (у левого края,
# середина, у правого); в каждой — медиана строки первого «бумажного» пикселя от фона за верхом (низом) листа.
END_EDGE_GUARD_MM = 1.0
# Поиск кромки — в ``END_SEARCH_MM`` от угла страниц варианта по высоте.
END_SEARCH_MM = 15.0


def paper_threshold(gray: np.ndarray, cols: np.ndarray, y0: int, y1: int) -> float:
    """Порог «бумага / фон» в столбцах ``cols`` строк ``[y0, y1)`` — Оцу по серому (фон коврика и обложки темнее)."""
    import cv2

    patch = gray[max(y0, 0) : max(y1, 1)][:, cols]
    thr, _ = cv2.threshold(patch.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr)


def strip_ends(sf: StripFrame, strip: Strip, v: VariantFit) -> dict[str, float]:
    """Выступ кромки полосы за углы страниц у верха и низа и её задир у краёв полосы, мм.

    Выступ — насколько кромка полосы в середине её ширины выше (у верха) или ниже (у низа) средней высоты углов двух
    страниц на линии корешка (плюс — полоса торчит за лист). Задир — насколько кромка у краёв полосы выступает дальше
    середины (плюс — края полосы торчат: V-образный разрыв или неплотная приклейка).

    Args:
        sf: Кадр стенда.
        strip: Полоса.
        v: Вариант с подгоночными областями (углы на корешке).

    Returns:
        ``{"top_out_mm", "bottom_out_mm", "top_flare_mm", "bottom_flare_mm"}`` (NaN — кромка не найдена).
    """
    import cv2

    gray = cv2.cvtColor(sf.frame.img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    reach = int(END_SEARCH_MM * SEG_PX_PER_MM)
    guard = END_EDGE_GUARD_MM * SEG_PX_PER_MM
    out = {}
    for end, corner_y in (
        ("top", (v.corners["Lt"] + v.corners["Rt"]) / 2),
        ("bottom", (v.corners["Lb"] + v.corners["Rb"]) / 2),
    ):
        y0, y1 = int(corner_y) - reach, int(corner_y) + reach
        xl, xr = strip.left.x_at(corner_y) + guard, strip.right.x_at(corner_y) - guard
        cols = np.arange(int(np.ceil(xl)), int(np.floor(xr)) + 1)
        cols = cols[(cols >= 0) & (cols < w)]
        if len(cols) < 6 or y0 < 0 or y1 > h:
            out[f"{end}_out_mm"] = out[f"{end}_flare_mm"] = float("nan")
            continue
        thr = paper_threshold(gray, cols, y0, y1)
        patch = gray[y0:y1][:, cols] > thr
        # Первая «бумажная» строка от фона: у верха — сверху вниз, у низа — снизу вверх.
        if end == "bottom":
            patch = patch[::-1]
        first = np.where(patch.any(axis=0), np.argmax(patch, axis=0), np.nan).astype(np.float64)
        edge_y = (y0 + first) if end == "top" else (y1 - 1 - first)
        thirds = np.array_split(edge_y, 3)
        sides = np.nanmedian(np.r_[thirds[0], thirds[2]])
        mid = np.nanmedian(thirds[1])
        sign = 1.0 if end == "top" else -1.0
        out[f"{end}_out_mm"] = float(sign * (corner_y - mid) / SEG_PX_PER_MM)
        out[f"{end}_flare_mm"] = float(sign * (mid - sides) / SEG_PX_PER_MM)
    return out
