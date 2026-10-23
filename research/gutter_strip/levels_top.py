"""Что добавил бы у верхних (нижних) краёв страниц SAM 3 по кадру с растянутой светлотой (``sam3.npz`` ``raw_lv``, опция v2 ``--dual-levels``) к основному SAM 3 (``raw`` ∪ ``raw_noink``): площадь в книге и вне её по кадру и оверлей полосы верхнего края с вырезкой у корешка."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from research.gutter_strip.band import load_strip_frame

# Зона верхних (нижних) краёв страниц: по столбцу от верхней (нижней) строки страниц v2 — столько наружу и внутрь, мм.
ZONE_OUT_MM = 10.0
ZONE_IN_MM = 15.0
# Вырезка у корешка: полуширина, мм; увеличение.
SPINE_HALF_MM = 30.0
ZOOM = 2.5
# Ширина полосы верхнего края на картинке, px.
VIEW_WIDTH = 1800
COLOUR_BASE = (220, 90, 20)  # контур основного SAM 3
COLOUR_LEVELS = (0, 150, 0)  # контур SAM 3 по растянутому кадру
COLOUR_ADD_BOOK = (200, 60, 200)  # добавка в книге
COLOUR_ADD_OUT = (0, 0, 220)  # добавка вне книги
COLOUR_ZONE = (170, 170, 170)
COLOUR_SPINE = (0, 165, 255)
ALPHA = 0.55


@dataclass(frozen=True)
class TopAdd:
    """Добавка растянутого SAM 3 у верхних краёв страниц кадра.

    Поля:
        added_book_mm2: Добавка в пределах грубой книги, мм².
        added_out_mm2: Добавка вне книги, мм².
        removed_mm2: Что основной SAM 3 видит, а растянутый — нет (для сведения), мм².
        near_spine_mm2: Добавка в книге в ``SPINE_HALF_MM`` от разделителя, мм².
    """

    added_book_mm2: float
    added_out_mm2: float
    removed_mm2: float
    near_spine_mm2: float


def edge_zone(pages: list[np.ndarray], shape: tuple[int, int], top: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Зона верхних (``top``) или нижних краёв страниц и крайняя строка страниц по столбцам (NaN — страниц нет).

    Зона по столбцу — от крайней строки страниц на ``ZONE_OUT_MM`` наружу (у верха — вверх, у низа — вниз) и на
    ``ZONE_IN_MM`` внутрь.
    """
    h, w = shape
    either = np.zeros(shape, bool)
    for p in pages:
        either |= p
    has = either.any(axis=0)
    first = np.argmax(either, axis=0) if top else h - 1 - np.argmax(either[::-1], axis=0)
    edge = np.where(has, first, np.nan).astype(np.float64)
    yy = np.arange(h)[:, None]
    out, inside = ZONE_OUT_MM * SEG_PX_PER_MM, ZONE_IN_MM * SEG_PX_PER_MM
    lo, hi = (edge - out, edge + inside) if top else (edge - inside, edge + out)
    zone = (yy >= lo[None, :]) & (yy <= hi[None, :]) & has[None, :]
    return zone, edge


def levels_add(work: Path, pages: list[np.ndarray], book: np.ndarray, top: bool = True) -> np.ndarray | None:
    """Добавка SAM 3 по растянутой светлоте к основному в зоне краёв страниц (``edge_zone``) в пределах книги.

    Args:
        work: Папка кадра в кэше v2 (``sam3.npz`` с ``raw_lv``).
        pages: Маски страниц v2.
        book: Грубая маска книги.
        top: Верхние края (иначе нижние).

    Returns:
        Маска добавки или None (второго прохода в кэше нет).
    """
    with np.load(work / "sam3.npz") as d:
        if "raw_lv" not in d.files:
            return None
        base = d["raw"] | (d["raw_noink"] if "raw_noink" in d.files else False)
        levels = d["raw_lv"]
    zone, _ = edge_zone(pages, base.shape, top)
    return levels & ~base & zone & book


def frame_top_add(
    cache: Path, key: str, out: Path, title: str, name: str, top: bool = True
) -> tuple[TopAdd, str] | None:
    """Добавка у верхних (``top``) или нижних краёв кадра и оверлей.

    Args:
        cache: Кэш v2 (папка с кадрами).
        key: Ключ кадра.
        out: Папка оверлеев (подпапки по поясам добавки).
        title: Первая строка шапки.
        name: Имя файла оверлея без расширения.
        top: Верхние края (иначе нижние).

    Returns:
        ``(добавка, пояс)`` или None (не разворот из двух страниц).
    """
    sf = load_strip_frame(cache, key)
    if sf is None:
        return None
    work = cache / key
    with np.load(work / "sam3.npz") as d:
        if "raw_lv" not in d.files:
            return None
        base = d["raw"] | (d["raw_noink"] if "raw_noink" in d.files else False)
        levels = d["raw_lv"]
    book = sf.frame.book
    img = sf.frame.img
    zone, edge = edge_zone([p.mask for p in sf.frame.pages], base.shape, top)
    add = levels & ~base & zone
    add_book, add_out = add & book, add & ~book
    removed = base & ~levels & zone
    mm2 = SEG_PX_PER_MM**2
    h, w = base.shape
    cols = np.arange(w)[None, :]
    spine = sf.spine0
    near = np.abs(cols - spine.x_at(np.arange(h, dtype=np.float64))[:, None]) <= SPINE_HALF_MM * SEG_PX_PER_MM
    res = TopAdd(add_book.sum() / mm2, add_out.sum() / mm2, removed.sum() / mm2, (add_book & near).sum() / mm2)
    # Картинка: кадр, добавка заливкой, контуры двух проходов, границы зоны, разделитель.
    canvas = img.copy()
    layer = canvas.copy()
    layer[add_book] = COLOUR_ADD_BOOK
    layer[add_out] = COLOUR_ADD_OUT
    cv2.addWeighted(layer, ALPHA, canvas, 1 - ALPHA, 0, canvas)
    for mask, colour in ((base, COLOUR_BASE), (levels, COLOUR_LEVELS)):
        cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, cnt, -1, colour, 1)
    zcnt, _ = cv2.findContours(zone.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, zcnt, -1, COLOUR_ZONE, 1)
    ys = np.arange(h)
    pts = np.stack([spine.x_at(ys), ys], 1).round().astype(np.int32)[:, None]
    cv2.polylines(canvas, [pts], False, COLOUR_SPINE, 1)
    ok = np.isfinite(edge)
    # Пролёт полосы — по 5–95 % края страниц: в крайних столбцах «верх» страницы бывает у самого низа.
    lo_e, hi_e = np.nanpercentile(edge, [5, 95]) if ok.any() else ((0.0, h / 4) if top else (3 * h / 4, h))
    above, below = (ZONE_OUT_MM, ZONE_IN_MM) if top else (ZONE_IN_MM, ZONE_OUT_MM)
    y0 = int(max(lo_e - (above + 5) * SEG_PX_PER_MM, 0))
    y1 = int(min(hi_e + (below + 5) * SEG_PX_PER_MM, h))
    band = canvas[y0:y1]
    s = VIEW_WIDTH / band.shape[1]
    band = cv2.resize(band, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cx = int(spine.x_at((y0 + y1) / 2))
    half = int(SPINE_HALF_MM * SEG_PX_PER_MM)
    crop = canvas[y0:y1, max(cx - half, 0) : min(cx + half, w)]
    crop = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    crop = np.pad(crop, ((8, 0), (0, max(0, VIEW_WIDTH - crop.shape[1])), (0, 0)), constant_values=255)[:, :VIEW_WIDTH]
    picture = framed(
        np.vstack([band, crop]),
        [title,
         f"добавка растянутого SAM 3 у {'верхних' if top else 'нижних'} краёв: в книге {res.added_book_mm2:.0f} мм² (у корешка ±{SPINE_HALF_MM:.0f} мм —"
         f" {res.near_spine_mm2:.0f}), вне книги {res.added_out_mm2:.0f} мм²; основной видит, растянутый нет — "
         f"{res.removed_mm2:.0f} мм²",
         f"зона: от {'верха' if top else 'низа'} страниц v2 на {ZONE_OUT_MM:.0f} мм наружу и {ZONE_IN_MM:.0f} мм внутрь;"
         f" ниже полосы — у корешка ×{ZOOM}"],
        [LegendEntry("добавка в книге", COLOUR_ADD_BOOK, alpha=ALPHA, style=SampleStyle.BOX),
         LegendEntry("добавка вне книги (на фон)", COLOUR_ADD_OUT, alpha=ALPHA, style=SampleStyle.BOX),
         LegendEntry("основной SAM 3 (raw ∪ raw_noink), контур", COLOUR_BASE),
         LegendEntry("SAM 3 по растянутой светлоте (raw_lv), контур", COLOUR_LEVELS),
         LegendEntry("зона краёв страниц", COLOUR_ZONE),
         LegendEntry("разделитель v2", COLOUR_SPINE)],
    )  # fmt: skip
    belt = "a_50+" if res.added_book_mm2 >= 50 else ("b_10-50" if res.added_book_mm2 >= 10 else "c_0-10")
    if res.added_out_mm2 >= 10:
        belt = "d_вне_книги_10+"
    (out / belt).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / belt / f"{name}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return res, belt
