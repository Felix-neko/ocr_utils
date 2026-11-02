"""Полнота рамок рисунков: разметка «что ещё относится к рисунку» по компонентам краски вокруг рамки и меры — доля краски рисунка вне рамки, захват букв строк."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout import WORK_DPI
from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.image import PageImage, Variant
from ocr_utils.page_layout.overlay_frame import header_strip
from ocr_utils.units import mm_to_px
from research.line_art_boxes.canvas import load_crop
from research.line_art_boxes.detect import PageArt
from research.line_art_boxes.evaluate import LABELS_VARIANT, BoxKind, Label
from research.line_art_boxes.fix import page_ink

# Кольцо вокруг рамки, в котором компоненты показываются на разметку, мм бумаги.
RING_MM = 12.0
# Пыль мельче этого по большей стороне на разметку не выводится, мм.
SPECK_MM = 0.4
# Какие рамки разметки — объекты для меры полноты (верные рамки рисунков).
OBJECT_KINDS = (BoxKind.DRAWING, BoxKind.LOGO, BoxKind.ORNAMENT)

COLOUR_BOX = (0, 140, 255)  # рамка line art боевого кода
COLOUR_CANDIDATE = (0, 0, 220)  # компонента-кандидат (не текст) с номером
COLOUR_TEXT = (200, 140, 60)  # компонента, признанная буквой строки


@dataclass(frozen=True)
class PageInkWork:
    """Краска страницы в копии ``WORK_DPI`` и множитель перевода рамок вырезки в неё."""

    bitonal: np.ndarray
    scale: float  # px вырезки на px копии


def work_ink(canvas_png: Path) -> PageInkWork:
    """Битональная копия ``WORK_DPI`` вырезки страницы (та же, что видит детектор line art).

    Args:
        canvas_png: PNG вырезки.

    Returns:
        :class:`PageInkWork`.
    """
    image, crop = load_crop(canvas_png)
    page = PageImage.from_array(image, dpi=crop.dpi, variant=Variant.SCAN, cache_name="line_art_boxes")
    return PageInkWork(page.bitonal_at(WORK_DPI), page.scale_to_native(WORK_DPI))


def to_work(box, scale: float) -> Box:
    """Рамка px вырезки → px копии ``WORK_DPI``."""
    return Box(*(int(round(v / scale)) for v in box))


def ring_components(work: PageInkWork, box: Box) -> list[dict]:
    """Компоненты краски в кольце RING_MM вокруг рамки (px копии), не целиком внутри неё и не пыль.

    Args:
        work: Краска страницы.
        box: Рамка, px копии.

    Returns:
        Список ``{"i": номер, "box": [x0, y0, x1, y1], "text": буква строки}`` в порядке слева направо, сверху вниз.
    """
    ink = page_ink(work.bitonal, WORK_DPI, [box])
    ring = mm_to_px(RING_MM, WORK_DPI)
    speck = mm_to_px(SPECK_MM, WORK_DPI)
    s = ink.spots
    near = (s[:, 2] >= box.x0 - ring) & (s[:, 0] <= box.x1 + ring) & (s[:, 3] >= box.y0 - ring)
    near &= s[:, 1] <= box.y1 + ring
    inside = (s[:, 0] >= box.x0) & (s[:, 1] >= box.y0) & (s[:, 2] <= box.x1) & (s[:, 3] <= box.y1)
    big = np.maximum(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) >= speck
    picked = np.flatnonzero(near & ~inside & big & ~ink.edge)
    order = sorted(picked, key=lambda i: (int(s[i, 1]) // 20, int(s[i, 0])))
    return [{"i": n + 1, "box": [int(v) for v in s[i]], "text": bool(ink.text[i])} for n, i in enumerate(order)]


def gt_tile(work: PageInkWork, box: Box, comps: list[dict], title: str, side: int = 640) -> np.ndarray:
    """Миниатюра разметки: рамка, компоненты-кандидаты с номерами, буквы строк — тонко.

    Args:
        work: Краска страницы.
        box: Рамка, px копии.
        comps: Компоненты кольца (:func:`ring_components`).
        title: Подпись.
        side: Большая сторона миниатюры, px.

    Returns:
        Миниатюра BGR с подписью.
    """
    ring = mm_to_px(RING_MM, WORK_DPI)
    height, width = work.bitonal.shape
    x0, y0 = max(0, box.x0 - ring), max(0, box.y0 - ring)
    x1, y1 = min(width, box.x1 + ring), min(height, box.y1 + ring)
    crop = cv2.cvtColor(work.bitonal[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    k = side / max(crop.shape[:2])
    tile = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)

    def p(x, y):
        return int((x - x0) * k), int((y - y0) * k)

    cv2.rectangle(tile, p(box.x0, box.y0), p(box.x1, box.y1), COLOUR_BOX, 2)
    for c in comps:
        bx = c["box"]
        colour = COLOUR_TEXT if c["text"] else COLOUR_CANDIDATE
        cv2.rectangle(tile, p(bx[0], bx[1]), p(bx[2], bx[3]), colour, 1)
        # Номера у всех: буква строки может оказаться частью рисунка (штрих у края, подпись).
        cv2.putText(tile, str(c["i"]), p(bx[2] + 1, bx[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.36, colour, 1)
    return np.vstack([header_strip([title], tile.shape[1]), tile])


def objects(labels: list[Label]) -> list[Label]:
    """Рамки разметки, по которым мерится полнота (рисунки, логотипы, орнаменты)."""
    return [label for label in labels if label.kind in OBJECT_KINDS]


@dataclass(frozen=True)
class GtSpec:
    """Разметка полноты одного объекта: что из кольца относится к рисунку.

    Поля:
        add: Номера не-текстовых компонент кольца.
        text_add: Номера компонент, признанных буквами, но относящихся к рисунку (волосы, линии движения у края).
        zone: Поля к рамке ``(верх, низ, лево, право)``, мм: компоненты кольца с центром в рамке, расширенной на них,
            относятся к рисунку (штрихи тени, трава, лучи — когда номеров слишком много).
    """

    add: tuple[int, ...] = ()
    text_add: tuple[int, ...] = ()
    zone: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def read_gt(path: Path) -> dict[tuple[str, int], GtSpec]:
    """Разметка полноты ``page,box,add,text_add,zone,note`` (номера и поля — через «;»).

    Args:
        path: CSV.

    Returns:
        ``(страница, рамка) → GtSpec``.
    """

    def ids(text: str) -> tuple[int, ...]:
        return tuple(int(v) for v in text.split(";") if v.strip())

    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            zone = tuple(float(v) for v in r["zone"].split(";")) if r["zone"].strip() else (0.0, 0.0, 0.0, 0.0)
            out[(r["page"], int(r["box"]))] = GtSpec(ids(r["add"]), ids(r["text_add"]), zone)
    return out


def gt_box(base: Box, comps: list[dict], spec: GtSpec) -> Box:
    """Рамка рисунка по разметке: рамка боевого кода плюс выбранные компоненты кольца и компоненты в полях (px копии).

    Args:
        base: Рамка боевого кода, px копии.
        comps: Компоненты кольца (:func:`ring_components`).
        spec: Разметка.

    Returns:
        Рамка рисунка, px копии.
    """
    top, bottom, left, right = (mm_to_px(v, WORK_DPI, minimum=0) for v in spec.zone)
    zone = Box(base.x0 - left, base.y0 - top, base.x1 + right, base.y1 + bottom)
    chosen = set(spec.add) | set(spec.text_add)
    boxes = [base]
    for c in comps:
        bx = Box(*c["box"])
        centre_x, centre_y = (bx.x0 + bx.x1) / 2, (bx.y0 + bx.y1) / 2
        in_zone = (
            spec.zone != (0.0, 0.0, 0.0, 0.0) and zone.x0 <= centre_x <= zone.x1 and zone.y0 <= centre_y <= zone.y1
        )
        if c["i"] in chosen or in_zone:
            boxes.append(bx)
    return Box(min(b.x0 for b in boxes), min(b.y0 for b in boxes), max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def outside_share(bitonal: np.ndarray, truth: Box, found: list[Box]) -> float:
    """Доля краски рисунка (в рамке разметки) вне рамок варианта.

    Args:
        bitonal: Краска копии (0 — краска).
        truth: Рамка рисунка по разметке.
        found: Рамки варианта, px копии.

    Returns:
        Доля 0…1.
    """
    ink = bitonal[truth.slice] == 0
    total = int(ink.sum())
    if total == 0:
        return 0.0
    covered = np.zeros_like(ink)
    for b in found:
        x0, y0 = max(b.x0, truth.x0) - truth.x0, max(b.y0, truth.y0) - truth.y0
        x1, y1 = min(b.x1, truth.x1) - truth.x0, min(b.y1, truth.y1) - truth.y0
        if x1 > x0 and y1 > y0:
            covered[y0:y1, x0:x1] = True
    return float((ink & ~covered).sum()) / total


def captured_letters(work: PageInkWork, before: list[Box], after: list[Box]) -> int:
    """Сколько букв строк (компонент, признанных текстом) вошло в рамки варианта сверх рамок ``before``.

    Args:
        work: Краска страницы.
        before: Рамки боевого кода, px копии.
        after: Рамки варианта, px копии.

    Returns:
        Число букв, центр которых внутри рамок ``after`` и вне рамок ``before``.
    """
    if not after:
        return 0
    ink = page_ink(work.bitonal, WORK_DPI, after)
    s = ink.spots
    cx, cy = (s[:, 0] + s[:, 2]) / 2, (s[:, 1] + s[:, 3]) / 2

    def within(boxes):
        hit = np.zeros(len(s), bool)
        for b in boxes:
            hit |= (cx >= b.x0) & (cx <= b.x1) & (cy >= b.y0) & (cy <= b.y1)
        return hit

    return int(np.count_nonzero(ink.text & within(after) & ~within(before)))


def save_objects(path: Path, rows: list[dict]) -> None:
    """Записать компоненты колец объектов разметки (JSON) — по ним номера из CSV переводятся в рамки."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False))


def art_boxes(arts: PageArt, scale: float) -> list[Box]:
    """Рамки line art разбора в px копии ``WORK_DPI``."""
    return [to_work(b.box, scale) for b in arts.boxes]


__all__ = [
    "LABELS_VARIANT",
    "art_boxes",
    "captured_letters",
    "gt_box",
    "gt_tile",
    "objects",
    "outside_share",
    "read_gt",
    "ring_components",
    "save_objects",
    "to_work",
    "work_ink",
]
