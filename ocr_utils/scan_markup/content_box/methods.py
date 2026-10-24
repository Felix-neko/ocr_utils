"""Способы зоны контента (зоны обрезки) выпрямленной страницы: A — фиксированные отступы, B — охват контента, C — краска без сора у краёв, D — отступы с защитой контента; меры потери контента и мусора внутри рамки."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from ocr_utils.units import mm_to_px, mm_to_px_float, px_to_mm
from ocr_utils.scan_markup.content_box.content import EDGE_BAND_MM, Box, Item, PageContent, Side


class Method(StrEnum):
    """Способ зоны контента."""

    FIXED = "A_fixed"  # фиксированные отступы от краёв страницы
    CONTENT = "B_content"  # охват контента (строки, блоки, области) без сора + запас
    INK = "C_ink"  # охват краски без касающейся края и мелкого сора у краёв + запас
    HYBRID = "D_hybrid"  # отступы A, раздвинутые до контента B, если он заходит за них


@dataclass(frozen=True)
class TrimParams:
    """Числа способов, мм бумаги.

    Поля:
        top_mm, bottom_mm, outer_mm, cut_mm: Отступы A от верха, низа, внешнего края и края у линии разреза.
        margin_mm: Запас вокруг контента (B, C, D).
        stray_gap_mm: Короткая строка дальше этого от тела страницы по горизонтали — сор (метка у корешка, «49»).
        speck_mm2: Компонента краски у края меньше этой площади и не рядом с контентом — сор (C).
        near_mm: «Рядом с контентом» — ближе этого к рамке элемента тела (C).
    """

    top_mm: float = 4.0
    bottom_mm: float = 4.0
    outer_mm: float = 3.0
    cut_mm: float = 5.0
    margin_mm: float = 2.0
    stray_gap_mm: float = 2.0
    speck_mm2: float = 6.0
    near_mm: float = 3.0


def fixed_box(content: PageContent, params: TrimParams) -> Box:
    """A: страница минус отступы по сторонам (у разреза — ``cut_mm``, у внешнего края — ``outer_mm``).

    Args:
        content: Контент страницы (размер, dpi, сторона разреза).
        params: Числа способов.

    Returns:
        Рамка.
    """
    left_mm = params.cut_mm if content.cut_side is Side.LEFT else params.outer_mm
    right_mm = params.cut_mm if content.cut_side is Side.RIGHT else params.outer_mm
    dpi = content.dpi
    return (
        mm_to_px(left_mm, dpi, minimum=0),
        mm_to_px(params.top_mm, dpi, minimum=0),
        content.width - mm_to_px(right_mm, dpi, minimum=0),
        content.height - mm_to_px(params.bottom_mm, dpi, minimum=0),
    )


def strays(content: PageContent, params: TrimParams) -> list[Item]:
    """Сор среди строк: короткие строки, лежащие по горизонтали вне тела страницы дальше ``stray_gap_mm``.

    Тело — длинные строки, блоки и области. Номер страницы в колонтитуле и под колонкой лежит внутри горизонтального
    охвата тела и сором не считается; повёрнутая цифра во внутреннем поле у корешка (ПХ 1974/01 IMG_0078_R, «49») — вне.

    Args:
        content: Контент страницы.
        params: Числа способов.

    Returns:
        Короткие строки-сор.
    """
    body = content.body
    if not body:
        return []
    bx0 = min(i.box[0] for i in body)
    bx1 = max(i.box[2] for i in body)
    gap = mm_to_px_float(params.stray_gap_mm, content.dpi)
    return [i for i in content.lines if i.short and (i.box[2] < bx0 - gap or i.box[0] > bx1 + gap)]


def core_items(content: PageContent, params: TrimParams) -> list[Item]:
    """Контент без сора (:func:`strays`)."""
    bad = set(map(id, strays(content, params)))
    return [i for i in content.items if id(i) not in bad]


def union_box(boxes: list[Box]) -> Box | None:
    """Общий охват рамок или None."""
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def grow(box: Box, px: float, content: PageContent) -> Box:
    """Рамка, расширенная на ``px`` с каждой стороны и обрезанная по странице."""
    x0, y0, x1, y1 = box
    p = int(round(px))
    return (max(0, x0 - p), max(0, y0 - p), min(content.width, x1 + p), min(content.height, y1 + p))


def content_box(content: PageContent, params: TrimParams) -> Box:
    """B: охват контента без сора плюс запас; без контента — рамка A."""
    core = union_box([i.box for i in core_items(content, params)])
    if core is None:
        return fixed_box(content, params)
    return grow(core, mm_to_px_float(params.margin_mm, content.dpi), content)


def ink_box(content: PageContent, params: TrimParams) -> Box:
    """C: охват компонент краски без касающихся края страницы (тень, кромка стопки) и без мелких пятен у краёв вдали от
    контента, плюс запас.

    Args:
        content: Контент страницы (маска краски рабочей копии).
        params: Числа способов.

    Returns:
        Рамка px страницы.
    """
    ink = content.ink.astype(np.uint8)
    s = content.ink_scale
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    h, w = ink.shape
    band = mm_to_px_float(EDGE_BAND_MM, content.dpi) / s
    near = mm_to_px_float(params.near_mm, content.dpi) / s
    speck = params.speck_mm2 * (content.dpi / 25.4 / s) ** 2
    core = [tuple(v / s for v in i.box) for i in core_items(content, params)]
    kept = []
    for k in range(1, n):
        x, y, cw, ch, area = stats[k]
        x1, y1 = x + cw, y + ch
        if x <= 0 or y <= 0 or x1 >= w or y1 >= h:
            continue  # касается края страницы: тень, кромка, край соседней страницы
        at_edge = x < band or y < band or x1 > w - band or y1 > h - band
        close = any(x < b[2] + near and x1 > b[0] - near and y < b[3] + near and y1 > b[1] - near for b in core)
        if at_edge and area < speck and not close:
            continue  # мелкое пятно у края вдали от контента
        kept.append((x, y, x1, y1))
    found = union_box(kept)
    if found is None:
        return fixed_box(content, params)
    box = tuple(int(round(v * s)) for v in found)
    return grow(box, mm_to_px_float(params.margin_mm, content.dpi), content)


def hybrid_box(content: PageContent, params: TrimParams) -> Box:
    """D: рамка A, раздвинутая по каждой стороне до контента без сора (с запасом), если он заходит за отступ."""
    fixed = fixed_box(content, params)
    core = union_box([i.box for i in core_items(content, params)])
    if core is None:
        return fixed
    c = grow(core, mm_to_px_float(params.margin_mm, content.dpi), content)
    return (min(fixed[0], c[0]), min(fixed[1], c[1]), max(fixed[2], c[2]), max(fixed[3], c[3]))


def box_of(method: Method, content: PageContent, params: TrimParams, cover: bool) -> Box:
    """Рамка способа; у обложки — вся страница (обрезки нет)."""
    if cover:
        return (0, 0, content.width, content.height)
    return {Method.FIXED: fixed_box, Method.CONTENT: content_box, Method.INK: ink_box, Method.HYBRID: hybrid_box}[
        method
    ](content, params)


@dataclass(frozen=True)
class TrimMeasures:
    """Меры рамки.

    Поля:
        lost_items: Элементов контента (без сора), не целиком внутри рамки — должно быть 0.
        lost_mm2: Площадь контента без сора вне рамки, мм².
        junk_mm2: Краска внутри рамки в полосе :data:`EDGE_BAND_MM` у краёв страницы вне контента, мм².
        strays_in: Строк-сора внутри рамки.
        cut_mm: Срезано по сторонам, мм: верх, низ, лево, право.
    """

    lost_items: int
    lost_mm2: float
    junk_mm2: float
    strays_in: int
    cut_mm: tuple[float, float, float, float]


def measures(box: Box, content: PageContent, params: TrimParams) -> TrimMeasures:
    """Меры рамки по контенту страницы.

    Args:
        box: Рамка.
        content: Контент страницы.
        params: Числа способов (определение сора).

    Returns:
        Меры.
    """
    x0, y0, x1, y1 = box
    core = core_items(content, params)
    lost_items, lost_px = 0, 0
    for item in core:
        a0, b0, a1, b1 = item.box
        area = max(0, a1 - a0) * max(0, b1 - b0)
        inside = max(0, min(a1, x1) - max(a0, x0)) * max(0, min(b1, y1) - max(b0, y0))
        if inside < area:
            lost_items += 1
            lost_px += area - inside
    # Мусор: краска рабочей копии в полосе у краёв страницы, внутри рамки и вне элементов контента (с полем 1 мм).
    s = content.ink_scale
    ink = content.ink.copy()
    h, w = ink.shape
    pad = mm_to_px_float(1.0, content.dpi) / s
    for item in content.items:
        a0, b0, a1, b1 = (v / s for v in item.box)
        ink[max(0, int(b0 - pad)) : int(b1 + pad) + 1, max(0, int(a0 - pad)) : int(a1 + pad) + 1] = False
    band = int(mm_to_px_float(EDGE_BAND_MM, content.dpi) / s)
    edge = np.zeros_like(ink)
    edge[:band], edge[-band:], edge[:, :band], edge[:, -band:] = True, True, True, True
    inside = np.zeros_like(ink)
    inside[int(y0 / s) : int(y1 / s), int(x0 / s) : int(x1 / s)] = True
    junk_px = float((ink & edge & inside).sum()) * s * s
    stray_in = sum(
        1 for i in strays(content, params) if i.box[0] >= x0 and i.box[2] <= x1 and i.box[1] >= y0 and i.box[3] <= y1
    )
    mm2 = (25.4 / content.dpi) ** 2
    cut = tuple(round(px_to_mm(v, content.dpi), 1) for v in (y0, content.height - y1, x0, content.width - x1))
    return TrimMeasures(lost_items, round(lost_px * mm2, 1), round(junk_px * mm2, 1), stray_in, cut)


def side_distances(content: PageContent, params: TrimParams) -> dict[str, float]:
    """Расстояния от контента без сора до краёв страницы, мм — для калибровки отступов A по паку.

    Args:
        content: Контент страницы.
        params: Числа способов.

    Returns:
        ``{"top", "bottom", "outer", "cut"}`` → мм (NaN — контента нет); у разворота ``LR`` «cut» — правый край.
    """
    core = union_box([i.box for i in core_items(content, params)])
    if core is None:
        return {"top": float("nan"), "bottom": float("nan"), "outer": float("nan"), "cut": float("nan")}
    x0, y0, x1, y1 = core
    left, right = px_to_mm(x0, content.dpi), px_to_mm(content.width - x1, content.dpi)
    cut_left = content.cut_side is Side.LEFT
    return {
        "top": px_to_mm(y0, content.dpi),
        "bottom": px_to_mm(content.height - y1, content.dpi),
        "outer": right if cut_left else left,
        "cut": left if cut_left else right,
    }


# Глубже этого мусор у края не ищется, мм: дальше — уже поле страницы с контентом.
JUNK_REACH_MM = 15.0


def junk_depths(content: PageContent, params: TrimParams) -> dict[str, float]:
    """Глубина мусора у каждого края, мм: насколько далеко внутрь заходят компоненты краски, касающиеся края страницы
    (тень, кромка стопки листов, край соседней страницы за разрезом), не пересекающиеся с контентом.

    Отступ A по стороне должен быть больше этой глубины (по паку — верхний перцентиль) и меньше расстояния от края до
    контента (:func:`side_distances`, нижний перцентиль).

    Args:
        content: Контент страницы.
        params: Числа способов.

    Returns:
        ``{"top", "bottom", "outer", "cut"}`` → мм (0 — мусора у края нет).
    """
    ink = content.ink.astype(np.uint8)
    s = content.ink_scale
    h, w = ink.shape
    reach = mm_to_px_float(JUNK_REACH_MM, content.dpi) / s
    core = [tuple(v / s for v in i.box) for i in core_items(content, params)]
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    depth = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
    for k in range(1, n):
        x, y, cw, ch, _area = stats[k]
        x1, y1 = x + cw, y + ch
        if any(x < b[2] and x1 > b[0] and y < b[3] and y1 > b[1] for b in core):
            continue  # пересекается с контентом — не мусор
        if y <= 0:
            depth["top"] = max(depth["top"], min(y1, reach))
        if y1 >= h:
            depth["bottom"] = max(depth["bottom"], min(h - y, reach))
        if x <= 0:
            depth["left"] = max(depth["left"], min(x1, reach))
        if x1 >= w:
            depth["right"] = max(depth["right"], min(w - x, reach))
    mm = {k: px_to_mm(v * s, content.dpi) for k, v in depth.items()}
    cut_left = content.cut_side is Side.LEFT
    return {
        "top": mm["top"],
        "bottom": mm["bottom"],
        "outer": mm["right"] if cut_left else mm["left"],
        "cut": mm["left"] if cut_left else mm["right"],
    }
