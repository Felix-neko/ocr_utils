"""Контент выпрямленной страницы для зоны обрезки: строки и блоки текста (``text_blocks``, движок по краске), растр, таблицы и line art (``page_layout`` по пикселям), компоненты краски; сторона у линии разреза по паспорту страницы."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.rectify.coons import PageRect
from ocr_utils.scan_cropping.v2.rectify.text_field import analyze_page, page_regions
from ocr_utils.units import mm_to_px, mm_to_px_float

# Полоса краски у края страницы, мм: здесь меряется «мусор внутри рамки».
EDGE_BAND_MM = 8.0
# Короткая строка — короче этого, мм: номер страницы, одиночная цифра, повёрнутая метка у корешка.
SHORT_LINE_MM = 12.0
# Краска — пиксели темнее фона страницы на столько уровней (рабочая копия 150 dpi, медианный фон).
INK_DROP = 60
# Плотность рабочей копии компонент краски, точек на дюйм.
INK_DPI = 150.0


class Side(Enum):
    """Сторона страницы."""

    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


Box = tuple[int, int, int, int]  # (x0, y0, x1, y1), правая и нижняя границы не включительно


@dataclass(frozen=True)
class Item:
    """Элемент контента страницы: рамка px страницы, вид и признаки.

    Поля:
        box: Рамка ``(x0, y0, x1, y1)``.
        kind: ``line`` (ось строки), ``block`` (текстовый блок), ``raster``, ``table``, ``line_art``.
        short: Короткая строка (короче :data:`SHORT_LINE_MM`) — кандидат в сор у края.
    """

    box: Box
    kind: str
    short: bool = False


@dataclass(frozen=True)
class PageContent:
    """Контент страницы и её привязка.

    Поля:
        width: Ширина страницы, px.
        height: Высота страницы, px.
        dpi: Разрешение страницы.
        cut_side: Сторона у линии разреза (``LEFT`` у правой страницы, ``RIGHT`` у левой) или None (разворот ``LR``).
        items: Строки, блоки и области.
        ink: Маска краски страницы на рабочей копии :data:`INK_DPI` (bool).
        ink_scale: px страницы на px рабочей копии краски.
    """

    width: int
    height: int
    dpi: float
    cut_side: Side | None
    items: tuple[Item, ...]
    ink: np.ndarray
    ink_scale: float

    @property
    def lines(self) -> list[Item]:
        """Оси строк."""
        return [i for i in self.items if i.kind == "line"]

    @property
    def body(self) -> list[Item]:
        """Тело страницы: блоки, растр, таблицы, line art и длинные строки (без коротких)."""
        return [i for i in self.items if i.kind != "line" or not i.short]


def cut_side_of(part: str) -> Side | None:
    """Сторона у линии разреза по части кадра из паспорта (``L`` → правая, ``R`` → левая, ``LR`` и ``P`` → нет)."""
    return {"L": Side.RIGHT, "R": Side.LEFT}.get(part)


def ink_mask(bgr: np.ndarray, dpi: float) -> tuple[np.ndarray, float]:
    """Маска краски на рабочей копии :data:`INK_DPI`: темнее медианы фона на :data:`INK_DROP` уровней.

    Args:
        bgr: Страница BGR.
        dpi: Её разрешение.

    Returns:
        Маска bool рабочей копии и px страницы на px копии.
    """
    f = min(1.0, INK_DPI / dpi)
    gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    # Фон — медиана в окне 15 мм: тень у края и неровное освещение не становятся краской.
    win = mm_to_px(15.0, INK_DPI) | 1
    background = cv2.medianBlur(gray, min(win, 255))
    return (background.astype(np.int16) - gray.astype(np.int16)) > INK_DROP, 1.0 / f


def page_content(bgr: np.ndarray, dpi: float, part: str) -> PageContent:
    """Разбор контента страницы: строки и блоки (``analyze_page`` выпрямления, движок по краске), растр, таблицы и line
    art (``page_regions``), краска.

    Args:
        bgr: Страница BGR (выход ``scan_cropping.v2 --output-pages``).
        dpi: Настоящее разрешение страницы (паспорт).
        part: Часть разворота из паспорта (``L``, ``R``, ``LR``).

    Returns:
        Контент страницы.
    """
    h, w = bgr.shape[:2]
    box = (0, 0, w, h)
    regions = page_regions(bgr, box, dpi)
    page = analyze_page(bgr, box, dpi, PageRect(0.0, 0.0, float(w), float(h)), regions)
    items: list[Item] = []
    short_px = mm_to_px_float(SHORT_LINE_MM, dpi)
    for axis in page.analysis.axes:
        pts = page.to_canvas(axis.points)
        half = axis.height * page.scale / 2
        x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
        y0, y1 = float(pts[:, 1].min() - half), float(pts[:, 1].max() + half)
        items.append(Item(_clip((x0, y0, x1, y1), w, h), "line", short=(x1 - x0) < short_px))
    scale, off = page.scale, page.offset
    for block in page.analysis.blocks:
        if not block.rows:
            continue
        # Рамка блока — по краскам рядов (x0/x1 ряда — края краски, y ± половина высоты), px рабочей копии → страницы.
        x0 = min(r.x0 for r in block.rows)
        x1 = max(r.x1 for r in block.rows)
        y0 = min(r.y - r.height / 2 for r in block.rows)
        y1 = max(r.y + r.height / 2 for r in block.rows)
        items.append(Item(_clip((x0 * scale + off[0], y0 * scale + off[1], x1 * scale + off[0], y1 * scale + off[1]),
                                w, h), "block"))  # fmt: skip
    items = unglue_edges(items, w, dpi)
    for region in regions:
        items.append(Item(_clip(region.box, w, h), str(region.kind.value)))
    ink, ink_scale = ink_mask(bgr, dpi)
    return PageContent(w, h, dpi, cut_side_of(part), tuple(items), ink, ink_scale)


# Строка «прилипла» к боковому краю, если её рамка ближе EDGE_TOUCH_MM к краю, а соседи в окне ±NEIGHBOR_MM по
# вертикали — дальше CLEAR_MM от края и выступают внутрь страницы больше чем на STEP_MM (ПХ 1974/01 IMG_0082_R: ось в
# 1.0 мм от края при колонке в 4.5 мм — тень кромки сцепилась со строкой).
EDGE_TOUCH_MM = 3.0
NEIGHBOR_MM = 25.0
CLEAR_MM = 3.0
STEP_MM = 2.0


def unglue_edges(items: list[Item], width: int, dpi: float) -> list[Item]:
    """Строки, прилипшие к боковому краю страницы крошкой краски, — обрезаются по краю соседних строк.

    Разбор иногда сцепляет со строкой пятно тени или кромки у самого края (ПХ 1974/01 IMG_0083_R: ось начинается в 3 px
    от края, хотя колонка — в 4 мм). Такая строка тянула рамку контента к краю. Если у строки, касающейся края, есть
    соседи по вертикали (±:data:`NEIGHBOR_MM`), отстоящие от этого края дальше :data:`CLEAR_MM`, её край ставится на
    наименьший из их краёв.

    Args:
        items: Элементы (строки и блоки).
        width: Ширина страницы, px.
        dpi: Разрешение.

    Returns:
        Элементы с поправленными строками.
    """
    touch = mm_to_px_float(EDGE_TOUCH_MM, dpi)
    near = mm_to_px_float(NEIGHBOR_MM, dpi)
    clear = mm_to_px_float(CLEAR_MM, dpi)
    step = mm_to_px_float(STEP_MM, dpi)
    lines = [i for i in items if i.kind == "line"]
    fixed = []
    for item in items:
        x0, y0, x1, y1 = item.box
        if item.kind == "line" and (x0 < touch or x1 > width - touch):
            cy = (y0 + y1) / 2
            peers = [o for o in lines if o is not item and abs((o.box[1] + o.box[3]) / 2 - cy) < near]
            left = [o.box[0] for o in peers if o.box[0] > clear]
            right = [o.box[2] for o in peers if o.box[2] < width - clear]
            if x0 < touch and left and min(left) - x0 > step:
                x0 = min(left)
            if x1 > width - touch and right and x1 - max(right) > step:
                x1 = max(right)
            item = Item((x0, y0, max(x1, x0 + 1), y1), item.kind, item.short)
        fixed.append(item)
    return fixed


def _clip(box, w: int, h: int) -> Box:
    """Рамка в целых px, обрезанная по странице."""
    x0, y0, x1, y1 = box
    return (max(0, int(np.floor(x0))), max(0, int(np.floor(y0))), min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1))))
