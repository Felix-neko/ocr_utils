"""Выход страниц: выпрямленный разворот режется по кривой линии разреза на страницы, область каждой достраивается закрасом до осевого прямоугольника; флаги кадра и вид кадра (разворот, одна страница) решают, какие страницы писать и резать ли вообще."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from ocr_utils.frame_flags.flags import FrameFlags, Side
from ocr_utils.scan_cropping.v2.fill import FillMethod, fill_to_rectangle, paper_bbox
from ocr_utils.scan_cropping.v2.rectify.coons import Layout, PageRect
from ocr_utils.scan_cropping.v2.rectify.text_run import TextResult
from ocr_utils.scan_cropping.v2.single_page import FrameKind
from ocr_utils.units import MM_PER_INCH


class PagePart(StrEnum):
    """Часть разворота в выходе страниц; значение — суффикс имени файла."""

    LEFT = "_L"  # левая страница
    RIGHT = "_R"  # правая страница
    SPREAD = "_LR"  # разворот целиком (``no-splitting``)
    SINGLE = "_P"  # одна страница кадра (обложка, крышка, лист без разреза): стороны разреза нет


# Сторона флагов кадра → часть выхода.
PART_OF_SIDE = {Side.LEFT: PagePart.LEFT, Side.RIGHT: PagePart.RIGHT}


class PagesStatus(StrEnum):
    """Исход выхода страниц кадра (колонка ``pages_status`` сводки)."""

    OK = "ok"  # страницы записаны (или все проигнорированы флагами)
    NO_FRAMES = "no_frames"  # нет подгоночных областей двух страниц — кадр не нарезан
    NO_PAPER = "no_paper"  # итог v2 без вырезки — нарезать нечего
    NO_PAGE = "no_page"  # в кадре нет страницы (закрытая крышка с ярлыком) — писать нечего
    ERROR = "error"  # ошибка выпрямления или закраса
    OFF = ""  # выход страниц не задан


def wanted_parts(
    flags: FrameFlags, kind: FrameKind = FrameKind.SPREAD, spread_index: int | None = None
) -> list[PagePart]:
    """Какие части кадра писать по флагам и виду кадра.

    Args:
        flags: Флаги кадра.
        kind: Вид кадра (``single_page.FrameKind``).
        spread_index: У кадра ``SLIVER`` — место широкой страницы в развороте (0 — левая, 1 — правая).

    Returns:
        Пусто при ``ignore-both`` и у кадра без страницы; ``[SINGLE]`` у одной страницы; у страницы с полоской
        соседней — её часть разворота (``LEFT``/``RIGHT``: корешок настоящий, обрезка учтёт сторону разреза), если она
        не игнорируется; у разворота — ``[SPREAD]`` при ``no-splitting``, иначе незаигнорированные из
        ``[LEFT, RIGHT]``.
    """
    if flags.skip_frame or kind is FrameKind.NO_PAGE:
        return []
    if kind is FrameKind.SINGLE:
        return [PagePart.SINGLE]
    if kind is FrameKind.SLIVER:
        side = Side.LEFT if spread_index == 0 else Side.RIGHT
        return [] if flags.ignored(side) else [PART_OF_SIDE[side]]
    if not flags.split:
        return [PagePart.SPREAD]
    return [PART_OF_SIDE[side] for side in (Side.LEFT, Side.RIGHT) if not flags.ignored(side)]


def page_areas(layout: Layout, dense: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, ...]:
    """Области левой и правой страницы на выпрямленном холсте (у одной страницы — одна область).

    Пиксель холста ``X`` составная карта берёт из точки ``X + D(X)`` холста Кунса (``D`` — накопленное поле по
    тексту). Пиксель принадлежит странице, если эта точка лежит в прямоугольнике страницы расклада Кунса. Поле у
    каждой страницы своё, поэтому линия разреза между областями — кривая, а сами области слегка отходят от
    прямоугольников там, где поле выпрямляло строки.

    Args:
        layout: Расклад холста Кунса (прямоугольники страниц встык по линии корешка).
        dense: Поле ``(D_x, D_y)`` размера холста итога (при растяжении у корешка он шире ``layout``) в координаты
            холста Кунса; нули — только Кунс.

    Returns:
        Маски bool левой и правой страницы размера холста итога (не пересекаются) или одна маска.
    """
    h, w = dense[0].shape
    xs = np.arange(w, dtype=np.float32)[None, :] + dense[0]
    ys = np.arange(h, dtype=np.float32)[:, None] + dense[1]
    if layout.single:
        return (inside(layout.rects[0], xs, ys),)
    left, right = layout.rects
    # Полуоткрытые интервалы: левая — до линии корешка не включая, правая — от неё; общих пикселей нет.
    in_left = inside(left, xs, ys) & (xs < layout.spine_x)
    in_right = inside(right, xs, ys) & (xs >= layout.spine_x)
    return in_left, in_right


def inside(rect: PageRect, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Маска точек ``(xs, ys)`` внутри прямоугольника ``[x0, x1) × [y0, y1)``.

    Args:
        rect: Прямоугольник страницы.
        xs: Абсциссы точек (массив).
        ys: Ординаты точек (того же размера).

    Returns:
        Маска bool размера ``xs``.
    """
    return (xs >= rect.x0) & (xs < rect.x1) & (ys >= rect.y0) & (ys < rect.y1)


@dataclass(frozen=True)
class PageImage:
    """Готовая страница выхода: часть разворота, картинка прямоугольника, его рамка на холсте и область страницы в ней.

    Поля:
        part: Часть разворота.
        image: Картинка страницы BGR (закрашена до прямоугольника).
        box: Рамка страницы на выпрямленном холсте ``(x0, y0, x1, y1)``.
        area: Область страницы внутри рамки (bool, размер ``image``): вне её — закрас.
    """

    part: PagePart
    image: np.ndarray
    box: tuple[int, int, int, int]
    area: np.ndarray


def cut_pages(
    result: TextResult, parts: list[PagePart], dpi: float, method: FillMethod, seam_mm: float
) -> list[PageImage]:
    """Нарезка выпрямленного разворота на страницы с закрасом до прямоугольника.

    Область страницы (``page_areas``) — опора закраса; рамка вырезки — её осевая рамка. Всё внутри рамки вне
    области (уголки, где область отошла от прямоугольника, и край соседней страницы за кривой линией разреза)
    закрашивается тем же способом, что основной итог v2. Разворот без разреза — объединение двух областей; закрас
    pull-push тогда видит стык страниц по линии корешка.

    Args:
        result: Итог выпрямления кадра.
        parts: Какие части писать (``wanted_parts``).
        dpi: Разрешение итога.
        method: Способ закраса до прямоугольника (``--fill``).
        seam_mm: Ширина мягкого шва для ``pull_push_seam``, мм.

    Returns:
        Страницы в порядке ``parts``; часть с пустой областью пропускается.
    """
    area_of = part_areas(result, parts)
    px_per_mm = dpi / MM_PER_INCH
    pages = []
    for part in parts:
        area = area_of[part]
        box = paper_bbox(area)
        if box is None:
            continue
        spine = result.canvas.spine_x if part is PagePart.SPREAD else None
        image = fill_to_rectangle(result.final, area, method, box=box, px_per_mm=px_per_mm, spine_x=spine,
                                  seam_mm=seam_mm)  # fmt: skip
        if image is not None:
            x0, y0, x1, y1 = box
            pages.append(PageImage(part, image, box, area[y0:y1, x0:x1]))
    return pages


def part_areas(result: TextResult, parts: list[PagePart]) -> dict[PagePart, np.ndarray]:
    """Области частей кадра на выпрямленном холсте.

    Args:
        result: Итог выпрямления кадра.
        parts: Части, которые пишутся.

    Returns:
        Часть → маска bool размера холста. У холста одной страницы любая часть (``SINGLE`` или часть широкой
        страницы кадра ``SLIVER``) — эта страница.
    """
    areas = page_areas(result.layout, result.dense)
    if result.layout.single:
        return {part: areas[0] for part in parts}
    left, right = areas
    return {PagePart.LEFT: left, PagePart.RIGHT: right, PagePart.SPREAD: left | right}


def page_mask_of(result: TextResult, mask: np.ndarray, page: PageImage) -> np.ndarray:
    """Маска в координатах итога v2 → маска страницы выхода (та же составная карта, что у картинки: Кунс + поле).

    Args:
        result: Итог выпрямления кадра (``maps`` — холст → итог v2).
        mask: Маска bool размера итога v2 (например, зона закраса пальцев).
        page: Страница выхода (рамка на холсте и область страницы).

    Returns:
        Маска bool размера страницы, только внутри её области.
    """
    x0, y0, x1, y1 = page.box
    map_x, map_y = (m[y0:y1, x0:x1] for m in result.maps)
    # Билинейная выборка доли маски и порог 0.5 — край маски ложится туда же, куда край картинки.
    warped = cv2.remap(mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return (warped > 0.5) & page.area


def cut_line(layout: Layout, dense: tuple[np.ndarray, np.ndarray], step: int = 4) -> np.ndarray:
    """Линия разреза на холсте: в каждой ``step``-й строке — первый X, чья точка Кунса правее линии корешка.

    Args:
        layout: Расклад холста Кунса.
        dense: Поле ``(D_x, D_y)`` размера холста итога (в координаты холста Кунса).
        step: Шаг по строкам, px.

    Returns:
        Точки линии (N, 2) float64 сверху вниз; у холста одной страницы — пусто (разреза нет).
    """
    if layout.single:
        return np.zeros((0, 2), np.float64)
    h, w = dense[0].shape
    cols = np.arange(w, dtype=np.float32)
    points = []
    for y in range(0, h, step):
        right = cols + dense[0][y] >= layout.spine_x
        x = int(np.argmax(right)) if right.any() else w - 1
        points.append((float(x), float(y)))
    return np.array(points, np.float64)
