"""Прямые линии разворота: разделитель страниц (кандидаты по трапециям и маскам) и ограничители внешних краёв страниц — параллельно краю книги, если он в кадре, иначе виртуально (по умолчанию параллельно разделителю; зеркально или как противоположный край — по выбору)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.keystone import vertical_sides
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM, PageQuad
from ocr_utils.scan_cropping.v2.side_fit import Side
from ocr_utils.scan_cropping.v2.trapezoid import SIDE_QUANTILE

# Край книги ближе этого к рамке кадра в строке — «упирается в рамку», мм бумаги (2 мм = 13.8 px сегментации).
BORDER_MM = 2.0
# Край книги «в кадре», если таких строк меньше этой доли строк пролёта.
BORDER_SHARE = 0.05
# Строки, где рука или зона закраса пальца ближе этого к краю книги, в прямую края не идут, мм (5 мм = 34.5 px).
HAND_REACH_MM = 5.0
# Пролёт края книги по высоте: средняя доля строк страниц (углы обложки и закругления у верха и низа отброшены).
BOOK_ROWS_SHARE = 0.8
# Меньше стольких строк — прямую не строить.
MIN_ROWS = 20
# Кожа дальше этого от зоны закраса пальца — не рука, даже если связана с ней: красный переплёт под пальцем
# (МТС 1991/06 IMG_0316) иначе целиком становится «рукой», мм (15 мм = 103.5 px).
HAND_FROM_PAINT_MM = 15.0
# Если после отбраковки у руки осталось меньше этой доли строк не у рамки — прямая по всем строкам не у рамки.
KEEP_MIN_SHARE = 0.5
# ``SpineMethod.AUTO``: наклон GAP_MID дальше этого от QUAD_MID — маска у корешка рваная или кадр не разворот,
# берётся QUAD_MID, градусы (пак-6: p90 разницы 0.02°; проклейка ПХ 1975/05 IMG_0021 — 1.9°).
AUTO_TILT_GAP_DEG = 1.0


class SpineMethod(StrEnum):
    """Способ провести прямую-разделитель между страницами разворота."""

    # Через середины верхних и нижних концов внутренних сторон трапеций.
    QUAD_MID = "quad_mid"
    # Хубер по середине между правой кромкой маски левой страницы и левой кромкой правой, по строкам.
    GAP_MID = "gap_mid"
    # Как GAP_MID, но кромка — конец сплошного участка маски от тела страницы (островки и языки у корешка не в счёт).
    GAP_MID_RUN = "gap_mid_run"
    # Хубер по точкам линии сгиба по яркости бумаги (стенд ``research/spread_lines``, сгиб — ``research/page_cutting``).
    FOLD = "fold"
    # GAP_MID, а при наклоне дальше ``AUTO_TILT_GAP_DEG`` от QUAD_MID — QUAD_MID (у приподнятой страницы сторона
    # трапеции уходит на изгиб на 2–5 мм от стыка, GAP_MID держит стык; рваная маска сбивает GAP_MID).
    AUTO = "auto"


class VirtualRule(StrEnum):
    """Наклон виртуального ограничителя, когда край книги с его стороны за кадром."""

    # Зеркально видимому противоположному краю относительно разделителя: 2·θ_разделителя − θ_противоположного.
    MIRROR = "mirror"
    # Параллельно разделителю.
    SPINE = "spine"
    # Параллельно видимому противоположному краю книги.
    OPPOSITE = "opposite"


class EdgeKind(StrEnum):
    """Откуда наклон ограничителя внешнего края страницы."""

    BOOK = "book"  # параллельно краю книги, край целиком в кадре
    VIRTUAL = "virtual"  # край книги за кадром: наклон отражён от противоположного края относительно разделителя
    NONE = "none"  # не построен


@dataclass(frozen=True)
class Line:
    """Прямая ``x(y) = x0 + k·(y − y0)`` на масштабе сегментации и её пролёт по высоте.

    Поля:
        x0: Абсцисса в строке ``y0``, px.
        y0: Опорная строка, px.
        k: Наклон dx/dy (плюс — низ правее верха).
        y_top: Верх пролёта для рисования, px.
        y_bot: Низ пролёта, px.
    """

    x0: float
    y0: float
    k: float
    y_top: float
    y_bot: float

    def x_at(self, y: np.ndarray | float) -> np.ndarray | float:
        """Абсцисса прямой в строках ``y``."""
        return self.x0 + self.k * (np.asarray(y, np.float64) - self.y0)

    @property
    def tilt_deg(self) -> float:
        """Наклон от вертикали, градусы (знак как у ``keystone.tilt``)."""
        return float(np.degrees(np.arctan(self.k)))

    def ends(self) -> np.ndarray:
        """Концы отрезка пролёта ``[[x, y_top], [x, y_bot]]``."""
        return np.array([[self.x_at(self.y_top), self.y_top], [self.x_at(self.y_bot), self.y_bot]], np.float64)


@dataclass(frozen=True)
class BookEdge:
    """Край книги с одной стороны: прямая, видимость и точки, по которым она строилась.

    Поля:
        side: Сторона (``Side.LEFT`` / ``Side.RIGHT``).
        line: Прямая по крайним точкам книги или None (точек мало).
        visible: Край целиком в кадре (доля строк у рамки меньше ``border_share``).
        border_share: Доля строк пролёта, где край книги у рамки кадра.
        points: Точки (N, 2) ``x, y``, вошедшие в прямую.
        dropped: Точки (M, 2), отброшенные: у рамки, рядом рука или палец.
    """

    side: Side
    line: Line | None
    visible: bool
    border_share: float
    points: np.ndarray
    dropped: np.ndarray


@dataclass(frozen=True)
class SpreadLines:
    """Прямые разворота на масштабе сегментации.

    Поля:
        method: Способ разделителя.
        spine: Разделитель страниц.
        left: Ограничитель левого края левой страницы или None.
        right: Ограничитель правого края правой страницы или None.
        left_kind: Откуда наклон левого ограничителя.
        right_kind: Откуда наклон правого.
        book_left: Левый край книги.
        book_right: Правый край книги.
    """

    method: SpineMethod
    spine: Line
    left: Line | None
    right: Line | None
    left_kind: EdgeKind
    right_kind: EdgeKind
    book_left: BookEdge
    book_right: BookEdge


def mm_px(mm: float) -> float:
    """Миллиметры бумаги → px сегментации."""
    return mm * SEG_PX_PER_MM


def fit_line(xs: np.ndarray, ys: np.ndarray, y_top: float, y_bot: float) -> Line | None:
    """Прямая ``x(y)`` Хубером (``cv2.fitLine``, ``DIST_HUBER``) по точкам.

    Args:
        xs: Абсциссы точек.
        ys: Строки точек.
        y_top: Верх пролёта прямой.
        y_bot: Низ пролёта.

    Returns:
        Прямая или None (точек меньше ``MIN_ROWS`` или прямая почти горизонтальна).
    """
    if len(xs) < MIN_ROWS:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    vx, vy, x0, y0 = (float(v) for v in cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
    if abs(vy) < 1e-3:
        return None
    return Line(x0, y0, vx / vy, y_top, y_bot)


def pages_span(pages: list[PageQuad]) -> tuple[float, float]:
    """Пролёт по высоте двух страниц: от верхней вершины трапеций до нижней, px."""
    quads = [p.quad for p in pages]
    return float(min(q[:, 1].min() for q in quads)), float(max(q[:, 1].max() for q in quads))


def row_edges(mask: np.ndarray, rows: np.ndarray, side: Side) -> tuple[np.ndarray, np.ndarray]:
    """Крайний пиксель маски по строкам с нужной стороны.

    Args:
        mask: Булева маска.
        rows: Строки.
        side: ``Side.LEFT`` — самый левый пиксель, ``Side.RIGHT`` — самый правый.

    Returns:
        ``(строки, где маска есть, абсциссы)``.
    """
    sub = mask[rows]
    has = sub.any(axis=1)
    w = mask.shape[1]
    xs = np.argmax(sub, axis=1) if side is Side.LEFT else w - 1 - np.argmax(sub[:, ::-1], axis=1)
    return rows[has], xs[has].astype(np.float64)


def run_edges(mask: np.ndarray, rows: np.ndarray, side: Side, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Конец сплошного участка маски от тела страницы по строкам: участок, содержащий ``centre[i]``, до первой дыры.

    Args:
        mask: Маска страницы.
        rows: Строки.
        side: С какой стороны искать конец участка.
        centre: Абсцисса тела страницы в каждой строке ``rows``.

    Returns:
        ``(строки, абсциссы конца участка)`` — строки, где тело страницы попадает в маску.
    """
    out_rows, out_xs = [], []
    w = mask.shape[1]
    for y, c in zip(rows, np.clip(centre.astype(int), 0, w - 1)):
        line = mask[y]
        if not line[c]:
            continue
        if side is Side.RIGHT:
            gap = np.flatnonzero(~line[c:])
            out_xs.append(c + (gap[0] - 1 if len(gap) else w - 1 - c))
        else:
            gap = np.flatnonzero(~line[: c + 1][::-1])
            out_xs.append(c - (gap[0] - 1 if len(gap) else c))
        out_rows.append(y)
    return np.asarray(out_rows), np.asarray(out_xs, np.float64)


def spine_quad_mid(pages: list[PageQuad]) -> Line:
    """Разделитель через середины концов внутренних сторон трапеций (``SpineMethod.QUAD_MID``)."""
    _, inner_left = vertical_sides(pages[0].quad)
    inner_right, _ = vertical_sides(pages[1].quad)
    top, bot = (inner_left + inner_right) / 2
    k = (bot[0] - top[0]) / max(bot[1] - top[1], 1e-6)
    return Line(float(top[0]), float(top[1]), float(k), *pages_span(pages))


def spine_gap_mid(pages: list[PageQuad], run: bool) -> Line | None:
    """Разделитель Хубером по середине между внутренними кромками масок страниц (``GAP_MID`` / ``GAP_MID_RUN``).

    Args:
        pages: Две страницы слева направо.
        run: True — кромка как конец сплошного участка от тела страницы (``run_edges``), False — крайний пиксель.

    Returns:
        Прямая или None (строк с обеими страницами мало).
    """
    left, right = pages[0].mask, pages[1].mask
    rows = np.flatnonzero(left.any(axis=1) & right.any(axis=1))
    if run:
        # Тело страницы в строке — середина её трапеции по горизонтали в этой строке.
        lc = np.interp(rows, *_quad_centre(pages[0].quad))
        rc = np.interp(rows, *_quad_centre(pages[1].quad))
        ly, lx = run_edges(left, rows, Side.RIGHT, lc)
        ry, rx = run_edges(right, rows, Side.LEFT, rc)
    else:
        ly, lx = row_edges(left, rows, Side.RIGHT)
        ry, rx = row_edges(right, rows, Side.LEFT)
    both, li, ri = np.intersect1d(ly, ry, return_indices=True)
    return fit_line((lx[li] + rx[ri]) / 2, both.astype(np.float64), *pages_span(pages))


def _quad_centre(quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Середина трапеции по горизонтали в строках её верха и низа: ``(строки, абсциссы)`` для ``np.interp``."""
    left, right = vertical_sides(quad)
    mid = (left + right) / 2
    return mid[:, 1], mid[:, 0]


def spine_candidates(pages: list[PageQuad]) -> dict[SpineMethod, Line]:
    """Разделители по способам ядра (без ``FOLD``) для двух страниц с трапециями; не построенные — пропущены.

    ``AUTO`` всегда есть: ``GAP_MID`` или откат на ``QUAD_MID`` (``auto_source``).
    """
    out = {SpineMethod.QUAD_MID: spine_quad_mid(pages)}
    for method, run in ((SpineMethod.GAP_MID, False), (SpineMethod.GAP_MID_RUN, True)):
        line = spine_gap_mid(pages, run)
        if line is not None:
            out[method] = line
    out[SpineMethod.AUTO] = out[auto_source(out)]
    return out


def auto_source(candidates: dict[SpineMethod, Line], max_gap_deg: float = AUTO_TILT_GAP_DEG) -> SpineMethod:
    """Какой способ стоит за ``SpineMethod.AUTO``: ``GAP_MID``, если он построен и его наклон не дальше
    ``max_gap_deg`` от ``QUAD_MID``, иначе ``QUAD_MID``.

    Args:
        candidates: Разделители по способам (обязательно с ``QUAD_MID``).
        max_gap_deg: Наибольшая разница наклонов, градусы.

    Returns:
        Способ-источник.
    """
    gap = candidates.get(SpineMethod.GAP_MID)
    quad = candidates[SpineMethod.QUAD_MID]
    if gap is not None and abs(gap.tilt_deg - quad.tilt_deg) <= max_gap_deg:
        return SpineMethod.GAP_MID
    return SpineMethod.QUAD_MID


def book_edge(
    book: np.ndarray,
    hands: np.ndarray,
    side: Side,
    span: tuple[float, float],
    border_mm: float = BORDER_MM,
    border_share: float = BORDER_SHARE,
    hand_reach_mm: float = HAND_REACH_MM,
) -> BookEdge:
    """Прямая края книги с одной стороны по крайним точкам грубой маски книги и её видимость.

    Строки — средние ``BOOK_ROWS_SHARE`` пролёта страниц. Отбрасываются строки, где край книги у рамки кадра (по ним же
    считается видимость), и где рука или зона закраса пальца ближе ``hand_reach_mm`` к краю (палец за край книги).

    Args:
        book: Грубая маска книги (не фон) масштаба сегментации.
        hands: Руки и зона закраса пальцев (bool) того же размера.
        side: ``Side.LEFT`` или ``Side.RIGHT``.
        span: Пролёт страниц по высоте ``(верх, низ)``, px.
        border_mm: Насколько близко к рамке край считается «упёршимся», мм.
        border_share: Наибольшая доля таких строк для «край в кадре».
        hand_reach_mm: Окно вокруг края, где рука или палец отбрасывают строку, мм.

    Returns:
        Край книги.
    """
    h, w = book.shape
    top, bot = span
    pad = (1 - BOOK_ROWS_SHARE) / 2 * (bot - top)
    rows = np.arange(max(int(top + pad), 0), min(int(bot - pad), h - 1) + 1)
    ys, xs = row_edges(book, rows, side)
    empty = np.zeros((0, 2))
    if len(ys) == 0:
        return BookEdge(side, None, False, 1.0, empty, empty)
    # Строки, где край у рамки кадра: книга продолжается за кадр.
    near = mm_px(border_mm)
    at_border = xs <= near if side is Side.LEFT else xs >= w - 1 - near
    share = float(at_border.mean())
    # Строки с рукой или пальцем у края: в окне ±hand_reach_mm от края по строке.
    reach = int(round(mm_px(hand_reach_mm)))
    near_hand = np.array([hands[y, max(int(x) - reach, 0) : int(x) + reach + 1].any() for y, x in zip(ys, xs)])
    keep = ~at_border & ~near_hand
    if keep.sum() < max(MIN_ROWS, KEEP_MIN_SHARE * (~at_border).sum()):
        # Рука вдоль большей части края (или кожей сочтён переплёт) — по остаткам прямую не строить: по всем строкам
        # не у рамки.
        keep = ~at_border
    points = np.stack([xs[keep], ys[keep]], axis=1)
    dropped = np.stack([xs[~keep], ys[~keep]], axis=1)
    line = fit_line(xs[keep], ys[keep].astype(np.float64), top, bot)
    return BookEdge(side, line, share < border_share and line is not None, share, points, dropped)


def finger_hands(skin: np.ndarray, paint: np.ndarray, reach_mm: float = HAND_FROM_PAINT_MM) -> np.ndarray:
    """Руки у края книги: компоненты кожи, касающиеся зоны закраса пальцев, в пределах ``reach_mm`` от неё, плюс сама
    зона.

    Кожа по цвету ловит и коричнево-красные переплёты (МТС 1991/01 IMG_0576): без связи с пальцем такая «кожа» —
    обложка, и строки у неё в прямую края книги идут. Если палец лежит на таком переплёте (МТС 1991/06 IMG_0316),
    компонента связана с пальцем во всю высоту книги — поэтому рукой считается только её часть у закраса.

    Args:
        skin: Кожа по цвету (bool) масштаба сегментации.
        paint: Зона закраса пальцев (bool) того же масштаба.
        reach_mm: Насколько далеко от зоны закраса кожа ещё считается рукой, мм бумаги.

    Returns:
        Булева маска рук.
    """
    count, labels = cv2.connectedComponents(skin.astype(np.uint8))
    touched = np.unique(labels[paint & skin])
    touched = touched[touched > 0]
    # Окрестность зоны закраса радиусом reach_mm: дальше связанная с пальцем кожа — уже переплёт или фон.
    r = int(round(mm_px(reach_mm)))
    near = cv2.dilate(paint.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    return (np.isin(labels, touched) & near) | paint


def page_bound(
    mask: np.ndarray, k: float, side: Side, span: tuple[float, float], quantile: float = SIDE_QUANTILE
) -> Line | None:
    """Ограничитель внешнего края страницы: прямая с наклоном ``k``, сдвинутая наружу до ``quantile`` кромки страницы.

    Args:
        mask: Маска страницы.
        k: Наклон dx/dy.
        side: Внешняя сторона страницы (``Side.LEFT`` у левой страницы, ``Side.RIGHT`` у правой).
        span: Пролёт для рисования ``(верх, низ)``, px.
        quantile: Доля точек кромки страницы, остающихся внутри (как у сторон трапеции).

    Returns:
        Прямая или None (кромки нет).
    """
    rows = np.flatnonzero(mask.any(axis=1))
    ys, xs = row_edges(mask, rows, side)
    if len(ys) < MIN_ROWS:
        return None
    y0 = float((span[0] + span[1]) / 2)
    # Сдвиг каждой точки кромки до прямой с наклоном k через опорную строку.
    offsets = xs - k * (ys - y0)
    x0 = np.quantile(offsets, 1 - quantile) if side is Side.LEFT else np.quantile(offsets, quantile)
    return Line(float(x0), y0, float(k), *span)


def mirror_k(spine: Line, opposite: Line) -> float:
    """Наклон, зеркальный наклону ``opposite`` относительно разделителя: ``tan(2·θ_spine − θ_opposite)``."""
    return float(np.tan(2 * np.arctan(spine.k) - np.arctan(opposite.k)))


def spread_lines(
    pages: list[PageQuad],
    spine: Line,
    method: SpineMethod,
    book: np.ndarray,
    hands: np.ndarray,
    rule: VirtualRule = VirtualRule.SPINE,
) -> SpreadLines:
    """Ограничители внешних краёв двух страниц по разделителю и краям книги.

    Наклон ограничителя: край книги с этой стороны в кадре — его наклон (``EdgeKind.BOOK``); за кадром, а
    противоположный в кадре — по правилу ``rule`` (параллельно разделителю, зеркально противоположному краю или как
    он); оба за кадром — наклон разделителя (оба ``EdgeKind.VIRTUAL``). Положение — внешняя кромка бумаги страницы (``page_bound``).

    Args:
        pages: Две страницы с трапециями слева направо.
        spine: Разделитель.
        method: Каким способом построен разделитель (для подписи).
        book: Грубая маска книги масштаба сегментации.
        hands: Руки и зона закраса пальцев (bool).
        rule: Наклон виртуального ограничителя при видимом противоположном крае (по умолчанию — параллельно разделителю:
            на паках 4 и 6 медиана ошибки 0.58° / 0.65° против 0.90° / 0.68° у зеркала, ``research/spread_lines``).

    Returns:
        Прямые разворота.
    """
    span = pages_span(pages)
    edges = {s: book_edge(book, hands, s, span) for s in (Side.LEFT, Side.RIGHT)}
    out = {}
    for page, side, other in ((pages[0], Side.LEFT, Side.RIGHT), (pages[1], Side.RIGHT, Side.LEFT)):
        own, opp = edges[side], edges[other]
        if own.visible:
            k, kind = own.line.k, EdgeKind.BOOK
        elif opp.visible:
            virtual = {VirtualRule.MIRROR: mirror_k(spine, opp.line), VirtualRule.SPINE: spine.k,
                       VirtualRule.OPPOSITE: opp.line.k}  # fmt: skip
            k, kind = virtual[rule], EdgeKind.VIRTUAL
        else:
            k, kind = spine.k, EdgeKind.VIRTUAL
        line = page_bound(page.mask, k, side, span)
        out[side] = (line, kind if line is not None else EdgeKind.NONE)
    return SpreadLines(method, spine, out[Side.LEFT][0], out[Side.RIGHT][0], out[Side.LEFT][1], out[Side.RIGHT][1],
                       edges[Side.LEFT], edges[Side.RIGHT])  # fmt: skip
