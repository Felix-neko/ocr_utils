"""Полоса-вклейка как область по итогам уточнения ленты: строки, где между краями страниц шире ``ROW_MIN_MM``, серии таких строк (полоса может идти по части линии разреза и менять ширину), доводка конца до края листа при обтекании страницами, исправление страниц SAM 3 по уточнённым краям, линия корешка и подгоночные области с зоной сплайна у корешка по сторонам."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from scipy import ndimage as ndi

from ocr_utils.scan_cropping.v2.gutter_strip import largest_component
from ocr_utils.scan_cropping.v2.page_boundary import CURVE_SAMPLES, EXTEND, SPINE_ZONE_MM
from ocr_utils.scan_cropping.v2.page_boundary_adaptive import fit_boundary_adaptive
from ocr_utils.scan_cropping.v2.page_frames import STRIP_GUARD_MM, PageFrame, cut_at, line_points
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM, PageQuad, page_quads
from ocr_utils.scan_cropping.v2.page_sides import Side
from ocr_utils.scan_cropping.v2.spread_lines import Line, SpineMethod, fit_line, spine_candidates
from research.gutter_strip.band import StripFrame
from research.gutter_strip.refine import SpineBand

# Строка «с полосой» — ширина между уточнёнными краями страниц не меньше, мм (узкая тёмная щель сгиба — 1–2 мм).
ROW_MIN_MM = 3.0
# Короткие разрывы серии по строкам закрываются, мм; серия короче ``RUN_MIN_MM`` — не полоса.
ROW_CLOSE_MM = 10.0
RUN_MIN_MM = 30.0
# Серия — полоса, если медиана её ширины не меньше, мм (``refine.STRIP_WIDTH_MM``: полосы ПХ 1975 — от 5.1, без
# полосы ПХ и ХиП — до 4.4).
STRIP_WIDTH_MM = 5.0
# Торцы полосы — по её бумаге в ``END_ZONE_MM`` от конца серии и за ним (``shape_ends``): не срез по строкам страниц.
END_ZONE_MM = 15.0
# Обтекание у конца: конец серии ближе ``END_SNAP_MM`` к верху (низу) страниц у полосы, а между ними — страница
# (перемычка над полосой, ``END_BRIDGE_SHARE`` столбцов полосы), — полоса доводится до края листа, перемычка — полосе.
END_SNAP_MM = 8.0
END_BRIDGE_SHARE = 0.6
# Верх (низ) страниц у полосы — по столбцам в ``PAGE_EDGE_FROM_MM … PAGE_EDGE_TO_MM`` от краёв полосы к страницам.
PAGE_EDGE_FROM_MM = 2.0
PAGE_EDGE_TO_MM = 6.0
# Зона страницы у корешка для правки карты экземпляров SAM 3, мм от разделителя (как лента уточнения).
CORRECT_HALF_MM = 25.0
# Полоса на части длины, если строк без неё не меньше этой доли пролёта обеих страниц: линия корешка — по стыку
# страниц и продолжающему его краю полосы (``region_spine``), иначе — по середине полосы.
JUNCTION_MIN_SHARE = 0.3
# Защита у полосы-области для верха и низа, мм: точки контура страницы ближе к полосе негодны. Страницы исправлены по
# краю полосы, вертикальная кромка маски — вплотную к ней; прежние 6 мм (``page_frames.STRIP_GUARD_MM``, против
# нырка за кромкой у прямой вспомогательной линии, ПХ 1975/09 IMG_0021) выбрасывали изгиб страницы у полосы.
REGION_GUARD_MM = 1.5
# Выемки края страницы (клочок обложки, перекрывший верх: ПХ 1975/12 IMG_0084, правая страница) закрываются перед
# подгонкой горизонтальным закрытием на ``NOTCH_CLOSE_MM`` — кроме стороны полосы: плавный изгиб страницы у полосы не
# выемка, за ним не страница, закрывать нечего.
NOTCH_CLOSE_MM = 15.0
# Отсев точек верха и низа у полосы (``strip_side_rejects``): отвесные — перепад края круче ``STEEP_SLOPE`` на 1 мм
# (срезанный наискось угол страницы у полосы: 04/IMG_0066 — 10 мм за 2 мм); выемка — глубина от жёсткой кривой
# наибольшая не у полосы, а у полосы мельче её на ``NOTCH_RETURN_MM`` (12/IMG_0084: 20.1 → 25.1 → 20.1 мм), негодны
# её столбцы глубже ``NOTCH_FLOOR_MM``. Монотонный загиб к полосе — 2–3 мм (04/IMG_0066, 0082) — остаётся.
STEEP_SLOPE = 1.0
NOTCH_RETURN_MM = 1.5
NOTCH_FLOOR_MM = 1.0
# Край страницы у полосы для кривой верха (низа) — по строкам в ``EDGE_ROWS_MM`` от точки кривой.
EDGE_ROWS_MM = 3.0
# Конец полосы «у верха (низа)» для подгонки: ближе этого к верху (низу) страниц у полосы, мм.
END_TOUCH_MM = 20.0
# Продолжение полосы вдоль линии корешка за каждый её конец, мм (``extend_along_spine``): отсекает тонкие мостики масок
# страниц SAM, обтекающих конец полосы (ПХ 1975/12 IMG_0066); ширина — как у конца полосы. Тем же расстоянием
# решается, подходит ли полоса к верху (низу) страниц.
EXTENSION_MM = 20.0
EXTENSION_SPAN_MM = 5.0
# Экстраполяция верха и низа через полосу и её продолжение до линии корешка (``extrapolate``): парабола по последним
# ``EXTRAP_FIT_MM`` кривой у края страницы; наклон в конце — не круче ``EXTRAP_MAX_SLOPE``, вклад кривизны на всём
# продолжении — не больше ``EXTRAP_CURVE_MM``.
EXTRAP_FIT_MM = 30.0
EXTRAP_MAX_SLOPE = 0.15
EXTRAP_CURVE_MM = 1.0


@dataclass
class StripRegion:
    """Полоса как область.

    Поля:
        mask: Маска полосы на масштабе сегментации.
        runs: Серии строк кадра сегментации ``(y_верх, y_низ)``, конец не включается.
        widths_mm: Ширина по строкам кадра сегментации в сериях (NaN вне серий).
        snapped: Доведённые до края листа концы: ``"top"``, ``"bottom"``.
        touch_top: Полоса доходит до верха страниц (``END_TOUCH_MM``) — верх у корешка строится как с полосой.
        touch_bottom: То же у низа.
        extension: Продолжение полосы вдоль линии корешка за её концы (``extend_along_spine``) или None.
    """

    mask: np.ndarray
    runs: list[tuple[int, int]]
    widths_mm: np.ndarray
    snapped: list[str] = field(default_factory=list)
    touch_top: bool = False
    touch_bottom: bool = False
    extension: np.ndarray | None = None

    def thirds_mm(self) -> tuple[float, float, float]:
        """Медиана ширины в верхней, средней и нижней трети пролёта полосы, мм (NaN — полосы там нет)."""
        rows = np.flatnonzero(np.isfinite(self.widths_mm))
        if not len(rows):
            return (float("nan"),) * 3
        parts = np.array_split(np.arange(rows[0], rows[-1] + 1), 3)
        return tuple(float(np.nanmedian(self.widths_mm[p])) if np.isfinite(self.widths_mm[p]).any()
                     else float("nan") for p in parts)  # fmt: skip

    def span_mm(self) -> float:
        """Пролёт полосы по высоте от верха первой серии до низа последней, мм."""
        return (self.runs[-1][1] - self.runs[0][0]) / SEG_PX_PER_MM if self.runs else 0.0


def seg_edges(band: SpineBand, edge: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Край по строкам ленты → абсцисса по строкам кадра сегментации (медиана строк ленты в строке кадра; NaN — нет)."""
    h = shape[0]
    ys = np.round(band.ys_seg).astype(int)
    xs = band.x_full(edge) / band.ratio
    out = np.full(h, np.nan)
    ok = (ys >= 0) & (ys < h) & np.isfinite(xs)
    if not ok.any():
        return out
    order = np.argsort(ys[ok])
    y_sorted, x_sorted = ys[ok][order], xs[ok][order]
    uniq, start = np.unique(y_sorted, return_index=True)
    for y, chunk in zip(uniq, np.split(x_sorted, start[1:])):
        out[y] = float(np.median(chunk))
    return out


def page_rows(pages: list[PageQuad]) -> np.ndarray:
    """Строки кадра, где есть обе страницы."""
    return pages[0].mask.any(axis=1) & pages[1].mask.any(axis=1)


def build_region(sf: StripFrame, band: SpineBand, left: np.ndarray, right: np.ndarray) -> StripRegion | None:
    """Полоса как область по уточнённым краям: строки шире ``ROW_MIN_MM`` в пролёте обеих страниц, серии не короче
    ``RUN_MIN_MM`` с медианой ширины не меньше ``STRIP_WIDTH_MM``; маска — между краями по строкам, в пределах книги.

    Args:
        sf: Кадр стенда.
        band: Лента уточнения.
        left: Правый край левой страницы по строкам ленты.
        right: Левый край правой страницы.

    Returns:
        Область или None (ни одной серии-полосы).
    """
    shape = sf.frame.img.shape[:2]
    xl, xr = seg_edges(band, left, shape), seg_edges(band, right, shape)
    width = (xr - xl) / SEG_PX_PER_MM
    rows_ok = page_rows(sf.frame.pages)
    present = rows_ok & np.isfinite(width) & (width >= ROW_MIN_MM)
    close = max(1, int(round(ROW_CLOSE_MM * SEG_PX_PER_MM)))
    # Закрытие только заполняет короткие разрывы (у края массива оно бы укоротило серию).
    present = (present | ndi.binary_closing(present, structure=np.ones(close, bool))) & rows_ok & np.isfinite(width)
    lab, n = ndi.label(present)
    runs, widths = [], np.full(shape[0], np.nan)
    for k in range(1, n + 1):
        ys = np.flatnonzero(lab == k)
        if len(ys) < RUN_MIN_MM * SEG_PX_PER_MM or np.median(width[ys]) < STRIP_WIDTH_MM:
            continue
        runs.append((int(ys[0]), int(ys[-1]) + 1))
        widths[ys] = width[ys]
    if not runs:
        return None
    h, w = shape
    cols = np.arange(w)[None, :]
    inside = np.isfinite(widths)
    lo = np.where(inside, xl, np.inf)[:, None]
    hi = np.where(inside, xr, -np.inf)[:, None]
    mask = (cols >= lo) & (cols < hi) & sf.frame.book
    return StripRegion(mask, runs, widths)


def shape_ends(region: StripRegion, img: np.ndarray, book: np.ndarray) -> StripRegion:
    """Торцы полосы — по её бумаге, а не срезом по строкам страниц: в ``END_ZONE_MM`` от конца каждой серии (и за ним
    на столько же) в столбцах полосы область — светлая бумага (порог Оцу по месту), связанная с телом полосы.

    Args:
        region: Область.
        img: Кадр масштаба сегментации, BGR.
        book: Грубая маска книги.

    Returns:
        Та же область с торцами по бумаге (маска меняется на месте — это её сборка).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    zone = int(END_ZONE_MM * SEG_PX_PER_MM)
    for a, b in region.runs:
        for top in (True, False):
            y_end = a if top else b - 1
            inner = slice(y_end, min(y_end + zone, b)) if top else slice(max(y_end - zone + 1, a), y_end + 1)
            cols = np.flatnonzero(region.mask[inner].any(axis=0))
            if len(cols) < 3:
                continue
            x0, x1 = int(cols.min()), int(cols.max()) + 1
            y0, y1 = (
                (max(y_end - zone, 0), min(y_end + zone, h)) if top else (max(y_end - zone, 0), min(y_end + zone, h))
            )
            patch = gray[y0:y1, x0:x1]
            thr, _ = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bright = (patch > thr) & book[y0:y1, x0:x1]
            # Столбцы — как у области в тех строках, где она есть; за концом серии — как у её крайней строки.
            allowed = np.zeros_like(bright)
            edge_row = region.mask[y_end, x0:x1]
            for i, y in enumerate(range(y0, y1)):
                inside = a <= y < b
                allowed[i] = region.mask[y, x0:x1] if inside and region.mask[y, x0:x1].any() else edge_row
            lab, n = ndi.label(bright & allowed, structure=np.ones((3, 3), bool))
            # Связная с телом полосы — компонента, накрывающая больше всего строк внутри серии.
            body_rows = slice(y_end - y0, y1 - y0) if top else slice(0, y_end - y0 + 1)
            ids, counts = np.unique(lab[body_rows][lab[body_rows] > 0], return_counts=True)
            if not len(ids):
                continue
            keep = lab == ids[np.argmax(counts)]
            region.mask[y0:y1, x0:x1] = np.where(allowed, keep, region.mask[y0:y1, x0:x1])
    return region


def extend_along_spine(region: StripRegion, spine: Line, ext_mm: float = EXTENSION_MM) -> StripRegion:
    """Продолжение полосы вдоль линии корешка на ``ext_mm`` за верхний конец первой серии и нижний конец последней:
    столбцы — как в крайней строке полосы, сдвинутые по наклону корешка.

    Args:
        region: Область.
        spine: Линия корешка.
        ext_mm: Дальность продолжения, мм.

    Returns:
        Та же область с маской ``extension`` (её сборка).
    """
    h, w = region.mask.shape
    ext = np.zeros((h, w), bool)
    reach = int(round(ext_mm * SEG_PX_PER_MM))
    rows = np.flatnonzero(region.mask.any(axis=1))
    if not len(rows):
        region.extension = ext
        return region
    near = max(1, int(round(EXTENSION_SPAN_MM * SEG_PX_PER_MM)))
    for y_end, step in ((int(rows[0]), -1), (int(rows[-1]), 1)):
        # Ширина — по строкам полосы в ``EXTENSION_SPAN_MM`` от конца (крайняя строка после ``shape_ends`` — кончик торца).
        band_rows = slice(y_end, y_end + near) if step < 0 else slice(max(y_end - near + 1, 0), y_end + 1)
        span = np.flatnonzero(region.mask[band_rows].any(axis=0))
        cols = np.arange(span.min(), span.max() + 1) if len(span) else np.zeros(0, int)
        for k in range(1, reach + 1):
            y = y_end + step * k
            if not 0 <= y < h:
                break
            shift = int(round(spine.x_at(y) - spine.x_at(y_end)))
            c = cols + shift
            c = c[(c >= 0) & (c < w)]
            ext[y, c] = True
    region.extension = ext & ~region.mask
    return region


def fit_masks(pages: list[PageQuad], region: StripRegion | None) -> list[np.ndarray]:
    """Маски страниц для подгонки: без полосы и её продолжения, наибольшая связная часть (тонкие мостики масок SAM 3
    над и под концом полосы отрезаются)."""
    if region is None:
        return [p.mask for p in pages]
    cut = region.mask | (region.extension if region.extension is not None else False)
    return [largest_component(p.mask & ~cut) for p in pages]


def page_edge_y(pages: list[PageQuad], region: StripRegion, run: tuple[int, int], top: bool) -> float:
    """Верх (низ) страниц у полосы: медиана крайней строки масок страниц в столбцах ``PAGE_EDGE_FROM_MM … TO_MM`` от
    краёв полосы на конце серии ``run``, px сегментации (NaN — столбцов нет)."""
    y_end = run[0] if top else run[1] - 1
    xs = np.flatnonzero(region.mask[y_end])
    if not len(xs):
        return float("nan")
    a, b = int(PAGE_EDGE_FROM_MM * SEG_PX_PER_MM), int(PAGE_EDGE_TO_MM * SEG_PX_PER_MM)
    vals = []
    for page, cols in ((pages[0], np.arange(xs[0] - b, xs[0] - a)), (pages[1], np.arange(xs[-1] + a, xs[-1] + b))):
        cols = cols[(cols >= 0) & (cols < page.mask.shape[1])]
        for c in cols:
            ys = np.flatnonzero(page.mask[:, c])
            if len(ys):
                vals.append(ys[0] if top else ys[-1])
    return float(np.median(vals)) if vals else float("nan")


def snap_ends(region: StripRegion, pages: list[PageQuad], paper: np.ndarray) -> StripRegion:
    """Доводка концов полосы до края листа при обтекании: конец серии ближе ``END_SNAP_MM`` к верху (низу) страниц у
    полосы, а в зазоре над (под) концом в столбцах полосы — бумага (перемычка страниц в ``END_BRIDGE_SHARE`` столбцов).

    Полоса продлевается в зазор и дальше по бумаге до её края в тех же столбцах (ширина — как у конца серии); строки
    области до края листа отмечаются, верх (низ) у корешка подгоняется как с полосой.

    Args:
        region: Область.
        pages: Страницы кадра (до исправления).
        paper: Бумага кадра.

    Returns:
        Та же область с дополненной маской и отметками (``snapped``, ``touch_*``).
    """
    reach = END_SNAP_MM * SEG_PX_PER_MM
    for top in (True, False):
        run = region.runs[0] if top else region.runs[-1]
        y_end = run[0] if top else run[1] - 1
        edge_y = page_edge_y(pages, region, run, top)
        if not np.isfinite(edge_y):
            continue
        gap = (y_end - edge_y) if top else (edge_y - y_end)
        cols = np.flatnonzero(region.mask[y_end])
        if gap <= END_TOUCH_MM * SEG_PX_PER_MM:  # ``END_TOUCH_MM`` = ``EXTENSION_MM``: продолжение дойдёт до края
            if top:
                region.touch_top = True
            else:
                region.touch_bottom = True
        if gap < 1 or gap > reach or not len(cols):
            continue
        y0, y1 = (int(edge_y), y_end) if top else (y_end + 1, int(edge_y) + 1)
        # Перемычка — бумага в зазоре в столбцах полосы (маски страниц v2 и так забирают туда бумагу полосы).
        bridge = paper[y0:y1][:, cols].any(axis=0).mean()
        if bridge < END_BRIDGE_SHARE:
            continue
        # Перемычка: полоса — в зазор и до края бумаги в своих столбцах.
        step = -1 if top else 1
        y = y_end + step
        while 0 <= y < paper.shape[0] and paper[y, cols].mean() >= 0.5:
            region.mask[y, cols] = True
            y += step
        region.snapped.append("top" if top else "bottom")
    return region


@dataclass(frozen=True)
class CorrectedPages:
    """Страницы после правки карты экземпляров SAM 3 по уточнённым краям.

    Поля:
        pages: Страницы слева направо (маски без полосы, трапеции).
        inst: Исправленная карта экземпляров.
        divider: Разделитель ``AUTO`` по новым страницам.
        moved_mm2: Бумага, перешедшая от страницы к странице или из страниц в полосу, мм².
    """

    pages: list[PageQuad]
    inst: np.ndarray
    divider: Line
    moved_mm2: float


def page_labels(inst: np.ndarray, spine: Line, rows: np.ndarray) -> tuple[int, int]:
    """Метки экземпляров SAM 3 левой и правой страницы: самые частые в столбцах ±15 мм от разделителя по ``rows``."""
    out = []
    for off in (-15.0, 15.0):
        xs = np.round(spine.x_at(rows) + off * SEG_PX_PER_MM).astype(int)
        ok = (xs >= 0) & (xs < inst.shape[1])
        v = inst[rows[ok], xs[ok]]
        v = v[v > 0]
        out.append(int(np.bincount(v).argmax()) if len(v) else -1)
    return out[0], out[1]


def correct_pages(sf: StripFrame, band: SpineBand, left: np.ndarray, right: np.ndarray,region: StripRegion | None, paper: np.ndarray, hulls: np.ndarray) -> CorrectedPages | None:  # fmt: skip
    """Правка карты экземпляров SAM 3 по уточнённым краям и страницы заново.

    В строках обеих страниц, в ``CORRECT_HALF_MM`` от разделителя: метка левой страницы правее её края снимается,
    бумага левее края — метка левой; зеркально — правой; область полосы — без метки и вне бумаги страниц. Страницы —
    ``page_pose.page_quads`` по бумаге без полосы (бумага раздаётся ближайшему экземпляру, ``glue.split_pages``).

    Args:
        sf: Кадр стенда.
        band: Лента уточнения.
        left: Правый край левой страницы по строкам ленты.
        right: Левый край правой страницы.
        region: Полоса или None.
        paper: Бумага кадра.
        hulls: Оболочки своей печати.

    Returns:
        Исправленные страницы или None (страниц с трапецией не две).
    """
    inst = sf.inst.astype(np.int32).copy()
    shape = inst.shape
    spine = sf.spine0
    rows = np.flatnonzero(page_rows(sf.frame.pages))
    lab_l, lab_r = page_labels(inst, spine, rows)
    if lab_l <= 0 or lab_r <= 0 or lab_l == lab_r:
        return None
    xl, xr = seg_edges(band, left, shape), seg_edges(band, right, shape)
    h, w = shape
    cols = np.arange(w)[None, :]
    sx = spine.x_at(np.arange(h, dtype=np.float64))[:, None]
    zone = (np.abs(cols - sx) <= CORRECT_HALF_MM * SEG_PX_PER_MM) & page_rows(sf.frame.pages)[:, None]
    ok = np.isfinite(xl)[:, None] & np.isfinite(xr)[:, None] & zone
    left_side = ok & (cols < np.nan_to_num(xl, nan=-1)[:, None])
    right_side = ok & (cols >= np.nan_to_num(xr, nan=w + 1)[:, None])
    between = ok & ~left_side & ~right_side
    inst[(inst == lab_l) & ok & ~left_side] = 0
    inst[(inst == lab_r) & ok & ~right_side] = 0
    inst[left_side & paper] = lab_l
    inst[right_side & paper] = lab_r
    inst[between] = 0
    strip = region.mask if region is not None else np.zeros(shape, bool)
    inst[strip] = 0
    gray = cv2.cvtColor(sf.frame.img, cv2.COLOR_BGR2GRAY)
    pages = [p for p in page_quads(paper & ~strip, sf.frame.book, hulls, inst, gray, sf.paint) if p.quad is not None]
    if len(pages) != 2:
        return None
    # Страница — её наибольшая связная часть: оторванные кусочки у края полосы (ПХ 1975/12 IMG_0082 — 0.7 мм² в 2–3 мм
    # за краем страницы у верха) сдвигали край страницы у полосы, и верх продолжался за данными касательной — скачок.
    pages = [replace(p, mask=largest_component(p.mask)) for p in pages]
    old = sf.frame.pages
    moved = (old[0].mask & ~pages[0].mask) | (old[1].mask & ~pages[1].mask)
    divider = spine_candidates(pages)[SpineMethod.AUTO]
    return CorrectedPages(pages, inst, divider, float(moved.sum() / SEG_PX_PER_MM**2))


def region_spine(region: StripRegion | None, divider: Line, pages: list[PageQuad]) -> tuple[Line, str]:
    """Линия корешка по исправленным страницам и полосе.

    - Полосы нет — разделитель по исправленным страницам.
    - Полоса почти во всю высоту (строк без неё меньше ``JUNCTION_MIN_SHARE`` пролёта обеих страниц) — прямая Хубера по
      серединам строк полосы.
    - Полоса на части длины — прямая Хубера по стыку страниц в строках без полосы (середина между концом левой и
      началом правой), а в строках с полосой — по тому её краю, что ближе к продолжению стыка: сгиб продолжает стык, а
      середина расширяющейся в одну сторону полосы уводила корешок в страницу (ПХ 1975/10 IMG_0081: полоса в нижних
      138 мм, правый край 205 → 213 мм, левый 201.5; по середине корешок у верха — на 3.6 мм в левой странице).

    Args:
        region: Полоса или None.
        divider: Разделитель по исправленным страницам.
        pages: Исправленные страницы.

    Returns:
        ``(линия корешка, откуда она — подпись)``.
    """
    if region is None:
        return divider, "разделитель по исправленным страницам"
    rows_both = np.flatnonzero(page_rows(pages))
    has = region.mask.any(axis=1)
    free = rows_both[~has[rows_both]]
    strip_rows = rows_both[has[rows_both]]
    if len(free) < JUNCTION_MIN_SHARE * max(len(rows_both), 1):
        mids = np.array([np.flatnonzero(region.mask[y]).mean() for y in strip_rows])
        line = fit_line(mids, strip_rows.astype(np.float64), divider.y_top, divider.y_bot)
        return (line, "середина полосы") if line is not None else (divider, "разделитель")
    lm, rm = pages[0].mask, pages[1].mask
    jx = np.array([(np.flatnonzero(lm[y]).max() + np.flatnonzero(rm[y]).min()) / 2 for y in free])
    junction = fit_line(jx, free.astype(np.float64), divider.y_top, divider.y_bot)
    if junction is None:
        return divider, "разделитель (стык не построился)"
    xs, ys = list(jx), list(free.astype(np.float64))
    for y in strip_rows:
        cols = np.flatnonzero(region.mask[y])
        guess = junction.x_at(y)
        xs.append(float(cols.min() if abs(cols.min() - guess) <= abs(cols.max() - guess) else cols.max()))
        ys.append(float(y))
    line = fit_line(np.array(xs), np.array(ys), divider.y_top, divider.y_bot)
    return (line, "стык страниц и продолжающий его край полосы") if line is not None else (junction, "стык страниц")


def region_frames(
    pages: list[PageQuad],
    spine: Line,
    region: StripRegion | None,
    paint: np.ndarray,
    core_zone: np.ndarray,
    texts: list[np.ndarray],
    guard_mm: float = REGION_GUARD_MM,
    zone_at_strip: bool = True,
) -> list[PageFrame]:
    """Подгоночные области страниц с полосой-областью.

    Точки контура в полосе, расширенной на ``guard_mm`` по горизонтали, негодны (вертикальная кромка маски страницы у
    полосы — вплотную к ней). Страницы исправлены по краю полосы, их сторона у корешка — край полосы, поэтому зона
    сплайна у корешка ``SPINE_ZONE_MM`` (узлы гуще, штраф слабее) отсчитывается от края полосы: верх и низ повторяют
    изгиб страницы до края полосы (ПХ 1975/04 IMG_0082, низ левой страницы поднимается на 2 мм в последних 6–8 мм),
    отвесные точки и выемки у полосы негодны, монотонный загиб остаётся (``strip_side_rejects``). Подгонка — по странице
    без полосы и её продолжения вдоль корешка (``fit_masks``: мостики масок SAM над концом полосы отрезаны, ПХ 1975/12
    IMG_0066); через полосу и продолжение — экстраполяция своей кривой параболой до линии корешка (``extrapolate``,
    наклон и кривизна ограничены). Сторона без полосы — касательной до линии корешка. Прежний вариант (``guard_mm`` = ``STRIP_GUARD_MM``,
    ``zone_at_strip`` = False: сторона, к которой подходит полоса, — без зоны) — для сравнения.

    Args:
        pages: Страницы слева направо.
        spine: Линия корешка.
        region: Полоса или None.
        paint: Зона закраса пальцев.
        core_zone: Ядро пальца с припуском.
        texts: Вершины текста по страницам.
        guard_mm: Защита у полосы, мм.
        zone_at_strip: Зона сплайна у корешка и на стороне, к которой подходит полоса.

    Returns:
        Области левой и правой страницы.
    """
    guard = max(1, int(round(guard_mm * SEG_PX_PER_MM)))
    cut = np.zeros(paint.shape, bool)
    if region is not None:
        cut = region.mask | (region.extension if region.extension is not None and zone_at_strip else False)
    zone = cv2.dilate(cut.astype(np.uint8), np.ones((1, 2 * guard + 1), np.uint8)) > 0
    masks = fit_masks(pages, region) if zone_at_strip else [p.mask for p in pages]
    close = max(3, int(round(NOTCH_CLOSE_MM * SEG_PX_PER_MM)) | 1)
    t_ext = np.linspace(-EXTEND, 1 + EXTEND, CURVE_SAMPLES)
    touch = {Side.TOP: region is not None and region.touch_top, Side.BOTTOM: region is not None and region.touch_bottom}
    bodies: list[dict] = []
    fars = []
    for page, page_mask, spine_right, text in (
        (pages[0], masks[0], True, texts[0]),
        (pages[1], masks[1], False, texts[1]),
    ):
        ys = page.quad[:, 1]
        fold = line_points(spine, float(ys.min()), float(ys.max()))
        mask = page_mask
        if zone_at_strip and region is not None:
            # Выемки края — закрытием по горизонтали, но не в полосе и не за ней.
            closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, close), np.uint8)) > 0
            mask = mask | (closed & ~zone)

        def fit(zone_mm, extra):  # noqa: E306 — замыкание над параметрами страницы (два вызова ниже)
            return fit_boundary_adaptive(mask, page.quad, spine_right, SEG_PX_PER_MM, paint | zone | extra, text,
                                         fold_points=fold, edge_exclude=core_zone | zone | extra,
                                         spine_zone_mm=zone_mm)[0]  # fmt: skip

        stiff = fit(None, np.zeros(mask.shape, bool))
        flexible = stiff
        if zone_at_strip and region is not None and any(touch.values()):
            # У полосы: отвесные точки (срезанный угол, кромка у полосы) и выемки (край уходит внутрь и возвращается —
            # клочок обложки, ПХ 1975/12 IMG_0084) негодны; монотонный загиб к полосе остаётся (04/IMG_0066, 0082).
            bad = strip_side_rejects(mask, stiff, page_left=spine_right, touch=touch)
            flexible = fit(SPINE_ZONE_MM, bad)
        elif zone_at_strip or region is None:
            flexible = fit(SPINE_ZONE_MM, np.zeros(mask.shape, bool))
        page_bodies = {}
        for side in (Side.TOP, Side.BOTTOM):
            stiff_side = touch[side] and not zone_at_strip
            b = stiff if stiff_side else flexible
            if zone_at_strip and touch[side]:
                body = body_to_edge(b.curves[side].points(t_ext), page_mask, outer_of(b, side, spine_right),
                                    spine_right)  # fmt: skip
                page_bodies[side] = ("extrapolate", body)
            else:
                pts = b.curves[side].points(t_ext)
                page_bodies[side] = ("cut", cut_at(pts, spine, outer_of(b, side, spine_right), spine_right))
        bodies.append(page_bodies)
        fars.append(flexible.left if spine_right else flexible.right)
    frames = []
    for k in (0, 1):
        parts = []
        for side in (Side.TOP, Side.BOTTOM):
            kind, line = bodies[k][side]
            if kind == "cut":
                parts.append(line)
                continue
            # Через полосу и её продолжение — экстраполяция своей кривой до линии корешка.
            parts.append(extrapolate(line, spine, page_left=k == 0))
        top, bottom = parts
        frames.append(PageFrame(top, bottom, fars[k], np.array([top[-1], bottom[-1]]), np.zeros((0, 2))))
    return frames


def edge_profile(mask: np.ndarray, xs: np.ndarray, top: bool) -> np.ndarray:
    """Край страницы по столбцам ``xs``: верхняя (``top``) или нижняя строка маски, NaN — столбец пуст."""
    out = np.full(len(xs), np.nan)
    for i, x in enumerate(xs):
        ys = np.flatnonzero(mask[:, x])
        if len(ys):
            out[i] = ys[0] if top else ys[-1]
    return out


def strip_side_rejects(mask: np.ndarray, stiff, page_left: bool, touch: dict) -> np.ndarray:
    """Негодные точки верха (низа) страницы у полосы, в ``SPINE_ZONE_MM`` от её края, на сторонах ``touch``.

    - Отвесные столбцы: край меняется круче ``STEEP_SLOPE`` на 1 мм (срезанный наискось угол страницы у полосы,
      вертикальная кромка маски).
    - Выемка: глубина края внутрь от жёсткой кривой (без зоны у корешка) наибольшая не у полосы, а у края полосы
      мельче наибольшей больше чем на ``NOTCH_RETURN_MM`` (край ушёл внутрь и вернулся: клочок обложки, 12/IMG_0084);
      негодны столбцы выемки глубже ``NOTCH_FLOOR_MM``. Монотонный загиб к полосе (04/IMG_0066, 0082) остаётся.

    Args:
        mask: Маска страницы.
        stiff: Рамка без зоны у корешка (``PageBoundary``).
        page_left: Левая страница (полоса справа).
        touch: Стороны, к которым подходит полоса.

    Returns:
        Маска негодных пикселей (точки контура в ней выпадают из подгонки).
    """
    h, w = mask.shape
    out = np.zeros((h, w), bool)
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(cols):
        return out
    edge = cols.max() if page_left else cols.min()
    reach = int(SPINE_ZONE_MM * SEG_PX_PER_MM)
    # Столбцы от края у полосы внутрь страницы (порядок — от полосы).
    xs = np.arange(edge, max(edge - reach, 0) - 1, -1) if page_left else np.arange(edge, min(edge + reach, w - 1) + 1)
    mm = max(1, int(round(SEG_PX_PER_MM)))
    t = np.linspace(-EXTEND, 1 + EXTEND, CURVE_SAMPLES)
    for side in (Side.TOP, Side.BOTTOM):
        if not touch[side]:
            continue
        top = side is Side.TOP
        y = edge_profile(mask, xs, top)
        rows = slice(0, h // 2) if top else slice(h // 2, h)
        # Отвесные: перепад края за 1 мм круче STEEP_SLOPE.
        dy = np.abs(y - np.roll(y, -mm))
        dy[-mm:] = 0
        steep = np.nan_to_num(dy, nan=np.inf) > STEEP_SLOPE * mm
        steep |= np.r_[False, steep[:-1]]
        # Глубина внутрь от жёсткой кривой (плюс — внутрь страницы).
        pts = stiff.curves[side].points(t)
        order = np.argsort(pts[:, 0])
        base = np.interp(xs, pts[order, 0], pts[order, 1])
        depth = (y - base) if top else (base - y)
        depth = np.where(steep, np.nan, depth) / SEG_PX_PER_MM
        bad_cols = steep.copy()
        if np.isfinite(depth).any():
            k = int(np.nanargmax(depth))
            near = depth[: max(k, 1)]
            near_val = float(np.nanmin(near[np.isfinite(near)][:3])) if np.isfinite(near).any() else depth[k]
            # Выемка: наибольшая глубина не у самой полосы, у полосы край заметно мельче.
            if depth[k] > NOTCH_FLOOR_MM and k > mm and depth[k] - near_val > NOTCH_RETURN_MM:
                bad_cols |= np.nan_to_num(depth, nan=0.0) > NOTCH_FLOOR_MM
        for x in xs[bad_cols]:
            out[rows, x] = True
    return out


def outer_of(b, side: Side, page_left: bool) -> np.ndarray:
    """Внешний угол верха или низа рамки ``b`` (``PageBoundary``)."""
    line = b.top if side is Side.TOP else b.bottom
    return line[0] if page_left else line[-1]


def body_to_edge(pts: np.ndarray, page: np.ndarray, outer: np.ndarray, page_left: bool) -> np.ndarray:
    """Кривая верха или низа от внешнего угла до края страницы у полосы: точки идут, пока не выйдут за крайний столбец
    маски страницы в строках ``±EDGE_ROWS_MM`` около самой точки (крайний столбец всей маски брал бы чужие строки)."""
    pts = pts[np.argsort(pts[:, 0])]
    if not page_left:
        pts = pts[::-1]
    h, w = page.shape
    half = max(1, int(round(EDGE_ROWS_MM * SEG_PX_PER_MM)))
    sign = 1.0 if page_left else -1.0
    start = int(np.argmin(np.abs(pts[:, 0] - outer[0])))
    body = [outer]
    for x, y in pts[start:]:
        y0, y1 = max(int(y) - half, 0), min(int(y) + half + 1, h)
        cols = np.flatnonzero(page[y0:y1].any(axis=0)) if y1 > y0 else np.zeros(0, int)
        if not len(cols):
            break
        edge = cols.max() if page_left else cols.min()
        if sign * (x - edge) > 0:
            break
        body.append(np.array([x, y]))
    body = np.array(body, np.float64)
    return body if len(body) >= 2 else np.vstack([outer, outer])


def extrapolate(body: np.ndarray, spine: Line, page_left: bool) -> np.ndarray:
    """Кривая верха или низа от края страницы до линии корешка — экстраполяция параболой.

    Парабола ``y(x)`` — по точкам последних ``EXTRAP_FIT_MM`` кривой; наклон в конце — не круче ``EXTRAP_MAX_SLOPE``,
    кривизна — такая, чтобы её вклад до линии корешка был не больше ``EXTRAP_CURVE_MM``.

    Args:
        body: Кривая от внешнего угла до края страницы у полосы (N, 2).
        spine: Линия корешка.
        page_left: Левая страница (корешок справа).

    Returns:
        Кривая с продолжением, последняя точка — на линии корешка.
    """
    end = body[-1]
    span = EXTRAP_FIT_MM * SEG_PX_PER_MM
    near = body[np.abs(body[:, 0] - end[0]) <= span]
    if len(near) >= 3 and np.ptp(near[:, 0]) > 1:
        a2, a1, _ = np.polyfit(near[:, 0] - end[0], near[:, 1] - end[1], 2)
    else:
        a2, a1 = 0.0, 0.0
    slope = float(np.clip(a1, -EXTRAP_MAX_SLOPE, EXTRAP_MAX_SLOPE))
    dx_max = abs(float(spine.x_at(end[1])) - end[0])
    if dx_max < 1:
        return np.vstack([body, [[float(spine.x_at(end[1])), end[1]]]])
    a2 = float(np.clip(a2, -EXTRAP_CURVE_MM * SEG_PX_PER_MM / dx_max**2, EXTRAP_CURVE_MM * SEG_PX_PER_MM / dx_max**2))
    sign = 1.0 if page_left else -1.0
    dxs = sign * np.linspace(0.0, dx_max * 1.2, 40)[1:]
    pts = np.stack([end[0] + dxs, end[1] + slope * dxs + a2 * dxs**2], axis=1)
    # Обрезка на линии корешка.
    side = (spine.x_at(pts[:, 1]) - pts[:, 0]) * sign
    inside = np.flatnonzero(side <= 0)
    if not len(inside):
        return np.vstack([body, pts])
    k = int(inside[0])
    prev = pts[k - 1] if k > 0 else end
    sa = (spine.x_at(prev[1]) - prev[0]) * sign
    sb = side[k]
    hit = prev + (pts[k] - prev) * (sa / (sa - sb)) if sa != sb else pts[k]
    return np.vstack([body, pts[:k], hit])


def segment_hit(p: np.ndarray, q: np.ndarray, spine: Line) -> np.ndarray:
    """Точка отрезка ``p → q`` на линии корешка (не пересекает — проекция середины на прямую)."""
    fp, fq = spine.x_at(p[1]) - p[0], spine.x_at(q[1]) - q[0]
    if fp != fq and fp * fq <= 0:
        return p + (q - p) * (fp / (fp - fq))
    mid = (p + q) / 2
    return np.array([spine.x_at(mid[1]), mid[1]])
