"""Выравнивание внешних боковых сторон плоских страниц: робастная гладкая кривая по краю маски (скользящая медиана + гаусс), подтяжка к переходу «жёлтая бумага → не бумага», срез выбросов и заливка ям."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM, PagePose, PageQuad

# Полоса у верхнего и нижнего концов стороны, где ямы не заливаются, мм: там скругления углов листа.
CORNER_MM = 5.0
# Робастная кривая стороны: скользящая медиана по окну MEDIAN_MM (ямы и выступы короче ~половины окна её не сдвигают),
# затем отсев точек дальше OUTLIER_MM от неё и гауссово сглаживание оставшихся с σ = SMOOTH_MM. Форма стороны крупнее
# окна (выгиб листа, волна у подшивки) сохраняется — парабола её не держала.
MEDIAN_MM = 25.0
SMOOTH_MM = 4.0
OUTLIER_MM = 1.5
# Подтяжка к настоящему краю листа. Пределы поиска края от кривой SAM 3 (наружу — плюс), мм: внутрь до 3, наружу до 8 —
# тёмная полоса кромки страницы в тени у РИСК 1991/05 IMG_0538 шириной ~7 мм, SAM 3 колеблется внутри неё.
SHIFT_IN_MM, SHIFT_OUT_MM = 3.0, 8.0
# Окно по обе стороны точки при оценке перепада желтизны (b* Lab, уровни OpenCV), мм.
STEP_WINDOW_MM = 1.2
# Край в строке — место наибольшего падения желтизны наружу, если оно не меньше STEP_MIN (бумага +4…+6, белая стопка −2).
STEP_MIN = 3.0
# Подтяжка включается, только если SAM 3 не уверен в стороне — разброс края (MAD) вокруг скользящей медианы с окном
# SNAP_GATE_MM не меньше SNAP_MIN_MAD px — и край найден хотя бы в доле строк SNAP_MIN_SHARE. Замер по 7 кадрам
# (окно 60 мм): у уверенных сторон MAD 0–2 px (там дальше по b* — торец или обложка, подтягиваться к ним нельзя:
# IMG_0240, IMG_0250 слева), у колеблющейся на загнутой кромке — 5 (РИСК 1991/05 IMG_0538 справа). Край у предела
# поиска — не край (упёрлись в окно).
SNAP_GATE_MM = 60.0
SNAP_MIN_MAD = 4.0
SNAP_MIN_SHARE = 0.5
# Срез: бумага снаружи кривой дальше этого, мм. Заливка: ямы не глубже этого, мм.
CUT_MARGIN_MM = 1.0
MAX_FILL_MM = 10.0
# Дыры, которые замкнула сама заливка ям: заливка идёт по строкам до самого внешнего пикселя страницы, и язычок бумаги
# под кончиком пальца оставлял за собой замкнутую полулуну, а закрас до рамки красил её светлым (МТС 1991/02 IMG_0634,
# 0641, 0666, 0669). Такая дыра доливается, если она светлая, как бумага вокруг: светлота кольца бумаги
# CLOSED_RING_MM вокруг минус светлота дыры (медианы, L OpenCV 0..255) не больше CLOSED_MAX_DROP, и площадь не больше
# CLOSED_MAX_MM2. Пак-6 (2480 кадров): замкнутых заливкой дыр 4824 (у пальцев 1134 в 395 кадрах), это бумага —
# ΔL 99 % = 11, |Δb*| 99 % = 1.6, площадь до 621 мм²; замкнутые ещё до заливки (полосы у корешка, печать) не трогаются.
CLOSED_RING_MM = 1.0
CLOSED_MAX_DROP = 25.0
CLOSED_MAX_MM2 = 1000.0
# Кромка листа за светлым полем: SAM 3 берёт цветную плашку, а светлое поле страницы вокруг неё — рваными языками
# (МТС 1990/04 IMG_1113: оранжевая обложка, кремовое поле 15 мм не взято, страница из-за этого «приподнята»). Если край
# маски стороны рваный (MAD как у подтяжки, SNAP_MIN_MAD), в каждой строке ищется первый узкий провал светлоты наружу от
# края маски в пределах EDGE_REACH_MM — тень кромки листа: темнее на EDGE_DIP уровней L (0..255) соседей в окне
# EDGE_DIP_MM с обеих сторон. По провалам — прямая RANSAC (допуск EDGE_TOL_MM); если на ней не меньше EDGE_MIN_SHARE
# строк стороны, поле между маской и прямой заливается (книга, не фон). Страницы со сходом больше EDGE_MAX_KEYSTONE
# (по-настоящему поднятые) не трогаются. IMG_1113: слева прямая x = 327 + 0.006·y по 83 % строк (край листа по замеру
# 325–334), сверху — 83 %; снизу провалов нет (маска у края).
EDGE_REACH_MM = 25.0
EDGE_DIP = 6.0
EDGE_DIP_MM = 1.5
EDGE_TOL_MM = 1.5
EDGE_MIN_SHARE = 0.5
EDGE_MAX_KEYSTONE = 12.0
# Только цветные страницы — медианная насыщенность страницы (a*, b* OpenCV от 128) не меньше EDGE_MIN_PAGE_CHROMA: у
# обычной страницы с текстом провал за краем маски — это торец стопки (полосы листов), и заливка его брала (пак-6:
# 89 кадров без этого условия, почти все — торец).
EDGE_MIN_PAGE_CHROMA = 25.0


class Side(Enum):
    """Внешняя боковая сторона страницы."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class SideCurve:
    """Выровненная сторона: абсцисса края по строкам (масштаб сегментации).

    Поля:
        side: Какая сторона.
        ys: Строки, где кривая задана (без полос у углов).
        xs: Абсцисса края на этих строках (по настоящему краю, если была подтяжка).
        y0: Первая строка стороны (по трапеции).
        y1: Последняя строка стороны.
        shift: Медианный сдвиг подтяжки к настоящему краю, px (наружу — плюс; 0 — без подтяжки).
    """

    side: Side
    ys: np.ndarray
    xs: np.ndarray
    y0: int
    y1: int
    shift: int

    def at(self, rows: np.ndarray) -> np.ndarray:
        """Абсцисса кривой на строках ``rows``; за её концами — значения на концах."""
        return np.interp(rows, self.ys, self.xs)


@dataclass(frozen=True)
class SideFit:
    """Итог выравнивания сторон.

    Поля:
        paper: Маска бумаги после выравнивания.
        added: Залитые ямы.
        cut: Срезанные выбросы.
        curves: Выровненные стороны.
    """

    paper: np.ndarray
    added: np.ndarray
    cut: np.ndarray
    curves: list[SideCurve]


def mm(value: float) -> int:
    """Миллиметры → px масштаба сегментации (не меньше 1)."""
    return max(1, int(round(value * SEG_PX_PER_MM)))


def outer_sides(index: int, count: int) -> list[Side]:
    """Внешние боковые стороны страницы по её месту в кадре (страницы слева направо).

    Args:
        index: Номер страницы.
        count: Число страниц.

    Returns:
        Стороны: у одиночной обе, у левой — левая, у правой — правая, у средних — ни одной.
    """
    if count == 1:
        return [Side.LEFT, Side.RIGHT]
    if index == 0:
        return [Side.LEFT]
    if index == count - 1:
        return [Side.RIGHT]
    return []


def side_rows(quad: np.ndarray, side: Side) -> tuple[int, int]:
    """Строки стороны по вершинам трапеции: две левые или две правые вершины.

    Args:
        quad: Вершины (4, 2).
        side: Сторона.

    Returns:
        ``(первая строка, последняя строка)``.
    """
    order = np.argsort(quad[:, 0])
    pair = quad[order[:2]] if side is Side.LEFT else quad[order[2:]]
    return int(np.floor(pair[:, 1].min())), int(np.ceil(pair[:, 1].max()))


def edge_points(mask: np.ndarray, side: Side, y0: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
    """Крайние пиксели маски со стороны ``side`` по строкам ``[y0, y1]``.

    Args:
        mask: Маска страницы.
        side: Сторона.
        y0: Первая строка.
        y1: Последняя строка.

    Returns:
        ``(строки, абсциссы края)``.
    """
    y0, y1 = max(0, y0), min(mask.shape[0] - 1, y1)
    band = mask[y0 : y1 + 1]
    has = band.any(axis=1)
    if side is Side.LEFT:
        xs = np.argmax(band, axis=1)
    else:
        xs = band.shape[1] - 1 - np.argmax(band[:, ::-1], axis=1)
    ys = np.arange(y0, y1 + 1)
    return ys[has], xs[has].astype(np.float64)


def robust_curve(ys: np.ndarray, xs: np.ndarray) -> np.ndarray | None:
    """Гладкая кривая края, устойчивая к ямам и выступам (см. ``MEDIAN_MM``, ``SMOOTH_MM``, ``OUTLIER_MM``).

    Пропуски строк заполняются линейно, затем два прохода: скользящая медиана → отсев точек дальше ``OUTLIER_MM`` →
    гауссово сглаживание оставшихся (нормированная свёртка, выброшенные точки не тянут кривую).

    Args:
        ys: Строки (по возрастанию).
        xs: Абсциссы края.

    Returns:
        Абсциссы кривой на строках ``ys`` или None (мало точек).
    """
    if len(ys) < 20:
        return None
    rows = np.arange(ys[0], ys[-1] + 1)
    full = np.interp(rows, ys, xs)
    curve = ndimage.median_filter(full, size=mm(MEDIAN_MM), mode="nearest")
    sigma = mm(SMOOTH_MM)
    for _ in range(2):
        keep = (np.abs(full - curve) <= mm(OUTLIER_MM)).astype(np.float64)
        if keep.sum() < 20:
            return None
        num = ndimage.gaussian_filter1d(full * keep, sigma, mode="nearest")
        den = ndimage.gaussian_filter1d(keep, sigma, mode="nearest")
        smooth = num / np.maximum(den, 1e-6)
        # Где рядом нет принятых точек (яма длиннее ±3σ) — линейно между соседними опорными строками. Медиана здесь
        # уже втянута ямой внутрь, и кривая шла ступенькой (прямоугольная выемка у пальца, МТС 1991/02 IMG_0622).
        ok = den > 0.05
        curve = np.interp(rows, rows[ok], smooth[ok]) if ok.any() else curve
    return np.interp(ys, rows, curve)


def step_points(b: np.ndarray, curve: np.ndarray, ys: np.ndarray, side: Side) -> tuple[np.ndarray, np.ndarray]:
    """Край листа по строкам: где желтизна сильнее всего падает наружу в окне поиска вокруг кривой.

    Args:
        b: Канал b* Lab (float, сглаженный).
        curve: Абсциссы кривой стороны на строках ``ys``.
        ys: Строки.
        side: Сторона.

    Returns:
        ``(абсциссы края, перепад)`` по строкам.
    """
    sign = -1 if side is Side.LEFT else 1
    shifts = np.arange(-mm(SHIFT_IN_MM), mm(SHIFT_OUT_MM) + 1)
    xc = np.round(curve).astype(int)
    xs = xc[:, None] + sign * shifts[None, :]
    offs = np.arange(1, mm(STEP_WINDOW_MM) + 1)
    rows = ys[:, None, None]
    inside = b[rows, np.clip(xs[..., None] - sign * offs, 0, b.shape[1] - 1)].mean(axis=2)
    outside = b[rows, np.clip(xs[..., None] + sign * offs, 0, b.shape[1] - 1)].mean(axis=2)
    drop = inside - outside
    best = np.argmax(drop, axis=1)
    value = drop[np.arange(len(ys)), best]
    # Край у предела поиска — не край.
    value[(best == 0) | (best == len(shifts) - 1)] = 0.0
    return xs[np.arange(len(ys)), best].astype(np.float64), value


def snap_curve(b: np.ndarray, curve: np.ndarray, ys: np.ndarray, xs: np.ndarray, side: Side) -> tuple[np.ndarray, int]:
    """Кривая по настоящему краю листа, если SAM 3 колеблется (см. ``SNAP_*``), иначе прежняя.

    Args:
        b: Канал b* Lab.
        curve: Кривая по краю маски на строках ``ys``.
        ys: Строки.
        xs: Абсциссы края маски.
        side: Сторона.

    Returns:
        ``(кривая на строках ys, медианный сдвиг наружу, px)``.
    """
    rows = np.arange(ys[0], ys[-1] + 1)
    full = np.interp(rows, ys, xs)
    res = full - ndimage.median_filter(full, size=mm(SNAP_GATE_MM), mode="nearest")
    if np.median(np.abs(res - np.median(res))) < SNAP_MIN_MAD:
        return curve, 0
    edge, value = step_points(b, curve, ys, side)
    ok = value >= STEP_MIN
    if ok.mean() < SNAP_MIN_SHARE:
        return curve, 0
    snapped = robust_curve(ys[ok], edge[ok])
    if snapped is None:
        return curve, 0
    snapped = np.interp(ys, ys[ok], snapped)
    sign = -1 if side is Side.LEFT else 1
    # Подтяжка только наружу: внутрь край уводят места, где за кромкой не стопка, а заливка LaMa или палец
    # (IMG_0538 внизу справа) — срезать по ним бумагу нельзя.
    snapped = curve + sign * np.maximum(sign * (snapped - curve), 0.0)
    return snapped, int(round(np.median(sign * (snapped - curve))))


def fit_side(page: np.ndarray, quad: np.ndarray, side: Side, b: np.ndarray, snap: bool) -> SideCurve | None:
    """Выровненная сторона страницы: робастная кривая по краю маски без углов и (по ``snap``) подтяжка к настоящему краю.

    Args:
        page: Маска страницы.
        quad: Трапеция страницы.
        side: Сторона.
        b: Канал b* Lab.
        snap: Подтягивать к настоящему краю листа, где SAM 3 колеблется.

    Returns:
        Сторона или None (мало точек).
    """
    y0, y1 = side_rows(quad, side)
    corner = mm(CORNER_MM)
    ys, xs = edge_points(page, side, y0 + corner, y1 - corner)
    curve = robust_curve(ys, xs)
    if curve is None:
        return None
    shift = 0
    if snap:
        curve, shift = snap_curve(b, curve, ys, xs, side)
    return SideCurve(side, ys, curve, y0, y1, shift)


def apply_side(paper: np.ndarray, page: np.ndarray, curve: SideCurve, bg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Срез выбросов и заливка ям по выровненной стороне.

    Срез — по всей стороне (выбросы бывают и у углов, IMG_0240), заливка — без полос у углов и не глубже
    ``MAX_FILL_MM``, только не фон.

    Args:
        paper: Маска бумаги кадра.
        page: Маска страницы.
        curve: Выровненная сторона.
        bg: Маска фона.

    Returns:
        ``(добавлено, срезано)``.
    """
    h, w = paper.shape
    added = np.zeros_like(paper)
    cut = np.zeros_like(paper)
    cols = np.arange(w)[None, :]
    y0, y1 = max(0, curve.y0), min(h - 1, curve.y1)
    ys = np.arange(y0, y1 + 1)
    xc = curve.at(ys)[:, None]
    if curve.side is Side.LEFT:
        outside = cols < xc - mm(CUT_MARGIN_MM)
        between = (cols >= xc) & (cols <= xc + mm(MAX_FILL_MM))
    else:
        outside = cols > xc + mm(CUT_MARGIN_MM)
        between = (cols <= xc) & (cols >= xc - mm(MAX_FILL_MM))
    cut[y0 : y1 + 1] = paper[y0 : y1 + 1] & outside
    # Заливка — от кривой до края маски страницы, по строкам, где страница есть, без углов.
    corner = mm(CORNER_MM)
    rows = page[y0 : y1 + 1].any(axis=1)
    rows &= (ys >= curve.y0 + corner) & (ys <= curve.y1 - corner)
    sub = page[y0 : y1 + 1]
    if curve.side is Side.LEFT:
        edge = np.where(rows, np.argmax(sub, axis=1), 0)[:, None]
        gap = between & (cols < edge)
    else:
        edge = np.where(rows, w - 1 - np.argmax(sub[:, ::-1], axis=1), w - 1)[:, None]
        gap = between & (cols > edge)
    gap &= rows[:, None] & ~bg[y0 : y1 + 1] & ~paper[y0 : y1 + 1]
    added[y0 : y1 + 1] = gap
    return added, cut


def regularize_sides(
    paper: np.ndarray, pages: list[PageQuad], bg: np.ndarray, img: np.ndarray, snap: bool = True
) -> SideFit:
    """Выравнивание внешних боковых сторон плоских страниц кадра.

    Приподнятые страницы не трогаются (их сторона сильно изогнута). Верх, низ и корешок не меняются.

    Args:
        paper: Маска бумаги (масштаб сегментации).
        pages: Страницы слева направо с трапециями и классом положения.
        bg: Маска фона.
        img: Кадр без пальцев BGR (масштаб сегментации).
        snap: Подтягивать сторону к настоящему краю листа, где SAM 3 колеблется.

    Returns:
        Маски и выровненные стороны.
    """
    b = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 2].astype(np.float32), (5, 5), 0)
    added = np.zeros_like(paper)
    cut = np.zeros_like(paper)
    curves: list[SideCurve] = []
    for k, page in enumerate(pages):
        if page.pose is not PagePose.FLAT or page.quad is None:
            continue
        for side in outer_sides(k, len(pages)):
            curve = fit_side(page.mask, page.quad, side, b, snap)
            if curve is None:
                continue
            a, c = apply_side(paper, page.mask, curve, bg)
            added |= a
            cut |= c
            curves.append(curve)
    added = (added | closed_holes(paper, (paper | added) & ~cut, added, bg, img)) & ~cut
    return SideFit((paper | added) & ~cut, added, cut, curves)


def closed_holes(paper: np.ndarray, new: np.ndarray, added: np.ndarray, bg: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Дыры, которые замкнула сама заливка ям и которые светлые, как бумага вокруг (см. ``CLOSED_*``).

    Args:
        paper: Бумага до выравнивания.
        new: Бумага после заливки и среза.
        added: Залитые ямы.
        bg: Маска фона.
        img: Кадр без пальцев BGR (масштаб сегментации).

    Returns:
        Маска дыр к доливке (bool).
    """
    holes = ndimage.binary_fill_holes(new) & ~new & ~ndimage.binary_fill_holes(paper) & ~bg
    if not holes.any():
        return np.zeros_like(paper)
    light = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    near_added = cv2.dilate(added.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ring_px = mm(CLOSED_RING_MM)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    max_px = CLOSED_MAX_MM2 * SEG_PX_PER_MM**2
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
    out = np.zeros_like(paper)
    for k in range(1, count):
        if stats[k, cv2.CC_STAT_AREA] > max_px:
            continue
        x, y, w, h = stats[k, :4]
        # Рамка дыры с запасом на кольцо.
        box = (slice(max(0, y - ring_px - 1), y + h + ring_px + 1), slice(max(0, x - ring_px - 1), x + w + ring_px + 1))
        hole = labels[box] == k
        if not (hole & near_added[box]).any():
            continue
        ring = (cv2.dilate(hole.astype(np.uint8), kernel) > 0) & ~hole & new[box]
        if ring.sum() < 3:
            continue
        if np.median(light[box][ring]) - np.median(light[box][hole]) <= CLOSED_MAX_DROP:
            out[box] |= hole
    return out


def ransac_line(t: np.ndarray, v: np.ndarray, tol: float, iters: int = 300) -> tuple[np.ndarray, np.ndarray] | None:
    """Прямая ``v ≈ a·t + b`` по точкам, устойчивая к выбросам (RANSAC по парам, уточнение МНК по опоре).

    Args:
        t: Координата вдоль стороны (строки или столбцы).
        v: Координата края.
        tol: Допуск опоры, px.
        iters: Число проб (генератор с фиксированным зерном — итог воспроизводим).

    Returns:
        ``((a, b), опора bool)`` или None (мало точек).
    """
    if len(t) < 20:
        return None
    rng = np.random.default_rng(0)
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(t), 2, replace=False)
        if t[i] == t[j]:
            continue
        a = (v[j] - v[i]) / (t[j] - t[i])
        inl = np.abs(v - (a * t + v[i] - a * t[i])) <= tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 2:
        return None
    return np.polyfit(t[best], v[best], 1), best


def edge_dips(light: np.ndarray, mask: np.ndarray, outward: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Первый узкий провал светлоты наружу от края маски по строкам (сторона — по знаку ``outward``).

    Args:
        light: Светлота L (0..255), float32; для верха и низа — транспонированная.
        mask: Маска страницы (так же транспонированная).
        outward: −1 — край слева (наружу влево), +1 — справа.

    Returns:
        ``(строки, абсциссы провала, число строк стороны)``.
    """
    has = mask.any(axis=1)
    if outward < 0:
        edge = np.argmax(mask, axis=1)
    else:
        edge = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    reach, w = mm(EDGE_REACH_MM), mm(EDGE_DIP_MM)
    rows, xs_found = [], []
    for row in np.nonzero(has)[0]:
        xs = edge[row] + outward * np.arange(1, reach)
        xs = xs[(xs >= w + 2) & (xs < light.shape[1] - w - 2)]
        if len(xs) < 2 * w + 4:
            continue
        prof = light[row, xs]
        for i in range(w, len(xs) - w):
            v = prof[i]
            # Провал: темнее соседей с обеих сторон на EDGE_DIP и локальный минимум.
            if v <= prof[i - w : i].max() - EDGE_DIP and v <= prof[i + 1 : i + w + 1].max() - EDGE_DIP:
                if v <= prof[max(0, i - 2) : i + 3].min():
                    rows.append(row)
                    xs_found.append(xs[i])
                    break
    return np.asarray(rows, float), np.asarray(xs_found, float), int(has.sum())


def ragged(mask: np.ndarray, outward: int) -> bool:
    """Рваный ли край маски стороны: MAD отклонения края от скользящей медианы ≥ ``SNAP_MIN_MAD`` (как у подтяжки)."""
    has = mask.any(axis=1)
    if has.sum() < 20:
        return False
    if outward < 0:
        edge = np.argmax(mask, axis=1)[has].astype(np.float64)
    else:
        edge = (mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1))[has].astype(np.float64)
    res = edge - ndimage.median_filter(edge, size=mm(SNAP_GATE_MM), mode="nearest")
    return float(np.median(np.abs(res - np.median(res)))) >= SNAP_MIN_MAD


def snap_sheet_edges(
    paper: np.ndarray, pages: list[PageQuad], bg: np.ndarray, book: np.ndarray, img: np.ndarray
) -> np.ndarray:
    """Поле страницы до кромки листа, где SAM 3 взял его рваными языками (см. ``EDGE_*``).

    Стороны — внешняя боковая (по месту страницы), верх и низ; корешок не трогается.

    Args:
        paper: Маска бумаги (масштаб сегментации).
        pages: Страницы слева направо с трапециями.
        bg: Маска фона.
        book: Грубая маска книги.
        img: Кадр без пальцев BGR (масштаб сегментации).

    Returns:
        Добавка (bool).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    light = cv2.GaussianBlur(lab[..., 0], (3, 3), 0)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    added = np.zeros_like(paper)
    h, w = paper.shape
    for k, page in enumerate(pages):
        if page.quad is None or page.features.keystone > EDGE_MAX_KEYSTONE:
            continue
        if not page.mask.any() or np.median(chroma[page.mask]) < EDGE_MIN_PAGE_CHROMA:
            continue
        # (наружу, транспонировано): внешняя боковая сторона по месту страницы, затем верх и низ.
        sides = [(-1 if side is Side.LEFT else 1, False) for side in outer_sides(k, len(pages))]
        sides += [(-1, True), (1, True)]
        for outward, transposed in sides:
            mask = page.mask.T if transposed else page.mask
            lum = light.T if transposed else light
            if not ragged(mask, outward):
                continue
            rows, xs, count = edge_dips(lum, mask, outward)
            fit = ransac_line(rows, xs, mm(EDGE_TOL_MM))
            if fit is None or fit[1].sum() < EDGE_MIN_SHARE * count:
                continue
            (a, b), inliers = fit
            # Только строки между крайними провалами на прямой: ниже и выше кромки этой стороны (угол листа) провала
            # нет, и заливка там ушла бы на картон под листом.
            span = rows[inliers]
            first, last = int(span.min()), int(span.max())
            # Поле: строки страницы, от края маски наружу до прямой.
            has = mask.any(axis=1)
            line = a * np.arange(mask.shape[0]) + b
            cols = np.arange(mask.shape[1])[None, :]
            if outward < 0:
                edge = np.argmax(mask, axis=1)
                band = (cols >= line[:, None]) & (cols < edge[:, None])
            else:
                edge = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
                band = (cols <= line[:, None]) & (cols > edge[:, None])
            band &= has[:, None]
            band[:first] = False
            band[last + 1 :] = False
            added |= band.T if transposed else band
    return added & book & ~bg & ~paper
