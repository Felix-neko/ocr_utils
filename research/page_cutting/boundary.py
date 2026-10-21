"""Криволинейная рамка страницы: четыре стороны маски v2, каждая — своим семейством кривых (полином или сглаживающий B-сплайн со штрафом) относительно стороны трапеции, робастно, с углами на пересечениях."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.interpolate import BSpline

from research.page_cutting.geometry import Side, SideProfile, inward_normal, quad_sides, side_profiles

# Итераций IRLS (перевзвешивание Хубера).
IRLS_ITERS = 8
# Порог Хубера, мм: отклонения контура от кривой больше — выбросы (зубцы рваного края, соринки, ямы маски).
HUBER_MM = 1.0
# Доля годных точек стороны, которая должна оказаться внутри рамки после сдвига наружу (как квантиль трапеции v2,
# но мягче: кривая и так идёт по краю, сдвиг лишь накрывает остаток зубцов).
OUTER_QUANTILE = 0.95
# Меньше годных точек — сторона остаётся прямой стороной трапеции.
MIN_FIT_POINTS = 30
# Насколько продлевать кривую за концы стороны (доля длины): углы ищутся на пересечении продлённых кривых.
EXTEND = 0.15
# Точек на кривой при выборке.
CURVE_SAMPLES = 400
# Точки верха и низа ближе к стороне у корешка в подгонку не идут, мм: у сгиба маска задирается «шпорой» к соседней
# странице (МТС 1991/05 IMG_0250), и полином уводит угол за край листа.
SPINE_EXCLUDE_MM = 5.0
# Зона у корешка на верхе и низе, мм: там край листа загибается к сгибу (страница уходит в корешок, ХиП 1991/02
# IMG_1130) — для сплайна здесь узлы гуще и штраф слабее, а выбросы наружу (шпора маски) почти не весят.
SPINE_ZONE_MM = 25.0
# Шаг узлов сплайна верха и низа в зоне у корешка, мм.
SPINE_KNOT_MM = 4.0
# Множитель штрафа на коэффициенты в зоне у корешка.
SPINE_SMOOTH_FACTOR = 0.05
# Вес выброса наружу в зоне у корешка относительно обычного Хубера.
SPINE_OUTWARD_WEIGHT = 0.05


class FitKind(Enum):
    """Вариант рамки."""

    # Трапеция v2 как есть (базовая линия).
    QUAD = "quad"
    # Полиномы: верх и низ — 3-й степени, дальняя сторона — 2-й, у корешка — 3-й.
    POLY = "poly"
    # Сглаживающие B-сплайны со штрафом на вторую разность: у корешка узлов больше (загиб у концов).
    SPLINE = "spline"


@dataclass(frozen=True)
class SideModel:
    """Семейство кривой для стороны.

    Поля:
        degree: Степень полинома (``POLY``).
        knots: Число внутренних отрезков B-сплайна (``SPLINE``).
        smooth: Вес штрафа на вторую разность коэффициентов B-сплайна (больше — ровнее).
    """

    degree: int
    knots: int
    smooth: float


# Семейства по сторонам: верх и низ гнутся вдоль строк (дуга, S-образный изгиб у корешка); дальняя сторона почти
# прямая (обрез листа в стороне от корешка); сторона у корешка загибается у концов — ей больше свободы.
SIDE_MODELS = {
    Side.TOP: SideModel(degree=3, knots=8, smooth=3.0),
    Side.BOTTOM: SideModel(degree=3, knots=8, smooth=3.0),
    Side.FAR: SideModel(degree=2, knots=4, smooth=10.0),
    Side.SPINE: SideModel(degree=3, knots=12, smooth=1.0),
}


@dataclass(frozen=True)
class SideCurve:
    """Подогнанная сторона.

    Поля:
        side: Какая сторона.
        a: Начало стороны трапеции (px).
        b: Конец стороны трапеции (px).
        normal: Единичная нормаль внутрь страницы.
        coef: Коэффициенты кривой отклонения ``d(t)`` (полином — по возрастанию степени, сплайн — B-коэффициенты).
        kind: Вариант рамки (как вычислять ``d(t)``).
        knots: Узлы B-сплайна (для ``SPLINE``) или None.
        resid_p90_mm: 90-й перцентиль |отклонения контура от кривой| по годным точкам, мм (NaN — не подгонялась).
        shift_mm: Сдвиг наружу, чтобы ``OUTER_QUANTILE`` точек и весь текст оказались внутри, мм.
        t_lo: Начало участка стороны, где были точки подгонки (доля); дальше кривая идёт по касательной.
        t_hi: Конец этого участка.
    """

    side: Side
    a: np.ndarray
    b: np.ndarray
    normal: np.ndarray
    coef: np.ndarray
    kind: FitKind
    knots: np.ndarray | None
    resid_p90_mm: float
    shift_mm: float
    t_lo: float = 0.0
    t_hi: float = 1.0

    def raw_offset(self, t: np.ndarray) -> np.ndarray:
        """Отклонение по самой кривой семейства (без продолжения по касательной), px."""
        if self.kind is FitKind.QUAD:
            return np.zeros_like(t)
        if self.kind is FitKind.POLY:
            return np.polynomial.polynomial.polyval(t, self.coef)
        return BSpline(self.knots, self.coef, 3, extrapolate=True)(t)

    def offset(self, t: np.ndarray) -> np.ndarray:
        """Отклонение кривой от прямой стороны в точках ``t`` (доли стороны), px, плюс — внутрь.

        Вне участка ``[t_lo, t_hi]``, где были точки подгонки, кривая продолжается по касательной: полином за краем
        данных уходит в сторону (угол улетал за лист).
        """
        t = np.asarray(t, np.float64)
        out = self.raw_offset(np.clip(t, self.t_lo, self.t_hi))
        eps = 1e-3
        for end, sign in ((self.t_lo, 1.0), (self.t_hi, -1.0)):
            beyond = (t < end) if sign > 0 else (t > end)
            if beyond.any():
                inner = end + sign * eps
                slope = (self.raw_offset(np.array([inner]))[0] - self.raw_offset(np.array([end]))[0]) / (sign * eps)
                out[beyond] = self.raw_offset(np.array([end]))[0] + slope * (t[beyond] - end)
        return out

    def points(self, t: np.ndarray) -> np.ndarray:
        """Точки кривой (N, 2), px, в положениях ``t`` вдоль стороны."""
        base = self.a[None, :] + t[:, None] * (self.b - self.a)[None, :]
        return base + self.offset(t)[:, None] * self.normal[None, :]


@dataclass(frozen=True)
class PageBoundary:
    """Криволинейная рамка страницы.

    Поля:
        top: Верх слева направо, (N, 2) px — от угла до угла.
        bottom: Низ слева направо.
        left: Левая сторона сверху вниз.
        right: Правая сторона сверху вниз.
        curves: Подогнанные стороны (меры подгонки).
        kind: Вариант рамки.
        text_outside: Доля вершин текстовых блоков вне рамки до сдвига (что пришлось накрыть).
    """

    top: np.ndarray
    bottom: np.ndarray
    left: np.ndarray
    right: np.ndarray
    curves: dict[Side, SideCurve]
    kind: FitKind
    text_outside: float

    def polygon(self) -> np.ndarray:
        """Замкнутый контур рамки по часовой стрелке: верх, правая, низ (обратно), левая (обратно)."""
        return np.vstack([self.top, self.right[1:], self.bottom[::-1][1:], self.left[::-1][1:-1]])


def bspline_basis(t: np.ndarray, n_segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Матрица кубического B-сплайна на [0, 1] с равномерными узлами (концы — кратные).

    Args:
        t: Положения точек (могут выходить за [0, 1] — продолжение крайних кусков).
        n_segments: Число отрезков между узлами.

    Returns:
        ``(матрица (N, n_segments + 3), узлы)``.
    """
    return bspline_basis_knots(t, np.linspace(0.0, 1.0, n_segments + 1))


def bspline_basis_knots(t: np.ndarray, inner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Матрица кубического B-сплайна по узлам ``inner`` (от 0 до 1 по возрастанию; концы делаются кратными).

    Args:
        t: Положения точек.
        inner: Узлы, включая 0 и 1.

    Returns:
        ``(матрица (N, len(inner) + 2), узлы)``.
    """
    knots = np.concatenate([[0.0] * 3, inner, [1.0] * 3])
    n_coef = len(knots) - 4
    basis = np.zeros((len(t), n_coef))
    for j in range(n_coef):
        coef = np.zeros(n_coef)
        coef[j] = 1.0
        basis[:, j] = BSpline(knots, coef, 3, extrapolate=True)(t)
    return basis, knots


def fit_offsets(
    t: np.ndarray,
    d: np.ndarray,
    kind: FitKind,
    model: SideModel,
    huber_px: float,
    spine_zone: tuple[float, float] | None = None,
    knot_step: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Робастная подгонка отклонения ``d(t)`` кривой выбранного семейства (IRLS с весами Хубера).

    Args:
        t: Положения годных точек вдоль стороны (доли).
        d: Их отклонения от прямой стороны, px.
        kind: ``POLY`` или ``SPLINE``.
        model: Семейство стороны.
        huber_px: Порог Хубера, px.
        spine_zone: Участок ``(t0, t1)`` у корешка (только верх и низ сплайном) или None. В нём узлы через
            ``knot_step``, штраф слабее в ``SPINE_SMOOTH_FACTOR`` раз, выбросы наружу весят ``SPINE_OUTWARD_WEIGHT``.
        knot_step: Шаг узлов в зоне у корешка, доля стороны.

    Returns:
        ``(коэффициенты, узлы или None)``.
    """
    zone = np.zeros(len(t), bool)
    if kind is FitKind.POLY:
        design = np.vander(t, model.degree + 1, increasing=True)
        knots, penalty = None, np.zeros((model.degree + 1, model.degree + 1))
    else:
        inner = np.linspace(0.0, 1.0, model.knots + 1)
        if spine_zone is not None and knot_step:
            z0, z1 = spine_zone
            extra = np.arange(z0, z1 + 1e-9, knot_step)
            inner = np.unique(np.concatenate([inner[(inner < z0) | (inner > z1)], extra, [0.0, 1.0]]))
            zone = (t >= z0) & (t <= z1)
        design, knots = bspline_basis_knots(t, inner)
        diff = np.diff(np.eye(design.shape[1]), n=2, axis=0)
        # Вес строки штрафа: в зоне у корешка слабее (там нужен резкий загиб края).
        row_w = np.ones(diff.shape[0])
        if spine_zone is not None:
            centres = knots[2 : 2 + diff.shape[0]] if len(knots) >= diff.shape[0] + 2 else np.zeros(diff.shape[0])
            in_zone = (centres >= spine_zone[0]) & (centres <= spine_zone[1])
            row_w[in_zone] = SPINE_SMOOTH_FACTOR
        # Штраф масштабируется числом точек: вес не зависит от плотности контура.
        penalty = model.smooth * len(t) / design.shape[1] * (diff * row_w[:, None]).T @ diff
    w = np.ones_like(d)
    coef = np.zeros(design.shape[1])
    for _ in range(IRLS_ITERS):
        dw = design * w[:, None]
        coef = np.linalg.solve(design.T @ dw + penalty + 1e-9 * np.eye(design.shape[1]), dw.T @ d)
        r = d - design @ coef
        w = np.where(np.abs(r) <= huber_px, 1.0, huber_px / np.maximum(np.abs(r), 1e-9))
        # У корешка точка снаружи кривой (r < 0) — шпора маски к соседней странице, а не край листа.
        w[zone & (r < -huber_px)] *= SPINE_OUTWARD_WEIGHT
    return coef, knots


def fit_side(
    profile: SideProfile,
    centre: np.ndarray,
    kind: FitKind,
    px_per_mm: float,
    keep: np.ndarray | None = None,
    spine_zone: tuple[float, float] | None = None,
) -> SideCurve:
    """Подгонка одной стороны.

    Args:
        profile: Точки контура у стороны.
        centre: Центр трапеции (для нормали внутрь).
        kind: Вариант рамки.
        px_per_mm: Плотность масштаба маски, px/мм.
        keep: Дополнительный отбор точек профиля (bool по точкам) или None.
        spine_zone: Участок стороны у корешка ``(t0, t1)`` для сплайна верха и низа (см. ``fit_offsets``) или None.

    Returns:
        Сторона (без сдвига наружу).
    """
    normal = inward_normal(profile.a, profile.b, centre)
    sel = profile.clean if keep is None else profile.clean & keep
    t, d = profile.t[sel], profile.d[sel]
    if kind is FitKind.QUAD or len(t) < MIN_FIT_POINTS:
        resid = float(np.percentile(np.abs(d), 90)) / px_per_mm if len(d) else float("nan")
        return SideCurve(profile.side, profile.a, profile.b, normal, np.zeros(1), FitKind.QUAD, None, resid, 0.0)
    length = float(np.linalg.norm(profile.b - profile.a))
    step = SPINE_KNOT_MM * px_per_mm / max(length, 1.0)
    coef, knots = fit_offsets(t, d, kind, SIDE_MODELS[profile.side], HUBER_MM * px_per_mm, spine_zone, step)
    lo, hi = float(t.min()), float(t.max())
    curve = SideCurve(profile.side, profile.a, profile.b, normal, coef, kind, knots, float("nan"), 0.0, lo, hi)
    resid = float(np.percentile(np.abs(d - curve.offset(t)), 90)) / px_per_mm
    return SideCurve(profile.side, profile.a, profile.b, normal, coef, kind, knots, resid, 0.0, lo, hi)


def fold_profile(fold_points: np.ndarray, profile: SideProfile, centre: np.ndarray) -> SideProfile:
    """Профиль стороны у корешка по точкам сгиба (вместо контура маски): положение и отклонение от прямой стороны."""
    a, b = profile.a, profile.b
    ab = b - a
    t = ((fold_points - a) @ ab) / (ab @ ab)
    d = (fold_points - a) @ inward_normal(a, b, centre)
    order = np.argsort(t)
    return SideProfile(profile.side, a, b, t[order], d[order], np.ones(len(t), bool))


def outward_shift(curve: SideCurve, profile: SideProfile, text: np.ndarray) -> tuple[float, int]:
    """Сдвиг стороны наружу (px), чтобы все вершины текста были внутри (``profile`` — для единообразия вызова).

    Args:
        curve: Подогнанная сторона.
        profile: Её точки контура.
        text: Вершины текстовых блоков страницы (N, 2), px.

    Returns:
        ``(сдвиг ≥ 0, число вершин текста за кривой до сдвига)``.
    """
    # Сдвиг по квантилю контура снят: один крючок маски в углу уводил всю сторону наружу (ХиП 1991/02 IMG_1168,
    # низ правой страницы на 0.8 мм ниже края листа). Кривая идёт по краю; наружу её сдвигает только текст.
    shift = 0.0
    n_out = 0
    if len(text):
        ab = curve.b - curve.a
        tt = ((text - curve.a) @ ab) / (ab @ ab)
        near = (tt > -EXTEND) & (tt < 1 + EXTEND)
        dd = (text[near] - curve.a) @ curve.normal
        beyond = curve.offset(tt[near]) - dd
        n_out = int((beyond > 0).sum())
        if n_out:
            shift = max(shift, float(beyond.max()))
    return max(shift, 0.0), n_out


def shifted(curve: SideCurve, shift_px: float, px_per_mm: float) -> SideCurve:
    """Сторона, сдвинутая наружу на ``shift_px`` (постоянный сдвиг кривой вдоль нормали)."""
    if shift_px <= 0:
        return curve
    a = curve.a - shift_px * curve.normal
    b = curve.b - shift_px * curve.normal
    return SideCurve(
        curve.side,
        a,
        b,
        curve.normal,
        curve.coef,
        curve.kind,
        curve.knots,
        curve.resid_p90_mm,
        shift_px / px_per_mm,
        curve.t_lo,
        curve.t_hi,
    )


def intersect_polylines(p: np.ndarray, q: np.ndarray) -> np.ndarray | None:
    """Точка пересечения двух ломаных (первое найденное пересечение отрезков) или None.

    Args:
        p: Ломаная (N, 2).
        q: Ломаная (M, 2).

    Returns:
        Точка (2,) или None.
    """
    a0, a1 = p[:-1, None, :], p[1:, None, :]
    b0, b1 = q[None, :-1, :], q[None, 1:, :]
    da, db = a1 - a0, b1 - b0
    denom = da[..., 0] * db[..., 1] - da[..., 1] * db[..., 0]
    diff = b0 - a0
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (diff[..., 0] * db[..., 1] - diff[..., 1] * db[..., 0]) / denom
        u = (diff[..., 0] * da[..., 1] - diff[..., 1] * da[..., 0]) / denom
    hit = (s >= 0) & (s <= 1) & (u >= 0) & (u <= 1) & np.isfinite(s)
    if not hit.any():
        return None
    i, j = np.argwhere(hit)[0]
    return (a0[i, 0] + s[i, j] * da[i, 0]).astype(np.float64)


def between(curve: SideCurve, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Кусок кривой от угла ``start`` до угла ``end`` (по проекциям углов на сторону), ``CURVE_SAMPLES`` точек."""
    ab = curve.b - curve.a
    t0 = float((start - curve.a) @ ab / (ab @ ab))
    t1 = float((end - curve.a) @ ab / (ab @ ab))
    pts = curve.points(np.linspace(t0, t1, CURVE_SAMPLES))
    pts[0], pts[-1] = start, end
    return pts


def fit_boundary(
    mask: np.ndarray,
    quad: np.ndarray,
    spine_right: bool | None,
    kind: FitKind,
    px_per_mm: float,
    exclude: np.ndarray | None = None,
    text: np.ndarray | None = None,
    fold_points: np.ndarray | None = None,
) -> PageBoundary:
    """Криволинейная рамка страницы.

    Точки контура маски раздаются сторонам трапеции; каждая сторона — своим семейством (``SIDE_MODELS``), Хубером;
    затем сдвиг наружу до ``OUTER_QUANTILE`` точек и всего текста; углы — пересечения продлённых соседних кривых.

    Args:
        mask: Маска страницы.
        quad: Трапеция страницы (4, 2), px маски.
        spine_right: Где корешок (см. ``geometry.quad_sides``).
        kind: Вариант рамки.
        px_per_mm: Плотность масштаба маски, px/мм.
        exclude: Зона, где точки контура негодны (палец), или None.
        text: Вершины текстовых блоков страницы (N, 2), px маски, или None.
        fold_points: Линия сгиба (N, 2) px маски или None. Если дана — сторона у корешка подгоняется по ней, без
            сдвига наружу (сгиб — точная граница страницы); иначе — по контуру маски.

    Returns:
        Рамка.
    """
    text = np.zeros((0, 2)) if text is None else text
    profiles = side_profiles(mask, quad, spine_right, exclude)
    centre = np.asarray(quad, np.float64).mean(axis=0)
    spine_len = float(np.linalg.norm(profiles[Side.TOP].b - profiles[Side.TOP].a))
    # Конец верха и низа у корешка: t = 1, если корешок справа (стороны идут слева направо), иначе t = 0.
    spine_end = 1.0 if spine_right in (True, None) else 0.0
    near_spine = SPINE_EXCLUDE_MM * px_per_mm / max(spine_len, 1.0)
    zone_len = SPINE_ZONE_MM * px_per_mm / max(spine_len, 1.0)
    zone = (1.0 - zone_len, 1.0) if spine_end == 1.0 else (0.0, zone_len)
    curves, n_out = {}, 0
    for side, profile in profiles.items():
        keep, side_zone = None, None
        if side in (Side.TOP, Side.BOTTOM) and spine_right is not None:
            if kind is FitKind.POLY:
                # Полином местный загиб у корешка не опишет, а шпора маски уводит угол: точки у корешка не берём.
                keep = np.abs(profile.t - spine_end) > near_spine
            else:
                side_zone = zone
        if side is Side.SPINE and fold_points is not None:
            profile = fold_profile(fold_points, profile, centre)
            # Сгиб — граница страницы как есть: ни квантиль контура, ни текст surya её не сдвигают.
            curves[side] = fit_side(profile, centre, kind if kind is not FitKind.QUAD else FitKind.SPLINE, px_per_mm)
            continue
        curve = fit_side(profile, centre, kind, px_per_mm, keep, side_zone)
        shift, out = outward_shift(curve, profile, text)
        n_out += out
        curves[side] = shifted(curve, shift, px_per_mm)
    # Ломаные продлённых кривых; углы — их пересечения (нет пересечения — угол трапеции со сдвигом).
    t_ext = np.linspace(-EXTEND, 1 + EXTEND, CURVE_SAMPLES)
    lines = {side: c.points(t_ext) for side, c in curves.items()}
    left_side, right_side = (Side.FAR, Side.SPINE) if spine_right in (True, None) else (Side.SPINE, Side.FAR)
    corners = {}
    for horiz in (Side.TOP, Side.BOTTOM):
        for vert, end in ((left_side, 0), (right_side, 1)):
            hit = intersect_polylines(lines[horiz], lines[vert])
            if hit is None:
                hit = curves[horiz].a if end == 0 else curves[horiz].b
            corners[(horiz, end)] = hit
    tl, tr = corners[(Side.TOP, 0)], corners[(Side.TOP, 1)]
    bl, br = corners[(Side.BOTTOM, 0)], corners[(Side.BOTTOM, 1)]
    return PageBoundary(
        top=between(curves[Side.TOP], tl, tr),
        bottom=between(curves[Side.BOTTOM], bl, br),
        left=between(curves[left_side], tl, bl),
        right=between(curves[right_side], tr, br),
        curves=curves,
        kind=kind,
        text_outside=n_out / max(len(text), 1),
    )
