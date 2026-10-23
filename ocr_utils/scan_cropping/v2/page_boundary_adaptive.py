"""Сплайн верха и низа страницы с 1–2 добавленными узлами: прежний сплайн ``page_boundary``, а там, где край листа вдали от корешка и от дальнего угла полого и систематически уходит от кривой (пологий излом низа страницы), — по узлу со слегка ослабленным штрафом; ямы маски, дыры в точках и углы не трогаются; боковые стороны — как в ``page_boundary``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.scan_cropping.v2.page_boundary import (
    CURVE_SAMPLES,
    EXTEND,
    HUBER_MM,
    IRLS_ITERS,
    MIN_FIT_POINTS,
    SIDE_MODELS,
    SPINE_KNOT_MM,
    SPINE_OUTWARD_WEIGHT,
    SPINE_SMOOTH_FACTOR,
    SPINE_ZONE_MM,
    FitKind,
    PageBoundary,
    SideCurve,
    between,
    bspline_basis_knots,
    fit_side,
    fold_profile,
    intersect_polylines,
    outward_shift,
    shifted,
)
from ocr_utils.scan_cropping.v2.page_sides import Side, SideProfile, inward_normal, side_profiles

# Добавленных узлов на сторону не больше: изгиб края — одно-два места, а не волна по всей стороне. Третий узел пробовали
# (ХиП 1991/03 IMG_0110): после двух узлов ухода по видимым точкам нет (медиана ≤ 0.17 мм), зазор — под ядром пальца.
MAX_EXTRA_KNOTS = 2
# Шаг сетки, на которой ищется место для узла, мм.
GRID_MM = 2.0
# Окно, в котором меряется уход контура от кривой (медиана остатка), мм: шире зубцов маски и крючков.
BIAS_WINDOW_MM = 10.0
# Уход — медиана остатка в окне от BIAS_MM до BIAS_MAX_MM: пологий излом края (ХиП 1991/02 IMG_1130, низ правой
# страницы: край на 0.5–0.8 мм выше прежнего сплайна). Глубже — яма маски (загнутый угол ПХ 1975/02 IMG_0038 — до
# 20 мм), за ней кривая не идёт.
BIAS_MM = 0.3
BIAS_MAX_MM = 1.5
# Меньше точек в окне — уход не мерится (дыра под пальцем, рамка кадра), доля от ожидаемого числа точек окна.
BIAS_MIN_COVERAGE = 0.4
# Множитель штрафа у добавленного узла (строки штрафа ближе BEND_HALF_MM): кривая изгибается слегка, не ломается.
BEND_SMOOTH_FACTOR = 0.2
BEND_HALF_MM = 8.0
# Узел принимается, если дальше LOCAL_MM от него: (1) там, где точек нет (продолжение дальше GAP_MM за краем данных), и в зоне
# корешка кривая сдвинулась не больше LOCAL_TOL_MM — узел в 30 мм от зоны корешка менял низ у корешка на 4 мм (МТС
# 1991/05 IMG_0250, маска у сгиба со шпорой до 20 мм); (2) по точкам контура кривая легла не хуже (p90 остатка вырос
# не больше LOCAL_RESID_MM) — горб низа шириной 40–60 мм с вершиной под пальцем (ХиП 1991/01 IMG_1052, 1991/02
# IMG_1143) поднимает кривую и в 20+ мм от узла, и там это к краю.
LOCAL_MM = 20.0
LOCAL_TOL_MM = 0.2
LOCAL_RESID_MM = 0.05
# Добавленные узлы не ближе друг к другу, мм.
KNOT_GAP_MM = 15.0
# Узлы не ставятся ближе SPINE_GUARD_MM к зоне корешка (там прежний сплайн хорош, а контур поворачивает вдоль сгиба;
# излом низа бывает в 30–50 мм от корешка — ХиП 1991/01 IMG_1052, так что отступ узкий, а узел, меняющий кривую у
# корешка, отсекает проверка локальности) и ближе FAR_GUARD_MM к дальнему концу стороны (углы страниц не скруглены —
# изгиб у угла — это маска, не лист; узел в 28 мм от угла ещё отгибал угол, ПХ 1975/02 IMG_0038, низ правой страницы).
SPINE_GUARD_MM = 5.0
FAR_GUARD_MM = 30.0
# Дыра в точках стороны (зона пальца, рамка кадра) — разрыв между соседними годными точками длиннее GAP_MM; узел
# ставится только на сплошном участке точек длиной от RUN_MIN_MM (или не дальше GAP_MM от него): островок 7 мм между
# двумя зонами пальца выгибал кривую над дырой (ХиП 1991/02 IMG_1144), а изгиб у края дыры бывает настоящим (IMG_1052).
GAP_MM = 3.0
RUN_MIN_MM = 15.0
# Яма маски — точки, отстоящие от прежнего сплайна дальше PIT_MM; узлы не ближе PIT_GUARD_MM к ним: на склонах ямы
# медиана остатка умеренная, и узел там тянул кривую в яму (ПХ 1975/02 IMG_0038: с отступом 15 мм — на 0.5 мм глубже
# прежнего сплайна).
PIT_MM = 3.0
PIT_GUARD_MM = 25.0
# Яма длиннее PIT_SIDE_MM вдоль стороны (а не короткий крючок маски) запрещает новые узлы на всей стороне: она тянет и
# прежний сплайн, бока ямы смотрятся «уходом», и узел даже в 30 мм от неё углублял кривую в яме (ПХ 1975/02 IMG_0038,
# верх правой страницы: на 0.3 мм). Такая сторона остаётся прежним сплайном.
PIT_SIDE_MM = 5.0
# Почти отвесные точки профиля — размах отклонений на отрезке STEEP_SPAN_MM вдоль стороны больше STEEP_SLOPE × отрезок:
# контур соседней стороны у угла, попавший верху или низу (ХиП 1991/02 IMG_1144, низ левой страницы: отвесно на
# 9 мм). В поиске ухода не участвуют.
STEEP_SPAN_MM = 1.0
STEEP_SLOPE = 2.0


@dataclass(frozen=True)
class AdaptiveFit:
    """Итог подгонки стороны с добавленными узлами.

    Поля:
        coef: B-коэффициенты.
        knots: Узлы (концы кратные).
        added: Положения добавленных узлов вдоль стороны (доли); пусто — кривая совпадает с прежним сплайном.
    """

    coef: np.ndarray
    knots: np.ndarray
    added: np.ndarray


def solve_irls(
    design: np.ndarray, d: np.ndarray, penalty: np.ndarray, huber_px: float, zone: np.ndarray, use: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Робастная подгонка с заданным штрафом (IRLS Хубера, как в ``page_boundary.fit_offsets``).

    Args:
        design: Матрица B-сплайна (N, K).
        d: Отклонения точек от прямой стороны, px.
        penalty: Матрица штрафа (K, K).
        huber_px: Порог Хубера, px.
        zone: Точки в зоне у корешка (bool, N): выброс наружу там — шпора маски, весит ``SPINE_OUTWARD_WEIGHT``.
        use: Точки, участвующие в подгонке (bool, N); остальные — с нулевым весом.

    Returns:
        ``(коэффициенты, остатки d − кривая)``.
    """
    w = use.astype(np.float64)
    coef = np.zeros(design.shape[1])
    for _ in range(IRLS_ITERS):
        dw = design * w[:, None]
        coef = np.linalg.solve(design.T @ dw + penalty + 1e-9 * np.eye(design.shape[1]), dw.T @ d)
        r = d - design @ coef
        w = np.where(np.abs(r) <= huber_px, 1.0, huber_px / np.maximum(np.abs(r), 1e-9))
        w[zone & (r < -huber_px)] *= SPINE_OUTWARD_WEIGHT
        w[~use] = 0.0
    return coef, d - design @ coef


def window_bias(t: np.ndarray, r: np.ndarray, centres: np.ndarray, half: float, min_points: int) -> np.ndarray:
    """Медиана остатка в окне ``±half`` вокруг каждого центра (NaN — точек меньше ``min_points``).

    Args:
        t: Положения точек (по возрастанию), доли стороны.
        r: Остатки точек, px.
        centres: Центры окон, доли.
        half: Полуширина окна, доля.
        min_points: Наименьшее число точек в окне.

    Returns:
        Медианы по центрам, px.
    """
    lo = np.searchsorted(t, centres - half)
    hi = np.searchsorted(t, centres + half, side="right")
    return np.array([np.median(r[i:j]) if j - i >= min_points else np.nan for i, j in zip(lo, hi)])


def steep_points(t: np.ndarray, d: np.ndarray, length_px: float, px_per_mm: float) -> np.ndarray:
    """Почти отвесные точки профиля: размах ``d`` на отрезке ``STEEP_SPAN_MM`` вокруг точки круче ``STEEP_SLOPE``.

    Args:
        t: Положения точек (по возрастанию), доли стороны.
        d: Отклонения, px.
        length_px: Длина стороны, px.
        px_per_mm: Плотность, px/мм.

    Returns:
        Маска отвесных точек (bool, N).
    """
    half = STEEP_SPAN_MM / 2 * px_per_mm / max(length_px, 1.0)
    lo = np.searchsorted(t, t - half)
    hi = np.searchsorted(t, t + half, side="right")
    spread = np.array([np.ptp(d[i:j]) for i, j in zip(lo, hi)])
    return spread > STEEP_SLOPE * STEEP_SPAN_MM * px_per_mm


def base_knots(spine_zone: tuple[float, float] | None, n_segments: int, knot_step: float) -> np.ndarray:
    """Внутренние узлы прежнего сплайна (``page_boundary.fit_offsets``): равномерные отрезки, у корешка — через шаг.

    Args:
        spine_zone: Участок у корешка ``(t0, t1)`` или None.
        n_segments: Число равномерных отрезков (``SIDE_MODELS``).
        knot_step: Шаг узлов в зоне у корешка, доля стороны.

    Returns:
        Узлы от 0 до 1 по возрастанию.
    """
    inner = np.linspace(0.0, 1.0, n_segments + 1)
    if spine_zone is None:
        return inner
    z0, z1 = spine_zone
    extra = np.arange(z0, z1 + 1e-9, knot_step)
    return np.unique(np.concatenate([inner[(inner < z0) | (inner > z1)], extra, [0.0, 1.0]]))


def blocked_nodes(
    t: np.ndarray,
    r0: np.ndarray,
    grid: np.ndarray,
    spine_zone: tuple[float, float] | None,
    length_px: float,
    px_per_mm: float,
) -> np.ndarray:
    """Узлы сетки, где новый узел ставить нельзя: зона корешка и дальний угол, края дыр в точках, ямы маски.

    Args:
        t: Положения годных точек (по возрастанию), доли.
        r0: Остатки точек от прежнего сплайна, px.
        grid: Сетка кандидатов, доли.
        spine_zone: Участок у корешка ``(t0, t1)`` или None (страница одна — оба конца дальние).
        length_px: Длина стороны, px.
        px_per_mm: Плотность, px/мм.

    Returns:
        Маска запретных узлов (bool по ``grid``).
    """
    mm = px_per_mm / max(length_px, 1.0)
    far = FAR_GUARD_MM * mm
    out = np.zeros(len(grid), bool)
    if spine_zone is None:
        out |= (grid < far) | (grid > 1.0 - far)
    else:
        guard = SPINE_GUARD_MM * mm
        out |= (grid >= spine_zone[0] - guard) & (grid <= spine_zone[1] + guard)
        # Дальний конец — противоположный зоне корешка.
        out |= (grid > 1.0 - far) if spine_zone[0] == 0.0 else (grid < far)
    # Узел — на сплошном участке точек не короче RUN_MIN_MM (дыры — разрывы длиннее GAP_MM).
    breaks = np.flatnonzero(np.diff(t) > GAP_MM * mm)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(t) - 1]])
    on_run = np.zeros(len(grid), bool)
    for a, b in zip(t[starts], t[ends]):
        if b - a >= RUN_MIN_MM * mm:
            on_run |= (grid >= a - GAP_MM * mm) & (grid <= b + GAP_MM * mm)
    out |= ~on_run
    # Ямы и выступы маски — точки дальше PIT_MM от прежнего сплайна; длинная яма закрывает всю сторону.
    pits = t[np.abs(r0) > PIT_MM * px_per_mm]
    if len(pits):
        out |= (np.abs(grid[:, None] - pits[None, :]) <= PIT_GUARD_MM * mm).any(axis=1)
        if longest_run(pits, mm) > PIT_SIDE_MM * mm:
            out[:] = True
    return out


def longest_run(pts: np.ndarray, mm: float) -> float:
    """Длина самого длинного сплошного участка точек (соседние ближе 1 мм) вдоль стороны, доли.

    Args:
        pts: Положения точек (по возрастанию), доли.
        mm: Миллиметр в долях стороны.

    Returns:
        Длина участка, доли (0 — точка одна).
    """
    breaks = np.flatnonzero(np.diff(pts) > mm)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(pts) - 1]])
    return float((pts[ends] - pts[starts]).max())


def in_zone(x: np.ndarray, spine_zone: tuple[float, float] | None) -> np.ndarray:
    """Положения ``x`` (доли) в зоне корешка (bool); зоны нет — все False."""
    if spine_zone is None:
        return np.zeros(len(x), bool)
    return (x >= spine_zone[0]) & (x <= spine_zone[1])


def fit_with_knots(
    t: np.ndarray,
    d: np.ndarray,
    model_smooth: float,
    base: np.ndarray,
    added: list[float],
    spine_zone: tuple[float, float] | None,
    bend_half: float,
    huber_px: float,
    use: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Сплайн по узлам прежнего плюс ``added``; штраф — как у прежнего, у добавленных узлов — ``BEND_SMOOTH_FACTOR``.

    Args:
        t: Положения точек (доли).
        d: Отклонения, px.
        model_smooth: Вес штрафа семейства стороны (``SIDE_MODELS``).
        base: Внутренние узлы прежнего сплайна.
        added: Добавленные узлы (доли).
        spine_zone: Участок у корешка или None.
        bend_half: Полуширина ослабления штрафа у добавленного узла, доля стороны.
        huber_px: Порог Хубера, px.
        use: Точки, участвующие в подгонке (bool); остальные — с нулевым весом, но с остатками.

    Returns:
        ``(коэффициенты, узлы, остатки всех точек)``.
    """
    design, knots = bspline_basis_knots(t, np.unique(np.concatenate([base, added])))
    diff = np.diff(np.eye(design.shape[1]), n=2, axis=0)
    # Положение строки второй разности — узел knots[j + 2], ровно как в page_boundary.fit_offsets: без добавленных
    # узлов кривая совпадает с прежней.
    centres = knots[2 : 2 + diff.shape[0]]
    row_w = np.ones(diff.shape[0])
    zone = np.zeros(len(t), bool)
    if spine_zone is not None:
        row_w[(centres >= spine_zone[0]) & (centres <= spine_zone[1])] = SPINE_SMOOTH_FACTOR
        zone = (t >= spine_zone[0]) & (t <= spine_zone[1])
    for k in added:
        row_w[np.abs(centres - k) <= bend_half] = BEND_SMOOTH_FACTOR
    # Штраф масштабируется числом участвующих точек (как в fit_offsets) и коэффициентов прежнего сплайна: добавленные
    # узлы не ослабляют штраф всей стороны.
    n_base = len(base) + 2
    penalty = model_smooth * int(use.sum()) / n_base * (diff * row_w[:, None]).T @ diff
    coef, r = solve_irls(design, d, penalty, huber_px, zone, use)
    return coef, knots, r


def fit_adaptive(
    t: np.ndarray,
    d: np.ndarray,
    base_use: np.ndarray,
    side: Side,
    length_px: float,
    px_per_mm: float,
    spine_zone: tuple[float, float] | None,
) -> AdaptiveFit:
    """Сплайн отклонения ``d(t)`` верха или низа: прежний плюс до ``MAX_EXTRA_KNOTS`` узлов на пологих изломах.

    Сначала — ровно прежний сплайн (узлы и штраф ``page_boundary.fit_offsets``). Затем, пока узлов меньше
    ``MAX_EXTRA_KNOTS``: медиана остатка в окне ``BIAS_WINDOW_MM`` у каждого узла сетки через ``GRID_MM`` (без отвесных
    точек); среди узлов вне запретных мест (``blocked_nodes``) и не ближе ``KNOT_GAP_MM`` к уже добавленным берётся
    узел с наибольшим уходом от ``BIAS_MM`` до ``BIAS_MAX_MM``; он добавляется в сплайн со штрафом
    ``BEND_SMOOTH_FACTOR`` рядом, и кривая подгоняется заново. Узел остаётся, только если дальше ``LOCAL_MM`` от него
    кривая без точек и в зоне корешка сдвинулась не больше ``LOCAL_TOL_MM``, а по точкам легла не хуже (p90 остатка
    вырос не больше ``LOCAL_RESID_MM``); иначе место запрещается.

    Прежний сплайн строится только по точкам ``base_use`` (точки прежнего сплайна, без зоны закраса пальца); точки
    сверх них (припуск зоны закраса у ядра пальца — край там настоящий) участвуют в поиске излома и в подгонке с
    добавленными узлами. Без добавленных узлов кривая — ровно прежняя.

    Args:
        t: Положения годных точек вдоль стороны (доли, по возрастанию).
        d: Их отклонения от прямой стороны, px (плюс — внутрь страницы).
        base_use: Точки прежнего сплайна (bool по ``t``).
        side: Верх или низ (семейство ``SIDE_MODELS``).
        length_px: Длина стороны, px.
        px_per_mm: Плотность масштаба маски, px/мм.
        spine_zone: Участок ``(t0, t1)`` у корешка или None (страница одна).

    Returns:
        Подгонка.
    """
    model = SIDE_MODELS[side]
    mm = px_per_mm / max(length_px, 1.0)
    base = base_knots(spine_zone, model.knots, SPINE_KNOT_MM * mm)
    huber_px = HUBER_MM * px_per_mm
    grid = np.arange(0.0, 1.0 + 1e-9, GRID_MM * mm)
    half = BIAS_WINDOW_MM / 2 * mm
    # Контур маски — примерно точка на пиксель вдоль стороны: столько ждём в окне.
    min_points = max(int(BIAS_MIN_COVERAGE * BIAS_WINDOW_MM * px_per_mm), 5)
    added: list[float] = []
    coef, knots, r = fit_with_knots(t, d, model.smooth, base, added, spine_zone, BEND_HALF_MM * mm, huber_px, base_use)
    blocked = blocked_nodes(t, r, grid, spine_zone, length_px, px_per_mm)
    every = np.ones(len(t), bool)
    flat = ~steep_points(t, d, length_px, px_per_mm)
    # Кривая по всей стороне — для проверки локальности (продолжение за точками — по касательной, как у SideCurve).
    probe = np.linspace(0.0, 1.0, CURVE_SAMPLES)
    lo, hi = float(t[base_use].min()), float(t[base_use].max())
    zero = np.zeros(2)
    current = SideCurve(side, zero, zero, zero, coef, FitKind.SPLINE, knots, 0.0, 0.0, lo, hi).offset(probe)
    while len(added) < MAX_EXTRA_KNOTS:
        bias = np.abs(np.nan_to_num(window_bias(t[flat], r[flat], grid, half, min_points)))
        ok = ~blocked & (bias > BIAS_MM * px_per_mm) & (bias < BIAS_MAX_MM * px_per_mm)
        for k in added:
            ok &= np.abs(grid - k) >= KNOT_GAP_MM * mm
        if not ok.any():
            break
        k = float(grid[np.flatnonzero(ok)[np.argmax(bias[ok])]])
        trial = fit_with_knots(t, d, model.smooth, base, [*added, k], spine_zone, BEND_HALF_MM * mm, huber_px, every)
        offset = SideCurve(side, zero, zero, zero, trial[0], FitKind.SPLINE, trial[1], 0.0, 0.0, lo, hi).offset(probe)
        # Вдали от узла: без точек и в зоне корешка — почти не сдвинулась; по точкам — легла не хуже.
        far_away = np.abs(probe - k) > LOCAL_MM * mm
        # Без точек — дальше GAP_MM за крайними точками: последний миллиметр у угла — продолжение того же сдвига, что у
        # крайних точек (ХиП 1991/03 IMG_0107: точки кончаются за 1.2 мм до угла, узел на изломе там сдвигал кривую
        # на 0.32 мм и отвергался).
        bare = (probe < lo - GAP_MM * mm) | (probe > hi + GAP_MM * mm) | in_zone(probe, spine_zone)
        far_pts = np.abs(t - k) > LOCAL_MM * mm
        worse = far_pts.any() and (
            np.percentile(np.abs(trial[2][far_pts]), 90)
            > np.percentile(np.abs(r[far_pts]), 90) + LOCAL_RESID_MM * px_per_mm
        )
        if worse or np.abs(offset - current)[far_away & bare].max(initial=0.0) > LOCAL_TOL_MM * px_per_mm:
            # Изгиб не местный — место запрещается, ищется следующее.
            blocked |= np.abs(grid - k) <= half
            continue
        added.append(k)
        coef, knots, r = trial
        current = offset
    return AdaptiveFit(coef, knots, np.array(sorted(added)))


def fit_side_adaptive(
    profile: SideProfile,
    base_clean: np.ndarray,
    centre: np.ndarray,
    px_per_mm: float,
    spine_zone: tuple[float, float] | None,
) -> tuple[SideCurve, AdaptiveFit | None]:
    """Подгонка верха или низа сплайном с добавленными узлами (``fit_adaptive``).

    Args:
        profile: Точки контура у стороны (годные — с узкой зоной пальца).
        base_clean: Годные точки прежнего сплайна (bool по точкам ``profile``, подмножество ``profile.clean``).
        centre: Центр трапеции (для нормали внутрь).
        px_per_mm: Плотность масштаба маски, px/мм.
        spine_zone: Участок стороны у корешка или None.

    Returns:
        ``(сторона без сдвига наружу, подгонка или None — точек мало, сторона прямая)``.
    """
    normal = inward_normal(profile.a, profile.b, centre)
    t, d = profile.t[profile.clean], profile.d[profile.clean]
    base_use = base_clean[profile.clean]
    if base_use.sum() < MIN_FIT_POINTS:
        resid = float(np.percentile(np.abs(d), 90)) / px_per_mm if len(d) else float("nan")
        return SideCurve(profile.side, profile.a, profile.b, normal, np.zeros(1), FitKind.QUAD, None, resid, 0.0), None
    length = float(np.linalg.norm(profile.b - profile.a))
    fit = fit_adaptive(t, d, base_use, profile.side, length, px_per_mm, spine_zone)
    # Участок данных — как у прежнего сплайна (за ним — продолжение по касательной); с узлами — по всем точкам.
    use = base_use if not len(fit.added) else np.ones(len(t), bool)
    lo, hi = float(t[use].min()), float(t[use].max())
    curve = SideCurve(
        profile.side, profile.a, profile.b, normal, fit.coef, FitKind.SPLINE, fit.knots, float("nan"), 0.0, lo, hi
    )
    # Остаток — по точкам прежнего сплайна: сравним с его остатком.
    resid = float(np.percentile(np.abs(d[base_use] - curve.offset(t[base_use])), 90)) / px_per_mm
    curve = SideCurve(
        profile.side, profile.a, profile.b, normal, fit.coef, FitKind.SPLINE, fit.knots, resid, 0.0, lo, hi
    )
    return curve, fit


def fit_boundary_adaptive(
    mask: np.ndarray,
    quad: np.ndarray,
    spine_right: bool | None,
    px_per_mm: float,
    exclude: np.ndarray | None = None,
    text: np.ndarray | None = None,
    fold_points: np.ndarray | None = None,
    edge_exclude: np.ndarray | None = None,
    spine_zone_mm: float | None = SPINE_ZONE_MM,
) -> tuple[PageBoundary, dict[Side, AdaptiveFit]]:
    """Криволинейная рамка страницы: верх и низ — сплайном с добавленными узлами, боковые — сплайном ``page_boundary``.

    Повторяет ``page_boundary.fit_boundary`` для ``FitKind.SPLINE`` (сдвиг наружу только текстом, углы — пересечения
    продлённых кривых), меняется лишь подгонка верха и низа.

    Args:
        mask: Маска страницы.
        quad: Трапеция страницы (4, 2), px маски.
        spine_right: Где корешок (см. ``page_sides.quad_sides``).
        px_per_mm: Плотность масштаба маски, px/мм.
        exclude: Зона, где точки контура негодны (палец), или None.
        text: Вершины текстовых блоков страницы (N, 2), px маски, или None.
        fold_points: Линия сгиба (N, 2) px маски или None (сторона у корешка — по ней, без сдвига).
        edge_exclude: Зона негодных точек для верха и низа (ядро пальца с узким припуском) или None — как ``exclude``.
            Вся зона закраса шире пальца: в её припуске край листа настоящий, и там бывает излом (ХиП 1991/01
            IMG_1052, низ правой страницы).
        spine_zone_mm: Зона у корешка на верхе и низе, где узлы гуще и штраф слабее, мм, или None — без неё (у полосы-
            вклейки точек у корешка нет вовсе, и гибкость там только нырнула бы за кромкой маски у полосы).

    Returns:
        ``(рамка, подгонки верха и низа)``.
    """
    text = np.zeros((0, 2)) if text is None else text
    profiles = side_profiles(mask, quad, spine_right, exclude)
    # Верх и низ — с узкой зоной пальца (точки контура те же, годных больше); годные точки прежнего сплайна — по
    # полной зоне.
    edge_profiles = side_profiles(mask, quad, spine_right, exclude if edge_exclude is None else edge_exclude)
    centre = np.asarray(quad, np.float64).mean(axis=0)
    spine_len = float(np.linalg.norm(profiles[Side.TOP].b - profiles[Side.TOP].a))
    zone_len = (spine_zone_mm or 0.0) * px_per_mm / max(spine_len, 1.0)
    spine_zone = None
    if spine_right is not None and spine_zone_mm is not None:
        spine_zone = (1.0 - zone_len, 1.0) if spine_right else (0.0, zone_len)
    curves, fits, n_out = {}, {}, 0
    for side, profile in profiles.items():
        if side is Side.SPINE and fold_points is not None:
            curves[side] = fit_side(fold_profile(fold_points, profile, centre), centre, FitKind.SPLINE, px_per_mm)
            continue
        if side in (Side.TOP, Side.BOTTOM):
            edge = edge_profiles[side]
            curve, fit = fit_side_adaptive(edge, edge.clean & profile.clean, centre, px_per_mm, spine_zone)
            if fit is not None:
                fits[side] = fit
        else:
            curve = fit_side(profile, centre, FitKind.SPLINE, px_per_mm)
        shift, out = outward_shift(curve, profile, text)
        n_out += out
        curves[side] = shifted(curve, shift, px_per_mm)
    # Углы — пересечения продлённых кривых (нет пересечения — конец верха или низа со сдвигом), как в fit_boundary.
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
    boundary = PageBoundary(
        top=between(curves[Side.TOP], tl, tr),
        bottom=between(curves[Side.BOTTOM], bl, br),
        left=between(curves[left_side], tl, bl),
        right=between(curves[right_side], tr, br),
        curves=curves,
        kind=FitKind.SPLINE,
        text_outside=n_out / max(len(text), 1),
    )
    return boundary, fits
