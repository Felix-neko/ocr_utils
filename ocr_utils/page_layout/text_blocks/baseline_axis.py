"""Вторая ось строки — по БАЗОВОЙ ЛИНИИ глифов: гладкая, не прыгающая на прописных и цифрах, согласованная с соседями.

Первая ось (:mod:`segment`) — центр масс краски по столбцам. Центр масс зависит от класса буквы:
над прописной, цифрой или знаком он на полразницы высот выше, чем над строчной, и ось подпрыгивает
на «1975», «От», «В 1974» (1975/08 IMG_0087_2R). Низ буквы такого перекоса не даёт: на базовой
линии стоят и строчные, и прописные, и цифры; вниз от неё свисают только выносные («р», «у»,
запятая), а выше неё висят только верхние индексы и тире. Поэтому здесь:

1. **Базовая линия** — сглаживающий штрафной сплайн (Уиттекер: данные + вторая разность) по НИЗАМ
   глифов строки на сетке в 1 мм. Отсев односторонний и итеративный: низ глифа, ушедший ниже
   текущей кривой больше чем на ``DESCENDER_XH`` строчной, — выносной, выше больше чем на
   ``RAISED_XH`` — индекс, тире или дефис; оба выбрасываются, кривая строится заново.
2. **Ось** — базовая линия, поднятая на половину высоты строчной ЭТОЙ строки, где высота
   строчной — медиана «база минус верх» по строчным без выносных. Ось идёт параллельно базовой
   линии и поэтому не прыгает.
3. **Соседи.** Изгиб строки у края полосы и трапеция (наклон меняется от строки к строке)
   держатся на последних одной-двух буквах, и своих данных там мало. Второй проход добавляет к
   штрафу член «наклон строки равен наклону соседних строк в той же точке x» (две строки сверху
   и две снизу, вес — обратно расстоянию в шагах). Вес члена велик там, где своих низов мало
   (концы строки, пропуски), и мал в плотной середине: там решает своя краска. Идея — как
   ``fit_parallel_rows`` у Tesseract (строки блока параллельны), но локально: поле по всей
   странице (``legacy_linking``) изгиб угла не ловило.

Все длины — в высотах строчной строки, окна — в миллиметрах бумаги; координаты — пиксели рабочей
копии (``WORK_DPI``), как у остальных осей.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ocr_utils.page_layout import mm_to_px
from ocr_utils.page_layout.text_blocks.lines import LineAxis

# Шаг сетки кривой, мм бумаги (как у первой оси: ``lines.AXIS_STEP_MM``).
GRID_STEP_MM = 1.0
# Жёсткость базовой линии: длина, на которой кривая может заметно изогнуться, в высотах строчной.
# Короче — сплайн начинает повторять отдельные буквы; длиннее — не успевает за изгибом у края.
SMOOTH_XH = 4.0
# Односторонний отсев низов (доли высоты строчной): ниже кривой — выносной («р», «у», запятая,
# нижний индекс), выше — верхний индекс, тире, дефис, кавычка.
DESCENDER_XH = 0.22
RAISED_XH = 0.30
# Первый отсев — по МЕСТНОЙ МЕДИАНЕ низов соседних глифов в окне ±``LOCAL_WINDOW_XH`` строчных:
# её не сдвигает и половина выносных, а крайний глиф строки («р», запятая) иначе стягивал к себе
# конец кривой и сам же проходил отсев (ошибка конца оси до 4 px на синтетике).
LOCAL_WINDOW_XH = 3.0
# Голос за базовую линию по верху глифа — только у глифа не ниже этой доли строчной; низ такого
# мелкого глифа голосует с весом ``SMALL_GLYPH_WEIGHT``.
TOP_VOTE_MIN_XH = 0.75
SMALL_GLYPH_WEIGHT = 0.3
# Итераций отсева.
ITERATIONS = 4
# Строчная без выносных: высота в пределах этой доли от высоты строчной строки.
X_CLASS_TOLERANCE = 0.22
# Высота строчной строки — медиана глифов не выше такой доли 30-го перцентиля высот (прописные,
# цифры и выносные выше строчных на треть и больше).
X_HEIGHT_QUANTILE = 30.0
X_HEIGHT_SPREAD = 1.2
# Глиф выше стольких строчных — мостик между строками или слипшиеся буквы: не берётся вовсе.
BRIDGE_XH = 2.3
# Меньше стольких принятых низов — у строки второй оси нет (слишком мало данных).
MIN_SAMPLES = 3
# Соседи: сколько строк сверху и снизу опрашивается, насколько далеко (в шагах строки), какое
# перекрытие по x нужно и какой вес у члена соседей там, где своих данных нет совсем.
NEIGHBOURS_EACH_SIDE = 2
NEIGHBOUR_MAX_PITCHES = 2.6
NEIGHBOUR_MIN_OVERLAP = 0.3
NEIGHBOUR_WEIGHT = 4.0
# Окно, в котором считается плотность своих низов (в высотах строчной), и сколько низов в нём —
# «данных достаточно, соседи почти не нужны».
DENSITY_WINDOW_XH = 1.5
DENSITY_FULL = 3.0
# Даже в плотной середине соседи чуть-чуть держат наклон: доля полного веса.
NEIGHBOUR_FLOOR = 0.05


@dataclass(frozen=True)
class BaselineFit:
    """Базовая линия одной строки на сетке и её высота строчной.

    Args:
        grid: Абсциссы сетки (пиксели рабочей копии).
        base: Ординаты базовой линии на сетке.
        x_height: Высота строчной строки (пиксели).
        xs: Абсциссы голосов (центры глифов, каждый дважды), по которым шла подгонка.
        bottoms: Сами голоса за базовую линию: низы глифов и их верхи, опущенные на высоту строчной.
        kept: Какие голоса приняты после отсева.
    """

    grid: np.ndarray
    base: np.ndarray
    x_height: float
    xs: np.ndarray
    bottoms: np.ndarray
    kept: np.ndarray

    def slope(self) -> np.ndarray:
        """Наклон базовой линии в узлах сетки (центральные разности)."""
        return np.gradient(self.base, self.grid) if self.grid.size > 1 else np.zeros_like(self.base)


def line_x_height(glyphs: np.ndarray) -> float:
    """Высота строчной строки по её глифам: медиана низших глифов, без прописных, цифр и выносных.

    Args:
        glyphs: Боксы глифов ``(n, 4)`` — ``x0, y0, x1, y1``.

    Returns:
        Высота в пикселях; 0.0, если глифов нет.
    """
    if glyphs.shape[0] == 0:
        return 0.0
    heights = glyphs[:, 3] - glyphs[:, 1]
    # Низший «этаж» высот — строчные без выносных; прописные, цифры и выносные выше.
    floor = float(np.percentile(heights, X_HEIGHT_QUANTILE))
    low = heights[heights <= X_HEIGHT_SPREAD * floor]
    return float(np.median(low)) if low.size else floor


def _second_difference(count: int) -> sparse.csr_matrix:
    """Матрица второй разности ``(count − 2, count)``."""
    if count < 3:
        return sparse.csr_matrix((0, count))
    return sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(count - 2, count), format="csr")


def _first_difference(count: int) -> sparse.csr_matrix:
    """Матрица первой разности ``(count − 1, count)``."""
    if count < 2:
        return sparse.csr_matrix((0, count))
    return sparse.diags([-1.0, 1.0], [0, 1], shape=(count - 1, count), format="csr")


def _interpolation(grid: np.ndarray, xs: np.ndarray) -> sparse.csr_matrix:
    """Матрица линейной интерполяции значений сетки в точки ``xs`` ``(len(xs), len(grid))``."""
    step = grid[1] - grid[0] if grid.size > 1 else 1.0
    position = np.clip((xs - grid[0]) / step, 0.0, grid.size - 1.000001)
    left = np.floor(position).astype(int)
    frac = position - left
    right = np.minimum(left + 1, grid.size - 1)
    rows = np.repeat(np.arange(xs.size), 2)
    cols = np.column_stack([left, right]).ravel()
    values = np.column_stack([1.0 - frac, frac]).ravel()
    return sparse.csr_matrix((values, (rows, cols)), shape=(xs.size, grid.size))


def _solve(
    grid: np.ndarray,
    xs: np.ndarray,
    bottoms: np.ndarray,
    weights: np.ndarray,
    stiffness: float,
    neighbour_slope: np.ndarray | None = None,
    neighbour_weight: np.ndarray | None = None,
    anchor: np.ndarray | None = None,
) -> np.ndarray:
    """Штрафная подгонка кривой на сетке: данные, вторая разность и (необязательно) наклон соседей.

    Минимизируется ``Σ w (B(x) − низ)² + λ Σ (Δ²B)² + Σ μ (ΔB − s·шаг)² + ε Σ (B − якорь)²``.
    Последний член — слабая привязка к исходной кривой: без неё сетка за пределами данных
    неопределена.

    Args:
        grid: Абсциссы сетки.
        xs, bottoms, weights: Низы глифов и их веса (0 — отсеян).
        stiffness: λ — вес второй разности.
        neighbour_slope: Наклон соседей в узлах сетки (между узлами берётся среднее) или ``None``.
        neighbour_weight: μ в узлах сетки.
        anchor: Исходная кривая на сетке (первая ось, сдвинутая к базовой линии).

    Returns:
        Ординаты кривой на сетке.
    """
    count = grid.size
    interp = _interpolation(grid, xs)
    data = interp.T @ sparse.diags(weights) @ interp
    rhs = interp.T @ (weights * bottoms)
    second = _second_difference(count)
    system = data + stiffness * (second.T @ second)
    if neighbour_slope is not None and neighbour_weight is not None and count > 1:
        step = grid[1] - grid[0]
        first = _first_difference(count)
        mu = 0.5 * (neighbour_weight[:-1] + neighbour_weight[1:])
        target = 0.5 * (neighbour_slope[:-1] + neighbour_slope[1:]) * step
        system = system + first.T @ sparse.diags(mu) @ first
        rhs = rhs + first.T @ (mu * target)
    if anchor is not None:
        epsilon = 1e-4
        system = system + epsilon * sparse.identity(count)
        rhs = rhs + epsilon * anchor
    return np.asarray(spsolve(system.tocsc(), rhs), dtype=np.float64)


def fit_baseline(axis: LineAxis, x_height: float | None = None) -> BaselineFit | None:
    """Базовая линия строки по низам и верхам её глифов (первый проход, без соседей).

    Args:
        axis: Ось строки с глифами (``axis.glyphs``).
        x_height: Высота строчной; ``None`` — по глифам строки (:func:`line_x_height`).

    Returns:
        :class:`BaselineFit` или ``None``, если глифов нет или принятых низов слишком мало.
    """
    glyphs = axis.glyphs
    if glyphs is None or glyphs.shape[0] < MIN_SAMPLES:
        return None
    x_h = x_height or line_x_height(glyphs)
    if x_h <= 0:
        return None
    heights = glyphs[:, 3] - glyphs[:, 1]
    usable = heights <= BRIDGE_XH * x_h
    centres = 0.5 * (glyphs[usable, 0] + glyphs[usable, 2])
    # Два голоса за базовую линию от каждого глифа: его низ и его верх, опущенный на высоту
    # строчной. Низ врёт у выносных вниз («р», запятая), верх — у прописных, цифр и выносных вверх;
    # отсев у обоих один и тот же, односторонний (ниже кривой — выносной или точка с тире, выше —
    # прописная или индекс). У «р» верх стоит на линии строчных и голосует верно: три выносных
    # подряд в конце строки («…ру,») больше не тянут конец оси вниз.
    # Верх голосует только у глифа не ниже ``TOP_VOTE_MIN_XH`` строчной: у индекса, точки и тире
    # верх не лежит на линии строчных (индекс «м²» тянул ось на 4 px вверх).
    tall = heights[usable] >= TOP_VOTE_MIN_XH * x_h
    xs = np.concatenate([centres, centres[tall]])
    bottoms = np.concatenate([glyphs[usable, 3], glyphs[usable, 1][tall] + x_h]).astype(np.float64)
    # Вес голоса: мелкий глиф (индекс, тире, точка) голосует низом слабо — у конца строки из «б»,
    # «м²» и точки нормальных строчных нет, и равные голоса индексов перетягивали базу вверх.
    weights = np.concatenate([np.where(tall, 1.0, SMALL_GLYPH_WEIGHT), np.ones(int(tall.sum()))])
    if centres.size < MIN_SAMPLES:
        return None
    step = mm_to_px(GRID_STEP_MM, axis.dpi)
    start, stop = float(min(axis.x0, xs.min())), float(max(axis.x1, xs.max()))
    grid = np.arange(start, stop + step * 0.5, step)
    if grid.size < 2:
        grid = np.array([start, start + step])
    # Начальная кривая — первая ось, опущенная на полвысоты строчной к базовой линии.
    anchor = np.interp(grid, axis.points[:, 0], axis.points[:, 1]) + x_h / 2.0
    stiffness = (SMOOTH_XH * x_h / step) ** 4 / max(xs.size / grid.size, 1e-3) * 1e-2
    kept = _local_kept(xs, bottoms, weights, x_h)
    if kept.sum() < MIN_SAMPLES:
        return None
    base = _solve(grid, xs, bottoms, kept * weights, stiffness, anchor=anchor)
    for _ in range(ITERATIONS):
        residual = bottoms - np.interp(xs, grid, base)
        kept = (residual <= DESCENDER_XH * x_h) & (residual >= -RAISED_XH * x_h)
        if kept.sum() < MIN_SAMPLES:
            return None
        base = _solve(grid, xs, bottoms, kept * weights, stiffness, anchor=anchor)
    return BaselineFit(grid=grid, base=base, x_height=x_h, xs=xs, bottoms=bottoms, kept=kept)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Взвешенная медиана: значение, на котором накопленный вес переходит половину."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _local_kept(xs: np.ndarray, bottoms: np.ndarray, weights: np.ndarray, x_height: float) -> np.ndarray:
    """Первый отсев голосов: сверка с местной взвешенной медианой соседей (окно ±``LOCAL_WINDOW_XH``).

    Args:
        xs: Абсциссы голосов.
        bottoms: Голоса за базовую линию.
        weights: Веса голосов (мелкие глифы — слабее).
        x_height: Высота строчной строки.

    Returns:
        Маска принятых голосов.
    """
    window = LOCAL_WINDOW_XH * x_height
    near = np.abs(xs[:, None] - xs[None, :]) <= window
    # Медиана по окну — построчно: у большинства соседей база одна, выносные и индексы в меньшинстве.
    local = np.array([_weighted_median(bottoms[row], weights[row]) for row in near])
    residual = bottoms - local
    return (residual <= DESCENDER_XH * x_height) & (residual >= -RAISED_XH * x_height)


def _own_density(fit: BaselineFit) -> np.ndarray:
    """Доля «своих данных достаточно» в узлах сетки: сколько принятых низов рядом, от 0 до 1."""
    window = DENSITY_WINDOW_XH * fit.x_height
    near = np.abs(fit.grid[:, None] - fit.xs[fit.kept][None, :]) <= window
    return np.clip(near.sum(axis=1) / DENSITY_FULL, 0.0, 1.0)


def _neighbours(fits: list[BaselineFit | None], axes: list[LineAxis]) -> list[list[tuple[int, float]]]:
    """Соседи каждой строки: до ``NEIGHBOURS_EACH_SIDE`` сверху и снизу с перекрытием по x.

    Returns:
        Для каждой строки — список ``(номер соседа, вес)``; вес — обратно расстоянию в шагах.
    """
    centres = np.array([axis.cy for axis in axes])
    # Шаг строк — медиана расстояний до ближайшего ряда, перекрытого по x.
    gaps = []
    for index, axis in enumerate(axes):
        best = np.inf
        for other, candidate in enumerate(axes):
            if other == index:
                continue
            overlap = min(axis.x1, candidate.x1) - max(axis.x0, candidate.x0)
            if overlap > NEIGHBOUR_MIN_OVERLAP * min(axis.x1 - axis.x0, candidate.x1 - candidate.x0):
                distance = abs(centres[other] - centres[index])
                if distance > 0.3 * axis.height:
                    best = min(best, distance)
        if np.isfinite(best):
            gaps.append(best)
    pitch = float(np.median(gaps)) if gaps else 0.0
    out: list[list[tuple[int, float]]] = []
    for index, axis in enumerate(axes):
        found: list[tuple[int, float]] = []
        if pitch <= 0 or fits[index] is None:
            out.append(found)
            continue
        for sign in (-1, 1):
            side = []
            for other, candidate in enumerate(axes):
                if other == index or fits[other] is None:
                    continue
                offset = (centres[other] - centres[index]) * sign
                if offset <= 0.3 * pitch or offset > NEIGHBOUR_MAX_PITCHES * pitch:
                    continue
                overlap = min(axis.x1, candidate.x1) - max(axis.x0, candidate.x0)
                if overlap < NEIGHBOUR_MIN_OVERLAP * min(axis.x1 - axis.x0, candidate.x1 - candidate.x0):
                    continue
                side.append((offset, other))
            for offset, other in sorted(side)[:NEIGHBOURS_EACH_SIDE]:
                found.append((other, pitch / offset))
        out.append(found)
    return out


def _neighbour_slope(
    fit: BaselineFit, neighbours: list[tuple[int, float]], fits: list[BaselineFit | None]
) -> tuple[np.ndarray, np.ndarray]:
    """Наклон соседей в узлах сетки строки и их суммарный вес там (0 — ни один сосед этот x не накрыл)."""
    total = np.zeros(fit.grid.size)
    weight = np.zeros(fit.grid.size)
    for other, own_weight in neighbours:
        neighbour = fits[other]
        inside = (fit.grid >= neighbour.grid[0]) & (fit.grid <= neighbour.grid[-1])
        slope = np.interp(fit.grid, neighbour.grid, neighbour.slope())
        total[inside] += own_weight * slope[inside]
        weight[inside] += own_weight
    slope = np.divide(total, weight, out=np.zeros_like(total), where=weight > 0)
    return slope, weight


def body_axes(axes: list[LineAxis]) -> list[LineAxis]:
    """Вторые оси всех строк страницы: по базовой линии, со вторым проходом по соседям.

    Args:
        axes: Оси страницы с глифами (у строк без глифов второй оси нет).

    Returns:
        Те же оси, у каждой проставлено ``body_points`` (``None``, если посчитать не из чего).
    """
    fits = [fit_baseline(axis) for axis in axes]
    neighbours = _neighbours(fits, axes)
    out: list[LineAxis] = []
    for axis, fit, own in zip(axes, fits, neighbours):
        if fit is None:
            out.append(replace(axis, body_points=None))
            continue
        base = fit.base
        if own:
            slope, weight = _neighbour_slope(fit, own, fits)
            if weight.any():
                # Где своих низов мало — соседи держат наклон почти целиком; в плотной середине —
                # едва-едва. Вес масштабируется так же, как данные (на узел сетки).
                density = _own_density(fit)
                mu = NEIGHBOUR_WEIGHT * np.where(weight > 0, np.maximum(1.0 - density, NEIGHBOUR_FLOOR), 0.0)
                step = fit.grid[1] - fit.grid[0]
                stiffness = (SMOOTH_XH * fit.x_height / step) ** 4 / max(fit.xs.size / fit.grid.size, 1e-3) * 1e-2
                base = _solve(
                    fit.grid,
                    fit.xs,
                    fit.bottoms,
                    fit.kept.astype(np.float64),
                    stiffness,
                    neighbour_slope=slope,
                    neighbour_weight=mu * (fit.xs.size / fit.grid.size),
                    anchor=fit.base,
                )
        lift = _body_lift(axis, fit, base)
        points = np.column_stack([fit.grid, base - lift])
        # Ось обрезается по охвату первой оси: сетка могла выйти за неё на крайний глиф.
        inside = (points[:, 0] >= axis.x0 - 1e-6) & (points[:, 0] <= axis.x1 + 1e-6)
        out.append(replace(axis, body_points=points[inside] if inside.sum() >= 2 else points))
    return out


def _body_lift(axis: LineAxis, fit: BaselineFit, base: np.ndarray) -> float:
    """На сколько поднять базовую линию до оси: полвысоты строчной, измеренной от самой базовой линии.

    Высота строчной — медиана «база минус верх» по строчным без выносных (их высота в пределах
    ``X_CLASS_TOLERANCE`` от высоты строчной строки). Нет таких — половина оценки высоты строчной.
    """
    glyphs = axis.glyphs
    heights = glyphs[:, 3] - glyphs[:, 1]
    x_class = np.abs(heights - fit.x_height) <= X_CLASS_TOLERANCE * fit.x_height
    if x_class.sum() >= MIN_SAMPLES:
        centres = 0.5 * (glyphs[x_class, 0] + glyphs[x_class, 2])
        measured = np.interp(centres, fit.grid, base) - glyphs[x_class, 1]
        return float(np.median(measured)) / 2.0
    return fit.x_height / 2.0


def use_body(axes: list[LineAxis]) -> list[LineAxis]:
    """Оси, у которых основной стала вторая ось (по базовой линии), где она есть.

    Прежняя ось остаётся в ``centre_points`` — для сравнения и отрисовки «было — стало».
    """
    return [
        replace(axis, points=axis.body_points, centre_points=axis.points) if axis.body_points is not None else axis
        for axis in axes
    ]


__all__ = ["BaselineFit", "body_axes", "fit_baseline", "line_x_height", "use_body"]
