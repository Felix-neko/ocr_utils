"""Растяжение у корешка по модели сжатия: одномерная монотонная карта ``x → x + dx(x, y)`` каждой страницы — с неподвижным внешним краем (режим ``shift``: лишняя длина уходит в пустое поле у корешка, сверх запаса — сжатием тела) или с расширением страницы со стороны корешка (режим ``widen``: растягивается только зона текста, разница по высоте — общим масштабом строки), смещение в формате поля ``text_field`` для сложения с ним; обратная карта — синтетическое сжатие для проверки мер."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.rectify.coons import Layout, PageRect
from research.gutter_compression.profile import LENGTH_MM, CompressionModel

# Шаг сетки интегрирования по расстоянию от корешка, мм.
GRID_MM = 0.05
# Рядов сетки по высоте страницы для амплитуды, зависящей от высоты (между ними — линейно).
Y_NODES = 9


@dataclass(frozen=True)
class StretchLimits:
    """Пределы растяжения.

    Поля:
        max_stretch: Наибольшее растяжение ``1/c`` (выше — амплитуда модели урезается).
        edge_safety_mm: Край текста у корешка после растяжения не ближе этого к линии корешка, мм.
        min_amplitude: Модель с амплитудой меньше этой не применяется (шум меры, по контролю).
        max_body_squeeze: Наибольшее равномерное сжатие тела страницы (доля), которым гасится лишняя длина сверх
            запаса поля у корешка; если не хватает — амплитуда модели урезается.
        min_excess_mm: Модель с лишней длиной текста меньше этой не применяется: на контроле без сжатия подгонка
            даёт короткие спады (L = 1.5 мм) с амплитудой до 0.3–0.38 и лишней длиной до 0.5–0.77 мм (синтетика p90 0.43 мм,
            контрольные кадры прогона stretch — до 0.77 мм).
    """

    max_stretch: float = 1.6
    edge_safety_mm: float = 1.0
    min_amplitude: float = 0.04
    max_body_squeeze: float = 0.01
    min_excess_mm: float = 0.8


@dataclass(frozen=True)
class SidePlan:
    """Растяжение одной страницы.

    Поля:
        model: Модель сжатия (амплитуда уже урезана пределом растяжения).
        width_mm: Расстояние от линии корешка до внешнего края страницы, мм.
        edge_mm: Отступ края текста от корешка до растяжения (медиана строк), мм.
        excess_mm: Лишняя длина текста строки ``∫(1/c − 1)`` от края текста до внешнего края (худшая высота), мм.
        body_squeeze: Доля равномерного сжатия тела страницы (лишнее сверх сдвига края текста; худшая высота),
            0 — не нужно.
        applied: Растяжение применяется.
        shift_mm: Сдвиг края текста к корешку, мм — один на всю высоту страницы: выровненная сторона блока у корешка
            остаётся прямой и при сжатии, меняющемся по высоте (лишнее сверх сдвига на каждой высоте гасит тело).
        outer_mm: Внешний край текста от корешка, мм: от него наружу карта тождественна, и внешняя выровненная
            сторона блока не сдвигается ни на какой высоте (тело между краями текста гасит остаток).
    """

    model: CompressionModel
    width_mm: float
    edge_mm: float
    excess_mm: float
    body_squeeze: float
    applied: bool
    shift_mm: float = 0.0
    outer_mm: float = 0.0


def capped(model: CompressionModel, limits: StretchLimits) -> CompressionModel:
    """Модель с амплитудой, урезанной так, чтобы растяжение ``1/c`` нигде не превышало ``max_stretch``.

    Args:
        model: Модель.
        limits: Пределы.

    Returns:
        Модель с амплитудой (и наклоном) не выше ``1 − 1/max_stretch`` на любой высоте.
    """
    top = 1.0 - 1.0 / limits.max_stretch
    worst = max(model.amplitude_at(0.0), model.amplitude_at(1.0), model.amplitude)
    if worst <= top:
        return model
    k = top / worst
    return CompressionModel(model.amplitude * k, model.length_mm, model.tilt * k, model.windows, model.lines,
                            model.residual, model.edge_mm)  # fmt: skip


def text_excesses(model: CompressionModel, width_mm: float, edge_mm: float) -> np.ndarray:
    """Лишняя длина текста ``∫(1/c − 1)`` от края текста до внешнего края на узлах высоты страницы, мм.

    Она сдвигает край текста к корешку; растяжение пустого поля у корешка край текста не двигает (поле уходит за
    линию корешка).

    Args:
        model: Модель.
        width_mm: Расстояние от корешка до внешнего края страницы, мм.
        edge_mm: Отступ края текста, мм.

    Returns:
        float[Y_NODES] — лишняя длина на долях высоты ``linspace(0, 1, Y_NODES)``, мм.
    """
    u = np.arange(min(edge_mm, width_mm), width_mm + GRID_MM, GRID_MM)
    return np.array([float(np.trapezoid(1.0 / model.scale(u, y) - 1.0, u)) for y in np.linspace(0.0, 1.0, Y_NODES)])


def text_excess(model: CompressionModel, width_mm: float, edge_mm: float) -> float:
    """Лишняя длина текста на худшей высоте страницы, мм (``text_excesses``)."""
    return float(text_excesses(model, width_mm, edge_mm).max())


def squeeze_need(model: CompressionModel, width_mm: float, edge_mm: float, budget: float) -> tuple[float, float]:
    """Сдвиг края текста и наибольшая длина, которую гасит тело страницы.

    Сдвиг один на всю высоту — не больше запаса поля и не больше лишней длины на наименее сжатой высоте; остаток на
    каждой высоте гасит тело.

    Args:
        model: Модель.
        width_mm: Расстояние от корешка до внешнего края, мм.
        edge_mm: Отступ края текста, мм.
        budget: Запас поля у корешка, мм.

    Returns:
        (сдвиг края, мм; наибольшая длина под сжатие тела, мм).
    """
    excesses = text_excesses(model, width_mm, edge_mm)
    shift = min(budget, float(excesses.min()))
    return shift, float((excesses - shift).max())


def scaled(model: CompressionModel, k: float) -> CompressionModel:
    """Модель с амплитудой и наклоном, умноженными на ``k``."""
    return CompressionModel(model.amplitude * k, model.length_mm, model.tilt * k, model.windows, model.lines,
                            model.residual, model.edge_mm)  # fmt: skip


def plan_side(model: CompressionModel, width_mm: float, edge_mm: float, limits: StretchLimits,
              outer_mm: float | None = None) -> SidePlan:  # fmt: skip
    """Растяжение страницы: урезание модели и раздел лишней длины между полем у корешка и телом страницы.

    Порядок: амплитуда урезается до ``max_stretch``; лишняя длина текста гасится сначала пустым полем у корешка
    (отступ минус ``edge_safety_mm``), остаток — равномерным сжатием тела не больше ``max_body_squeeze``; если и этого
    мало, амплитуда уменьшается (деление пополам по множителю), пока лишняя длина не влезет.

    Args:
        model: Модель сжатия.
        width_mm: Расстояние от корешка до внешнего края страницы, мм.
        edge_mm: Отступ края текста от корешка, мм.
        limits: Пределы.
        outer_mm: Внешний край текста от корешка, мм (медиана строк), или None — внешний край страницы.

    Returns:
        План стороны.
    """
    # Дальше внешнего края текста карта тождественна: лишняя длина и сжатие тела считаются до него.
    span = min(outer_mm, width_mm) if outer_mm is not None and outer_mm > edge_mm else width_mm
    model = capped(model, limits)
    measured = text_excess(model, span, edge_mm)
    budget = max(0.0, edge_mm - limits.edge_safety_mm)
    body = max(span - edge_mm, 1.0)
    allowed = limits.max_body_squeeze * body
    if squeeze_need(model, span, edge_mm, budget)[1] > allowed:
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if squeeze_need(scaled(model, mid), span, edge_mm, budget)[1] <= allowed else (lo, mid)
        model = scaled(model, lo)
    shift, over = squeeze_need(model, span, edge_mm, budget)
    applied = model.amplitude >= limits.min_amplitude and measured >= limits.min_excess_mm and span > edge_mm
    return SidePlan(model, width_mm, edge_mm, text_excess(model, span, edge_mm), over / body if applied else 0.0,
                    applied, shift if applied else 0.0, span)  # fmt: skip


def output_of_source(plan: SidePlan, u_mm: np.ndarray, y_frac: float) -> np.ndarray:
    """Расстояние от корешка на итоге для точки источника: ``t(u) = u − (F(o) − F(min(u, o))) + (Eₜ − S)·h(u)``.

    ``F(u) = ∫₀ᵘ (1/c − 1)`` — набранная к ``u`` лишняя длина, ``o`` — внешний край текста (``outer_mm``),
    ``Eₜ = F(o) − F(e)`` — лишняя длина текста на этой высоте, ``S`` — сдвиг края текста (``shift_mm``, один на всю
    высоту), ``h`` — 1 до края текста ``e`` и линейно до 0 у ``o``: остаток ``Eₜ − S`` тело между краями текста гасит
    равномерным сжатием. Край текста встаёт на ``e − S`` на любой высоте, ``t(u) = u`` от ``o`` наружу — внешняя
    сторона блока и край страницы на месте.

    Args:
        plan: План стороны.
        u_mm: Сетка расстояний источника от корешка, мм (возрастает от 0 до ``width_mm``).
        y_frac: Доля высоты страницы.

    Returns:
        Расстояния на итоге, мм (монотонно возрастают).
    """
    u_mm = np.asarray(u_mm, np.float64)
    if not plan.applied:
        return u_mm.copy()
    outer = plan.outer_mm if plan.outer_mm > plan.edge_mm else plan.width_mm
    gain = 1.0 / plan.model.scale(u_mm, y_frac) - 1.0
    f = np.concatenate([[0.0], np.cumsum((gain[1:] + gain[:-1]) / 2 * np.diff(u_mm))])
    f_outer = float(np.interp(outer, u_mm, f))
    g = np.interp(np.minimum(u_mm, outer), u_mm, f)
    over = max(0.0, f_outer - float(np.interp(plan.edge_mm, u_mm, f)) - plan.shift_mm)
    h = np.clip((outer - u_mm) / max(outer - plan.edge_mm, 1e-6), 0.0, 1.0)
    return u_mm - (f_outer - g) + over * h


def side_displacement(plan: SidePlan, layout: Layout, left: bool, px_per_mm: float) -> np.ndarray:
    """Смещение ``dx`` страницы на холсте: пиксель итога ``x`` берёт источник ``x + dx``.

    Args:
        plan: План стороны.
        layout: Раскладка холста.
        left: Левая страница (корешок справа от неё).
        px_per_mm: Плотность, px/мм.

    Returns:
        float32 ``(H, W)`` — смещение по x в пикселях; вне страницы — 0.
    """
    dx = np.zeros((layout.height, layout.width), np.float32)
    if not plan.applied:
        return dx
    rect = layout.rects[0 if left else 1]
    spine = layout.spine_x
    cols = np.arange(layout.width, dtype=np.float64)
    inside = (cols < spine) if left else (cols >= spine)
    t_mm = np.abs(cols - spine) / px_per_mm
    u_grid = np.arange(0.0, plan.width_mm + GRID_MM, GRID_MM)
    nodes = np.linspace(0.0, 1.0, Y_NODES)
    rows = np.arange(layout.height, dtype=np.float64)
    y_frac = np.clip((rows - rect.y0) / max(rect.height, 1.0), 0.0, 1.0)
    shifts = []
    for node in nodes:
        t_grid = output_of_source(plan, u_grid, float(node))
        # Обратная карта: расстояние источника для расстояния итога (t монотонна по u).
        u_of_t = np.interp(t_mm, t_grid, u_grid, left=np.nan, right=np.nan)
        shift = (u_of_t - t_mm) * px_per_mm * (1.0 if not left else -1.0)
        # За внешним краем страницы и в отрицательных t (не бывает) — без смещения.
        shifts.append(np.where(inside & np.isfinite(shift), shift, 0.0))
    shifts = np.array(shifts)
    # Линейная интерполяция между рядами-узлами по высоте.
    pos = y_frac * (Y_NODES - 1)
    lo = np.clip(np.floor(pos).astype(int), 0, Y_NODES - 2)
    w = (pos - lo)[:, None]
    dx[:] = ((1 - w) * shifts[lo] + w * shifts[lo + 1]).astype(np.float32)
    return dx


def stretch_displacement(plans: tuple[SidePlan, SidePlan], layout: Layout, px_per_mm: float) -> np.ndarray:
    """Смещение ``dx`` растяжения обеих страниц на холсте.

    Args:
        plans: Планы левой и правой страницы.
        layout: Раскладка.
        px_per_mm: Плотность, px/мм.

    Returns:
        float32 ``(H, W)``.
    """
    return side_displacement(plans[0], layout, True, px_per_mm) + side_displacement(plans[1], layout, False, px_per_mm)


def compress_image(image: np.ndarray, spine_x: float, models: tuple[CompressionModel, CompressionModel],
                   px_per_mm: float) -> np.ndarray:  # fmt: skip
    """Синтетическое сжатие у корешка (проверка мер): строки у корешка сжимаются по модели, край текста на месте.

    Пиксель на расстоянии ``u ≥ e`` от корешка берётся из исходника на расстоянии ``e + ∫ₑᵘ 1/c``: отрезок у края
    текста короче в ``c`` раз, поле у корешка (``u < e``) не трогается, внешний край уезжает наружу на лишнюю длину
    (для синтетики это неважно). ``e`` — ``edge_mm`` модели стороны.

    Args:
        image: Выпрямленный холст (серый или BGR).
        spine_x: X линии корешка, px.
        models: Модели сжатия левой и правой страницы (без наклона по высоте).
        px_per_mm: Плотность, px/мм.

    Returns:
        Сжатый холст того же размера.
    """
    height, width = image.shape[:2]
    cols = np.arange(width, dtype=np.float64)
    u_mm = np.abs(cols - spine_x) / px_per_mm
    src = cols.copy()
    for model, inside in zip(models, (cols < spine_x, cols >= spine_x)):
        grid = np.arange(0.0, u_mm.max() + GRID_MM, GRID_MM)
        inv = np.where(grid < model.edge_mm, 1.0, 1.0 / model.scale(grid))
        t_grid = np.concatenate([[0.0], np.cumsum((inv[1:] + inv[:-1]) / 2 * np.diff(grid))])
        t_mm = np.interp(u_mm, grid, t_grid)
        src = np.where(inside, spine_x + np.sign(cols - spine_x) * t_mm * px_per_mm, src)
    map_x = np.repeat(src[None, :].astype(np.float32), height, axis=0)
    map_y = np.repeat(np.arange(height, dtype=np.float32)[:, None], width, axis=1)
    return cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


class StretchMode(StrEnum):
    """Куда уходит лишняя длина разжатых строк."""

    SHIFT = "shift"  # ширина страницы прежняя: край текста сдвигается к корешку, остаток гасит тело (прогоны final*)
    WIDEN = "widen"  # страница шире со стороны корешка; растягивается только зона текста, поле у корешка — нет


@dataclass(frozen=True)
class WidenPlan:
    """Растяжение страницы с расширением со стороны корешка: растягивается только зона текста.

    Поле у корешка без букв не трогается (складка и разрез на месте, клина нет). Текст от начала зоны текста до
    внешнего края разжимается по модели, а разная по высоте прибавка длины строки гасится общим горизонтальным
    масштабом строки ``k(y)``: обе выровненные стороны блока остаются прямыми (решение пользователя 2026-10-25 после
    отсмотра: растяжение поля давало клин у корешка и перерастягивало концы строк, лежащие ближе медианного края).

    Поля:
        model: Модель сжатия (``edge_mm`` — медианный край текста стороны, начало спада).
        width_mm: Расстояние от линии корешка до внешнего края страницы, мм.
        edge_mm: Медианный край текста от корешка, мм.
        outer_mm: Внешний край текста от корешка, мм.
        widen_mm: Расширение страницы ``W`` — медиана прибавки строки по высотам со строками, мм (0 — не применено).
        excesses_mm: Прибавка строки ``Eₜ(y)`` на узлах высоты ``linspace(0, 1, Y_NODES)``, мм.
        applied: Растяжение применяется.
        amp_cap: Предел амплитуды на любой высоте ``1 − 1/max_stretch``: обрезается только там, где модель выше, —
            масштабирование всей модели (``capped``) при крутом наклоне урезало бы и середину, и низ страницы.
        text_starts_mm: Начало зоны текста ``e₀(y)`` на узлах высоты — край строк на этой высоте (скользящая медиана
            краёв соседних строк): модель отсчитывается от него, внутренняя сторона блока остаётся на месте по
            построению, а концы строк ближе медианного края растягиваются как текст, а не как поле.
        y_range: Доли высоты, занятые строками стороны: по ним берётся медиана прибавки (края сплайна без строк — нет).
    """

    model: CompressionModel
    width_mm: float
    edge_mm: float
    outer_mm: float
    widen_mm: float
    excesses_mm: tuple[float, ...]
    applied: bool
    amp_cap: float = 1.0
    text_starts_mm: tuple[float, ...] = ()
    y_range: tuple[float, float] = (0.0, 1.0)

    def text_start(self, y_frac: float) -> float:
        """Начало зоны текста на доле высоты, мм (без заданных краёв строк — медианный край)."""
        if not self.text_starts_mm:
            return self.edge_mm
        return float(np.interp(y_frac, np.linspace(0.0, 1.0, Y_NODES), self.text_starts_mm))

    def line_scale(self, y_frac: float) -> float:
        """Общий масштаб строки ``k(y) = (o − e₀ + W) / (o − e₀ + Eₜ(y))`` на доле высоты (1 — не применено)."""
        if not self.applied:
            return 1.0
        body = self.outer_mm - self.text_start(y_frac)
        excess = float(np.interp(y_frac, np.linspace(0.0, 1.0, Y_NODES), self.excesses_mm))
        return (body + self.widen_mm) / (body + excess)

    def scale_range(self) -> tuple[float, float]:
        """Наименьший и наибольший масштаб строки на высотах со строками."""
        ks = [self.line_scale(float(y)) for y in np.linspace(self.y_range[0], self.y_range[1], Y_NODES)]
        return min(ks), max(ks)


# Спад ближе этого к верхнему пределу подгонки (``profile.LENGTH_MM``) — модель не применяется, мм.
LENGTH_AT_BOUND_MM = 0.5
# Лишняя длина для порога применения считается в этой полосе от края текста, мм: слабый длинный спад (01/IMG_0987 R,
# 0.07 при L 30 мм) даёт в ней 0.6 мм, контроль с коротким спадом (0.38 при L 1.5 мм) — ~0.7 мм; порог 0.8 мм.
NEAR_EXCESS_MM = 10.0


def text_scale(model: CompressionModel, u_mm: np.ndarray, y_frac: float, amp_cap: float = 1.0,
               text_start_mm: float | None = None) -> np.ndarray:  # fmt: skip
    """Сжатие ``c`` в зоне текста: ``1 − B(y)·exp(−max(u − e, 0)/L)`` при ``u ≥ e₀``, ближе к корешку — поле, ``c = 1``.

    Спад отсчитывается от медианного края ``e`` (``model.edge_mm``) — как в подгонке модели (``fit_model``,
    ``shape.side_data``); в полосе ``[e₀, e)`` между нижней огибающей краёв строк и медианой сжатие постоянно и равно
    сжатию у края (висячие знаки, последние буквы строк, кончающихся ближе к корешку). Отсчёт спада от огибающей
    недорастягивал основную часть строк: мера у края на сжатых сторонах 0.998 → 0.980.

    Args:
        model: Модель (``edge_mm`` — медианный край текста, начало спада, если ``text_start_mm`` не задан).
        u_mm: Расстояния от линии корешка, мм.
        y_frac: Доля высоты страницы.
        amp_cap: Предел амплитуды (растяжение ``1/c`` не больше ``1/(1 − amp_cap)``).
        text_start_mm: Начало зоны текста ``e₀`` на этой высоте (край строк) или None — медианный край.

    Returns:
        ``c`` той же формы, что ``u_mm``.
    """
    u = np.asarray(u_mm, np.float64)
    start = model.edge_mm if text_start_mm is None else text_start_mm
    amp = np.minimum(model.amplitude_at(y_frac), amp_cap)
    return np.where(u >= start, 1.0 - amp * np.exp(-np.maximum(u - model.edge_mm, 0.0) / model.length_mm), 1.0)


def widen_excess(model: CompressionModel, start_mm: float, outer_mm: float, y_frac: float, amp_cap: float) -> float:
    """Прибавка строки ``∫ₑ₀ᵒ (1/c − 1)`` на доле высоты, мм (амплитуда не выше ``amp_cap``).

    Args:
        model: Модель (``edge_mm`` — медианный край).
        start_mm: Начало зоны текста ``e₀``, мм.
        outer_mm: Конец интегрирования, мм.
        y_frac: Доля высоты.
        amp_cap: Предел амплитуды.

    Returns:
        Прибавка, мм.
    """
    u = np.arange(start_mm, max(outer_mm, start_mm) + GRID_MM, GRID_MM)
    if len(u) < 2:
        return 0.0
    return float(np.trapezoid(1.0 / text_scale(model, u, y_frac, amp_cap, start_mm) - 1.0, u))


def plan_widen(model: CompressionModel, width_mm: float, edge_mm: float, limits: StretchLimits,
               outer_mm: float | None = None, text_starts_mm: tuple[float, ...] = (),
               y_range: tuple[float, float] = (0.0, 1.0)) -> WidenPlan:  # fmt: skip
    """План растяжения с расширением: прибавка строки по высоте, расширение — медиана прибавки, порог применения.

    Args:
        model: Модель сжатия.
        width_mm: Расстояние от корешка до внешнего края страницы, мм.
        edge_mm: Медианный край текста от корешка, мм.
        limits: Пределы (``max_stretch``, ``min_amplitude``, ``min_excess_mm``; запас края и сжатие тела не нужны).
        outer_mm: Внешний край текста от корешка, мм, или None — внешний край страницы.
        text_starts_mm: Начало зоны текста на узлах высоты ``linspace(0, 1, Y_NODES)``, мм, или пусто — медианный
            край на любой высоте.
        y_range: Доли высоты, занятые строками стороны.

    Returns:
        План стороны.
    """
    outer = min(outer_mm, width_mm) if outer_mm is not None and outer_mm > edge_mm else width_mm
    model = replace(model, edge_mm=edge_mm)
    cap = 1.0 - 1.0 / limits.max_stretch
    nodes = np.linspace(0.0, 1.0, Y_NODES)
    starts = tuple(float(v) for v in text_starts_mm) if len(text_starts_mm) == Y_NODES else (edge_mm,) * Y_NODES
    excesses = tuple(widen_excess(model, e0, outer, float(y), cap) for y, e0 in zip(nodes, starts))
    # Расширение — медиана прибавки по высотам со строками: масштаб строки и сжимает, и растягивает поровну
    # (выбор пользователя 2026-10-25; замер excess_spread — ±1.7 % медиана, ±4 % p90 на 834 сжатых сторонах).
    lo, hi = y_range
    inside = [e for y, e in zip(nodes, excesses) if lo - 1e-9 <= y <= hi + 1e-9]
    if len(inside) < 2:
        inside = list(np.interp(np.linspace(lo, hi, Y_NODES), nodes, excesses))
    widen = float(np.median(inside))
    worst_amp = max(float(model.amplitude_at(0.0)), float(model.amplitude_at(1.0)))
    # Порог применения — по лишней длине у края текста: слабая амплитуда на длинном спаде (L у предела 30 мм)
    # набирает по всей строке больше порога без всякого видимого сжатия (01/IMG_0987 R: 0.07 → 2.19 мм).
    near = 0.0
    for y, e0 in zip(nodes, starts):
        if lo - 1e-9 <= y <= hi + 1e-9:
            near = max(near, widen_excess(model, e0, min(outer, e0 + NEAR_EXCESS_MM), float(y), cap))
    # Спад у верхнего предела подгонки — ровный сдвиг меры по всей строке, а не сжатие у корешка: шаг shape — ни одной
    # такой стороны среди 834 сжатых (пояса 1–3), 34 % у сторон без сжатия (01/IMG_0987 R, 03/IMG_0089 R).
    decays = model.length_mm < LENGTH_MM[1] - LENGTH_AT_BOUND_MM
    applied = near >= limits.min_excess_mm and worst_amp >= limits.min_amplitude and outer > edge_mm and decays
    return WidenPlan(model, width_mm, edge_mm, outer, widen if applied else 0.0, excesses, applied, cap, starts,
                     (lo, hi))  # fmt: skip


def widen_output_of_source(plan: WidenPlan, u_mm: np.ndarray, y_frac: float) -> np.ndarray:
    """Расстояние от корешка на итоге для точки источника при расширении страницы.

    - Поле у корешка ``u < e₀``: ``t = u`` — без растяжения.
    - Зона текста ``e₀ ≤ u ≤ o``: ``t = e₀ + k(y)·((u − e₀) + Fₑ₀(u))``, ``Fₑ₀(u) = ∫ₑ₀ᵘ (1/c − 1)`` — строка
      разжимается по модели и масштабируется общим ``k(y)`` так, что её конец встаёт на ``o + W``.
    - Дальше внешнего края текста: ``t = u + W`` — внешняя сторона блока и край страницы сдвинуты на ``W`` на любой
      высоте.

    Монотонна (``k > 0``, ``1/c ≥ 1``); излом наклона только в ``e₀`` — на чистой бумаге.

    Args:
        plan: План стороны.
        u_mm: Сетка расстояний источника от корешка, мм (возрастает от 0).
        y_frac: Доля высоты страницы.

    Returns:
        Расстояния на итоге, мм (монотонно возрастают).
    """
    u = np.asarray(u_mm, np.float64)
    if not plan.applied:
        return u.copy()
    e0, o, w = plan.text_start(y_frac), plan.outer_mm, plan.widen_mm
    gain = 1.0 / text_scale(plan.model, u, y_frac, plan.amp_cap, e0) - 1.0
    f = np.concatenate([[0.0], np.cumsum((gain[1:] + gain[:-1]) / 2 * np.diff(u))])
    f_start, f_outer = float(np.interp(e0, u, f)), float(np.interp(o, u, f))
    # Общий масштаб строки: разжатая строка этой высоты (o − e₀ + Eₜ) встаёт ровно на o − e₀ + W.
    k = (o - e0 + w) / (o - e0 + (f_outer - f_start))
    text = e0 + k * ((u - e0) + (np.interp(np.clip(u, e0, o), u, f) - f_start))
    return np.where(u < e0, u, np.where(u > o, u + w, text))


def widen_shift_px(plan: WidenPlan, px_per_mm: float) -> float:
    """Расширение страницы в пикселях холста (0 — не применено)."""
    return plan.widen_mm * px_per_mm if plan.applied else 0.0


def widened_layout(layout: Layout, plans: tuple[WidenPlan, WidenPlan], px_per_mm: float) -> Layout:
    """Раскладка холста итога с расширенными страницами: левая растёт к корешку, правая сдвигается и растёт.

    Внешний край левой страницы на месте, линия корешка сдвинута вправо на ``W_L``, правая страница целиком — на
    ``W_L``, её внешний край — на ``W_L + W_R``. Высота и верх страниц не меняются.

    Args:
        layout: Раскладка Кунса.
        plans: Планы левой и правой страницы.
        px_per_mm: Плотность, px/мм.

    Returns:
        Раскладка холста итога (координаты — нового холста).
    """
    wl, wr = (widen_shift_px(p, px_per_mm) for p in plans)
    left, right = layout.rects
    spine = layout.spine_x + wl
    rects = (PageRect(left.x0, left.y0, spine, left.y1), PageRect(spine, right.y0, right.x1 + wl + wr, right.y1))
    return Layout(int(np.ceil(layout.width + wl + wr)), layout.height, rects, spine)


def widen_side_source(plan: WidenPlan, layout: Layout, wide: Layout, left: bool, px_per_mm: float) -> np.ndarray:
    """X источника (холст Кунса) для пикселей своей стороны расширенного холста; NaN — пиксель другой стороны.

    Args:
        plan: План стороны.
        layout: Раскладка Кунса.
        wide: Расширенная раскладка (:func:`widened_layout`).
        left: Левая страница.
        px_per_mm: Плотность, px/мм.

    Returns:
        float64 ``(H, W')`` — X источника.
    """
    cols = np.arange(wide.width, dtype=np.float64)
    inside = (cols < wide.spine_x) if left else (cols >= wide.spine_x)
    t_mm = np.abs(cols - wide.spine_x) / px_per_mm
    u_grid = np.arange(0.0, plan.width_mm + GRID_MM, GRID_MM)
    rect = layout.rects[0 if left else 1]
    rows = np.arange(layout.height, dtype=np.float64)
    y_frac = np.clip((rows - rect.y0) / max(rect.height, 1.0), 0.0, 1.0)
    sources = []
    for node in np.linspace(0.0, 1.0, Y_NODES):
        t_grid = widen_output_of_source(plan, u_grid, float(node))
        # Обратная карта; за краем сетки (поле холста за страницей) — сдвиг на расширение.
        u = np.where(t_mm <= t_grid[-1], np.interp(t_mm, t_grid, u_grid), t_mm - (t_grid[-1] - u_grid[-1]))
        sources.append(layout.spine_x - u * px_per_mm if left else layout.spine_x + u * px_per_mm)
    sources = np.array(sources)
    pos = y_frac * (Y_NODES - 1)
    lo = np.clip(np.floor(pos).astype(int), 0, Y_NODES - 2)
    w = (pos - lo)[:, None]
    out = (1 - w) * sources[lo] + w * sources[lo + 1]
    return np.where(inside[None, :], out, np.nan)


def widen_displacement(
    plans: tuple[WidenPlan, WidenPlan], layout: Layout, px_per_mm: float
) -> tuple[Layout, np.ndarray]:
    """Смещение растяжения с расширением: пиксель ``X'`` расширенного холста берёт точку ``X' + s`` холста Кунса.

    Args:
        plans: Планы левой и правой страницы.
        layout: Раскладка Кунса.
        px_per_mm: Плотность, px/мм.

    Returns:
        (расширенная раскладка; float32 ``(H, W')`` — смещение по x в координаты холста Кунса).
    """
    wide = widened_layout(layout, plans, px_per_mm)
    left = widen_side_source(plans[0], layout, wide, True, px_per_mm)
    right = widen_side_source(plans[1], layout, wide, False, px_per_mm)
    source = np.where(np.isfinite(left), left, right)
    return wide, (source - np.arange(wide.width, dtype=np.float64)[None, :]).astype(np.float32)
