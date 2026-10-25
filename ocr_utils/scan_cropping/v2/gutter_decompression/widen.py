"""Растяжение страницы у корешка с расширением холста: растягивается только зона текста (поле у корешка тождественно), разная по высоте прибавка длины строки гасится общим горизонтальным масштабом строки, обе выровненные стороны блока остаются прямыми; смещение в формате поля ``text_field`` для сложения с ним."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from ocr_utils.scan_cropping.v2.gutter_decompression.fit import LENGTH_MM
from ocr_utils.scan_cropping.v2.gutter_decompression.model import CompressionModel
from ocr_utils.scan_cropping.v2.rectify.coons import Layout, PageRect

# Шаг сетки интегрирования по расстоянию от корешка, мм.
GRID_MM = 0.05
# Рядов сетки по высоте страницы (между ними — линейно).
Y_NODES = 9
# Начало зоны текста у корешка: скользящий минимум краёв строк по этому числу соседних строк; строки ближе к корешку,
# чем медиана стороны минус TEXT_START_SLACK_MM (сор у складки, шапки), в огибающую не идут, мм.
TEXT_START_WINDOW = 5
TEXT_START_SLACK_MM = 2.0
# Внешний край текста — этот перцентиль внешних концов строк (концы абзацев его не задают).
OUTER_QUANTILE = 0.9
# Лишняя длина для порога применения считается в этой полосе от начала зоны текста, мм.
NEAR_EXCESS_MM = 10.0
# Спад ближе этого к верхнему пределу подгонки (``fit.LENGTH_MM``) — ровный сдвиг меры по всей строке, а не сжатие у
# корешка: модель не применяется, мм (стенд, шаг shape: ни одной такой стороны среди 834 сжатых, 34 % у несжатых).
LENGTH_AT_BOUND_MM = 0.5


@dataclass(frozen=True)
class WidenLimits:
    """Пределы и порог применения растяжения.

    Поля:
        max_stretch: Наибольшее местное растяжение ``1/c``. 2.0 — амплитуда до 0.5: предел 1.6 на 30 кадрах ХиП 1991
            недорастягивал (буквы у края после 0.92 против 0.97), перерастяжения без предела нет (ни одной стороны
            с буквами у края > 1.1).
        min_amplitude: Наибольшая по высоте амплитуда меньше этой — не применяется.
        min_excess_mm: Лишняя длина строки в первых :data:`NEAR_EXCESS_MM` мм зоны текста (наибольшая по высотам
            со строками) меньше этой — не применяется, мм. 0.5: пропускает короткие сильные спады (ХиП 1991/07
            IMG_0133 L — последняя буква 0.83, 0.65 мм; 05/IMG_0325 L — 0.875, 0.66 мм); стороны контроля — 0.37–0.46 мм.
    """

    max_stretch: float = 2.0
    min_amplitude: float = 0.04
    min_excess_mm: float = 0.5


@dataclass(frozen=True)
class WidenPlan:
    """Растяжение страницы с расширением со стороны корешка: растягивается только зона текста.

    Поле у корешка без букв не трогается (складка и разрез на месте, клина нет). Текст от начала зоны текста ``e₀(y)``
    до внешнего края ``o`` разжимается по модели, а разная по высоте прибавка длины строки ``Eₜ(y)`` гасится общим
    масштабом строки ``k(y) = (o − e₀ + W) / (o − e₀ + Eₜ(y))``: обе выровненные стороны блока остаются прямыми.

    Поля:
        model: Модель сжатия (``origin_mm`` — начало спада, наименьшее начало зоны текста стороны).
        width_mm: Расстояние от линии корешка до внешнего края страницы, мм.
        outer_mm: Внешний край текста от корешка, мм.
        widen_mm: Расширение страницы ``W`` — медиана прибавки строки по высотам со строками, мм (0 — не применено).
        excesses_mm: Прибавка строки ``Eₜ(y)`` на узлах высоты ``linspace(0, 1, Y_NODES)``, мм.
        near_mm: Наибольшая по высотам со строками лишняя длина в первых :data:`NEAR_EXCESS_MM` мм зоны текста, мм.
        applied: Растяжение применяется.
        amp_cap: Предел амплитуды на любой высоте ``1 − 1/max_stretch`` (обрезается только там, где модель выше).
        text_starts_mm: Начало зоны текста ``e₀(y)`` на узлах высоты, мм.
        y_range: Доли высоты, занятые строками стороны.
    """

    model: CompressionModel
    width_mm: float
    outer_mm: float
    widen_mm: float = 0.0
    excesses_mm: tuple[float, ...] = (0.0,) * Y_NODES
    near_mm: float = 0.0
    applied: bool = False
    amp_cap: float = 1.0
    text_starts_mm: tuple[float, ...] = ()
    y_range: tuple[float, float] = (0.0, 1.0)

    def text_start(self, y_frac: float) -> float:
        """Начало зоны текста на доле высоты, мм (без заданных краёв строк — начало спада модели)."""
        if not self.text_starts_mm:
            return self.model.origin_mm
        return float(np.interp(y_frac, np.linspace(0.0, 1.0, Y_NODES), self.text_starts_mm))

    def line_scale(self, y_frac: float) -> float:
        """Общий масштаб строки ``k(y)`` на доле высоты (1 — не применено)."""
        if not self.applied:
            return 1.0
        body = self.outer_mm - self.text_start(y_frac)
        excess = float(np.interp(y_frac, np.linspace(0.0, 1.0, Y_NODES), self.excesses_mm))
        return (body + self.widen_mm) / (body + excess)


def text_zone(letters: pd.DataFrame, rect: PageRect) -> tuple[tuple[float, ...], tuple[float, float], float | None]:
    """Начало зоны текста у корешка на узлах высоты, высоты со строками и внешний край текста.

    Начало зоны на высоте — нижняя огибающая краёв строк: скользящий минимум по :data:`TEXT_START_WINDOW` соседним
    строкам. Края у корешка чередуются (строки, кончающиеся дефисом или точкой, ближе к корешку, чем кончающиеся
    буквой); сжатие зависит от места на бумаге, поэтому промежуток между висячими знаками и буквами растягивается
    вместе с текстом. Одиночные «строки» ближе :data:`TEXT_START_SLACK_MM` к корешку, чем медиана стороны (сор у
    складки, шапки), в огибающую не идут: с ними пустое поле растягивалось бы как текст.

    Args:
        letters: Буквы стороны (``letters.page_letters``: ``line``, ``y``, ``edge_mm``, ``outer_mm``).
        rect: Прямоугольник страницы на холсте.

    Returns:
        (начало зоны текста на узлах ``linspace(0, 1, Y_NODES)``, мм; доли высоты первой и последней строки; внешний
        край текста от корешка, мм, или None — строк нет).
    """
    if len(letters) == 0:
        return (), (0.0, 1.0), None
    lines = letters.groupby("line").agg(edge=("edge_mm", "first"), outer=("outer_mm", "first"), y=("y", "first"))
    lines = lines.sort_values("y")
    median = float(lines.edge.median())
    ys = np.clip((lines.y.to_numpy() - rect.y0) / max(rect.height, 1.0), 0.0, 1.0)
    kept = lines[lines.edge >= median - TEXT_START_SLACK_MM]
    envelope = kept.edge.rolling(TEXT_START_WINDOW, center=True, min_periods=1).min().clip(upper=median)
    kept_ys = np.clip((kept.y.to_numpy() - rect.y0) / max(rect.height, 1.0), 0.0, 1.0)
    starts = np.interp(np.linspace(0.0, 1.0, Y_NODES), kept_ys, envelope.to_numpy())
    outer = float(lines.outer.quantile(OUTER_QUANTILE))
    return tuple(float(v) for v in starts), (float(ys.min()), float(ys.max())), outer


def text_scale(model: CompressionModel, u_mm: np.ndarray, y_frac: float, amp_cap: float,
               text_start_mm: float) -> np.ndarray:  # fmt: skip
    """Сжатие ``c`` в зоне текста: ``1 − B(y)·exp(−max(u − e, 0)/L)`` при ``u ≥ e₀``, ближе к корешку — поле, ``c = 1``.

    Args:
        model: Модель (``origin_mm`` — начало спада ``e``).
        u_mm: Расстояния от линии корешка, мм.
        y_frac: Доля высоты страницы.
        amp_cap: Предел амплитуды (растяжение ``1/c`` не больше ``1/(1 − amp_cap)``).
        text_start_mm: Начало зоны текста ``e₀`` на этой высоте, мм.

    Returns:
        ``c`` той же формы, что ``u_mm``.
    """
    u = np.asarray(u_mm, np.float64)
    amp = np.minimum(model.amplitude_at(y_frac), amp_cap)
    decay = np.exp(-np.maximum(u - model.origin_mm, 0.0) / model.length_mm)
    return np.where(u >= text_start_mm, 1.0 - amp * decay, 1.0)


def widen_excess(model: CompressionModel, start_mm: float, end_mm: float, y_frac: float, amp_cap: float) -> float:
    """Прибавка строки ``∫ₑ₀ᵉⁿᵈ (1/c − 1)`` на доле высоты, мм.

    Args:
        model: Модель.
        start_mm: Начало зоны текста ``e₀``, мм.
        end_mm: Конец интегрирования, мм.
        y_frac: Доля высоты.
        amp_cap: Предел амплитуды.

    Returns:
        Прибавка, мм.
    """
    u = np.arange(start_mm, max(end_mm, start_mm) + GRID_MM, GRID_MM)
    if len(u) < 2:
        return 0.0
    return float(np.trapezoid(1.0 / text_scale(model, u, y_frac, amp_cap, start_mm) - 1.0, u))


def plan_widen(model: CompressionModel, width_mm: float, outer_mm: float | None, limits: WidenLimits,
               text_starts_mm: tuple[float, ...], y_range: tuple[float, float]) -> WidenPlan:  # fmt: skip
    """План растяжения: прибавка строки по высоте, расширение — медиана прибавки, порог применения.

    Args:
        model: Модель сжатия.
        width_mm: Расстояние от корешка до внешнего края страницы, мм.
        outer_mm: Внешний край текста от корешка, мм, или None — внешний край страницы.
        limits: Пределы и порог.
        text_starts_mm: Начало зоны текста на узлах высоты, мм (пусто — начало спада модели).
        y_range: Доли высоты, занятые строками стороны.

    Returns:
        План стороны.
    """
    origin = model.origin_mm
    outer = min(outer_mm, width_mm) if outer_mm is not None and outer_mm > origin else width_mm
    if model.null:
        return WidenPlan(model, width_mm, outer)
    cap = 1.0 - 1.0 / limits.max_stretch
    nodes = np.linspace(0.0, 1.0, Y_NODES)
    starts = tuple(float(v) for v in text_starts_mm) if len(text_starts_mm) == Y_NODES else (origin,) * Y_NODES
    excesses = tuple(widen_excess(model, e0, outer, float(y), cap) for y, e0 in zip(nodes, starts))
    # Расширение — медиана прибавки по высотам со строками: масштаб строки и сжимает, и растягивает поровну
    # (стенд: ±1.7 % медиана, ±4 % p90 на 834 сжатых сторонах); края сплайна без строк не в счёт.
    lo, hi = y_range
    inside = [(float(y), e0, e) for y, e0, e in zip(nodes, starts, excesses) if lo - 1e-9 <= y <= hi + 1e-9]
    if len(inside) < 2:
        ys = np.linspace(lo, hi, Y_NODES)
        inside = [(float(y), float(np.interp(y, nodes, starts)), float(np.interp(y, nodes, excesses))) for y in ys]
    widen = float(np.median([e for _, _, e in inside]))
    worst_amp = float(np.max(model.amplitude_at(nodes)))
    # Порог — по лишней длине у начала зоны текста: слабая амплитуда на длинном спаде набирает по всей строке больше
    # порога без всякого видимого сжатия (ХиП 1991/01 IMG_0987 R: 0.07 → 2.19 мм).
    near = max(widen_excess(model, e0, min(outer, e0 + NEAR_EXCESS_MM), y, cap) for y, e0, _ in inside)
    decays = model.length_mm < LENGTH_MM[1] - LENGTH_AT_BOUND_MM
    applied = near >= limits.min_excess_mm and worst_amp >= limits.min_amplitude and outer > origin and decays
    return WidenPlan(model, width_mm, outer, widen if applied else 0.0, excesses, near, applied, cap, starts,
                     (lo, hi))  # fmt: skip


def widen_output_of_source(plan: WidenPlan, u_mm: np.ndarray, y_frac: float) -> np.ndarray:
    """Расстояние от корешка на итоге для точки источника при расширении страницы.

    - Поле у корешка ``u < e₀``: ``t = u`` — без растяжения.
    - Зона текста ``e₀ ≤ u ≤ o``: ``t = e₀ + k(y)·((u − e₀) + Fₑ₀(u))``, ``Fₑ₀(u) = ∫ₑ₀ᵘ (1/c − 1)`` — строка
      разжимается по модели и масштабируется общим ``k(y)`` так, что её конец встаёт на ``o + W``.
    - Дальше внешнего края текста: ``t = u + W`` — внешняя сторона блока и край страницы сдвинуты на ``W``.

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
        layout: Раскладка Кунса (разворот).
        plans: Планы левой и правой страницы.
        px_per_mm: Плотность, px/мм.

    Returns:
        Раскладка холста итога (координаты нового холста).
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
    # Между узлами высоты — линейно.
    pos = y_frac * (Y_NODES - 1)
    lo = np.clip(np.floor(pos).astype(int), 0, Y_NODES - 2)
    w = (pos - lo)[:, None]
    out = (1 - w) * sources[lo] + w * sources[lo + 1]
    return np.where(inside[None, :], out, np.nan)


def widen_displacement(plans: tuple[WidenPlan, WidenPlan], layout: Layout,
                       px_per_mm: float) -> tuple[Layout, np.ndarray]:  # fmt: skip
    """Смещение растяжения: пиксель ``X'`` расширенного холста берёт точку ``X' + s`` холста Кунса.

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


def add_wide_displacement(total: tuple[np.ndarray, np.ndarray], shift_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Сложение поля с растяжением на холсте другого размера: ``D'(X') = s(X') + D(X' + s(X'))``.

    Args:
        total: Накопленное поле ``(D_x, D_y)`` холста Кунса.
        shift_x: Смещение ``s`` по x (H, W') в координаты холста Кунса; по y смещения нет.

    Returns:
        Поле ``(D'_x, D'_y)`` размера ``s``.
    """
    h, w = shift_x.shape
    xs = np.arange(w, dtype=np.float32)[None, :] + shift_x
    ys = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    dx = shift_x + cv2.remap(total[0], xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dy = cv2.remap(total[1], xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return dx, dy


__all__ = ["WidenLimits", "WidenPlan", "add_wide_displacement", "plan_widen", "text_zone", "widen_displacement"]
