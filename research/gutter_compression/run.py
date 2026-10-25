"""Кадр с растяжением у корешка: выпрямление в два преобразования (боевой ``text_frame``), мера сжатия на его итоге, модель и растяжение, добавленное к полю (``D'(X) = s(X) + D(X + s(X))``), одна выборка из итога v2, меры «до | после» и лист; плюс синтетическое сжатие для проверки мер."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ocr_utils.scan_cropping.v2.pipeline import read_bgr
from ocr_utils.scan_cropping.v2.rectify.coons import Layout, rectify_image
from ocr_utils.scan_cropping.v2.rectify.measure import TextMeasures, text_measures
from ocr_utils.scan_cropping.v2.rectify.text_field import (
    CROP_MARGIN_MM,
    PageRegion,
    analyze_page,
    compose_maps,
    crop_box,
)
from ocr_utils.scan_cropping.v2.rectify.text_run import TextParams, TextResult, add_displacement, text_frame
from ocr_utils.units import mm_to_px_float
from research.gutter_compression.frames import FrameRef, frame_geometry
from research.gutter_compression.letter_fit import MIN_LETTERS, FitSource, fit_letters, letter_data, letter_summary
from research.gutter_compression.letter_profile import spread_letters
from research.gutter_compression.measure import Method, MeasureParams, Side, spread_windows
from research.gutter_compression.profile import MIN_LINES, CompressionModel, correct_position, fit_model, side_summary
from research.gutter_compression.shape import MIN_WINDOWS, HeightShape, fit_shape, side_data
from research.gutter_compression.sheets import before_after_sheet
from research.gutter_compression.stretch import (
    SidePlan,
    StretchLimits,
    Y_NODES,
    StretchMode,
    WidenPlan,
    compress_image,
    plan_side,
    plan_widen,
    stretch_displacement,
    widen_displacement,
)


@dataclass(frozen=True)
class StretchConfig:
    """Параметры растяжения кадра.

    Поля:
        method: Мера, по которой подгоняется модель.
        use_tilt: Амплитуда сжатия меняется по высоте страницы (наклон).
        limits: Пределы растяжения.
        measure: Параметры меры.
        text: Параметры второго преобразования (как у ``run_page_rectify_text.sh``: шаг поля и изгиб у корешка 3 мм).
        bias: Поправка на место в строке (``position_bias.csv`` команды ``outer``) или None — без поправки; окна до и
            после растяжения поправляются одинаково.
        mode: Куда уходит лишняя длина: сдвиг края текста и сжатие тела или расширение страницы.
        tilt_max: Предел модуля наклона амплитуды по высоте или None — ``profile.TILT_MAX`` (0.3).
        spline_penalty: Амплитуда по высоте — P-сплайн с этим штрафом (``shape.fit_shape``) или None — линия
            (``use_tilt``) либо константа. Только для режима ``widen``.
        fit_source: По чему подгоняется модель: окна 5 мм или ширина букв (только ``widen``). По буквам спад
            отсчитывается от начала зоны текста (наименьшее ``e₀`` стороны), а не от медианного края: полосы
            постоянного сжатия ``[e₀, e]`` нет.
    """

    method: Method = Method.ACF
    use_tilt: bool = False
    limits: StretchLimits = StretchLimits()
    measure: MeasureParams = MeasureParams()
    text: TextParams = TextParams(field_step_mm=3.0, gutter_smooth_mm=3.0)
    bias: pd.DataFrame | None = None
    mode: StretchMode = StretchMode.SHIFT
    tilt_max: float | None = None
    spline_penalty: float | None = None
    fit_source: FitSource = FitSource.WINDOWS


Plan = SidePlan | WidenPlan
# Начало зоны текста у корешка: скользящий минимум краёв строк по этому числу соседних строк; строки ближе к корешку,
# чем медиана стороны минус TEXT_START_SLACK_MM (сор у складки, шапки), в огибающую не идут, мм.
TEXT_START_WINDOW = 5
TEXT_START_SLACK_MM = 2.0


@dataclass(frozen=True)
class StretchResult:
    """Итог кадра.

    Поля:
        result: Выпрямление без растяжения (боевое).
        stretched: Итог с растяжением.
        plans: Растяжение левой и правой страницы.
        before: Окна строк до растяжения.
        after: Окна строк после.
        measures_before: Прямизна строк и сторон без растяжения (последний проход).
        measures_after: То же с растяжением (разбор итога заново).
        dx_max_mm: Наибольший |dx| растяжения, мм.
        wide: Раскладка холста итога с растяжением (в режиме ``widen`` — шире раскладки Кунса).
        letters_before: Буквы строк до растяжения (``letter_profile.spread_letters``).
        letters_after: Буквы строк после.
    """

    result: TextResult
    stretched: np.ndarray
    plans: tuple[Plan, Plan]
    before: pd.DataFrame
    after: pd.DataFrame
    measures_before: TextMeasures
    measures_after: TextMeasures
    dx_max_mm: float
    wide: Layout
    letters_before: pd.DataFrame
    letters_after: pd.DataFrame


def side_plans(table: pd.DataFrame, result: TextResult, px_per_mm: float, cfg: StretchConfig,
               letters: pd.DataFrame | None = None) -> tuple[Plan, Plan]:  # fmt: skip
    """Модели сжатия и планы растяжения обеих страниц по окнам или буквам строк итога (режим — ``cfg.mode``).

    Args:
        table: Окна строк итога без растяжения (зона текста, внешний край; модель — при ``cfg.fit_source`` WINDOWS).
        result: Выпрямление.
        px_per_mm: Плотность, px/мм.
        cfg: Параметры.
        letters: Буквы строк итога без растяжения (нужны при ``cfg.fit_source`` LETTERS).

    Returns:
        Планы левой и правой страницы.
    """
    layout = result.layout
    plans = []
    for side, rect in zip(Side, layout.rects):
        part = table[table.side == side.value] if len(table) else table
        width_mm = abs((rect.x0 if side is Side.LEFT else rect.x1) - layout.spine_x) / px_per_mm
        planner = plan_widen if cfg.mode is StretchMode.WIDEN else plan_side
        if len(part) == 0:
            plans.append(planner(CompressionModel(), width_mm, 0.0, cfg.limits))
            continue
        starts, y_range = text_zone(part, rect)
        if cfg.fit_source is FitSource.LETTERS:
            # Спад — от начала зоны текста: концы строк ближе медианного края растягиваются по модели, а не
            # постоянным сжатием у медианного края (07/IMG_0078 L: полоса 1.25–2.38 мм давала половину прибавки).
            origin = min(starts)
            own = letters[letters.side == side.value] if letters is not None and len(letters) else pd.DataFrame()
            model = letter_side_model(own, rect.y0, rect.height, px_per_mm, origin, cfg)
            outer = float(part.outer_mm.quantile(0.9)) if "outer_mm" in part else None
            plans.append(plan_widen(model, width_mm, origin, cfg.limits, outer, starts, y_range))
            continue
        model = side_model(part, rect.y0, rect.height, cfg)
        # Внешний край текста — 90-й перцентиль внешних концов строк (короткие строки концов абзацев его не задают).
        outer = float(part.outer_mm.quantile(0.9)) if "outer_mm" in part else None
        edge = float(part.edge_mm.median())
        if cfg.mode is StretchMode.WIDEN:
            plans.append(plan_widen(model, width_mm, edge, cfg.limits, outer, starts, y_range))
        else:
            plans.append(plan_side(model, width_mm, edge, cfg.limits, outer))
    return plans[0], plans[1]


def text_zone(part: pd.DataFrame, rect) -> tuple[tuple[float, ...], tuple[float, float]]:
    """Начало зоны текста у корешка на узлах высоты и высоты, занятые строками стороны.

    Начало зоны на высоте — нижняя огибающая краёв строк (``edge_mm`` окна — край своей строки): скользящий минимум по
    :data:`TEXT_START_WINDOW` соседним строкам. Края у корешка чередуются: строки, кончающиеся дефисом или точкой,
    ближе к корешку, чем кончающиеся буквой (01/IMG_1060 L: 2.46 и 3.07 мм); по медиане последние буквы строк
    с краем 2.46 мм оставались нерастянутыми. Сжатие зависит от места на бумаге, поэтому промежуток между висячими
    знаками и буквами растягивается вместе с текстом. Одиночные «строки» ближе :data:`TEXT_START_SLACK_MM` к корешку,
    чем медиана стороны (сор у складки, шапки), в огибающую не идут: с ними пустое поле растягивалось как текст.

    Args:
        part: Окна стороны.
        rect: Прямоугольник страницы на холсте (``y0``, ``height``).

    Returns:
        (начало зоны текста на узлах ``linspace(0, 1, Y_NODES)``, мм; доли высоты первой и последней строки).
    """
    lines = part.groupby("line").agg(edge=("edge_mm", "first"), y=("y", "median")).sort_values("y")
    median = float(lines.edge.median())
    ys = np.clip((lines.y.to_numpy() - rect.y0) / max(rect.height, 1.0), 0.0, 1.0)
    kept = lines[lines.edge >= median - TEXT_START_SLACK_MM]
    envelope = kept.edge.rolling(TEXT_START_WINDOW, center=True, min_periods=1).min().clip(upper=median)
    kept_ys = np.clip((kept.y.to_numpy() - rect.y0) / max(rect.height, 1.0), 0.0, 1.0)
    starts = np.interp(np.linspace(0.0, 1.0, Y_NODES), kept_ys, envelope.to_numpy())
    return tuple(float(v) for v in starts), (float(ys.min()), float(ys.max()))


def side_model(part: pd.DataFrame, page_top: float, page_height: float, cfg: StretchConfig) -> CompressionModel:
    """Модель сжатия стороны: амплитуда по высоте — P-сплайн (``cfg.spline_penalty``), линия или константа.

    Args:
        part: Окна стороны (поправленные на место в строке).
        page_top: Верх страницы на холсте, px.
        page_height: Высота страницы, px.
        cfg: Параметры.

    Returns:
        Модель (нулевая — строк или окон мало).
    """
    if cfg.spline_penalty is None:
        return fit_model(part, cfg.method, page_height, page_top, use_tilt=cfg.use_tilt, tilt_max=cfg.tilt_max)
    data = side_data(part, cfg.method, page_top, page_height)
    edge = float(part.edge_mm.median())
    lines = int(part.line.nunique())
    if lines < MIN_LINES or len(data.r) < MIN_WINDOWS:
        return CompressionModel(windows=len(data.r), lines=lines, edge_mm=edge)
    fit = fit_shape(data.x, data.y, data.r, HeightShape.SPLINE, cfg.spline_penalty)
    return CompressionModel(float(fit.amplitude(0.5)), fit.length_mm, 0.0, fit.windows, lines, fit.residual, edge,
                            fit.coef)  # fmt: skip


def letter_side_model(letters: pd.DataFrame, page_top: float, page_height: float, px_per_mm: float,
                      origin_mm: float, cfg: StretchConfig) -> CompressionModel:  # fmt: skip
    """Модель сжатия стороны по ширине букв (``letter_fit.fit_letters``): амплитуда по высоте — P-сплайн, линия или
    константа, как у :func:`side_model`.

    Args:
        letters: Буквы стороны.
        page_top: Верх страницы на холсте, px.
        page_height: Высота страницы, px.
        px_per_mm: Плотность, px/мм.
        origin_mm: Начало спада от корешка (начало зоны текста), мм.
        cfg: Параметры.

    Returns:
        Модель с ``edge_mm`` = ``origin_mm`` (нулевая — букв или строк мало).
    """
    lines = int(letters.line.nunique()) if len(letters) else 0
    data = letter_data(letters, page_top, page_height, px_per_mm) if len(letters) else None
    if data is None or lines < MIN_LINES or len(data.r) < MIN_LETTERS:
        return CompressionModel(windows=0 if data is None else len(data.r), lines=lines, edge_mm=origin_mm)
    if cfg.spline_penalty is not None:
        shape = HeightShape.SPLINE
    else:
        shape = HeightShape.LINEAR if cfg.use_tilt else HeightShape.CONST
    fit = fit_letters(data, shape, origin_mm, cfg.spline_penalty or 0.0)
    if shape is HeightShape.SPLINE:
        return CompressionModel(float(fit.amplitude(0.5)), fit.length_mm, 0.0, fit.windows, lines, fit.residual,
                                origin_mm, fit.coef)  # fmt: skip
    # Линия Лежандра по z = 2y − 1: B(y) = c₀ + c₁·(2y − 1), то есть амплитуда в середине c₀ и наклон 2·c₁.
    tilt = 2.0 * fit.coef[1] if shape is HeightShape.LINEAR else 0.0
    return CompressionModel(fit.coef[0], fit.length_mm, tilt, fit.windows, lines, fit.residual, origin_mm)


def shifted_region(region: PageRegion, dx: float) -> PageRegion:
    """Область страницы, сдвинутая по x на ``dx`` px (рамка и многоугольник)."""
    x0, y0, x1, y1 = region.box
    polygon = None if region.polygon is None else np.asarray(region.polygon, np.float64) + np.array([dx, 0.0])
    return replace(region, box=(x0 + dx, y0, x1 + dx, y1), polygon=polygon)


def reanalyze(image: np.ndarray, result: TextResult, dpi: float, wide: Layout | None = None) -> TextMeasures:
    """Прямизна строк и сторон итога с растяжением: разбор страниц заново теми же вырезками и преградами.

    Args:
        image: Итог с растяжением.
        result: Выпрямление без растяжения (раскладка, преграды разбора).
        dpi: Плотность, точек на дюйм.
        wide: Раскладка холста итога, если он шире раскладки Кунса (режим ``widen``), или None — та же. Преграды
            разбора правой страницы сдвигаются на сдвиг её внешнего края (у корешка ошибка — до расширения левой).

    Returns:
        Меры прямизны.
    """
    layout = wide or result.layout
    margin = mm_to_px_float(CROP_MARGIN_MM, dpi)
    pages = []
    for i, (rect, page) in enumerate(zip(layout.rects, result.pages)):
        if page.after is None:
            continue
        dx = 0.0 if i == 0 else rect.x1 - result.layout.rects[1].x1
        regions = tuple(shifted_region(r, dx) for r in page.after.regions) if dx else page.after.regions
        box = crop_box(rect, image.shape[:2], margin, layout.spine_x, i == 0)
        pages.append(analyze_page(image, box, dpi, rect, regions))
    return text_measures(pages, dpi)


def add_wide_displacement(total: tuple[np.ndarray, np.ndarray], shift_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Сложение поля с растяжением на холсте другого размера: ``D'(X') = s(X') + D(X' + s(X'))``.

    То же, что ``text_run.add_displacement``, но размер итога берётся у смещения ``s`` (расширенный холст), а ``D`` —
    поле холста Кунса.

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


def stretch_frame(ref: FrameRef, cfg: StretchConfig) -> StretchResult:
    """Выпрямление кадра с растяжением у корешка.

    Args:
        ref: Кадр.
        cfg: Параметры.

    Returns:
        Итог с промежуточными данными.
    """
    ppm = ref.frame_set.px_per_mm
    dpi = ref.frame_set.dpi
    image = read_bgr(ref.v2_path)
    geometry = frame_geometry(ref)
    result = text_frame(image, geometry, dpi, cfg.text)
    layout = result.layout
    gray = cv2.cvtColor(result.final, cv2.COLOR_BGR2GRAY)
    before = correct_position(spread_windows(gray, layout.rects, layout.spine_x, ppm, cfg.measure), cfg.bias)
    letters_before = spread_letters(gray, layout.rects, layout.spine_x, ppm, cfg.measure)
    plans = side_plans(before, result, ppm, cfg, letters_before)
    if cfg.mode is StretchMode.WIDEN:
        # Холст итога шире раскладки Кунса; Кунс считается по прежней раскладке (точки X' + D' — в её координатах).
        wide, dx = widen_displacement(plans, layout, ppm)
        total = add_wide_displacement(result.dense, dx)
        maps = compose_maps(result.patches, replace(layout, width=wide.width), *total)
        # Сдвиг текста относительно места на холсте Кунса: правая страница — без общего переноса на W_L.
        cols = np.arange(wide.width, dtype=np.float32)[None, :]
        moved = np.abs(np.where(cols >= wide.spine_x, dx + (wide.spine_x - layout.spine_x), dx))
    else:
        wide = layout
        dx = stretch_displacement(plans, layout, ppm)
        total = add_displacement(result.dense, (dx, np.zeros_like(dx)))
        maps = compose_maps(result.patches, layout, *total)
        moved = np.abs(dx)
    stretched = rectify_image(result.equalized, maps, dpi)
    gray_after = cv2.cvtColor(stretched, cv2.COLOR_BGR2GRAY)
    after = correct_position(spread_windows(gray_after, wide.rects, wide.spine_x, ppm, cfg.measure), cfg.bias)
    letters_after = spread_letters(gray_after, wide.rects, wide.spine_x, ppm, cfg.measure)
    measures_after = reanalyze(stretched, result, dpi, wide if wide is not layout else None)
    return StretchResult(result, stretched, plans, before, after, result.after, measures_after,
                         float(moved.max()) / ppm, wide, letters_before, letters_after)  # fmt: skip


def summary_row(key: str, out: StretchResult) -> dict:
    """Строка сводки кадра: модели, лишняя длина, меры у края до и после, прямизна строк и сторон.

    Args:
        key: Ключ кадра.
        out: Итог кадра.

    Returns:
        Словарь полей.
    """
    row: dict = {"key": key, "dx_max_mm": round(out.dx_max_mm, 2)}
    for side, plan in zip(Side, out.plans):
        p = side.value
        row.update({f"{p}_amp": round(plan.model.amplitude, 3), f"{p}_len_mm": round(plan.model.length_mm, 2),
                    f"{p}_tilt": round(plan.model.tilt, 3), f"{p}_resid": round(plan.model.residual, 3),
                    f"{p}_edge_mm": round(plan.edge_mm, 2), f"{p}_applied": int(plan.applied)})  # fmt: skip
        if isinstance(plan, WidenPlan):
            # Расширение, разброс прибавки по высоте и общий масштаб строки, которым он гасится.
            k_lo, k_hi = plan.scale_range()
            row.update({f"{p}_widen_mm": round(plan.widen_mm, 2), f"{p}_excess_mm": round(max(plan.excesses_mm), 2),
                        f"{p}_excess_min_mm": round(min(plan.excesses_mm), 2),
                        f"{p}_text_start_mm": round(min(plan.text_starts_mm or (plan.edge_mm,)), 2), f"{p}_scale_min": round(k_lo, 4),
                        f"{p}_scale_max": round(k_hi, 4)})  # fmt: skip
        else:
            row.update({f"{p}_excess_mm": round(plan.excess_mm, 2), f"{p}_squeeze": round(plan.body_squeeze, 4)})
        for label, table in (("before", out.before), ("after", out.after)):
            part = table[table.side == p] if len(table) else table
            if len(part):
                summ = side_summary(part)
                row.update({f"{p}_{label}_{k}": summ[k] for k in ("near_acf", "near_stem", "near_comp", "near_height",
                                                                  "mid_acf", "edge_mm")})  # fmt: skip
        # Ширина букв у края до и после — прямая проверка растяжения, не зависящая от окон.
        for label, letters in (("before", out.letters_before), ("after", out.letters_after)):
            part = letters[letters.side == p] if len(letters) else letters
            row.update({f"{p}_{label}_{k}": v for k, v in letter_summary(part).items()})
    row["canvas_widen_px"] = out.wide.width - out.result.layout.width
    row.update(out.measures_before.row("lines_before_"))
    row.update(out.measures_after.row("lines_after_"))
    return row


def write_sheet(path: Path, key: str, out: StretchResult, px_per_mm: float) -> None:
    """Лист «было | стало» кадра.

    Args:
        path: Путь JPEG.
        key: Ключ кадра.
        out: Итог кадра.
        px_per_mm: Плотность, px/мм.
    """
    layout = out.result.layout
    top = max(r.y0 for r in layout.rects)
    bottom = min(r.y1 for r in layout.rects)
    ys = [top + 0.2 * (bottom - top), top + 0.5 * (bottom - top), top + 0.8 * (bottom - top)]
    notes = []
    for y in ys:
        parts = []
        for label, table in (("было", out.before), ("стало", out.after)):
            near = table[(abs(table.y - y) < 12 * px_per_mm) & (table.from_edge_mm <= 5.6)] if len(table) else table
            vals = [f"{s}: {near[near.side == s].acf_ratio.median():.2f}" for s in ("L", "R")]
            parts.append(f"{label} шаг у края " + ", ".join(vals))
        notes.append("; ".join(parts))
    plans = "; ".join(plan_note(s, p) for s, p in zip(Side, out.plans))
    mb, ma = out.measures_before, out.measures_after
    widen = isinstance(out.plans[0], WidenPlan)
    title = "страница шире со стороны корешка" if widen else "внешний край на месте"
    header = [
        f"{key} — растяжение у корешка по модели c(d, y) = 1 − B(y)·exp(−(d − e)/L), {title}",
        plans,
        f"строки p95 {mb.line_p95_mm:.2f} → {ma.line_p95_mm:.2f} мм, стороны p95 {mb.side_p95_mm:.2f} → "
        f"{ma.side_p95_mm:.2f} мм; наибольший |dx| {out.dx_max_mm:.2f} мм",
    ]
    sheet = before_after_sheet(out.result.final, out.stretched, layout.spine_x, ys, px_per_mm, header, notes,
                               spine_after=out.wide.spine_x)  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def plan_note(side: Side, plan: Plan) -> str:
    """Подпись плана стороны в шапке листа: модель, лишняя длина, отступ и куда она ушла."""
    m = plan.model
    if m.spline_coef:
        top, mid, bottom = (float(m.amplitude_at(y)) for y in (0.0, 0.5, 1.0))
        head = f"{side.value}: B(y) сплайн {top:.2f}/{mid:.2f}/{bottom:.2f} (верх/середина/низ) L={m.length_mm:.1f} мм"
    else:
        head = f"{side.value}: B={m.amplitude:.2f}{m.tilt:+.2f}·(y−½) L={m.length_mm:.1f} мм"
    tail = "" if plan.applied else " (не применено)"
    if isinstance(plan, WidenPlan):
        k_lo, k_hi = plan.scale_range()
        return (f"{head}, прибавка строки {min(plan.excesses_mm):.2f}–{max(plan.excesses_mm):.2f} мм по высоте, "
                f"зона текста от {min(plan.text_starts_mm or (plan.edge_mm,)):.1f} мм (медиана {plan.edge_mm:.1f}), страница шире на "
                f"{plan.widen_mm:.2f} мм, масштаб строки {k_lo:.3f}–{k_hi:.3f}{tail}")  # fmt: skip
    return (f"{head}, лишняя длина {plan.excess_mm:.2f} мм, отступ {plan.edge_mm:.1f} мм, "
            f"сжатие тела {100 * plan.body_squeeze:.2f} %{tail}")  # fmt: skip


def synth_frame(gray: np.ndarray, layout, px_per_mm: float, models: list[CompressionModel],
                params: MeasureParams) -> tuple[list[float], list[pd.DataFrame]]:  # fmt: skip
    """Синтетическое сжатие холста по каждой модели (от края текста своей стороны) и окна строк сжатого холста.

    Args:
        gray: Выпрямленный холст без сжатия (контроль).
        layout: Раскладка холста.
        px_per_mm: Плотность, px/мм.
        models: Модели сжатия (истина; ``edge_mm`` подставляется по стороне).
        params: Параметры меры.

    Returns:
        (отступы края текста левой и правой страницы, мм; таблицы окон по моделям в том же порядке).
    """
    base = spread_windows(gray, layout.rects, layout.spine_x, px_per_mm, params)
    edges = [float(base[base.side == s.value].edge_mm.median()) if len(base) else 0.0 for s in Side]
    tables = []
    for m in models:
        pair = tuple(CompressionModel(m.amplitude, m.length_mm, edge_mm=e) for e in edges)
        squeezed = compress_image(gray, layout.spine_x, pair, px_per_mm)
        tables.append(spread_windows(squeezed, layout.rects, layout.spine_x, px_per_mm, params))
    return edges, tables
