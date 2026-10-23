"""Листы второго преобразования «было | стало»: 01 освещение, 02 разбор текста и выборки поля, 03 поле, 04 итог с осями строк до и после, 05 вырезки с увеличением."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed, header_strip
from ocr_utils.units import mm_to_px_float, px_to_mm
from ocr_utils.scan_cropping.v2.overlay import COLOUR_FRAME
from research.page_rectify.overlay import FRAME_PX, patch_outline
from research.page_rectify.text_field import FieldMode, PageText, SampleKind, Samples, line_curve
from research.page_rectify.text_run import TextResult

# Высота каждой половины листа, px.
SHEET_HEIGHT = 1400
GAP = 24
# Разделитель стадий на листе: чёрная полоса посреди зазора, px листа (просьба пользователя 2026-10-22: ярко-зелёная
# забирала внимание).
COLOUR_SEPARATOR = (0, 0, 0)
SEPARATOR_PX = 10
# Палитра проекта (навык draw-overlay).
COLOUR_AXIS = (40, 170, 40)  # базовая линия строки
COLOUR_ALIGNED = (220, 90, 20)  # выровненная сторона блока
COLOUR_RAGGED = (0, 0, 220)  # невыровненная сторона
COLOUR_BLOCK = (150, 170, 120)  # граница блока, справочная
# Области page_layout: растр, таблица, line art.
COLOUR_REGION = {"raster": (60, 60, 200), "table": (200, 60, 60), "line_art": (0, 140, 255), "equation": (160, 60, 160)}
REGION_TITLE = {"raster": "растр (опор поля нет)", "table": "таблица", "line_art": "line art (опор поля нет)",
                "equation": "формула surya (опор поля нет)"}  # fmt: skip
# Рамка — полупрозрачная, без заливки внутри (просьба пользователя 2026-10-22).
REGION_ALPHA = 0.55
REGION_PX = 3
COLOUR_RULE = (40, 170, 40)  # линейка — тот же зелёный, что ось строки, но пунктиром (опора поля того же рода)
# Пунктир линейки на листе: штрих и пропуск, px листа.
RULE_DASH = (6, 4)
COLOUR_ARROW = (0, 165, 255)  # выборка: точка → цель
COLOUR_GRID = (200, 60, 200)  # деформированная сетка
COLOUR_ZOOM = (180, 105, 255)  # рамка вырезки
COLOUR_CUT = (255, 110, 210)  # линия разреза страниц (корешок) — ярко-сиреневая (просьба пользователя 2026-10-22)
CUT_PX = 1
# Шаг по строкам холста, с которым строится образ линии разреза на итоге, px.
CUT_STEP_PX = 4
BLOCK_ALPHA = 0.25
FIELD_ALPHA = 0.45
# Стрелки выборок увеличены во столько раз, иначе смещения в 0.5 мм на листе не видны.
ARROW_GAIN = 5.0
# Шаг деформированной сетки, мм.
GRID_MM = 10.0
# Вырезки с увеличением: размер на бумаге, мм, и ширина половины на листе, px.
ZOOM_MM = (70.0, 40.0)
ZOOM_WIDTH = 1000


def fit(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Картинка, ужатая до :data:`SHEET_HEIGHT` по высоте, и масштаб."""
    f = SHEET_HEIGHT / image.shape[0]
    return cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), f


def side_by_side(images: list[np.ndarray]) -> np.ndarray:
    """Картинки одной высоты в ряд; между стадиями — толстая чёрная полоса посреди белого зазора."""
    gap = np.full((images[0].shape[0], GAP, 3), 255, np.uint8)
    mid = (GAP - SEPARATOR_PX) // 2
    gap[:, mid : mid + SEPARATOR_PX] = COLOUR_SEPARATOR
    parts = []
    for i, image in enumerate(images):
        parts += [gap, image] if i else [image]
    return np.hstack(parts)


def polylines(image: np.ndarray, curves: list[np.ndarray], colour, scale: float, width: int = 1, alpha: float = 1.0):
    """Ломаные (px полного размера) на картинке листа, полупрозрачно при ``alpha < 1`` (на месте)."""
    if not curves:
        return
    layer = image.copy() if alpha < 1 else image
    pts = [np.round(np.asarray(c) * scale).astype(np.int32) for c in curves if len(c) >= 2]
    cv2.polylines(layer, pts, False, colour, width, cv2.LINE_AA)
    if alpha < 1:
        cv2.addWeighted(layer, alpha, image, 1 - alpha, 0, image)


def dashed(image: np.ndarray, curves: list[np.ndarray], colour, scale: float) -> None:
    """Пунктирные ломаные толщиной 1 px (px полного размера) на картинке листа (на месте).

    Args:
        image: Картинка листа BGR.
        curves: Ломаные (N, 2) в px полного размера.
        colour: Цвет BGR.
        scale: Масштаб листа к полному размеру.
    """
    dash, gap = RULE_DASH
    for curve in curves:
        pts = np.asarray(curve, np.float64) * scale
        if len(pts) < 2:
            continue
        # Длина дуги по точкам; штрихи — куски дуги [k·(dash+gap), k·(dash+gap)+dash].
        arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        for start in np.arange(0.0, arc[-1], dash + gap):
            t = np.linspace(start, min(start + dash, arc[-1]), 4)
            seg = np.stack([np.interp(t, arc, pts[:, 0]), np.interp(t, arc, pts[:, 1])], 1)
            cv2.polylines(image, [np.round(seg).astype(np.int32)], False, colour, 1, cv2.LINE_AA)


def axes_of(page: PageText | None) -> list[np.ndarray]:
    """Базовые линии строк страницы на холсте (без базы — оси центра масс, :func:`text_field.line_curve`)."""
    return (
        [] if page is None else [page.to_canvas(line_curve(a)) for a in page.analysis.axes if len(line_curve(a)) >= 2]
    )


def sides_of(page: PageText | None) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Выровненные стороны, невыровненные стороны и контуры блоков страницы на холсте."""
    aligned, ragged, outlines = [], [], []
    if page is None:
        return aligned, ragged, outlines
    for block, alignment in zip(page.analysis.blocks, page.analysis.alignments):
        envelope = block.envelope
        if len(envelope.polygon) >= 3:
            outlines.append(np.vstack([page.to_canvas(envelope.polygon), page.to_canvas(envelope.polygon[:1])]))
        for stats, core, outer in ((alignment.left, envelope.core_left, envelope.left),
                                   (alignment.right, envelope.core_right, envelope.right)):  # fmt: skip
            curve = core if core is not None else outer
            if curve is None or len(curve) < 2:
                continue
            (aligned if stats.aligned else ragged).append(page.to_canvas(curve))
    return aligned, ragged, outlines


def page_note(result: TextResult) -> str:
    """Подпись страниц: уровень цветности, строк в поле, участвует ли страница."""
    notes = []
    for name, page in zip(("левая", "правая"), result.pages):
        if page.mode is FieldMode.FULL:
            state = "в поле"
        elif page.mode is FieldMode.SPARSE:
            state = "мало строк — поле у опор"
        else:
            state = "цветная — без поля" if page.before is None else "мало строк — без поля"
        notes.append(f"{name}: {page.colour.value}, строк {page.lines}, {state}")
    return "; ".join(notes)


def sheet_illumination(result: TextResult, header: list[str]) -> np.ndarray:
    """01: итог v2 | после выравнивания освещения | фон страниц (псевдоцвет, тёмное — тень)."""
    before, f = fit(result.image)
    after, _ = fit(result.equalized)
    bg = np.full(result.image.shape[:2], 255, np.float32)
    notes = []
    for name, page in zip(("левая", "правая"), result.pages):
        if page.background is None:
            notes.append(f"{name}: без выравнивания")
            continue
        bg[page.mask] = page.background.surface[page.mask]
        surface = page.background.surface[page.mask]
        notes.append(f"{name}: фон {surface.min():.0f}…{surface.max():.0f}, способ {page.background.method}, "
                     f"годно {page.background.mask.mean():.0%}")  # fmt: skip
    lo = float(bg.min())
    norm = np.clip((bg - lo) / max(255 - lo, 1) * 255, 0, 255).astype(np.uint8)
    heat, _ = fit(cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS))
    legend = [LegendEntry(f"фон страницы: {lo:.0f} (тёмно-синий) … 255 (жёлтый); вне страниц — 255", (0, 200, 230),
                          style=SampleStyle.BOX)]  # fmt: skip
    return framed(side_by_side([before, after, heat]), header + ["итог v2 | выровненное освещение | фон", "; ".join(notes)],
                  legend)  # fmt: skip


def draw_regions(image: np.ndarray, pages: list[PageText | None], scale: float) -> None:
    """Растр, таблицы и line art страниц — полупрозрачными рамками без заливки (на месте)."""
    layer = image.copy()
    for page in pages:
        for region in () if page is None else page.regions:
            x0, y0, x1, y1 = (int(round(v * scale)) for v in region.box)
            cv2.rectangle(layer, (x0, y0), (x1, y1), COLOUR_REGION[region.kind.value], REGION_PX)
    cv2.addWeighted(layer, REGION_ALPHA, image, 1 - REGION_ALPHA, 0, image)


def draw_text(image: np.ndarray, pages: list[PageText | None], scale: float) -> None:
    """Области page_layout, блоки, стороны, оси строк и линейки страниц на картинке листа (на месте)."""
    draw_regions(image, pages, scale)
    for page in pages:
        aligned, ragged, outlines = sides_of(page)
        polylines(image, outlines, COLOUR_BLOCK, scale, 2, BLOCK_ALPHA)
        polylines(image, axes_of(page), COLOUR_AXIS, scale, 1)
        polylines(image, aligned, COLOUR_ALIGNED, scale, 2)
        polylines(image, ragged, COLOUR_RAGGED, scale, 1)
        if page is not None:
            dashed(image, [r.points for r in page.rules], COLOUR_RULE, scale)


def text_legend() -> list[LegendEntry]:
    """Легенда разбора текста."""
    return [
        LegendEntry("базовая линия строки", COLOUR_AXIS),
        LegendEntry("выровненная сторона блока", COLOUR_ALIGNED),
        LegendEntry("невыровненная сторона", COLOUR_RAGGED),
        LegendEntry("граница блока", COLOUR_BLOCK, alpha=BLOCK_ALPHA),
        LegendEntry("линейка (в поле — как строка или сторона)", COLOUR_RULE, style=SampleStyle.DASHED),
    ] + [LegendEntry(REGION_TITLE[k], c, alpha=REGION_ALPHA) for k, c in COLOUR_REGION.items()]


def arrows(image: np.ndarray, samples: Samples, weight: np.ndarray, scale: float) -> None:
    """Стрелки «точка → цель» (×:data:`ARROW_GAIN`) по выборкам строк и сторон; выброшенные — красной точкой."""
    for src, at, w, kind in zip(samples.source, samples.at, weight, samples.kind):
        if kind == SampleKind.ZERO:
            continue
        p = tuple(np.round(src * scale).astype(int))
        if w <= 0:
            cv2.circle(image, p, 2, COLOUR_RAGGED, -1, cv2.LINE_AA)
            continue
        q = tuple(np.round((src + ARROW_GAIN * (at - src)) * scale).astype(int))
        if p != q:
            cv2.arrowedLine(image, p, q, COLOUR_ARROW, 1, cv2.LINE_AA, tipLength=0.3)


def sheet_analysis(result: TextResult, header: list[str]) -> np.ndarray:
    """02: выпрямленный кадр с разбором текста | тот же кадр со стрелками выборок поля."""
    left, f = fit(result.first)
    right = left.copy()
    draw_text(left, [p.before for p in result.pages], f)
    field = result.field
    arrows(right, field.v_samples, field.dy.weight, f)
    arrows(right, field.h_samples, field.dx.weight, f)
    legend = text_legend() + [
        LegendEntry(f"выборка: точка оси/стороны → цель, ×{ARROW_GAIN:.0f}", COLOUR_ARROW),
        LegendEntry("выборка выброшена (остаток > 3 мм)", COLOUR_RAGGED, style=SampleStyle.BOX),
    ]
    n_v = int((field.v_samples.kind == SampleKind.LINE).sum())
    n_h = int((field.h_samples.kind == SampleKind.SIDE).sum())
    return framed(side_by_side([left, right]),
                  header + [f"после первого преобразования: разбор текста | выборки поля (строк {n_v}, сторон {n_h})",
                            page_note(result)], legend)  # fmt: skip


def sheet_field(result: TextResult, header: list[str], dpi: float) -> np.ndarray:
    """03: модуль поля (псевдоцвет) | деформированная сетка: откуда берутся прямые линии итога."""
    dx, dy = result.dense
    norm_mm = px_to_mm(np.hypot(dx, dy), dpi)
    top = max(float(norm_mm.max()), 0.1)
    heat_full = cv2.applyColorMap(np.clip(norm_mm / top * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    base, f = fit(result.first)
    heat, _ = fit(heat_full)
    left = cv2.addWeighted(heat, FIELD_ALPHA, base, 1 - FIELD_ALPHA, 0)
    right = base.copy()
    h, w = dx.shape
    step = mm_to_px_float(GRID_MM, dpi)
    curves = []
    for x in np.arange(0, w, step):  # вертикали сетки итога → откуда они на выпрямленном кадре
        ys = np.arange(0, h, 4.0)
        xi = np.full(len(ys), int(min(x, w - 1)))
        yi = ys.astype(int)
        curves.append(np.stack([x + dx[yi, xi], ys + dy[yi, xi]], 1))
    for y in np.arange(0, h, step):
        xs = np.arange(0, w, 4.0)
        yi = np.full(len(xs), int(min(y, h - 1)))
        xi = xs.astype(int)
        curves.append(np.stack([xs + dx[yi, xi], y + dy[yi, xi]], 1))
    polylines(right, curves, COLOUR_GRID, f, 1, 0.8)
    legend = [
        LegendEntry(f"модуль поля: 0 (чёрный) … {top:.2f} мм (светлый)", (60, 120, 230), alpha=FIELD_ALPHA,
                    style=SampleStyle.BOX),
        LegendEntry(f"сетка итога {GRID_MM:.0f} мм — где она лежит на выпрямленном кадре", COLOUR_GRID, alpha=0.8),
    ]  # fmt: skip
    note = (f"поле: |dx| до {px_to_mm(np.abs(dx).max(), dpi):.2f} мм, |dy| до {px_to_mm(np.abs(dy).max(), dpi):.2f} мм"
            + (", обрезано по потолку" if result.field.clipped else ""))  # fmt: skip
    return framed(side_by_side([left, right]), header + [note, page_note(result)], legend)


def measures_line(result: TextResult) -> str:
    """Меры прямизны до и после одной строкой."""
    b, a = result.before, result.after
    return (f"размах строк по высоте, мм (медиана / 95%): {b.line_med_mm:.2f} / {b.line_p95_mm:.2f} → "
            f"{a.line_med_mm:.2f} / {a.line_p95_mm:.2f}; наклон {b.slope_med_deg:.2f}° → {a.slope_med_deg:.2f}°; "
            f"выровненные стороны, мм: {b.side_med_mm:.2f} / {b.side_p95_mm:.2f} → {a.side_med_mm:.2f} / "
            f"{a.side_p95_mm:.2f}")  # fmt: skip


def cut_on_final(result: TextResult) -> np.ndarray:
    """Линия разреза страниц на итоге: в каждой строке холста — столбец, где составная карта переходит с левой страницы
    на правую (первый ``X`` с ``X + D_x(X) ≥ spine_x``). После второго преобразования это кривая.

    Args:
        result: Итог кадра.

    Returns:
        Ломаная (N, 2) px итога сверху вниз.
    """
    dx = result.dense[0]
    h, w = dx.shape
    cols = np.arange(w, dtype=np.float32)
    points = []
    for y in range(0, h, CUT_STEP_PX):
        right = cols + dx[y] >= result.layout.spine_x
        if right.any():
            points.append((float(np.argmax(right)), float(y)))
    return np.array(points)


def sheet_result(result: TextResult, header: list[str]) -> np.ndarray:
    """04: итог v2 после выравнивания освещения с подгоночными областями страниц | после первого преобразования | итог.

    На втором и третьем — оси строк и стороны, найденные на этой же картинке, и линия разреза страниц (на втором —
    вертикаль корешка, на итоге — её образ после поля).
    """
    source, e = fit(result.equalized)
    polylines(source, [np.vstack([patch_outline(p), patch_outline(p)[:1]]) for p in result.patches], COLOUR_FRAME, e,
              FRAME_PX)  # fmt: skip
    left, f = fit(result.first)
    right, g = fit(result.final)
    draw_text(left, [p.before for p in result.pages], f)
    draw_text(right, [p.after for p in result.pages], g)
    spine = np.array([[result.layout.spine_x, 0.0], [result.layout.spine_x, float(result.layout.height - 1)]])
    polylines(left, [spine], COLOUR_CUT, f, CUT_PX)
    polylines(right, [cut_on_final(result)], COLOUR_CUT, g, CUT_PX)
    return framed(side_by_side([source, left, right]),
                  header + ["итог v2 после выравнивания освещения | после первого преобразования (Кунс) | итог (Кунс + "
                            "поле по тексту)", measures_line(result)],
                  [LegendEntry("подгоночная область страницы", COLOUR_FRAME)] + text_legend()
                  + [LegendEntry("линия разреза страниц (корешок)", COLOUR_CUT)])  # fmt: skip


def zoom_centres(result: TextResult) -> list[tuple[str, np.ndarray]]:
    """Центры вырезок: самая изогнутая строка, наибольший модуль поля на страницах, середина корешка."""
    centres = []
    v = result.field.v_samples
    lines = v.kind == SampleKind.LINE
    if lines.any():
        i = int(np.argmax(np.abs(np.where(lines, v.value, 0))))
        centres.append(("наибольший сдвиг строки", v.source[i]))
    dx, dy = result.dense
    norm = np.hypot(dx, dy)
    for rect in result.layout.rects:
        y0, y1, x0, x1 = int(rect.y0), int(rect.y1), int(rect.x0), int(rect.x1)
        part = norm[max(y0, 0) : y1, max(x0, 0) : x1]
        if part.size:
            r, c = np.unravel_index(int(np.argmax(part)), part.shape)
            centres.append(("наибольшее поле на странице", np.array([c + max(x0, 0), r + max(y0, 0)], float)))
    centres.append(("корешок", np.array([result.layout.spine_x, result.layout.height / 2])))
    return centres[:4]


def sheet_zoom(result: TextResult, header: list[str], dpi: float) -> np.ndarray:
    """05: вырезки ×3 «после первого | итог» в одних координатах холста."""
    half_w, half_h = (mm_to_px_float(m, dpi) / 2 for m in ZOOM_MM)
    rows = []
    for title, (cx, cy) in zoom_centres(result):
        h, w = result.first.shape[:2]
        x0, y0 = int(np.clip(cx - half_w, 0, max(w - 2 * half_w, 0))), int(
            np.clip(cy - half_h, 0, max(h - 2 * half_h, 0))
        )
        x1, y1 = int(min(x0 + 2 * half_w, w)), int(min(y0 + 2 * half_h, h))
        tiles = []
        for image in (result.first, result.final):
            crop = image[y0:y1, x0:x1]
            k = ZOOM_WIDTH / crop.shape[1]
            tiles.append(cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC))
        row = side_by_side(tiles)
        caption = f"{title}: x {px_to_mm(x0, dpi):.0f}…{px_to_mm(x1, dpi):.0f} мм, y {px_to_mm(y0, dpi):.0f}…{px_to_mm(y1, dpi):.0f} мм"
        rows.append(np.vstack([header_strip([caption], row.shape[1]), row]))
    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), 255, np.uint8)]) for r in rows]
    return framed(np.vstack(rows), header + ["вырезки: после первого преобразования | итог"], [])
