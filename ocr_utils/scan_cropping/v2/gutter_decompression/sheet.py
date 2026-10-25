"""Отладочный лист растяжения у корешка: шапка с моделями и планами страниц, разворот «было | стало» уменьшенным и вырезки у корешка ×2 на трёх высотах; поля сводки кадра."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ocr_utils.page_layout.overlay_frame import LegendEntry, _font, framed
from ocr_utils.scan_cropping.v2.gutter_decompression.widen import WidenPlan
from ocr_utils.scan_cropping.v2.rectify.coons import Layout

# Цвета (BGR, палитра ``/draw-overlay``): линия корешка — синяя, высоты вырезок — оранжевые.
COLOUR_SPINE = (200, 80, 0)
COLOUR_ROW = (0, 140, 255)
# Кегль подписи вырезки, px.
CAPTION_SIZE = 18
# Вырезка у корешка: половина ширины и высоты, мм, и увеличение.
CROP_HALF_MM = (22.0, 9.0)
CROP_ZOOM = 2.0
# Высоты вырезок — доли общей высоты страниц.
CROP_ROWS = (0.2, 0.5, 0.8)


def caption(image: np.ndarray, text: str) -> np.ndarray:
    """Картинка с подписью над ней в белом поле (не поверх страницы).

    Args:
        image: Картинка BGR.
        text: Подпись.

    Returns:
        Новая картинка: полоса подписи и картинка.
    """
    strip = Image.new("RGB", (image.shape[1], CAPTION_SIZE + 8), (255, 255, 255))
    ImageDraw.Draw(strip).text((4, 3), text, fill=(0, 0, 0), font=_font(CAPTION_SIZE))
    return np.vstack([cv2.cvtColor(np.asarray(strip), cv2.COLOR_RGB2BGR), image])


def gutter_crop(image: np.ndarray, spine_x: float, y_centre: float, px_per_mm: float) -> np.ndarray:
    """Вырезка у корешка с увеличением и рисками линии корешка у верхнего и нижнего края (не поверх букв).

    Args:
        image: Холст BGR.
        spine_x: X линии корешка, px.
        y_centre: Центр вырезки по высоте, px.
        px_per_mm: Плотность, px/мм.

    Returns:
        Вырезка BGR (за краем холста — белое).
    """
    hw, hh = (int(v * px_per_mm) for v in CROP_HALF_MM)
    x0, y0 = int(spine_x) - hw, int(y_centre) - hh
    canvas = np.full((2 * hh, 2 * hw, 3), 255, np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(image.shape[1], x0 + 2 * hw), min(image.shape[0], y0 + 2 * hh)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[sy0:sy1, sx0:sx1]
    out = cv2.resize(canvas, None, fx=CROP_ZOOM, fy=CROP_ZOOM, interpolation=cv2.INTER_CUBIC)
    xs = int(round((spine_x - x0) * CROP_ZOOM))
    tick = max(6, int(1.5 * px_per_mm * CROP_ZOOM))
    cv2.line(out, (xs, 0), (xs, tick), COLOUR_SPINE, 2)
    cv2.line(out, (xs, out.shape[0] - tick), (xs, out.shape[0] - 1), COLOUR_SPINE, 2)
    return out


def plan_note(name: str, plan: WidenPlan) -> str:
    """Модель и план страницы одной строкой для шапки листа.

    Args:
        name: Подпись страницы (``L``/``R``).
        plan: План страницы.

    Returns:
        Строка.
    """
    model = plan.model
    if model.null:
        return f"{name}: без модели (букв {model.letters}, строк {model.lines})"
    amps = model.amplitude_at(np.linspace(*plan.y_range, 5))
    state = f"шире на {plan.widen_mm:.2f} мм" if plan.applied else "не применено"
    return (f"{name}: B {amps.min():.2f}–{amps.max():.2f}, L {model.length_mm:.1f} мм, начало {model.origin_mm:.1f} мм, "
            f"прибавка у края {plan.near_mm:.2f} мм — {state}")  # fmt: skip


def gutter_sheet(before: np.ndarray, after: np.ndarray, layout: Layout, canvas: Layout,
                 plans: tuple[WidenPlan, WidenPlan], px_per_mm: float, header: list[str]) -> np.ndarray:  # fmt: skip
    """Лист «было | стало» растяжения у корешка.

    Args:
        before: Итог без растяжения (Кунс + поле), холст ``layout``.
        after: Итог с растяжением, холст ``canvas``.
        layout: Раскладка Кунса.
        canvas: Раскладка холста итога (шире при растяжении).
        plans: Планы левой и правой страницы.
        px_per_mm: Плотность, px/мм.
        header: Строки шапки кадра.

    Returns:
        Лист BGR.
    """
    top = max(r.y0 for r in layout.rects)
    bottom = min(r.y1 for r in layout.rects)
    ys = [top + f * (bottom - top) for f in CROP_ROWS]
    rows = []
    for y in ys:
        a = gutter_crop(before, layout.spine_x, y, px_per_mm)
        b = gutter_crop(after, canvas.spine_x, y, px_per_mm)
        gap = np.full((a.shape[0], 12, 3), 255, np.uint8)
        rows.append(caption(np.hstack([a, gap, b]), f"было | стало, ×{CROP_ZOOM:.0f} у корешка"))
    width = rows[0].shape[1]
    scale = width / (before.shape[1] + after.shape[1] + 12)
    small = [cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for im in (before, after)]
    gap_w = max(width - small[0].shape[1] - small[1].shape[1], 0)
    overview = np.hstack([small[0], np.full((small[0].shape[0], gap_w, 3), 255, np.uint8), small[1]])[:, :width]
    if overview.shape[1] < width:
        overview = np.hstack([overview, np.full((overview.shape[0], width - overview.shape[1], 3), 255, np.uint8)])
    for x in (int(layout.spine_x * scale), int(canvas.spine_x * scale) + small[0].shape[1] + gap_w):
        cv2.line(overview, (x, 0), (x, 8), COLOUR_SPINE, 2)
    for y in ys:
        cv2.line(overview, (0, int(y * scale)), (6, int(y * scale)), COLOUR_ROW, 2)
    body = np.vstack([caption(overview, "разворот: было | стало (оранжевые риски слева — высоты вырезок)")] + rows)
    lines = header + ["растяжение у корешка: c(d, y) = 1 − B(y)·exp(−(d − e)/L), подгонка по ширине букв",
                      "; ".join(plan_note(n, p) for n, p in zip("LR", plans))]  # fmt: skip
    legend = [LegendEntry("линия корешка (риски у краёв)", COLOUR_SPINE),
              LegendEntry("высоты вырезок (риски у левого края обзора)", COLOUR_ROW)]  # fmt: skip
    return framed(body, lines, legend)


def summary_fields(plans: tuple[WidenPlan, WidenPlan] | None) -> dict:
    """Поля сводки кадра: применено ли растяжение, расширение, модель каждой страницы.

    Args:
        plans: Планы левой и правой страницы или None — растяжения не было.

    Returns:
        ``gutter_<L|R>_{applied, widen_mm, amp_max, length_mm, near_mm}``; пусто — растяжения не было.
    """
    if plans is None:
        return {}
    out = {}
    for name, plan in zip("LR", plans):
        amps = plan.model.amplitude_at(np.linspace(*plan.y_range, 5))
        out |= {f"gutter_{name}_applied": int(plan.applied), f"gutter_{name}_widen_mm": round(plan.widen_mm, 2),
                f"gutter_{name}_amp_max": round(float(amps.max()), 3),
                f"gutter_{name}_length_mm": round(plan.model.length_mm, 2),
                f"gutter_{name}_near_mm": round(plan.near_mm, 2)}  # fmt: skip
    return out


__all__ = ["gutter_sheet", "plan_note", "summary_fields"]
