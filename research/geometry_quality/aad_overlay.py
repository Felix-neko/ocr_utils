"""Оверлей AAD по рамке line art: вырезка B, вырезка A в кадре B, карта отклонений AAD поверх A; меры и пороги в шапке."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout import px_to_mm
from ocr_utils.page_layout.overlay_frame import LegendEntry, header_strip, legend_strip
from ocr_utils.geometry_regression.quality.lineart_flow import NEIGHBOUR_MM, WORK_DPI, FlowMeasure

# Ширина одной панели на листе, пиксели (вырезка 150 dpi увеличивается до неё: черты схемы должны читаться).
PANEL_WIDTH = 760
# Ширина панели широкой рамки (панели одна под другой).
WIDE_WIDTH = 1800
# Карта отклонений: полупрозрачно поверх A (навык draw-overlay), шкала — от нуля до SCALE_FACTOR × порога AAD.
HEAT_ALPHA = 0.55
SCALE_FACTOR = 3.0
# Пиксели черт раздуваются, чтобы тонкая линия карты была видна на увеличении.
HEAT_DILATE_PX = 1
# Отрезки LSD: полупрозрачно; отрезок без параллельных соседей — серым.
SEGMENT_ALPHA = 0.75
NO_NEIGHBOUR_COLOUR = (150, 150, 150)


def _panel(gray: np.ndarray, scale: float) -> np.ndarray:
    """Серая вырезка → BGR, увеличенная до ширины панели."""
    size = (int(round(gray.shape[1] * scale)), int(round(gray.shape[0] * scale)))
    return cv2.cvtColor(cv2.resize(gray, size, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)


def heat_panel(crop_a: np.ndarray, heat: np.ndarray, threshold_mm: float, scale: float) -> np.ndarray:
    """A с картой отклонения AAD поверх: цвет — сдвиг черты от общего по строке или столбцу, от 0 до 3 порогов.

    Args:
        crop_a: Вырезка A в кадре B.
        heat: Отклонение AAD по пикселям черт, пиксели 150 dpi (ноль вне черт).
        threshold_mm: Порог AAD, мм (шкала цвета — ``SCALE_FACTOR`` порогов).
        scale: Увеличение панели.

    Returns:
        Панель BGR.
    """
    canvas = _panel(crop_a, scale)
    top = SCALE_FACTOR * threshold_mm * WORK_DPI / 25.4
    level = np.clip(heat / max(top, 1e-6), 0.0, 1.0)
    colours = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = (heat > 0).astype(np.uint8)
    if HEAT_DILATE_PX:
        mask = cv2.dilate(mask, np.ones((2 * HEAT_DILATE_PX + 1,) * 2, np.uint8))
    size = (canvas.shape[1], canvas.shape[0])
    colours = cv2.resize(colours, size, interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST).astype(bool)
    layer = canvas.copy()
    layer[mask] = colours[mask]
    cv2.addWeighted(layer, HEAT_ALPHA, canvas, 1 - HEAT_ALPHA, 0, canvas)
    return canvas


def segments_panel(crop_b: np.ndarray, segments: np.ndarray, threshold_mm: float, scale: float) -> np.ndarray:
    """B с отрезками LSD поверх: цвет — сдвиг отрезка поперёк относительно соседних параллельных, от 0 до 3 порогов.

    Отрезки любой ориентации (наклонные — лучи, стрелки, скосы соединителей), от ``SEGMENT_MIN_MM``; отрезок без
    параллельных соседей в пределах ``NEIGHBOUR_MM`` — серым. Толсто и полупрозрачно, чтобы под ним читалась черта.

    Args:
        crop_b: Вырезка B.
        segments: ``(n, 5)`` — концы в пикселях вырезки и сдвиг, пиксели 150 dpi (NaN — без соседей).
        threshold_mm: Порог метрики сдвига соседей, мм (шкала — ``SCALE_FACTOR`` порогов).
        scale: Увеличение панели.

    Returns:
        Панель BGR.
    """
    canvas = _panel(crop_b, scale)
    layer = canvas.copy()
    top = SCALE_FACTOR * threshold_mm * WORK_DPI / 25.4
    thickness = max(2, int(round(2 * scale)))
    # Сначала серые (без соседей), потом цветные по возрастанию сдвига: сильные поверх слабых.
    order = sorted(range(len(segments)), key=lambda i: -1.0 if np.isnan(segments[i, 4]) else segments[i, 4])
    for i in order:
        x0, y0, x1, y1, shift = segments[i]
        if np.isnan(shift):
            colour = NO_NEIGHBOUR_COLOUR
        else:
            level = int(np.clip(shift / max(top, 1e-6), 0.0, 1.0) * 255)
            colour = tuple(int(v) for v in cv2.applyColorMap(np.uint8([[level]]), cv2.COLORMAP_JET)[0, 0])
        cv2.line(layer, (int(round(x0 * scale)), int(round(y0 * scale))), (int(round(x1 * scale)), int(round(y1 * scale))),
                 colour, thickness, cv2.LINE_AA)  # fmt: skip
    cv2.addWeighted(layer, SEGMENT_ALPHA, canvas, 1 - SEGMENT_ALPHA, 0, canvas)
    return canvas


def scale_strip(width: int, threshold_mm: float, title: str = "") -> np.ndarray:
    """Полоса шкалы цвета: 0 … 3 порога, с отметками в мм и подписью шкалы (в поле картинки, под листом)."""
    height = 46
    strip = np.full((height, width, 3), 255, np.uint8)
    bar = cv2.applyColorMap(np.tile(np.linspace(0, 255, width - 40).astype(np.uint8), (14, 1)), cv2.COLORMAP_JET)
    # Цвет на бумаге — с той же прозрачностью, что на карте.
    bar = (HEAT_ALPHA * bar + (1 - HEAT_ALPHA) * 255).astype(np.uint8)
    strip[6:20, 20 : width - 20] = bar
    top = SCALE_FACTOR * threshold_mm
    for k in range(4):
        x = 20 + int(round(k / 3 * (width - 41)))
        cv2.line(strip, (x, 20), (x, 26), (20, 20, 20), 1)
        cv2.putText(
            strip,
            f"{k / 3 * top:.2f} mm",
            (min(max(0, x - 22), width - 62), 40),  # последняя отметка не должна уйти за край полосы
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    if title:
        return np.vstack([header_strip([title], width), strip])
    return strip


def aad_sheet(
    measure: FlowMeasure, title: str, threshold_mm: float, mode: str, verdict: str = "", shift_threshold_mm: float = 1.0
) -> np.ndarray:
    """Лист одной рамки: «B | A в кадре B | карта AAD поверх A», шапка с мерами и порогами, легенда и шкала.

    Args:
        measure: Мера рамки с картами (``measure_box(..., keep_maps=True)``).
        title: Страница и рамка.
        threshold_mm: Порог AAD, мм.
        mode: Режим выравнивания (в шапку).
        verdict: Вердикт страницы v17 и правило (в шапку), если есть.
        shift_threshold_mm: Порог сдвига соседних отрезков, мм (шкала панели отрезков).

    Returns:
        Картинка BGR.
    """
    crop_b, crop_a = measure.crop_b, measure.crop_a
    # Широкая рамка (схема во всю полосу) — панели одна под другой шириной WIDE_WIDTH; узкая — в ряд по PANEL_WIDTH.
    # Вырезка не уменьшается: мелкие черты схемы должны читаться.
    wide = crop_b.shape[1] > crop_b.shape[0]
    scale = max(1.0, (WIDE_WIDTH if wide else PANEL_WIDTH) / crop_b.shape[1])
    panels = [_panel(crop_b, scale), _panel(crop_a, scale)]
    titles = ["без коррекции (B)", "с коррекцией (A), в кадре B"]
    if measure.heat is not None:
        panels.append(heat_panel(crop_a, measure.heat, threshold_mm, scale))
        titles.append("карта отклонений AAD поверх A")
    if measure.segments is not None and len(measure.segments):
        panels.append(segments_panel(crop_b, measure.segments, shift_threshold_mm, scale))
        titles.append("отрезки LSD (любой ориентации) на B: сдвиг поперёк относительно соседних параллельных")
    titled = [np.vstack([header_strip([name], panel.shape[1]), panel]) for panel, name in zip(panels, titles)]
    if wide:
        gap = np.full((12, titled[0].shape[1], 3), 255, np.uint8)
        canvas = np.vstack([part for item in titled for part in (item, gap)][:-1])
    else:
        gap = np.full((titled[0].shape[0], 12, 3), 255, np.uint8)
        canvas = np.hstack([part for item in titled for part in (item, gap)][:-1])
    width = canvas.shape[1]
    x0, y0, x1, y1 = measure.box
    flag = "≥ порога" if measure.aad_mean_mm >= threshold_mm else "ниже порога"
    header = [
        f"{title}   рамка line art разбора B, вырезка с полями 3 мм: ({x0:.0f}, {y0:.0f}) – ({x1:.0f}, {y1:.0f}) px 150 dpi, "
        f"{px_to_mm(x1 - x0, WORK_DPI):.0f}×{px_to_mm(y1 - y0, WORK_DPI):.0f} мм"
        + (f"   вердикт v17: {verdict}" if verdict else ""),
        f"AAD {measure.aad_mean_mm:.3f} мм ({flag}; порог {threshold_mm:g} мм, непрощаемая)   p95 {measure.aad_p95_mm:.2f} мм   "
        f"перекос p95 {measure.shear_p95_deg:.1f}°, анизотропия p95 {measure.aniso_p95:.3f} (справочно)",
        f"сдвиг соседних параллельных отрезков (любой ориентации) p95 {measure.seg_rel_mm:.2f} мм (порог {shift_threshold_mm:g} мм, "
        f"непрощаемая);   "
        f"AAD только по линиям {measure.aad_lines_mm:.3f} мм, изгиб отрезков {measure.seg_bend_mm:.3f} мм (справочно)",
        f"выравнивание A → B: {mode}, местный сдвиг ({measure.shift[0]}, {measure.shift[1]}) px; корреляция вырезок "
        f"{measure.ncc:.2f} ({'рисунок найден' if measure.found else 'не найден — AAD не считается'})",
    ]
    legend = [
        LegendEntry("карта AAD: отклонение черты от общего сдвига её строки (горизонтали) или столбца (вертикали), после "
                    "снятия аффинной части рамки; синий — ноль, красный — ≥ 3 порогов AAD", (0, 0, 255), HEAT_ALPHA),
        LegendEntry("отрезок LSD: цвет — p95 поперечного сдвига относительно соседних параллельных отрезков (в пределах "
                    f"{NEIGHBOUR_MM:g} мм); синий — ноль, красный — ≥ 3 порогов сдвига", (0, 0, 255), SEGMENT_ALPHA),
        LegendEntry("отрезок LSD без параллельных соседей (сдвиг не меряется)", NO_NEIGHBOUR_COLOUR, SEGMENT_ALPHA),
    ]  # fmt: skip
    return np.vstack(
        [
            header_strip(header, width),
            canvas,
            scale_strip(width, threshold_mm, "шкала карты AAD"),
            scale_strip(width, shift_threshold_mm, "шкала отрезков LSD (сдвиг относительно соседей)"),
            legend_strip(legend, width, "обозначения"),
        ]
    )


__all__ = ["aad_sheet", "heat_panel", "scale_strip", "segments_panel"]
