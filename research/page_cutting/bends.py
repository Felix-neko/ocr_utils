"""Изгибы верха и низа страницы вдали от корешка: прежний сплайн ``page_boundary`` против сплайна с добавленными узлами (``adaptive``) на входе шага ``07_lines`` v2 — расхождение кривых, остатки, оверлей кадра с вырезками в местах наибольшего расхождения."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, framed
from ocr_utils.scan_cropping.v2.page_boundary import FitKind, PageBoundary, SideCurve, fit_boundary
from ocr_utils.scan_cropping.v2.page_sides import Side
from ocr_utils.scan_cropping.v2.page_boundary_adaptive import AdaptiveFit, fit_boundary_adaptive
from research.page_cutting.cases import FrameCase, load_case
from research.page_cutting.runs import RunSpec, read_image

# Цвета (BGR, палитра ``/draw-overlay``): прежний сплайн — тёмно-красный, как на шаге 07_lines v2; новый — синий
# (главная линия); контур маски — зелёный тонко; добавленные узлы — розовые точки.
COLOUR_OLD = (0, 0, 160)
COLOUR_NEW = (220, 90, 20)
COLOUR_MASK = (0, 150, 0)
COLOUR_ADDED = (180, 105, 255)
# Ширина кадра на оверлее, px.
OVERLAY_WIDTH = 1600
# Вырезка у места наибольшего расхождения: ширина и высота на бумаге, мм; увеличение относительно сегментации.
INSET_W_MM = 38.0
INSET_H_MM = 16.0
INSET_ZOOM = 3
# Вырезка печатается, если кривые расходятся больше, мм.
INSET_MIN_MM = 0.15
# Припуск к ядру пальца, где точки контура верха и низа негодны, мм: шов закраса LaMa у ядра; дальше, в остальной
# зоне закраса, край листа настоящий (ХиП 1991/01 IMG_1052: излом низа правой страницы в припуске зоны).
CORE_MARGIN_MM = 3.0
# Точек по стороне при мере расхождения.
DIFF_SAMPLES = 400


@dataclass(frozen=True)
class BendRow:
    """Строка ``bends.csv``: верх или низ одной страницы кадра.

    Поля:
        run, key: Кадр.
        page: ``L``, ``R`` или ``S``.
        side: ``top`` или ``bottom``.
        diff_max_mm: Наибольшее расхождение новой кривой и прежнего сплайна по нормали, мм (с учётом сдвига
            текстом).
        diff_at_mm: Где вдоль стороны это расхождение, мм от левого конца.
        diff_sign: Плюс — новая кривая глубже внутри страницы (режет больше), минус — снаружи.
        resid_old_mm, resid_new_mm: 90-й перцентиль |контур − кривая| по годным точкам, мм.
        added: Число добавленных узлов (0 — кривая совпадает с прежней).
        added_at_mm: Где добавлены узлы, мм от левого конца, через ``;``.
    """

    run: str
    key: str
    page: str
    side: str
    diff_max_mm: float
    diff_at_mm: float
    diff_sign: int
    resid_old_mm: float
    resid_new_mm: float
    added: int
    added_at_mm: str


def depth_along(curve: SideCurve, origin: np.ndarray, normal: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Глубина кривой внутрь страницы от исходной прямой стороны (px) в положениях ``t``, со сдвигом наружу."""
    return (curve.points(t) - origin[None, :]) @ normal


def side_diff(old: SideCurve, new: SideCurve) -> tuple[np.ndarray, np.ndarray]:
    """Расхождение двух кривых одной стороны по нормали на участке, где были точки подгонки.

    Args:
        old: Прежний сплайн (со сдвигом).
        new: Адаптивный (со сдвигом).

    Returns:
        ``(t, new − old)`` — положения и разность глубин, px (плюс — новая глубже внутри).
    """
    lo, hi = max(old.t_lo, new.t_lo), min(old.t_hi, new.t_hi)
    t = np.linspace(lo, hi, DIFF_SAMPLES)
    # Обе глубины — от одной точки (начало прежней стороны): сдвиг текстом у каждой кривой учтён в её точках.
    return t, depth_along(new, old.a, old.normal, t) - depth_along(old, old.a, old.normal, t)


def finger_core_zone(case: FrameCase) -> np.ndarray:
    """Ядро пальца с припуском ``CORE_MARGIN_MM`` на масштабе сегментации (точки верха и низа там негодны).

    Args:
        case: Кадр.

    Returns:
        Маска (bool) размера маски страницы.
    """
    with np.load(case.run.cache_dir / case.key / "fingers.npz") as data:
        core = data["core"]
    shape = case.exclude.shape
    zone = cv2.resize(core.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST)
    r = max(int(round(CORE_MARGIN_MM * case.seg_px_per_mm)), 1)
    return cv2.dilate(zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


def page_name(case: FrameCase, index: int) -> str:
    """Имя страницы кадра: ``L``/``R`` для разворота, ``S`` для одной страницы."""
    return "LR"[index] if len(case.pages) == 2 else "S"


@dataclass(frozen=True)
class PageBends:
    """Обе рамки страницы и подгонки адаптивного сплайна (для оверлея)."""

    name: str
    old: PageBoundary
    new: PageBoundary
    fits: dict[Side, AdaptiveFit]
    mask: np.ndarray


def frame_bends(run: RunSpec, key: str) -> tuple[FrameCase, list[PageBends], list[BendRow]]:
    """Прежняя и адаптивная рамки страниц кадра на входе шага 07_lines v2 (маска и трапеция v2, без сгиба).

    Args:
        run: Прогон.
        key: Ключ кадра.

    Returns:
        ``(кадр, рамки по страницам, строки мер)``.
    """
    case = load_case(run, key)
    ppm = case.seg_px_per_mm
    pages, rows = [], []
    core = finger_core_zone(case)
    for i, page in enumerate(case.pages):
        old = fit_boundary(page.mask, page.quad, page.spine_right, FitKind.SPLINE, ppm, case.exclude, page.text)
        new, fits = fit_boundary_adaptive(
            page.mask, page.quad, page.spine_right, ppm, case.exclude, page.text, edge_exclude=core
        )
        name = page_name(case, i)
        pages.append(PageBends(name, old, new, fits, page.mask))
        for side in (Side.TOP, Side.BOTTOM):
            c_old, c_new = old.curves[side], new.curves[side]
            length_mm = float(np.linalg.norm(c_old.b - c_old.a)) / ppm
            if side not in fits:
                continue
            t, diff = side_diff(c_old, c_new)
            j = int(np.argmax(np.abs(diff)))
            rows.append(
                BendRow(
                    run=run.name,
                    key=key,
                    page=name,
                    side=side.value,
                    diff_max_mm=round(float(abs(diff[j])) / ppm, 2),
                    diff_at_mm=round(float(t[j]) * length_mm, 1),
                    diff_sign=int(np.sign(diff[j])),
                    resid_old_mm=round(c_old.resid_p90_mm, 2),
                    resid_new_mm=round(c_new.resid_p90_mm, 2),
                    added=len(fits[side].added),
                    added_at_mm=";".join(f"{k * length_mm:.0f}" for k in fits[side].added),
                )
            )
    return case, pages, rows


def draw_curves(canvas: np.ndarray, pages: list[PageBends], scale: float, offset: np.ndarray, width: int) -> None:
    """Контуры масок, прежние и адаптивные рамки на холсте (координаты сегментации × ``scale`` − ``offset``)."""
    for p in pages:
        contours, _ = cv2.findContours(p.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, [np.round(c * scale - offset).astype(np.int32) for c in contours], -1, COLOUR_MASK, 1)
        for boundary, colour in ((p.old, COLOUR_OLD), (p.new, COLOUR_NEW)):
            poly = np.round(boundary.polygon() * scale - offset).astype(np.int32)
            cv2.polylines(canvas, [poly], True, colour, width, cv2.LINE_AA)
        # Добавленные узлы — точками на новой кривой.
        for side, fit in p.fits.items():
            for x, y in p.new.curves[side].points(fit.added) * scale - offset:
                cv2.circle(canvas, (int(round(x)), int(round(y))), max(width + 2, 4), COLOUR_ADDED, -1, cv2.LINE_AA)


def inset(seg: np.ndarray, pages: list[PageBends], centre: np.ndarray, ppm: float, caption: str) -> np.ndarray:
    """Вырезка ``INSET_W_MM × INSET_H_MM`` у точки ``centre`` (px сегментации) с увеличением ``INSET_ZOOM``.

    Args:
        seg: Картинка сегментации кадра.
        pages: Рамки страниц.
        centre: Центр вырезки, px сегментации.
        ppm: Плотность сегментации, px/мм.
        caption: Подпись над вырезкой.

    Returns:
        Вырезка BGR с подписью.
    """
    hw, hh = INSET_W_MM / 2 * ppm, INSET_H_MM / 2 * ppm
    x0 = int(np.clip(centre[0] - hw, 0, seg.shape[1] - 2 * hw))
    y0 = int(np.clip(centre[1] - hh, 0, seg.shape[0] - 2 * hh))
    crop = seg[y0 : y0 + int(2 * hh), x0 : x0 + int(2 * hw)]
    big = cv2.resize(crop, None, fx=INSET_ZOOM, fy=INSET_ZOOM, interpolation=cv2.INTER_CUBIC)
    draw_curves(big, pages, INSET_ZOOM, np.array([x0, y0]) * INSET_ZOOM, 2)
    return framed(big, [caption], [])


def draw_bends(case: FrameCase, pages: list[PageBends], rows: list[BendRow]) -> np.ndarray:
    """Оверлей кадра: обе рамки на всём кадре и вырезки у мест расхождения больше ``INSET_MIN_MM``.

    Args:
        case: Кадр.
        pages: Рамки страниц.
        rows: Меры сторон (по ним — места вырезок).

    Returns:
        Картинка BGR с шапкой и легендой в полях.
    """
    seg = read_image(case.run.cache_dir / case.key / "seg.png")
    ppm = case.seg_px_per_mm
    scale = OVERLAY_WIDTH / seg.shape[1]
    canvas = cv2.resize(seg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    draw_curves(canvas, pages, scale, np.zeros(2), 2)
    insets = []
    by_name = {p.name: p for p in pages}
    for row in sorted(rows, key=lambda r: -r.diff_max_mm):
        if row.diff_max_mm < INSET_MIN_MM:
            continue
        curve = by_name[row.page].new.curves[Side(row.side)]
        length_mm = float(np.linalg.norm(curve.b - curve.a)) / ppm
        point = curve.points(np.array([row.diff_at_mm / length_mm]))[0]
        where = "внутрь" if row.diff_sign > 0 else "наружу"
        caption = (
            f"{row.page} {'верх' if row.side == 'top' else 'низ'}, {row.diff_at_mm:.0f} мм от левого конца: "
            f"расхождение {row.diff_max_mm:.2f} мм {where}; остаток p90 {row.resid_old_mm:.2f} → {row.resid_new_mm:.2f} мм"
        )
        insets.append(inset(seg, pages, point, ppm, caption))
    body = canvas
    if insets:
        # Вырезки — по две в ряд под кадром, добиты белым до ширины кадра.
        rows_img = []
        for k in range(0, len(insets), 2):
            pair = insets[k : k + 2]
            h = max(p.shape[0] for p in pair)
            cells = [np.pad(p, ((0, h - p.shape[0]), (0, 12), (0, 0)), constant_values=255) for p in pair]
            line = np.hstack(cells)
            line = cv2.resize(line, None, fx=min(1.0, OVERLAY_WIDTH / line.shape[1]), fy=min(1.0, OVERLAY_WIDTH / line.shape[1]), interpolation=cv2.INTER_AREA)  # fmt: skip
            rows_img.append(np.pad(line, ((8, 0), (0, OVERLAY_WIDTH - line.shape[1]), (0, 0)), constant_values=255))
        body = np.vstack([canvas, *rows_img])
    worst = max((r.diff_max_mm for r in rows), default=0.0)
    legend = [
        LegendEntry("контур маски страницы v2", COLOUR_MASK),
        LegendEntry("прежний сплайн (верх и низ — 8 отрезков, узлы гуще только у корешка)", COLOUR_OLD),
        LegendEntry("новый сплайн верха и низа: прежний + до 2 узлов на пологих изломах края; бока — прежние", COLOUR_NEW),
        LegendEntry("добавленный узел (штраф рядом слегка ослаблен)", COLOUR_ADDED),
    ]  # fmt: skip
    head = [f"{case.run.name} {case.key} — изгибы верха и низа: прежний сплайн против нового",
            f"наибольшее расхождение {worst:.2f} мм; вырезки ×{INSET_ZOOM} — где расхождение больше {INSET_MIN_MM} мм"]  # fmt: skip
    return framed(body, head, legend)
