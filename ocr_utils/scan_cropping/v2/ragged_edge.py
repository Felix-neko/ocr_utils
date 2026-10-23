"""Детектор «рваного края» внешней (дальней) стороны страницы: извилистость контура маски к сглаживающему сплайну стороны.

Рваный край — маска SAM 3 языками заходит в белые торцы нижних листов и рваную бумагу обложки. Главная мера —
извилистость: длина контура маски, пройденная дословно (контур сглажен окном :data:`SMOOTH_MM`), к длине сплайна
стороны (``page_boundary.fit_side``, ``FitKind.SPLINE``) на том же участке. Справочные — p90 отклонения от сплайна,
доля длины с отклонением больше :data:`SPIKE_MM`, наибольшие выбросы наружу и внутрь. Отсмотр по поясам
(ПХ 1975, ХиП-1991, пак-6; ai_slop/stack_edges/ragged.py): до 1,02 ровно, 1,03–1,05 смешано, от 1,05 почти всё рваное.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.page_boundary import FitKind, SideCurve, fit_side
from ocr_utils.scan_cropping.v2.page_sides import Side, mask_contour, quad_sides, side_profiles

# Порог извилистости: сторона с извилистостью не ниже — рваная (стартовый, до калибровки по разметке поясов).
RAGGED_TORTUOSITY = 1.05
# Окно сглаживания контура перед измерением длины, мм: шум пикселей сегментации не считается рваностью.
SMOOTH_MM = 1.0
# Отклонение от сплайна, с которого точка — «выброс», мм.
SPIKE_MM = 1.0
# Края стороны (доли длины), не входящие в меру: у углов контур уходит на верх и низ.
END_SHARE = 0.05
# Пропуски покрытия (палец, рамка) длиннее этого вычитаются из длины сплайна, мм.
GAP_MM = 2.0
# Меньше годных точек на стороне — меры нет.
MIN_POINTS = 50


@dataclass(frozen=True)
class RaggedMeasure:
    """Меры рваности дальней стороны страницы.

    Поля:
        tortuosity: Извилистость — длина сглаженного контура к длине сплайна на том же участке (1 — ровно).
        path_mm: Длина сглаженного контура дальней стороны, мм.
        spline_mm: Длина сплайна на покрытом контуром участке, мм.
        p90_mm: p90 модуля отклонения контура от сплайна, мм.
        share_gt1: Доля точек с отклонением больше :data:`SPIKE_MM`.
        out_max_mm: Наибольший выброс контура наружу от сплайна (язык), мм.
        in_max_mm: Наибольший выброс внутрь (выемка), мм.
        length_mm: Длина стороны трапеции, мм.
    """

    tortuosity: float
    path_mm: float
    spline_mm: float
    p90_mm: float
    share_gt1: float
    out_max_mm: float
    in_max_mm: float
    length_mm: float

    @property
    def ragged(self) -> bool:
        """Рваная ли сторона по порогу :data:`RAGGED_TORTUOSITY`."""
        return is_ragged(self)


def is_ragged(m: RaggedMeasure | None, threshold: float = RAGGED_TORTUOSITY) -> bool:
    """Рваная ли сторона.

    Args:
        m: Меры стороны или None (мерить было нечего).
        threshold: Порог извилистости.

    Returns:
        True — извилистость не ниже порога; без меры — False.
    """
    return m is not None and np.isfinite(m.tortuosity) and m.tortuosity >= threshold


def measure_far_side(
    mask: np.ndarray, quad: np.ndarray, spine_right: bool, exclude: np.ndarray | None, px_per_mm: float
) -> RaggedMeasure | None:
    """Меры рваности дальней стороны страницы.

    Args:
        mask: Маска страницы (bool), масштаб сегментации.
        quad: Трапеция страницы (4, 2), px той же маски.
        spine_right: Корешок справа от страницы (левая страница разворота) — True, слева — False.
        exclude: Зона, точки в которой не меряются (палец), или None.
        px_per_mm: Масштаб маски, px/мм.

    Returns:
        Меры или None, если годных точек контура на дальней стороне меньше :data:`MIN_POINTS`.
    """
    prof = side_profiles(mask, quad, spine_right, exclude)[Side.FAR]
    centre = np.asarray(quad, np.float64).mean(axis=0)
    curve = fit_side(prof, centre, FitKind.SPLINE, px_per_mm)
    t, d = prof.t[prof.clean], prof.d[prof.clean]
    if len(t) < MIN_POINTS:
        return None
    # Остаток от сплайна, мм: плюс — контур внутри кривой (выемка), минус — наружу (язык).
    r = (d - curve.offset(t)) / px_per_mm
    length_mm = float(np.linalg.norm(prof.b - prof.a)) / px_per_mm
    path_mm, spline_mm = contour_lengths(mask, quad, spine_right, exclude, curve, px_per_mm)
    return RaggedMeasure(
        tortuosity=path_mm / spline_mm if spline_mm > 0 else float("nan"),
        path_mm=path_mm,
        spline_mm=spline_mm,
        p90_mm=float(np.percentile(np.abs(r), 90)),
        share_gt1=float((np.abs(r) > SPIKE_MM).mean()),
        out_max_mm=float(-r.min()),
        in_max_mm=float(r.max()),
        length_mm=length_mm,
    )


def contour_lengths(
    mask: np.ndarray,
    quad: np.ndarray,
    spine_right: bool,
    exclude: np.ndarray | None,
    curve: SideCurve,
    px_per_mm: float,
) -> tuple[float, float]:
    """Длина сглаженного контура маски вдоль дальней стороны и длина сплайна на покрытом участке, мм.

    Точка контура относится к стороне трапеции, ближайшей по расстоянию до отрезка; берутся точки дальней стороны в
    средних ``1 − 2·END_SHARE`` её длины, не в ``exclude`` и не у рамки кадра. Пропуски покрытия длиннее
    :data:`GAP_MM` (палец, рамка) не входят ни в контур, ни в сплайн.

    Args:
        mask: Маска страницы (bool).
        quad: Трапеция страницы (4, 2).
        spine_right: Где корешок (см. :func:`measure_far_side`).
        exclude: Негодная зона (палец) или None.
        curve: Сплайн дальней стороны (``fit_side``).
        px_per_mm: Масштаб, px/мм.

    Returns:
        ``(длина контура, длина сплайна)``, мм; нет точек — ``(nan, nan)``.
    """
    pts = mask_contour(mask)
    if len(pts) < MIN_POINTS:
        return float("nan"), float("nan")
    # Сглаживание контура скользящим средним по кругу (контур замкнут).
    k = max(3, int(round(SMOOTH_MM * px_per_mm)) | 1)
    pad = k // 2
    ker = np.ones(k) / k
    sm = np.stack(
        [np.convolve(np.r_[pts[-pad:, i], pts[:, i], pts[:pad, i]], ker, mode="valid") for i in (0, 1)], axis=1
    )
    # Чья точка: ближайшая сторона трапеции по расстоянию до отрезка.
    sides = quad_sides(quad, spine_right)
    dists = []
    for a, b in sides.values():
        ab = b - a
        tt = np.clip(((pts - a) @ ab) / (ab @ ab + 1e-12), 0.0, 1.0)
        dists.append(np.linalg.norm(pts - (a + tt[:, None] * ab), axis=1))
    owner = np.argmin(np.stack(dists), axis=0)
    far_i = list(sides).index(Side.FAR)
    a, b = sides[Side.FAR]
    ab = b - a
    tt = ((pts - a) @ ab) / (ab @ ab + 1e-12)
    h, w = mask.shape
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    ok = (owner == far_i) & (tt >= END_SHARE) & (tt <= 1 - END_SHARE)
    if exclude is not None:
        ok &= ~exclude[yi, xi]
    ok &= (pts[:, 0] > 2) & (pts[:, 1] > 2) & (pts[:, 0] < w - 3) & (pts[:, 1] < h - 3)
    if not ok.any():
        return float("nan"), float("nan")
    # Длина контура — по отрезкам, у которых обе точки годны.
    seg = np.linalg.norm(np.roll(sm, -1, axis=0) - sm, axis=1)
    path = float(seg[ok & np.roll(ok, -1)].sum())
    # Длина сплайна на том же участке t, без пропусков покрытия длиннее GAP_MM.
    ts = np.linspace(tt[ok].min(), tt[ok].max(), 400)
    spl = float(np.linalg.norm(np.diff(curve.points(ts), axis=0), axis=1).sum())
    covered = np.zeros(len(ts), bool)
    idx = np.minimum(((tt[ok] - ts[0]) / max(ts[-1] - ts[0], 1e-9) * (len(ts) - 1)).astype(int), len(ts) - 1)
    covered[idx] = True
    step_len = spl / (len(ts) - 1)
    gap = np.diff(np.flatnonzero(covered))
    spl -= float(np.sum((gap[gap * step_len > GAP_MM * px_per_mm] - 1) * step_len))
    return path / px_per_mm, spl / px_per_mm


# Окрестность итоговой страницы, в которой берётся сырая маска SAM 3 этой страницы, мм: языки рваного края уходят
# за итоговую сторону на 5–15 мм.
PAGE_GROW_MM = 20.0


def page_raw_mask(raw: np.ndarray, page: np.ndarray, other: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Сырая маска SAM 3 одной страницы: объединение экземпляров в окрестности итоговой страницы, без соседней.

    Рваность меряется по сырой маске: дальше по конвейеру (``plain``, выравнивание сторон) край сглаживается, а
    остаётся не на месте (ПХ 1975/01 IMG_0035 L: raw 1,16, итог 1,01).

    Args:
        raw: Сырое объединение экземпляров SAM 3 (``sam3.npz["raw"]``), bool.
        page: Итоговая маска этой страницы (``PageQuad.mask``).
        other: Итоговая маска соседней страницы разворота.
        px_per_mm: Масштаб, px/мм.

    Returns:
        Маска (bool) той же формы.
    """
    k = max(3, int(round(2 * PAGE_GROW_MM * px_per_mm)) | 1)
    region = cv2.dilate(page.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))) > 0
    return raw.astype(bool) & region & ~other.astype(bool)
