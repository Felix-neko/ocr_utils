"""Закрас не-бумаги без участия краски у края: маска подозрительного в полосе бумаги у края (медиана и разброс участков), pull-push по пирамиде с опорой на чистую бумагу, раздельно по страницам с растушёвкой."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

# Глубина полосы бумаги у края, по которой ищется краска и берётся цвет бумаги, мм.
BAND_MM = 10.0
# Размер участка периферии (клетки сетки), мм.
CELL_MM = 25.0
# Наименьшее число пикселей полосы в участке, чтобы его статистика считалась.
CELL_MIN_PX = 30
# Подозрительное отклонение от медианы участка: ΔE больше max(абсолютного порога, k·робастной σ).
SUSPECT_DE = 10.0
SUSPECT_K = 4.0
# Подозрительно темнее медианы участка по L (Lab, 0..100).
SUSPECT_DARK_L = 10.0
# Расширение маски подозрительного, мм.
SUSPECT_GROW_MM = 1.0
# Растушёвка закрасов двух страниц у продолжения стыка, мм.
FEATHER_MM = 8.0


def to_lab(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → Lab float32 (L 0..100, a и b в единицах CIE)."""
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


@dataclass(frozen=True)
class Periphery:
    """Разбор полосы бумаги у края на ¼ копии.

    Поля:
        lab: Lab ¼ копии.
        band: Полоса бумаги у края (``BAND_MM``).
        suspect: Подозрительное в полосе (краска, печать, край обложки) с расширением.
        page: Номер страницы пикселя (0 — левая, 1 — правая; при одной странице — нули).
        medians: Медиана Lab участка, отображённая на пиксели полосы (H, W, 3), NaN вне полосы.
        sigma: Робастная σ (по MAD) яркости бумаги участка, на пиксели полосы, уровни BGR.
    """

    lab: np.ndarray
    band: np.ndarray
    suspect: np.ndarray
    page: np.ndarray
    medians: np.ndarray
    sigma: np.ndarray


def page_map(shape: tuple[int, int], spine_x: float | None) -> np.ndarray:
    """Номер страницы каждого пикселя по абсциссе стыка.

    Args:
        shape: ``(H, W)``.
        spine_x: Абсцисса стыка или None.

    Returns:
        Массив int8: 0 — левее стыка, 1 — правее.
    """
    page = np.zeros(shape, np.int8)
    if spine_x is not None:
        page[:, int(round(spine_x)) :] = 1
    return page


def periphery(small: np.ndarray, paper: np.ndarray, page: np.ndarray, mm: float) -> Periphery:
    """Полоса бумаги у края, статистика участков и маска подозрительного (вариант «маска прозрачности»).

    Полоса ``BAND_MM`` внутрь от края бумаги режется сеткой ``CELL_MM`` и по страницам; в каждом участке — медиана
    Lab и робастный разброс (MAD). Подозрительно: ΔE от медианы больше ``max(SUSPECT_DE, SUSPECT_K·σ)`` или темнее
    медианы на ``SUSPECT_DARK_L``. Цветная сплошная печать страницы — «бумага» своего участка (медиана местная).

    Args:
        small: ¼ копия BGR.
        paper: Маска бумаги ¼ копии.
        page: Номер страницы пикселя.
        mm: px/мм ¼ копии.

    Returns:
        Разбор периферии.
    """
    lab = to_lab(small)
    h, w = paper.shape
    inside = ndimage.distance_transform_edt(paper)
    band = paper & (inside <= BAND_MM * mm)
    cell_px = max(int(round(CELL_MM * mm)), 4)
    wc = -(-w // cell_px)
    ys, xs = np.nonzero(band)
    medians = np.full((h, w, 3), np.nan, np.float32)
    sigma = np.full((h, w), np.nan, np.float32)
    suspect = np.zeros((h, w), bool)
    if len(ys):
        # Участок = клетка сетки × страница.
        cell = ((ys // cell_px) * wc + xs // cell_px) * 2 + page[ys, xs]
        order = np.argsort(cell, kind="stable")
        ys, xs, cell = ys[order], xs[order], cell[order]
        starts = np.r_[0, np.nonzero(np.diff(cell))[0] + 1]
        ends = np.r_[starts[1:], len(cell)]
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for s, e in zip(starts, ends):
            if e - s < CELL_MIN_PX:
                continue
            yy, xx = ys[s:e], xs[s:e]
            vals = lab[yy, xx]
            med = np.median(vals, axis=0)
            de = np.linalg.norm(vals - med, axis=1)
            # Робастная σ отклонения: 1.4826·MAD по ΔE (ΔE ≥ 0, поэтому MAD от медианы ΔE).
            spread = 1.4826 * np.median(np.abs(de - np.median(de)))
            bad = (de > max(SUSPECT_DE, SUSPECT_K * spread)) | (vals[:, 0] < med[0] - SUSPECT_DARK_L)
            # Статистика участка — по неподозрительным пикселям.
            good = ~bad
            if good.sum() >= CELL_MIN_PX // 2:
                med = np.median(vals[good], axis=0)
            g = gray[yy[good], xx[good]] if good.any() else gray[yy, xx]
            medians[yy, xx] = med
            sigma[yy, xx] = 1.4826 * np.median(np.abs(g - np.median(g)))
            suspect[yy[bad], xx[bad]] = True
        # Участки без статистики (мало полосы) — значением ближайшего пикселя полосы с ней (той же страницы не
        # проверяется: такие участки малы и лежат у края своей страницы).
        has = ~np.isnan(medians[..., 0]) & band
        if has.any():
            _, (iy, ix) = ndimage.distance_transform_edt(~has, return_indices=True)
            fill = band & ~has
            medians[fill] = medians[iy[fill], ix[fill]]
            sigma[fill] = sigma[iy[fill], ix[fill]]
    grow = max(int(round(SUSPECT_GROW_MM * mm)), 1)
    suspect = cv2.dilate(suspect.astype(np.uint8), np.ones((2 * grow + 1,) * 2, np.uint8)) > 0
    return Periphery(lab, band, suspect & paper, page, medians, sigma)


def pull_push(img: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Pull-push по пирамиде (Gortler et al. 1996, «The Lumigraph»; Kraus 2009): гладкая интерполяция по весам.

    Pull: на каждом уровне вниз — взвешенное среднее (``pyrDown`` от ``w·I`` и ``w``), вес обрезается до 1.
    Push: вверх — ``I = w·I + (1 − w)·up(I_грубее)``. Пиксели с весом 0 (краска, не-бумага) в среднее не входят.

    Args:
        img: Картинка float32 (H, W, C).
        weight: Вес float32 (H, W) в 0..1.

    Returns:
        Заполненная картинка float32 (H, W, C).
    """
    levels_i, levels_w = [img * weight[..., None]], [weight]
    while min(levels_w[-1].shape) > 2:
        wi, ww = cv2.pyrDown(levels_i[-1]), cv2.pyrDown(levels_w[-1])
        if wi.ndim == 2:
            wi = wi[..., None]
        levels_i.append(wi)
        levels_w.append(ww)
    # Нормированные уровни: цвет = Σw·I / Σw, вес обрезается до 1 после нормировки (по Gortler).
    colours = [li / np.maximum(lw, 1e-6)[..., None] for li, lw in zip(levels_i, levels_w)]
    weights = [np.clip(lw * 4.0**k, 0, 1) if k else np.clip(lw, 0, 1) for k, lw in enumerate(levels_w)]
    out = colours[-1]
    for k in range(len(colours) - 2, -1, -1):
        h, w = colours[k].shape[:2]
        up = cv2.pyrUp(out, dstsize=(w, h))
        if up.ndim == 2:
            up = up[..., None]
        wk = weights[k][..., None]
        out = wk * colours[k] + (1 - wk) * up
    return out


def pull_push_known(img: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Pull-push с весом 1 на ``known`` и 0 вне его (для ``by_page``)."""
    return pull_push(img, known.astype(np.float32))


def by_page(
    small: np.ndarray, known: np.ndarray, page: np.ndarray, paper: np.ndarray, fill_fn, feather_px: float
) -> np.ndarray:
    """Закрас отдельно по каждой странице и плавное смешение по разности расстояний до бумаги страниц.

    Голубая полоса и белая страница разворота не смешиваются вдали от стыка; у продолжения стыка закрасы страниц
    смешиваются на полосе ``feather_px`` (вес правой страницы ``0.5 + (d_лев − d_прав) / (2·feather)``), без резкой
    границы.

    Args:
        small: ¼ копия float32 BGR.
        known: Опора закраса.
        page: Номер страницы пикселя.
        paper: Маска бумаги.
        fill_fn: ``fill_fn(img, known) -> img`` — закрас одной страницы.
        feather_px: Ширина растушёвки, px.

    Returns:
        Закрас float32.
    """
    pages = np.unique(page[paper]) if paper.any() else np.array([0])
    if len(pages) < 2:
        return fill_fn(small, known)
    left = fill_fn(small, known & (page == 0))
    right = fill_fn(small, known & (page == 1))
    d_left = ndimage.distance_transform_edt(~(paper & (page == 0)))
    d_right = ndimage.distance_transform_edt(~(paper & (page == 1)))
    w = np.clip(0.5 + (d_left - d_right) / (2 * feather_px), 0, 1).astype(np.float32)[..., None]
    return (1 - w) * left + w * right


def fill_pull_push(small: np.ndarray, paper: np.ndarray, mm: float, spine_x: float | None) -> np.ndarray:
    """Закрас ¼ копии pull-push: опора — полоса бумаги у края без подозрительного, по страницам с растушёвкой.

    Args:
        small: ¼ копия BGR uint8.
        paper: Маска бумаги ¼ копии (вписанные пиксели).
        mm: px/мм ¼ копии.
        spine_x: Абсцисса стыка страниц на ¼ копии или None.

    Returns:
        ¼ копия BGR uint8, заполненная везде (вклеивать только вне бумаги).
    """
    page = page_map(paper.shape, spine_x)
    per = periphery(small, paper, page, mm)
    support = per.band & ~per.suspect
    filled = by_page(small.astype(np.float32), support, page, paper, pull_push_known, FEATHER_MM * mm)
    return np.clip(filled, 0, 255).astype(np.uint8)
