"""Варианты закраса не-бумаги до прямоугольника: Telea (как в v2), Telea без краски у края, pull-push по пирамиде, цвет участков с размытием, гармоническое продолжение, зерно бумаги, pull-push с мягким швом у края."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.scan_cropping.v2.fill import FILL_SCALE, TELEA_RADIUS
from ocr_utils.scan_cropping.v2.border_fill import (
    BAND_MM,
    FEATHER_MM,
    Periphery,
    by_page,
    page_map,
    periphery,
    pull_push,
    SEAM_MAX_DELTA,
    pull_push_known,
    seam_ramp,
)

# Размытие заливки цветом участков, мм (σ гаусса).
REGION_BLUR_MM = 6.0
# Число итераций Якоби гармонического продолжения (на ¼ копии, начиная с pull-push).
HARMONIC_ITERS = 300
# Доля робастной σ бумаги, с которой накладывается зерно.
GRAIN_SHARE = 0.8


class Variant(str, Enum):
    """Вариант закраса (код стенда)."""

    # A — как сейчас: Telea по «бумага ∪ защита».
    TELEA_KEEP = "A_telea_keep"
    # B — Telea по бумаге, защита поверх (основной пакет после правки).
    TELEA_PAPER = "B_telea_paper"
    # C — Telea по бумаге без подозрительного у края (подозрительное стало дырой).
    TELEA_CLEAN = "C_telea_clean"
    # D — pull-push по пирамиде: веса — бумага полосы у края без подозрительного, по страницам.
    PULL_PUSH = "D_pull_push"
    # E — робастный цвет участка у ближайшего края, размытый.
    REGION = "E_region"
    # F — гармоническое (Лаплас) продолжение от бумаги без подозрительного, старт — pull-push.
    HARMONIC = "F_harmonic"
    # G — pull-push плюс зерно бумаги.
    PULL_PUSH_GRAIN = "G_pull_push_grain"
    # H — pull-push с рампой от цвета каймы листа к закрасу на 3, 5, 8 мм (без печати и чужого цвета в кайме).
    SEAM_3 = "H3_seam"
    SEAM_5 = "H5_seam"
    SEAM_8 = "H8_seam"
    # H5n — рампа 5 мм без фильтра каймы и ограничения отклонения (нужен ли фильтр).
    SEAM_5_LOOSE = "H5n_seam_loose"
    # H5w — рампа 5 мм со строгим фильтром каймы и потолком отклонения 40 уровней вместо 25.
    SEAM_5_WIDE = "H5w_seam_wide"


# Вариант H: ширина рампы шва, мм; строгий ли фильтр каймы; потолок отклонения края от закраса, уровни.
SEAM_PARAMS = {
    Variant.SEAM_3: (3.0, True, SEAM_MAX_DELTA),
    Variant.SEAM_5: (5.0, True, SEAM_MAX_DELTA),
    Variant.SEAM_8: (8.0, True, SEAM_MAX_DELTA),
    Variant.SEAM_5_LOOSE: (5.0, False, SEAM_MAX_DELTA),
    Variant.SEAM_5_WIDE: (5.0, True, 40.0),
}


@dataclass(frozen=True)
class FillInput:
    """Вход закраса одного прямоугольника (полный масштаб).

    Поля:
        base: Кадр без пальцев, без copy back, BGR uint8 (рамка вырезки).
        top: Кадр с copy back — его пиксели кладутся поверх бумаги и защиты.
        paper: Маска бумаги.
        keep: Маска защиты контента сверх бумаги.
        px_per_mm: Плотность, px/мм.
        spine_x: Абсцисса стыка страниц в координатах прямоугольника или None (одна страница).
    """

    base: np.ndarray
    top: np.ndarray
    paper: np.ndarray
    keep: np.ndarray
    px_per_mm: float
    spine_x: float | None = None


def telea(small: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Telea по ¼ копии: неизвестно всё вне ``known`` (дыра расширена на 1 px, как в ``fill_non_paper``)."""
    hole = cv2.dilate((~known).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    return cv2.inpaint(small, hole, TELEA_RADIUS, cv2.INPAINT_TELEA)


def harmonic(img: np.ndarray, known: np.ndarray, iters: int) -> np.ndarray:
    """Гармоническое продолжение: итерации Якоби на неизвестном, известное держится.

    Args:
        img: Начальное приближение float32 (H, W, C) (pull-push).
        known: Маска известного.
        iters: Число итераций.

    Returns:
        Картинка float32.
    """
    out = img.copy()
    fixed = img[known]
    kernel = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], np.float32)
    for _ in range(iters):
        out = cv2.filter2D(out, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        out[known] = fixed
    return out


def region_colour(per: Periphery, paper: np.ndarray, mm: float) -> np.ndarray:
    """Вариант E: медиана участка (Lab → BGR) у ближайшего пикселя полосы, затем размытие ``REGION_BLUR_MM``.

    Args:
        per: Разбор периферии.
        paper: Маска бумаги ¼ копии.
        mm: px/мм.

    Returns:
        Закрас float32 BGR 0..255.
    """
    has = per.band & ~np.isnan(per.medians[..., 0])
    _, (iy, ix) = ndimage.distance_transform_edt(~has, return_indices=True)
    lab = per.medians[iy, ix]
    bgr = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR) * 255
    sigma = REGION_BLUR_MM * mm
    return cv2.GaussianBlur(bgr, (0, 0), sigma)


def add_grain(out: np.ndarray, region: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Зерно бумаги в закрасе: гауссов шум с σ бумаги, сглаженный на 1 px (полный масштаб).

    Args:
        out: Прямоугольник BGR uint8.
        region: Где накладывать (закрас).
        sigma: σ зерна, уровни.
        seed: Зерно генератора (детерминированно).

    Returns:
        Прямоугольник BGR uint8.
    """
    if sigma <= 0 or not region.any():
        return out
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, out.shape[:2]).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.8)
    res = out.astype(np.float32)
    res[region] += noise[region][:, None]
    return np.clip(res, 0, 255).astype(np.uint8)


def fill_variant(inp: FillInput, variant: Variant) -> np.ndarray:
    """Закрас прямоугольника вариантом стенда: всё вне бумаги и защиты закрашено, бумага и защита — из ``top``.

    Args:
        inp: Вход.
        variant: Вариант.

    Returns:
        Прямоугольник BGR uint8.
    """
    h, w = inp.paper.shape
    visible = inp.paper | inp.keep
    small = cv2.resize(inp.base, None, fx=FILL_SCALE, fy=FILL_SCALE, interpolation=cv2.INTER_AREA)
    size = (small.shape[1], small.shape[0])
    mm = inp.px_per_mm * FILL_SCALE
    # Бумага на ¼ копии — по вписанным пикселям (INTER_AREA доли ≥ 0.99): краевые пиксели смешаны с фоном.
    paper_s = cv2.resize(inp.paper.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.99
    if variant == Variant.TELEA_KEEP:
        # Как в v2 до правки: опора — бумага ∪ защита на кадре с copy back.
        small_top = cv2.resize(inp.top, size, interpolation=cv2.INTER_AREA)
        vis_s = cv2.resize(visible.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.99
        filled = telea(small_top, vis_s).astype(np.float32)
    elif variant == Variant.TELEA_PAPER:
        filled = telea(small, paper_s).astype(np.float32)
    else:
        page = page_map(paper_s.shape, None if inp.spine_x is None else inp.spine_x * FILL_SCALE)
        per = periphery(small, paper_s, page, mm)
        known = paper_s & ~per.suspect
        if variant == Variant.TELEA_CLEAN:
            filled = telea(small, known).astype(np.float32)
        elif variant == Variant.REGION:
            filled = region_colour(per, paper_s, mm)
        else:
            # Pull-push: опора — полоса у края без подозрительного (глубже полосы текст страницы не участвует).
            support = per.band & ~per.suspect
            img = small.astype(np.float32)
            filled = by_page(img, support, page, paper_s, pull_push_known, FEATHER_MM * mm)
            if variant == Variant.HARMONIC:
                filled = by_page(
                    filled, support, page, paper_s, partial(harmonic, iters=HARMONIC_ITERS), FEATHER_MM * mm
                )
            elif variant in SEAM_PARAMS:
                seam_mm, strict, max_delta = SEAM_PARAMS[variant]
                filled = seam_ramp(small, filled, paper_s, page, per, mm, seam_mm, FEATHER_MM * mm, strict, max_delta)
    big = cv2.resize(np.clip(filled, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    out = big
    if variant == Variant.PULL_PUSH_GRAIN:
        # σ на ¼ копии занижена усреднением 4×4; на полном масштабе берём σ высоких частот бумаги полосы.
        sig = full_grain_sigma(inp.base, inp.paper, inp.px_per_mm) * GRAIN_SHARE
        out = add_grain(out, ~visible, sig)
    out = out.copy()
    out[visible] = inp.top[visible]
    return out


def full_grain_sigma(bgr: np.ndarray, paper: np.ndarray, px_per_mm: float) -> float:
    """σ зерна бумаги на полном масштабе: робастный разброс высоких частот серого в полосе бумаги у края.

    Args:
        bgr: Прямоугольник BGR.
        paper: Маска бумаги.
        px_per_mm: Плотность.

    Returns:
        σ, уровни.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 2.0)
    depth = int(BAND_MM * px_per_mm)
    eroded = cv2.erode(paper.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    inner = cv2.erode(paper.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=depth // 2) > 0
    band = eroded & ~inner
    if band.sum() < 100:
        return 0.0
    v = hp[band]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))
