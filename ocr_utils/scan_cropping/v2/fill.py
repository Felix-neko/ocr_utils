"""Закрас до прямоугольника: осевая рамка маски бумаги, всё вне бумаги — Telea, pull-push от чистой бумаги у края (с мягким швом от каймы листа или без), или прежней заливкой v1 (replicate)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.border_fill import SEAM_MM, fill_pull_push
from ocr_utils.scan_cropping.v2.scale import REF_PX_PER_MM, Scale
from ocr_utils.scan_cropping.background_fill import (
    BG_FILL_EROSION_PX,
    CROP_FILL_REPLICATE,
    book_mean_color,
    voronoi_fill,
)
from ocr_utils.scan_cropping.cropping import CROP_FILL_BLUR_PX, CROP_FILL_FADE, crop_pixel_exact
from ocr_utils.scan_cropping.page_detection import WORK_SIDE

# Масштаб, на котором считается заливка Telea и pull-push (на полном кадре они медленные), на калибровочном кадре
# (13.8 px/мм); на кадре другой плотности копия приводится к той же плотности (:func:`fill_factor`), чтобы радиус
# Telea и счётчики пикселей копии значили одно и то же на бумаге. Вклейка — только в не-бумагу.
FILL_SCALE = 0.25
# Радиус Telea на уменьшенной копии, px (копия калибровочной плотности: 9 px ≈ 2.6 мм).
TELEA_RADIUS = 9
# Запас вокруг выемок, которые дозаливаются в replicate: опорная бумага берётся не ближе этого, px калибровочного
# кадра (0.43 мм); на кадре другой плотности пересчитывается через ``Scale.frame_px``.
HOLE_MARGIN_PX = 6


def fill_factor(px_per_mm: float) -> float:
    """Масштаб копии закраса относительно кадра: ¼ на калибровочном кадре, та же плотность копии на любом другом.

    Args:
        px_per_mm: Плотность кадра, px/мм.

    Returns:
        Множитель для ``cv2.resize`` (при 13.8 px/мм — ровно ``FILL_SCALE``).
    """
    return FILL_SCALE * (REF_PX_PER_MM / px_per_mm)


class FillMethod(str, Enum):
    """Способ закраса не-бумаги внутри прямоугольника."""

    # Telea на ¼ копии, вклейка только вне бумаги (стенды ``newspaper_crop`` и ``bound_volumes``).
    TELEA = "telea"
    # Прежняя заливка v1: продление края бумаги по нормали к сторонам рамки, размытие и выцветание к цвету книги
    # (``cropping.crop_pixel_exact`` с ``fill_method=replicate``).
    REPLICATE = "replicate"
    # Pull-push по пирамиде на ¼ копии с опорой на полосу бумаги у края без подозрительного (печать, край обложки,
    # тень), по страницам с растушёвкой (``border_fill``; стенд ``research/border_fill``): без шлейфов и лучей Telea.
    PULL_PUSH = "pull_push"
    # Pull-push с мягким швом: у края закрас начинается с цвета каймы листа (завал, торец) и за ``seam_mm`` уходит к
    # pull-push; печать и чужой цвет в кайме не участвуют (``border_fill.seam_ramp``; стенд ``research/border_fill``,
    # вариант H5: резкий шов на 37 кадрах пака-6 13.4 → 8.9 тыс. мм, шлейфы не выросли).
    PULL_PUSH_SEAM = "pull_push_seam"


@dataclass(frozen=True)
class ReplicateParams:
    """Параметры прежней заливки v1 (см. ``cropping.crop_pixel_exact``).

    Поля:
        blur_px: Наибольшая σ размытия заливки, растущая с расстоянием от бумаги, px.
        fade: Доля выцветания к среднему цвету книги на самом дальнем пикселе (0 — не выцветать).
    """

    blur_px: float = CROP_FILL_BLUR_PX
    fade: float = CROP_FILL_FADE


def fill_non_paper(bgr: np.ndarray, paper: np.ndarray, px_per_mm: float = REF_PX_PER_MM) -> np.ndarray:
    """Дорисовывает не-бумагу цветом соседней бумаги (Telea на уменьшенной копии, вклейка только вне листа).

    Args:
        bgr: Картинка BGR.
        paper: Булева маска бумаги того же размера.
        px_per_mm: Плотность картинки, px/мм: копия для Telea приводится к ¼ калибровочной плотности.

    Returns:
        Картинка BGR, где пиксели бумаги не тронуты.
    """
    if paper.all():
        return bgr
    h, w = paper.shape
    factor = fill_factor(px_per_mm)
    small = cv2.resize(bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    hole = cv2.resize(
        (~paper).astype(np.uint8) * 255, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    # Дыра чуть шире маски: краевые пиксели бумаги после уменьшения смешаны с фоном, их цвет Telea не берёт.
    hole = cv2.dilate(hole, np.ones((3, 3), np.uint8))
    filled = cv2.inpaint(small, hole, TELEA_RADIUS, cv2.INPAINT_TELEA)
    big = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
    out = bgr.copy()
    out[~paper] = big[~paper]
    return out


def fill_non_paper_pull_push(
    bgr: np.ndarray, paper: np.ndarray, px_per_mm: float, spine_x: float | None, seam_mm: float = 0.0
) -> np.ndarray:
    """Дорисовывает не-бумагу pull-push от чистой бумаги у края (``border_fill.fill_pull_push``), вклейка вне листа.

    Args:
        bgr: Картинка BGR.
        paper: Булева маска бумаги того же размера.
        px_per_mm: Плотность картинки, px/мм.
        spine_x: Абсцисса стыка страниц в координатах картинки или None (одна страница).
        seam_mm: Ширина мягкого шва от каймы листа, мм; 0 — без шва.

    Returns:
        Картинка BGR, где пиксели бумаги не тронуты.
    """
    if paper.all():
        return bgr
    h, w = paper.shape
    factor = fill_factor(px_per_mm)
    small = cv2.resize(bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    size = (small.shape[1], small.shape[0])
    # Бумага на ¼ копии — только вписанные пиксели: краевые смешаны с фоном.
    paper_small = cv2.resize(paper.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.99
    spine = None if spine_x is None else spine_x * factor
    filled = fill_pull_push(small, paper_small, px_per_mm * factor, spine, seam_mm)
    big = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
    out = bgr.copy()
    out[~paper] = big[~paper]
    return out


def paper_bbox(paper: np.ndarray) -> tuple[int, int, int, int] | None:
    """Осевая рамка маски бумаги.

    Args:
        paper: Булева маска бумаги.

    Returns:
        ``(x0, y0, x1, y1)`` (правая и нижняя границы — не включительно) или None, если маска пуста.
    """
    ys, xs = np.nonzero(paper)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fill_to_rectangle(
    bgr: np.ndarray,
    paper: np.ndarray,
    method: FillMethod,
    replicate: ReplicateParams = ReplicateParams(),
    box: tuple[int, int, int, int] | None = None,
    px_per_mm: float = REF_PX_PER_MM,
    spine_x: float | None = None,
    seam_mm: float = SEAM_MM,
) -> np.ndarray | None:
    """Вырезка по осевой рамке маски бумаги с закрасом всего, что вне бумаги, выбранным способом.

    Args:
        bgr: Кадр BGR в исходном разрешении (после закраса пальцев).
        paper: Булева маска бумаги того же размера — опора закраса: только её пиксели считаются известными.
        method: Способ закраса.
        replicate: Параметры прежней заливки (только для ``REPLICATE``).
        box: Рамка вырезки ``(x0, y0, x1, y1)``; None — осевая рамка ``paper``. Рамка шире бумаги нужна, когда
            вырезка охватывает и защиту контента, а закрашивать надо от точного края бумаги.
        px_per_mm: Плотность кадра, px/мм (копия закраса, запас выемок и эрозия книги в ``REPLICATE``; по умолчанию —
            калибровочная).
        spine_x: Абсцисса стыка страниц в координатах кадра или None (для ``PULL_PUSH``).
        seam_mm: Ширина мягкого шва, мм (только для ``PULL_PUSH_SEAM``).

    Returns:
        Картинка BGR прямоугольника или None, если бумаги нет.
    """
    if box is None:
        box = paper_bbox(paper)
    if box is None or not paper.any():
        return None
    x0, y0, x1, y1 = box
    if method == FillMethod.TELEA:
        return fill_non_paper(bgr[y0:y1, x0:x1], paper[y0:y1, x0:x1], px_per_mm)
    if method in (FillMethod.PULL_PUSH, FillMethod.PULL_PUSH_SEAM):
        spine = None if spine_x is None else spine_x - x0
        seam = seam_mm if method == FillMethod.PULL_PUSH_SEAM else 0.0
        return fill_non_paper_pull_push(bgr[y0:y1, x0:x1], paper[y0:y1, x0:x1], px_per_mm, spine, seam)
    # Прежняя заливка: рамка без поворота (угол 0, центр в начале координат — ext задаётся прямо в px кадра);
    # известным считается только бумага внутри рамки, всё остальное продлевается от её края.
    mask = paper.astype(np.uint8) * 255
    scale = Scale(px_per_mm)
    out = crop_pixel_exact(
        bgr,
        0.0,
        0.0,
        0.0,
        (x0, y0, x1, y1),
        # Эрозия маски книги перед средним цветом — px калибровочного кадра, пересчитанные по плотности.
        fade_color=book_mean_color(bgr, mask, erosion_px=scale.frame_px(BG_FILL_EROSION_PX)),
        blur_px=replicate.blur_px,
        fade_strength=replicate.fade,
        fill_method=CROP_FILL_REPLICATE,
        content_mask=mask,
    )
    return fill_enclosed(out, paper[y0:y1, x0:x1], px_per_mm)


def fill_then_overlay(
    base: np.ndarray,
    top: np.ndarray,
    paper: np.ndarray,
    keep: np.ndarray,
    method: FillMethod,
    replicate: ReplicateParams = ReplicateParams(),
    px_per_mm: float = REF_PX_PER_MM,
    spine_x: float | None = None,
    seam_mm: float = SEAM_MM,
) -> np.ndarray | None:
    """Закрас до прямоугольника от точного края бумаги, затем поверх — бумага и сохраняемое из ``top``.

    Рамка вырезки — осевая рамка ``paper | keep`` (защита контента расширяет вырезку). Опора закраса — только
    ``paper`` на ``base``: печать, край обложки и кусок пальца, попавшие в защиту или copy back у края, не становятся
    известными пикселями Telea и не тянутся шлейфом. После закраса пиксели ``paper | keep`` берутся из ``top``.

    Args:
        base: Кадр, по которому закрашивается (после закраса пальцев, без copy back), BGR.
        top: Кадр, пиксели которого кладутся поверх закраса (после copy back), BGR того же размера.
        paper: Булева маска бумаги — опора закраса.
        keep: Булева маска того, что сохраняется сверх бумаги (защита контента).
        method: Способ закраса.
        replicate: Параметры прежней заливки (только для ``REPLICATE``).
        px_per_mm: Плотность кадра, px/мм (для ``PULL_PUSH``).
        spine_x: Абсцисса стыка страниц в координатах кадра или None (для ``PULL_PUSH``).
        seam_mm: Ширина мягкого шва, мм (только для ``PULL_PUSH_SEAM``).

    Returns:
        Картинка BGR прямоугольника или None, если бумаги нет.
    """
    visible = paper | keep
    box = paper_bbox(visible)
    out = fill_to_rectangle(base, paper, method, replicate, box, px_per_mm, spine_x, seam_mm)
    if out is None:
        return None
    x0, y0, x1, y1 = box
    # Поверх закраса — бумага и защита из кадра с copy back.
    inside = visible[y0:y1, x0:x1]
    out[inside] = top[y0:y1, x0:x1][inside]
    return out


def fill_enclosed(img: np.ndarray, paper: np.ndarray, px_per_mm: float = REF_PX_PER_MM) -> np.ndarray:
    """Дозаливка выемок не-бумаги, которые прежняя заливка v1 оставляет исходными пикселями.

    ``clamp_to_edge`` продлевает край по столбцу только за первую и последнюю бумагу столбца: пиксель не-бумаги между
    двумя кусками бумаги того же столбца (выемка от пальца или руки на странице, щель у корешка) остаётся как был. В v1
    маска книги почти выпуклая и таких мест нет; маска бумаги v2 — нет. Выемки заливаются цветом ближайшей бумаги
    (``voronoi_fill`` v1).

    Args:
        img: Прямоугольник BGR после ``crop_pixel_exact``.
        paper: Маска бумаги того же размера.
        px_per_mm: Плотность кадра, px/мм (по ней пересчитывается запас ``HOLE_MARGIN_PX``).

    Returns:
        Прямоугольник BGR; бумага и уже залитое не тронуты.
    """
    # Выемка: не-бумага между первой и последней бумагой своего столбца.
    h = paper.shape[0]
    has = paper.any(axis=0)
    first = np.where(has, np.argmax(paper, axis=0), h)
    last = np.where(has, h - 1 - np.argmax(paper[::-1], axis=0), -1)
    rows = np.arange(h)[:, None]
    holes = ~paper & (rows >= first[None, :]) & (rows <= last[None, :])
    if not holes.any():
        return img
    # Опора — всё, кроме выемок с запасом: на уменьшенной копии край выемки смешан с пальцем.
    margin = Scale(px_per_mm).frame_px(HOLE_MARGIN_PX)
    near = cv2.dilate(holes.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)) > 0
    filled = voronoi_fill(img, (~near).astype(np.uint8) * 255, WORK_SIDE)
    out = img.copy()
    out[holes] = filled[holes]
    return out
