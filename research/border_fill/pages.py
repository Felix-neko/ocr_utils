"""Закрас страниц второго преобразования v2: площадь повтора концов строк от складки карты у корешка и варианты опоры закраса без краски (Саувола, тёмное относительно бумаги, строже пороги) с мерами клякс и шлейфа."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_sauvola

import ocr_utils.scan_cropping.v2.border_fill as bf
import ocr_utils.scan_cropping.v2.rectify.pages as pages_mod
from ocr_utils.frame_flags.flags import Side, read_flags
from ocr_utils.scan_cropping.v2.border_fill import (
    FEATHER_MM,
    SEAM_MAX_DELTA,
    SEAM_MM,
    by_page,
    page_map,
    periphery,
    pull_push_known,
    rim_support,
    smoothstep_ramp,
)
from ocr_utils.scan_cropping.v2.fill import fill_factor, paper_bbox
from ocr_utils.scan_cropping.v2.pipeline import read_bgr
from ocr_utils.scan_cropping.v2.rectify.text_run import PageControl, TextParams, text_frame
from ocr_utils.scan_cropping.v2.spread_lines import SpineMethod
from ocr_utils.units import MM_PER_INCH
from research.border_fill.measure import blot_area, trail_score
from research.page_rectify.frames import cached_frames

# Саувола для маски краски опоры (вариант B): окно, мм, и k; рост маски, мм.
SAUVOLA_WINDOW_MM = 3.0
SAUVOLA_K = 0.2
SAUVOLA_GROW_MM = 0.5
# Тёмное относительно бумаги участка (вариант C): темнее медианы участка по серому на столько уровней; рост, мм.
DARK_LEVELS = 25.0
DARK_GROW_MM = 1.0
# Строже пороги периферии (вариант D): подозрительно темнее медианы на столько по L, краска в кайме — на столько.
STRICT_SUSPECT_DARK_L = 6.0
STRICT_INK_DARK_L = 12.0
# Масштаб сохраняемых картинок страниц для листов.
THUMB_SCALE = 0.25


class PageVariant(str, Enum):
    """Вариант закраса страницы второго преобразования (код стенда)."""

    # Прежняя область страницы (с полосой повтора у корешка) и боевой закрас.
    FOLD = "0_fold"
    # A — область без складки, боевой закрас (pull-push с мягким швом 5 мм).
    BASE = "A_base"
    # B — A, из опоры и каймы убрана краска по Сауволе.
    SAUVOLA = "B_sauvola"
    # C — A, из опоры и каймы убрано тёмное относительно бумаги участка.
    DARK = "C_dark"
    # D — A со строже порогами периферии и каймы.
    STRICT = "D_strict"


@dataclass(frozen=True)
class PageRow:
    """Строка сводки по странице и варианту.

    Поля:
        key: ``<выпуск>/<кадр>``.
        side: ``L`` или ``R``.
        plain: Обычная страница (не цветная и не тонированная, ``PageState.plain``).
        variant: Код варианта.
        fold_mm2: Площадь полосы повтора у корешка (прежняя область минус новая), мм² — одна на страницу.
        fold_width_mm: Наибольшая ширина полосы повтора в строке, мм.
        blot_mm2: Чёрное по Сауволе (k 0.2) в полосе закраса 3 мм у края, мм².
        blot_soft_mm2: То же при k 0.05 (чувствительнее), мм².
        trail_mm2: Площадь шлейфа ΔE > 15 в закрасе, мм².
        fill_mm2: Площадь закраса, мм².
        message: Ошибка или пусто.
    """

    key: str
    side: str
    plain: bool
    variant: str
    fold_mm2: float = 0.0
    fold_width_mm: float = 0.0
    blot_mm2: float = 0.0
    blot_soft_mm2: float = 0.0
    trail_mm2: float = 0.0
    fill_mm2: float = 0.0
    message: str = ""


def sauvola_ink(small: np.ndarray, mm: float) -> np.ndarray:
    """Краска по Сауволе на ¼ копии с ростом ``SAUVOLA_GROW_MM``.

    Args:
        small: ¼ копия BGR uint8.
        mm: px/мм копии.

    Returns:
        Маска bool.
    """
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    window = max(int(round(SAUVOLA_WINDOW_MM * mm)) | 1, 3)
    ink = gray < threshold_sauvola(gray, window_size=window, k=SAUVOLA_K, r=128.0)
    grow = max(int(round(SAUVOLA_GROW_MM * mm)), 1)
    return cv2.dilate(ink.astype(np.uint8), np.ones((2 * grow + 1,) * 2, np.uint8)) > 0


def dark_ink(small: np.ndarray, medians: np.ndarray, mm: float) -> np.ndarray:
    """Тёмное относительно бумаги участка: серый темнее медианы участка на ``DARK_LEVELS`` с ростом ``DARK_GROW_MM``.

    Args:
        small: ¼ копия BGR uint8.
        medians: Медианы Lab участков периферии (NaN вне полосы).
        mm: px/мм копии.

    Returns:
        Маска bool (только в полосе периферии, где есть медиана).
    """
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # L медианы участка (0..100) → серый 0..255 приближённо: L·2.55.
    paper_gray = medians[..., 0] * 2.55
    with np.errstate(invalid="ignore"):
        ink = gray < paper_gray - DARK_LEVELS
    grow = max(int(round(DARK_GROW_MM * mm)), 1)
    return cv2.dilate(ink.astype(np.uint8), np.ones((2 * grow + 1,) * 2, np.uint8)) > 0


def fill_small(small: np.ndarray, paper: np.ndarray, mm: float, variant: PageVariant) -> np.ndarray:
    """Закрас ¼ копии одной страницы по варианту (повторяет ``border_fill.fill_pull_push`` с мягким швом).

    Args:
        small: ¼ копия BGR uint8.
        paper: Область страницы на копии (вписанные пиксели).
        mm: px/мм копии.
        variant: Вариант опоры.

    Returns:
        ¼ копия BGR uint8, заполненная везде.
    """
    page = page_map(paper.shape, None)
    if variant is PageVariant.STRICT:
        # Строже пороги: подменяются константы модуля на время вызова (стенд, ядро не меняется).
        saved = bf.SUSPECT_DARK_L, bf.INK_DARK_L
        bf.SUSPECT_DARK_L, bf.INK_DARK_L = STRICT_SUSPECT_DARK_L, STRICT_INK_DARK_L
        try:
            per = periphery(small, paper, page, mm)
            rim = rim_support(per, paper, mm)
        finally:
            bf.SUSPECT_DARK_L, bf.INK_DARK_L = saved
    else:
        per = periphery(small, paper, page, mm)
        rim = rim_support(per, paper, mm)
    support = per.band & ~per.suspect
    if variant is PageVariant.SAUVOLA:
        ink = sauvola_ink(small, mm)
        support, rim = support & ~ink, rim & ~ink
    elif variant is PageVariant.DARK:
        ink = dark_ink(small, per.medians, mm)
        support, rim = support & ~ink, rim & ~ink
    feather = FEATHER_MM * mm
    filled = by_page(small.astype(np.float32), support, page, paper, pull_push_known, feather)
    if rim.any():
        # Мягкий шов (как ``border_fill.seam_ramp``, strict): край — pull-push по кайме, отклонение ограничено.
        edge = by_page(small.astype(np.float32), rim, page, paper, pull_push_known, feather)
        diff = edge - filled
        gray = diff @ np.array([0.114, 0.587, 0.299], np.float32)
        diff = diff * np.minimum(1.0, SEAM_MAX_DELTA / np.maximum(np.abs(gray), 1e-6))[..., None]
        dist = ndimage.distance_transform_edt(~paper)
        filled = filled + diff * smoothstep_ramp(dist, SEAM_MM * mm)[..., None]
    return np.clip(filled, 0, 255).astype(np.uint8)


def fill_page(crop: np.ndarray, paper: np.ndarray, px_per_mm: float, variant: PageVariant) -> np.ndarray:
    """Закрас вырезки страницы до прямоугольника по варианту (вклейка только вне области).

    Args:
        crop: Вырезка итога BGR (осевая рамка области).
        paper: Область страницы в вырезке.
        px_per_mm: Плотность, px/мм.
        variant: Вариант опоры.

    Returns:
        Прямоугольник BGR.
    """
    if paper.all():
        return crop
    h, w = paper.shape
    factor = fill_factor(px_per_mm)
    small = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    size = (small.shape[1], small.shape[0])
    paper_small = cv2.resize(paper.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.99
    filled = fill_small(small, paper_small, px_per_mm * factor, variant)
    big = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
    out = crop.copy()
    out[~paper] = big[~paper]
    return out


def all_rows(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Подмена ``pages.unfolded`` для прежней области: складка не исключается."""
    ones = np.ones(xs.shape, bool)
    return ones, ones


def frame_pages(
    output: Path, cache: Path, photos: Path, issue: str, frame: str, px_per_mm: float, thumbs: Path
) -> list[PageRow]:
    """Пересчёт второго преобразования кадра (без GPU) и варианты закраса его страниц.

    Args:
        output: Папка итогов v2 прогона (``output/``).
        cache: Кэш прогона (``cache/``).
        photos: Корень исходных фото набора (выпуски внутри).
        issue: Выпуск (подпапка).
        frame: Имя кадра без расширения.
        px_per_mm: Плотность выпуска, px/мм.
        thumbs: Куда писать ¼ картинки страниц по вариантам (``<выпуск>__<кадр>_<сторона>_<вариант>.jpg``).

    Returns:
        Строки сводки (страница × вариант); при ошибке — одна строка с сообщением.
    """
    key = f"{issue}/{frame}"
    src = photos / issue / f"{frame}.jpg"
    try:
        image = read_bgr(output / issue / f"{frame}.tiff")
        dpi = px_per_mm * MM_PER_INCH
        geometry = cached_frames(src, cache / issue, frame, True, SpineMethod.AUTO)
        if geometry.box is None or geometry.frames is None or len(geometry.frames) != 2:
            return [PageRow(key, "", True, "", message="нет двух областей страниц")]
        flags = read_flags(src)
        controls = tuple(
            PageControl(equalize=flags.equalize(s), field=not flags.ignored(s) and flags.text_rectify(s))
            for s in (Side.LEFT, Side.RIGHT)
        )
        result = text_frame(image, geometry, dpi, TextParams(), controls)
        new_areas = pages_mod.page_areas(result.canvas, result.dense)
        saved = pages_mod.unfolded
        pages_mod.unfolded = all_rows
        try:
            old_areas = pages_mod.page_areas(result.canvas, result.dense)
        finally:
            pages_mod.unfolded = saved
    except Exception as error:  # noqa: BLE001 — стенд: ошибка кадра в сводку, прогон идёт дальше
        return [PageRow(key, "", True, "", message=f"{type(error).__name__}: {error}")]
    rows = []
    mm2 = px_per_mm * px_per_mm
    for i, side in enumerate("LR"):
        old, new = old_areas[i], new_areas[i]
        repeat = old & ~new
        fold_mm2 = float(repeat.sum() / mm2)
        fold_width = float(repeat.sum(axis=1).max() / px_per_mm) if repeat.any() else 0.0
        plain = bool(result.pages[i].plain)
        for variant in PageVariant:
            area = old if variant is PageVariant.FOLD else new
            box = paper_bbox(area)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            crop, paper = result.final[y0:y1, x0:x1], area[y0:y1, x0:x1]
            filled = fill_page(crop, paper, px_per_mm, PageVariant.BASE if variant is PageVariant.FOLD else variant)
            keep = np.zeros_like(paper)
            blots = blot_area(filled, paper, keep, px_per_mm)
            trail = trail_score(filled, paper, keep, px_per_mm)
            rows.append(PageRow(key, side, plain, variant.value, fold_mm2, fold_width, blots[0], blots[1],
                                trail.area_mm2, trail.fill_mm2))  # fmt: skip
            small = cv2.resize(filled, None, fx=THUMB_SCALE, fy=THUMB_SCALE, interpolation=cv2.INTER_AREA)
            name = f"{issue}__{frame}_{side}_{variant.value}.jpg"
            cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(thumbs / name))
    return rows


def frame_pages_task(args: tuple[Path, Path, Path, str, str, float, Path]) -> list[PageRow]:
    """Обёртка :func:`frame_pages` для пула (кортеж аргументов)."""
    return frame_pages(*args)
