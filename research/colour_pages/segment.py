"""Шаг «бумага и страницы» по кэшу v2 с правилами для цветных страниц (рост по палитре, защита контента при любой позе) и мера «контент за бумагой» — как ``pipeline.cpu_pages`` и защита из ``finish_frame``, без записи в кэш."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.finger_removal.text_protection import polygons_to_mask
from ocr_utils.scan_cropping.v2 import pipeline as pl
from ocr_utils.scan_cropping.v2.content_guard import protect_content
from ocr_utils.scan_cropping.v2.copy_back import LAYOUT_PAD_PX
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM, PagePose, PageQuad
from ocr_utils.scan_cropping.v2.paper import smooth_lab
from ocr_utils.scan_cropping.v2.side_fit import EDGE_MAX_KEYSTONE, regularize_sides, snap_sheet_edges
from ocr_utils.scan_cropping.v2.trapezoid import quad_mask
from research.colour_pages.classify import Colour, chroma_of, page_colour
from research.colour_pages.palette import edge_light, grow_by_palette

# Масштаб сегментации пака-6 относительно кадра (``Scale.seg`` при 13.8 px/мм).
PACK6_SEG = 0.5
# Мера «контент за бумагой»: трапеция страницы с запасом, мм.
CONTENT_QUAD_MM = 10.0


class Rule(Enum):
    """Правило для целиком цветных страниц."""

    PALETTE = "palette"  # рост бумаги у края по палитре уверенного центра страницы
    PROTECT_ANY_POSE = "protect"  # защита контента (блоки surya) и у «приподнятой» цветной страницы


@dataclass(frozen=True)
class PageStep:
    """Итог шага «бумага и страницы» кадра.

    Поля:
        img: Кадр сегментации BGR.
        paper: Бумага после выравнивания сторон (bool).
        protected: Добавлено защитой контента (bool).
        pages: Страницы.
        levels: Уровни цветности страниц (по порядку ``pages``).
        palette_added: Добавлено ростом по палитре (bool).
        content: Контент — блоки surya с припуском ∪ краска (bool).
        book: Грубая маска книги (bool).
        bg: Фон (bool).
    """

    img: np.ndarray
    paper: np.ndarray
    protected: np.ndarray
    pages: list[PageQuad]
    levels: list[Colour]
    palette_added: np.ndarray
    content: np.ndarray
    book: np.ndarray
    bg: np.ndarray


def colour_pages_flat(pages: list[PageQuad], levels: list[Colour]) -> list[PageQuad]:
    """Страницы для защиты контента: целиком цветные со сходом до ``EDGE_MAX_KEYSTONE`` считаются лежащими плоско."""
    return [
        (
            replace(p, pose=PagePose.FLAT)
            if level is Colour.FULL and p.quad is not None and p.features.keystone <= EDGE_MAX_KEYSTONE
            else p
        )
        for p, level in zip(pages, levels)
    ]


def page_step(cache_dir: Path, rules: frozenset[Rule], seg: float = PACK6_SEG) -> PageStep | None:
    """Бумага, страницы и защита контента кадра по кэшу с правилами ``rules`` (пусто — как в пакете).

    Защита контента — по грубой маске книги, без вычитания рук (в пакете — «не фон» без рук, ``book_zone`` по
    исходному кадру; для меры «до | после» разница одна и та же).

    Args:
        cache_dir: Папка кадра в кэше v2.
        rules: Правила для целиком цветных страниц.
        seg: Масштаб сегментации относительно кадра.

    Returns:
        Итог или None, если нет файлов.
    """
    if not (cache_dir / "seg.png").exists() or not (cache_dir / "sam3.npz").exists():
        return None
    img = cv2.imread(str(cache_dir / "seg.png"))
    book = dict(np.load(cache_dir / "book.npz"))
    sam3 = dict(np.load(cache_dir / "sam3.npz"))
    paint = pl.paint_on_seg(np.load(cache_dir / "fingers.npz"), img.shape[:2])
    masks = pl.frame_paper(img, book, sam3, pl.REFINE_DEPTH)
    pages = pl.frame_pages(img, book, sam3, paint, masks)
    edge = snap_sheet_edges(masks.paper, pages, book["bg"], book["book"], img)
    if edge.any():
        masks = replace(masks, paper=masks.paper | edge)
        pages = pl.frame_pages(img, book, sam3, paint, masks)
    chroma = chroma_of(img)
    palette_added = np.zeros_like(masks.paper)
    if Rule.PALETTE in rules:
        lab, light = smooth_lab(img), edge_light(img)
        for page in pages:
            if page.quad is not None and page_colour(chroma, page.mask).level is Colour.FULL:
                palette_added |= grow_by_palette(
                    lab, light, page.mask, page.quad, masks.paper, book["book"], book["bg"]
                )
        if palette_added.any():
            masks = replace(masks, paper=masks.paper | palette_added)
            pages = pl.frame_pages(img, book, sam3, paint, masks)
    side = regularize_sides(masks.paper, pages, book["bg"], img, False)
    pages = pl.frame_pages(img, book, sam3, paint, replace(masks, paper=side.paper))
    levels = [page_colour(chroma, p.mask).level for p in pages]
    polys = pl.read_layout(cache_dir / "layout.json") or []
    guard_pages = colour_pages_flat(pages, levels) if Rule.PROTECT_ANY_POSE in rules else pages
    protected = protect_content(side.paper, polys, guard_pages, book["book"], seg, LAYOUT_PAD_PX)
    content = content_mask(polys, img.shape[:2], seg)
    return PageStep(img, side.paper, protected, pages, levels, palette_added, content, book["book"], book["bg"])


def content_mask(polys: list[np.ndarray], shape: tuple[int, int], seg: float) -> np.ndarray:
    """Контент кадра на масштабе сегментации — блоки surya без припуска.

    Краска (``book.ink``) не берётся: она ловит тёмные линии, к печати не относящиеся, — тень кромки у края страницы
    (МТС 1990/02 IMG_0896), пятно у пальца (1991/02 IMG_0622).
    """
    if not polys:
        return np.zeros(shape, bool)
    return polygons_to_mask(shape, [np.asarray(p, np.float32) * seg for p in polys], 0) > 0


def colour_region(steps: list[PageStep], levels: frozenset[Colour]) -> np.ndarray:
    """Где мерить контент за бумагой: трапеции страниц уровней ``levels`` по всем шагам с запасом ``CONTENT_QUAD_MM``
    — одна область для «было» и «стало» (рост по палитре расширяет трапеции).

    Args:
        steps: Итоги шагов одного кадра.
        levels: Уровни цветности страниц.

    Returns:
        Маска (bool).
    """
    shape = steps[0].paper.shape
    near = np.zeros(shape, np.uint8)
    for step in steps:
        for page, level in zip(step.pages, step.levels):
            if level in levels and page.quad is not None:
                near |= quad_mask(page.quad, shape).astype(np.uint8)
    k = 2 * int(round(CONTENT_QUAD_MM * SEG_PX_PER_MM)) + 1
    return cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def content_outside(step: PageStep, region: np.ndarray) -> float:
    """Контент за бумагой в ``region``, мм²: блоки surya в книге, не фон, вне ``paper ∪ protected``.

    Args:
        step: Итог шага.
        region: Где мерить (``colour_region``).

    Returns:
        Площадь, мм² (масштаб сегментации).
    """
    lost = step.content & step.book & ~step.bg & ~(step.paper | step.protected) & region
    return float(lost.sum() / SEG_PX_PER_MM**2)
