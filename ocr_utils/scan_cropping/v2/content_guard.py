"""Защита контента при вырезке: куски блоков surya страниц на блоке (плоских и опущенных) внутри «не фона» без рук — в бумагу."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.scan_cropping.finger_removal.text_protection import polygons_to_mask
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM, PageQuad, page_owner
from ocr_utils.scan_cropping.v2.trapezoid import quad_mask

# Припуск блоков surya при защите — общий с copy back: ``copy_back.LAYOUT_PAD_PX`` (``--layout-pad-mm``).
# Запас трапеции плоской страницы для ограничения защиты, мм (``protect_content(..., limit_to_quads=True)``). Рамка
# рисунка surya бывает шире страницы и захватывает торец блока и зону пальца у края (РИСК 1991/05 IMG_0591: логотип у
# нижнего края — в итог ушла полоса торца). Пока ВЫКЛЮЧЕНО по решению пользователя (2026-10-12): торец в итоге не
# страшен, важно только не захватывать чёрный фон, а его отсекает «не фон»; включать, если фон будет попадать часто.
QUAD_MARGIN_MM = 2.0


def flat_blocks(polys: list[np.ndarray], pages: list[PageQuad], seg: float, shape: tuple[int, int]) -> list[np.ndarray]:
    """Блоки surya, принадлежащие страницам на блоке — плоским и опущенным (по ближайшей странице к центру блока).

    Args:
        polys: Полигоны блоков surya, px кадра.
        pages: Страницы кадра с классом положения.
        seg: Масштаб сегментации относительно кадра.
        shape: ``(H, W)`` масштаба сегментации.

    Returns:
        Полигоны блоков страниц на блоке, px масштаба сегментации.
    """
    small = [np.asarray(p, np.float32) * seg for p in polys]
    if not pages or all(p.pose.on_book for p in pages):
        return small
    owner = page_owner(pages, shape)
    h, w = shape
    out = []
    for poly in small:
        cx, cy = poly.mean(axis=0)
        k = owner[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))]
        if pages[k].pose.on_book:
            out.append(poly)
    return out


def flat_quads(pages: list[PageQuad], shape: tuple[int, int]) -> np.ndarray:
    """Трапеции страниц на блоке (плоских и опущенных) с запасом ``QUAD_MARGIN_MM`` (без трапеции — маска страницы).

    Args:
        pages: Страницы кадра.
        shape: ``(H, W)`` масштаба сегментации.

    Returns:
        Булева маска; без страниц — пустая.
    """
    out = np.zeros(shape, np.uint8)
    for page in pages:
        if not page.pose.on_book:
            continue
        out |= quad_mask(page.quad, shape) if page.quad is not None else page.mask
    k = 2 * int(round(QUAD_MARGIN_MM * SEG_PX_PER_MM)) + 1
    return cv2.dilate(out.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def protect_content(
    paper: np.ndarray,
    polys: list[np.ndarray],
    pages: list[PageQuad],
    zone: np.ndarray,
    seg: float,
    pad_px: tuple[int, int],
    limit_to_quads: bool = False,
) -> np.ndarray:
    """Куски блоков surya страниц на блоке (плоских и опущенных) внутри «не фона» без рук, которых нет в бумаге.

    Args:
        paper: Итоговая маска бумаги (масштаб сегментации).
        polys: Полигоны всех блоков surya, px кадра.
        pages: Страницы кадра с классом положения (``page_pose.page_quads``).
        zone: «Не фон» без рук (масштаб сегментации): грубая маска книги минус руки.
        seg: Масштаб сегментации относительно кадра.
        pad_px: Припуск блоков, px кадра (по X, по Y).
        limit_to_quads: Ограничить защиту трапециями плоских страниц с запасом ``QUAD_MARGIN_MM`` (пока выключено).

    Returns:
        Булева маска «добавлено защитой» (масштаб сегментации); ``paper | она`` — бумага с защищённым контентом.
    """
    # Без страниц (обложка подшивки, бумага не найдена) защищать нечего: иначе «бумагой» станет наклейка или надпись
    # на обложке (РИСК 1991/05 IMG_0702), и кадр обрежется по ней.
    if not polys or not pages:
        return np.zeros_like(paper)
    blocks = flat_blocks(polys, pages, seg, paper.shape)
    if not blocks:
        return np.zeros_like(paper)
    pad = (max(1, round(pad_px[0] * seg)), max(1, round(pad_px[1] * seg)))
    mask = polygons_to_mask(paper.shape, blocks, pad) > 0
    added = mask & zone & ~paper
    return added & flat_quads(pages, paper.shape) if limit_to_quads else added
