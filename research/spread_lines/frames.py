"""Кадр прогона v2 из кэша для стенда: страницы, маски книги и рук, кандидаты разделителя (с линией сгиба) и прямые разворота по каждому способу."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.background import hand_mask
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM, PageQuad
from ocr_utils.scan_cropping.v2.pipeline import PaperMasks, frame_pages, paint_on_seg
from ocr_utils.scan_cropping.v2.spread_lines import (
    auto_source,
    finger_hands,
    Line,
    SpineMethod,
    SpreadLines,
    fit_line,
    pages_span,
    spine_candidates,
    spread_lines,
)
from research.page_cutting.fold import find_fold


@dataclass(frozen=True)
class FrameLines:
    """Кадр кэша v2 и прямые разворота по всем способам разделителя.

    Поля:
        key: Ключ кадра в кэше (``<подпапки>__<кадр>``).
        img: Кадр без пальцев на масштабе сегментации (``seg.png``).
        book: Грубая маска книги (после вето фона, если оно было).
        pages: Две страницы с трапециями слева направо.
        lines: Прямые разворота по способу разделителя.
    """

    key: str
    img: np.ndarray
    book: np.ndarray
    pages: list[PageQuad]
    lines: dict[SpineMethod, SpreadLines]


def fold_spine(img: np.ndarray, pages: list[PageQuad]) -> Line | None:
    """Разделитель ``SpineMethod.FOLD``: Хубер по точкам линии сгиба по яркости бумаги (``page_cutting.fold``).

    Args:
        img: Кадр масштаба сегментации, BGR.
        pages: Две страницы.

    Returns:
        Прямая или None (сгиб не найден).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fold = find_fold(gray, pages[0].mask, pages[1].mask, SEG_PX_PER_MM)
    if fold is None:
        return None
    return fit_line(fold.points[:, 0], fold.points[:, 1], *pages_span(pages))


def load_frame(cache: Path, key: str) -> FrameLines | None:
    """Кадр из кэша v2 и его прямые по всем способам; None — страниц с трапецией не две.

    Руки — кожа по цвету на ``seg.png`` (кадр без пальцев: кожа осталась только вне закраса) плюс зона закраса пальцев.

    Args:
        cache: Корень кэша прогона (``--cache`` v2).
        key: Ключ кадра.

    Returns:
        Кадр с прямыми или None.
    """
    work = cache / key
    img = cv2.imread(str(work / "seg.png"))
    book = dict(np.load(work / ("book_veto.npz" if (work / "book_veto.npz").exists() else "book.npz")))
    sam3 = dict(np.load(work / "sam3.npz"))
    with np.load(work / "paper.npz") as data:
        masks = PaperMasks(**{n: data[n] for n in PaperMasks.__dataclass_fields__ if n in data.files})
    fingers = np.load(work / "fingers.npz")
    paint = paint_on_seg(fingers, img.shape[:2])
    pages = [p for p in frame_pages(img, book, sam3, paint, masks) if p.quad is not None]
    if len(pages) != 2:
        return None
    core = cv2.resize(fingers["core"].astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    hands = finger_hands(hand_mask(img, core) & book["book"], paint)
    spines = spine_candidates(pages)
    fold = fold_spine(img, pages)
    if fold is not None:
        spines[SpineMethod.FOLD] = fold
    # У AUTO в подписи прямых — способ, который за ним стоит на этом кадре.
    source = auto_source(spines)
    lines = {
        m: spread_lines(pages, s, source if m is SpineMethod.AUTO else m, book["book"], hands)
        for m, s in spines.items()
    }
    return FrameLines(key, img, book["book"], pages, lines)
