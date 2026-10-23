"""Меры рваности дальних сторон страниц кадра кэша v2 по шагам сегментации: сырой SAM 3, ``plain``, итоговая бумага."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM, PageQuad
from ocr_utils.scan_cropping.v2.pipeline import PaperMasks, frame_pages, paint_on_seg
from ocr_utils.scan_cropping.v2.ragged_edge import RaggedMeasure, measure_far_side, page_raw_mask

log = logging.getLogger(__name__)

# Шаги, на которых меряется сторона: сырой SAM 3 (детектор), plain и итог (справочно).
STAGES = ("raw", "plain", "paper")


@dataclass
class CacheFrame:
    """Кадр кэша v2 для мер: кадр сегментации, бумага по шагам, сырой SAM 3, страницы, зона пальцев.

    Поля:
        key: Ключ кадра.
        img: Кадр без пальцев на масштабе сегментации (``seg.png``).
        raw: Сырое объединение экземпляров SAM 3.
        masks: Маски бумаги по шагам (``paper.npz``).
        pages: Страницы с трапециями (левая, правая — для разворота).
        paint: Зона закраса пальцев на масштабе сегментации.
    """

    key: str
    img: np.ndarray
    raw: np.ndarray
    masks: PaperMasks
    pages: list[PageQuad]
    paint: np.ndarray


def load_frame(work: Path) -> CacheFrame:
    """Кадр кэша v2 (папка кадра) для мер.

    Args:
        work: Папка кадра в кэше (``seg.png``, ``book*.npz``, ``sam3.npz``, ``paper.npz``, ``fingers.npz``).

    Returns:
        Кадр; страницы — только с трапецией.
    """
    img = cv2.imread(str(work / "seg.png"))
    book = dict(np.load(work / ("book_veto.npz" if (work / "book_veto.npz").exists() else "book.npz")))
    sam3 = dict(np.load(work / "sam3.npz"))
    with np.load(work / "paper.npz") as data:
        masks = PaperMasks(**{n: data[n] for n in PaperMasks.__dataclass_fields__ if n in data.files})
    paint = paint_on_seg(np.load(work / "fingers.npz"), img.shape[:2])
    pages = [p for p in frame_pages(img, book, sam3, paint, masks) if p.quad is not None]
    return CacheFrame(work.name, img, sam3["raw"].astype(bool), masks, pages, paint)


def stage_page_mask(frame: CacheFrame, stage: str, index: int) -> np.ndarray:
    """Маска страницы ``index`` (0 — левая, 1 — правая) на шаге ``stage``.

    Args:
        frame: Кадр.
        stage: ``raw`` — сырой SAM 3 в окрестности страницы; ``plain`` — то же по маске ``plain``; ``paper`` —
            итоговая маска страницы.
        index: Номер страницы.

    Returns:
        Маска (bool).
    """
    page, other = frame.pages[index].mask, frame.pages[1 - index].mask
    if stage == "paper":
        return page
    src = frame.raw if stage == "raw" else frame.masks.plain
    return page_raw_mask(src, page, other, SEG_PX_PER_MM)


def side_measures(frame: CacheFrame, index: int) -> dict[str, RaggedMeasure | None]:
    """Меры дальней стороны страницы по шагам :data:`STAGES`.

    Args:
        frame: Кадр-разворот из двух страниц.
        index: 0 — левая страница (дальняя сторона слева), 1 — правая.

    Returns:
        Шаг → меры (None — мерить нечего).
    """
    quad, spine_right = frame.pages[index].quad, index == 0
    return {
        s: measure_far_side(stage_page_mask(frame, s, index), quad, spine_right, frame.paint, SEG_PX_PER_MM)
        for s in STAGES
    }


def frame_rows(args: tuple[str, Path, str]) -> list[dict]:
    """Строки CSV двух дальних сторон кадра (для пула).

    Args:
        args: ``(имя набора, кэш, ключ кадра)``.

    Returns:
        По строке на сторону: набор, кэш, ключ, страница, меры по шагам (``<шаг>_<мера>``); не разворот — пусто;
        ошибка — одна строка с ``error``.
    """
    name, cache, key = args
    try:
        frame = load_frame(cache / key)
    except Exception as exc:  # noqa: BLE001 — кадр с ошибкой не роняет прогон
        return [{"set": name, "cache": str(cache), "key": key, "error": repr(exc)}]
    if len(frame.pages) != 2:
        return []
    rows = []
    for index, page in ((0, "L"), (1, "R")):
        row: dict = {"set": name, "cache": str(cache), "key": key, "page": page}
        try:
            for stage, m in side_measures(frame, index).items():
                for k, v in (asdict(m) if m else {}).items():
                    row[f"{stage}_{k}"] = round(float(v), 4)
        except Exception as exc:  # noqa: BLE001
            row["error"] = repr(exc)
        rows.append(row)
    return rows
