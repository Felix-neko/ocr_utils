"""Кадры прогона scan_cropping v2 для стенда: итоговый прямоугольник, маски бумаги и защиты в его координатах, ключ кэша."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.fill import paper_bbox
from ocr_utils.scan_cropping.v2.pipeline import to_frame


@dataclass(frozen=True)
class FrameMasks:
    """Маски кадра в размере кадра.

    Поля:
        paper: Итоговая бумага (опора закраса).
        protected: Защита контента (добавлено сверх бумаги).
        box: Рамка вырезки ``(x0, y0, x1, y1)`` — осевая рамка ``paper | protected``, или None.
    """

    paper: np.ndarray
    protected: np.ndarray
    box: tuple[int, int, int, int] | None

    def crop(self, mask: np.ndarray) -> np.ndarray:
        """Маска в координатах прямоугольника вырезки."""
        x0, y0, x1, y1 = self.box
        return mask[y0:y1, x0:x1]


def key_of(rel: Path) -> str:
    """Ключ кадра в кэше v2: относительный путь без расширения, ``/`` → ``__`` (как ``FrameJob.key``)."""
    return "__".join(rel.with_suffix("").parts)


def frame_shape(cache: Path) -> tuple[int, int]:
    """Размер кадра ``(H, W)`` по маске закраса пальцев в кэше (она в размере кадра).

    Args:
        cache: Папка кадра в кэше v2.

    Returns:
        ``(H, W)``.
    """
    with np.load(cache / "fingers.npz") as data:
        return tuple(data["paint"].shape[:2])


def load_masks(cache: Path) -> FrameMasks:
    """Бумага и защита кадра из ``paper.npz`` кэша v2, переведённые в размер кадра (как в ``finish_frame``).

    Args:
        cache: Папка кадра в кэше v2.

    Returns:
        Маски кадра.
    """
    shape = frame_shape(cache)
    with np.load(cache / "paper.npz") as data:
        paper_seg = data["paper"]
        protected_seg = data["protected"] if "protected" in data else np.zeros_like(paper_seg)
    # Как в finish_frame: «бумага ∪ защита» переводится одной маской, бумага — отдельно.
    visible = to_frame(paper_seg | protected_seg, shape)
    paper = to_frame(paper_seg, shape)
    return FrameMasks(paper, visible & ~paper, paper_bbox(visible))


def read_image(path: Path) -> np.ndarray:
    """Картинка BGR uint8 (TIFF, PNG, JPEG) с путём в юникоде."""
    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"не читается: {path}")
    return img
