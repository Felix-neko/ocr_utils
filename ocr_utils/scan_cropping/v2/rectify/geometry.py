"""Геометрия разворота для выпрямления: подгоночные области страниц (px сегментации), рамка вырезки итога, формулы surya и перевод точек в px итога."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.scan_cropping.v2.page_frames import PageFrame

# Метка формулы в блоках surya.
EQUATION_LABEL = "Equation"


@dataclass(frozen=True)
class SpreadGeometry:
    """Области страниц кадра и привязка к итогу v2.

    Поля:
        frames: Области левой и правой страницы (px сегментации) или None — страниц с трапецией не две.
        box: Рамка вырезки итога ``(x0, y0, x1, y1)`` в px кадра или None — бумага не найдена.
        seg_scale: Масштаб сегментации относительно кадра по X и Y.
        equations: Блоки surya ``Equation`` (многоугольники в px кадра).
    """

    frames: list[PageFrame] | None
    box: tuple[int, int, int, int] | None
    seg_scale: tuple[float, float]
    equations: tuple[np.ndarray, ...] = ()

    def frame_to_output(self, points: np.ndarray) -> np.ndarray:
        """Точки (N, 2) px кадра → px итога (минус левый верхний угол рамки вырезки)."""
        return np.asarray(points, np.float64) - np.array([self.box[0], self.box[1]], np.float64)

    def to_output(self, points: np.ndarray) -> np.ndarray:
        """Точки (N, 2) px сегментации → px итога (кадр минус левый верхний угол рамки вырезки)."""
        x0, y0 = self.box[0], self.box[1]
        return np.asarray(points, np.float64) / np.array(self.seg_scale) - np.array([x0, y0], np.float64)


def equations_of(polys: list, labels: list) -> tuple[np.ndarray, ...]:
    """Многоугольники формул surya из блоков layout.

    Формулы — только surya ``Equation`` (точность 0.96, полнота 0.92 — ``research/surya_equations``): пиксельного
    детектора формул нет.

    Args:
        polys: Многоугольники блоков surya в px кадра.
        labels: Метки блоков в том же порядке.

    Returns:
        Многоугольники формул (N, 2) float64.
    """
    return tuple(np.asarray(p, np.float64) for p, label in zip(polys, labels) if label == EQUATION_LABEL)
