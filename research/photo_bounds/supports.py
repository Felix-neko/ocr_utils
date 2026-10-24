"""Стороны фото как опоры выпрямления: выборки ``Samples`` по сторонам кривого четырёхугольника (верх и низ — как оси строк, боковые — как выровненные стороны блоков) и область фото многоугольником вместо рамки для преград разбора и отсева выборок."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.rectify.text_field import PageRegion, RegionKindR, Samples
from research.photo_bounds.bounds import PhotoBound


@dataclass(frozen=True)
class PhotoRegion(PageRegion):
    """Область фото многоугольником: ``box`` — его охват (для преград разбора), ``contains`` — по многоугольнику.

    Поля:
        polygon: Многоугольник фото (M, 2) px холста — кривой четырёхугольник или прямоугольник.
        photo: Найденное фото (стороны, чёткость) или None.
    """

    polygon: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    photo: PhotoBound | None = None

    def contains(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Какие точки (N, 2) внутри многоугольника или ближе ``margin`` px к нему снаружи.

        Args:
            points: Точки px холста.
            margin: Поле, px.

        Returns:
            Маска bool (N,).
        """
        points = np.asarray(points, np.float64).reshape(-1, 2)
        if len(self.polygon) < 3 or not len(points):
            return super().contains(points, margin)
        contour = self.polygon.astype(np.float32).reshape(-1, 1, 2)
        dist = np.array([cv2.pointPolygonTest(contour, (float(x), float(y)), True) for x, y in points])
        return dist >= -margin


def region_of(photo: PhotoBound, quad: bool = True) -> PhotoRegion:
    """Область фото для разбора и отсева: многоугольник (кривой четырёхугольник, если все стороны чёткие, иначе
    уточнённая рамка), охват — рамка многоугольника.

    Args:
        photo: Найденное фото.
        quad: Брать четырёхугольник (иначе — уточнённую рамку даже у чётких фото).

    Returns:
        Область.
    """
    if quad and photo.corners is not None:
        polygon = photo.polygon()
    else:
        x0, y0, x1, y1 = photo.box
        polygon = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    box = (float(polygon[:, 0].min()), float(polygon[:, 1].min()), float(polygon[:, 0].max()),
           float(polygon[:, 1].max()))  # fmt: skip
    return PhotoRegion(RegionKindR.RASTER, box, polygon, photo)


def photo_samples(photo: PhotoBound, dpi: float) -> tuple[Samples, Samples, int]:
    """Выборки поля по сторонам фото — боевой ``text_field.photo_side_samples`` (верх и низ — ``dy`` как оси строк,
    боковые — ``dx`` как выровненные стороны блоков).

    Args:
        photo: Найденное фото (px холста).
        dpi: Разрешение холста.

    Returns:
        ``(выборки dx, выборки dy, число горизонтальных сторон)``.
    """
    from ocr_utils.scan_cropping.v2.rectify.text_field import photo_side_samples

    return photo_side_samples(photo, dpi)
