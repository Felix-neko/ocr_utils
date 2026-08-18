"""Оценка фокуса по областям строк текста, найденным surya-ocr.

Включается флагом ``--use-surya-lines``; по умолчанию пакет работает по-старому, замеряя
резкость по равномерной сетке тайлов. Зачем это нужно и чем отличается — в докстрингах
``regions`` (наклонные строки), ``measure`` (два балла: оптика и читаемость) и
``zonal_tiles`` (зональная карта по сетке 3x3).
"""

from ocr_utils.defocus_detection.lines.detect import DetectCache, DetectParams, LineDetector
from ocr_utils.defocus_detection.lines.measure import LineMeasurements, measure_lines
from ocr_utils.defocus_detection.lines.regions import Chunk, LineRegion, line_chunks
from ocr_utils.defocus_detection.lines.zonal_tiles import TileZonalResult, tile_zonal

__all__ = [
    "Chunk",
    "LineMeasurements",
    "DetectCache",
    "DetectParams",
    "LineDetector",
    "LineRegion",
    "TileZonalResult",
    "line_chunks",
    "measure_lines",
    "tile_zonal",
]
