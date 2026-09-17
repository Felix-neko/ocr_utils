"""Пары PDF «с коррекцией | без» и рендер их страниц.

Страницы сопоставляются по индексу: оба PDF собраны FineReader из одного промежуточного
PDF, порядок страниц он сохраняет. Число страниц сверяется; выпуск с расхождением
пропускается с записью в отчёт, а не выравнивается наугад (как в
``curved_lines.finereader_compare``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np

from research.geometry_regression import WORK_DPI

# Рендер идёт при 300 dpi (нужен ``line_fit``: центр-линии он берёт с копии 300 dpi),
# рабочая копия 150 dpi делается из него усреднением.
RENDER_DPI = 300


@dataclass(frozen=True)
class PdfPair:
    """Один выпуск в обоих прогонах."""

    name: str  # full_ГГГГ_НН
    geo: Path
    nogeo: Path
    pages: int

    @property
    def year(self) -> str:
        parts = self.name.split("_")
        return parts[1] if len(parts) > 1 else self.name


def page_count(pdf: Path) -> int:
    with fitz.open(pdf) as document:
        return document.page_count


def pair_pdfs(geo_dir: Path, nogeo_dir: Path) -> tuple[list[PdfPair], list[str]]:
    """Пары по имени файла; в замечания — PDF без пары и с разным числом страниц."""
    pairs: list[PdfPair] = []
    notes: list[str] = []
    nogeo_names = {p.name: p for p in nogeo_dir.glob("*.pdf")}
    for geo in sorted(geo_dir.glob("*.pdf")):
        nogeo = nogeo_names.pop(geo.name, None)
        if nogeo is None:
            notes.append(f"{geo.name}: нет версии без коррекции")
            continue
        n_geo, n_nogeo = page_count(geo), page_count(nogeo)
        if n_geo != n_nogeo:
            notes.append(f"{geo.name}: страниц {n_geo} с коррекцией, {n_nogeo} без — выпуск пропущен")
            continue
        pairs.append(PdfPair(geo.stem, geo, nogeo, n_geo))
    for name in sorted(nogeo_names):
        notes.append(f"{name}: нет версии с коррекцией")
    return pairs, notes


def render_gray(document: fitz.Document, index: int, dpi: int = RENDER_DPI) -> np.ndarray:
    """Серый рендер страницы ``index`` (с нуля)."""
    zoom = dpi / 72.0
    pixmap = document[index].get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width).copy()


def to_work(gray: np.ndarray, dpi: int = RENDER_DPI) -> np.ndarray:
    """Рабочая копия ``WORK_DPI`` усреднением (не ближайшим соседом: бинарный рендер
    иначе теряет тонкие штрихи)."""
    factor = WORK_DPI / dpi
    size = (max(1, round(gray.shape[1] * factor)), max(1, round(gray.shape[0] * factor)))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
