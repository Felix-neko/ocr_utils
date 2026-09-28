"""Скан пака по бинаризованным PDF: концы строк у вертикальных сторон блоков с признаками крайнего глифа.

По каждой полосе прогоняется детектор текстовых блоков (движок ``ink``, вторая ось — как в разборе v3) и по
каждой вертикальной стороне каждого блока берётся устойчивое выравнивание (``AlignMethod.ROBUST``). Для
каждого конца строки пишется отклонение от стороны и признаки крайнего глифа строки у этой стороны: размер,
площадь краски, заполнение бокса, зазор до соседнего глифа, положение относительно базовой линии. По
выступам наружу (``resid_mm`` < 0) потом ищутся соринки и отчёркивания, притянувшие конец строки.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks.baseline_axis import line_x_height
from ocr_utils.page_layout.text_blocks.blocks import Row
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.page import AxisKind, PageAnalysis, analyse_gray, render_page
from ocr_utils.page_layout.text_blocks.sides import AlignMethod, SideKind, side_alignment
from research.text_block_specks.pages import PdfPage

# Пиксель рабочей копии в мм.
MM_PER_PX = 25.4 / WORK_DPI


@dataclass(frozen=True)
class EndRecord:
    """Конец строки у вертикальной стороны блока и признаки крайнего глифа (всё в мм, если не сказано иное).

    Attributes:
        key: Ключ полосы.
        pdf: Имя PDF.
        page: Номер страницы PDF с единицы.
        block: Номер блока на полосе (порядок ``analysis.blocks``).
        side: ``left`` или ``right``.
        row: Номер ряда в блоке.
        rows: Сколько рядов в блоке.
        x: Абсцисса конца строки (край краски), пиксели рабочей копии.
        y: Ордината оси строки у конца, пиксели рабочей копии.
        resid_mm: Отклонение от устойчивой кривой стороны; «внутрь блока — плюс», выступ наружу — минус.
        status: Статус конца (``on`` / ``off`` / ``indent``).
        x_h_mm: Высота строчной строки.
        glyph_x_mm: Насколько край крайнего глифа отстоит от края краски ряда (внутрь — плюс).
        last_w_mm, last_h_mm: Ширина и высота бокса крайнего глифа.
        last_area_mm2: Площадь краски крайнего глифа (по бинарной рабочей копии).
        last_fill: Заполнение бокса крайнего глифа краской.
        gap_mm: Зазор от крайнего глифа до ближайшего глифа строки со стороны текста.
        bottom_xh: Низ крайнего глифа относительно базовой линии, в высотах строчной (ниже — плюс).
        top_xh: Верх крайнего глифа относительно верха строчных, в высотах строчной (выше — минус).
    """

    key: str
    pdf: str
    page: int
    block: int
    side: str
    row: int
    rows: int
    x: float
    y: float
    resid_mm: float
    status: str
    x_h_mm: float
    glyph_x_mm: float
    last_w_mm: float
    last_h_mm: float
    last_area_mm2: float
    last_fill: float
    gap_mm: float
    bottom_xh: float
    top_xh: float


FIELDS = [field.name for field in fields(EndRecord)]


def analyse(pdf_dir: Path, page: PdfPage) -> tuple[PageAnalysis, np.ndarray]:
    """Разбор страницы PDF тем же детектором, что в разборе пака v3 (движок ``ink``, вторая ось).

    Args:
        pdf_dir: Каталог PDF варианта (nogeo или geo).
        page: Страница.

    Returns:
        Пара: разбор страницы и серый рендер ``RENDER_DPI`` (для признаков и оверлеев).
    """
    gray = render_page(pdf_dir / page.pdf, page.page)
    analysis = analyse_gray(gray, InkEngine(), name=page.key, page=page.page, variant="nogeo", axis=AxisKind.BODY)
    return analysis, gray


def work_binary(gray: np.ndarray) -> np.ndarray:
    """Бинарная рабочая копия ``WORK_DPI``: краска — ``True`` (порог Оцу, как в ``segment.component_mask``).

    Args:
        gray: Серый рендер ``RENDER_DPI``.

    Returns:
        Маска краски размера рабочей копии.
    """
    size = (round(gray.shape[1] * WORK_DPI / RENDER_DPI), round(gray.shape[0] * WORK_DPI / RENDER_DPI))
    work = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary > 0


def row_glyphs(row: Row) -> np.ndarray:
    """Боксы глифов всех осей ряда одним массивом ``(n, 4)`` (``x0, y0, x1, y1``, пиксели рабочей копии)."""
    parts = [axis.glyphs for axis in row.axes if axis.glyphs is not None and axis.glyphs.shape[0]]
    return np.vstack(parts) if parts else np.zeros((0, 4))


def end_features(row: Row, side: SideKind, binary: np.ndarray) -> dict[str, float]:
    """Признаки крайнего глифа ряда у стороны ``side``.

    Args:
        row: Ряд блока.
        side: ``LEFT`` — крайний слева глиф, ``RIGHT`` — справа.
        binary: Бинарная рабочая копия (:func:`work_binary`).

    Returns:
        Словарь полей :class:`EndRecord` от ``x_h_mm`` до ``top_xh``; при двух глифах и меньше — NaN.
    """
    nan = float("nan")
    empty = {name: nan for name in FIELDS[FIELDS.index("x_h_mm") :]}
    glyphs = row_glyphs(row)
    if glyphs.shape[0] < 3:
        return empty
    x_h = line_x_height(glyphs)
    if x_h <= 0:
        return empty
    # Крайний глиф — с наибольшим правым краем (у правой стороны) или наименьшим левым (у левой).
    right = side is SideKind.RIGHT
    order = np.argsort(-glyphs[:, 2] if right else glyphs[:, 0])
    last, rest = glyphs[order[0]], glyphs[order[1:]]
    # Зазор до ближайшего глифа со стороны текста: от левого края крайнего до правого края остальных.
    gap = last[0] - rest[:, 2].max() if right else rest[:, 0].min() - last[2]
    # Базовая линия ряда — медиана низов глифов строчного класса (высота около x_h), верх строчных — медиана их верхов.
    heights = glyphs[:, 3] - glyphs[:, 1]
    x_class = np.abs(heights - x_h) <= 0.22 * x_h
    base = float(np.median(glyphs[x_class, 3])) if x_class.any() else float(np.median(glyphs[:, 3]))
    top = float(np.median(glyphs[x_class, 1])) if x_class.any() else base - x_h
    # Площадь краски крайнего глифа — пиксели бинарной копии в его боксе.
    x0, y0, x1, y1 = (int(round(v)) for v in last)
    crop = binary[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
    area = float(crop.sum())
    w, h = float(last[2] - last[0]), float(last[3] - last[1])
    edge = row.x1 if right else row.x0
    return {
        "x_h_mm": x_h * MM_PER_PX,
        "glyph_x_mm": ((edge - last[2]) if right else (last[0] - edge)) * MM_PER_PX,
        "last_w_mm": w * MM_PER_PX,
        "last_h_mm": h * MM_PER_PX,
        "last_area_mm2": area * MM_PER_PX**2,
        "last_fill": area / max(1.0, w * h),
        "gap_mm": float(gap) * MM_PER_PX,
        "bottom_xh": (last[3] - base) / x_h,
        "top_xh": (last[1] - top) / x_h,
    }


def page_records(analysis: PageAnalysis, gray: np.ndarray, page: PdfPage) -> list[EndRecord]:
    """Записи концов строк всех блоков страницы по обеим вертикальным сторонам.

    Args:
        analysis: Разбор страницы.
        gray: Серый рендер ``RENDER_DPI``.
        page: Страница.

    Returns:
        Записи :class:`EndRecord`; блоки меньше чем из двух рядов пропускаются (у них нет стороны).
    """
    binary = work_binary(gray)
    out = []
    for block_index, block in enumerate(analysis.blocks):
        for side in (SideKind.LEFT, SideKind.RIGHT):
            alignment = side_alignment(block, side, AlignMethod.ROBUST)
            if alignment is None:
                continue
            for end in alignment.ends:
                row = block.rows[end.row]
                out.append(
                    EndRecord(
                        key=page.key,
                        pdf=page.pdf,
                        page=page.page,
                        block=block_index,
                        side=side.value,
                        row=end.row,
                        rows=len(block.rows),
                        x=round(float(end.point[0]), 1),
                        y=round(float(end.point[1]), 1),
                        resid_mm=round(float(end.resid_mm), 3),
                        status=end.status.value,
                        **{k: round(v, 3) for k, v in end_features(row, side, binary).items()},
                    )
                )
    return out


def scan_page(task: tuple[Path, PdfPage]) -> tuple[PdfPage, list[EndRecord], float, str]:
    """Задача пула: разбор одной страницы и её записи.

    Args:
        task: Пара «каталог PDF, страница».

    Returns:
        ``(страница, записи, секунды, ошибка)``; при ошибке записи пусты, а ошибка — текст исключения.
    """
    pdf_dir, page = task
    started = time.monotonic()
    try:
        analysis, gray = analyse(pdf_dir, page)
        return page, page_records(analysis, gray, page), time.monotonic() - started, ""
    except Exception as error:  # noqa: BLE001 — одна кривая страница не должна ронять прогон пака
        return page, [], time.monotonic() - started, f"{type(error).__name__}: {error}"


def write_header(path: Path) -> None:
    """Создать CSV концов с шапкой, если файла ещё нет."""
    if not path.exists():
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerow(FIELDS)


def append_records(path: Path, records: list[EndRecord]) -> None:
    """Дописать записи в CSV концов."""
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        for record in records:
            writer.writerow([getattr(record, name) for name in FIELDS])


__all__ = [
    "EndRecord",
    "FIELDS",
    "analyse",
    "append_records",
    "end_features",
    "page_records",
    "row_glyphs",
    "scan_page",
    "work_binary",
    "write_header",
]
