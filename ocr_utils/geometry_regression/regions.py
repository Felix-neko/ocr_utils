"""Области страницы на рендере без коррекции: line art (с таблицами) и строки текста.

Рамки таблиц и line art с v13 даёт единый разбор ``ocr_utils.page_layout`` — тот же, что на
этапе ``detect`` и в правке текстового слоя: детектор таблиц, блоки surya (из кэша по
варианту ``fr_nogeo``, GPU в воркерах не нужен) и связные пятна/скопления линеек, всё через
пиксельную проверку. Пиксельный запасной ход (:func:`lineart_boxes`, одни пятна и линейки,
как в v12) остаётся для страниц без surya и для тестов. Строки текста — сегментация
``curved_lines.detectors.line_fit.line_samples``, она же кормит попарные метрики строк и
кромки колонок.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ocr_utils.page_layout.line_art.features import analyse_gray, params_for_dpi
from ocr_utils.scan_markup.curved_lines.detectors.line_fit import LineSample, line_samples
from ocr_utils.scan_markup.curved_lines.fitting import LineFit, fit_line
from ocr_utils.page_layout.orientation.image_io import frame_from_gray
from ocr_utils.geometry_regression import WORK_DPI
from ocr_utils.geometry_regression.render import RENDER_DPI

Box = tuple[int, int, int, int]

# ``line_fit`` строит копии 300 и 150 dpi сам (``WORK_DPI_FINE`` в ``orientation``):
# боксы строк он отдаёт на копии 150 dpi, центр-линии — на копии 300 dpi.
LINE_FIT_BOX_DPI = RENDER_DPI // 2


@dataclass(frozen=True)
class TextLine:
    """Строка текста на рабочей копии с аппроксимацией её центр-линии.

    Бокс и высота — в пикселях рабочей копии; ``fit`` посчитан на копии ``RENDER_DPI``,
    поэтому его линейные величины делятся на ``fit_scale`` при переводе.
    """

    x0: int
    y0: int
    x1: int
    y1: int
    height: float
    fit: LineFit
    fit_scale: float  # пикселей аппроксимации на пиксель рабочей копии
    xs: np.ndarray  # центр-линия по своим сгусткам, координаты копии RENDER_DPI
    ys: np.ndarray
    column: int = 0  # номер колонки, в которой лежит строка (0 — первая); через межколонник — −1

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def length(self) -> float:
        return float(self.x1 - self.x0)

    @property
    def slope_deg(self) -> float:
        return self.fit.slope_deg

    @property
    def sag_rel(self) -> float:
        """Прогиб параболы в долях высоты строки."""
        return abs(self.fit.sagitta) / max(1.0, self.fit_scale * self.height)

    @property
    def wobble_rel(self) -> float:
        """Остаток центр-линии от параболы в долях высоты: волнистость внутри строки."""
        return self.fit.resid_quad / max(1.0, self.fit_scale * self.height)

    @property
    def box(self) -> Box:
        return (self.x0, self.y0, self.x1, self.y1)


def lineart_boxes(gray: np.ndarray, dpi: float = WORK_DPI) -> list[Box]:
    """Рамки крупного штриха (x0, y0, x1, y1) на рабочей копии — одни пиксели, без surya (запасной ход)."""
    findings = analyse_gray(gray, params_for_dpi(int(round(dpi))))
    return [tuple(int(v) for v in box) for box in findings.boxes]


def layout_boxes(document, page_index: int, dpi: float = WORK_DPI, surya=None) -> list[Box]:
    """Рамки таблиц и line art страницы PDF без коррекции в пикселях ``dpi`` — разбором ``page_layout``.

    Args:
        document: Открытый ``fitz.Document`` без коррекции геометрии (B).
        page_index: Номер страницы с нуля.
        dpi: Разрешение рабочей копии, в котором отдаются рамки.
        surya: ``SuryaSource`` (кэш только для чтения в воркере, кэш + модель в родителе) или
            ``None`` — без surya, по одним пикселям.

    Returns:
        Рамки ``(x0, y0, x1, y1)`` таблиц и line art; таблица для детектора порчи — та же
        «нетекстовая геометрия», что и рисунок: её линейки нельзя ни наклонять, ни гнуть.
    """
    from ocr_utils.page_layout.analysis import Find, LayoutOptions, PageLayout
    from ocr_utils.page_layout.image import PageImage, Variant

    image = PageImage.from_pdf_page(document, page_index, Variant.FR_NOGEO)
    options = LayoutOptions(use_surya=surya is not None and surya.enabled)
    layout = PageLayout(image, {Find.TABLES, Find.LINE_ART}, options).process(surya)
    k = dpi / image.dpi
    boxes: list[Box] = []
    for region in layout.tables + layout.line_arts:
        box = region.box.scaled(k)
        boxes.append((int(box.x0), int(box.y0), int(box.x1), int(box.y1)))
    return boxes


def text_lines(gray300: np.ndarray, dpi: float = WORK_DPI) -> tuple[list[TextLine], list[tuple[int, int]]]:
    """Строки с аппроксимациями и межколонники, приведённые к пикселям рабочей копии ``dpi``.

    Args:
        gray300: серый рендер страницы в ``RENDER_DPI``.
        dpi: разрешение рабочей копии, в котором отдаются боксы.

    Returns:
        Строки (с номером колонки) и межколонники ``(x0, x1)`` — по лентам высоты, чтобы
        заголовок на всю ширину их не ломал.
    """
    frame = frame_from_gray(gray300, RENDER_DPI, "page", Path("page"))
    samples, separators = line_samples(frame, banded=True)
    k = dpi / LINE_FIT_BOX_DPI
    separators = [(round(a * k), round(b * k)) for a, b in separators]
    columns = column_spans(separators, round(gray300.shape[1] * dpi / RENDER_DPI), dpi)
    lines: list[TextLine] = []
    for sample in samples:
        fit = _fit(sample)
        if fit is not None:
            box = (round(sample.x * k), round(sample.y * k), round(sample.x_end * k), round(sample.y_end * k))
            lines.append(
                TextLine(*box, sample.h_line * k, fit, RENDER_DPI / dpi, sample.xs, sample.ys, column_of(box, columns))
            )
    return lines, separators


# Уже этого (мм) колонка не бывает: пустые полосы внутри узкой врезки курсивом (1970/04 с.32 в A)
# сливаются с соседями, иначе номера колонок в A и B расходятся.
MIN_COLUMN_MM = 20.0


def column_spans(separators: list[tuple[int, int]], width: int, dpi: float = WORK_DPI) -> list[tuple[int, int]]:
    """Колонки ``(x0, x1)`` — промежутки между межколонниками (поля по краям — тоже межколонники).

    Промежуток уже ``MIN_COLUMN_MM`` колонкой не считается и присоединяется к соседу.
    """
    bounds = [0] + [x for pair in separators for x in pair] + [width]
    spans = [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds), 2) if bounds[i + 1] > bounds[i]]
    min_px = MIN_COLUMN_MM * dpi / 25.4
    merged: list[tuple[int, int]] = []
    for x0, x1 in spans:
        if merged and (x1 - x0 < min_px or merged[-1][1] - merged[-1][0] < min_px):
            merged[-1] = (merged[-1][0], x1)
        else:
            merged.append((x0, x1))
    return merged


def column_of(box: Box, columns: list[tuple[int, int]]) -> int:
    """Номер колонки, в которой строка лежит целиком (допуск по краям 5 % ширины колонки); через межколонник — −1."""
    x0, _, x1, _ = box
    for index, (c0, c1) in enumerate(columns):
        pad = 0.05 * (c1 - c0)
        if x0 >= c0 - pad and x1 <= c1 + pad:
            return index
    return -1


def text_boxes(lines: list[TextLine]) -> list[Box]:
    return [line.box for line in lines]


def _fit(sample: LineSample) -> LineFit | None:
    return fit_line(sample.xs - 2 * sample.x, sample.ys - 2 * sample.y, sample.weights)
