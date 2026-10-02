"""Картинки к отчёту reports/text_block_envelope.md: построение границы текстового блока по шагам на вырезке одного блока.

Скрипт ничего не считает сам: он вызывает те же функции, что и рабочий разбор пака
(:func:`ocr_utils.page_layout.pack_analysis.final.text_blocks` с осью ``BODY``, функции
:mod:`ocr_utils.page_layout.text_blocks.blocks`, :mod:`~.baseline_axis`, :mod:`~.sides`), и рисует их
промежуточные значения. Подсказки внешних детекторов (запреты, барьеры, линейки) берутся из JSON полосы
готового разбора пака (``--analysis-dir``), как в итоговой стадии; нет JSON — разбор без подсказок.

Запуск::

    uv run python scripts/text_block_envelope_report.py --sharpened-dir "$SHARPENED_DIR" \\
        --analysis-dir /mnt/hotstore/scan_processing/mts/pack1_page_analysis_v3 --out-dir reports/text_block_envelope \\
        --page curved=1971/10/IMG_0046_2R --page trapezoid=1969/07/IMG_0037_1L#2
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.image import PageImage, Variant
from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from ocr_utils.page_layout.pack_analysis.final import PATCH_COLOR, SIDE_ALIGN_METHOD, SIDE_COLOR, text_blocks
from ocr_utils.page_layout.pack_analysis.stages import DEFAULT_DPI, page_key
from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks import overlay as blocks_overlay
from ocr_utils.page_layout.text_blocks.baseline_axis import fit_baseline
from ocr_utils.page_layout.text_blocks.blocks import (
    EDGE_QUANTILE,
    TRIM_MM,
    CapKind,
    _cap_line,
    _quantile_trend,
    _row_band,
    _row_span,
)
from ocr_utils.page_layout.text_blocks.page import AxisKind, text_ink
from ocr_utils.page_layout.text_blocks.sides import RowStatus, SideKind, filled_side, side_alignment, sides_of
from ocr_utils.page_layout.text_blocks.sides_overlay import draw_filled_line
from ocr_utils.page_layout import mm_to_px

logger = logging.getLogger(__name__)

# Палитра (BGR) — по навыку draw-overlay: те же значения, что на оверлеях пакета.
AXIS = blocks_overlay.COLOUR_AXIS  # вторая ось (основная)
FIRST_AXIS = (0, 165, 255)  # первая ось, по центру масс краски — оранжевым полупрозрачно
ENVELOPE = blocks_overlay.COLOUR_ENVELOPE  # итоговая граница блока
ENVELOPE_INK = blocks_overlay.COLOUR_ENVELOPE_INK
COARSE = blocks_overlay.COLOUR_COARSE
POINT = blocks_overlay.COLOUR_POINT  # края рядов
TAIL = blocks_overlay.COLOUR_TAIL
OK = (0, 150, 0)
BAD = (0, 0, 220)
HINT = (200, 140, 60)  # вспомогательное: боксы глифов, грубый тренд
INDENT = (0, 165, 255)
INK_TINT = (120, 120, 120)  # краска текста — приглушённо: здесь важна разметка, а не буквы
SIDE_LABEL = {
    SideKind.LEFT: SIDE_COLOR[SideKind.LEFT],
    SideKind.RIGHT: SIDE_COLOR[SideKind.RIGHT],
    SideKind.TOP: (0, 150, 0),
    SideKind.BOTTOM: (160, 60, 160),
}
ALPHA = 0.55
# Ширина вырезки блока на картинке и поле вокруг блока (мм бумаги).
CROP_WIDTH = 1100
MARGIN_MM = 4.0
# Ширина обзорной картинки страницы.
PAGE_WIDTH = 1100
# Увеличение вырезок углов блока («верх | низ») относительно рабочей копии.
CORNER_ZOOM = 5
CORNER_ROWS = 3
DASH = 6


@dataclass
class Example:
    """Разобранный пример: полоса, её разбор и выбранный блок."""

    slug: str
    name: str
    gray300: np.ndarray
    analysis: object
    hints: object
    ink: np.ndarray
    block_index: int

    @property
    def block(self):
        return self.analysis.blocks[self.block_index]


class View:
    """Вырезка страницы вокруг прямоугольника (пиксели рабочей копии) с увеличением: перевод координат и холст."""

    def __init__(self, gray300: np.ndarray, box: tuple[float, float, float, float], zoom: float) -> None:
        """
        Args:
            gray300: Серый рендер страницы ``RENDER_DPI``.
            box: ``(x0, y0, x1, y1)`` вырезки в пикселях рабочей копии.
            zoom: Пикселей холста на пиксель рабочей копии.
        """
        k = RENDER_DPI / WORK_DPI
        height, width = gray300.shape
        self.x0, self.y0 = max(0.0, box[0]), max(0.0, box[1])
        x1, y1 = min(width / k, box[2]), min(height / k, box[3])
        self.zoom = zoom
        crop = gray300[int(self.y0 * k) : int(y1 * k), int(self.x0 * k) : int(x1 * k)]
        size = (int(round((x1 - self.x0) * zoom)), int(round((y1 - self.y0) * zoom)))
        self.gray = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC if zoom * 1.0 > k else cv2.INTER_AREA)

    def canvas(self, faded: bool = False) -> np.ndarray:
        """Холст BGR: серая вырезка; ``faded`` — краска приглушена до светло-серого (картинки «по краске»)."""
        gray = self.gray if not faded else (255 - (255 - self.gray.astype(np.float32)) * 0.35).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def pts(self, points) -> np.ndarray:
        """Точки рабочей копии ``(N, 2)`` → целые пиксели холста."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return np.round((points - [self.x0, self.y0]) * self.zoom).astype(np.int32)

    def line(self, canvas, points, colour, thickness=2, closed=False) -> None:
        """Ломаная (``points`` в пикселях рабочей копии)."""
        if points is None or len(points) < 2:
            return
        cv2.polylines(canvas, [self.pts(points)], closed, colour, thickness, cv2.LINE_AA)

    def dashed(self, canvas, points, colour, thickness=2) -> None:
        """Ломаная пунктиром по длине."""
        pts = self.pts(points).astype(np.float64)
        if len(pts) < 2:
            return
        along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        for start in np.arange(0.0, along[-1], 2 * DASH):
            stop = min(start + DASH, along[-1])
            xs = np.interp([start, stop], along, pts[:, 0])
            ys = np.interp([start, stop], along, pts[:, 1])
            cv2.line(canvas, (int(xs[0]), int(ys[0])), (int(xs[1]), int(ys[1])), colour, thickness, cv2.LINE_AA)

    def dot(self, canvas, point, colour, radius=4, filled=True) -> None:
        """Кружок в точке рабочей копии."""
        x, y = self.pts(point)[0]
        cv2.circle(canvas, (int(x), int(y)), radius, colour, -1 if filled else 1, cv2.LINE_AA)


def load(slug: str, spec: str, sharpened: Path, analysis_dir: Path | None) -> Example:
    """Разобрать полосу так же, как итоговая стадия разбора пака v3 (вторая ось, подсказки из JSON полосы).

    Args:
        slug: Короткое имя примера (префикс файлов).
        spec: ``год/выпуск/полоса`` и, через ``#``, номер блока; без номера — самый многострочный блок.
        sharpened: Корень заострённых копий.
        analysis_dir: Корень готового разбора пака (``pages/<ключ>.json``) или ``None``.

    Returns:
        Пример.
    """
    name, _, index = spec.partition("#")
    image = PageImage.from_file(sharpened / f"{name}.jpg", Variant.SHARPENED, name, default_dpi=DEFAULT_DPI)
    record, objects = {"page": name, "loose_rules": []}, []
    source = analysis_dir / "pages" / f"{page_key(name)}.json" if analysis_dir else None
    if source is not None and source.is_file():
        saved = json.loads(source.read_text())
        record["loose_rules"], objects = saved["loose_rules"], saved["objects"]
    analysis, hints = text_blocks(image, record, objects, AxisKind.BODY)
    gray300 = image.gray_at(RENDER_DPI)
    blocks = analysis.blocks
    chosen = int(index) if index else max(range(len(blocks)), key=lambda i: len(blocks[i].rows))
    return Example(slug, name, gray300, analysis, hints, text_ink(gray300, WORK_DPI), chosen)


def block_view(example: Example) -> View:
    """Вырезка вокруг блока с полем ``MARGIN_MM``, шириной ``CROP_WIDTH`` на картинке."""
    polygon = example.block.envelope.polygon
    margin = mm_to_px(MARGIN_MM, WORK_DPI)
    box = (*(polygon.min(axis=0) - margin), *(polygon.max(axis=0) + margin))
    return View(example.gray300, box, CROP_WIDTH / (box[2] - box[0]))


def save(out: Path, example: Example, number: int, step: str, canvas, header: str, legend) -> str:
    """Записать картинку шага со шапкой и легендой в полях; вернуть имя файла."""
    title = f"{example.name.replace('/', ' ')}, блок {example.block.column}.{example.block.index} — {header}"
    entries = [LegendEntry(*item) if isinstance(item, tuple) else item for item in legend]
    picture = framed(canvas, [title], entries)
    file_name = f"{example.slug}_{number:02d}_{step}.jpg"
    cv2.imwrite(str(out / file_name), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return file_name


def blend(canvas: np.ndarray, layer: np.ndarray, alpha: float = ALPHA) -> None:
    """Подмешать слой к холсту с прозрачностью (на месте)."""
    cv2.addWeighted(layer, alpha, canvas, 1.0 - alpha, 0, canvas)


# --- Шаги -------------------------------------------------------------------------------------------


def step_page(example: Example, out: Path) -> str:
    """Шаг 0: вся полоса — разбор целиком, выбранный блок в красной рамке."""
    analysis = example.analysis
    scale = PAGE_WIDTH / analysis.width
    small = cv2.resize(example.gray300, (PAGE_WIDTH, int(round(analysis.height * scale))), interpolation=cv2.INTER_AREA)
    canvas = blocks_overlay.draw(analysis, small, scale=scale, hints=example.hints)
    polygon = example.block.envelope.polygon * scale
    x0, y0 = polygon.min(axis=0).astype(int) - 6
    x1, y1 = polygon.max(axis=0).astype(int) + 6
    cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1), int(y1)), BAD, 3)
    legend = [
        *blocks_overlay.legend_entries(hints=example.hints, body_axis=True),
        LegendEntry("блок, который разбирается по шагам", BAD),
    ]
    return save(out, example, 0, "page", canvas, "вся полоса", legend)


def step_ink(example: Example, view: View, out: Path) -> str:
    """Шаг 1: краска текста (маска глифов рендера) и боксы глифов строк."""
    canvas = view.canvas(faded=True)
    k = RENDER_DPI / WORK_DPI
    height, width = view.gray.shape
    ink = example.ink[int(view.y0 * k) :, int(view.x0 * k) :]
    ink = cv2.resize(ink.astype(np.uint8), None, fx=view.zoom / k, fy=view.zoom / k, interpolation=cv2.INTER_NEAREST)
    ink = ink[:height, :width]
    layer = canvas.copy()
    layer[: ink.shape[0], : ink.shape[1]][ink > 0] = (40, 40, 40)
    blend(canvas, layer, 0.8)
    for row in example.block.rows:
        for axis in row.axes:
            for x0, y0, x1, y1 in axis.glyphs if axis.glyphs is not None else ():
                cv2.rectangle(canvas, tuple(view.pts([x0, y0])[0]), tuple(view.pts([x1, y1])[0]), HINT, 1)
    legend = [("краска текста (маска глифов)", (40, 40, 40), 0.8, SampleStyle.BOX), ("бокс глифа строки", HINT)]
    return save(out, example, 1, "ink", canvas, "краска текста и глифы", legend)


def step_axes(example: Example, view: View, out: Path) -> str:
    """Шаг 2: первая ось (центр масс) и вторая (базовая линия глифов поднята на полстрочной); голоса за базу."""
    canvas = view.canvas()
    layer = canvas.copy()
    for row in example.block.rows:
        for axis in row.axes:
            if axis.centre_points is not None:
                view.line(layer, axis.centre_points, FIRST_AXIS, 3)
    blend(canvas, layer)
    for row in example.block.rows:
        for axis in row.axes:
            fit = fit_baseline(axis)
            if fit is not None:
                view.line(canvas, np.column_stack([fit.grid, fit.base]), (0, 90, 0), 1)
                for x, y, kept in zip(fit.xs, fit.bottoms, fit.kept):
                    view.dot(canvas, (x, y), OK if kept else BAD, 2)
            view.line(canvas, axis.points, AXIS, 2)
    legend = [
        ("первая ось: центр масс краски (прежняя)", FIRST_AXIS, ALPHA),
        ("базовая линия: сплайн по низам глифов", (0, 90, 0)),
        ("голос за базу принят", OK),
        ("голос отсеян (выносной, индекс, тире)", BAD),
        ("вторая ось = база + полвысоты строчной", AXIS),
    ]
    return save(out, example, 2, "axes", canvas, "первая и вторая оси строки", legend)


def step_rows(example: Example, view: View, out: Path) -> str:
    """Шаг 3: ряды блока — края по краске, профили краски сверху и снизу, полоса вокруг оси."""
    canvas = view.canvas()
    layer = canvas.copy()
    margin = 0.0
    for row in example.block.rows:
        band = _row_band(row, margin, CapKind.BODY)
        if band is not None:
            cv2.fillPoly(layer, [view.pts(band)], (230, 200, 150))
    blend(canvas, layer, 0.45)
    for row in example.block.rows:
        view.line(canvas, row.top_edge, HINT, 1)
        view.line(canvas, row.bottom_edge, HINT, 1)
        for axis in row.axes:
            view.line(canvas, axis.points, AXIS, 2)
        for x in (row.x0, row.x1):
            view.dot(canvas, (x, row.y), POINT, 5, filled=False)
        left, right = _row_span(row, CapKind.BODY)
        for x in (left, right):
            view.dot(canvas, (x, row.y), POINT, 2)
    legend = [
        ("полоса ряда вокруг оси (квантиль 0.70 профиля)", (230, 200, 150), 0.45, SampleStyle.BOX),
        ("профиль краски ряда сверху и снизу", HINT),
        ("ось строки (вторая)", AXIS),
        ("край ряда по краске (кружок) / по концу оси (точка)", POINT),
    ]
    return save(out, example, 3, "rows", canvas, "ряды: края и профили краски", legend)


def _ends(block) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ординаты рядов и их края слева и справа — как в ``envelope_of`` для кромки-полосы."""
    ys = np.array([row.y for row in block.rows])
    spans = np.array([_row_span(row, CapKind.BODY) for row in block.rows])
    return ys, spans[:, 0], spans[:, 1]


def step_trend(example: Example, view: View, out: Path) -> str:
    """Шаг 4: тренд сторон — грубый квантиль, отбор рядов тела, локальная прямая."""
    block = example.block
    envelope = block.envelope
    canvas = view.canvas()
    ys, lefts, rights = _ends(block)
    grid = envelope.core_left[:, 1]
    window = envelope.smooth_pitches * block.pitch_px
    trim = mm_to_px(TRIM_MM, block.dpi)
    for xs, core, quantile, inward in (
        (lefts, envelope.core_left, EDGE_QUANTILE, 1.0),
        (rights, envelope.core_right, 1.0 - EDGE_QUANTILE, -1.0),
    ):
        rough = _quantile_trend(ys, xs, grid, window * 2.0, quantile)
        view.dashed(canvas, np.column_stack([rough, grid]), HINT, 2)
        view.line(canvas, core, COARSE, 2)
        fitted = np.interp(ys, core[:, 1], core[:, 0])
        for x, y, f in zip(xs, ys, fitted):
            view.dot(canvas, (x, y), BAD if inward * (x - f) > trim else OK, 4)
    legend = [
        ("грубый квантильный тренд (0.25 слева, 0.75 справа)", HINT, 1.0, SampleStyle.DASHED),
        ("край ряда в теле блока", OK),
        (f"край ряда ушёл внутрь > {TRIM_MM:g} мм (отступ, конец абзаца)", BAD),
        ("тренд стороны: локальная прямая по телу", COARSE),
    ]
    return save(out, example, 4, "trend", canvas, "тренд левой и правой сторон", legend)


def step_outward(example: Example, view: View, out: Path) -> str:
    """Шаг 5: сторона отодвинута наружу до самых дальних краёв рядов и обрезана верхом и низом."""
    block = example.block
    envelope = block.envelope
    canvas = view.canvas()
    ys, lefts, rights = _ends(block)
    view.line(canvas, envelope.core_left, COARSE, 1)
    view.line(canvas, envelope.core_right, COARSE, 1)
    view.line(canvas, envelope.left, ENVELOPE, 3)
    view.line(canvas, envelope.right, ENVELOPE, 3)
    view.line(canvas, envelope.top, (0, 150, 0), 2)
    view.line(canvas, envelope.bottom, (160, 60, 160), 2)
    for x, y in zip(np.concatenate([lefts, rights]), np.concatenate([ys, ys])):
        view.dot(canvas, (x, y), POINT, 3)
    legend = [
        ("тренд стороны (шаг 4)", COARSE),
        ("сторона, отодвинутая наружу (0.3 мм за краем)", ENVELOPE),
        ("верх: полоса над осью первой строки", (0, 150, 0)),
        ("низ: полоса под осью последней строки", (160, 60, 160)),
        ("края рядов", POINT),
    ]
    return save(out, example, 5, "outward", canvas, "стороны наружу, верх и низ", legend)


def _corner_views(example: Example) -> list[View]:
    """Две крупные вырезки блока: первые и последние ``CORNER_ROWS`` рядов на всю ширину."""
    block = example.block
    margin = mm_to_px(MARGIN_MM, WORK_DPI)
    polygon = block.envelope.polygon
    x0, x1 = polygon[:, 0].min() - margin, polygon[:, 0].max() + margin
    rows = block.rows
    top = rows[min(CORNER_ROWS, len(rows)) - 1]
    bottom = rows[max(0, len(rows) - CORNER_ROWS)]
    boxes = [
        (x0, polygon[:, 1].min() - margin, x1, top.y + top.height),
        (x0, bottom.y - bottom.height, x1, polygon[:, 1].max() + margin),
    ]
    zoom = min(CORNER_ZOOM, CROP_WIDTH / (x1 - x0))
    return [View(example.gray300, box, zoom) for box in boxes]


def step_caps(example: Example, out: Path) -> str:
    """Шаг 6: верх и низ крупно — линия строчной, отступ полосы, хвост короткой последней строки."""
    block = example.block
    envelope = block.envelope
    parts = []
    for view, row in zip(_corner_views(example), (block.rows[0], block.rows[-1])):
        canvas = view.canvas()
        for own in block.rows:
            for axis in own.axes:
                view.line(canvas, axis.points, AXIS, 2)
            view.line(canvas, own.top_edge, HINT, 1)
            view.line(canvas, own.bottom_edge, HINT, 1)
        reference = _cap_line(row)
        view.dashed(canvas, reference, (60, 60, 60), 1)
        if row.tail is not None:
            view.line(canvas, row.tail, TAIL, 2)
            view.line(canvas, row.cut, TAIL, 1)
        view.line(canvas, envelope.top, (0, 150, 0), 2)
        view.line(canvas, envelope.bottom, (160, 60, 160), 2)
        parts.append(canvas)
    width = max(part.shape[1] for part in parts)
    parts = [
        cv2.copyMakeBorder(p, 0, 12, 0, width - p.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in parts
    ]
    legend = [
        ("ось строки (вторая)", AXIS),
        ("линия середины строчной, от которой меряется отступ", (60, 60, 60), 1.0, SampleStyle.DASHED),
        ("профиль краски ряда", HINT),
        ("верх блока", (0, 150, 0)),
        ("низ блока", (160, 60, 160)),
        ("хвост последней строки и линия отсечки", TAIL),
    ]
    return save(
        out, example, 6, "caps", np.vstack(parts), "верх и низ блока крупно (сверху — верх, снизу — низ)", legend
    )


def step_polygon(example: Example, view: View, out: Path) -> str:
    """Шаг 7: контур из четырёх кривых, он же после поглощения рядов, и раздутый на полсимвола."""
    envelope = example.block.envelope
    canvas = view.canvas()
    assembled = np.vstack([envelope.left, envelope.bottom, envelope.right[::-1], envelope.top[::-1]])
    view.dashed(canvas, np.vstack([assembled, assembled[:1]]), HINT, 2)
    layer = canvas.copy()
    view.line(layer, envelope.polygon, ENVELOPE, 3, closed=True)
    blend(canvas, layer)
    if envelope.polygon_dilated is not None:
        view.line(canvas, envelope.polygon_dilated, (0, 200, 255), 1, closed=True)
    for row in example.block.rows:
        for axis in row.axes:
            view.line(canvas, axis.points, AXIS, 1)
    legend = [
        ("контур обходом: левая, низ, правая, верх", HINT, 1.0, SampleStyle.DASHED),
        ("итоговая граница (после поглощения рядов)", ENVELOPE, ALPHA),
        ("граница, раздутая на полсимвола", (0, 200, 255)),
        ("ось строки", AXIS),
    ]
    return save(out, example, 7, "polygon", canvas, "сборка контура", legend)


def step_variants(example: Example, view: View, out: Path) -> str:
    """Шаг 8: три огибающие блока — полоса вокруг оси, по краске, крупная."""
    block = example.block
    canvas = view.canvas()
    layer = canvas.copy()
    view.line(layer, block.envelope.polygon, ENVELOPE, 3, closed=True)
    if block.envelope_ink is not None:
        view.line(layer, block.envelope_ink.polygon, ENVELOPE_INK, 3, closed=True)
    blend(canvas, layer)
    view.line(canvas, block.envelope_coarse.left, COARSE, 2)
    view.line(canvas, block.envelope_coarse.right, COARSE, 2)
    legend = [
        ("граница: полоса вокруг оси (главная)", ENVELOPE, ALPHA),
        ("граница по краске, справочно", ENVELOPE_INK, ALPHA),
        ("крупная огибающая (окно ×3)", COARSE),
    ]
    return save(out, example, 8, "variants", canvas, "главная, по краске и крупная огибающие", legend)


def step_sides(example: Example, view: View, out: Path) -> str:
    """Шаг 9: стороны контура по построению; углы и неуверенные концы — пунктиром."""
    sides = sides_of(example.block)
    canvas = view.canvas()
    points = sides.polygon
    n = len(points)
    for index in range(n):
        segment = points[[index, (index + 1) % n]]
        colour = SIDE_LABEL[sides.labels[index]]
        if sides.excluded[index]:
            view.dashed(canvas, segment, colour, 3)
        else:
            view.line(canvas, segment, colour, 3)
    legend = [
        ("левая сторона", SIDE_LABEL[SideKind.LEFT]),
        ("правая сторона", SIDE_LABEL[SideKind.RIGHT]),
        ("верх", SIDE_LABEL[SideKind.TOP]),
        ("низ", SIDE_LABEL[SideKind.BOTTOM]),
        ("угол или неуверенный конец (в меры не идёт)", (90, 90, 90), 1.0, SampleStyle.DASHED),
    ]
    return save(out, example, 9, "sides", canvas, "стороны границы (метод construct)", legend)


def step_alignment(example: Example, view: View, out: Path) -> str:
    """Шаг 10: выравнивание концов строк по вертикальным сторонам (метод robust)."""
    block = example.block
    canvas = view.canvas()
    colours = {RowStatus.ON: OK, RowStatus.OFF: BAD, RowStatus.INDENT: INDENT}
    for side in (SideKind.LEFT, SideKind.RIGHT):
        alignment = side_alignment(block, side, SIDE_ALIGN_METHOD)
        if alignment is None:
            continue
        if alignment.curve is not None:
            view.line(canvas, alignment.curve, COARSE, 2)
        for end in alignment.ends:
            view.dot(canvas, end.point, colours[end.status], 5)
            if end.aligned:
                view.dot(canvas, end.point, (0, 0, 0), 8, filled=False)
    legend = [
        ("кривая стороны: Тейл–Сен / RANSAC-парабола по концам строк", COARSE),
        ("конец строки на кривой (±0.8 мм)", OK),
        ("конец строки мимо кривой", BAD),
        ("отступ внутрь ≥ 2.5 мм (абзац, конец абзаца)", INDENT),
        ("ряд в выровненной серии (обведён)", (0, 0, 0)),
    ]
    return save(out, example, 10, "alignment", canvas, "концы строк у сторон", legend)


def step_filled(example: Example, view: View, out: Path) -> str:
    """Шаг 11: дополнительные линии сторон — без невыровненных концов, с PCHIP-заплатками."""
    block = example.block
    sides = sides_of(block)
    canvas = view.canvas()
    view.line(canvas, block.envelope.polygon, ENVELOPE, 1, closed=True)
    layer = canvas.copy()
    for side in (SideKind.LEFT, SideKind.RIGHT):
        alignment = side_alignment(block, side, SIDE_ALIGN_METHOD)
        line = None if alignment is None else filled_side(sides, alignment, block)
        if line is None:
            continue
        # Линия рисуется в координатах вырезки: сдвиг на начало вырезки и увеличение.
        shifted = replace(line, points=line.points - [view.x0, view.y0])
        draw_filled_line(layer, shifted, view.zoom, 6, SIDE_COLOR[side], PATCH_COLOR)
    blend(canvas, layer)
    legend = [
        ("граница блока", ENVELOPE),
        ("левая сторона: доп. линия", SIDE_COLOR[SideKind.LEFT], ALPHA),
        ("правая сторона: доп. линия", SIDE_COLOR[SideKind.RIGHT], ALPHA),
        ("заплатка PCHIP на месте выброса", PATCH_COLOR, ALPHA, SampleStyle.DASHED),
    ]
    return save(out, example, 11, "filled", canvas, "дополнительные линии сторон", legend)


@click.command()
@click.option("--sharpened-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--analysis-dir", default=None, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--page",
    "pages",
    multiple=True,
    required=True,
    help="имя=год/выпуск/полоса[#блок], например curved=1971/10/IMG_0046_2R",
)
def main(sharpened_dir: Path, analysis_dir: Path | None, out_dir: Path, pages: tuple[str, ...]) -> None:
    """Нарисовать шаги построения границы текстового блока для каждого примера."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in pages:
        slug, _, spec = item.partition("=")
        example = load(slug, spec, sharpened_dir, analysis_dir)
        view = block_view(example)
        block = example.block
        logger.info(
            "%s: блок %d.%d, рядов %d, шаг %.1f px",
            example.name,
            block.column,
            block.index,
            len(block.rows),
            block.pitch_px,
        )
        names = [
            step_page(example, out_dir),
            step_ink(example, view, out_dir),
            step_axes(example, view, out_dir),
            step_rows(example, view, out_dir),
            step_trend(example, view, out_dir),
            step_outward(example, view, out_dir),
            step_caps(example, out_dir),
            step_polygon(example, view, out_dir),
            step_variants(example, view, out_dir),
            step_sides(example, view, out_dir),
            step_alignment(example, view, out_dir),
            step_filled(example, view, out_dir),
        ]
        logger.info("%s: %s", slug, ", ".join(names))


if __name__ == "__main__":
    main()
