"""Итоговый разбор полосы: классы объектов (растр, таблицы, line art по решению DeepSeek, формулы, повёрнутый текст), текстовые блоки с запретами, оверлей и папка по классам."""

from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.line_art.classes import ObjectClass
from ocr_utils.page_layout.line_art.deepseek.decide import DECISION_VERSION, Outcome, is_thin_stroke
from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, header_strip, legend_strip
from ocr_utils.page_layout.pack_analysis.stages import PageTask, decide_candidates, load_image, page_key, write_json
from ocr_utils.page_layout.regions import RegionKind
from ocr_utils.page_layout import version_tag
from ocr_utils.page_layout.text_blocks import RENDER_DPI, TEXT_BLOCKS_VERSION, WORK_DPI
from ocr_utils.page_layout.text_blocks import overlay as blocks_overlay
from ocr_utils.page_layout.text_blocks.blocks import DEFAULT_BLOCKS_MODE, BlocksMode
from ocr_utils.page_layout.text_blocks.columns import GutterMode
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.from_layout import build
from ocr_utils.page_layout.text_blocks.glyph_maps import load_maps
from ocr_utils.page_layout.text_blocks.page import AxisKind, analyse_gray
from ocr_utils.page_layout.text_blocks.report import filled_json
from ocr_utils.page_layout.text_blocks.segment import SPLIT_ROWS_DEFAULT
from ocr_utils.page_layout.text_blocks.sides import (
    AlignMethod,
    FilledSide,
    SideKind,
    filled_side,
    side_alignment,
    sides_of,
)
from ocr_utils.page_layout.text_blocks.sides_overlay import FILLED_ALPHA, draw_filled_line
from ocr_utils.page_layout.text_blocks.store import STORE_FORMAT_VERSION, page_arrays, write_arrays


class OverlayMode(str, Enum):
    """Писать ли оверлей полосы: на прогонах ради данных (PDF FineReader под детектор геометрии) — нет."""

    ALL = "all"
    NONE = "none"


class PageClass(str, Enum):
    """Класс объекта полосы — он же имя папки оверлея «не только текст»."""

    COLOR = "цветной_растр"
    GRAYSCALE = "серый_растр"
    COLOR_TEXT = "цветной_текст"
    STAMP = "печать"
    TABLE = "таблица"
    DRAWING = "рисунок"
    FORMULA = "формула"
    ROTATED_TEXT = "повёрнутый_текст"
    UNCLEAR = "неясно"


# Растровые виды ``page_layout`` → класс полосы.
RASTER_CLASS = {
    RegionKind.COLOR.value: PageClass.COLOR,
    RegionKind.GRAYSCALE.value: PageClass.GRAYSCALE,
    RegionKind.COLOR_TEXT.value: PageClass.COLOR_TEXT,
    RegionKind.STAMP_SUSPECT.value: PageClass.STAMP,
}
# Класс объекта DeepSeek → класс полосы.
OBJECT_CLASS = {
    ObjectClass.DRAWING.value: PageClass.DRAWING,
    ObjectClass.TABLE.value: PageClass.TABLE,
    ObjectClass.FORMULA.value: PageClass.FORMULA,
}

# Папки оверлеев.
ONLY_TEXT = "только_текст"
NOT_ONLY_TEXT = "не_только_текст"
SEVERAL = "несколько_классов"
DISPUTED_ORIENTATION = "ориентация_спорная"

# Цвета (BGR) классов на оверлее: растр и таблицы — как метки CVAT (page_layout/cli.py), line art —
# синий итогового объекта стенда, формула — оранжевый, неясно — красный «отвергнутого».
CLASS_COLOR = {
    PageClass.COLOR: (0, 230, 118),
    PageClass.GRAYSCALE: (255, 176, 0),
    PageClass.COLOR_TEXT: (98, 17, 197),
    PageClass.STAMP: (0, 109, 255),
    PageClass.TABLE: (254, 79, 48),
    PageClass.DRAWING: (180, 0, 180),
    PageClass.FORMULA: (0, 165, 255),
    PageClass.ROTATED_TEXT: (92, 105, 0),
    PageClass.UNCLEAR: (0, 0, 220),
}
TITLE_COLOR = (150, 150, 150)  # надпись, снятая с line art (справочно, пунктир не нужен — тонкая рамка)
MARK_COLOR = (180, 105, 255)  # пометка на полях, снятая с line art (розовый — не занят на этом оверлее), тонкая рамка
STROKE_COLOR = (0, 165, 255)  # одиночная черта, снятая с line art (оранжевый — не занят на этом оверлее), тонкая рамка
RULE_COLOR = (60, 90, 210)  # линейки-сироты (text_blocks.overlay.COLOUR_BARRIER)
FILL_ALPHA = 0.18
OVERLAY_WIDTH = 1600

# Дополнительные линии левой и правой стороны блока (:func:`sides.filled_side`): невыровненные концы
# выброшены, невыровненные куски в середине заменены гладкой интерполяцией PCHIP (заплатка — пунктиром).
# Главная граница блока синяя, поэтому стороны — своими цветами, не занятыми на этом оверлее: левая —
# золотистая, правая — голубая. Толсто и полупрозрачно, как на стенде сторон (``sides_overlay``).
SIDE_COLOR = {SideKind.LEFT: (0, 190, 230), SideKind.RIGHT: (230, 210, 0)}
# Заплатка (место выброса, замененное интерполяцией) — пунктиром своего красно-оранжевого цвета у обеих
# сторон: пунктир цвета стороны на синей границе блока не различался (промежутки сливались с ней).
PATCH_COLOR = (0, 60, 255)
SIDE_THICKNESS = 4
# Метод выравнивания, по которому размечены выровненные участки стороны: robust — устойчивая подгонка к
# концам строк в системе координат блока (выбран на стенде сторон 2026-09-25).
SIDE_ALIGN_METHOD = AlignMethod.ROBUST


def assemble(record: dict, decisions: list) -> dict:
    """Объекты полосы по классам из разбора и решений по кандидатам line art.

    Args:
        record: JSON полосы стадии кандидатов.
        decisions: Пары (кандидат, :class:`Decision`).

    Returns:
        ``{"objects": [{"class", "box", "source"}], "titles": [рамки], "marks": [рамки]}`` в пикселях полосы;
        пометки на полях (``Outcome.MARK``) — отдельным списком: не объект полосы, не надпись.
    """
    objects = []
    for region in record["raster"]:
        objects.append({"class": RASTER_CLASS[region["kind"]].value, "box": region["box"], "source": "растр"})
    for region in record["tables"]:
        objects.append({"class": PageClass.TABLE.value, "box": region["box"], "source": "таблицы"})
    for region in record["formulas"]:
        objects.append({"class": PageClass.FORMULA.value, "box": region["box"], "source": "surya Equation"})
    for region in record["rotated_text"]:
        objects.append({"class": PageClass.ROTATED_TEXT.value, "box": region["box"], "source": "повёрнутый текст"})
    titles, marks = [], []
    for candidate, decision in decisions:
        if decision.outcome is Outcome.OBJECTS:
            for obj in decision.objects:
                objects.append(
                    {
                        "class": OBJECT_CLASS[obj["class"]].value,
                        "box": obj["box"],
                        "source": f"line art: {obj['box_source']}",
                        "candidate": candidate["id"],
                    }
                )
        elif decision.outcome is Outcome.UNCLEAR:
            objects.append(
                {
                    "class": PageClass.UNCLEAR.value,
                    "box": candidate["crop"]["box"],
                    "source": "line art",
                    "candidate": candidate["id"],
                }
            )
        elif decision.outcome is Outcome.MARK:
            marks.append(candidate["crop"]["box"])
        else:
            titles.append(candidate["crop"]["box"])
    return {"objects": objects, "titles": titles, "marks": marks}


def split_strokes(objects: list[dict], dpi: float) -> tuple[list[dict], list]:
    """Снять с объектов полосы одиночные черты: рамки рисунков и «неясно» тоньше ``decide.STROKE_MAX_MM``.

    Концевая черта статьи, линейка колонки или бланка, край обложки, кусок фото или заголовка в рамке 3 мм — не
    рисунок (просмотр пользователя 2026-09-30, 57 рамок пака-1): не запрет для текстовых блоков и не line art для
    детектора порчи геометрии. Работает и на объектах прошлого разбора (``reblock_page``).

    Args:
        objects: Объекты полосы (:func:`assemble`).
        dpi: Разрешение пикселей полосы.

    Returns:
        ``(объекты без черт, рамки черт)``.
    """
    kept, strokes = [], []
    for obj in objects:
        if obj["class"] in (PageClass.DRAWING.value, PageClass.UNCLEAR.value) and is_thin_stroke(obj["box"], dpi):
            strokes.append(list(obj["box"]))
        else:
            kept.append(obj)
    return kept, strokes


def text_blocks(
    image,
    record: dict,
    objects: list[dict],
    axis: AxisKind = AxisKind.CENTRE,
    gutter_mode: GutterMode = GutterMode.SHORT,
    join_leaders: bool = True,
    split_rows: bool = SPLIT_ROWS_DEFAULT,
    blocks_mode: BlocksMode = DEFAULT_BLOCKS_MODE,
    glyph_maps: dict | None = None,
):
    """Текстовые блоки полосы с запретами: растр, печати, таблицы, line art, формулы; барьеры — рамки и линейки.

    «Неясно» и надписи в запрет не идут (решение пользователя: «неясно» — отдельный класс для
    просмотра, надписи разбирает детектор строк). Повёрнутый текст — боковые зоны.

    Args:
        image: Страница.
        record: JSON полосы.
        objects: Объекты :func:`assemble`.
        axis: По какой оси строки собирать ряды и блоки: ``CENTRE`` — первая, по центру масс краски;
            ``BODY`` — вторая, по базовой линии глифов (:mod:`text_blocks.baseline_axis`).
        gutter_mode: Как межколонники превращаются в запреты сцепки (``columns.GutterMode``).
        join_leaders: Сращивать ли строки, сошедшиеся на общей точке отточия (``text_blocks.leader_join``).
        split_rows: Резать ли сгустки RLSA, собравшие буквы двух рядов (``segment._split_two_rows``).
        blocks_mode: Способ группировки строк в блоки и их границы (``text_blocks.blocks.BlocksMode``).
        glyph_maps: Карты CRAFT и pero полосы (``text_blocks.glyph_maps.load_maps``) — второй проход защиты
            сторон; ``None`` — только пометка выступов недостоверными.

    Returns:
        Пара: ``PageAnalysis`` детектора текстовых блоков (пиксели рабочей копии 150 dpi) и подсказки.
    """
    gray300 = image.gray_at(RENDER_DPI)
    scale = WORK_DPI / image.dpi
    width, height = int(round(gray300.shape[1] * WORK_DPI / RENDER_DPI)), int(
        round(gray300.shape[0] * WORK_DPI / RENDER_DPI)
    )

    def work(box) -> tuple[int, int, int, int]:
        return tuple(int(round(v * scale)) for v in box)  # type: ignore[return-value]

    forbidden_classes = {c.value for c in PageClass} - {PageClass.UNCLEAR.value, PageClass.ROTATED_TEXT.value}
    forbidden = [work(o["box"]) for o in objects if o["class"] in forbidden_classes]
    barriers = [work(o["box"]) for o in objects if o["class"] in (PageClass.TABLE.value, PageClass.DRAWING.value)]
    sideways = [work(o["box"]) for o in objects if o["class"] == PageClass.ROTATED_TEXT.value]
    rules = [tuple((x * scale, y * scale) for x, y in rule["points"]) for rule in record["loose_rules"]]
    hints = build(forbidden, barriers, sideways, width, height, WORK_DPI, rules)
    return (
        analyse_gray(
            gray300,
            InkEngine(hints=hints, gutter_mode=gutter_mode, join_leaders=join_leaders, split_rows=split_rows),
            hints=hints,
            name=page_key(record["page"]),
            variant=image.variant.value,
            axis=axis,
            blocks_mode=blocks_mode,
            glyph_maps=glyph_maps,
        ),
        hints,
    )


def side_lines(analysis) -> list[dict[SideKind, FilledSide | None]]:
    """Дополнительные линии левой и правой стороны каждого блока.

    Стороны размечаются методом по умолчанию (``sides_of``), выровненные участки — методом
    ``SIDE_ALIGN_METHOD``; по ним :func:`sides.filled_side` строит линию без невыровненных концов и с
    PCHIP-заплатками на невыровненных кусках в середине. Недостоверные участки огибающей
    (``unreliable_left/right``: ступеньки выноса, выступы сора ``edge_guard``) тоже идут заплаткой — линия
    стороны держится нормальных участков.

    Args:
        analysis: Разбор текстовых блоков.

    Returns:
        По блоку (в порядке ``analysis.blocks``) словарь «сторона → линия»; ``None`` — у стороны линия не
        построилась (мало звеньев или выровненных точек).
    """
    out = []
    for block in analysis.blocks:
        sides = sides_of(block)
        lines: dict[SideKind, FilledSide | None] = {}
        for side in (SideKind.LEFT, SideKind.RIGHT):
            alignment = side_alignment(block, side, SIDE_ALIGN_METHOD)
            unreliable = tuple(getattr(block.envelope, f"unreliable_{side.value}", ()))
            lines[side] = None if alignment is None else filled_side(sides, alignment, block, unreliable)
        out.append(lines)
    return out


def table_traces(image, objects: list[dict]) -> list[tuple[int, object]]:
    """Трассы линеек каждой таблицы полосы (``tables.traces.trace_rules``) в пикселях рабочей копии блоков.

    Трассы строятся по вырезке таблицы в ``RENDER_DPI`` (как у сетки по кривым) и переносятся в
    ``WORK_DPI`` кадра полосы: там же лежат оси строк и стороны блоков.

    Args:
        image: Страница (``PageImage``).
        objects: Объекты полосы (:func:`assemble`); берутся объекты класса «таблица», рамки — в родных пикселях.

    Returns:
        Пары ``(номер объекта в objects, RuleTrace)``.
    """
    from ocr_utils.page_layout.tables.traces import trace_rules

    tables = [(index, obj) for index, obj in enumerate(objects) if obj["class"] == PageClass.TABLE.value]
    if not tables:
        return []
    gray = image.gray_at(RENDER_DPI)
    to_render = RENDER_DPI / image.dpi
    out = []
    for index, obj in tables:
        # Рамка таблицы в пикселях рендера, обрезанная кадром.
        x0, y0, x1, y1 = (int(round(v * to_render)) for v in obj["box"])
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        traced = trace_rules(gray[y0:y1, x0:x1], RENDER_DPI)
        # Вырезка рендера → кадр рабочей копии: масштаб WORK/RENDER, сдвиг — угол вырезки в рабочих пикселях.
        scale = WORK_DPI / RENDER_DPI
        out += [(index, trace.mapped(scale, x0 * scale, y0 * scale)) for trace in traced.all]
    return out


def blocks_json(analysis, lines: list[dict]) -> list[dict]:
    """Блоки полосы для итогового JSON: контур, ряды, выключка, недостоверные участки и линии сторон.

    Args:
        analysis: Разбор текстовых блоков.
        lines: Дополнительные линии сторон (:func:`side_lines`).

    Returns:
        По блоку словарь (координаты — пиксели рабочей копии).
    """
    out = []
    for block, own, alignment in zip(analysis.blocks, lines, analysis.alignments):
        envelope = block.envelope
        out.append(
            {
                "polygon": np.asarray(envelope.polygon).round(1).tolist(),
                # Ряды и оси блока: мера края большого блока должна весить больше, чем у заголовка в 2–3 строки.
                "rows": len(block.rows),
                "lines": sum(len(row.axes) for row in block.rows),
                # Выключка (``alignment.AlignKind``) и выровненность каждой стороны: меры краёв смотрят только
                # выровненные стороны.
                "alignment": {
                    "kind": alignment.kind.value,
                    "left": bool(alignment.left.aligned),
                    "right": bool(alignment.right.aligned),
                    "left_core_share": round(float(alignment.left.core_share), 3),
                    "right_core_share": round(float(alignment.right.core_share), 3),
                },
                # Недостоверные участки сторон (отрезки по высоте): ступеньки по выносу за колонку и выступы
                # от сора, оставшиеся после защиты сторон (``text_blocks.edge_guard``).
                "unreliable_left": [
                    [round(float(v), 1) for v in span] for span in getattr(envelope, "unreliable_left", ())
                ],
                "unreliable_right": [
                    [round(float(v), 1) for v in span] for span in getattr(envelope, "unreliable_right", ())
                ],
                # Дополнительные линии сторон (метод выравнивания — ``SIDE_ALIGN_METHOD``).
                "sides": {side.value: filled_json(line) for side, line in own.items()},
            }
        )
    return out


def versions() -> dict:
    """Версии семейств ``page_layout`` и детектора текстовых блоков — в JSON полосы, для инвалидации потребителями."""
    from ocr_utils.page_layout import (
        FORMULA_VERSION,
        LINE_ART_VERSION,
        ORIENTATION_VERSION,
        RASTER_VERSION,
        ROTATED_TEXT_VERSION,
        TABLES_VERSION,
    )

    return {
        "tag": version_tag(),
        "raster": RASTER_VERSION,
        "tables": TABLES_VERSION,
        "formulas": FORMULA_VERSION,
        "line_art": LINE_ART_VERSION,
        "rotated_text": ROTATED_TEXT_VERSION,
        "orientation": ORIENTATION_VERSION,
        "text_blocks": TEXT_BLOCKS_VERSION,
        "line_art_decision": DECISION_VERSION,
        "arrays": STORE_FORMAT_VERSION,
    }


def folder_of(objects: list[dict]) -> Path:
    """Папка оверлея по классам объектов: только текст, класс или «несколько_классов»."""
    classes = {o["class"] for o in objects}
    if not classes:
        return Path(ONLY_TEXT)
    return Path(NOT_ONLY_TEXT) / (classes.pop() if len(classes) == 1 else SEVERAL)


def draw(
    image,
    analysis,
    hints,
    record: dict,
    objects: list[dict],
    titles: list,
    orientation: dict,
    lines: list[dict] | None = None,
    body_axis: bool = False,
    marks: list | None = None,
    strokes: list | None = None,
) -> np.ndarray:
    """Оверлей полосы: текстовые блоки (огибающие, оси), стороны блоков, объекты по классам, линейки, легенда.

    Args:
        image: Страница.
        analysis: Разбор текстовых блоков.
        hints: Подсказки детектора блоков (для отрисовки линеек).
        record: JSON полосы.
        objects: Объекты полосы.
        titles: Рамки надписей, снятых с line art.
        orientation: Вердикт ориентации полосы.
        lines: Дополнительные линии сторон блоков (:func:`side_lines`); ``None`` — не рисовать.
        body_axis: Ряды и блоки собраны по второй оси строки (подпись в легенде).
        marks: Рамки пометок на полях, снятых с line art.
        strokes: Рамки одиночных черт, снятых с line art (:func:`split_strokes`).

    Returns:
        Картинка BGR.
    """
    gray = image.gray_at(RENDER_DPI)
    canvas_scale = OVERLAY_WIDTH / analysis.width
    small = cv2.resize(gray, (OVERLAY_WIDTH, int(round(analysis.height * canvas_scale))), interpolation=cv2.INTER_AREA)
    canvas = blocks_overlay.draw(analysis, small, scale=canvas_scale)
    if lines:
        # Линии сторон — толстые, поэтому в слой и полупрозрачно: под ними читаются буквы и граница блока.
        layer = canvas.copy()
        for block_lines in lines:
            for side, line in block_lines.items():
                if line is not None:
                    draw_filled_line(layer, line, canvas_scale, SIDE_THICKNESS, SIDE_COLOR[side], PATCH_COLOR)
        cv2.addWeighted(layer, FILLED_ALPHA, canvas, 1 - FILLED_ALPHA, 0, canvas)
    k = OVERLAY_WIDTH / image.width

    def rect(box) -> tuple[tuple[int, int], tuple[int, int]]:
        return (int(box[0] * k), int(box[1] * k)), (int(box[2] * k), int(box[3] * k))

    layer = canvas.copy()
    for obj in objects:
        color = CLASS_COLOR[PageClass(obj["class"])]
        p0, p1 = rect(obj["box"])
        cv2.rectangle(layer, p0, p1, color, -1)
    cv2.addWeighted(layer, FILL_ALPHA, canvas, 1 - FILL_ALPHA, 0, canvas)
    for obj in objects:
        p0, p1 = rect(obj["box"])
        cv2.rectangle(canvas, p0, p1, CLASS_COLOR[PageClass(obj["class"])], 3)
    for box in titles:
        p0, p1 = rect(box)
        cv2.rectangle(canvas, p0, p1, TITLE_COLOR, 1)
    for box in marks or []:
        p0, p1 = rect(box)
        cv2.rectangle(canvas, p0, p1, MARK_COLOR, 2)
    for box in strokes or []:
        p0, p1 = rect(box)
        cv2.rectangle(canvas, p0, p1, STROKE_COLOR, 2)
    for rule in record["loose_rules"]:
        points = np.array([[int(x * k), int(y * k)] for x, y in rule["points"]], np.int32)
        cv2.polylines(canvas, [points], False, RULE_COLOR, 2)
    return _frame(canvas, record, objects, orientation, analysis, bool(lines), body_axis)


def _frame(
    canvas: np.ndarray,
    record: dict,
    objects: list[dict],
    orientation: dict,
    analysis,
    with_sides: bool = False,
    body_axis: bool = False,
) -> np.ndarray:
    """Шапка (полоса, поворот, счёт объектов и блоков) и легенды классов и текстовых блоков — в полях.

    Обе легенды печатаются ПОД страницей, а не поверх неё (:mod:`ocr_utils.page_layout.overlay_frame`):
    поверх страницы легенда закрывала бы угол полосы.

    Args:
        canvas: Холст страницы с разметкой.
        record: JSON полосы (имя полосы).
        objects: Объекты полосы.
        orientation: Вердикт ориентации.
        analysis: Разбор текстовых блоков (счёт блоков и осей).
        with_sides: Нарисованы ли дополнительные линии сторон (их строки в легенде).
        body_axis: Основная ось строки — вторая (подпись оси в легенде).

    Returns:
        Картинка BGR: шапка, страница, легенда объектов, легенда текстовых блоков.
    """
    counts = {}
    for obj in objects:
        counts[obj["class"]] = counts.get(obj["class"], 0) + 1
    turn = orientation.get("rotate_cw", 0)
    turn_note = (
        f"повёрнута на {turn}°"
        if orientation.get("apply")
        else ("ориентация спорная" if orientation.get("disputed") else "прямая")
    )
    header = [
        f"{record['page']}   {turn_note}   текстовых блоков: {len(analysis.blocks)}, осей строк: {len(analysis.axes)}",
        "объекты: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "только текст"),
    ]
    width = canvas.shape[1]
    classes = [LegendEntry(c.value, CLASS_COLOR[c], FILL_ALPHA, SampleStyle.BOX) for c in PageClass] + [
        LegendEntry("надпись (снята с line art)", TITLE_COLOR),
        LegendEntry("пометка на полях (снята с line art)", MARK_COLOR),
        LegendEntry("одиночная черта (снята с line art)", STROKE_COLOR),
        LegendEntry("линейка-сирота", RULE_COLOR),
    ]
    blocks = blocks_overlay.legend_entries(body_axis=body_axis)
    if with_sides:
        blocks += [
            LegendEntry("левая сторона блока (доп. линия)", SIDE_COLOR[SideKind.LEFT], FILLED_ALPHA),
            LegendEntry("правая сторона блока (доп. линия)", SIDE_COLOR[SideKind.RIGHT], FILLED_ALPHA),
            LegendEntry("сторона: заплатка PCHIP на месте выброса", PATCH_COLOR, FILLED_ALPHA, SampleStyle.DASHED),
        ]
    return np.vstack(
        [
            header_strip(header, width),
            canvas,
            legend_strip(classes, width, "объекты"),
            legend_strip(blocks, width, "текстовые блоки"),
        ]
    )


def final_page(
    task: PageTask,
    orientation: dict,
    deepseek: dict,
    work: Path,
    out: Path,
    axis: AxisKind = AxisKind.CENTRE,
    gutter_mode: GutterMode = GutterMode.SHORT,
    blocks_mode: BlocksMode = DEFAULT_BLOCKS_MODE,
    overlay: OverlayMode = OverlayMode.ALL,
) -> dict:
    """Стадия 6: итог полосы — объекты, текстовые блоки, JSON и оверлей в папку по классам.

    Args:
        task: Полоса.
        orientation: Вердикт стадии 1.
        deepseek: Вывод DeepSeek по кандидатам полосы (см. :func:`stages.decide_candidates`).
        work: Рабочая папка.
        out: Корень выхода (``pages/``, ``overlays/``).
        axis: По какой оси строки собирать ряды и блоки (см. :func:`text_blocks`).
        gutter_mode: Как межколонники становятся запретами сцепки (``columns.GutterMode``).
        blocks_mode: Способ группировки строк в блоки и их границы (``BlocksMode``).
        overlay: Писать ли оверлей полосы.

    Returns:
        Строка описи: полоса, папка, классы, число блоков, время.
    """
    started = time.time()
    record = json.loads((work / "pages" / f"{page_key(task.name)}.json").read_text())
    image = load_image(task, record["rotate_cw"])
    decisions = decide_candidates(record, image, deepseek, work)
    assembled = assemble(record, decisions)
    candidates = [{"id": c["id"], "box": c["crop"]["box"], **d.to_json()} for c, d in decisions]
    return finish_page(
        task, image, record, assembled["objects"], assembled["titles"], candidates, orientation, out, axis,
        gutter_mode, started, blocks_mode, overlay=overlay, marks=assembled["marks"],
    )  # fmt: skip


def _row_jumps(analysis) -> int:
    """Число осей, перескочивших на соседнюю строку: у них разбор проставил участки ``LineAxis.jump_spans``."""
    return sum(1 for axis in analysis.axes if axis.jump_spans)


def _row_jump_spans(analysis) -> list[dict]:
    """Участки перескока осей полосы для JSON.

    Args:
        analysis: Разбор текстовых блоков полосы (``PageAnalysis``).

    Returns:
        Список ``{"axis", "x0", "x1", "y0", "y1"}`` в пикселях рабочей копии: номер оси, отрезок по x и
        ордината оси на его концах.
    """
    out = []
    for index, axis in enumerate(analysis.axes):
        for start, stop in axis.jump_spans:
            out.append(
                {
                    "axis": index,
                    "x0": round(float(start), 1),
                    "x1": round(float(stop), 1),
                    "y0": round(axis.y_at(start), 1),
                    "y1": round(axis.y_at(stop), 1),
                }
            )
    return out


def page_record(final: dict) -> dict:
    """Запись стадии кандидатов, собранная из итогового JSON полосы: всё, что нужно текстовым блокам.

    Текстовым блокам и оверлею из записи кандидатов нужны только ``page``, ``size``, ``dpi``,
    ``loose_rules`` и ``rotate_cw`` — всё это есть в итоговом JSON (поворот — из вердикта ориентации,
    если он применён). Так текстовые блоки пересчитываются по готовому разбору, даже когда его
    рабочая папка (``work/``) почищена.

    Args:
        final: Итоговый JSON полосы (:func:`finish_page`).

    Returns:
        Словарь в форме записи ``work/pages/<полоса>.json``.
    """
    orientation = final.get("orientation") or {}
    return {
        "page": final["page"],
        "size": final["size"],
        "dpi": final["dpi"],
        "loose_rules": final.get("loose_rules", []),
        "rotate_cw": orientation.get("rotate_cw", 0) if orientation.get("apply") else 0,
    }


def reblock_page(
    task: PageTask,
    source: Path,
    out: Path,
    axis: AxisKind = AxisKind.BODY,
    gutter_mode: GutterMode = GutterMode.SHORT,
    blocks_mode: BlocksMode = DEFAULT_BLOCKS_MODE,
    glyph_maps_dir: Path | None = None,
    overlay: OverlayMode = OverlayMode.ALL,
) -> dict:
    """Пересчитать только текстовые блоки полосы по готовому разбору пака; объекты, надписи и кандидаты — как были.

    Нужна после правки детектора текстовых блоков, когда остальные стадии (растр, line art с
    DeepSeek) пересчитывать незачем, а их рабочие файлы могут быть уже почищены: всё берётся из
    итогового JSON прошлого разбора ``<source>/pages/<полоса>.json``.

    Args:
        task: Полоса.
        source: Корень прошлого разбора.
        out: Корень выхода (``pages/``, ``overlays/``).
        axis: Ось строки для рядов и блоков.
        gutter_mode: Как межколонники становятся запретами сцепки.
        blocks_mode: Способ группировки строк в блоки и их границы (``BlocksMode``).
        glyph_maps_dir: Кэш карт CRAFT и pero (``<детектор>/<полоса>.png``) — второй проход защиты сторон, если
            карты полосы там есть; ``None`` — без второго прохода.
        overlay: Писать ли оверлей полосы.

    Returns:
        Строка описи, как у :func:`final_page`.
    """
    started = time.time()
    final = json.loads((source / "pages" / f"{page_key(task.name)}.json").read_text())
    record = page_record(final)
    image = load_image(task, record["rotate_cw"])
    return finish_page(
        task, image, record, final["objects"], final["titles"], final["candidates"], final["orientation"], out,
        axis, gutter_mode, started, blocks_mode, glyph_maps_dir, overlay, final.get("marks", []),
    )  # fmt: skip


def render_for_maps(task: PageTask, source: Path, png_dir: Path) -> str:
    """Рендер полосы ``RENDER_DPI`` для карт CRAFT и pero — тот же, что разбирает детектор текстовых блоков.

    Args:
        task: Полоса.
        source: Корень разбора (поворот полосы — из его итогового JSON).
        png_dir: Куда писать PNG.

    Returns:
        Путь PNG.
    """
    final = json.loads((source / "pages" / f"{page_key(task.name)}.json").read_text())
    image = load_image(task, page_record(final)["rotate_cw"])
    path = png_dir / f"{page_key(task.name)}.png"
    cv2.imwrite(str(path), image.gray_at(RENDER_DPI))
    return str(path)


def finish_page(
    task: PageTask,
    image,
    record: dict,
    objects: list[dict],
    titles: list,
    candidates: list[dict],
    orientation: dict,
    out: Path,
    axis: AxisKind,
    gutter_mode: GutterMode,
    started: float,
    blocks_mode: BlocksMode = DEFAULT_BLOCKS_MODE,
    glyph_maps_dir: Path | None = None,
    overlay: OverlayMode = OverlayMode.ALL,
    marks: list | None = None,
) -> dict:
    """Текстовые блоки полосы, оверлей в папку по классам и итоговый JSON — общий хвост :func:`final_page` и :func:`reblock_page`.

    Args:
        task: Полоса.
        image: Страница (``PageImage``).
        record: Запись стадии кандидатов (или :func:`page_record`).
        objects: Объекты полосы (:func:`assemble`).
        titles: Рамки надписей, снятых с line art.
        candidates: Кандидаты line art с решениями DeepSeek (в JSON как есть).
        orientation: Вердикт ориентации.
        out: Корень выхода.
        axis: Ось строки для рядов и блоков.
        gutter_mode: Как межколонники становятся запретами сцепки.
        started: Время начала обработки полосы (``time.time()``) — для поля ``seconds``.
        blocks_mode: Способ группировки строк в блоки и их границы (``BlocksMode``).
        glyph_maps_dir: Кэш карт CRAFT и pero — второй проход защиты сторон, если карты полосы там есть.
        overlay: Писать ли оверлей полосы (``OverlayMode.NONE`` — только JSON и сайдкар).
        marks: Рамки пометок на полях, снятых с line art (в JSON — ``marks``; не запрет для текстовых блоков).

    Одиночные черты (рамки рисунков и «неясно» тоньше ``decide.STROKE_MAX_MM``) снимаются с объектов здесь же
    (:func:`split_strokes`) — в JSON это ``strokes``.

    Рядом с JSON пишется сайдкар ``<полоса>.npz`` (:mod:`text_blocks.store`): оси строк с участками
    перескоков и точек, стороны блоков (сырые и с заплатками), трассы линеек таблиц.

    Returns:
        Строка описи: полоса, папка, классы, число блоков, время.
    """
    # Одиночные черты — не рисунок: снимаются с объектов до текстовых блоков (не запрет для строк) и до JSON.
    objects, strokes = split_strokes(objects, record["dpi"])
    maps = load_maps(glyph_maps_dir, page_key(task.name)) if glyph_maps_dir is not None else None
    analysis, hints = text_blocks(image, record, objects, axis, gutter_mode, blocks_mode=blocks_mode, glyph_maps=maps)
    lines = side_lines(analysis)
    folder = folder_of(objects)
    if overlay is OverlayMode.ALL:
        picture = draw(
            image, analysis, hints, record, objects, titles, orientation, lines, axis is AxisKind.BODY, marks, strokes
        )
        target = out / "overlays" / folder / f"{page_key(task.name)}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if orientation.get("disputed"):
            spare = out / "overlays" / DISPUTED_ORIENTATION / f"{page_key(task.name)}.jpg"
            spare.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(spare), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
    # Сайдкар — до JSON: JSON полосы означает «полоса готова» и ссылается на сайдкар.
    arrays_name = f"{page_key(task.name)}.npz"
    write_arrays(out / "pages" / arrays_name, page_arrays(analysis, lines, table_traces(image, objects)))
    result = {
        "page": task.name,
        "size": record["size"],
        "dpi": record["dpi"],
        "orientation": orientation,
        "objects": objects,
        "titles": titles,
        # Пометки на полях, снятые с line art (``line_art.deepseek.marks``): рамки кандидатов в пикселях полосы.
        "marks": list(marks or []),
        # Одиночные черты, снятые с line art (:func:`split_strokes`): рамки в пикселях полосы.
        "strokes": strokes,
        "candidates": candidates,
        "loose_rules": record["loose_rules"],
        "text_blocks": {
            "dpi": WORK_DPI,
            "axis": axis.value,
            "gutter_mode": gutter_mode.value,
            "blocks_mode": BlocksMode(blocks_mode).value,
            # Перескоков оси на соседнюю строку (``metrics.row_jumps_of``) — мера качества полосы.
            "row_jumps": _row_jumps(analysis),
            # Участки перескока (``LineAxis.jump_spans``): номер оси в разборе, отрезок по x и ордината
            # оси на его концах — туда меры наклона и формы строки не смотрят.
            "row_jump_spans": _row_jump_spans(analysis),
            # Защита выровненных сторон от сора: выступы, второй проход CRAFT + pero, выброшенная краска.
            "edge_guard": analysis.edge_guard.to_json() if analysis.edge_guard is not None else None,
            "count": len(analysis.blocks),
            "axes": len(analysis.axes),
            "blocks": blocks_json(analysis, lines),
            # Оси строк, стороны блоков и трассы линеек таблиц — в сайдкаре (``text_blocks.store.load_page``).
            "arrays": arrays_name,
        },
        "folder": str(folder),
        "versions": versions(),
        "seconds": round(time.time() - started, 2),
    }
    write_json(out / "pages" / f"{page_key(task.name)}.json", result)
    return {
        "page": task.name,
        "folder": str(folder),
        "classes": ",".join(sorted({o["class"] for o in objects})),
        "blocks": len(analysis.blocks),
        "rotate_cw": record["rotate_cw"],
        "seconds": result["seconds"],
    }


__all__ = [
    "OverlayMode",
    "PageClass",
    "blocks_json",
    "assemble",
    "final_page",
    "finish_page",
    "folder_of",
    "page_record",
    "reblock_page",
    "render_for_maps",
    "side_lines",
    "table_traces",
    "text_blocks",
    "versions",
]
