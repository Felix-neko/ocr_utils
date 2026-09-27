"""Итоговый разбор полосы: классы объектов (растр, таблицы, line art по решению DeepSeek, формулы, повёрнутый текст), текстовые блоки с запретами, оверлей и папка по классам."""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.line_art.classes import ObjectClass
from ocr_utils.page_layout.line_art.deepseek.decide import Outcome
from ocr_utils.page_layout.pack_analysis.stages import PageTask, decide_candidates, load_image, page_key, write_json
from ocr_utils.page_layout.regions import RegionKind
from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks import overlay as blocks_overlay
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.from_layout import build
from ocr_utils.page_layout.text_blocks.page import analyse_gray


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
RULE_COLOR = (60, 90, 210)  # линейки-сироты (text_blocks.overlay.COLOUR_BARRIER)
FILL_ALPHA = 0.18
OVERLAY_WIDTH = 1600
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def assemble(record: dict, decisions: list) -> dict:
    """Объекты полосы по классам из разбора и решений по кандидатам line art.

    Args:
        record: JSON полосы стадии кандидатов.
        decisions: Пары (кандидат, :class:`Decision`).

    Returns:
        ``{"objects": [{"class", "box", "source"}], "titles": [рамки]}`` в пикселях полосы.
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
    titles = []
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
        else:
            titles.append(candidate["crop"]["box"])
    return {"objects": objects, "titles": titles}


def text_blocks(image, record: dict, objects: list[dict]):
    """Текстовые блоки полосы с запретами: растр, печати, таблицы, line art, формулы; барьеры — рамки и линейки.

    «Неясно» и надписи в запрет не идут (решение пользователя: «неясно» — отдельный класс для
    просмотра, надписи разбирает детектор строк). Повёрнутый текст — боковые зоны.

    Args:
        image: Страница.
        record: JSON полосы.
        objects: Объекты :func:`assemble`.

    Returns:
        ``PageAnalysis`` детектора текстовых блоков (пиксели рабочей копии 150 dpi).
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
        analyse_gray(gray300, InkEngine(hints=hints), hints=hints, name=page_key(record["page"]), variant="sharpened"),
        hints,
    )


def folder_of(objects: list[dict]) -> Path:
    """Папка оверлея по классам объектов: только текст, класс или «несколько_классов»."""
    classes = {o["class"] for o in objects}
    if not classes:
        return Path(ONLY_TEXT)
    return Path(NOT_ONLY_TEXT) / (classes.pop() if len(classes) == 1 else SEVERAL)


def draw(image, analysis, hints, record: dict, objects: list[dict], titles: list, orientation: dict) -> np.ndarray:
    """Оверлей полосы: текстовые блоки (огибающие, оси), объекты по классам полупрозрачно, линейки, легенда.

    Args:
        image: Страница.
        analysis: Разбор текстовых блоков.
        hints: Подсказки детектора блоков (для отрисовки линеек).
        record: JSON полосы.
        objects: Объекты полосы.
        titles: Рамки надписей, снятых с line art.
        orientation: Вердикт ориентации полосы.

    Returns:
        Картинка BGR.
    """
    gray = image.gray_at(RENDER_DPI)
    canvas_scale = OVERLAY_WIDTH / analysis.width
    small = cv2.resize(gray, (OVERLAY_WIDTH, int(round(analysis.height * canvas_scale))), interpolation=cv2.INTER_AREA)
    canvas = blocks_overlay.draw(analysis, small, scale=canvas_scale)
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
    for rule in record["loose_rules"]:
        points = np.array([[int(x * k), int(y * k)] for x, y in rule["points"]], np.int32)
        cv2.polylines(canvas, [points], False, RULE_COLOR, 2)
    return _frame(canvas, record, objects, orientation, analysis)


def _frame(canvas: np.ndarray, record: dict, objects: list[dict], orientation: dict, analysis) -> np.ndarray:
    """Шапка (полоса, поворот, счёт объектов и блоков) и легенда классов под картинкой."""
    font = ImageFont.truetype(FONT_PATH, 18)
    small = ImageFont.truetype(FONT_PATH, 15)
    counts = {}
    for obj in objects:
        counts[obj["class"]] = counts.get(obj["class"], 0) + 1
    turn = orientation.get("rotate_cw", 0)
    turn_note = (
        f"повёрнута на {turn}°"
        if orientation.get("apply")
        else ("ориентация спорная" if orientation.get("disputed") else "прямая")
    )
    lines = [
        f"{record['page']}   {turn_note}   текстовых блоков: {len(analysis.blocks)}, осей строк: {len(analysis.axes)}",
        "объекты: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "только текст"),
    ]
    width = canvas.shape[1]
    head = Image.new("RGB", (width, 12 + 26 * len(lines)), (255, 255, 255))
    draw_head = ImageDraw.Draw(head)
    for index, line in enumerate(lines):
        draw_head.text((10, 6 + 26 * index), line, font=font, fill=(20, 20, 20))
    legend = [(CLASS_COLOR[c], c.value) for c in PageClass] + [
        (TITLE_COLOR, "надпись (снята с line art)"),
        (RULE_COLOR, "линейка-сирота"),
    ]
    columns = 4
    rows = (len(legend) + columns - 1) // columns
    foot = Image.new("RGB", (width, 12 + 24 * rows + 24), (255, 255, 255))
    draw_foot = ImageDraw.Draw(foot)
    for index, (color, name) in enumerate(legend):
        x, y = 10 + (index % columns) * (width // columns), 6 + 24 * (index // columns)
        fill = tuple(int(FILL_ALPHA * c + (1 - FILL_ALPHA) * 255) for c in color)
        draw_foot.rectangle((x, y + 3, x + 28, y + 17), fill=fill[::-1], outline=color[::-1], width=2)
        draw_foot.text((x + 36, y), name, font=small, fill=(20, 20, 20))
    draw_foot.text(
        (10, 6 + 24 * rows),
        "текстовые блоки — огибающие и оси строк (легенда детектора блоков в углу картинки)",
        font=small,
        fill=(20, 20, 20),
    )
    to_bgr = lambda picture: cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR)  # noqa: E731
    return np.vstack([to_bgr(head), canvas, to_bgr(foot)])


def final_page(task: PageTask, orientation: dict, deepseek: dict, work: Path, out: Path) -> dict:
    """Стадия 6: итог полосы — объекты, текстовые блоки, JSON и оверлей в папку по классам.

    Args:
        task: Полоса.
        orientation: Вердикт стадии 1.
        deepseek: Вывод DeepSeek по кандидатам полосы (см. :func:`stages.decide_candidates`).
        work: Рабочая папка.
        out: Корень выхода (``pages/``, ``overlays/``).

    Returns:
        Строка описи: полоса, папка, классы, число блоков, время.
    """
    started = time.time()
    record = __import__("json").loads((work / "pages" / f"{page_key(task.name)}.json").read_text())
    image = load_image(task, record["rotate_cw"])
    decisions = decide_candidates(record, image, deepseek, work)
    assembled = assemble(record, decisions)
    analysis, hints = text_blocks(image, record, assembled["objects"])
    folder = folder_of(assembled["objects"])
    picture = draw(image, analysis, hints, record, assembled["objects"], assembled["titles"], orientation)
    target = out / "overlays" / folder / f"{page_key(task.name)}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if orientation.get("disputed"):
        spare = out / "overlays" / DISPUTED_ORIENTATION / f"{page_key(task.name)}.jpg"
        spare.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(spare), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
    result = {
        "page": task.name,
        "size": record["size"],
        "dpi": record["dpi"],
        "orientation": orientation,
        "objects": assembled["objects"],
        "titles": assembled["titles"],
        "candidates": [{"id": c["id"], "box": c["crop"]["box"], **d.to_json()} for c, d in decisions],
        "loose_rules": record["loose_rules"],
        "text_blocks": {
            "dpi": WORK_DPI,
            "count": len(analysis.blocks),
            "axes": len(analysis.axes),
            "blocks": [{"polygon": np.asarray(b.envelope.polygon).round(1).tolist()} for b in analysis.blocks],
        },
        "folder": str(folder),
        "seconds": round(time.time() - started, 2),
    }
    write_json(out / "pages" / f"{page_key(task.name)}.json", result)
    return {
        "page": task.name,
        "folder": str(folder),
        "classes": ",".join(sorted({o["class"] for o in assembled["objects"]})),
        "blocks": len(analysis.blocks),
        "rotate_cw": record["rotate_cw"],
        "seconds": result["seconds"],
    }


__all__ = ["PageClass", "assemble", "final_page", "folder_of", "text_blocks"]
