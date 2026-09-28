"""Выборка крайних компонентов строк для судей: решения правила по набору, вырезки с обводкой, разметка глазами.

Компоненты берутся из журналов решений прогона с правилом (``<eval>/<правило>/*.json``): всё, что правило
признало мусором, все «спорно», пометки, надстрочные и запятые — целиком, точки — выборкой. Вырезка —
окрестность компонента с красной рамкой вокруг него (для разметки: знак или мусор).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import fitz
import numpy as np

from ocr_utils.geometry_regression.render import render_gray
from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from research.text_block_specks.filters import End, EndComponent
from research.text_block_specks.rules import Kind, kind_of
from research.text_block_specks.sheets import labelled

K = RENDER_DPI / WORK_DPI
# Сколько точек взять выборкой (их на наборе больше тысячи).
PERIOD_SAMPLE = 100
# Поле вырезки для разметки в высотах строчной: вглубь строки, наружу, вверх и вниз.
VIEW_IN_XH, VIEW_OUT_XH, VIEW_UP_XH, VIEW_DOWN_XH = 7.0, 3.0, 2.0, 1.5
# Высота вырезки на листе, пиксели.
VIEW_HEIGHT = 110
# Рамка компонента на вырезке.
COLOUR_BOX = (0, 0, 230)


def collect(eval_dir: Path, rule: str, seed: int = 20260928) -> list[dict]:
    """Компоненты для разметки из журналов решений прогона ``rule``.

    Args:
        eval_dir: Корень прогона стенда.
        rule: Имя правила (папка с JSON).
        seed: Зерно выборки точек.

    Returns:
        Словари: ``id``, ``key``, ``pdf``, ``page``, ``end``, ``source``, ``kind``, ``noise``, ``box`` и признаки.
    """
    keep_all = {Kind.SPECK, Kind.UNSURE, Kind.MARK, Kind.SUPERSCRIPT, Kind.COMMA}
    chosen, periods = [], []
    for path in sorted((eval_dir / rule).glob("*.json")):
        page = json.loads(path.read_text())
        for decision in page["decisions"]:
            component = EndComponent(box=tuple(decision["box"]), **decision["component"])
            kind = kind_of(component, End(decision["end"]))
            record = {
                "key": page["key"],
                "pdf": page["pdf"],
                "page": page["page"],
                "end": decision["end"],
                "source": decision["source"],
                "kind": kind.value,
                "noise": decision["noise"],
                "box": decision["box"],
                **decision["component"],
            }
            if kind in keep_all:
                chosen.append(record)
            elif kind is Kind.PERIOD:
                periods.append(record)
    random.Random(seed).shuffle(periods)
    out = chosen + periods[:PERIOD_SAMPLE]
    for number, record in enumerate(out, 1):
        record["id"] = f"k{number:04d}"
    return out


def view(gray: np.ndarray, record: dict) -> np.ndarray:
    """Вырезка вокруг компонента с красной рамкой, приведённая к высоте ``VIEW_HEIGHT``."""
    x0, y0, x1, y1 = (v * K for v in record["box"])
    x_h = record["x_h"] * K
    if record["end"] == "right":
        left, right = x0 - VIEW_IN_XH * x_h, x1 + VIEW_OUT_XH * x_h
    else:
        left, right = x0 - VIEW_OUT_XH * x_h, x1 + VIEW_IN_XH * x_h
    top, bottom = y0 - VIEW_UP_XH * x_h, y1 + VIEW_DOWN_XH * x_h
    box = [int(max(0, left)), int(max(0, top)), int(min(gray.shape[1], right)), int(min(gray.shape[0], bottom))]
    crop = cv2.cvtColor(gray[box[1] : box[3], box[0] : box[2]], cv2.COLOR_GRAY2BGR)
    pad = 3
    cv2.rectangle(
        crop,
        (int(x0) - box[0] - pad, int(y0) - box[1] - pad),
        (int(x1) - box[0] + pad, int(y1) - box[1] + pad),
        COLOUR_BOX,
        1,
    )
    scale = VIEW_HEIGHT / crop.shape[0]
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def views(records: list[dict], pdf_dir: Path) -> dict[str, np.ndarray]:
    """Вырезки для разметки с подписью ``id``, по одному рендеру на страницу."""
    by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        by_page[(record["pdf"], int(record["page"]))].append(record)
    out = {}
    for (pdf, page), items in by_page.items():
        with fitz.open(pdf_dir / pdf) as document:
            gray = render_gray(document, page - 1)
        for record in items:
            out[record["id"]] = labelled(view(gray, record), f"{record['id']} {record['kind'][:4]}")
    return out


def grid(images: list[np.ndarray], columns: int = 5, rows: int = 8, width: int = 330) -> list[np.ndarray]:
    """Листы с вырезками разной ширины: каждая вписывается в ячейку ``width`` пикселей (обрезка справа)."""
    cell_h = max(image.shape[0] for image in images) + 4
    sheets = []
    per = columns * rows
    for start in range(0, len(images), per):
        chunk = images[start : start + per]
        used = (len(chunk) + columns - 1) // columns
        sheet = np.full((used * cell_h, columns * (width + 4), 3), 200, dtype=np.uint8)
        for index, image in enumerate(chunk):
            r, c = divmod(index, columns)
            part = image[:, :width]
            sheet[r * cell_h : r * cell_h + part.shape[0], c * (width + 4) : c * (width + 4) + part.shape[1]] = part
        sheets.append(sheet)
    return sheets


__all__ = ["collect", "grid", "view", "views"]
