"""Листы отсмотра: блоки разбора цветом выключки по способам в ряд у стыков множества, номера рядов, риски стыков."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from ocr_utils.page_layout.text_blocks import WORK_DPI
from ocr_utils.page_layout.text_blocks.alignment import AlignKind
from ocr_utils.page_layout.text_blocks.overlay import VERDICT_COLOUR
from research.flush_split.labels import Expect, Joint, joint_cut, mm_to_work

# Прозрачность заливки рядов.
ALPHA = 0.35
# Окно вокруг стыка по высоте, мм.
WINDOW_MM = 30.0
# Риски стыков у левого края страницы: ожидается разрез — красная, один блок — серая.
CUT_COLOUR = (0, 0, 220)
KEEP_COLOUR = (150, 150, 150)
# Увеличение вырезки на листе.
ZOOM = 2.0

LEGEND = [
    LegendEntry("блок по формату", VERDICT_COLOUR[AlignKind.BOTH], alpha=ALPHA, style=SampleStyle.BOX),
    LegendEntry("блок вправо или влево", VERDICT_COLOUR[AlignKind.RIGHT], alpha=ALPHA, style=SampleStyle.BOX),
    LegendEntry("блок без выключки", VERDICT_COLOUR[AlignKind.RAGGED], alpha=ALPHA, style=SampleStyle.BOX),
    LegendEntry("блок по центру", VERDICT_COLOUR[AlignKind.CENTER], alpha=ALPHA, style=SampleStyle.BOX),
    LegendEntry("стык: ожидается разрез (риска слева, ✓/✗ — разрезан ли)", CUT_COLOUR),
    LegendEntry("стык: ожидается один блок", KEEP_COLOUR),
]


def work_gray(png: Path) -> np.ndarray:
    """Серая вырезка в ``WORK_DPI`` (рабочая копия разбора, координаты рядов сводки)."""
    meta = json.loads(png.with_suffix(".json").read_text())
    gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    f = WORK_DPI / meta["dpi"]
    return cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def draw_blocks(gray: np.ndarray, blocks: list[dict]) -> np.ndarray:
    """Ряды блоков полупрозрачно цветом выключки, рамки блоков и номер блока.ряда слева от каждого ряда.

    Args:
        gray: Рабочая копия.
        blocks: Блоки сводки разбора.

    Returns:
        BGR той же формы.
    """
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    layer = canvas.copy()
    for b in blocks:
        colour = VERDICT_COLOUR[AlignKind(b["align"])]
        for y, x0, x1, h, *_ in b["rows"]:
            cv2.rectangle(layer, (int(x0), int(y - h / 2)), (int(x1), int(y + h / 2)), colour, -1)
    cv2.addWeighted(layer, ALPHA, canvas, 1 - ALPHA, 0, canvas)
    for bi, b in enumerate(blocks):
        colour = VERDICT_COLOUR[AlignKind(b["align"])]
        ys = [r[0] - r[3] / 2 for r in b["rows"]] + [r[0] + r[3] / 2 for r in b["rows"]]
        x0, x1 = min(r[1] for r in b["rows"]), max(r[2] for r in b["rows"])
        cv2.rectangle(canvas, (int(x0) - 2, int(min(ys)) - 2), (int(x1) + 2, int(max(ys)) + 2), colour, 1)
    return canvas


def mark_joints(canvas: np.ndarray, blocks: list[dict], joints: list[Joint]) -> None:
    """Риски стыков у левого края страницы и знак ✓/✗ — разрезан ли стык в этом разборе (рисует на ``canvas``)."""
    for j in joints:
        y = int(mm_to_work(j.y_mm))
        colour = CUT_COLOUR if j.expect is Expect.CUT else KEEP_COLOUR
        cv2.line(canvas, (0, y), (14, y), colour, 2)
        cut = joint_cut(blocks, j)
        sign = "?" if cut is None else ("+" if cut else "-")
        cv2.putText(canvas, sign, (16, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)


def joint_sheet(rec: dict, modes: tuple[str, ...], joints: list[Joint], y_mm: float) -> np.ndarray:
    """Лист одного стыка: окно ±``WINDOW_MM`` вокруг ``y_mm``, разборы способов в ряд.

    Args:
        rec: Запись прогона (``bench.run_page``).
        modes: Способы в ряд.
        joints: Стыки страницы (рисуются все, попавшие в окно).
        y_mm: Середина окна, мм.

    Returns:
        Лист BGR со шапкой и легендой.
    """
    gray = work_gray(Path(rec["path"]))
    y0 = max(0, int(mm_to_work(y_mm - WINDOW_MM)))
    y1 = min(gray.shape[0], int(mm_to_work(y_mm + WINDOW_MM)))
    panels = []
    for mode in modes:
        img = draw_blocks(gray, rec[mode])
        mark_joints(img, rec[mode], joints)
        crop = cv2.resize(img[y0:y1], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
        title = np.full((22, crop.shape[1], 3), 255, np.uint8)
        cv2.putText(title, mode, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        panels.append(np.vstack([title, crop]))
    sep = np.full((panels[0].shape[0], 12, 3), 255, np.uint8)
    sep[:, 5:7] = (0, 0, 200)
    body = panels[0]
    for panel in panels[1:]:
        body = np.hstack([body, sep, panel])
    kinds = ", ".join(sorted({f"{j.kind}:{j.expect.value}" for j in joints if abs(j.y_mm - y_mm) <= WINDOW_MM}))
    header = [f"{rec['page']} — y {y_mm:.0f} мм ±{WINDOW_MM:.0f}, ×{ZOOM:.0f}; стыки: {kinds}"]
    return framed(body, header, LEGEND)
