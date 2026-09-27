"""Слияние областей детектора line art с вердиктом DeepSeek-OCR-2 как классификатора: рисунок, таблица, формула или надпись.

Правило (по решению пользователя, reports/line_art_titles.md): DeepSeek точно называет
``image`` / ``table`` / ``equation``, но неточно обводит — поэтому класс берётся у DeepSeek, а рамка
по возможности у нашего детектора.

* Блоки DeepSeek (промпт ``markdown``), лежащие на рамке области хотя бы наполовину своей площади.
* Ни одного нетекстового класса — правило «надпись» (``features.is_title_like_deepseek``):
  надпись или «неясно».
* Один класс — один объект этого класса с **рамкой детектора**.
* Два и больше — по объекту на каждый нетекстовый блок, рамка от DeepSeek, переведённая в пиксели
  полосы и **достроенная** до краёв пятен, которые режет (``line_art.expand.grow_to_components``).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.line_art.classes import ObjectClass
from ocr_utils.page_layout.line_art.expand import FIGURE_GROW_MM, FORMULA_GROW_MM, grow_to_components
from research.line_art_titles.features import is_title_like_deepseek, on_region

# Правила — из основного пакета (перенесены 2026-09-27); здесь остаются слияние стенда и оверлей.
from ocr_utils.page_layout.line_art.deepseek.rules import (  # noqa: E402
    CLASS_BY_LABEL,
    CROP_DPI,
    HALLUCINATION,
    HALLUCINATION_MIN,
    MATH,
    block_class,
    is_hallucinated,
)


class Verdict(str, Enum):
    """Итог области, если в ней нет нетекстового объекта: надпись или неясно (папки оверлеев)."""

    TITLE = ObjectClass.TITLE.value
    UNCLEAR = "неясно"


class BoxSource(str, Enum):
    """Чья рамка у объекта."""

    DETECTOR = "детектор"
    DEEPSEEK = "DeepSeek"


def crop_to_page(element: dict, row: dict) -> Box:
    """Рамка блока DeepSeek (пиксели вырезки 300 dpi) → пиксели полосы.

    Args:
        element: Блок DeepSeek (``x0 … y1`` в вырезке).
        row: Строка ``regions.jsonl`` (``box`` — рамка области на полосе, ``crop_inner`` — она же в вырезке, ``dpi``).

    Returns:
        Рамка в пикселях полосы.
    """
    scale = row["dpi"] / CROP_DPI
    ox = row["box"][0] - row["crop_inner"][0] * scale
    oy = row["box"][1] - row["crop_inner"][1] * scale
    return Box(
        int(round(ox + element["x0"] * scale)),
        int(round(oy + element["y0"] * scale)),
        int(round(ox + element["x1"] * scale)),
        int(round(oy + element["y1"] * scale)),
    )


def merge_region(row: dict, features: dict, blocks: list[dict], ink: np.ndarray, barriers: list[Box]) -> dict:
    """Итог одной области: список объектов с классом и рамкой либо вердикт «надпись» / «неясно».

    Args:
        row: Строка ``regions.jsonl``.
        features: Строка ``features_deepseek_vllm.csv`` этой области.
        blocks: Блоки DeepSeek ``markdown`` в пикселях вырезки.
        ink: Маска краски полосы в родном разрешении (ненулевое — краска).
        barriers: Известные растр и таблицы полосы — за них достройка не заходит.

    Returns:
        ``{"id", "page", "classes", "objects": [{"class", "box", "box_source", "failed"}], "verdict"}``;
        ``verdict`` — ``None``, если объекты есть.
    """
    regional = [b for b in blocks if block_class(b) is not None and on_region(b, row["crop_inner"])]
    classes = sorted({block_class(b).value for b in regional})
    height, width = ink.shape[:2]
    objects: list[dict] = []
    verdict = None
    if not classes:
        verdict = (Verdict.TITLE if is_title_like_deepseek(features) else Verdict.UNCLEAR).value
    elif len(classes) == 1:
        box, failed = Box(*row["box"]), []
        if classes[0] == ObjectClass.FORMULA.value:
            # Формула: рамка детектора уточняется, чтобы не резать пятна (индексы, дробная черта).
            grown = grow_to_components(box, ink, row["dpi"], FORMULA_GROW_MM, barriers)
            box, failed = grown.box, list(grown.failed)
        objects.append(
            {"class": classes[0], "box": list(box.as_tuple()), "box_source": BoxSource.DETECTOR.value, "failed": failed}
        )
    else:
        for block in regional:
            cls = block_class(block)
            limit = FORMULA_GROW_MM if cls is ObjectClass.FORMULA else FIGURE_GROW_MM
            box = crop_to_page(block, row).clipped(width, height)
            grown = grow_to_components(box, ink, row["dpi"], limit, barriers)
            objects.append(
                {
                    "class": cls.value,
                    "box": list(grown.box.as_tuple()),
                    "deepseek_box": list(box.as_tuple()),
                    "box_source": BoxSource.DEEPSEEK.value,
                    "failed": list(grown.failed),
                }
            )
    return {"id": row["id"], "page": row["page"], "classes": classes, "objects": objects, "verdict": verdict}


def merge_page(
    page: str, rows: list[dict], features: dict, blocks: dict, barriers: list[Box], sharpened_dir: Path
) -> list[dict]:
    """Слияние всех областей одной полосы (полоса читается один раз, краска — только при нужде).

    Args:
        page: Имя полосы.
        rows: Строки ``regions.jsonl`` полосы.
        features: Признаки областей по id.
        blocks: Блоки DeepSeek областей по id.
        barriers: Растр и таблицы полосы из базы.
        sharpened_dir: Корень заострённых копий.

    Returns:
        Итоги областей (:func:`merge_region`).
    """
    from ocr_utils.page_layout.image import PageImage, Variant

    image = PageImage.from_file(sharpened_dir / f"{page}.jpg", Variant.SHARPENED, page, default_dpi=600)
    ink = image.bitonal_at(image.dpi) == 0
    return [merge_region(row, features[row["id"]], blocks.get(row["id"], []), ink, barriers) for row in rows]


__all__ = [
    "BoxSource",
    "CLASS_BY_LABEL",
    "MATH",
    "block_class",
    "Verdict",
    "crop_to_page",
    "merge_page",
    "merge_region",
]


# Оверлей итога: окрестность области, рамка детектора, блоки DeepSeek, итоговые объекты по классам.
CONTEXT_MM = 6.0
VIEW_MAX_SIDE = 1400
COLOR_DETECTOR = (170, 170, 170)
COLOR_BLOCK = (200, 60, 200)
COLOR_BY_CLASS = {
    ObjectClass.DRAWING.value: (220, 90, 20),
    ObjectClass.TABLE.value: (0, 150, 0),
    ObjectClass.FORMULA.value: (0, 165, 255),
}
# Папка оверлея, если объектов несколько разных классов.
MIXED_FOLDER = "несколько_классов"


def folder_of(result: dict) -> str:
    """Папка оверлея по итогу: класс объекта, «несколько_классов» или вердикт (надпись / неясно)."""
    if result["verdict"] is not None:
        return result["verdict"]
    classes = {o["class"] for o in result["objects"]}
    return classes.pop() if len(classes) == 1 else MIXED_FOLDER


def draw_merged(
    gray: np.ndarray, dpi: int, row: dict, blocks: list[dict], result: dict, label: str | None
) -> np.ndarray:
    """Оверлей итога слияния одной области.

    Args:
        gray: Серая полоса в родном разрешении.
        dpi: Разрешение полосы.
        row: Строка ``regions.jsonl``.
        blocks: Блоки DeepSeek в пикселях вырезки.
        result: Итог :func:`merge_region`.
        label: Ручная разметка, если есть.

    Returns:
        Картинка BGR с шапкой и легендой.
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    height, width = gray.shape[:2]
    margin = int(round(CONTEXT_MM / 25.4 * dpi))
    box = Box(*row["box"])
    boxes = [box] + [Box(*o["box"]) for o in result["objects"]]
    view = Box(
        min(b.x0 for b in boxes) - margin,
        min(b.y0 for b in boxes) - margin,
        max(b.x1 for b in boxes) + margin,
        max(b.y1 for b in boxes) + margin,
    ).clipped(width, height)
    scale = min(1.0, VIEW_MAX_SIDE / max(view.width, view.height))
    canvas = cv2.cvtColor(
        cv2.resize(gray[view.slice], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR
    )

    def rect(b: Box, color, thickness, inset=0):
        p0 = (int((b.x0 - view.x0) * scale) + inset, int((b.y0 - view.y0) * scale) + inset)
        p1 = (int((b.x1 - view.x0) * scale) - inset, int((b.y1 - view.y0) * scale) - inset)
        cv2.rectangle(canvas, p0, p1, color, thickness)
        return p0

    rect(box, COLOR_DETECTOR, 2)
    for block in blocks:
        p0 = rect(crop_to_page(block, row), COLOR_BLOCK, 1, 2)
        cv2.putText(canvas, block["label"], (p0[0] + 3, p0[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BLOCK, 1)
    for obj in result["objects"]:
        rect(Box(*obj["box"]), COLOR_BY_CLASS.get(obj["class"], (0, 0, 220)), 3, -3)
    if canvas.shape[1] < 800:
        canvas = np.hstack([canvas, np.full((canvas.shape[0], 800 - canvas.shape[1], 3), 255, np.uint8)])
    what = result["verdict"] or ", ".join(f"{o['class']} (рамка: {o['box_source']})" for o in result["objects"])
    lines = [
        f"{row['page']}, область {row['id'].rsplit('_', 1)[-1]}   итог: {what}"
        + (f"   разметка: {label}" if label else ""),
        f"блоки DeepSeek: {', '.join(b['label'] for b in blocks) or '—'}",
    ]
    legend = [(COLOR_DETECTOR, "рамка детектора line art"), (COLOR_BLOCK, "блок DeepSeek-OCR-2 с типом")] + [
        (color, f"итог: {name}") for name, color in COLOR_BY_CLASS.items()
    ]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    head = Image.new("RGB", (canvas.shape[1], 10 + 22 * len(lines)), (255, 255, 255))
    draw = ImageDraw.Draw(head)
    for i, line in enumerate(lines):
        draw.text((8, 5 + 22 * i), line, font=font, fill=(20, 20, 20))
    foot = Image.new("RGB", (canvas.shape[1], 10 + 20 * ((len(legend) + 1) // 2)), (255, 255, 255))
    draw = ImageDraw.Draw(foot)
    for i, (color, name) in enumerate(legend):
        x, y = 8 + (i % 2) * (canvas.shape[1] // 2), 5 + 20 * (i // 2)
        draw.rectangle((x, y + 3, x + 26, y + 15), outline=color[::-1], width=2)
        draw.text((x + 34, y), name, font=font, fill=(20, 20, 20))
    to_bgr = lambda image: cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  # noqa: E731
    return np.vstack([to_bgr(head), canvas, to_bgr(foot)])
