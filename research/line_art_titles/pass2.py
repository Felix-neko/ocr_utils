"""Второй проход стенда: логика — в основном пакете (``ocr_utils.page_layout.line_art.deepseek.pass2``), здесь — оверлей."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.line_art.deepseek.pass2 import (  # noqa: F401 — реэкспорт для CLI стенда
    classic_boxes,
    despeckle,
    fill_words,
    has_body,
    strip_rules,
    tight_box,
    word_boxes_grown,
)
from ocr_utils.page_layout.line_art.deepseek.pass2 import verdict_pass2 as _objects_pass2


def verdict_pass2(binary: np.ndarray, inner, blocks: list[dict], classic: list) -> dict:
    """Формат стенда: ``{"verdict": "объект" | "надпись", "objects": [...]}`` поверх функции пакета."""
    objects = _objects_pass2(binary, inner, blocks, classic)
    return {"verdict": "объект" if objects else "надпись", "objects": objects}


class Pass2Verdict(str, __import__("enum").Enum):
    """Итог второго прохода по области (папка оверлея стенда)."""

    OBJECT = "объект"
    TITLE = "надпись"


# Оверлей: слева исходная вырезка с залитыми словами, справа залитая с находками второго прохода.
COLOR_WORD_FILL = (0, 165, 255)
COLOR_BLOCK = (200, 60, 200)
COLOR_CLASSIC = (120, 120, 120)
COLOR_FRAME = (170, 170, 170)
COLOR_OBJECT = (220, 90, 20)
WORD_ALPHA = 0.45
PANEL_MAX = 900


def draw_pass2(
    gray: np.ndarray,
    binary: np.ndarray,
    row: dict,
    filled: list,
    blocks: list[dict],
    classic: list,
    result: dict,
    label: str | None,
) -> np.ndarray:
    """Оверлей второго прохода по области.

    Args:
        gray: Исходная серая вырезка.
        binary: Залитая вырезка.
        row: Строка ``regions.jsonl``.
        filled: Залитые рамки слов.
        blocks: Блоки DeepSeek второго прохода.
        classic: Рамки классики.
        result: Итог :func:`verdict_pass2`.
        label: Ручная разметка.

    Returns:
        Картинка BGR.
    """
    from PIL import Image, ImageDraw, ImageFont

    left = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    layer = left.copy()
    for x0, y0, x1, y1 in filled:
        cv2.rectangle(layer, (x0, y0), (x1, y1), COLOR_WORD_FILL, -1)
    cv2.addWeighted(layer, WORD_ALPHA, left, 1 - WORD_ALPHA, 0, left)
    right = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for x0, y0, x1, y1 in classic:
        cv2.rectangle(right, (x0, y0), (x1, y1), COLOR_CLASSIC, 2)
    for block in blocks:
        cv2.rectangle(right, (block["x0"] + 2, block["y0"] + 2), (block["x1"] - 2, block["y1"] - 2), COLOR_BLOCK, 1)
        cv2.putText(
            right, block["label"], (block["x0"] + 4, block["y0"] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BLOCK, 1
        )
    for obj in result["objects"]:
        x0, y0, x1, y1 = obj["box"]
        cv2.rectangle(right, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), COLOR_OBJECT, 3)
    for panel in (left, right):
        x0, y0, x1, y1 = row["crop_inner"]
        cv2.rectangle(panel, (x0, y0), (x1, y1), COLOR_FRAME, 1)
    scale = min(1.0, PANEL_MAX / max(left.shape[:2]))
    left, right = (cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for p in (left, right))
    body = np.hstack([left, np.full((left.shape[0], 12, 3), 255, np.uint8), right])
    if body.shape[1] < 900:
        body = np.hstack([body, np.full((body.shape[0], 900 - body.shape[1], 3), 255, np.uint8)])
    what = ", ".join(f"{o['class']} (рамка: {o['box_source']})" for o in result["objects"]) or "надпись"
    lines = [
        f"{row['page']}, область {row['id'].rsplit('_', 1)[-1]}   второй проход: {what}"
        + (f"   разметка: {label}" if label else ""),
        f"залито слов: {len(filled)};  блоки DeepSeek по залитой: {', '.join(b['label'] for b in blocks) or '—'};  находок классики: {len(classic)}",
        "слева — исходная вырезка и залитые слова, справа — залитая вырезка и что на ней нашлось",
    ]
    legend = [
        (
            (
                int(WORD_ALPHA * COLOR_WORD_FILL[0] + (1 - WORD_ALPHA) * 255),
                int(WORD_ALPHA * COLOR_WORD_FILL[1] + (1 - WORD_ALPHA) * 255),
                int(WORD_ALPHA * COLOR_WORD_FILL[2] + (1 - WORD_ALPHA) * 255),
            ),
            True,
            "залитое слово (рамка DeepSeek, достроенная по буквам)",
        ),
        (COLOR_BLOCK, False, "блок DeepSeek второго прохода с типом"),
        (COLOR_CLASSIC, False, "находка классического детектора line art"),
        (COLOR_OBJECT, False, "итоговый объект"),
        (COLOR_FRAME, False, "рамка области"),
    ]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    head = Image.new("RGB", (body.shape[1], 10 + 22 * len(lines)), (255, 255, 255))
    draw = ImageDraw.Draw(head)
    for i, line in enumerate(lines):
        draw.text((8, 5 + 22 * i), line, font=font, fill=(20, 20, 20))
    foot = Image.new("RGB", (body.shape[1], 10 + 20 * ((len(legend) + 1) // 2)), (255, 255, 255))
    draw = ImageDraw.Draw(foot)
    for i, (color, filled_sample, name) in enumerate(legend):
        x, y = 8 + (i % 2) * (body.shape[1] // 2), 5 + 20 * (i // 2)
        if filled_sample:
            draw.rectangle((x, y + 3, x + 26, y + 15), fill=color[::-1])
        else:
            draw.rectangle((x, y + 3, x + 26, y + 15), outline=color[::-1], width=2)
        draw.text((x + 34, y), name, font=font, fill=(20, 20, 20))
    to_bgr = lambda image: cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  # noqa: E731
    return np.vstack([to_bgr(head), body, to_bgr(foot)])
