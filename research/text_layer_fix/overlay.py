"""Оверлеи страницы: рамки зон, таблиц, картинок FineReader и слов слоя по вердиктам.

Цвета (BGR для OpenCV): красный — DELETE, оранжевый — SANITIZE (выброшенные глифы —
красным внутри), жёлтый — SUSPECT, голубой — KEEP_ROTATED, зелёный — таблицы детектора,
синий — зоны повёрнутого текста, пурпурный — зоны с принятым чтением (вставка), рыжий —
картинки FineReader. Подпись зоны — прочитанный текст.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_utils.text_layer_fix.classify import Verdict

# Разрешение оверлея: четверть от 600 dpi, страница ~1000×1600 px — читается и весит немного.
OVERLAY_DPI = 150

COLORS = {
    Verdict.DELETE: (0, 0, 230),
    Verdict.SANITIZE: (0, 140, 255),
    Verdict.SUSPECT: (0, 210, 230),
    Verdict.KEEP_ROTATED: (230, 200, 0),
}
COLOR_TABLE = (0, 170, 0)
COLOR_ZONE = (230, 90, 0)
COLOR_INSERT = (200, 0, 200)
COLOR_FIGURE = (0, 120, 255)
COLOR_MISSING = (120, 0, 200)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _scaled(box, factor: float) -> tuple[int, int, int, int]:
    return tuple(int(round(v * factor)) for v in box)


def draw_overlay(gray: np.ndarray, payload: dict, out_path: Path, dpi_from: float, caption: str = "") -> None:
    """Нарисовать оверлей страницы по JSON кэша и сохранить PNG.

    Args:
        gray: Растр страницы в родном разрешении.
        payload: JSON результата (:meth:`pipeline.PageResult.to_json`).
        out_path: Куда сохранить PNG.
        dpi_from: Разрешение ``gray``.
        caption: Подпись в верхнем левом углу.
    """
    factor = OVERLAY_DPI / dpi_from
    small = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    for figure in payload.get("figures", []):
        x0, y0, x1, y1 = _scaled(figure, factor)
        cv2.rectangle(rgb, (x0, y0), (x1, y1), COLOR_FIGURE, 2)
    for table in payload.get("tables", []):
        x0, y0, x1, y1 = _scaled(table["box"], factor)
        cv2.rectangle(rgb, (x0, y0), (x1, y1), COLOR_TABLE, 2)
    readings = payload.get("readings", {})
    labels: list[tuple[int, int, str, tuple[int, int, int]]] = []
    for index, zone in enumerate(payload.get("zones", [])):
        x0, y0, x1, y1 = _scaled(zone["box"], factor)
        reading = readings.get(str(index))
        accepted = bool(reading and reading.get("accepted"))
        color = COLOR_INSERT if accepted else (COLOR_MISSING if zone["kind"].endswith("upright") else COLOR_ZONE)
        cv2.rectangle(rgb, (x0, y0), (x1, y1), color, 2)
        if reading and reading.get("text"):
            labels.append((x0, max(0, y0 - 12), reading["text"][:40] + ("" if accepted else " ✗"), color))
    for word in payload.get("words", []):
        verdict = Verdict(word["verdict"])
        if verdict == Verdict.KEEP:
            continue
        x0, y0, x1, y1 = _scaled(word["bbox_px"], factor)
        cv2.rectangle(rgb, (x0, y0), (x1, y1), COLORS[verdict], 1 if verdict == Verdict.SUSPECT else 2)
    image = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(FONT_PATH, 11)
    except OSError:
        font = ImageFont.load_default()
    for x, y, text, color in labels:
        draw.text((x, y), text, fill=(color[2], color[1], color[0]), font=font)
    if caption:
        draw.rectangle((0, 0, 8 + 6 * len(caption), 16), fill=(255, 255, 255))
        draw.text((4, 2), caption, fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, optimize=True)
