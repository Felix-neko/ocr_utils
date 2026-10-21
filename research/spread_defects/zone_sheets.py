"""Лист отсмотра карты смаза: весь кадр с зонами поверх и крупы 1:1 у максимума каждой зоны, шапка и легенда в полях."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from research.spread_defects.frames import read_rgb
from research.spread_defects.motion import TILE

# Цвета зон (BGR, палитра навыка draw-overlay): ореол — оранжевый, двоение по кепстру — синий, оба — фиолетовый.
COLOUR_MIDTONE = (0, 165, 255)
COLOUR_CEPSTRUM = (220, 90, 20)
COLOUR_BOTH = (200, 60, 200)
# Сгиб — серый, как межколонник.
COLOUR_FOLD = (170, 170, 170)
ZONE_ALPHA = 0.35
# Ширина обзорного кадра на листе, px; круп — 768 px полного кадра в ячейке 600 px (двоение 4–5 px ещё видно).
OVERVIEW_WIDTH = 1800
CROP_SIDE, CROP_CELL = 768, 600
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def zone_colour(zone: dict) -> tuple[int, int, int]:
    """Цвет зоны по источнику: ореол, кепстр или оба.

    Args:
        zone: Строка ``zones.csv`` (поля ``by_midtone``, ``by_cepstrum``).

    Returns:
        Цвет BGR.
    """
    if zone["by_midtone"] and zone["by_cepstrum"]:
        return COLOUR_BOTH
    return COLOUR_CEPSTRUM if zone["by_cepstrum"] else COLOUR_MIDTONE


def put_text(image: np.ndarray, text: str, origin: tuple[int, int], size: int) -> np.ndarray:
    """Подпись кириллицей на белой подложке (PIL: векторный шрифт OpenCV кириллицу не знает).

    Args:
        image: Картинка BGR.
        text: Подпись.
        origin: Левый верхний угол, px.
        size: Кегль, px.

    Returns:
        Новая картинка BGR с подписью.
    """
    picture = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(picture)
    font = ImageFont.truetype(FONT, size)
    box = draw.textbbox(origin, text, font=font)
    draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(255, 255, 255))
    draw.text(origin, text, font=font, fill=(20, 20, 20))
    return cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR)


def overview(bgr: np.ndarray, zones: list[dict], fold_x: float) -> np.ndarray:
    """Весь кадр, ужатый до ``OVERVIEW_WIDTH``, с полупрозрачными зонами, их номерами и сгибом.

    Args:
        bgr: Кадр полного разрешения BGR.
        zones: Строки ``zones.csv`` кадра.
        fold_x: x сгиба, px полного кадра (NaN — не рисовать).

    Returns:
        Обзор BGR.
    """
    scale = OVERVIEW_WIDTH / bgr.shape[1]
    canvas = cv2.resize(bgr, (OVERVIEW_WIDTH, int(bgr.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    layer = canvas.copy()
    for zone in zones:
        # Заливка зоны поплиточно не нужна: прямоугольник охвата читается лучше и не дробит текст.
        x0, y0, x1, y1 = (int(v * scale) for v in (zone["x0"], zone["y0"], zone["x1"], zone["y1"]))
        cv2.rectangle(layer, (x0, y0), (x1, y1), zone_colour(zone), -1)
    cv2.addWeighted(layer, ZONE_ALPHA, canvas, 1 - ZONE_ALPHA, 0, canvas)
    for zone in zones:
        x0, y0, x1, y1 = (int(v * scale) for v in (zone["x0"], zone["y0"], zone["x1"], zone["y1"]))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), zone_colour(zone), 2)
    if np.isfinite(fold_x):
        x = int(fold_x * scale)
        cv2.line(canvas, (x, 0), (x, canvas.shape[0] - 1), COLOUR_FOLD, 1)
    for zone in zones:
        x0, y0 = int(zone["x0"] * scale), int(zone["y0"] * scale)
        canvas = put_text(canvas, f"{zone['zone']}: {zone['mid_peak']:.2f}", (x0 + 4, y0 + 4), 22)
    return canvas


def zone_crop(bgr: np.ndarray, zone: dict) -> np.ndarray:
    """Круп ``CROP_SIDE`` px полного кадра вокруг плитки максимума ореола зоны, в ячейке ``CROP_CELL``.

    Args:
        bgr: Кадр полного разрешения BGR.
        zone: Строка ``zones.csv``.

    Returns:
        Ячейка BGR с подписью зоны снизу (в поле, не поверх текста).
    """
    cx, cy = int((zone["peak_col"] + 0.5) * TILE), int((zone["peak_row"] + 0.5) * TILE)
    x0 = min(max(cx - CROP_SIDE // 2, 0), max(bgr.shape[1] - CROP_SIDE, 0))
    y0 = min(max(cy - CROP_SIDE // 2, 0), max(bgr.shape[0] - CROP_SIDE, 0))
    part = cv2.resize(
        bgr[y0 : y0 + CROP_SIDE, x0 : x0 + CROP_SIDE], (CROP_CELL, CROP_CELL), interpolation=cv2.INTER_AREA
    )
    caption = np.full((56, CROP_CELL, 3), 255, np.uint8)
    source = "+".join(name for name, flag in (("ореол", zone["by_midtone"]), ("кепстр", zone["by_cepstrum"])) if flag)
    # Две строки: в одну подпись в ячейку 600 px не помещается.
    caption = put_text(caption, f"зона {zone['zone']}: {zone['side']} {zone['band']}, плиток {zone['tiles']}, {source}", (4, 4), 16)
    caption = put_text(caption, f"ореол {zone['mid_peak']:.2f}, кепстр {zone['agree_peak']:.1f}", (4, 30), 16)
    # Рамка цвета зоны — чтобы круп связывался с прямоугольником на обзоре.
    cv2.rectangle(part, (0, 0), (CROP_CELL - 1, CROP_CELL - 1), zone_colour(zone), 4)
    return np.vstack([part, caption])


def render_zone_sheet(frame: dict, zones: list[dict], fold_x: float, target: Path, label: str | None = None) -> str:
    """Рисует и пишет лист отсмотра одного кадра (воркер пула).

    Args:
        frame: Строка ``zone_frames.csv`` (ключ, путь, ``mid_peak``, ``frame_rel``, число зон).
        zones: Строки ``zones.csv`` кадра (сильнейшие).
        fold_x: x сгиба, px полного кадра.
        target: Файл листа.
        label: Вердикт отсмотра для шапки (например, «тяжёлый смаз»); None — без него.

    Returns:
        Путь записанного листа строкой.
    """
    bgr = cv2.cvtColor(read_rgb(Path(frame["path"])), cv2.COLOR_RGB2BGR)
    top = overview(bgr, zones, fold_x)
    # Крупы в ряд под обзором, добитые белым до ширины обзора.
    crops = [zone_crop(bgr, zone) for zone in zones]
    rows = []
    per_row = OVERVIEW_WIDTH // CROP_CELL
    for n in range(0, len(crops), per_row):
        row = np.hstack(crops[n : n + per_row])
        pad = np.full((row.shape[0], OVERVIEW_WIDTH - row.shape[1], 3), 255, np.uint8)
        rows.append(np.hstack([row, pad]))
    canvas = np.vstack([top, *rows]) if rows else top
    header = [
        f"{frame['key']} — карта смаза: зон {frame['zones']}, сильнейший ореол {frame['mid_peak']:.2f} "
        f"(к медиане выпуска), ореол всего кадра {frame['frame_rel']:.2f}"
    ]
    if label:
        header = [f"{frame['key'][:7]} {Path(frame['path']).stem} — {label}", *header]
    legend = [
        LegendEntry("зона по полутоновому ореолу", COLOUR_MIDTONE, alpha=ZONE_ALPHA, style=SampleStyle.BOX),
        LegendEntry("зона по двоению штриха (кепстр)", COLOUR_CEPSTRUM, alpha=ZONE_ALPHA, style=SampleStyle.BOX),
        LegendEntry("зона по обоим признакам", COLOUR_BOTH, alpha=ZONE_ALPHA, style=SampleStyle.BOX),
        LegendEntry("сгиб", COLOUR_FOLD),
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), framed(canvas, header, legend), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(target)
