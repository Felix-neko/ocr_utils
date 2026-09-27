"""Оверлей одной области line art: из чего правило «надпись» сложило вердикт — пятна, слова, три числа с порогами."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from research.line_art_titles.features import FEW_BLOBS, ink_blobs

# Цвета (BGR) — палитра проекта (навык draw-overlay): принято / отвергнуто, подсказка, ось над меткой.
COLOR_LETTER = (0, 150, 0)  # буквоподобное пятно — идёт в долю букв
COLOR_OTHER = (0, 0, 220)  # не буква — тянет долю букв вниз
COLOR_LARGEST = (200, 140, 60)  # крупнейшее пятно — правило его не считает
COLOR_WORD = (0, 165, 255)  # слово движка распознавания (tesseract из букв / DeepSeek — все)
COLOR_BLOCK = (200, 60, 200)  # блок вёрстки DeepSeek-OCR-2 (промпт markdown) с его типом
COLOR_FRAME = (170, 170, 170)  # рамка области
COLOR_TEXT = (20, 20, 20)
# Вырезка, поданная в OCR целиком (tesseract и DeepSeek видят всю картинку): рамка вокруг неё и
# полупрозрачная заливка поля вне рамки области — поле даёт OCR контекст, но в признаки краски не идёт.
COLOR_CROP = (60, 60, 60)
COLOR_PAD = (160, 120, 60)
PAD_ALPHA = 0.18

# Прозрачность окраски краски: буквы под ней должны читаться.
BLOB_ALPHA = 0.55

# Размер картинки области: узкая вырезка растягивается до этой ширины (не больше чем в 4 раза),
# крупная ужимается до этой длинной стороны.
MIN_WIDTH = 900
MAX_UPSCALE = 4.0
MAX_SIDE = 1600

# Шрифт подписей с кириллицей.
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_SIZE = 17
LINE_HEIGHT = 24


def _font(size: int = FONT_SIZE):
    """Шрифт DejaVu Sans нужного кегля (или встроенный PIL, если DejaVu нет)."""
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _on_paper(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    """Цвет полупрозрачной окраски на белой бумаге — для образца в легенде."""
    return tuple(int(round(alpha * c + (1 - alpha) * 255)) for c in color)


def _text_strip(lines: list[tuple[str, tuple[int, int, int]]], width: int) -> np.ndarray:
    """Белая полоса с подписями: по строке на пару (текст, цвет BGR)."""
    strip = Image.new("RGB", (width, 10 + LINE_HEIGHT * len(lines)), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    for index, (text, color) in enumerate(lines):
        draw.text((10, 6 + index * LINE_HEIGHT), text, font=_font(), fill=color[::-1])
    return cv2.cvtColor(np.asarray(strip), cv2.COLOR_RGB2BGR)


def _legend(width: int, engine: str, with_blocks: bool) -> np.ndarray:
    """Легенда; образцы окраски краски — с той же прозрачностью, что на картинке."""
    word_label = (
        "слово tesseract из букв, любая уверенность" if engine == "tesseract" else f"слово {engine} (все найденные)"
    )
    items = [
        (_on_paper(COLOR_LETTER, BLOB_ALPHA), True, "буквоподобное пятно"),
        (_on_paper(COLOR_OTHER, BLOB_ALPHA), True, "не буква"),
        (_on_paper(COLOR_LARGEST, BLOB_ALPHA), True, "крупнейшее пятно (правило не считает)"),
        (COLOR_WORD, False, word_label),
        (COLOR_FRAME, False, "рамка области line art"),
        (_on_paper(COLOR_PAD, PAD_ALPHA), True, "поле: видит OCR, в признаки краски не идёт"),
        (COLOR_CROP, False, "вся вырезка, поданная в OCR"),
    ]
    if with_blocks:
        items.append((COLOR_BLOCK, False, f"блок вёрстки {engine} с типом"))
    strip = Image.new("RGB", (width, 12 + LINE_HEIGHT * ((len(items) + 1) // 2)), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    column = width // 2
    for index, (color, filled, label) in enumerate(items):
        x = 10 + (index % 2) * column
        y = 8 + (index // 2) * LINE_HEIGHT
        box = (x, y + 2, x + 26, y + 16)
        if filled:
            draw.rectangle(box, fill=color[::-1])
        else:
            draw.rectangle(box, outline=color[::-1], width=2)
        draw.text((x + 34, y), label, font=_font(15), fill=COLOR_TEXT[::-1])
    return cv2.cvtColor(np.asarray(strip), cv2.COLOR_RGB2BGR)


def _mark(passed: bool) -> str:
    """Галочка или крестик для условия правила."""
    return "✓" if passed else "✗"


def draw_region(
    gray: np.ndarray,
    row: dict,
    features: dict,
    words: list[dict],
    dpi: int,
    label: str | None,
    title: bool,
    conditions: list[tuple[str, bool]],
    engine: str,
    blocks: list[dict] = (),
    notes: list[str] = (),
) -> np.ndarray:
    """Картинка одной области: окрашенная краска, слова и блоки движка, шапка с вердиктом и числами правила, легенда.

    Args:
        gray: Серая вырезка области (с полем).
        row: Строка ``regions.jsonl``.
        features: Признаки области (строка ``features.csv`` или ``features_deepseek.csv``).
        words: Рамки слов движка распознавания в пикселях вырезки (``x0, y0, x1, y1``).
        dpi: Разрешение вырезки.
        label: Класс ручной разметки, если область размечена, иначе ``None``.
        title: Вердикт правила «надпись».
        conditions: Условия правила — (подпись с числом и порогом, выполнено ли); в шапке ✓/✗.
        engine: Имя движка для легенды и шапки.
        blocks: Блоки вёрстки DeepSeek-OCR-2 (``label`` — тип блока); пусто у tesseract.
        notes: Справочные строки шапки (числа, которые в правило не входят).

    Returns:
        Картинка BGR.
    """
    blobs = ink_blobs(gray, row["crop_inner"], dpi)
    few = blobs.areas.size <= FEW_BLOBS

    # Окраска краски по классу пятна: буква / не буква / крупнейшее (при ≤ 3 пятнах крупнейшее
    # не выделяется — правило тогда считает все пятна).
    kind = np.zeros(blobs.areas.size + 1, np.uint8)  # 0 — бумага, 1 — буква, 2 — не буква, 3 — крупнейшее
    kind[1:] = np.where(blobs.letter_like, 1, 2)
    if not few and blobs.largest >= 0:
        kind[blobs.largest + 1] = 3
    per_pixel = kind[blobs.labels]
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    layer = canvas.copy()
    for value, color in ((1, COLOR_LETTER), (2, COLOR_OTHER), (3, COLOR_LARGEST)):
        layer[per_pixel == value] = color
    cv2.addWeighted(layer, BLOB_ALPHA, canvas, 1 - BLOB_ALPHA, 0, canvas)

    # Поле вокруг области (вне рамки) — полупрозрачная заливка: OCR его видит, признаки краски — нет.
    x0, y0, x1, y1 = row["crop_inner"]
    tinted = canvas.copy()
    tinted[:] = COLOR_PAD
    blended = cv2.addWeighted(tinted, PAD_ALPHA, canvas, 1 - PAD_ALPHA, 0)
    outside = np.ones(canvas.shape[:2], bool)
    outside[y0:y1, x0:x1] = False
    canvas[outside] = blended[outside]
    crop_mm = (gray.shape[1] / dpi * 25.4, gray.shape[0] / dpi * 25.4)
    pads_mm = (x0 / dpi * 25.4, (gray.shape[1] - x1) / dpi * 25.4, y0 / dpi * 25.4, (gray.shape[0] - y1) / dpi * 25.4)

    # Масштаб до читаемого размера; рамки рисуются уже на масштабированной картинке, чтобы
    # толщина линий не зависела от размера области.
    height, width = canvas.shape[:2]
    scale = min(MAX_UPSCALE, max(1.0, MIN_WIDTH / max(1, width)))
    scale = min(scale, MAX_SIDE / max(height, width)) if max(height, width) * scale > MAX_SIDE else scale
    canvas = cv2.resize(
        canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    )
    for word in words:
        corner0 = (int(word["x0"] * scale), int(word["y0"] * scale))
        corner1 = (int(word["x1"] * scale), int(word["y1"] * scale))
        cv2.rectangle(canvas, corner0, corner1, COLOR_WORD, 2)
    for block in blocks:
        corner0 = (int(block["x0"] * scale) + 3, int(block["y0"] * scale) + 3)
        corner1 = (int(block["x1"] * scale) - 3, int(block["y1"] * scale) - 3)
        cv2.rectangle(canvas, corner0, corner1, COLOR_BLOCK, 2)
        cv2.putText(
            canvas, block["label"], (corner0[0] + 4, corner0[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLOCK, 2
        )
    x0, y0, x1, y1 = (int(v * scale) for v in row["crop_inner"])
    cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOR_FRAME, 2)
    cv2.rectangle(canvas, (1, 1), (canvas.shape[1] - 2, canvas.shape[0] - 2), COLOR_CROP, 3)

    # Шапка: вердикт, условия правила с числами и порогами, справочные строки.
    verdict = "НАДПИСЬ" if title else "НЕ НАДПИСЬ"
    manual = f"   ручная разметка: {label}" if label else ""
    lines = [(f"{verdict} ({engine})   {row['page']}, область {row['id'].rsplit('_', 1)[-1]}{manual}", COLOR_TEXT)]
    lines += [(f"{_mark(passed)} {text}", COLOR_LETTER if passed else COLOR_OTHER) for text, passed in conditions]
    lines += [(note, COLOR_TEXT) for note in notes]
    lines.append(
        (
            f"в OCR подана вся вырезка {crop_mm[0]:.0f}×{crop_mm[1]:.0f} мм: область + поле "
            f"слева {pads_mm[0]:.0f}, справа {pads_mm[1]:.0f}, сверху {pads_mm[2]:.0f}, снизу {pads_mm[3]:.0f} мм",
            COLOR_TEXT,
        )
    )
    width = max(canvas.shape[1], MIN_WIDTH)
    if canvas.shape[1] < width:
        pad = np.full((canvas.shape[0], width - canvas.shape[1], 3), 255, np.uint8)
        canvas = np.hstack([canvas, pad])
    image = np.vstack([_text_strip(lines, width), canvas, _legend(width, engine, bool(blocks))])
    return image


def overlay_path(out_dir: Path, row: dict, title: bool) -> Path:
    """Куда класть картинку области: ``<out_dir>/надпись|не_надпись/<год>_<выпуск>_<полоса>_<номер>.jpg``.

    Args:
        out_dir: Корень оверлеев движка.
        row: Строка ``regions.jsonl``.
        title: Вердикт правила «надпись».

    Returns:
        Путь файла картинки.
    """
    folder = "надпись" if title else "не_надпись"
    return out_dir / folder / f"{row['id']}.jpg"
