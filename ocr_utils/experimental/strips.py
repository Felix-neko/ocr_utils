"""Полоса → горизонтальные полосы 1×N с перекрытием: каждая картинка укладывается в потолок площади модели без ужатия.

Зачем. У DeepSeek V4.1 Flash потолок ≈ 800 токенов на картинку достигается при площади
≈ 1.8 Мпкс (замер: 2100–2300 px² на токен, сессия 65786440); всё крупнее провайдер ужимает
сам. Сетка `tiling.grid` режет полосу пака-1 (3450×6000) на 1×2, и каждый тайл уходит как
2200×2230 ≈ 4.9 Мпкс — то есть модель видит его ужатым до ≈ 1330 px по ширине (≈ 131 dpi,
строчные буквы ≈ 12 px). Шаг сетки там общий по обеим осям, поэтому «1×3» через опции не
получить — уменьшение шага добавляет второй столбец. Здесь столбец всегда один, число полос
задаётся прямо; при пяти полосах полоса 1966/03 уходит как 2200×777 ≈ 1.7 Мпкс — без ужатия.

Формат результата тот же ``PreparedImage`` с ``TileBox(col=0, row=i)``, так что ``describe`` и
промпт («N tiles: 1 column × N rows») работают без изменений. Стенд подменяет
``ocr.prepare_tiles`` на :func:`prepare_strips` (``scripts/replay_page.py --rows``).
"""

from __future__ import annotations

from pathlib import Path

from ocr_utils.external_ocr_services.tiling import (
    DEFAULT_MAX_MODEL_TILE,
    DEFAULT_QUALITY,
    PreparedImage,
    TileBox,
    encode,
    load,
)

# Перекрытие соседних полос — доля ВЫСОТЫ ПОЛОСЫ (не кадра, как в ``tiling.TILE_OVERLAP``: 8 % от
# 6000 px — это 480 px, при пяти полосах почти половина каждой). 15 % от полосы в 1200 px — 180 px,
# две-три строки текста, чтобы строка на стыке попала целиком хотя бы в одну полосу.
STRIP_OVERLAP = 0.15
# Потолок площади картинки у DeepSeek V4.1 Flash: ≈ 800 токенов, дальше картинка ужимается (замер сессии 65786440).
DEEPSEEK_MAX_AREA = 1.8e6


def strip_boxes(width: int, height: int, rows: int, overlap: float = STRIP_OVERLAP) -> list[TileBox]:
    """Прямоугольники N горизонтальных полос кадра с перекрытием соседей.

    Перекрытие — доля высоты ПОЛОСЫ, поделённая пополам между соседями; у крайних полос только
    внутренняя сторона.

    Args:
        width: Ширина кадра в пикселях исходника.
        height: Высота кадра.
        rows: Сколько полос; 1 — весь кадр целиком.
        overlap: Перекрытие соседей, доля высоты полосы.

    Returns:
        ``TileBox`` сверху вниз, все в столбце 0; первая начинается с 0, последняя кончается на ``height``.
    """
    rows = max(1, rows)
    margin = int(height / rows * overlap / 2) if rows > 1 else 0
    boxes = []
    for row in range(rows):
        top = max(0, int(row * height / rows) - margin)
        bottom = min(height, int((row + 1) * height / rows) + margin)
        boxes.append(TileBox(col=0, row=row, left=0, top=top, right=width, bottom=bottom))
    return boxes


def prepare_strips(
    path: Path,
    rows: int,
    max_model_tile: int = DEFAULT_MAX_MODEL_TILE,
    quality: int = DEFAULT_QUALITY,
    grayscale: bool = True,
    overlap: float = STRIP_OVERLAP,
) -> list[PreparedImage]:
    """Полоса с диска → N горизонтальных полос к отправке, сверху вниз.

    Args:
        path: Файл полосы.
        rows: Сколько полос (см. докстринг модуля: 5 для полос пака-1 при 2200 px — без ужатия).
        max_model_tile: Длинная сторона каждой полосы после уменьшения, px (у горизонтальной
            полосы это её ширина).
        quality: Качество JPEG.
        grayscale: Обесцвечивать ли.
        overlap: Перекрытие соседей, доля высоты полосы.

    Returns:
        Готовые к отправке картинки в порядке чтения (столбец один, строки сверху вниз).
    """
    image = load(path)
    boxes = strip_boxes(image.width, image.height, rows, overlap)
    if len(boxes) == 1:
        return [encode(image, boxes[0], max_model_tile, quality, grayscale)]
    return [
        encode(image.crop((box.left, box.top, box.right, box.bottom)), box, max_model_tile, quality, grayscale)
        for box in boxes
    ]


def rows_for_area(
    width: int, height: int, max_model_tile: int, max_area: float = DEEPSEEK_MAX_AREA, overlap: float = STRIP_OVERLAP
) -> int:
    """Сколько полос нужно, чтобы каждая (с перекрытием) после уменьшения до ``max_model_tile`` по ширине не превышала ``max_area``.

    Args:
        width: Ширина полосы в пикселях исходника.
        height: Высота полосы.
        max_model_tile: Длинная сторона после уменьшения (ширина горизонтальной полосы).
        max_area: Потолок площади картинки у модели, px².
        overlap: Перекрытие соседей, доля высоты полосы (добавляется к высоте каждой полосы).

    Returns:
        Наименьшее N ≥ 1, при котором ``max_model_tile × высота полосы с перекрытием после масштаба ≤ max_area``.
    """
    scale = min(1.0, max_model_tile / width)
    sent_width = width * scale
    rows = 1
    while True:
        strip_height = height / rows * (1.0 + (overlap if rows > 1 else 0.0)) * scale
        if sent_width * strip_height <= max_area:
            return rows
        rows += 1
