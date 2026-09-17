"""Нарезка полосы на тайлы по сетке в пикселях исходника и подготовка каждого к отправке.

Зачем тайлы. DeepSeek ужимает любую картинку до ~1300 px по стороне (1024 токенов), и целая
полоса 6000 px превращается в ≈128 dpi — на плотных полосах модель теряет текст. Стенд
(``research/external_ocr_models``) резал полосу на N горизонтальных кусков; здесь вместо N —
сетка по ``max_src_tile``: шаг сетки в пикселях ИСХОДНИКА, число столбцов и строк —
``ceil(сторона / шаг)``. Так обычная полоса пака-1 (3000–4500 × 5000–6700 при 600 dpi) даёт 1×2,
полоса, повёрнутая на 90°, — 2×1, а склеенный разворот (6444×5336) — 2×2, и всё это одним
параметром, без знания о том, что за кадр перед нами.

Перекрытие соседей — константа ``TILE_OVERLAP``, доля стороны кадра вдоль оси, добавляется
СВЕРХ шага (тайл чуть больше ``max_src_tile``): при 8 % от 6000 px это 480 px, пять-шесть строк
текста — модель точно увидит место стыка. Порядок тайлов — по столбцам: сверху вниз внутри
столбца, затем следующий столбец; это порядок чтения полосы с двумя страницами разворота.

Каждый тайл режется в исходном разрешении и только потом уменьшается до ``max_model_tile`` по
длинной стороне (2200 px ≈ 216 dpi для полосы 10″: строчные основного текста ≈ 20 px, петит
≈ 13 px — ниже VLM начинают путать буквы), обесцвечивается и сжимается в JPEG.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

# Шаг сетки в пикселях исходника — для пака-1 при 600 dpi (замер по базе: ширина полос
# 2998–4499, высота 4961–6692, разворот 6444×5336).
DEFAULT_MAX_SRC_TILE = 4500
# Длинная сторона тайла после уменьшения, px.
DEFAULT_MAX_MODEL_TILE = 2200
DEFAULT_QUALITY = 85
# Перекрытие соседних тайлов — доля стороны кадра вдоль оси (делится пополам между соседями).
TILE_OVERLAP = 0.08


@dataclass(frozen=True)
class TileBox:
    """Один тайл сетки: позиция в сетке и прямоугольник в пикселях исходника."""

    col: int
    row: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def size(self) -> tuple[int, int]:
        return self.right - self.left, self.bottom - self.top


@dataclass(frozen=True)
class PreparedImage:
    """Готовый к отправке тайл: JPEG-байты, размер после уменьшения и его место в сетке."""

    data: bytes
    width: int
    height: int
    box: TileBox
    mime: str = "image/jpeg"

    def data_url(self) -> str:
        return f"data:{self.mime};base64,{base64.b64encode(self.data).decode('ascii')}"


def grid_shape(width: int, height: int, max_src_tile: int) -> tuple[int, int]:
    """Число (столбцов, строк) сетки: ``ceil(сторона / шаг)``, не меньше одного."""
    if max_src_tile <= 0:
        raise ValueError("шаг сетки должен быть положительным")
    return max(1, math.ceil(width / max_src_tile)), max(1, math.ceil(height / max_src_tile))


def grid(width: int, height: int, max_src_tile: int, overlap: float = TILE_OVERLAP) -> list[TileBox]:
    """Тайлы сетки в порядке чтения: по столбцам, внутри столбца сверху вниз.

    Перекрытие добавляется только там, где есть сосед (по оси с одним тайлом его нет).
    """
    ncols, nrows = grid_shape(width, height, max_src_tile)
    margin_x = int(width * overlap / 2) if ncols > 1 else 0
    margin_y = int(height * overlap / 2) if nrows > 1 else 0
    boxes: list[TileBox] = []
    for col in range(ncols):
        left = max(0, int(col * width / ncols) - margin_x)
        right = min(width, int((col + 1) * width / ncols) + margin_x)
        for row in range(nrows):
            top = max(0, int(row * height / nrows) - margin_y)
            bottom = min(height, int((row + 1) * height / nrows) + margin_y)
            boxes.append(TileBox(col, row, left, top, right, bottom))
    return boxes


def load(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image.load()
    return image


def encode(
    image: Image.Image, box: TileBox, max_model_tile: int, quality: int, grayscale: bool = True
) -> PreparedImage:
    """Уменьшить (только вниз), обесцветить и сжать вырезанный тайл."""
    if grayscale and image.mode != "L":
        image = ImageOps.grayscale(image)
    scale = max_model_tile / max(image.size)
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return PreparedImage(buffer.getvalue(), image.width, image.height, box)


def prepare_tiles(
    path: Path,
    max_src_tile: int = DEFAULT_MAX_SRC_TILE,
    max_model_tile: int = DEFAULT_MAX_MODEL_TILE,
    quality: int = DEFAULT_QUALITY,
    grayscale: bool = True,
    overlap: float = TILE_OVERLAP,
) -> list[PreparedImage]:
    """Полоса с диска -> тайлы к отправке, в порядке чтения."""
    image = load(path)
    boxes = grid(image.width, image.height, max_src_tile, overlap)
    if len(boxes) == 1:
        return [encode(image, boxes[0], max_model_tile, quality, grayscale)]
    return [
        encode(image.crop((box.left, box.top, box.right, box.bottom)), box, max_model_tile, quality, grayscale)
        for box in boxes
    ]


def describe(tiles: list[PreparedImage]) -> dict:
    """Сводка сетки для .meta.json: размеры, число столбцов и строк, прямоугольники."""
    ncols = 1 + max(tile.box.col for tile in tiles)
    nrows = 1 + max(tile.box.row for tile in tiles)
    return {
        "ncols": ncols,
        "nrows": nrows,
        "tiles": [
            {
                "col": tile.box.col,
                "row": tile.box.row,
                "src": [tile.box.left, tile.box.top, tile.box.right, tile.box.bottom],
                "sent_px": [tile.width, tile.height],
                "bytes": len(tile.data),
            }
            for tile in tiles
        ],
    }
