"""Подготовка полосы к отправке: уменьшить, обесцветить, сжать, закодировать.

Сканы лежат в 600 dpi RGB по 8-22 МБ; в таком виде их слать нельзя (base64 — ещё +33%,
а провайдеры режут запросы на 4-32 МБ) и незачем: модели всё равно приводят картинку к
своей сетке (Gemini — плитки 768 px, DeepSeek — ~1300 px по стороне). По умолчанию длинная
сторона 2200 px ≈ 216 dpi для полосы 10 дюймов: у петита 7-8 pt высота строчных ≈ 12-14 px,
у основного текста 9-10 pt ≈ 20 px — граница, ниже которой VLM начинают путать буквы.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

DEFAULT_MAX_SIDE = 2200
DEFAULT_QUALITY = 85
# Перекрытие соседних полос при нарезке, доля высоты страницы: строка текста в 600 dpi
# занимает ~80 px, 8% от 6000 — 480 px, то есть 5-6 строк — модель точно увидит место стыка.
STRIP_OVERLAP = 0.08


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    width: int
    height: int
    mime: str = "image/jpeg"

    def data_url(self) -> str:
        return f"data:{self.mime};base64,{base64.b64encode(self.data).decode('ascii')}"


def _encode(image: Image.Image, max_side: int, quality: int, grayscale: bool) -> PreparedImage:
    if grayscale and image.mode != "L":
        image = ImageOps.grayscale(image)
    scale = max_side / max(image.size)
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return PreparedImage(buffer.getvalue(), image.width, image.height)


def load(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image.load()
    return image


def prepare(
    path: Path,
    max_side: int = DEFAULT_MAX_SIDE,
    quality: int = DEFAULT_QUALITY,
    grayscale: bool = True,
    strips: int = 1,
    overlap: float = STRIP_OVERLAP,
) -> list[PreparedImage]:
    """Одна картинка целиком или ``strips`` горизонтальных полос сверху вниз с перекрытием.

    Полосы режутся по исходному разрешению и только потом уменьшаются до ``max_side`` — так
    каждая полоса несёт больше пикселей, чем досталось бы ей от целой страницы.
    """
    image = load(path)
    if strips <= 1:
        return [_encode(image, max_side, quality, grayscale)]
    height = image.height
    step = height / strips
    margin = int(height * overlap / 2)
    pieces: list[PreparedImage] = []
    for index in range(strips):
        top = max(0, int(index * step) - margin)
        bottom = min(height, int((index + 1) * step) + margin)
        pieces.append(_encode(image.crop((0, top, image.width, bottom)), max_side, quality, grayscale))
    return pieces
