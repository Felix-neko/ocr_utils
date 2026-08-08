"""Чтение изображений и сбор входных файлов.

Поддерживаются RAF (берётся встроенное JPEG-превью 4416×2944 — полноразмерный RAW для
оценки фокуса не нужен, а распаковывается он в сотни раз дольше), а также TIFF, PNG, JPG.
"""

import io
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image, ImageOps

# Pillow по умолчанию отказывается открывать очень большие файлы (защита от decompression bomb).
# Сканы газетных полос легко перешагивают лимит, а источник у нас доверенный.
Image.MAX_IMAGE_PIXELS = None

RAW_SUFFIXES = {".raf"}
PLAIN_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SUPPORTED_SUFFIXES = RAW_SUFFIXES | PLAIN_SUFFIXES


def _to_uint8_gray(img: Image.Image) -> np.ndarray:
    """Приводит картинку Pillow к 8-битному полутоновому массиву.

    16-битные TIFF масштабируются по фактическому диапазону (а не делением на 257):
    сканы часто не используют полную шкалу, и наивное деление съело бы контраст,
    от которого зависят почти все метрики резкости.

    Args:
        img: Изображение Pillow в любом режиме.

    Returns:
        Полутоновый массив uint8.
    """
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img).astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        return ((arr - lo) * (255.0 / (hi - lo))).astype(np.uint8)
    return np.asarray(img.convert("L"))


def read_gray(path: Path) -> np.ndarray | None:
    """Загружает изображение в оттенках серого с учётом EXIF-поворота.

    Args:
        path: Путь к файлу (RAF / TIFF / PNG / JPG).

    Returns:
        Полутоновый массив uint8 либо None, если файл не читается или в RAF нет превью.
    """
    suffix = path.suffix.lower()
    try:
        if suffix in RAW_SUFFIXES:
            with rawpy.imread(str(path)) as raw:
                thumb = raw.extract_thumb()
                if thumb.format != rawpy.ThumbFormat.JPEG:
                    return None
                img = Image.open(io.BytesIO(bytes(thumb.data)))
                img = ImageOps.exif_transpose(img)
                return _to_uint8_gray(img)
        with Image.open(str(path)) as img:
            img = ImageOps.exif_transpose(img)
            return _to_uint8_gray(img)
    except Exception:
        return None


def collect_images(folder: Path, recursive: bool = False) -> list[Path]:
    """Собирает поддерживаемые изображения в папке.

    Args:
        folder: Папка со сканами (либо путь к одиночному файлу).
        recursive: Обходить ли вложенные папки.

    Returns:
        Отсортированный по имени список путей.
    """
    if folder.is_file():
        return [folder]
    pattern = "**/*" if recursive else "*"
    files = [p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files)
