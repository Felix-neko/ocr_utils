"""Сбор входных файлов и сохранение результата в нужном формате."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image as PILImage

# Поддерживаемые форматы входных изображений (без учёта регистра расширения)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    """Собирает изображения (по расширению, без учёта регистра).

    ``recursive=False`` — только верхний уровень каталога; ``True`` — рекурсивно.
    """
    it = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def imwrite_params(suffix: str) -> list[int]:
    """Параметры cv2.imwrite под формат (качество JPEG / сжатие PNG / сжатие TIFF)."""
    s = suffix.lower()
    if s in (".jpg", ".jpeg"):
        return [cv2.IMWRITE_JPEG_QUALITY, 95]
    if s == ".png":
        return [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if s in (".tif", ".tiff"):
        # LZW — сжатие БЕЗ потерь (в отличие от JPEG-in-TIFF); задаём явно, чтобы
        # не зависеть от дефолта cv2. Код 5 = COMPRESSION_LZW (libtiff).
        return [cv2.IMWRITE_TIFF_COMPRESSION, 5]
    return []


def write_image(out_path: Path, img: np.ndarray, params: list[int], force_dpi: Optional[int]) -> None:
    """Сохраняет изображение. Без ``force_dpi`` — быстрый ``cv2.imwrite``.

    cv2.imwrite не умеет прописывать разрешение (DPI). Раньше при ``force_dpi`` файл
    после cv2 перечитывался PIL и пересохранялся с тегом dpi — это ВТОРОЙ проход
    кодека (для TIFF-LZW на 30-48 Мп — несколько секунд впустую, всё в один поток).
    Теперь TIFF с DPI пишется ОДНИМ проходом через PIL (LZW без потерь + тег
    разрешения). PNG-ветка (DPI нужен редко) оставлена прежней двухпроходной.
    """
    if force_dpi is None:
        cv2.imwrite(str(out_path), img, params)
        return

    if out_path.suffix.lower() in (".tif", ".tiff"):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        PILImage.fromarray(rgb).save(str(out_path), format="TIFF", compression="tiff_lzw", dpi=(force_dpi, force_dpi))
        return

    cv2.imwrite(str(out_path), img, params)
    with PILImage.open(out_path) as im:
        im.save(out_path, dpi=(force_dpi, force_dpi))


def resolve_output_suffix(orig_suffix: str, output_format: Optional[str]) -> str:
    """Суффикс выходного файла: как у входа, если ``output_format`` не задан."""
    if output_format is None:
        return orig_suffix
    fmt = output_format.lower()
    if fmt == "png":
        return ".png"
    if fmt in ("jpg", "jpeg"):
        return ".jpg"
    return ".tiff"


def read_dpi(path: Path, default: Optional[int] = None) -> Optional[int]:
    """Разрешение исходного файла (точек на дюйм) или ``default``, если тега нет.

    Нужно, чтобы протащить DPI из входного файла в выходной: ``cv2.imwrite`` тег
    разрешения не пишет вовсе, а OCR-движки по нему судят о кегле — скан 600 dpi,
    сохранённый без тега, читается как страница в 96 dpi и распознаётся заметно хуже.
    Пишет это значение обратно ``write_image(..., force_dpi=...)``.
    """
    try:
        with PILImage.open(path) as img:
            dpi = img.info.get("dpi")
    except OSError:
        return default
    if not dpi:
        return default
    # PIL отдаёт пару (x, y), иногда как Rational — берём горизонтальное.
    value = int(round(float(dpi[0])))
    return value or default
