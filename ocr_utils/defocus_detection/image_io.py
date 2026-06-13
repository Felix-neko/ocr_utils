"""Ввод/вывод: загрузка превью из RAF/JPEG, сбор входных файлов, тепловые карты.

Полноразмерный RAW для оценки фокуса не нужен — внутри RAF лежит JPEG-превью
4416×2944, его разрешения с запасом хватает и извлекается оно мгновенно.
"""

import io
from pathlib import Path

import click
import cv2
import numpy as np
import rawpy
from PIL import Image, ImageOps

# Поддерживаемые расширения входных файлов
RAW_SUFFIXES = {".raf"}
JPEG_SUFFIXES = {".jpg", ".jpeg"}


def read_image_gray(path: Path) -> np.ndarray | None:
    """Загружает изображение в оттенках серого.

    Для RAF извлекает встроенное JPEG-превью (быстро), для JPEG грузит напрямую.
    Изображение поворачивается согласно EXIF.

    Args:
        path: Путь к входному файлу (RAF или JPEG).

    Returns:
        Полутоновый numpy.ndarray (uint8) либо None, если прочитать не удалось.
    """
    suffix = path.suffix.lower()
    try:
        if suffix in RAW_SUFFIXES:
            with rawpy.imread(str(path)) as raw:
                thumb = raw.extract_thumb()
                if thumb.format != rawpy.ThumbFormat.JPEG:
                    return None
                img = Image.open(io.BytesIO(bytes(thumb.data)))
        else:
            img = Image.open(str(path))
        img = ImageOps.exif_transpose(img)
        return np.array(img.convert("L"))
    except Exception:
        return None


def collect_inputs(inputs: tuple[str, ...]) -> list[Path]:
    """Разворачивает аргументы в список файлов.

    Директории обходятся нерекурсивно; берутся только поддерживаемые расширения.

    Args:
        inputs: Пути к файлам и/или директориям.

    Returns:
        Отсортированный список путей к файлам.
    """
    supported = RAW_SUFFIXES | JPEG_SUFFIXES
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(c for c in sorted(p.iterdir()) if c.suffix.lower() in supported)
        elif p.is_file():
            files.append(p)
        else:
            click.echo(f"Предупреждение: путь не найден — {p}", err=True)
    return files


def save_heatmap(path: Path, result: dict, vmax: float) -> None:
    """Сохраняет тепловую карту резкости тайлов (для отладки и подбора порога).

    Холодный цвет — мало резкости/муара (подозрение на расфокус), тёплый — резко.
    Серым показаны непечатные тайлы (поля), исключённые из статистики.

    Args:
        path: Куда сохранить PNG.
        result: Результат pipeline.analyze() с картами.
        vmax: Значение метрики, отображаемое как максимум шкалы.
    """
    sharp_map = result["sharp_map"]
    printed = result["printed_mask"]
    norm = np.clip(sharp_map / max(vmax, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    vis[~printed] = (60, 60, 60)
    scale = 24
    vis = cv2.resize(vis, (sharp_map.shape[1] * scale, sharp_map.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), vis)
