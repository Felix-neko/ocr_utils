"""Копия дерева кадров с пересчётом разрешения (300 → 450 dpi): стенды с порогами в px пака-4 работают в тех же мм."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import cv2
import numpy as np
import threadpoolctl
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Расширения кадров, которые пересчитываются (остальные файлы — Thumbs.db и т. п. — пропускаются).
FRAME_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff"}


def _init_worker() -> None:
    """Инициализатор воркера: без hugepage и с одним потоком BLAS/OpenCV (правило пулов проекта)."""
    np._core.multiarray._set_madvise_hugepage(False)
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def rescale_one(task: tuple[Path, Path, float, int]) -> str:
    """Пересчитать один кадр и записать JPEG.

    Args:
        task: Кортеж (исходный файл, файл назначения, множитель размера, качество JPEG 1–100).

    Returns:
        Путь записанного файла строкой; пустая строка, если файл уже был и пересчёт пропущен.
    """
    source, target, factor, quality = task
    # Уже пересчитанный кадр не трогаем: прогон можно перезапускать после обрыва.
    if target.exists():
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        size = (round(image.width * factor), round(image.height * factor))
        resized = image.resize(size, Image.Resampling.LANCZOS)
    # Запись через временный файл: оборванный воркер не оставит битый кадр под окончательным именем.
    partial = target.with_suffix(".part.jpg")
    resized.save(partial, quality=quality, subsampling=0)
    partial.rename(target)
    return str(target)


@click.command()
@click.option(
    "--src",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Корень исходного дерева (только чтение).",
)
@click.option(
    "--dst",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Корень копии; структура папок повторяет исходную.",
)
@click.option("--src-dpi", default=300.0, show_default=True, help="Разрешение исходных кадров.")
@click.option("--dst-dpi", default=450.0, show_default=True, help="Целевое разрешение копии.")
@click.option("--quality", default=95, show_default=True, help="Качество JPEG копии.")
@click.option("--jobs", default=12, show_default=True, help="Воркеров (исходники на NTFS-3G — узкое место).")
def main(src: Path, dst: Path, src_dpi: float, dst_dpi: float, quality: int, jobs: int) -> None:
    """Пересчитать все кадры дерева ``src`` в ``dst`` с множителем ``dst_dpi / src_dpi``.

    Args:
        src: Корень исходного дерева.
        dst: Корень копии.
        src_dpi: Разрешение исходников.
        dst_dpi: Целевое разрешение.
        quality: Качество JPEG.
        jobs: Число процессов пула.
    """
    if dst.resolve().is_relative_to(Path("/mnt/dump3")):
        raise click.UsageError("копию нельзя писать на /mnt/dump3 (Я.Диск затирает исходники)")
    factor = dst_dpi / src_dpi
    # Задачи: каждый кадр дерева → тот же относительный путь в копии.
    tasks = [
        (path, dst / path.relative_to(src), factor, quality)
        for path in sorted(src.rglob("*"))
        if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES
    ]
    with ProcessPoolExecutor(
        max_workers=jobs, mp_context=multiprocessing.get_context("forkserver"), initializer=_init_worker
    ) as pool:
        written = [name for name in pool.map(rescale_one, tasks, chunksize=2) if name]
    click.echo(f"кадров {len(tasks)}, пересчитано {len(written)}, множитель {factor:.3f} → {dst}")


if __name__ == "__main__":
    main()
