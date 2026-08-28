"""Уменьшенные копии полос для разметки в CVAT.

CVAT не масштабирует кадры сам: ``image_quality`` — это только качество JPEG-пережатия,
разрешение чанка всегда 1:1 с исходником (``ZipCompressedChunkWriter._compress_image`` в
``media_extractors.py`` — там нет ни одного resize). Полноразмерный скан 600 dpi он режет
на чанки по 5 кадров, и листание становится невыносимым. Поэтому уменьшаем заранее.

Логика уменьшения — та же, что в ``docker/downscale_for_cvat.py``, с одним отличием:
делитель у каждой полосы свой, он посчитан на шаге ``detect`` из DPI файла (600 dpi -> 8,
450 dpi -> 6) и лежит в базе. Сначала обрезка справа-снизу до кратности, потом деление —
мотивировка в ``scan_markup.geometry``.
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Ужимаем щадяще: поверх этого CVAT наложит своё image_quality, а пережимать дважды
# агрессивно — значит без нужды сыпать артефакты на тонкие оттиски печатей, которые
# разметчику как раз и предстоит обводить кистью.
JPEG_QUALITY = 95


@dataclass
class ImageJob:
    """Одна полоса: что откуда куда уменьшать."""

    src: Path
    dst: Path
    divisor: int
    page_id: int
    force: bool = False  # перезаписать готовую копию: исходник изменился


@dataclass
class ImageResult:
    """Результат по одной полосе."""

    page_id: int
    status: str  # done | skipped | failed
    cvat_width: int = 0
    cvat_height: int = 0
    error: str = ""


def downscale_one(job: ImageJob, force: bool = False) -> ImageResult:
    """Уменьшает одну полосу; повторный запуск пропускает готовое, если нет ``force``.

    Запись атомарная, через ``.part`` и ``replace``: прерванный прогон не должен оставить
    обрезанных JPEG, которые следующий запуск сочтёт готовыми.

    ``force`` бывает общий (перезаписать вообще всё) и на полосу (``job.force``): второй
    ставится тем полосам, чей исходник изменился с прошлого прогона. Без него уменьшенная
    копия осталась бы от старого файла — в CVAT висел бы старый кадр при новом оригинале,
    и разметка на нём выглядела бы верной.
    """
    if job.dst.exists() and not (force or job.force):
        try:
            with Image.open(job.dst) as done:
                return ImageResult(job.page_id, "skipped", done.width, done.height)
        except Exception:  # noqa: BLE001 — битый файл проще перезаписать, чем чинить
            pass

    try:
        job.dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(job.src) as im:
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            crop_w = (im.width // job.divisor) * job.divisor
            crop_h = (im.height // job.divisor) * job.divisor
            if not crop_w or not crop_h:
                return ImageResult(job.page_id, "failed", error=f"кадр меньше делителя: {im.size}")
            if (crop_w, crop_h) != im.size:
                im = im.crop((0, 0, crop_w, crop_h))
            im = im.resize((crop_w // job.divisor, crop_h // job.divisor), Image.Resampling.LANCZOS)

            tmp = job.dst.with_suffix(job.dst.suffix + ".part")
            im.save(tmp, "JPEG", quality=JPEG_QUALITY, subsampling=0)
            tmp.replace(job.dst)
            return ImageResult(job.page_id, "done", im.width, im.height)
    except Exception as exc:  # noqa: BLE001 — одна битая полоса не должна валить прогон
        return ImageResult(job.page_id, "failed", error=str(exc))


def _worker(job: ImageJob, force: bool) -> ImageResult:
    """Обёртка для ``ProcessPoolExecutor.map`` (лямбду не запикль)."""
    return downscale_one(job, force)


def cvat_rel_path(pack_name: str, page_rel_path: str) -> str:
    """Путь картинки внутри share-каталога: ``<пак>/<год>/<выпуск>/<имя>.jpg``.

    Пак включается в путь, потому что share один на все паки, а имена полос между паками
    повторяются.
    """
    return f"{pack_name}/{Path(page_rel_path).with_suffix('.jpg').as_posix()}"


def prepare_images(jobs: list[ImageJob], share_root: Path, workers: int, force: bool = False) -> list[ImageResult]:
    """Готовит все уменьшенные копии; возвращает результат по каждой полосе."""
    share_root.mkdir(parents=True, exist_ok=True)
    results: list[ImageResult] = []
    if not jobs:
        return results

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in tqdm(
            pool.map(_worker, jobs, [force] * len(jobs), chunksize=8), total=len(jobs), desc="уменьшение", unit="полоса"
        ):
            if result.status == "failed":
                tqdm.write(f"ОШИБКА уменьшения page_id={result.page_id}: {result.error}")
            results.append(result)
    return results
