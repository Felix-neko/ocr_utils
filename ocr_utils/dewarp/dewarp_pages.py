"""CLI геометрического выпрямления страниц (распрямление строк).

Выбирает движок dewarp по имени, прогоняет по изображениям из ``--input-dir`` и кладёт
результат каждого движка в свою подпапку ``<output-base>/<method>/``.

Примеры:
    # основной движок (DocScanner), 10 страниц
    uv run python -m ocr_utils.dewarp --input-dir IN --method docscanner --limit 10

    # все движки последовательно, каждый в свою папку
    uv run python -m ocr_utils.dewarp --input-dir IN --method all
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import click
import cv2
import torch
from tqdm import tqdm

from ocr_utils.dewarp.engines import ENGINES, get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
# По умолчанию результаты складываем внутрь самого подпакета dewarp
DEFAULT_OUTPUT_BASE = Path(__file__).resolve().parent


def collect_images(input_dir: Path, limit: int) -> list[Path]:
    """Собирает изображения в каталоге (без рекурсии), опционально первые ``limit``."""
    files = [f for f in sorted(input_dir.iterdir()) if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    if limit > 0:
        files = files[:limit]
    return files


def run_pagedewarp_parallel(files: list[Path], out_dir: Path, no_binary: bool, jobs: int, extended: bool) -> None:
    """Параллельный прогон page-dewarp по пулу процессов (CPU-алгоритм, без GPU)."""
    from ocr_utils.dewarp.engines.pagedewarp import pagedewarp_one

    todo = [(str(src), str(out_dir / f"{src.stem}.jpg")) for src in files if not (out_dir / f"{src.stem}.jpg").exists()]
    if not todo:
        return
    logger.info(
        "page-dewarp: %d файлов, процессов: %d, no_binary=%s, режим=%s",
        len(todo),
        jobs,
        no_binary,
        "extended" if extended else "vanilla",
    )
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(pagedewarp_one, src, dst, no_binary, extended): src for src, dst in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="pagedewarp", unit="img"):
            src, ok, info = fut.result()
            if not ok:
                tqdm.write(f"  Пропущен {Path(src).name}: {info}")


def run_engine(
    name: str, files: list[Path], output_base: Path, device: str, no_binary: bool, jobs: int, pd_extended: bool
) -> None:
    """Грузит движок ``name`` и прогоняет по файлам, результат — в ``output_base/name``."""
    out_dir = output_base / name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== %s → %s ===", name, out_dir)

    # page-dewarp — классический CPU-алгоритм: гоним параллельно по ядрам
    if name == "pagedewarp":
        run_pagedewarp_parallel(files, out_dir, no_binary, jobs, pd_extended)
        return

    engine = get_engine(name)
    try:
        engine.load(device)
    except Exception as e:
        logger.error("Движок %s не загрузился: %s", name, e)
        return

    for src in tqdm(files, desc=name, unit="img"):
        dst = out_dir / f"{src.stem}.jpg"
        if dst.exists():
            continue
        try:
            img = cv2.imread(str(src))
            if img is None:
                tqdm.write(f"  Не удалось прочитать: {src.name}")
                continue
            result = engine.dewarp(img)
            if result is None:
                tqdm.write(f"  Пропущен (движок не обработал): {src.name}")
                continue
            cv2.imwrite(str(dst), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as e:
            tqdm.write(f"  Ошибка {src.name}: {e}")
            import traceback

            tqdm.write(traceback.format_exc())


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Каталог с исходными изображениями",
)
@click.option(
    "--output-base",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_OUTPUT_BASE,
    show_default=True,
    help="Куда складывать подпапки движков (<output-base>/<method>/)",
)
@click.option(
    "--method",
    type=click.Choice([*ENGINES.keys(), "all"]),
    default="uvdoc",
    show_default=True,
    help="Движок выпрямления (или 'all' — все по очереди)",
)
@click.option("--limit", default=0, show_default=True, help="Обработать только первые N файлов (0 — все)")
@click.option("--device", default=None, help="cuda / cpu (по умолчанию авто)")
@click.option(
    "--no-binary/--binary",
    default=True,
    show_default=True,
    help="page-dewarp: отключить бинаризацию (оставить grayscale) или включить порог",
)
@click.option(
    "--jobs",
    default=4,
    show_default=True,
    help="page-dewarp: число процессов (по умолчанию 4; 0 — все ядра CPU)",
)
@click.option(
    "--pagedewarp-extended/--pagedewarp-vanilla",
    "pd_extended",
    default=False,
    show_default=True,
    help="page-dewarp: vanilla (чистая библиотека) или extended (поля 0, ширина как у входа, цвет)",
)
def main(
    input_dir: Path,
    output_base: Path,
    method: str,
    limit: int,
    device: Optional[str],
    no_binary: bool,
    jobs: int,
    pd_extended: bool,
) -> None:
    """Выпрямляет страницы (и строки) выбранным движком."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if jobs <= 0:
        jobs = os.cpu_count() or 1

    files = collect_images(input_dir, limit)
    if not files:
        logger.warning("Изображения не найдены в %s", input_dir)
        return

    methods = list(ENGINES.keys()) if method == "all" else [method]
    logger.info("Файлов: %d | устройство: %s | движки: %s", len(files), device, ", ".join(methods))

    for name in methods:
        run_engine(name, files, output_base, device, no_binary, jobs, pd_extended)

    logger.info("Готово. Результаты в %s/<движок>/", output_base)


if __name__ == "__main__":
    main()
