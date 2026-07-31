"""CLI сглаживания фона: ``python -m ocr_utils.background_smoothing``.

Здесь ТОЛЬКО click: объявление опций, их разбор и сборка ``SmoothParams``.
Расчёт по кадру — в ``processing``, обход пачки — в ``pipeline.run_batch``.

    uv run python -m ocr_utils.background_smoothing \\
        --input-dir /path/to/1966/03 --output-dir /path/to/1966/03_smooth
"""

import logging
from pathlib import Path
from typing import Optional

import click

from ocr_utils.background_smoothing.pipeline import BLUR_MODE_MASKED, BLUR_MODES, SmoothParams, run_batch
from ocr_utils.background_smoothing.processing import (
    DEFAULT_BLUR_MULT,
    DEFAULT_SAUVOLA_K,
    DEFAULT_THRESHOLD_BIAS,
    MASK_METHODS,
    METHOD_OTSU,
)
from ocr_utils.scan_cropping.image_io import collect_images

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Папка с исходниками.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Папка результата; относительные пути входа сохраняются.",
)
@click.option(
    "--debug-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Папка для debug-оверлеев (исходник + полупрозрачные маски). Не задана — оверлеи не рисуются.",
)
@click.option(
    "--output-format",
    default=None,
    type=click.Choice(["tif", "tiff", "png", "jpg", "jpeg"], case_sensitive=False),
    help="Формат выходных файлов. Не задан — тот же, что у входного файла.",
)
@click.option(
    "--recursive/--no-recursive",
    "recursive",
    default=True,
    show_default=True,
    help="Обходить подкаталоги; структура каталогов зеркалится в output.",
)
@click.option(
    "--skip-if-exists/--no-skip-if-exists",
    "skip_if_exists",
    default=True,
    show_default=True,
    help="Пропускать кадры, для которых выходной файл уже есть.",
)
@click.option(
    "--gray/--color",
    "to_gray",
    default=False,
    show_default=True,
    help="Писать 8-битный серый вместо цвета: для текста и line art цвет ничего не несёт, а файл втрое меньше.",
)
@click.option(
    "--method",
    default=METHOD_OTSU,
    show_default=True,
    type=click.Choice(MASK_METHODS, case_sensitive=False),
    help="Способ построения первичной маски контента.",
)
@click.option(
    "--threshold-bias",
    default=DEFAULT_THRESHOLD_BIAS,
    show_default=True,
    type=float,
    help=(
        "Сдвиг глобального порога от Оцу в сторону бумаги, доля расстояния до неё (0..1). "
        "Больше — маска щедрее, под защиту попадает больше сомнительных пикселей. "
        "0 — чистый Оцу, 1 — уровень бумаги (маской станет почти весь кадр)."
    ),
)
@click.option(
    "--sauvola-k",
    default=DEFAULT_SAUVOLA_K,
    show_default=True,
    type=float,
    help="Параметр k формулы Саволы: меньше — порог ближе к локальному среднему, маска щедрее.",
)
@click.option(
    "--sauvola-window",
    default=None,
    type=int,
    help="Окно Саволы в пикселях (приводится к нечётному). Не задано — из длинной стороны кадра (~101 px при 600 dpi).",
)
@click.option(
    "--dilate-px",
    default=None,
    type=float,
    help=(
        "Радиус припуска защитной маски, пикс. Не задан — из длинной стороны кадра (~15 px при 600 dpi, "
        "то есть ядро ≈30x30 — половинка средней буквы)."
    ),
)
@click.option(
    "--blur-mult",
    default=DEFAULT_BLUR_MULT,
    show_default=True,
    type=float,
    help="Во сколько раз радиус размытия фона больше радиуса припуска.",
)
@click.option(
    "--blur-mode",
    default=BLUR_MODE_MASKED,
    show_default=True,
    type=click.Choice(BLUR_MODES, case_sensitive=False),
    help=(
        "masked — нормированное размытие, пиксели защитной маски исключены из усреднения; "
        "plain — обычное размытие всего кадра (для сравнения: затягивает чернила в фон)."
    ),
)
@click.option(
    "--log-level",
    default="WARNING",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="INFO показывает тайминги шагов по каждому кадру.",
)
def main(
    input_dir: Path,
    output_dir: Path,
    debug_dir: Optional[Path],
    output_format: Optional[str],
    recursive: bool,
    skip_if_exists: bool,
    to_gray: bool,
    method: str,
    threshold_bias: float,
    sauvola_k: float,
    sauvola_window: Optional[int],
    dilate_px: Optional[float],
    blur_mult: float,
    blur_mode: str,
    log_level: str,
) -> None:
    """Сглаживает фон сканов, не трогая контент: подготовка к бинаризации в FineReader."""
    logging.getLogger().setLevel(log_level.upper())
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    params = SmoothParams(
        input_dir=input_dir,
        output_dir=output_dir,
        debug_dir=debug_dir,
        recursive=recursive,
        skip_if_exists=skip_if_exists,
        output_format=output_format,
        to_gray=to_gray,
        method=method.lower(),
        threshold_bias=threshold_bias,
        sauvola_k=sauvola_k,
        sauvola_window=sauvola_window,
        dilate_px=dilate_px,
        blur_mult=blur_mult,
        blur_mode=blur_mode.lower(),
    )

    files = collect_images(input_dir, recursive)
    if not files:
        logger.warning("В %s не найдено изображений", input_dir)
        return

    logger.info(
        "Файлов: %d | маска: %s (bias=%.2f%s) | припуск: %s | размытие: x%.1f, режим %s | выход: %s%s",
        len(files),
        params.method,
        params.threshold_bias,
        f", k={params.sauvola_k}" if params.method != METHOD_OTSU else "",
        f"{params.dilate_px} px" if params.dilate_px is not None else "из размера кадра",
        params.blur_mult,
        params.blur_mode,
        params.output_format or "как у входа",
        ", серый" if params.to_gray else "",
    )

    run_batch(files, params)
    logger.info("Готово. Результат → %s%s", output_dir, f", оверлеи → {debug_dir}" if debug_dir else "")


if __name__ == "__main__":
    main()
