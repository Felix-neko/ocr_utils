"""CLI вырезки сканов v2: ``python -m ocr_utils.scan_cropping.v2`` — SAM 3 и склейка страниц вместо YOLO-World + SAM.

Здесь только click: объявление опций, их разбор и сборка :class:`pipeline.RunParams`.

    uv run python -m ocr_utils.scan_cropping.v2 \\
        --input-dir IN --output OUT --cache CACHE --debug DBG --fill telea --px-per-mm 13.8
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from ocr_utils.scan_cropping.cropping import CROP_FILL_BLUR_PX, CROP_FILL_FADE
from ocr_utils.scan_cropping.image_io import collect_images
from ocr_utils.scan_cropping.v2.copy_back import LAYOUT_PAD_PX
from ocr_utils.scan_cropping.v2.fill import FillMethod, ReplicateParams
from ocr_utils.scan_cropping.v2.fingers import LAMA_HOLE_MAX_PX, LIGHT_INCREMENT, TIGHT_DILATE_PX
from ocr_utils.scan_cropping.v2.glue_refine import REFINE_DEPTH
from ocr_utils.scan_cropping.v2.pipeline import DEFAULT_JOBS, RunParams, run_batch, visible_frames
from ocr_utils.scan_cropping.v2.scale import REF_PX_PER_MM, Scale

log = logging.getLogger(__name__)


def parse_pad(value: str) -> tuple[int, int]:
    """Разбор припуска ``"N"`` или ``"X,Y"``.

    Args:
        value: Строка опции.

    Returns:
        ``(по X, по Y)``.
    """
    parts = [int(v) for v in value.split(",")]
    return (parts[0], parts[0]) if len(parts) == 1 else (parts[0], parts[1])


@click.command()
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Папка входных кадров.")  # fmt: skip
@click.option("--output", "--output-dir", "output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Папка итоговых прямоугольников (структура подпапок входа повторяется).")  # fmt: skip
@click.option("--cache", "--work-dir", "work_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Папка кэша шагов по кадрам: маски, сводка; повторный прогон берёт готовое.")  # fmt: skip
@click.option("--debug", "--debug-dir", "debug_dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Папка оверлеев (JPEG); не задана — без оверлеев.")  # fmt: skip
@click.option("--save-fingers/--no-save-fingers", default=False, show_default=True,
              help="Оставлять в --cache кадр без пальцев (clean.png, ~18 МБ на кадр).")  # fmt: skip
@click.option("--recursive", is_flag=True, help="Обходить подпапки (структура повторяется в выходе).")
@click.option("--skip-if-exists", is_flag=True, help="Пропускать кадры, итог которых уже есть.")
@click.option("--redo", is_flag=True, help="Пересчитать кэш шагов в --cache.")
@click.option(
    "--jobs", default=DEFAULT_JOBS, show_default=True, help="Число CPU-воркеров (GPU-шаги — в главном процессе)."
)
@click.option("--px-per-mm", default=REF_PX_PER_MM, show_default=True,
              help="Плотность кадра, px/мм бумаги; по умолчанию — калибровочные кадры подшивок МТО 1990 (пак-6). "
                   "Мерить заранее и задавать в run-скрипте.")  # fmt: skip
@click.option("--fill", type=click.Choice([m.value for m in FillMethod]), default=FillMethod.PULL_PUSH.value,
              show_default=True,
              help="Закрас до прямоугольника: pull_push (от чистой бумаги у края, без печатей и края обложки; "
                   "research/border_fill), telea или прежняя заливка v1 (replicate).")  # fmt: skip
@click.option("--crop-fill-blur-px", default=CROP_FILL_BLUR_PX, show_default=True,
              help="replicate: наибольшая σ размытия заливки, px.")  # fmt: skip
@click.option("--crop-fill-fade", default=CROP_FILL_FADE, show_default=True,
              help="replicate: доля выцветания к среднему цвету книги на самом дальнем пикселе.")  # fmt: skip
@click.option("--lama-hole-max-px", default=LAMA_HOLE_MAX_PX, show_default=True,
              help="Предел длинной стороны рамки дыры в масштабе LaMa, px; 0 — прежнее ужатие ROI до 512 px.")  # fmt: skip
@click.option("--finger-dilate-px", default=TIGHT_DILATE_PX, show_default=True,
              help="Припуск зоны закраса пальца, px калибровочного кадра (пересчитывается по --px-per-mm).")  # fmt: skip
@click.option("--layout-pad-px", default=f"{LAYOUT_PAD_PX[0]},{LAYOUT_PAD_PX[1]}", show_default=True,
              help="Припуск блоков surya, px калибровочного кадра (пересчитывается по --px-per-mm): N или X,Y. Один "
                   "и тот же для copy back в зоне пальца и для защиты контента от вырезки.")  # fmt: skip
@click.option("--finger-zone-light-increment", default=f"{LIGHT_INCREMENT[0]:g},{LIGHT_INCREMENT[1]:g}",
              show_default=True,
              help="Высветление зоны пальца перед LaMa, как в v1: N или «слева,справа» уровней; 0 — без.")  # fmt: skip
@click.option("--copy-back-over-fingers/--no-copy-back-over-fingers", default=True, show_default=True,
              help="Copy back блоков surya, задетых закрасом, и поверх ядра пальца и рук (весь «не фон»): при ложном "
                   "срабатывании детектора пальцев текст не теряется, при верном возвращается кусок пальца.")  # fmt: skip
@click.option("--fill-before-copy-back/--no-fill-before-copy-back", default=True, show_default=True,
              help="Закрас до прямоугольника от точного края бумаги, защита контента и copy back — поверх готового "
                   "закраса: печать и край обложки в блоке surya у края не тянутся шлейфом Telea.")  # fmt: skip
@click.option("--keystone-layout/--no-keystone-layout", default=True, show_default=True,
              help="surya layout по кадру, выпрямленному по трапециям страниц; блоки возвращаются наклонными "
                   "четырёхугольниками (рамки по тексту, а не по осям кадра).")  # fmt: skip
@click.option("--fit-sides/--no-fit-sides", default=True, show_default=True,
              help="Устранять выступы и ямы на внешних боковых сторонах плоских страниц (срез выбросов, заливка ям не "
                   "глубже 10 мм, фон не заливается).")  # fmt: skip
@click.option("--snap-sides/--no-snap-sides", default=False, show_default=True,
              help="При --fit-sides подтягивать сторону к краю листа (переход «жёлтая бумага → не бумага»), где SAM 3 "
                   "колеблется. Выключено: на МТС 1991/05 IMG_0217, 0222, 0224 уводит сторону на торец.")  # fmt: skip
@click.option("--ink-fill/--no-ink-fill", default=True, show_default=True,
              help="Второй проход SAM 3 по кадру без крупной краски (рамки, линейки, плашки закрашены цветом бумаги); "
                   "бумага — объединение двух проходов.")  # fmt: skip
@click.option("--refine-depth", default=REFINE_DEPTH, show_default=True,
              help="Глубина уточнения склейки от фона, px масштаба сегментации (35 ≈ 5 мм).")  # fmt: skip
@click.option("--compensate-levels", is_flag=True, help="Растянуть уровни итога по перцентилям.")
@click.option("--finger-shadow/--no-finger-shadow", default=False, show_default=True,
              help="После copy back и уровней осветлять тень пальца на бумаге по карте: зона ядро + 15 мм в книге, "
                   "опора — pull-push по кольцу бумаги, прибавка (research/finger_repaint). С ним высветление зоны "
                   "пальца перед LaMa не нужно: --finger-zone-light-increment 0.")  # fmt: skip
@click.option("--output-format", type=click.Choice(["png", "tiff"]), default=None,
              help="Формат итога; по умолчанию как у входа.")  # fmt: skip
@click.option("--force-dpi", type=int, default=None, help="Записать в итог это разрешение, dpi.")
@click.option("--log-level", default="INFO", show_default=True, help="Уровень логов.")
def main(
    input_dir: Path,
    output_dir: Path,
    work_dir: Path,
    debug_dir: Path | None,
    recursive: bool,
    skip_if_exists: bool,
    redo: bool,
    jobs: int,
    save_fingers: bool,
    px_per_mm: float,
    fill: str,
    crop_fill_blur_px: float,
    crop_fill_fade: float,
    lama_hole_max_px: int,
    finger_dilate_px: int,
    layout_pad_px: str,
    finger_zone_light_increment: str,
    ink_fill: bool,
    fit_sides: bool,
    keystone_layout: bool,
    snap_sides: bool,
    copy_back_over_fingers: bool,
    fill_before_copy_back: bool,
    refine_depth: int,
    compensate_levels: bool,
    finger_shadow: bool,
    output_format: str | None,
    force_dpi: int | None,
    log_level: str,
) -> None:
    """Вырезка разворотов v2: пальцы (LaMa, inkback) → фон → SAM 3 → склейка страниц → закрас до прямоугольника."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = RunParams(
        input_dir=input_dir,
        output_dir=output_dir,
        work_dir=work_dir,
        debug_dir=debug_dir,
        scale=Scale(px_per_mm),
        fill=FillMethod(fill),
        replicate=ReplicateParams(crop_fill_blur_px, crop_fill_fade),
        hole_max_px=lama_hole_max_px or None,
        finger_dilate_px=finger_dilate_px,
        layout_pad_px=parse_pad(layout_pad_px),
        light_increment=tuple(float(v) for v in parse_pad(finger_zone_light_increment)),
        ink_fill=ink_fill,
        fit_sides=fit_sides,
        keystone_layout=keystone_layout,
        snap_sides=snap_sides,
        copy_back_over_fingers=copy_back_over_fingers,
        fill_before_copy_back=fill_before_copy_back,
        refine_depth=refine_depth,
        compensate_levels=compensate_levels,
        finger_shadow=finger_shadow,
        output_format=output_format,
        force_dpi=force_dpi,
        skip_if_exists=skip_if_exists,
        redo=redo,
        jobs=jobs,
        save_fingers=save_fingers,
    )
    files = visible_frames(collect_images(input_dir, recursive), input_dir)
    log.info("кадров: %d; масштаб сегментации %.3f, закрас до прямоугольника: %s", len(files), params.scale.seg, fill)
    stats = run_batch(files, params)
    bad = [s for s in stats if s.status != "ok"]
    log.info("готово: %d кадров, не ok: %d %s", len(stats), len(bad), [(s.key, s.status) for s in bad])
