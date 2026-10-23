"""CLI вырезки сканов v2: ``python -m ocr_utils.scan_cropping.v2`` — SAM 3 и склейка страниц вместо YOLO-World + SAM.

Здесь только click: объявление опций, их разбор и сборка :class:`pipeline.RunParams`.

    uv run python -m ocr_utils.scan_cropping.v2 \\
        --input-dir IN --output OUT --cache CACHE --debug DBG --fill telea --dpi 450
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from ocr_utils.scan_cropping.cropping import CROP_FILL_FADE
from ocr_utils.scan_cropping.image_io import collect_images
from ocr_utils.scan_cropping.levels import LEVELS_MAX_BLACK
from ocr_utils.scan_cropping.v2.ragged_edge import RAGGED_TORTUOSITY
from ocr_utils.scan_cropping.v2.spread_lines import SpineMethod
from ocr_utils.scan_cropping.v2.border_fill import SEAM_MM
from ocr_utils.scan_cropping.v2.fill import FillMethod, ReplicateParams
from ocr_utils.scan_cropping.v2.fingers import LAMA_HOLE_MAX_PX, LIGHT_INCREMENT
from ocr_utils.scan_cropping.v2.pipeline import (
    CORE_GROW_MM,
    CROP_FILL_BLUR_MM,
    DEFAULT_FINGER_THREADS,
    DEFAULT_JOBS,
    FINGER_DILATE_MM,
    LAYOUT_PAD_MM,
    REFINE_DEPTH_MM,
    RunParams,
    run_batch,
    visible_frames,
)
from ocr_utils.scan_cropping.v2.scale import REF_PX_PER_MM, Scale

log = logging.getLogger(__name__)


def parse_pad(value: str) -> tuple[float, float]:
    """Разбор пары ``"N"`` или ``"X,Y"`` (припуск в мм, высветление в уровнях).

    Args:
        value: Строка опции.

    Returns:
        ``(по X, по Y)``.
    """
    parts = [float(v) for v in value.split(",")]
    return (parts[0], parts[0]) if len(parts) == 1 else (parts[0], parts[1])


def frame_scale(px_per_mm: float | None, dpi: float | None) -> Scale:
    """Масштабы прогона по ``--px-per-mm`` или ``--dpi`` (не обоим сразу).

    Args:
        px_per_mm: Плотность кадра, px/мм бумаги, или None.
        dpi: Разрешение бумаги на кадре, px/дюйм, или None.

    Returns:
        :class:`Scale`; ни одна опция не задана — калибровочная плотность ``REF_PX_PER_MM``.

    Raises:
        click.UsageError: Заданы обе опции.
    """
    if px_per_mm is not None and dpi is not None:
        raise click.UsageError("--px-per-mm и --dpi задают одно и то же — нужна одна из них")
    if dpi is not None:
        return Scale.from_dpi(dpi)
    return Scale(REF_PX_PER_MM if px_per_mm is None else px_per_mm)


@click.command()
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Папка входных кадров.")  # fmt: skip
@click.option("--output", "--output-dir", "output_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Папка итоговых прямоугольников (структура подпапок входа повторяется).")  # fmt: skip
@click.option("--cache", "--work-dir", "work_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Папка кэша шагов по кадрам: маски, сводка; повторный прогон берёт готовое.")  # fmt: skip
@click.option("--debug", "--debug-dir", "debug_dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Папка оверлеев (JPEG), по подпапке на тип: main/, shadow/, surya/ (и шаги при --debug-steps); "
                   "не задана — без оверлеев.")  # fmt: skip
@click.option("--debug-steps", is_flag=True,
              help="С --debug: ещё картинки шагов сегментации бумаги, каждая в свою подпапку (01_book — фон и книга, "
                   "02_sam3, 03_snap — plain и доводка края, 04_glue — склейка и уточнение, 05_sides — стороны и "
                   "защита, 06_pages — трапеции).")  # fmt: skip
@click.option("--spine-method", type=click.Choice([m.value for m in SpineMethod if m is not SpineMethod.FOLD]),
              default=SpineMethod.AUTO.value, show_default=True,
              help="Способ прямой-разделителя страниц на оверлеях (только диагностика): quad_mid — середины трапеций, "
                   "gap_mid — середина между кромками масок, gap_mid_run — то же по сплошному участку маски, "
                   "auto — gap_mid, а при наклоне дальше 1° от quad_mid — quad_mid.")  # fmt: skip
@click.option("--save-fingers/--no-save-fingers", default=False, show_default=True,
              help="Оставлять в --cache кадр без пальцев (clean.png, ~18 МБ на кадр).")  # fmt: skip
@click.option("--recursive", is_flag=True, help="Обходить подпапки (структура повторяется в выходе).")
@click.option("--skip-if-exists", is_flag=True, help="Пропускать кадры, итог которых уже есть.")
@click.option("--redo", is_flag=True, help="Пересчитать кэш шагов в --cache.")
@click.option(
    "--jobs", default=DEFAULT_JOBS, show_default=True, help="Число CPU-воркеров (GPU-шаги — в главном процессе)."
)
@click.option("--finger-threads", default=DEFAULT_FINGER_THREADS, show_default=True,
              help="Потоков закраса пальцев (сети — под общей блокировкой GPU; 1 — по кадру, как раньше).")  # fmt: skip
@click.option("--px-per-mm", type=float, default=None,
              help=f"Плотность кадра, px/мм бумаги; без неё и без --dpi — {REF_PX_PER_MM} (калибровочные кадры "
                   "подшивок МТО 1990, пак-6). Мерить заранее и задавать в run-скрипте.")  # fmt: skip
@click.option("--dpi", type=float, default=None,
              help="То же, что --px-per-mm, в px на дюйм бумаги (450 у новых паков: 17.72 px/мм). Все размеры в мм "
                   "переводятся в пиксели по этой плотности.")  # fmt: skip
@click.option("--fill", type=click.Choice([m.value for m in FillMethod]), default=FillMethod.PULL_PUSH_SEAM.value,
              show_default=True,
              help="Закрас до прямоугольника: pull_push_seam (pull-push от чистой бумаги у края, без печатей и края "
                   "обложки, с мягким швом от каймы листа; research/border_fill, вариант H5), pull_push (то же без "
                   "шва — прежний), telea или прежняя заливка v1 (replicate).")  # fmt: skip
@click.option("--fill-seam-mm", default=SEAM_MM, show_default=True,
              help="pull_push_seam: ширина рампы от цвета каймы листа к закрасу, мм (3 — заметно резче, 8 — шлейф "
                   "+8 %; research/border_fill).")  # fmt: skip
@click.option("--crop-fill-blur-mm", default=CROP_FILL_BLUR_MM, show_default=True,
              help="replicate: наибольшая σ размытия заливки, мм бумаги (48 px при 13.8 px/мм).")  # fmt: skip
@click.option("--crop-fill-fade", default=CROP_FILL_FADE, show_default=True,
              help="replicate: доля выцветания к среднему цвету книги на самом дальнем пикселе.")  # fmt: skip
@click.option("--lama-hole-max-px", default=LAMA_HOLE_MAX_PX, show_default=True,
              help="Предел длинной стороны рамки дыры в масштабе LaMa, px; 0 — прежнее ужатие ROI до 512 px.")  # fmt: skip
@click.option("--finger-dilate-mm", default=FINGER_DILATE_MM, show_default=True,
              help="Припуск зоны закраса пальца, мм бумаги (40 px при 13.8 px/мм).")  # fmt: skip
@click.option("--layout-pad-mm", default=f"{LAYOUT_PAD_MM[0]:g},{LAYOUT_PAD_MM[1]:g}", show_default=True,
              help="Припуск блоков surya, мм бумаги: N или X,Y (18,36 px при 13.8 px/мм). Один и тот же для copy back "
                   "в зоне пальца и для защиты контента от вырезки.")  # fmt: skip
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
@click.option("--refine-depth-mm", default=REFINE_DEPTH_MM, show_default=True,
              help="Глубина уточнения склейки от фона, мм бумаги (35 px масштаба сегментации).")  # fmt: skip
@click.option("--compensate-levels", is_flag=True, help="Растянуть уровни итога по перцентилям.")
@click.option("--levels-max-black", default=LEVELS_MAX_BLACK, show_default=True,
              help="Потолок чёрной точки растяжки уровней, 0..255: пустой разворот без краски не растягивается "
                   "«бумага → весь диапазон» (пятна цвета); 255 — без потолка.")  # fmt: skip
@click.option("--finger-shadow/--no-finger-shadow", default=False, show_default=True,
              help="После copy back и уровней осветлять тень пальца на бумаге по карте: зона ядро + 15 мм в книге, "
                   "опора — pull-push по кольцу бумаги, прибавка (research/finger_repaint). С ним высветление зоны "
                   "пальца перед LaMa не нужно: --finger-zone-light-increment 0.")  # fmt: skip
@click.option("--copy-back-core-grow-mm", default=CORE_GROW_MM, show_default=True,
              help="Припуск ядра пальца при copy back, мм бумаги (4 px при 13.8 px/мм): кайма кожи у края ядра SAM 3 "
                   "не возвращается с оригинала; 0 — без припуска.")  # fmt: skip
@click.option("--colour-pages/--no-colour-pages", default=True, show_default=True,
              help="Правила для целиком цветных страниц: рост бумаги у края по палитре страницы вверх и вниз, защита "
                   "контента при любой позе.")  # fmt: skip
@click.option("--bg-veto/--no-bg-veto", default=True, show_default=True,
              help="Вето фона: фон отдаётся книге, где SAM 3 видит страницу, а SAM 2 по точкам коврика — не фон "
                   "(рукописный номер, логотип, линейка у края листа не уводят фон на страницу; +SAM 2 на GPU, "
                   "~0.4 с на кадр).")  # fmt: skip
@click.option("--gutter-strip/--no-gutter-strip", default=True, show_default=True,
              help="Полоса-вклейка у корешка (щель SAM 3 ≥ 7.8 мм) — в бумагу кадра: склейка страниц закрывает щель только "
                   "до ~11.6 мм, белая часть полосы шире иначе закрашивается до прямоугольника.")  # fmt: skip
@click.option("--refine-ragged/--no-refine-ragged", default=False, show_default=True,
              help="Тонкое уточнение рваных внешних сторон страниц разворота: детектор по сырой маске SAM 3, сегментация "
                   "ленты вдоль стороны в полном разрешении (SAM 3 и SAM 2.1 прогона, сокращённый набор способов, при "
                   "неудаче — полный); 2–6 с на рваную сторону.")  # fmt: skip
@click.option("--ragged-threshold", type=float, default=RAGGED_TORTUOSITY, show_default=True,
              help="Порог извилистости сырой маски SAM 3 внешней стороны страницы для --refine-ragged.")  # fmt: skip
@click.option("--dual-levels/--no-dual-levels", default=False, show_default=True,
              help="Двойная грубая сегментация: SAM 3 без растяжения уровней — база, в зонах у внешних кромок книги — "
                   "проход по кадру с растянутыми уровнями (+1 проход SAM 3 на кадр); промахи прохода — зона на "
                   "базе.")  # fmt: skip
@click.option("--levels-edges/--no-levels-edges", default=True, show_default=True,
              help="При --dual-levels: у верхних и нижних краёв страниц к базе SAM 3 добавлять то, что видит проход по "
                   "растянутым уровням (объединение, не замена; заделывает выемки края у корешка).")  # fmt: skip
@click.option("--sam3-levels/--no-sam3-levels", default=False, show_default=True,
              help="Проба: растягивать уровни светлоты кадра по бумаге перед SAM 3 (против рваных краёв). Кэш SAM 3 "
                   "флагом не помечается — сравнивать в отдельном --cache.")  # fmt: skip
@click.option("--save-before-shadow", is_flag=True,
              help="Сохранять в --cache итог до осветления тени и маски входа карты (стенд research/finger_repaint "
                   "edge, ~40 МБ на кадр).")  # fmt: skip
@click.option("--output-format", type=click.Choice(["png", "tiff"]), default=None,
              help="Формат итога; по умолчанию как у входа.")  # fmt: skip
@click.option("--force-dpi", type=int, default=None, help="Записать в итог это разрешение, dpi.")
@click.option("--log-level", default="INFO", show_default=True, help="Уровень логов.")
def main(
    input_dir: Path,
    output_dir: Path,
    work_dir: Path,
    debug_dir: Path | None,
    debug_steps: bool,
    spine_method: str,
    recursive: bool,
    skip_if_exists: bool,
    redo: bool,
    jobs: int,
    finger_threads: int,
    save_fingers: bool,
    px_per_mm: float | None,
    dpi: float | None,
    fill: str,
    fill_seam_mm: float,
    crop_fill_blur_mm: float,
    crop_fill_fade: float,
    lama_hole_max_px: int,
    finger_dilate_mm: float,
    layout_pad_mm: str,
    finger_zone_light_increment: str,
    ink_fill: bool,
    fit_sides: bool,
    keystone_layout: bool,
    snap_sides: bool,
    copy_back_over_fingers: bool,
    fill_before_copy_back: bool,
    refine_depth_mm: float,
    compensate_levels: bool,
    levels_max_black: int,
    finger_shadow: bool,
    save_before_shadow: bool,
    copy_back_core_grow_mm: float,
    colour_pages: bool,
    bg_veto: bool,
    gutter_strip: bool,
    sam3_levels: bool,
    dual_levels: bool,
    levels_edges: bool,
    refine_ragged: bool,
    ragged_threshold: float,
    output_format: str | None,
    force_dpi: int | None,
    log_level: str,
) -> None:
    """Вырезка разворотов v2: пальцы (LaMa, inkback) → фон → SAM 3 → склейка страниц → закрас до прямоугольника."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    scale = frame_scale(px_per_mm, dpi)
    params = RunParams(
        input_dir=input_dir,
        output_dir=output_dir,
        work_dir=work_dir,
        debug_dir=debug_dir,
        debug_steps=debug_steps,
        spine_method=SpineMethod(spine_method),
        scale=scale,
        fill=FillMethod(fill),
        fill_seam_mm=fill_seam_mm,
        # Прежняя заливка v1 считает σ в px кадра — переводим из мм по плотности кадра.
        replicate=ReplicateParams(scale.mm_px_float(crop_fill_blur_mm), crop_fill_fade),
        hole_max_px=lama_hole_max_px or None,
        finger_dilate_mm=finger_dilate_mm,
        layout_pad_mm=parse_pad(layout_pad_mm),
        light_increment=tuple(float(v) for v in parse_pad(finger_zone_light_increment)),
        ink_fill=ink_fill,
        fit_sides=fit_sides,
        keystone_layout=keystone_layout,
        snap_sides=snap_sides,
        copy_back_over_fingers=copy_back_over_fingers,
        fill_before_copy_back=fill_before_copy_back,
        refine_depth_mm=refine_depth_mm,
        compensate_levels=compensate_levels,
        levels_max_black=levels_max_black,
        finger_shadow=finger_shadow,
        save_before_shadow=save_before_shadow,
        copy_back_core_grow_mm=copy_back_core_grow_mm,
        colour_pages=colour_pages,
        bg_veto=bg_veto,
        gutter_strip=gutter_strip,
        sam3_levels=sam3_levels,
        dual_levels=dual_levels,
        levels_edges=levels_edges,
        refine_ragged=refine_ragged,
        ragged_threshold=ragged_threshold,
        output_format=output_format,
        force_dpi=force_dpi,
        skip_if_exists=skip_if_exists,
        redo=redo,
        jobs=jobs,
        finger_threads=finger_threads,
        save_fingers=save_fingers,
    )
    files = visible_frames(collect_images(input_dir, recursive), input_dir)
    log.info(
        "кадров: %d; %.2f px/мм, масштаб сегментации %.3f, закрас до прямоугольника: %s",
        len(files),
        params.scale.px_per_mm,
        params.scale.seg,
        fill,
    )
    stats = run_batch(files, params)
    bad = [s for s in stats if s.status != "ok"]
    log.info("готово: %d кадров, не ok: %d %s", len(stats), len(bad), [(s.key, s.status) for s in bad])
