"""CLI подсистемы очистки пака: ``python -m ocr_utils.scan_cleanup <команда>``.

Команды:

* ``run`` — боевой прогон: закрас разметки и размытие фона за один проход;
* ``compare-inpaint`` — LaMa против Stable Diffusion на выборке полос;
* ``compare-masks`` — защитные маски разными алгоритмами и радиусами.

Здесь только сбор опций: вся работа — в ``runner`` и ``compare``.
"""

import logging
from pathlib import Path
import click

from ocr_utils.background_smoothing.processing import (
    BLUR_MODE_MASKED,
    BLUR_MODES,
    DEFAULT_BLUR_MULT,
    DEFAULT_SAUVOLA_K,
    DEFAULT_THRESHOLD_BIAS,
    MASK_METHODS,
    METHOD_SAUVOLA,
    MIN_GLYPH_AREA,
    SURE_GLYPH_AREA,
)
from ocr_utils.paper import AUTO_INK_LEVEL, INK_LEVEL, PAPER_BLUR_PX, PAPER_DILATE_PX
from ocr_utils.inpainting.backends import BACKEND_LAMA, BACKENDS, DEFAULT_SD_MODEL, SdParams
from ocr_utils.inpainting.grouping import DEFAULT_GROUP_DILATE_FRAC
from ocr_utils.scan_cleanup.inpaint import (
    DEFAULT_LAMA_HOLE_MAX_PX,
    GROUP_MIN_DILATE_PX,
    GROUP_ROI_SCALE,
    InpaintOptions,
)
from ocr_utils.scan_cleanup.prompts import PROMPT_COLOUR, PROMPT_HALFTONE, PROMPT_OTHER_SUFFIX, PROMPT_PAPER
from ocr_utils.scan_cleanup.prompts import NEGATIVE_COMMON, PromptSet
from ocr_utils.scan_cleanup.protect import ProtectOptions
from ocr_utils.scan_cleanup.runner import CleanupParams, run_cleanup, summary
from ocr_utils.scan_cleanup.smoothing import DEFAULT_DILATE_PX, SmoothOptions

logger = logging.getLogger(__name__)

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

DIR_IN = click.Path(exists=True, file_okay=False, path_type=Path)
DIR_OUT = click.Path(file_okay=False, path_type=Path)
FILE_IN = click.Path(exists=True, dir_okay=False, path_type=Path)


def _common_source_options(f):
    """Опции «откуда брать полосы» — общие у всех трёх команд."""
    f = click.option("--db", "db_path", required=True, type=FILE_IN, help="База разметки (та, что после from-cvat).")(f)
    f = click.option("--pack-dir", required=True, type=DIR_IN, help="Корень пака с оригиналами.")(f)
    f = click.option("--pack-name", default=None, help="Имя пака в базе. Не задано — имя папки --pack-dir.")(f)
    f = click.option("--only-year", default=None, help="Только этот год (имя годового комплекта).")(f)
    f = click.option("--only-issue", default=None, help="Только этот выпуск.")(f)
    f = click.option(
        "--pages-file",
        default=None,
        type=FILE_IN,
        help="Файл со списком полос (по относительному пути на строку); строки с # игнорируются.",
    )(f)
    f = click.option("--page", "pages", multiple=True, help="Относительный путь полосы; можно повторять.")(f)
    f = click.option("--limit", default=None, type=int, help="Взять не больше стольких полос (для пробы).")(f)
    f = click.option(
        "--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS, case_sensitive=False)
    )(f)
    return f


def _sd_options(f):
    """Опции Stable Diffusion, включая все четыре промпта."""
    f = click.option("--sd-model", default=DEFAULT_SD_MODEL, show_default=True, help="Модель diffusers для инпейнта.")(
        f
    )
    f = click.option("--sd-steps", default=30, show_default=True, type=int)(f)
    f = click.option("--sd-guidance", default=7.0, show_default=True, type=float)(f)
    f = click.option("--sd-size", default=512, show_default=True, type=int, help="Длинная сторона ROI перед сетью.")(f)
    f = click.option(
        "--sd-seed", default=0, show_default=True, type=int, help="Фиксируется, иначе прогоны несравнимы."
    )(f)
    f = click.option("--sd-prompt-paper", default=PROMPT_PAPER, help="Промпт для зон на чистой бумаге.")(f)
    f = click.option("--sd-prompt-colour", default=PROMPT_COLOUR, help="Промпт для зон внутри цветной иллюстрации.")(f)
    f = click.option(
        "--sd-prompt-halftone", default=PROMPT_HALFTONE, help="Промпт для зон внутри полутоновой иллюстрации."
    )(f)
    f = click.option(
        "--sd-prompt-other", default=PROMPT_OTHER_SUFFIX, help="Добавка к промпту для вида other_removal."
    )(f)
    f = click.option("--sd-negative", default=NEGATIVE_COMMON, help="Негативный промпт, общий для всех зон.")(f)
    return f


def _prompt_set(sd_prompt_paper, sd_prompt_colour, sd_prompt_halftone, sd_prompt_other, sd_negative) -> PromptSet:
    return PromptSet(
        paper=sd_prompt_paper,
        colour=sd_prompt_colour,
        halftone=sd_prompt_halftone,
        other_suffix=sd_prompt_other,
        negative=sd_negative,
    )


def parse_ink_level(value: str) -> "float | None":
    """``--ink-level``: число, ``off`` (не учитывать отражение) или ``auto``.

    ``auto`` — порог Оцу в долях уровня бумаги, посчитанный по самой полосе: он не
    зависит ни от бумаги, ни от экспозиции. На замерах садится около 0.6, то есть
    строже умолчания 0.65 и подтверждает на девять пунктов меньше площади
    пересвеченного текста, — поэтому умолчанием взято фиксированное значение.
    """
    text = value.strip().lower()
    if text == "off":
        return None
    if text == "auto":
        return AUTO_INK_LEVEL
    try:
        return float(text)
    except ValueError:
        raise click.BadParameter(f"ожидается число, 'off' или 'auto', получено {value!r}")


@click.group()
def main() -> None:
    """Очистка пака сканов по разметке из CVAT: закрас размеченного и размытие фона."""


@main.command("run")
@_common_source_options
@_sd_options
@click.option("--out-dir", required=True, type=DIR_OUT, help="Папка результата; структура подпапок зеркалится.")
@click.option(
    "--debug-dir", default=None, type=DIR_OUT, help="Папка debug-оверлеев (всегда JPG). Не задана — не рисуются."
)
@click.option("--inpaint/--no-inpaint", "do_inpaint", default=True, show_default=True, help="Закрашивать разметку.")
@click.option("--smooth/--no-smooth", "do_smooth", default=True, show_default=True, help="Размывать фон.")
@click.option(
    "--backend", default=BACKEND_LAMA, show_default=True, type=click.Choice(BACKENDS), help="Чем закрашивать."
)
@click.option(
    "--group-min-dilate-px",
    default=GROUP_MIN_DILATE_PX,
    show_default=True,
    type=int,
    help=(
        "Нижняя граница припуска ПРИ ГРУППИРОВКЕ, пикс. В сеть маска идёт без припуска: "
        "закрашивается ровно обведённое."
    ),
)
@click.option(
    "--group-dilate-frac",
    default=DEFAULT_GROUP_DILATE_FRAC,
    show_default=True,
    type=float,
    help="Доля своего размера, на которую раздувается связная область при группировке. 0 — не группировать вовсе.",
)
@click.option(
    "--roi-scale",
    default=GROUP_ROI_SCALE,
    show_default=True,
    type=float,
    help="Во сколько раз ROI больше рамки группы.",
)
@click.option(
    "--lama-hole-max-px",
    default=DEFAULT_LAMA_HOLE_MAX_PX,
    show_default=True,
    type=int,
    help="До какого размера ужимается ДЫРА перед LaMa. Крупнее — сеть заливает кляксой; 0 — не ужимать.",
)
@click.option(
    "--method", default=METHOD_SAUVOLA, show_default=True, type=click.Choice(MASK_METHODS, case_sensitive=False)
)
@click.option("--threshold-bias", default=DEFAULT_THRESHOLD_BIAS, show_default=True, type=float)
@click.option("--sauvola-k", default=DEFAULT_SAUVOLA_K, show_default=True, type=float)
@click.option("--sauvola-window", default=None, type=int)
@click.option(
    "--min-glyph-area",
    default=MIN_GLYPH_AREA,
    show_default=True,
    type=int,
    help=(
        "Минимальная площадь связной области первичной маски, пикс. Мельче — остаётся, только "
        "если примыкает к прошедшему глобальный порог или лежит рядом с подтверждённой "
        "областью (см. despeckle). 0 — не чистить вовсе."
    ),
)
@click.option(
    "--paper-dilate-px",
    default=PAPER_DILATE_PX,
    show_default=True,
    type=int,
    help=(
        "Радиус раздутия светлого при оценке уровня бумаги, пикс. Должен перекрывать толщину "
        "штриха. В пикселях, а не долей кадра: в паке встречаются обрезанные страницы, и доля "
        "от их высоты дала бы втрое меньшее окно на том же наборе."
    ),
)
@click.option(
    "--paper-blur-px",
    default=PAPER_BLUR_PX,
    show_default=True,
    type=int,
    help="Сторона окна размытия при оценке уровня бумаги, пикс.: крупнее буквы, мельче неровности света.",
)
@click.option(
    "--sure-glyph-area",
    default=SURE_GLYPH_AREA,
    show_default=True,
    type=int,
    help=(
        "Площадь, с которой область подтверждается САМА ПО СЕБЕ, без оглядки на отражение. "
        "Без неё правило вырождается в «крупная И тёмная», и бледная линейка таблицы в тысячи "
        "пикселей удаляется целиком."
    ),
)
@click.option(
    "--ink-level",
    default=str(INK_LEVEL),
    show_default=True,
    callback=lambda _ctx, _param, value: parse_ink_level(value),
    help=(
        "Доля уровня бумаги, темнее которой связная область считается настоящей краской. "
        "Отражение, а не яркость: на пересвеченной полосе краска бывает светлее, чем просвет "
        "с оборота на обычной. off — не учитывать, auto — порог Оцу по самой полосе."
    ),
)
@click.option(
    "--trust-strong/--no-trust-strong",
    default=False,
    show_default=True,
    help="Подтверждать всё, что прошло глобальный порог (прежнее поведение).",
)
@click.option(
    "--halftone-guard/--no-halftone-guard",
    default=False,
    show_default=True,
    help="Отбрасывать полосу целиком, если на ней НАЙДЕН растр помимо размеченного. "
    "Здесь растр размечен руками, и предохранитель даёт только ложные срабатывания.",
)
@click.option(
    "--dilate-px", default=DEFAULT_DILATE_PX, show_default=True, type=float, help="Радиус защитного припуска, пикс."
)
@click.option("--blur-px", default=None, type=float, help="Радиус размытия, пикс. Не задан — dilate-px x blur-mult.")
@click.option("--blur-mult", default=DEFAULT_BLUR_MULT, show_default=True, type=float)
@click.option("--blur-mode", default=BLUR_MODE_MASKED, show_default=True, type=click.Choice(BLUR_MODES))
@click.option(
    "--protect-stamp-suspect/--no-protect-stamp-suspect",
    default=False,
    show_default=True,
    help="Защищать ли от размытия прямоугольники «подозрение на печать» (разметчик их НЕ подтвердил).",
)
@click.option("--only-with-masks", is_flag=True, help="Только полосы, где есть что закрашивать.")
@click.option("--skip-if-exists/--no-skip-if-exists", default=True, show_default=True)
@click.option("--output-format", default=None, type=click.Choice(["tif", "tiff", "png", "jpg", "jpeg"]))
@click.option("--jobs", default=8, show_default=True, type=int, help="Воркеров на размытие (закрас всегда в родителе).")
@click.option("--report-csv", default=None, type=click.Path(dir_okay=False, path_type=Path))
def run_command(
    db_path,
    pack_dir,
    pack_name,
    only_year,
    only_issue,
    pages_file,
    pages,
    limit,
    log_level,
    sd_model,
    sd_steps,
    sd_guidance,
    sd_size,
    sd_seed,
    sd_prompt_paper,
    sd_prompt_colour,
    sd_prompt_halftone,
    sd_prompt_other,
    sd_negative,
    out_dir,
    debug_dir,
    do_inpaint,
    do_smooth,
    backend,
    group_min_dilate_px,
    group_dilate_frac,
    roi_scale,
    lama_hole_max_px,
    method,
    threshold_bias,
    sauvola_k,
    sauvola_window,
    min_glyph_area,
    ink_level,
    sure_glyph_area,
    paper_dilate_px,
    paper_blur_px,
    trust_strong,
    halftone_guard,
    dilate_px,
    blur_px,
    blur_mult,
    blur_mode,
    protect_stamp_suspect,
    only_with_masks,
    skip_if_exists,
    output_format,
    jobs,
    report_csv,
) -> None:
    """Закрашивает разметку и размывает фон, выдавая пак того же формата и структуры."""
    logging.getLogger().setLevel(log_level.upper())
    params = CleanupParams(
        db_path=db_path,
        pack_name=pack_name or pack_dir.name,
        pack_dir=pack_dir,
        out_dir=out_dir,
        debug_dir=debug_dir,
        do_inpaint=do_inpaint,
        do_smooth=do_smooth,
        only_year=only_year,
        only_issue=only_issue,
        only_with_masks=only_with_masks,
        pages_file=pages_file,
        explicit_pages=tuple(pages),
        limit=limit,
        skip_if_exists=skip_if_exists,
        output_format=output_format,
        jobs=jobs,
        report_csv=report_csv,
        inpaint=InpaintOptions(
            backend=backend,
            group_min_dilate_px=group_min_dilate_px,
            group_dilate_frac=group_dilate_frac,
            roi_scale=roi_scale,
            lama_hole_max_px=lama_hole_max_px,
            sd=SdParams(model=sd_model, steps=sd_steps, guidance=sd_guidance, size=sd_size, seed=sd_seed),
            prompts=_prompt_set(sd_prompt_paper, sd_prompt_colour, sd_prompt_halftone, sd_prompt_other, sd_negative),
        ),
        smooth=SmoothOptions(
            method=method.lower(),
            threshold_bias=threshold_bias,
            sauvola_k=sauvola_k,
            sauvola_window=sauvola_window,
            min_glyph_area=min_glyph_area,
            ink_level=ink_level,
            sure_glyph_area=sure_glyph_area,
            paper_dilate_px=paper_dilate_px,
            paper_blur_px=paper_blur_px,
            trust_strong=trust_strong,
            halftone_guard=halftone_guard,
            dilate_px=dilate_px,
            blur_px=blur_px,
            blur_mult=blur_mult,
            blur_mode=blur_mode,
        ),
        protect=ProtectOptions(protect_stamp_suspect=protect_stamp_suspect),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    reports = run_cleanup(params)
    click.echo(summary(reports))
    click.echo(f"Результат → {out_dir}" + (f", оверлеи → {debug_dir}" if debug_dir else ""))


@main.command("compare-masks")
@_common_source_options
@click.option("--out-dir", required=True, type=DIR_OUT, help="Куда складывать сравнение.")
@click.option("--method", "methods", multiple=True, default=("otsu", "sauvola"), show_default=True)
@click.option("--sauvola-k", "sauvola_ks", multiple=True, type=float, default=(DEFAULT_SAUVOLA_K,), show_default=True)
@click.option("--dilate-px", "dilate_pxs", multiple=True, type=float, default=(15.0, 25.0, 35.0), show_default=True)
@click.option("--blur-px", "blur_pxs", multiple=True, type=float, default=(60.0, 120.0, 240.0), show_default=True)
@click.option(
    "--ink-level",
    "ink_levels",
    multiple=True,
    default=(str(INK_LEVEL),),
    show_default=True,
    help="Порог отражения; можно повторять. off — не учитывать, auto — по самой полосе.",
)
@click.option(
    "--show-removed", is_flag=True, help="Класть врезки 1:1 с тем, что вариант УБРАЛ из маски первого варианта."
)
@click.option(
    "--sample", default=0, type=int, help="Взять столько полос случайно (детерминированно), если не задан --page."
)
@click.option("--crop-side", default=1200, show_default=True, type=int, help="Сторона врезки 1:1, пикс.")
@click.option("--crops", default=3, show_default=True, type=int, help="Сколько врезок на полосу.")
@click.option(
    "--montage-cols",
    default=4,
    show_default=True,
    type=int,
    help="Столбцов в сетке врезок. 0 — выложить всё одним рядом (при десятке вариантов он нечитаем).",
)
@click.option("--jobs", default=8, show_default=True, type=int)
def compare_masks_command(
    db_path,
    pack_dir,
    pack_name,
    only_year,
    only_issue,
    pages_file,
    pages,
    limit,
    log_level,
    out_dir,
    methods,
    sauvola_ks,
    dilate_pxs,
    blur_pxs,
    ink_levels,
    show_removed,
    sample,
    crop_side,
    crops,
    montage_cols,
    jobs,
) -> None:
    """Строит защитные маски всеми сочетаниями метод × k × припуск × размытие и раскладывает их рядом."""
    from ocr_utils.scan_cleanup.compare import CompareMasksParams, run_compare_masks

    logging.getLogger().setLevel(log_level.upper())
    params = CompareMasksParams(
        db_path=db_path,
        pack_name=pack_name or pack_dir.name,
        pack_dir=pack_dir,
        out_dir=out_dir,
        only_year=only_year,
        only_issue=only_issue,
        pages_file=pages_file,
        explicit_pages=tuple(pages),
        limit=limit,
        sample=sample,
        methods=tuple(m.lower() for m in methods),
        sauvola_ks=tuple(sauvola_ks),
        dilate_pxs=tuple(dilate_pxs),
        blur_pxs=tuple(blur_pxs),
        ink_levels=tuple(parse_ink_level(v) for v in ink_levels),
        show_removed=show_removed,
        crop_side=crop_side,
        crops=crops,
        montage_cols=montage_cols,
        jobs=jobs,
    )
    click.echo(run_compare_masks(params))


@main.command("compare-inpaint")
@_common_source_options
@_sd_options
@click.option("--out-dir", required=True, type=DIR_OUT, help="Куда складывать сравнение.")
@click.option("--backend", "backends", multiple=True, default=BACKENDS, show_default=True)
@click.option("--kind", "kinds", multiple=True, default=(), help="Виды разметки; по умолчанию все три.")
@click.option("--sample", default=30, show_default=True, type=int, help="Сколько полос с масками взять.")
@click.option(
    "--group-min-dilate-px",
    default=GROUP_MIN_DILATE_PX,
    show_default=True,
    type=int,
    help=(
        "Нижняя граница припуска ПРИ ГРУППИРОВКЕ, пикс. В сеть маска идёт без припуска: "
        "закрашивается ровно обведённое."
    ),
)
@click.option(
    "--group-dilate-frac",
    "group_dilate_fracs",
    multiple=True,
    type=float,
    default=(DEFAULT_GROUP_DILATE_FRAC,),
    show_default=True,
    help="Можно повторять; 0 — контрольный вариант «без группировки».",
)
@click.option("--roi-scale", "roi_scales", multiple=True, type=float, default=(GROUP_ROI_SCALE,), show_default=True)
@click.option(
    "--lama-hole-max-px", "lama_holes", multiple=True, type=int, default=(DEFAULT_LAMA_HOLE_MAX_PX,), show_default=True
)
@click.option("--montage-cols", default=4, show_default=True, type=int, help="Столбцов в сетке вариантов.")
def compare_inpaint_command(
    db_path,
    pack_dir,
    pack_name,
    only_year,
    only_issue,
    pages_file,
    pages,
    limit,
    log_level,
    sd_model,
    sd_steps,
    sd_guidance,
    sd_size,
    sd_seed,
    sd_prompt_paper,
    sd_prompt_colour,
    sd_prompt_halftone,
    sd_prompt_other,
    sd_negative,
    out_dir,
    backends,
    kinds,
    sample,
    group_dilate_fracs,
    roi_scales,
    lama_holes,
    group_min_dilate_px,
    montage_cols,
) -> None:
    """Закрашивает выборку полос всеми бэкендами и вариантами и складывает результаты рядом."""
    from ocr_utils.scan_cleanup.compare import CompareInpaintParams, run_compare_inpaint

    logging.getLogger().setLevel(log_level.upper())
    params = CompareInpaintParams(
        db_path=db_path,
        pack_name=pack_name or pack_dir.name,
        pack_dir=pack_dir,
        out_dir=out_dir,
        only_year=only_year,
        only_issue=only_issue,
        pages_file=pages_file,
        explicit_pages=tuple(pages),
        limit=limit,
        sample=sample,
        backends=tuple(backends),
        kinds=tuple(kinds),
        group_dilate_fracs=tuple(group_dilate_fracs),
        roi_scales=tuple(roi_scales),
        lama_holes=tuple(lama_holes),
        group_min_dilate_px=group_min_dilate_px,
        montage_cols=montage_cols,
        sd=SdParams(model=sd_model, steps=sd_steps, guidance=sd_guidance, size=sd_size, seed=sd_seed),
        prompts=_prompt_set(sd_prompt_paper, sd_prompt_colour, sd_prompt_halftone, sd_prompt_other, sd_negative),
    )
    click.echo(run_compare_inpaint(params))


if __name__ == "__main__":
    main()
