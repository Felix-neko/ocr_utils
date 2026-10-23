"""CLI стенда «выпрямление страниц по подгоночным областям»: итог прогона scan_cropping v2 → кадр, где криволинейная область каждой страницы стала прямоугольником (``run``); то же плюс выравнивание освещения и второе преобразование по тексту (``text-run``); перекладка прогона в общую раскладку (``relayout``)."""

from __future__ import annotations

import csv
import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.scan_cropping.image_io import collect_images, imwrite_params, read_dpi, write_image
from ocr_utils.scan_cropping.v2.pipeline import FrameJob, init_worker, read_bgr, visible_frames
from ocr_utils.scan_cropping.v2.spread_lines import SpineMethod
from ocr_utils.units import mm_to_px_float, px_to_mm
from research.page_rectify.coons import TANGENT_MM, page_patch, plan_layout, rectify_image, rectify_maps, sagitta
from research.page_rectify.frames import CachedFrames, cached_frames
from research.page_rectify.overlay import draw_sheet
from research.page_rectify.relayout import apply_moves, count_files, plan_moves
from research.page_rectify.text_field import ACROSS_SMOOTH_MM, FIELD_SMOOTH_MM, FIELD_STEP_MM, GUTTER_SMOOTH_MM
from research.page_rectify.text_overlay import sheet_analysis, sheet_field, sheet_illumination, sheet_result
from research.page_rectify.text_run import TextParams, text_frame

log = logging.getLogger(__name__)

# Воркеров по умолчанию: кадр читает оригинал с Я.Диска (NTFS-3G) и кэш v2, держит ~1 ГБ карт и копий — упор в диск
# и память раньше, чем в 16 ядер.
DEFAULT_JOBS = 12
# Суффикс итогов v2 (``--output-format tiff`` боевых ключей).
V2_SUFFIX = ".tiff"


class SheetStep(StrEnum):
    """Листы второго преобразования: подпапка ``debug/<шаг>/<набор>/``."""

    ILLUMINATION = "01_illumination"
    ANALYSIS = "02_analysis"
    FIELD = "03_field"
    RESULT = "04_result"


class Status(StrEnum):
    """Исход кадра."""

    OK = "ok"  # выпрямлен
    MISSING = "missing"  # нет итога v2
    SKIPPED = "skipped"  # итог выпрямления уже есть (``--skip-if-exists``)
    NO_PAPER = "no_paper"  # бумага не найдена — итог v2 не вырезан, копируется как есть
    NO_FRAMES = "no_frames"  # не две страницы или области не построились — копируется как есть
    CACHE_MISMATCH = "cache_mismatch"  # рамка вырезки по кэшу не совпала с размером итога — кэш перезаписан, копия
    ERROR = "error"  # исключение — копируется как есть


@dataclass(frozen=True)
class RunConfig:
    """Параметры прогона.

    Поля:
        input_dir: Папка исходных кадров прогона v2.
        cache: Папка кэша прогона v2.
        v2_output: Папка итогов прогона v2.
        output: Папка выпрямленных итогов.
        debug: Папка листов «было | стало» или None.
        bg_veto: Прогон v2 шёл с вето фона.
        spine_method: Способ разделителя страниц прогона v2.
        tangent_mm: Отрезок у конца стороны для касательной продолжения, мм бумаги.
        skip_if_exists: Пропускать кадры с готовым итогом.
        label: Подпись набора в шапке листа.
    """

    input_dir: Path
    cache: Path
    v2_output: Path
    output: Path
    debug: Path | None
    bg_veto: bool
    spine_method: SpineMethod
    tangent_mm: float
    skip_if_exists: bool
    label: str


def copy_as_is(src: Path, dst: Path) -> None:
    """Кладёт итог v2 в выход без правки (кадр не выпрямляется)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def rectify_frame(args: tuple[Path, RunConfig, TextParams | None]) -> dict:
    """Выпрямление одного кадра (для пула): только Кунс или Кунс плюс поле по тексту.

    Args:
        args: ``(исходный кадр, параметры прогона, параметры второго преобразования или None — только Кунс)``.

    Returns:
        Строка сводки.
    """
    src, cfg, text = args
    job = FrameJob(src, src.relative_to(cfg.input_dir))
    v2_path = cfg.v2_output / job.rel.with_suffix(V2_SUFFIX)
    dst = cfg.output / job.rel.with_suffix(V2_SUFFIX)
    row: dict = {"key": job.key, "status": Status.OK.value}
    t0 = time.perf_counter()
    if not v2_path.exists():
        return row | {"status": Status.MISSING.value}
    if cfg.skip_if_exists and dst.exists():
        return row | {"status": Status.SKIPPED.value}
    try:
        status = (
            rectify_one(job, cfg, v2_path, dst, row) if text is None else text_one(job, cfg, text, v2_path, dst, row)
        )
    except Exception as exc:  # кадр с ошибкой не роняет прогон
        log.exception("%s: ошибка выпрямления", job.key)
        copy_as_is(v2_path, dst)
        status, row["message"] = Status.ERROR, repr(exc)
    row["status"] = status.value
    row["seconds"] = round(time.perf_counter() - t0, 2)
    return row


def load_frame(job: FrameJob, cfg: RunConfig, v2_path: Path, dst: Path, row: dict):
    """Итог v2, его dpi и области страниц по кэшу; кадр, который выпрямлять нельзя, копируется как есть.

    Args:
        job: Кадр.
        cfg: Параметры прогона.
        v2_path: Итог v2.
        dst: Путь выпрямленного итога.
        row: Строка сводки (сюда — пояснение расхождения кэша).

    Returns:
        ``(итог, dpi, области)`` или исход кадра, если выпрямлять нельзя (итог уже скопирован).
    """
    image = read_bgr(v2_path)
    dpi = read_dpi(v2_path)
    if dpi is None:
        raise ValueError(f"в итоге v2 нет тега dpi: {v2_path}")
    cf = cached_frames(job.src, cfg.cache, job.key, cfg.bg_veto, cfg.spine_method)
    status = None
    if cf.box is None:
        status = Status.NO_PAPER
    elif (cf.box[3] - cf.box[1], cf.box[2] - cf.box[0]) != image.shape[:2]:
        # Кэш кадра перезаписан другим прогоном — области не от этого итога.
        x0, y0, x1, y1 = cf.box
        row["message"] = f"рамка {x1 - x0}×{y1 - y0}, итог {image.shape[1]}×{image.shape[0]}"
        status = Status.CACHE_MISMATCH
    elif cf.frames is None:
        status = Status.NO_FRAMES
    if status is not None:
        copy_as_is(v2_path, dst)
        return status
    return image, dpi, cf


def rectify_one(job: FrameJob, cfg: RunConfig, v2_path: Path, dst: Path, row: dict) -> Status:
    """Тело :func:`rectify_frame` (только Кунс): области по кэшу, карта, запись итога и листа; меры — в ``row``.

    Args:
        job: Кадр.
        cfg: Параметры прогона.
        v2_path: Итог v2.
        dst: Путь выпрямленного итога.
        row: Строка сводки (дополняется мерами).

    Returns:
        Исход кадра.
    """
    loaded = load_frame(job, cfg, v2_path, dst, row)
    if isinstance(loaded, Status):
        return loaded
    image, dpi, cf = loaded
    tangent_px = mm_to_px_float(cfg.tangent_mm, dpi)
    patches = (page_patch(cf.frames[0], True, cf.to_output, tangent_px),
               page_patch(cf.frames[1], False, cf.to_output, tangent_px))  # fmt: skip
    layout = plan_layout(patches, image.shape[:2])
    out = rectify_image(image, rectify_maps(patches, layout), dpi)
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_image(dst, out, imwrite_params(dst.suffix), dpi)
    row.update(
        width=out.shape[1],
        height=out.shape[0],
        v2_width=image.shape[1],
        v2_height=image.shape[0],
        left_w_mm=mm1(layout.rects[0].width, dpi),
        left_h_mm=mm1(layout.rects[0].height, dpi),
        right_w_mm=mm1(layout.rects[1].width, dpi),
        right_h_mm=mm1(layout.rects[1].height, dpi),
        spine_dh_mm=mm1(abs(layout.rects[0].height - layout.rects[1].height), dpi),
        top_dy_mm=mm1(abs(layout.rects[0].y0 - layout.rects[1].y0), dpi),
        bend_mm=mm1(max(sagitta(c) for p in patches for c in (p.top, p.bottom, p.left, p.right)), dpi),
    )
    if cfg.debug is not None:
        bends = [
            "/".join(f"{px_to_mm(sagitta(c), dpi):.1f}" for c in (p.top, p.bottom, p.left if i == 0 else p.right))
            for i, p in enumerate(patches)
        ]
        header = [
            f"{cfg.label}/{job.rel} — выпрямление по подгоночным областям страниц (поверхность Кунса)",
            f"было: итог v2 {image.shape[1]}×{image.shape[0]} px | стало: {out.shape[1]}×{out.shape[0]} px, {dpi} dpi",
            f"левая {row['left_w_mm']}×{row['left_h_mm']} мм, правая {row['right_w_mm']}×{row['right_h_mm']} мм; "
            f"высоты у корешка разнятся на {row['spine_dh_mm']} мм; изгиб верх/низ/внешняя, мм: "
            f"левая {bends[0]}, правая {bends[1]}",
        ]
        sheet = draw_sheet(image, out, patches, layout, header)
        sheet_path = cfg.debug / job.rel.with_suffix(".jpg")
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Status.OK


def text_one(job: FrameJob, cfg: RunConfig, text: TextParams, v2_path: Path, dst: Path, row: dict) -> Status:
    """Тело :func:`rectify_frame` со вторым преобразованием: освещение, Кунс, поле по тексту, итог, листы 01–05.

    Args:
        job: Кадр.
        cfg: Параметры прогона (``debug`` — корень листов: ``debug/<шаг>/<label>/<кадр>.jpg``).
        text: Параметры второго преобразования.
        v2_path: Итог v2.
        dst: Путь итога.
        row: Строка сводки (дополняется мерами).

    Returns:
        Исход кадра.
    """
    loaded = load_frame(job, cfg, v2_path, dst, row)
    if isinstance(loaded, Status):
        return loaded
    image, dpi, cf = loaded
    result = text_frame(image, cf, dpi, text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_image(dst, result.final, imwrite_params(dst.suffix), dpi)
    dx, dy = result.dense
    row.update(
        width=result.final.shape[1],
        height=result.final.shape[0],
        colours="/".join(p.colour.value for p in result.pages),
        lines="/".join(str(p.lines) for p in result.pages),
        fielded="/".join(p.mode.value for p in result.pages),
        dx_max_mm=round(px_to_mm(float(np.abs(dx).max()), dpi), 2),
        dy_max_mm=round(px_to_mm(float(np.abs(dy).max()), dpi), 2),
        clipped=int(result.field.clipped),
        **result.before.row("before_"),
        **result.after.row("after_"),
    )
    if cfg.debug is not None:
        header = [f"{cfg.label}/{job.rel} — освещение, Кунс и поле по тексту, {dpi} dpi"]
        sheets = {
            SheetStep.ILLUMINATION: sheet_illumination(result, header),
            SheetStep.ANALYSIS: sheet_analysis(result, header),
            SheetStep.FIELD: sheet_field(result, header, dpi),
            SheetStep.RESULT: sheet_result(result, header),
        }
        for step, sheet in sheets.items():
            path = cfg.debug / step.value / cfg.label / job.rel.with_suffix(".jpg")
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Status.OK


def mm1(px: float, dpi: float) -> float:
    """Пиксели → мм бумаги с одним знаком (меры сводки)."""
    return round(px_to_mm(px, dpi), 1)


def keep_previous(rows: list[dict], summary: Path) -> list[dict]:
    """Строки пропущенных кадров (``skipped``) — из прежней сводки, если она есть: перезапуск её не затирает.

    Args:
        rows: Строки этого прогона.
        summary: CSV прежней сводки.

    Returns:
        Строки с подставленными прежними для пропущенных кадров.
    """
    if not summary.exists():
        return rows
    with summary.open(newline="") as fh:
        previous = {r["key"]: r for r in csv.DictReader(fh)}
    return [previous.get(r["key"], r) if r["status"] == Status.SKIPPED.value else r for r in rows]


def even_sample(files: list[Path], n: int) -> list[Path]:
    """``n`` кадров, равномерно по отсортированному списку (по выпускам и номерам), без случайности: середины ``n``
    равных долей списка — первый и последний кадр (обычно обложки) в выборку не попадают.

    Args:
        files: Кадры.
        n: Сколько взять; 0 или больше длины — все.

    Returns:
        Выборка в исходном порядке.
    """
    files = sorted(files)
    if n <= 0 or n >= len(files):
        return files
    return [files[i] for i in np.unique(((np.arange(n) + 0.5) * len(files) / n).astype(int))]


def run_pool(sets: list[tuple[RunConfig, Path, int]], text: TextParams | None, recursive: bool, jobs: int) -> None:
    """Прогон кадров всех наборов в одном пуле и запись сводки каждого набора (пропущенные кадры берут строки прежней).

    Один общий пул: при поочерёдных наборах по 2 кадра (выборка ХиП — по выпуску) из 12 воркеров работали 2, а в
    полном прогоне пул простаивал на хвосте каждого выпуска.

    Args:
        sets: Наборы: параметры прогона, CSV сводки и сколько кадров взять равномерно (0 — все).
        text: Параметры второго преобразования или None — только Кунс.
        recursive: Кадры во вложенных папках.
        jobs: Воркеров пула.
    """
    tasks, owners = [], []
    for index, (cfg, _, sample) in enumerate(sets):
        files = even_sample(visible_frames(collect_images(cfg.input_dir, recursive), cfg.input_dir), sample)
        log.info("%s: кадров %d → %s", cfg.label, len(files), cfg.output)
        tasks += [(f, cfg, text) for f in files]
        owners += [index] * len(files)
    log.info("всего кадров: %d, наборов: %d, воркеров: %d", len(tasks), len(sets), jobs)
    rows: list[list[dict]] = [[] for _ in sets]
    ctx = get_context("forkserver")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=init_worker) as pool:
        for i, (owner, row) in enumerate(zip(owners, pool.map(rectify_frame, tasks)), 1):
            rows[owner].append(row)
            if i % 25 == 0 or i == len(tasks):
                log.info("%d/%d", i, len(tasks))
    for (cfg, summary, _), set_rows in zip(sets, rows):
        write_summary(summary, set_rows, cfg.label)


def write_summary(summary: Path, rows: list[dict], label: str) -> None:
    """CSV сводки набора (строки пропущенных кадров — из прежней сводки) и итог в лог.

    Args:
        summary: Путь CSV.
        rows: Строки кадров набора.
        label: Подпись набора для лога.
    """
    summary.parent.mkdir(parents=True, exist_ok=True)
    rows = keep_previous(rows, summary)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with summary.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fields)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    log.info("%s готово: %d кадров; %s; сводка %s", label, len(rows),
             ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), summary)  # fmt: skip


@click.group()
def main() -> None:
    """Стенд выпрямления страниц по подгоночным областям scan_cropping v2."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@main.command()
@click.option("--input-dir", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True,
              help="Исходные кадры прогона v2 (ключи кэша — от этой папки).")  # fmt: skip
@click.option("--cache", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True,
              help="Кэш прогона v2 (--cache).")  # fmt: skip
@click.option("--v2-output", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True,
              help="Итоги прогона v2 (--output).")  # fmt: skip
@click.option("--output", type=click.Path(path_type=Path), required=True, help="Папка выпрямленных итогов.")
@click.option("--debug", type=click.Path(path_type=Path), default=None, help="Папка листов «было | стало».")
@click.option("--summary", type=click.Path(path_type=Path), default=None,
              help="CSV сводки (по умолчанию — rectify.csv рядом с --output).")  # fmt: skip
@click.option("--label", default="", help="Подпись набора в шапке листа.")
@click.option("--sample", type=int, default=0, show_default=True,
              help="Взять столько кадров, равномерно по выпускам; 0 — все.")  # fmt: skip
@click.option("--bg-veto/--no-bg-veto", default=True, show_default=True, help="Прогон v2 шёл с вето фона.")
@click.option("--spine-method", type=click.Choice([m.value for m in SpineMethod if m is not SpineMethod.FOLD]),
              default=SpineMethod.AUTO.value, show_default=True, help="Разделитель страниц прогона v2.")  # fmt: skip
@click.option("--tangent-mm", type=float, default=TANGENT_MM, show_default=True,
              help="Отрезок у конца стороны для касательной продолжения, мм бумаги.")  # fmt: skip
@click.option("--recursive/--no-recursive", default=True, show_default=True, help="Кадры во вложенных папках.")
@click.option("--skip-if-exists", is_flag=True, help="Пропускать кадры с готовым итогом.")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def run(
    input_dir: Path,
    cache: Path,
    v2_output: Path,
    output: Path,
    debug: Path | None,
    summary: Path | None,
    label: str,
    sample: int,
    bg_veto: bool,
    spine_method: str,
    tangent_mm: float,
    recursive: bool,
    skip_if_exists: bool,
    jobs: int,
) -> None:
    """Выпрямить итоги прогона v2: область каждой страницы → прямоугольник, поля — продолжением карты."""
    cfg = RunConfig(input_dir, cache, v2_output, output, debug, bg_veto, SpineMethod(spine_method), tangent_mm,
                    skip_if_exists, label or input_dir.name)  # fmt: skip
    run_pool([(cfg, summary or output.parent / "rectify.csv", sample)], None, recursive, jobs)


@main.command("text-run")
@click.option("--set", "sets", type=(str, click.Path(path_type=Path, exists=True, file_okay=False),
              click.Path(path_type=Path, exists=True, file_okay=False), click.Path(path_type=Path, exists=True, file_okay=False), int),
              multiple=True, required=True,
              help="Набор: ПОДПИСЬ ВХОД КЭШ ИТОГИ_V2 N (подпись — подпуть в output/, debug/<шаг>/, summary/; N — сколько "
                   "кадров взять равномерно по выпускам, 0 — все); повторяется, все кадры всех наборов — в одном пуле.")  # fmt: skip
@click.option("--root", type=click.Path(path_type=Path), required=True,
              help="Корень прогона: output/<набор>/, debug/<шаг>/<набор>/, summary/<набор>.csv.")  # fmt: skip
@click.option("--debug/--no-debug", default=True, show_default=True, help="Рисовать листы 01–04.")
@click.option("--bg-veto/--no-bg-veto", default=True, show_default=True, help="Прогон v2 шёл с вето фона.")
@click.option("--spine-method", type=click.Choice([m.value for m in SpineMethod if m is not SpineMethod.FOLD]),
              default=SpineMethod.AUTO.value, show_default=True, help="Разделитель страниц прогона v2.")  # fmt: skip
@click.option("--tangent-mm", type=float, default=TANGENT_MM, show_default=True,
              help="Отрезок у конца стороны для касательной продолжения, мм бумаги.")  # fmt: skip
@click.option("--recursive/--no-recursive", default=True, show_default=True, help="Кадры во вложенных папках.")
@click.option("--skip-if-exists", is_flag=True, help="Пропускать кадры с готовым итогом.")
@click.option("--field-step-mm", type=float, default=FIELD_STEP_MM, show_default=True, help="Шаг узлов поля, мм.")
@click.option("--field-smooth-mm", type=float, default=FIELD_SMOOTH_MM, show_default=True,
              help="Длина изгиба поля, мм: короче — поле гибче.")  # fmt: skip
@click.option("--gutter-smooth-mm", type=float, default=GUTTER_SMOOTH_MM, show_default=True,
              help="Длина изгиба поля у корешка, мм (лист уходит в сгиб — строки загибаются круче).")  # fmt: skip
@click.option("--across-smooth-mm", type=float, default=ACROSS_SMOOTH_MM, show_default=True,
              help="Длина изгиба поля поперёк строк, мм: поле не меняется от строки к строке.")  # fmt: skip
@click.option("--illumination/--no-illumination", default=True, show_default=True,
              help="Выравнивать освещение не-цветных страниц (как Equalize illumination в ScanTailor).")  # fmt: skip
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def text_run(
    sets: tuple[tuple[str, Path, Path, Path, int], ...],
    root: Path,
    debug: bool,
    bg_veto: bool,
    spine_method: str,
    tangent_mm: float,
    recursive: bool,
    skip_if_exists: bool,
    field_step_mm: float,
    field_smooth_mm: float,
    gutter_smooth_mm: float,
    across_smooth_mm: float,
    illumination: bool,
    jobs: int,
) -> None:
    """Выпрямить итоги прогона v2 Кунсом и полем по тексту (оси строк, выровненные стороны блоков), с освещением."""
    configs = [
        (RunConfig(input_dir, cache, v2_output, root / "output" / label, root / "debug" if debug else None, bg_veto,
                   SpineMethod(spine_method), tangent_mm, skip_if_exists, label),
         root / "summary" / f"{label}.csv", sample)
        for label, input_dir, cache, v2_output, sample in sets
    ]  # fmt: skip
    text = TextParams(tangent_mm, field_step_mm, field_smooth_mm, gutter_smooth_mm, across_smooth_mm, illumination)
    run_pool(configs, text, recursive, jobs)


@main.command()
@click.option("--root", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True,
              help="Корень прогона прежней раскладки.")  # fmt: skip
@click.option("--dry-run", is_flag=True, help="Только показать переносы.")
def relayout(root: Path, dry_run: bool) -> None:
    """Переложить прогон в общую раскладку: output/<набор>, debug/<набор>, summary/<набор>.csv."""
    before = count_files(root)
    moves = plan_moves(root)
    for move in moves:
        log.info("%s → %s", move.src.relative_to(root), move.dst.relative_to(root))
    if dry_run:
        log.info("переносов: %d (dry-run), файлов: %d", len(moves), before)
        return
    removed = apply_moves(root, moves)
    after = count_files(root)
    log.info("перенесено: %d, удалено пустых папок: %d, файлов до/после: %d/%d", len(moves), removed, before, after)
    if before != after:
        raise click.ClickException("число файлов изменилось")
