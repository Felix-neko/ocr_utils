"""CLI стенда «нарезка сканов»: меры кадров прогонов v2 и кандидаты в тестовую коллекцию по группам (``select``), листы отсмотра наборов и кандидатов (``sheets``)."""

from __future__ import annotations

import csv
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from enum import Enum
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import header_strip
from ocr_utils.scan_cropping.v2.overlay import OverlayTag, overlay_path
from ocr_utils.scan_cropping.v2.pipeline import init_worker
from ocr_utils.scan_cropping.v2.page_boundary import FitKind
from research.page_cutting.bends import BendRow, draw_bends, frame_bends
from research.page_cutting.cutting import CutRow, cut_frame
from research.page_cutting.scantailor import default_profile_xml
from research.page_cutting.fitting import PageRow, process_frame
from research.page_cutting.runs import RunSpec, SetRow, parse_run, read_image, read_set, rel_of, run_keys, write_image
from research.page_cutting.selection import FrameMeasures, frame_measures

log = logging.getLogger(__name__)

# Ширина миниатюры кадра на листе, px: оверлей v2 «кадр | итог» — по ~500 px на половину (как превью в ScanTailor).
THUMB_WIDTH = 1000
# Миниатюр в ряду и рядов на листе.
SHEET_COLS = 2
SHEET_ROWS = 4
# Высота нижней полосы оверлея v2 с легендой, доля высоты картинки: на листе легенда не нужна, срезается.
DEBUG_LEGEND_SHARE = 0.17


class Group(str, Enum):
    """Группа тестовой коллекции: по какой мере отбираются кандидаты."""

    # Рваный или волнистый край маски SAM: рваность и изгиб дальней стороны.
    RAGGED = "ragged"
    # Сильный изгиб сторон страницы: верх и низ, сторона у корешка.
    CURVED = "curved"
    # Текст вплотную к стороне у корешка: наименьший зазор.
    TIGHT_GUTTER = "tight_gutter"
    # Цветные страницы (обложки, форзацы): цветность центра.
    COLOUR = "colour"
    # Приподнятая страница (дубль разворота): ``raised`` в ``pages.json``.
    RAISED = "raised"
    # Служебные кадры: без текстовых блоков на странице.
    SERVICE = "service"
    # Контроль: плоские страницы с ровными краями и полями у корешка.
    FLAT = "flat"


def measure_task(args: tuple[RunSpec, str]) -> FrameMeasures | None:
    """Меры одного кадра для пула; сбой кадра — в лог и None."""
    run, key = args
    try:
        return frame_measures(run, key)
    except Exception as exc:  # noqa: BLE001 — один битый кэш не роняет обход
        log.warning("%s %s: %s", run.name, key, exc)
        return None


def group_score(m: FrameMeasures, group: Group) -> float | None:
    """Мера кадра для группы: больше — сильнее признак; None — кадр в группу не идёт.

    Args:
        m: Меры кадра.
        group: Группа.

    Returns:
        Мера или None.
    """
    if group is Group.RAGGED:
        return m.ragged_mm * 3 + m.bend_far_mm
    if group is Group.CURVED:
        return max(m.bend_tb_mm, m.bend_spine_mm) if m.blocks > 0 else None
    if group is Group.TIGHT_GUTTER:
        return None if math.isnan(m.gutter_gap_mm) else -m.gutter_gap_mm
    if group is Group.COLOUR:
        return m.chroma
    if group is Group.RAISED:
        return m.keystone if "raised" in m.poses else None
    if group is Group.SERVICE:
        return 1.0 if m.blocks == 0 else None
    # Контроль: ровные стороны, текст не у корешка, две плоские страницы; меньше «дефекта» — выше.
    if m.pages != 2 or "raised" in m.poses or m.blocks == 0 or math.isnan(m.gutter_gap_mm):
        return None
    return -(m.ragged_mm * 3 + m.bend_far_mm + m.bend_tb_mm + m.bend_spine_mm)


@click.group()
def main() -> None:
    """Стенд «нарезка сканов»: криволинейная рамка страницы, выпрямление, ScanTailor CLI, судья dewarp."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


RUN_HELP = "Прогон v2: «имя=папка,подпапка,px_per_mm» (подпапка — внутри output/cache/debug, может быть пустой)."


@main.command("select")
@click.option("--run", "runs", multiple=True, required=True, help=RUN_HELP)
@click.option(
    "--out-dir", type=click.Path(path_type=Path), required=True, help="Куда писать measures.csv и кандидатов."
)
@click.option("--per-run", type=int, default=8, show_default=True, help="Кандидатов на группу от каждого прогона.")
@click.option("--jobs", type=int, default=16, show_default=True, help="Воркеров пула (упор — чтение кэша с SSD).")
def select_cmd(runs: tuple[str, ...], out_dir: Path, per_run: int, jobs: int) -> None:
    """Меры всех кадров прогонов (``measures.csv``) и кандидаты по группам (``candidates/<группа>.txt``)."""
    specs = [parse_run(r) for r in runs]
    tasks = [(spec, key) for spec in specs for key in run_keys(spec)]
    log.info("кадров: %d в %d прогонах", len(tasks), len(specs))
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [m for m in pool.map(measure_task, tasks, chunksize=8) if m is not None]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "measures.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, [fl.name for fl in fields(FrameMeasures)])
        writer.writeheader()
        writer.writerows(asdict(m) for m in rows)
    log.info("мер: %d → %s", len(rows), out_dir / "measures.csv")
    for group in Group:
        lines = [f"# кандидаты группы {group.value}: по {per_run} лучших от прогона (мера группы — в пометке)"]
        for spec in specs:
            scored = [(group_score(m, group), m) for m in rows if m.run == spec.name]
            scored = sorted([s for s in scored if s[0] is not None], key=lambda s: -s[0])[:per_run]
            for score, m in scored:
                lines.append(f"{m.run} {m.key}  # {group.value}={score:.2f} {describe(m)}")
        path = out_dir / "candidates" / f"{group.value}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        log.info("%s: %d кандидатов", group.value, len(lines) - 1)


def describe(m: FrameMeasures) -> str:
    """Меры кадра одной строкой для пометки в наборе и подписи на листе."""
    return (
        f"рв {m.ragged_mm:.1f} ({m.ragged_side}) изг в/н {m.bend_tb_mm:.1f} кор {m.bend_spine_mm:.1f} "
        f"даль {m.bend_far_mm:.1f} мм; зазор {m.gutter_gap_mm:.1f} мм; сход {m.keystone:.1f}°; "
        f"цв {m.chroma:.0f}; блоков {m.blocks}; {m.poses}"
    )


def debug_thumb(spec: RunSpec, row: SetRow) -> np.ndarray:
    """Миниатюра оверлея v2 кадра («кадр с разметкой | итог») без нижней легенды, с подписью-шапкой.

    Args:
        spec: Прогон кадра.
        row: Строка набора.

    Returns:
        Картинка BGR шириной ``THUMB_WIDTH``.
    """
    path = overlay_path(spec.debug_dir, rel_of(row.key), OverlayTag.MAIN)
    img = read_image(path)
    img = img[: int(img.shape[0] * (1 - DEBUG_LEGEND_SHARE))]
    scale = THUMB_WIDTH / img.shape[1]
    img = cv2.resize(img, (THUMB_WIDTH, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    head = header_strip([f"{row.run} {row.key}", row.note[:150]], THUMB_WIDTH)
    return np.vstack([head, img])


def sheet_of(thumbs: list[np.ndarray], cols: int) -> np.ndarray:
    """Лист миниатюр сеткой ``cols`` в ряд; ряды выравниваются по высоте белым полем.

    Args:
        thumbs: Миниатюры одной ширины.
        cols: Миниатюр в ряду.

    Returns:
        Лист BGR.
    """
    pad = 12
    rows = []
    for i in range(0, len(thumbs), cols):
        chunk = thumbs[i : i + cols]
        h = max(t.shape[0] for t in chunk)
        cells = [np.pad(t, ((0, h - t.shape[0]), (0, pad), (0, 0)), constant_values=255) for t in chunk]
        while len(cells) < cols:
            cells.append(np.full((h, THUMB_WIDTH + pad, 3), 255, np.uint8))
        rows.append(np.pad(np.hstack(cells), ((0, pad), (0, 0), (0, 0)), constant_values=255))
    return np.vstack(rows)


@main.command("sheets")
@click.option("--run", "runs", multiple=True, required=True, help=RUN_HELP)
@click.option(
    "--set",
    "set_files",
    multiple=True,
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Набор или файл кандидатов (строка «прогон ключ  # пометка»).",
)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Куда писать листы.")
def sheets_cmd(runs: tuple[str, ...], set_files: tuple[Path, ...], out_dir: Path) -> None:
    """Листы отсмотра: оверлеи v2 кадров набора по ``SHEET_COLS × SHEET_ROWS`` на лист, с мерами в подписи."""
    specs = {s.name: s for s in (parse_run(r) for r in runs)}
    per_sheet = SHEET_COLS * SHEET_ROWS
    for set_file in set_files:
        rows = [r for r in read_set(set_file) if r.run in specs]
        for n, start in enumerate(range(0, len(rows), per_sheet), 1):
            thumbs = []
            for row in rows[start : start + per_sheet]:
                try:
                    thumbs.append(debug_thumb(specs[row.run], row))
                except ValueError as exc:
                    log.warning("%s", exc)
            if thumbs:
                path = out_dir / f"{set_file.stem}_{n:02d}.jpg"
                write_image(path, sheet_of(thumbs, SHEET_COLS), quality=85)
                log.info("лист: %s (%d кадров)", path, len(thumbs))


def fit_task(args: tuple[RunSpec, str, tuple[FitKind, ...], Path, bool]) -> list[PageRow]:
    """Рамки и выпрямление одного кадра для пула; сбой кадра — в лог и пустой список."""
    run, key, kinds, out_dir, save_pages = args
    try:
        return process_frame(run, key, kinds, out_dir, save_pages)
    except Exception as exc:  # noqa: BLE001 — один кадр не роняет прогон
        log.warning("%s %s: %s", run.name, key, exc)
        return []


@main.command("fit")
@click.option("--run", "runs", multiple=True, required=True, help=RUN_HELP)
@click.option(
    "--set",
    "set_files",
    multiple=True,
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Набор кадров (строка «прогон ключ  # пометка»).",
)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Куда писать оверлеи, листы и fit.csv.")
@click.option(
    "--kind",
    "kinds",
    multiple=True,
    type=click.Choice([k.value for k in FitKind]),
    default=[k.value for k in FitKind],
    show_default=True,
    help="Варианты рамки.",
)
@click.option("--save-pages/--no-save-pages", default=False, show_default=True, help="Писать выпрямленные страницы.")
@click.option("--jobs", type=int, default=8, show_default=True, help="Воркеров (кадр — до ~3 ГБ ОЗУ на remap).")
def fit_cmd(
    runs: tuple[str, ...],
    set_files: tuple[Path, ...],
    out_dir: Path,
    kinds: tuple[str, ...],
    save_pages: bool,
    jobs: int,
) -> None:
    """Рамки страниц по вариантам, выпрямление, прямизна строк (``fit.csv``), оверлеи и листы сравнения."""
    specs = {s.name: s for s in (parse_run(r) for r in runs)}
    kind_list = tuple(FitKind(k) for k in kinds)
    rows = [r for f in set_files for r in read_set(f) if r.run in specs]
    tasks = [(specs[r.run], r.key, kind_list, out_dir, save_pages) for r in rows]
    log.info("кадров: %d, вариантов: %d", len(tasks), len(kind_list))
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        result = [row for chunk in pool.map(fit_task, tasks) for row in chunk]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "fit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, [fl.name for fl in fields(PageRow)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in result)
    log.info("страниц × вариантов: %d → %s", len(result), out_dir / "fit.csv")


# Пояса расхождения адаптивного сплайна и прежнего (``bends``), мм: оверлеи кадра раскладываются по поясу худшей стороны.
BEND_BANDS = (0.15, 0.3, 0.5)


def band_of(diff_mm: float) -> str:
    """Имя папки пояса расхождения: ``lt0.15``, ``0.15-0.3``, ``0.3-0.5``, ``ge0.5``."""
    if diff_mm < BEND_BANDS[0]:
        return f"lt{BEND_BANDS[0]:g}"
    for lo, hi in zip(BEND_BANDS, BEND_BANDS[1:]):
        if diff_mm < hi:
            return f"{lo:g}-{hi:g}"
    return f"ge{BEND_BANDS[-1]:g}"


def bends_task(args: tuple[RunSpec, str, Path, float]) -> list[BendRow]:
    """Изгибы верха и низа одного кадра для пула: меры и оверлей (если расхождение не меньше порога).

    Args:
        args: ``(прогон, ключ, корень выхода, порог расхождения для оверлея в мм)``.

    Returns:
        Строки мер (сбой кадра — в лог и пустой список).
    """
    run, key, out_dir, overlay_min_mm = args
    try:
        case, pages, rows = frame_bends(run, key)
        worst = max((r.diff_max_mm for r in rows), default=0.0)
        if worst >= overlay_min_mm:
            path = out_dir / "overlays" / band_of(worst) / f"{run.name}__{key}.jpg"
            write_image(path, draw_bends(case, pages, rows), quality=88)
        return rows
    except Exception as exc:  # noqa: BLE001 — один битый кэш не роняет обход
        log.warning("%s %s: %s", run.name, key, exc)
        return []


@main.command("bends")
@click.option("--run", "runs", multiple=True, required=True, help=RUN_HELP)
@click.option(
    "--set",
    "set_files",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Набор кадров; без наборов — все кадры прогонов с pages.json и layout.json.",
)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Куда писать bends.csv и оверлеи.")
@click.option(
    "--overlay-min-mm",
    type=float,
    default=0.0,
    show_default=True,
    help="Оверлей кадра — только если наибольшее расхождение кривых не меньше, мм.",
)
@click.option("--jobs", type=int, default=16, show_default=True, help="Воркеров пула (CPU: подгонка и чтение кэша).")
def bends_cmd(
    runs: tuple[str, ...], set_files: tuple[Path, ...], out_dir: Path, overlay_min_mm: float, jobs: int
) -> None:
    """Верх и низ страниц: прежний сплайн против адаптивного (``bends.csv``, оверлеи по поясам расхождения)."""
    specs = {s.name: s for s in (parse_run(r) for r in runs)}
    if set_files:
        keys = [(specs[r.run], r.key) for f in set_files for r in read_set(f) if r.run in specs]
    else:
        keys = [(spec, key) for spec in specs.values() for key in run_keys(spec)]
    tasks = [(spec, key, out_dir, overlay_min_mm) for spec, key in keys]
    log.info("кадров: %d", len(tasks))
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        result = [row for chunk in pool.map(bends_task, tasks, chunksize=4) for row in chunk]
    out_dir.mkdir(parents=True, exist_ok=True)
    result.sort(key=lambda r: -r.diff_max_mm)
    with (out_dir / "bends.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, [fl.name for fl in fields(BendRow)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in result)
    log.info("сторон: %d → %s", len(result), out_dir / "bends.csv")


def cut_task(args: tuple[RunSpec, str, Path, Path, Path]) -> list[CutRow]:
    """Нарезка одного кадра для пула; сбой кадра — в лог и пустой список."""
    run, key, out_dir, cli, template = args
    try:
        return cut_frame(run, key, out_dir, cli, template)
    except Exception as exc:  # noqa: BLE001 — один кадр не роняет прогон
        log.warning("%s %s: %s", run.name, key, exc)
        return []


@main.command("cut")
@click.option("--run", "runs", multiple=True, required=True, help=RUN_HELP)
@click.option(
    "--set",
    "set_files",
    multiple=True,
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Набор кадров (строка «прогон ключ  # пометка»).",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Куда писать страницы, вывод ScanTailor, листы и cut.csv.",
)
@click.option(
    "--cli", "cli_path", type=click.Path(exists=True, path_type=Path), required=True, help="Путь к scantailor-cli."
)
@click.option(
    "--jobs", type=int, default=6, show_default=True, help="Воркеров (ScanTailor и разбор строк — до ~3 ГБ ОЗУ)."
)
def cut_cmd(runs: tuple[str, ...], set_files: tuple[Path, ...], out_dir: Path, cli_path: Path, jobs: int) -> None:
    """Нарезка кадров наборов: рамка spline со сгибом → выпрямление → ScanTailor без и с dewarp → судья (``cut.csv``)."""
    specs = {s.name: s for s in (parse_run(r) for r in runs)}
    out_dir.mkdir(parents=True, exist_ok=True)
    template = out_dir / "profiles" / "default.stp"
    if not template.exists():
        template.parent.mkdir(parents=True, exist_ok=True)
        default_profile_xml(cli_path, template)
    rows = [r for f in set_files for r in read_set(f) if r.run in specs]
    tasks = [(specs[r.run], r.key, out_dir, cli_path, template) for r in rows]
    log.info("кадров: %d", len(tasks))
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        result = [row for chunk in pool.map(cut_task, tasks) for row in chunk]
    with (out_dir / "cut.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, [fl.name for fl in fields(CutRow)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in result)
    log.info("страниц: %d → %s", len(result), out_dir / "cut.csv")
