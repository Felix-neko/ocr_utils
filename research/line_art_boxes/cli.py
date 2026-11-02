"""CLI стенда «завышенные рамки line art во втором преобразовании v2»: холсты страниц (``canvas``), разбор было/стало (``detect``), листы отсмотра (``sheets``), сверка с разметкой (``evaluate``)."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import click

from ocr_utils.scan_cropping.v2.pipeline import init_worker

log = logging.getLogger(__name__)

# Воркеров по умолчанию: рядом идёт прогон v2 на 12 воркеров, стенду — не больше половины ядер.
DEFAULT_JOBS = 8


def parse_px_per_mm(text: str) -> dict[str, float]:
    """Разобрать ``"01=17.9,02=17.1"`` → ``{"01": 17.9, "02": 17.1}`` (плотность бумаги по выпускам, px/мм).

    Args:
        text: Пары ``префикс выпуска=px/мм`` через запятую; префикс сравнивается с началом имени папки выпуска.

    Returns:
        Словарь префикс → px/мм.
    """
    out: dict[str, float] = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        key, value = item.split("=")
        out[key.strip()] = float(value)
    return out


def px_per_mm_of(issue: str, table: dict[str, float], default: float | None) -> float:
    """Плотность выпуска по префиксу имени папки; нет префикса — ``default`` (или ошибка).

    Args:
        issue: Имя папки выпуска.
        table: Префикс → px/мм.
        default: Плотность по умолчанию или None.

    Returns:
        px/мм.

    Raises:
        click.UsageError: Плотность выпуска не задана.
    """
    for key, value in table.items():
        if issue.startswith(key):
            return value
    if default is None:
        raise click.UsageError(f"нет плотности для выпуска {issue!r}: задайте --px-per-mm или --px-per-mm-default")
    return default


def _canvas_one(args: tuple) -> tuple[str, str]:
    """Воркер ``canvas``: холст одного кадра и вырезки страниц на диск.

    Args:
        args: ``(пути набора, выпуск, кадр, px/мм, папка выхода, пропускать готовые)``.

    Returns:
        ``(кадр, исход)``: ``ok``, ``skip``, ``no_frames`` или текст ошибки.
    """
    from research.line_art_boxes.canvas import frame_canvas, save_crops

    paths, issue, frame, px_per_mm, out_dir, skip = args
    key = f"{issue}/{frame}"
    if skip and (out_dir / issue / f"{frame}_L.json").exists():
        return key, "skip"
    try:
        built = frame_canvas(paths, issue, frame, px_per_mm)
        if built is None:
            return key, "no_frames"
        save_crops(*built, out_dir / issue, frame)
        return key, "ok"
    except Exception as exc:  # noqa: BLE001 — кадр не роняет прогон, исход — в лог
        return key, f"ошибка: {exc!r}"


@click.group()
@click.option("--log-level", default="INFO", show_default=True)
def main(log_level: str) -> None:
    """Стенд «завышенные рамки line art» (research/line_art_boxes)."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--photos", type=click.Path(path_type=Path, exists=True), required=True,
              help="Папка исходных кадров набора (Я.Диск, только чтение).")  # fmt: skip
@click.option("--run", "run_root", type=click.Path(path_type=Path, exists=True), required=True,
              help="Корень набора прогона v2 (с подпапками cache и output).")  # fmt: skip
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True,
              help="Папка вырезок страниц холста (hotstore).")  # fmt: skip
@click.option("--px-per-mm", "px_table", default="", help="Плотность по выпускам: «01=17.9,02=17.1».")
@click.option("--px-per-mm-default", type=float, default=None, help="Плотность выпусков без записи в --px-per-mm.")
@click.option("--issues", default="", help="Только эти выпуски (префиксы через запятую).")
@click.option("--limit", type=int, default=0, help="Не больше стольких кадров на выпуск (0 — все).")
@click.option("--skip-if-exists/--no-skip-if-exists", default=True, show_default=True)
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def canvas(
    photos: Path,
    run_root: Path,
    out_dir: Path,
    px_table: str,
    px_per_mm_default: float | None,
    issues: str,
    limit: int,
    skip_if_exists: bool,
    jobs: int,
) -> None:
    """Холсты первого преобразования (освещение + Кунс) и вырезки страниц, как в ``rectify.page_regions``."""
    from research.line_art_boxes.canvas import V2_SUFFIX, RunPaths

    paths = RunPaths(photos, run_root / "cache", run_root / "output")
    table = parse_px_per_mm(px_table)
    wanted = [s.strip() for s in issues.split(",") if s.strip()]
    tasks = []
    for issue_dir in sorted(p for p in paths.output.iterdir() if p.is_dir()):
        if wanted and not any(issue_dir.name.startswith(w) for w in wanted):
            continue
        rate = px_per_mm_of(issue_dir.name, table, px_per_mm_default)
        frames = sorted(p.name[: -len(V2_SUFFIX)] for p in issue_dir.glob(f"*{V2_SUFFIX}"))
        if limit:
            frames = frames[:limit]
        tasks += [(paths, issue_dir.name, frame, rate, out_dir, skip_if_exists) for frame in frames]
    log.info("кадров: %d, воркеров: %d", len(tasks), jobs)
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        futures = [pool.submit(_canvas_one, t) for t in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            key, status = future.result()
            bucket = status if not status.startswith("ошибка") else "error"
            counts[bucket] = counts.get(bucket, 0) + 1
            if bucket == "error":
                log.warning("%s: %s", key, status)
            if done % 50 == 0:
                log.info("готово %d/%d: %s", done, len(tasks), counts)
    log.info("итог: %s", counts)


def _detect_one(args: tuple) -> tuple[str, str]:
    """Воркер ``detect``: разбор одной вырезки всеми вариантами и запись JSON.

    Args:
        args: ``(PNG вырезки, JSON выхода, варианты, пропускать готовые)``.

    Returns:
        ``(вырезка, исход)``.
    """
    from dataclasses import asdict

    from research.line_art_boxes.canvas import load_crop
    from research.line_art_boxes.detect import analyze_crop, load_page_art, save_page_art
    from research.line_art_boxes.fix import FixVariant

    png, out, variants, skip = args
    try:
        arts = {}
        if out.exists():
            _, arts = load_page_art(out)
        todo = [v for v in variants if not (skip and v in arts)]
        if not todo:
            return str(png), "skip"
        image, crop = load_crop(png)
        for variant in todo:
            arts[variant] = analyze_crop(image, crop.dpi, FixVariant(variant))
        save_page_art(out, arts, asdict(crop))
        return str(png), "ok"
    except Exception as exc:  # noqa: BLE001 — страница не роняет прогон
        return str(png), f"ошибка: {exc!r}"


def crop_files(canvas_dir: Path) -> list[Path]:
    """Все PNG вырезок страниц в кэше холстов (по выпускам, по порядку)."""
    return sorted(canvas_dir.glob("*/*.png"))


@main.command()
@click.option("--canvas", "canvas_dir", type=click.Path(path_type=Path, exists=True), required=True,
              help="Папка вырезок (выход canvas).")  # fmt: skip
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True, help="Папка JSON разборов.")
@click.option("--variants", default="base", show_default=True, help="Варианты через запятую (FixVariant).")
@click.option("--skip-if-exists/--no-skip-if-exists", default=True, show_default=True,
              help="Не пересчитывать вариант, уже записанный в JSON страницы.")  # fmt: skip
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def detect(canvas_dir: Path, out_dir: Path, variants: str, skip_if_exists: bool, jobs: int) -> None:
    """Разбор line art каждой вырезки (как ``page_regions``) боевым кодом и вариантами прототипа."""
    from research.line_art_boxes.fix import FixVariant

    names = [FixVariant(v.strip()).value for v in variants.split(",") if v.strip()]
    tasks = [(p, out_dir / p.parent.name / f"{p.stem}.json", names, skip_if_exists) for p in crop_files(canvas_dir)]
    log.info("страниц: %d, варианты: %s", len(tasks), names)
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        futures = [pool.submit(_detect_one, t) for t in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            key, status = future.result()
            bucket = status if not status.startswith("ошибка") else "error"
            counts[bucket] = counts.get(bucket, 0) + 1
            if bucket == "error":
                log.warning("%s: %s", key, status)
            if done % 200 == 0:
                log.info("готово %d/%d: %s", done, len(tasks), counts)
    log.info("итог: %s", counts)


def page_image(meta: dict, canvas_dir: Path | None, path: Path):
    """Картинка страницы разбора в родных px: вырезка холста из кэша или рендер страницы PDF (для ``regress``).

    Args:
        meta: Паспорт из JSON разбора.
        canvas_dir: Папка вырезок или None.
        path: JSON разбора (имя вырезки — по нему).

    Returns:
        Картинка BGR.
    """
    import cv2

    from research.line_art_boxes.canvas import load_crop

    if "pdf" in meta:
        from ocr_utils.page_layout.image import PageImage, Variant

        page = PageImage.from_pdf_file(Path(meta["pdf"]), int(meta["index"]), Variant.FR_NOGEO)
        return cv2.cvtColor(page.gray_at(page.dpi), cv2.COLOR_GRAY2BGR)
    if canvas_dir is None:
        raise click.UsageError("для разборов canvas нужна --canvas")
    return load_crop(canvas_dir / path.parent.name / f"{path.stem}.png")[0]


def _rectify_page(meta: dict) -> bool:
    """Зовёт ли ``rectify`` детектор на этой странице: шла в поле по автоматике и по флагам кадра."""
    return bool(meta.get("plain", True)) and bool(meta.get("field", True))


@main.command()
@click.option("--canvas", "canvas_dir", type=click.Path(path_type=Path, exists=True), default=None,
              help="Папка вырезок (выход canvas); для разборов regress не нужна — страница рендерится из PDF.")  # fmt: skip
@click.option("--detect", "detect_dir", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True, help="Папка листов.")
@click.option("--variant", default="legacy", show_default=True, help="Вариант листов отсмотра («было»).")
@click.option("--compare", "compare_to", default="", help="Вариант «стало»: листы только страниц, где рамки сменились.")
@click.option("--columns", type=int, default=5, show_default=True)
@click.option("--per-sheet", type=int, default=10, show_default=True, help="Миниатюр (страниц) на лист.")
@click.option("--all-pages", is_flag=True, help="И страницы, где rectify детектор не зовёт (цветные, флаги).")
def sheets(
    canvas_dir: Path | None,
    detect_dir: Path,
    out_dir: Path,
    variant: str,
    compare_to: str,
    columns: int,
    per_sheet: int,
    all_pages: bool,
) -> None:
    """Листы миниатюр страниц с рамками line art (или «было | стало» двух вариантов) для отсмотра глазами."""
    import cv2

    from research.line_art_boxes.detect import load_page_art
    from research.line_art_boxes.sheets import grid, legend_entries, page_tile, pair_tile

    picked = []
    for path in sorted(detect_dir.glob("*/*.json")):
        meta, arts = load_page_art(path)
        if not all_pages and not _rectify_page(meta):
            continue
        before = arts.get(variant)
        if before is None:
            continue
        if compare_to:
            after = arts.get(compare_to)
            if after is None or [b.box for b in after.boxes] == [b.box for b in before.boxes]:
                continue
        elif not before.boxes:
            continue
        picked.append((path, meta, before, arts.get(compare_to) if compare_to else None))
    log.info("страниц на листы: %d", len(picked))
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(picked), per_sheet):
        tiles = []
        for path, meta, before, after in picked[start : start + per_sheet]:
            image = page_image(meta, canvas_dir, path)
            title = [
                f"{path.parent.name[:2]}/{path.stem}" if canvas_dir is not None else f"{path.parent.name}/{path.stem}"
            ]
            if after is None:
                tiles.append(page_tile(image, before, title + [f"рамок {len(before.boxes)}"]))
            else:
                tiles.append(pair_tile(image, before, after, title))
        number = start // per_sheet + 1
        what = f"«было {variant} | стало {compare_to}»" if compare_to else f"вариант {variant}"
        header = [
            f"line art второго преобразования v2 — {what}, лист {number}, страницы {start + 1}–{start + len(tiles)}"
        ]
        sheet = grid(tiles, max(1, columns // 2) if compare_to else columns, header, legend_entries())
        name = f"{'compare_' + variant + '_' + compare_to if compare_to else variant}_{number:03d}.jpg"
        cv2.imwrite(str(out_dir / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


@main.command()
@click.option("--detect", "detect_dir", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--labels", type=click.Path(path_type=Path, exists=True), default=None,
              help="Разметка рамок прежнего кода (sets/*.csv); без неё — только счёт смен против legacy.")  # fmt: skip
@click.option("--variants", default="", help="Варианты через запятую (по умолчанию — все, что есть в разборах).")
def report(detect_dir: Path, labels: Path | None, variants: str) -> None:
    """Сводка вариантов: рамок, смен против legacy и (с разметкой) вредных рамок, потерь, добора малых."""
    from research.line_art_boxes.evaluate import changes, load_pages, read_labels, tally

    pages = load_pages(detect_dir)
    names = [v.strip() for v in variants.split(",") if v.strip()]
    if not names:
        names = sorted({v for arts in pages.values() for v in arts}, key=lambda v: (v != "legacy", v != "base", v))
    click.echo(f"страниц: {len(pages)}")
    for name in names:
        total = sum(len(a[name].boxes) for a in pages.values() if name in a)
        changed, removed, added, moved = changes(pages, name)
        line = f"{name:10s} рамок {total:4d} | страниц со сменой {changed:3d}: снято {removed}, новых {added}, сдвинуто {moved}"
        click.echo(line)
    if labels is None:
        return
    marks = read_labels(labels)
    for name in names:
        t = tally(marks, pages, name)
        harmful = ", ".join(f"{k} {v}" for k, v in t.harmful_left.items())
        click.echo(
            f"{name:10s} вредных {sum(t.harmful_left.values()):3d} ({harmful}); безвредных ложных {t.harmless_left}; "
            f"завышенных исправлено {t.overgrown_fixed[0]}/{t.overgrown_fixed[1]}; малых добрано "
            f"{t.undersized_grown[0]}/{t.undersized_grown[1]}; верных потеряно {len(t.correct_lost)}"
            + (f": {', '.join(t.correct_lost)}" if t.correct_lost else "")
        )


@main.command("gt-sheets")
@click.option("--canvas", "canvas_dir", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--labels", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True, help="Папка листов разметки.")
@click.option("--objects", "objects_json", type=click.Path(path_type=Path), required=True,
              help="JSON компонент колец (по нему номера из разметки переводятся в рамки).")  # fmt: skip
@click.option("--per-sheet", type=int, default=4, show_default=True)
def gt_sheets(canvas_dir: Path, labels: Path, out_dir: Path, objects_json: Path, per_sheet: int) -> None:
    """Листы разметки полноты: у каждой верной рамки — компоненты кольца с номерами (что из них часть рисунка)."""
    import cv2

    from research.line_art_boxes.coverage import gt_tile, objects, ring_components, save_objects, work_ink
    from research.line_art_boxes.detect import load_page_art
    from research.line_art_boxes.evaluate import LABELS_VARIANT, read_labels
    from research.line_art_boxes.sheets import grid

    rows, tiles = [], []
    for label in objects(read_labels(labels)):
        issue = next(p.name for p in canvas_dir.iterdir() if p.name.startswith(label.page[:2]))
        name = label.page.split("/")[1]
        work = work_ink(canvas_dir / issue / f"{name}.png")
        _, arts = load_page_art(canvas_dir.parent / "detect" / issue / f"{name}.json")
        native = arts[LABELS_VARIANT].boxes[label.box - 1].box
        from research.line_art_boxes.coverage import to_work

        box = to_work(native, work.scale)
        comps = ring_components(work, box)
        rows.append({"page": label.page, "box": label.box, "kind": label.kind.value, "work_box": list(box.as_tuple()),
                     "comps": comps})  # fmt: skip
        tiles.append(gt_tile(work, box, comps, f"{label.page}#{label.box} {label.kind.value}"))
    save_objects(objects_json, rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    legend = []
    for start in range(0, len(tiles), per_sheet):
        sheet = grid(tiles[start : start + per_sheet], 2, ["разметка полноты: красные — кандидаты с номерами, "
                                                            "синие — буквы строк, оранжевая — рамка"], legend)  # fmt: skip
        cv2.imwrite(str(out_dir / f"gt_{start // per_sheet + 1:02d}.png"), sheet)
    log.info("объектов: %d", len(rows))


@main.command()
@click.option("--canvas", "canvas_dir", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--detect", "detect_dir", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--objects", "objects_json", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--gt", "gt_csv", type=click.Path(path_type=Path, exists=True), required=True,
              help="Разметка полноты (sets/hp_1991_gt.csv).")  # fmt: skip
@click.option("--variants", default="base,grow,ring,close", show_default=True)
def coverage(canvas_dir: Path, detect_dir: Path, objects_json: Path, gt_csv: Path, variants: str) -> None:
    """Полнота рамок: доля краски рисунка вне рамки (по разметке) и захват букв строк по всем страницам с рамками."""
    import json

    import numpy as np

    from research.line_art_boxes.coverage import art_boxes, captured_letters, gt_box, outside_share, read_gt, work_ink
    from research.line_art_boxes.detect import load_page_art
    from ocr_utils.page_layout.geometry import Box

    names = [v.strip() for v in variants.split(",") if v.strip()]
    rows = json.loads(objects_json.read_text())
    gt = read_gt(gt_csv)
    shares: dict[str, list[float]] = {n: [] for n in names}
    per_object: list[str] = []
    for row in rows:
        key = (row["page"], row["box"])
        if key not in gt:
            continue
        truth = gt_box(Box(*row["work_box"]), row["comps"], gt[key])
        issue = next(p.name for p in canvas_dir.iterdir() if p.name.startswith(row["page"][:2]))
        name = row["page"].split("/")[1]
        work = work_ink(canvas_dir / issue / f"{name}.png")
        _, arts = load_page_art(detect_dir / issue / f"{name}.json")
        line = [f"{row['page']}#{row['box']}"]
        for n in names:
            share = outside_share(work.bitonal, truth, art_boxes(arts[n], work.scale)) if n in arts else float("nan")
            shares[n].append(share)
            line.append(f"{n} {100 * share:.1f}%")
        per_object.append(" | ".join(line))
    click.echo("\n".join(per_object))
    for n in names:
        values = np.array([v for v in shares[n] if v == v])
        click.echo(f"{n:8s} объектов {len(values)}: доля краски вне рамки — средняя {100 * values.mean():.2f}%, "
                   f"медиана {100 * np.median(values):.2f}%, макс {100 * values.max():.1f}%, "
                   f"рамок с потерей > 1% — {int((values > 0.01).sum())}")  # fmt: skip
    # Захват букв строк: по всем страницам, где у варианта и base есть рамки.
    for n in names:
        if n == "base":
            continue
        total, pages = 0, 0
        for path in sorted(detect_dir.glob("*/*.json")):
            meta, arts = load_page_art(path)
            if n not in arts or "base" not in arts or not arts[n].boxes:
                continue
            if [b.box for b in arts[n].boxes] == [b.box for b in arts["base"].boxes]:
                continue
            work = work_ink(canvas_dir / path.parent.name / f"{path.stem}.png")
            got = captured_letters(work, art_boxes(arts["base"], work.scale), art_boxes(arts[n], work.scale))
            total += got
            pages += int(got > 0)
        click.echo(f"{n:8s} захвачено букв строк сверх base: {total} на {pages} страницах")


def _carry_one(args: tuple) -> tuple[str, str]:
    """Воркер ``carry``: кадр без переноса областей и с переносом (:func:`research.line_art_boxes.carry.compare_frame`).

    Args:
        args: ``(пути набора, выпуск, кадр, px/мм, папка выхода)``.

    Returns:
        ``(кадр, исход)``.
    """
    from research.line_art_boxes.carry import compare_frame

    paths, issue, frame, px_per_mm, out_dir = args
    key = f"{issue}/{frame}"
    if (out_dir / f"{issue[:2]}_{frame}.json").exists():
        return key, "skip"
    try:
        return key, "ok" if compare_frame(paths, issue, frame, px_per_mm, out_dir) is not None else "no_frames"
    except Exception as exc:  # noqa: BLE001 — кадр не роняет прогон
        return key, f"ошибка: {exc!r}"


@main.command()
@click.option("--photos", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--run", "run_root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True, help="Папка листов и JSON.")
@click.option("--frames", required=True, help="Кадры «выпуск/кадр» через запятую (выпуск — префикс папки).")
@click.option("--px-per-mm", "px_table", default="", help="Плотность по выпускам: «01=17.9,02=17.1».")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула (~2.5 ГБ ОЗУ на кадр).")
def carry(photos: Path, run_root: Path, out_dir: Path, frames: str, px_table: str, jobs: int) -> None:
    """Перенос областей второго преобразования за содержимым: отставание рамок на итоге и меры «было | стало»."""
    import json

    import numpy as np

    from research.line_art_boxes.canvas import RunPaths

    paths = RunPaths(photos, run_root / "cache", run_root / "output")
    table = parse_px_per_mm(px_table)
    issues = sorted(p.name for p in paths.output.iterdir() if p.is_dir())
    tasks = []
    for item in filter(None, (f.strip() for f in frames.split(","))):
        prefix, frame = item.split("/")
        issue = next(name for name in issues if name.startswith(prefix))
        tasks.append((paths, issue, frame, px_per_mm_of(issue, table, None), out_dir))
    log.info("кадров: %d, воркеров: %d", len(tasks), jobs)
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for future in as_completed([pool.submit(_carry_one, t) for t in tasks]):
            key, status = future.result()
            log.info("%s: %s", key, status)
    # Сводка: отставание рамок и меры итога «было | стало».
    lags = {"fixed": [], "carry": []}
    content = {"fixed": [], "carry": []}
    rows = []
    for path in sorted(out_dir.glob("*.json")):
        record = json.loads(path.read_text())
        for side in lags:
            lags[side] += [r["lag"] for r in record[side]["lag"] if r["lag"] is not None]
            content[side] += [r["lag"] for r in record[side].get("content", [])]
        before, after = record["fixed"]["measures"], record["carry"]["measures"]
        rows.append(f"{record['frame']}: строк {before['lines']}→{after['lines']}, размах строк p95 "
                    f"{before['line_p95_mm']}→{after['line_p95_mm']} мм, сторон {before['sides']}→{after['sides']}, "
                    f"размах сторон p95 {before['side_p95_mm']}→{after['side_p95_mm']} мм")  # fmt: skip
    click.echo("\n".join(rows))
    for side, values in lags.items():
        v = np.array(values)
        if len(v):
            click.echo(f"{side}: рамок {len(v)}, отличие от рамки, найденной заново по итогу — медиана {np.median(v):.2f} "
                       f"мм, p90 {np.percentile(v, 90):.2f}, макс {v.max():.2f}, > 1 мм — {int((v > 1).sum())}")  # fmt: skip
        c = np.array(content[side])
        if len(c):
            click.echo(f"{side}: областей {len(c)}, сдвиг содержимого внутри рамки — медиана {np.median(c):.2f} мм, "
                       f"p90 {np.percentile(c, 90):.2f}, макс {c.max():.2f}, > 0.5 мм — {int((c > 0.5).sum())}")  # fmt: skip


def _regress_one(args: tuple) -> tuple[str, str]:
    """Воркер ``regress``: страница PDF пака с блоками surya из кэша — разбор вариантами и запись JSON.

    Args:
        args: ``(PDF, номер страницы, корень кэша surya, JSON выхода, варианты)``.

    Returns:
        ``(страница, исход)``: ``ok``, ``no_cache`` или текст ошибки.
    """
    from ocr_utils.page_layout.image import PageImage, Variant
    from ocr_utils.page_layout.surya.cache import SuryaCache
    from ocr_utils.page_layout.surya.source import SuryaSource
    from research.line_art_boxes.detect import analyze_page, load_page_art, save_page_art
    from research.line_art_boxes.fix import FixVariant

    pdf, index, cache_root, out, variants = args
    key = f"{pdf.stem}/p{index:04d}"
    try:
        cache = SuryaCache(cache_root, readonly=True)
        if not cache.has(Variant.FR_NOGEO, key):
            return key, "no_cache"
        arts = load_page_art(out)[1] if out.exists() else {}
        todo = [v for v in variants if v not in arts]
        if not todo:
            return key, "skip"
        for variant in todo:
            # Страница открывается заново на каждый вариант: копии разных dpi кэшируются внутри PageImage.
            page = PageImage.from_pdf_file(pdf, index, Variant.FR_NOGEO)
            arts[variant] = analyze_page(page, FixVariant(variant), SuryaSource(cache))
        save_page_art(out, arts, {"pdf": str(pdf), "index": index, "dpi": page.dpi, "plain": True, "field": True})
        return key, "ok"
    except Exception as exc:  # noqa: BLE001 — страница не роняет прогон
        return key, f"ошибка: {exc!r}"


@main.command()
@click.option("--pdf-dir", type=click.Path(path_type=Path, exists=True), required=True,
              help="Папка PDF FineReader без коррекции геометрии (страница = полоса пиксель в пиксель).")  # fmt: skip
@click.option("--surya-cache", type=click.Path(path_type=Path, exists=True), required=True,
              help="Корень кэша surya layout (вариант fr_nogeo), только чтение.")  # fmt: skip
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True, help="Папка JSON разборов.")
@click.option("--variants", default="legacy,base", show_default=True)
@click.option("--step", type=int, default=4, show_default=True, help="Каждая N-я страница каждого PDF.")
@click.option("--pdf-limit", type=int, default=0, help="Не больше стольких PDF (0 — все), равномерно по списку.")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def regress(
    pdf_dir: Path, surya_cache: Path, out_dir: Path, variants: str, step: int, pdf_limit: int, jobs: int
) -> None:
    """Регрессия на паке с surya (путь scan_markup / final_pdfs): разбор страниц PDF боевым кодом и прототипом."""
    import fitz

    names = [v.strip() for v in variants.split(",") if v.strip()]
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if pdf_limit and len(pdfs) > pdf_limit:
        pdfs = [pdfs[round(i * (len(pdfs) - 1) / (pdf_limit - 1))] for i in range(pdf_limit)]
    tasks = []
    for pdf in pdfs:
        with fitz.open(str(pdf)) as document:
            count = document.page_count
        tasks += [(pdf, i, surya_cache, out_dir / pdf.stem / f"p{i:04d}.json", names) for i in range(0, count, step)]
    log.info("страниц: %d из %d PDF, варианты: %s", len(tasks), len(pdfs), names)
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        futures = [pool.submit(_regress_one, t) for t in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            key, status = future.result()
            bucket = status if not status.startswith("ошибка") else "error"
            counts[bucket] = counts.get(bucket, 0) + 1
            if bucket == "error":
                log.warning("%s: %s", key, status)
            if done % 200 == 0:
                log.info("готово %d/%d: %s", done, len(tasks), counts)
    log.info("итог: %s", counts)


if __name__ == "__main__":
    main()
