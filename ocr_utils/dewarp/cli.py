"""Командная строка: ``run`` — прогнать выбранные движки по полосам, папка на движок.

СХЕМА. Сперва каждая полоса приводится к рабочему разрешению и кладётся в
``original/`` — это общий вход для всех движков и «было» для пар сравнения. Дальше
CPU-движки (``textline``, ``pagedewarp``) едут в пул процессов, нейросетевые — по одному
в родителе на GPU (загрузил, прогнал все полосы, освободил видеопамять). В конце —
пары «было | стало» и безэталонная оценка ``quality.csv`` / ``quality.md``.

ВЫХОД: ``<out-dir>/<движок>/<имя>.jpg``, ``<out-dir>/original/``, ``<out-dir>/compare/<движок>/``,
``<out-dir>/quality.{csv,md}``. Имя полосы — ``год_выпуск_имя`` от корня ``--root``, либо
имя симлинка при ``--from-links`` (оно уже несёт год, выпуск и номер страницы).
"""

from __future__ import annotations

import csv
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
from PIL import Image

from ocr_utils.dewarp import compare, quality
from ocr_utils.dewarp.engines import CPU_ENGINES, ENGINES, get_engine
from ocr_utils.scan_cropping.image_io import IMAGE_EXTS

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

JPEG_QUALITY = 92
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
MIN_PLAUSIBLE_DPI = 72


@dataclass(frozen=True)
class Page:
    source: Path
    name: str


def _set_log_level(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s %(name)s: %(message)s")


def _page_name(path: Path, root: Optional[Path]) -> str:
    """``год_выпуск_имя`` от корня, иначе просто имя файла."""
    if root is not None:
        try:
            parts = path.resolve().relative_to(root.resolve()).parts
        except ValueError:
            parts = ()
        if len(parts) >= 3:
            return "_".join(p.replace(" ", "_") for p in parts[:-1]) + "_" + path.stem
    return path.stem


def collect_pages(
    input_dir: Optional[Path],
    root: Optional[Path],
    only: tuple[str, ...],
    from_links: Optional[Path],
    from_csv: Optional[Path],
    min_score: float,
) -> list[Page]:
    pages: list[Page] = []
    if from_links is not None:
        for link in sorted(from_links.iterdir()):
            if link.suffix.lower() not in IMAGE_EXTS:
                continue
            target = link.resolve()
            if target.is_file():
                pages.append(Page(target, link.stem))
    elif from_csv is not None:
        base = root or Path(".")
        with from_csv.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    score = float(row.get("combo_score") or "nan")
                except ValueError:
                    continue
                if score != score or score < min_score:
                    continue
                path = base / row["полоса"]
                if path.is_file():
                    pages.append(Page(path, _page_name(path, root)))
    elif only:
        for item in only:
            path = Path(item)
            if not path.is_absolute() and root is not None:
                path = root / path
            if not path.is_file():
                raise click.UsageError(f"нет файла {path}")
            pages.append(Page(path, _page_name(path, root)))
    elif input_dir is not None:
        for path in sorted(input_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("."):
                pages.append(Page(path, _page_name(path, root or input_dir)))
    return pages


def _read_dpi(path: Path, default: int) -> int:
    try:
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
    except OSError:
        return default
    if not dpi:
        return default
    try:
        value = int(round(float(dpi[0])))
    except (TypeError, ValueError):
        return default
    return value if value >= MIN_PLAUSIBLE_DPI else default


def _prepare(
    page: Page, original_dir: Path, out_dpi: Optional[int], source_dpi: int, overwrite: bool
) -> tuple[str, int, str]:
    """Полоса в рабочем разрешении → original/. Возвращает (имя, dpi, ошибка)."""
    dst = original_dir / f"{page.name}.jpg"
    dpi = _read_dpi(page.source, source_dpi)
    work_dpi = out_dpi or dpi
    if dst.exists() and not overwrite:
        return page.name, work_dpi, ""
    image = cv2.imread(str(page.source), cv2.IMREAD_COLOR)
    if image is None:
        return page.name, work_dpi, "не прочитан"
    if work_dpi != dpi:
        factor = work_dpi / dpi
        size = (max(1, round(image.shape[1] * factor)), max(1, round(image.shape[0] * factor)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return page.name, work_dpi, ""


_ENGINE_CACHE: dict[str, object] = {}


def _cpu_job(engine_name: str, name: str, original: Path, dst: Path, dpi: int) -> tuple[str, str, bool, float, str]:
    """Один CPU-движок на одну полосу (воркер пула). Не бросает."""
    started = time.perf_counter()
    try:
        engine = _ENGINE_CACHE.get(engine_name)
        if engine is None:
            engine = get_engine(engine_name)
            engine.load("cpu")
            _ENGINE_CACHE[engine_name] = engine
        engine.dpi = dpi
        image = cv2.imread(str(original), cv2.IMREAD_COLOR)
        if image is None:
            return engine_name, name, False, 0.0, "не прочитан"
        result = engine.dewarp(image)
        if result is None:
            return engine_name, name, False, time.perf_counter() - started, "движок не обработал"
        cv2.imwrite(str(dst), result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return engine_name, name, True, time.perf_counter() - started, ""
    except Exception as error:  # noqa: BLE001 — сбой одной полосы не роняет прогон
        return engine_name, name, False, time.perf_counter() - started, str(error)[:200]


def _gpu_engine(
    engine_name: str, todo: list[tuple[str, Path, Path, int]], device: str
) -> list[tuple[str, str, bool, float, str]]:
    """Нейросетевой движок: загрузить, прогнать все полосы, освободить."""
    rows = []
    try:
        engine = get_engine(engine_name)
        engine.load(device)
    except Exception as error:  # noqa: BLE001
        logger.error("Движок %s не загрузился: %s", engine_name, error)
        return [(engine_name, name, False, 0.0, f"не загрузился: {str(error)[:160]}") for name, _, _, _ in todo]
    for name, original, dst, dpi in todo:
        started = time.perf_counter()
        try:
            engine.dpi = dpi
            image = cv2.imread(str(original), cv2.IMREAD_COLOR)
            result = engine.dewarp(image) if image is not None else None
            if result is None:
                rows.append((engine_name, name, False, time.perf_counter() - started, "движок не обработал"))
                continue
            cv2.imwrite(str(dst), result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            rows.append((engine_name, name, True, time.perf_counter() - started, ""))
        except Exception as error:  # noqa: BLE001
            rows.append((engine_name, name, False, time.perf_counter() - started, str(error)[:200]))
        click.echo(f"  {engine_name}: {name} {rows[-1][3]:.1f} с{'' if rows[-1][2] else ' — ' + rows[-1][4]}")
    engine.unload()
    del engine
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return rows


def _quality_job(name: str, path: Path, dpi: int) -> tuple[str, dict[str, float]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return name, {}
    try:
        return name, quality.measure(image, dpi)
    except Exception as error:  # noqa: BLE001
        logger.warning("quality %s: %s", path, error)
        return name, {}


def _compare_job(original: Path, result: Path, dst: Path) -> None:
    before = cv2.imread(str(original), cv2.IMREAD_COLOR)
    after = cv2.imread(str(result), cv2.IMREAD_COLOR)
    if before is None or after is None:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), compare.side_by_side(before, after), [cv2.IMWRITE_JPEG_QUALITY, 85])


def _init_worker() -> None:
    # Один поток на воркер — и у OpenCV, и у BLAS. Без этого пул из четырёх процессов считал
    # textline по 18 с на полосу против 1 с в одиночку: OpenBLAS numpy (32 потока) занят
    # активным ожиданием на каждом крошечном polyfit, и четыре процесса по 32 потока душат
    # друг друга. Переменные окружения тут не помогают: forkserver уже поднял numpy.
    cv2.setNumThreads(1)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # noqa: BLE001 — без threadpoolctl остаётся только окружение
        pass
    # ГЛАВНОЕ. numpy помечает большие массивы madvise(MADV_HUGEPAGE), а ядро при
    # transparent_hugepage/defrag=madvise синхронно уплотняет память под каждую такую
    # страницу. Один процесс этого не замечает, а четыре, одновременно выделяющие по 90 МБ
    # под карты ремапа, встают в очередь на уплотнение: замер — np.tile на 22 Мпк 0.18 с в
    # одиночку и 8-10 с при четырёх воркерах, remap 0.5 с против 8 с. С отключённой пометкой
    # те же четыре воркера укладываются в 0.2 и 0.6 с.
    try:
        np._core.multiarray._set_madvise_hugepage(False)
    except AttributeError:
        try:
            np.core.multiarray._set_madvise_hugepage(False)  # numpy < 2
        except Exception:  # noqa: BLE001
            pass
    try:
        os.nice(10)
    except OSError:
        pass


def _pool(jobs: int) -> ProcessPoolExecutor:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    return ProcessPoolExecutor(
        max_workers=max(1, jobs), mp_context=multiprocessing.get_context("forkserver"), initializer=_init_worker
    )


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main() -> None:
    """Выпрямление страниц несколькими движками, папка на движок."""


@main.command("run")
@click.option(
    "--input-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), help="папка с полосами (рекурсивно)"
)
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="корень дерева год/выпуск — для имён и --only",
)
@click.option("--only", multiple=True, help="конкретная полоса (путь или путь от --root); можно повторять")
@click.option(
    "--from-links", type=click.Path(exists=True, file_okay=False, path_type=Path), help="каталог симлинков детектора"
)
@click.option(
    "--from-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV детектора (колонки «полоса», combo_score)",
)
@click.option("--min-score", default=1.0, show_default=True, type=float, help="порог combo_score для --from-csv")
@click.option("--engines", default="all", show_default=True, help="через запятую; all — все")
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--out-dpi", type=int, help="рабочее и выходное разрешение; по умолчанию — как у исходника")
@click.option(
    "--source-dpi", default=600, show_default=True, type=int, help="разрешение исходника, если в файле нет тега"
)
@click.option("--jobs", default=8, show_default=True, type=int, help="процессов на CPU-этапы")
@click.option("--device", default=None, help="cuda / cpu (по умолчанию авто)")
@click.option("--limit", type=int, help="первые N полос — для пробы")
@click.option("--compare/--no-compare", "compare_flag", default=True, show_default=True, help="пары «было | стало»")
@click.option(
    "--quality/--no-quality", "want_quality", default=True, show_default=True, help="метрики кривизны до/после"
)
@click.option("--skip-existing/--overwrite", default=True, show_default=True, help="не пересчитывать готовые файлы")
@click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS))
def run_command(
    input_dir,
    root,
    only,
    from_links,
    from_csv,
    min_score,
    engines,
    out_dir,
    out_dpi,
    source_dpi,
    jobs,
    device,
    limit,
    compare_flag,
    want_quality,
    skip_existing,
    log_level,
) -> None:
    """Прогнать движки по полосам."""
    _set_log_level(log_level)
    names = list(ENGINES) if engines.strip() == "all" else [n.strip() for n in engines.split(",") if n.strip()]
    for name in names:
        if name not in ENGINES:
            raise click.UsageError(f"нет движка {name!r}; есть: {', '.join(ENGINES)}")
    pages = collect_pages(input_dir, root, only, from_links, from_csv, min_score)
    if limit:
        pages = pages[:limit]
    if not pages:
        raise click.UsageError("полос не нашлось: задайте --input-dir, --only, --from-links или --from-csv")
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"
    overwrite = not skip_existing
    out_dir.mkdir(parents=True, exist_ok=True)
    original_dir = out_dir / "original"
    original_dir.mkdir(exist_ok=True)
    click.echo(f"Полос: {len(pages)}. Движки: {', '.join(names)}. Устройство: {device}")

    # 1. Общий вход в рабочем разрешении.
    dpis: dict[str, int] = {}
    with _pool(jobs) as pool:
        for name, dpi, error in pool.map(
            _prepare,
            pages,
            [original_dir] * len(pages),
            [out_dpi] * len(pages),
            [source_dpi] * len(pages),
            [overwrite] * len(pages),
        ):
            if error:
                click.echo(f"  {name}: {error}")
                continue
            dpis[name] = dpi
    ready = [page for page in pages if page.name in dpis]

    # 2. Движки.
    rows: dict[tuple[str, str], quality.QualityRow] = {}

    def todo_for(engine_name: str) -> list[tuple[str, Path, Path, int]]:
        (out_dir / engine_name).mkdir(exist_ok=True)
        items = []
        for page in ready:
            dst = out_dir / engine_name / f"{page.name}.jpg"
            if dst.exists() and not overwrite:
                rows[(engine_name, page.name)] = quality.QualityRow(page.name, engine_name, True, 0.0, "готово ранее")
                continue
            items.append((page.name, original_dir / f"{page.name}.jpg", dst, dpis[page.name]))
        return items

    cpu_names = [n for n in names if n in CPU_ENGINES]
    gpu_names = [n for n in names if n not in CPU_ENGINES]
    cpu_jobs = [(engine_name, *item) for engine_name in cpu_names for item in todo_for(engine_name)]
    if cpu_jobs:
        click.echo(f"CPU-движки ({', '.join(cpu_names)}): {len(cpu_jobs)} заданий, процессов {jobs}")
        with _pool(jobs) as pool:
            for engine_name, name, ok, seconds, note in pool.map(_cpu_job, *zip(*cpu_jobs)):
                rows[(engine_name, name)] = quality.QualityRow(name, engine_name, ok, seconds, note)
                click.echo(f"  {engine_name}: {name} {seconds:.1f} с{'' if ok else ' — ' + note}")
    for engine_name in gpu_names:
        items = todo_for(engine_name)
        if not items:
            continue
        click.echo(f"GPU-движок {engine_name}: {len(items)} полос")
        for engine_name_, name, ok, seconds, note in _gpu_engine(engine_name, items, device):
            rows[(engine_name_, name)] = quality.QualityRow(name, engine_name_, ok, seconds, note)

    # 3. Оценка до/после и пары.
    if want_quality:
        with _pool(jobs) as pool:
            before = dict(
                pool.map(
                    _quality_job,
                    [p.name for p in ready],
                    [original_dir / f"{p.name}.jpg" for p in ready],
                    [dpis[p.name] for p in ready],
                )
            )
            after_jobs = [
                (f"{engine_name}|{name}", out_dir / engine_name / f"{name}.jpg", dpis[name])
                for (engine_name, name), row in rows.items()
                if row.ok and (out_dir / engine_name / f"{name}.jpg").exists()
            ]
            after = dict(pool.map(_quality_job, *zip(*after_jobs))) if after_jobs else {}
        for (engine_name, name), row in rows.items():
            row.before = before.get(name, {})
            row.after = after.get(f"{engine_name}|{name}", {})
        quality.write_csv(out_dir / "quality.csv", list(rows.values()))
        (out_dir / "quality.md").write_text(
            f"# Оценка выпрямления: {out_dir}\n\n" + quality.markdown_table(list(rows.values())) + "\n",
            encoding="utf-8",
        )
        click.echo(f"Оценка: {out_dir / 'quality.md'}")
    if compare_flag:
        compare_jobs = [
            (
                original_dir / f"{name}.jpg",
                out_dir / engine_name / f"{name}.jpg",
                out_dir / "compare" / engine_name / f"{name}.jpg",
            )
            for (engine_name, name), row in rows.items()
            if row.ok and (out_dir / engine_name / f"{name}.jpg").exists()
        ]
        if compare_jobs:
            with _pool(jobs) as pool:
                list(pool.map(_compare_job, *zip(*compare_jobs)))
            click.echo(f"Пары «было | стало»: {out_dir / 'compare'}")

    failed = [(e, n, r.note) for (e, n), r in rows.items() if not r.ok]
    click.echo(f"Готово: {len(rows) - len(failed)} результатов, сбоев {len(failed)}")
    for engine_name, name, note in failed:
        click.echo(f"  сбой {engine_name} / {name}: {note}")


if __name__ == "__main__":
    main()
