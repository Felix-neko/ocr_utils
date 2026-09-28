"""Команды стенда границ блоков: ``capture`` — кэш входа блоковой стадии, ``select`` — кандидаты в тестовое множество, ``run`` — алгоритмы, ``engines`` — чужие движки, ``measure`` и ``compare``."""

from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.pack_analysis.stages import page_key
from ocr_utils.page_layout.text_blocks.sides_overlay import side_by_side
from research.block_envelopes.capture import cache_path, capture_page, save
from research.block_envelopes.capture import load as load_input
from research.block_envelopes.engines import EngineKind, run_engine
from research.block_envelopes.overlay import page_picture
from research.block_envelopes.boundary import Boundary, block_of
from research.block_envelopes.grouping import Grouping, groups_of
from research.block_envelopes.measures import page_measures, shapes_from_blocks, shapes_from_json

logger = logging.getLogger(__name__)


def _init_worker() -> None:
    """Воркер пула: без hugepage-пометки и без потоков BLAS/OpenCV (см. .claude/rules/gpu_and_pools.md)."""
    import threadpoolctl

    try:
        np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def read_keys(path: Path) -> list[str]:
    """Ключи полос из файла: строка на ключ, после запятой — что угодно, ``#`` — комментарий.

    Args:
        path: Файл со списком.

    Returns:
        Ключи ``год/выпуск/полоса`` в порядке файла, без повторов.
    """
    keys: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            key = line.split(",")[0].strip()
            if key not in keys:
                keys.append(key)
    return keys


def only_text_keys(pack_dir: Path) -> list[str]:
    """Полосы разбора v3 из папки «только текст» без поворота (``index.csv``).

    Args:
        pack_dir: Выход разбора пака.

    Returns:
        Ключи по алфавиту.
    """
    with (pack_dir / "index.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(row["page"] for row in rows if row["folder"] == "только_текст" and int(row["rotate_cw"]) == 0)


def _capture_one(args: tuple) -> tuple[str, str]:
    """Воркер ``capture``: разобрать полосу и записать кэш; ошибка полосы не рушит прогон.

    Args:
        args: ``(ключ, папка разбора пака, заострённые сканы, папка кэша, папка записей или None)``.

    Returns:
        ``(ключ, "ok" | "skip" | текст ошибки)``.
    """
    key, pack_dir, sharpened_dir, cache_dir, records_dir = args
    target = cache_path(cache_dir, key)
    if target.exists():
        return key, "skip"
    try:
        save(capture_page(key, pack_dir, sharpened_dir, records_dir), target)
    except Exception as error:  # noqa: BLE001 — полоса с ошибкой уходит в лог, прогон идёт дальше
        logger.exception("Полоса %s: ошибка захвата", key)
        return key, f"{type(error).__name__}: {error}"
    return key, "ok"


@click.group()
def main() -> None:
    """Стенд границ и объединения текстовых блоков."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@main.command()
@click.option("--pack-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--sharpened-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--keys", "keys_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option(
    "--records-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Записи кандидатов полос вместо <pack-dir>/work/pages (например, work/pages_before_rules_v2).",
)
@click.option("--jobs", default=16, show_default=True, help="Воркеров пула.")
def capture(
    pack_dir: Path, sharpened_dir: Path, cache_dir: Path, keys_file: Path | None, records_dir: Path | None, jobs: int
) -> None:
    """Кэш входа блоковой стадии по полосам (идемпотентно: готовые пропускаются); без ``--keys`` — все «только текст»."""
    keys = read_keys(keys_file) if keys_file else only_text_keys(pack_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(key, pack_dir, sharpened_dir, cache_dir, records_dir) for key in keys]
    counts = {"ok": 0, "skip": 0, "error": 0}
    errors = []
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        for done, (key, status) in enumerate(pool.map(_capture_one, tasks, chunksize=2), start=1):
            if status in counts:
                counts[status] += 1
            else:
                counts["error"] += 1
                errors.append(f"{key}\t{status}")
            if done % 200 == 0:
                logger.info("Полос %d / %d: %s", done, len(tasks), counts)
    if errors:
        (cache_dir.parent / "capture_errors.txt").write_text("\n".join(errors) + "\n", encoding="utf-8")
    logger.info("Захват: %s → %s", counts, cache_dir)


def _measure_json(args: tuple) -> dict | None:
    """Воркер ``select``: меры полосы по готовому JSON разбора.

    Args:
        args: ``(ключ, путь к JSON)``.

    Returns:
        Строка таблицы мер или ``None`` при ошибке.
    """
    key, path = args
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        shapes, axes_mid = shapes_from_json(payload)
        measures = page_measures(shapes, axes_mid, float(payload["dpi"]))
    except Exception:  # noqa: BLE001 — полоса с ошибкой уходит в лог
        logger.exception("Полоса %s: ошибка мер", key)
        return None
    return {"key": key, **asdict(measures)}


# Виды сбоев для отбора: мера, порог «точно проблема» и сколько полос брать в кандидаты.
SELECT_KINDS = {
    "split": ("split_pairs", 1, 40),
    "hmerge": ("hmerge_blocks", 1, 25),
    "overshoot": ("overshoot_mm", 1.5, 30),
    "saw": ("saw_mm", 6.0, 30),
    "wedge": ("wedge_share", 0.06, 30),
    "lost": ("lost_axes", 1, 20),
}


@main.command()
@click.option("--pack-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option(
    "--pages-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="JSON разбора полос с рядами блоков (прогон research.gutter_crossing legacy).",
)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--jobs", default=8, show_default=True, help="Воркеров пула (задача лёгкая, упирается в чтение JSON).")
def select(pack_dir: Path, pages_dir: Path, out_dir: Path, seed: int, jobs: int) -> None:
    """Меры по всем полосам «только текст» (``measures.csv``) и кандидаты по видам сбоев (``candidates.csv``), плюс чистые (``clean_candidates.txt``)."""
    import random

    keys = only_text_keys(pack_dir)
    tasks = [(key, pages_dir / f"{key.replace('/', '_')}.json") for key in keys]
    tasks = [task for task in tasks if task[1].exists()]
    out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        rows = [row for row in pool.map(_measure_json, tasks, chunksize=16) if row is not None]
    with (out_dir / "measures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    generator = random.Random(seed)
    candidates: list[tuple[str, str, float]] = []
    for kind, (field, threshold, quota) in SELECT_KINDS.items():
        flagged = [row for row in rows if float(row[field]) >= threshold]
        # Половина — самые тяжёлые, половина — случайные из остальных помеченных: не только крайности.
        flagged.sort(key=lambda row: -float(row[field]))
        top = flagged[: quota // 2]
        rest = flagged[quota // 2 :]
        picked = top + generator.sample(rest, min(len(rest), quota - len(top)))
        candidates += [(row["key"], kind, float(row[field])) for row in picked]
        logger.info("Вид %s: помечено %d полос из %d, в кандидаты %d", kind, len(flagged), len(rows), len(picked))
    with (out_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "kind", "value"])
        writer.writerows(candidates)
    # Чистые: ни одна мера не сработала даже на половине порога.
    clean = [
        row["key"]
        for row in rows
        if all(float(row[field]) < threshold / 2 for field, threshold, _ in SELECT_KINDS.values())
    ]
    picked_clean = sorted(generator.sample(clean, min(len(clean), 120)))
    (out_dir / "clean_candidates.txt").write_text("\n".join(picked_clean) + "\n", encoding="utf-8")
    logger.info("Чистых полос %d, в кандидаты %d → %s", len(clean), len(picked_clean), out_dir)


@main.command()
@click.option("--pack-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--sharpened-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--cache-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--keys", "keys_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option(
    "--engine", "engine_names", multiple=True, type=click.Choice([kind.value for kind in EngineKind]), required=True
)
def engines(
    pack_dir: Path, sharpened_dir: Path, cache_dir: Path, out_dir: Path, keys_file: Path, engine_names: tuple[str, ...]
) -> None:
    """Чужие движки по полосам ПОСЛЕДОВАТЕЛЬНО (GPU один на всех): ``<out>/engines/<движок>/pages/*.json``, ``overlays/*.jpg``.

    Идемпотентно: полоса с готовым JSON не пересчитывается, только перерисовывается оверлей.
    """
    keys = read_keys(keys_file)
    for name in engine_names:
        kind = EngineKind(name)
        target = out_dir / "engines" / kind.value
        (target / "pages").mkdir(parents=True, exist_ok=True)
        (target / "overlays").mkdir(parents=True, exist_ok=True)
        seconds = []
        for done, key in enumerate(keys, start=1):
            path = target / "pages" / f"{page_key(key)}.json"
            try:
                if path.exists():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                else:
                    payload = run_engine(kind, key, pack_dir, sharpened_dir)
                    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                seconds.append(payload["seconds"])
                cached = cache_path(cache_dir, key)
                if not cached.exists():
                    continue
                inp = load_input(cached)
                picture = page_picture(
                    inp.gray,
                    [np.asarray(region) for region in payload["regions"]],
                    [
                        f"{key}   движок: {kind.value}",
                        f"регионов {len(payload['regions'])}, строк {len(payload['lines'])}, "
                        f"{payload['seconds']:.1f} с; серым — блоки боевого алгоритма ({len(inp.legacy)})",
                    ],
                    axes=[np.asarray(line) for line in payload["lines"]] or [np.asarray(a.points) for a in inp.axes],
                    reference=[block.envelope.polygon for block in inp.legacy],
                    labels=payload["labels"] if kind is EngineKind.SURYA else None,
                    block_word="регион движка",
                )
                cv2.imwrite(str(target / "overlays" / f"{page_key(key)}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
            except Exception:  # noqa: BLE001 — сбой движка на полосе уходит в лог, прогон идёт дальше
                logger.exception("Движок %s, полоса %s: ошибка", kind.value, key)
            if done % 10 == 0:
                logger.info("Движок %s: полос %d / %d", kind.value, done, len(keys))
        logger.info(
            "Движок %s: готово, медиана %.1f с/полосу → %s", kind.value, float(np.median(seconds or [0])), target
        )


def algo_name(grouping: Grouping, boundary: Boundary) -> str:
    """Имя пары алгоритмов для папок и таблиц: ``<группировка>__<граница>``."""
    return f"{grouping.value}__{boundary.value}"


def run_algo_page(
    key: str, cache_dir: Path, target: Path, grouping: Grouping, boundary: Boundary, overlays: bool = True
) -> dict:
    """Блоки полосы парой алгоритмов по кэшу входа: меры, JSON контуров и оверлей.

    Args:
        key: Полоса.
        cache_dir: Кэш входа блоковой стадии.
        target: Папка пары алгоритмов (``overlays/``, ``pages/``).
        grouping: Способ группировки.
        boundary: Способ границы.
        overlays: Писать ли оверлей полосы.

    Returns:
        Строка сводки: ключ, меры :class:`measures.PageMeasures`, время.
    """
    import time

    inp = load_input(cache_path(cache_dir, key))
    started = time.monotonic()
    blocks = [block_of(group, boundary, inp) for group in groups_of(grouping, inp)]
    seconds = time.monotonic() - started
    shapes, axes_mid, lost = shapes_from_blocks(blocks, inp.axes)
    measures = page_measures(shapes, axes_mid, inp.dpi, lost)
    name = page_key(key)
    payload = {
        "key": key,
        "algo": algo_name(grouping, boundary),
        "blocks": [
            {
                "polygon": np.round(block.envelope.polygon, 1).tolist(),
                "rows": [[round(r.y, 1), round(r.x0, 1), round(r.x1, 1), round(r.height, 1)] for r in block.rows],
            }
            for block in blocks
        ],
    }
    (target / "pages" / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    if overlays:
        picture = page_picture(
            inp.gray,
            [block.envelope.polygon for block in blocks],
            [
                f"{key}   алгоритм: {algo_name(grouping, boundary)}",
                f"блоков {measures.blocks}; выход за краску {measures.overshoot_mm} мм, пила {measures.saw_mm}, "
                f"клин {measures.wedge_share}, пар-разрезов {measures.split_pairs}, слитых {measures.hmerge_blocks}, "
                f"потеряно осей {measures.lost_axes}",
            ],
            axes=[np.asarray(axis.points) for axis in inp.axes],
            reference=(
                [block.envelope.polygon for block in inp.legacy]
                if grouping is not Grouping.LEGACY or boundary is not Boundary.LEGACY
                else None
            ),
        )
        cv2.imwrite(str(target / "overlays" / f"{name}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"key": key, **asdict(measures), "seconds": round(seconds, 3)}


def _run_algo_one(args: tuple) -> dict | None:
    """Обёртка для пула: ошибка полосы уходит в лог."""
    try:
        return run_algo_page(*args)
    except Exception:  # noqa: BLE001 — полоса с ошибкой уходит в лог, прогон идёт дальше
        logger.exception("Полоса %s, алгоритм %s: ошибка", args[0], algo_name(args[3], args[4]))
        return None


@main.command()
@click.option("--cache-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--keys", "keys_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option(
    "--algo",
    "algos",
    multiple=True,
    required=True,
    help="Пара «группировка__граница», например reach__fit; можно несколько.",
)
@click.option("--overlays/--no-overlays", default=True, help="Писать оверлеи полос.")
@click.option("--jobs", default=8, show_default=True, help="Воркеров пула.")
def run(cache_dir: Path, out_dir: Path, keys_file: Path, algos: tuple[str, ...], overlays: bool, jobs: int) -> None:
    """Прогнать пары алгоритмов по кэшу входа: ``<out>/algos/<пара>/overlays``, ``pages``, ``summary.csv``."""
    keys = [key for key in read_keys(keys_file) if cache_path(cache_dir, key).exists()]
    for algo in algos:
        grouping_name, boundary_name = algo.split("__")
        grouping, boundary = Grouping(grouping_name), Boundary(boundary_name)
        target = out_dir / "algos" / algo_name(grouping, boundary)
        (target / "overlays").mkdir(parents=True, exist_ok=True)
        (target / "pages").mkdir(parents=True, exist_ok=True)
        tasks = [(key, cache_dir, target, grouping, boundary, overlays) for key in keys]
        with ProcessPoolExecutor(
            max_workers=jobs, mp_context=get_context("forkserver"), initializer=_init_worker
        ) as pool:
            rows = [row for row in pool.map(_run_algo_one, tasks, chunksize=2) if row is not None]
        with (target / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Алгоритм %s: полос %d → %s", algo, len(rows), target)


def _set_of(keys_file: Path) -> list[str]:
    """Ключи множества из файла (``research/block_envelopes/sets/*.txt``)."""
    return read_keys(keys_file)


# Пороги «полоса с дефектом» для сводки: те же, что у отбора кандидатов (``SELECT_KINDS``).
REPORT_FLAGS = {
    "выход за краску > 1.5 мм": ("overshoot_mm", 1.5),
    "пила > 6": ("saw_mm", 6.0),
    "клин > 0.06": ("wedge_share", 0.06),
    "пар-разрезов ≥ 1": ("split_pairs", 1),
    "слитых ≥ 1": ("hmerge_blocks", 1),
    "потеряно осей ≥ 1": ("lost_axes", 1),
}


@main.command()
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--sets-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
def report(out_dir: Path, sets_dir: Path) -> None:
    """Сводка по парам алгоритмов: доля полос с дефектом по каждой мере, отдельно проблемные и чистые → ``report.md``, ``report.csv``."""
    sets = {name: set(_set_of(sets_dir / f"{name}.txt")) for name in ("problem", "clean")}
    legacy_rows = {}
    legacy_path = out_dir / "algos" / "legacy__legacy" / "summary.csv"
    if legacy_path.exists():
        legacy_rows = {row["key"]: row for row in csv.DictReader(legacy_path.open(encoding="utf-8"))}
    table = []
    for folder in sorted((out_dir / "algos").iterdir()):
        path = folder / "summary.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        for set_name, keys in sets.items():
            own = [row for row in rows if row["key"] in keys]
            if not own:
                continue
            line = {"алгоритм": folder.name, "множество": set_name, "полос": len(own)}
            for label, (field, threshold) in REPORT_FLAGS.items():
                line[label] = sum(1 for row in own if float(row[field]) >= threshold)
            # Изменение числа блоков против боевого: сколько полос стали дробнее и сколько — крупнее.
            changed = [
                (int(row["blocks"]) - int(legacy_rows[row["key"]]["blocks"]))
                for row in own
                if row["key"] in legacy_rows
            ]
            line["блоков меньше"] = sum(1 for d in changed if d < 0)
            line["блоков больше"] = sum(1 for d in changed if d > 0)
            line["с/полосу"] = round(float(np.median([float(row["seconds"]) for row in own])), 2)
            table.append(line)
    with (out_dir / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    header = list(table[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(row[name]) for name in header) + " |" for row in table]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    click.echo("\n".join(lines))


@main.command()
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--keys", "keys_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--algo", "algos", multiple=True, required=True, help="Пары алгоритмов по порядку слева направо.")
@click.option("--engine", "engine_names", multiple=True, help="Движки справа (их оверлеи из engines/).")
@click.option("--name", default="compare", show_default=True, help="Подпапка склеек в <out>/compare/.")
@click.option("--crop", default=1.0, show_default=True, help="Какую долю высоты полосы (сверху) оставлять.")
def compare(
    out_dir: Path, keys_file: Path, algos: tuple[str, ...], engine_names: tuple[str, ...], name: str, crop: float
) -> None:
    """Склейки одной полосы по алгоритмам и движкам рядом: ``<out>/compare/<name>/<ключ>.jpg``."""
    target = out_dir / "compare" / name
    target.mkdir(parents=True, exist_ok=True)
    sources = [(algo, out_dir / "algos" / algo / "overlays") for algo in algos]
    sources += [(engine, out_dir / "engines" / engine / "overlays") for engine in engine_names]
    made = 0
    for key in read_keys(keys_file):
        pictures, titles = [], []
        for title, folder in sources:
            picture = cv2.imread(str(folder / f"{page_key(key)}.jpg"))
            if picture is None:
                continue
            pictures.append(picture[: int(picture.shape[0] * crop)])
            titles.append(title)
        if len(pictures) < 2:
            continue
        cv2.imwrite(
            str(target / f"{page_key(key)}.jpg"), side_by_side(pictures, titles), [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        made += 1
    logger.info("Склеек %d → %s", made, target)
