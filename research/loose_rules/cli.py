"""Команды стенда: ``features`` — признаки всех сирот пака, ``sheets`` — листы глазами, ``evaluate`` — правила-кандидаты на разметке и по паку, ``compare`` — текстовые блоки без ложных барьеров «было | стало»."""

from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.geometry import LooseRule
from ocr_utils.page_layout.overlay_frame import LegendEntry, framed
from ocr_utils.page_layout.pack_analysis.stages import PageTask, load_image, page_key
from ocr_utils.page_layout.tables.ruling import WORK_DPI
from research.loose_rules.compare import compare_page
from research.loose_rules.evaluate import Thresholds, confusion, reasons
from research.loose_rules.features import formula_boxes, page_ink, rule_features
from research.loose_rules.labels import RuleLabel, read_labels

logger = logging.getLogger(__name__)

# Трасса на листах: красный «отвергнуто» палитры (``stages.COLOUR_BAD``), сдвинута вбок от штриха,
# чтобы не закрывать саму краску, которую и надо разглядеть.
COLOUR_TRACE = (0, 0, 220)
TRACE_SHIFT_MM = 1.7
# Поле вокруг линейки на плитке листа и размер плитки.
TILE_PAD_MM = 12.7
TILE_SIZE = 400
TILE_CAPTION = 30
SHEET_COLUMNS = 6


def _init_worker() -> None:
    """Воркер пула: без hugepage-пометки и без потоков BLAS/OpenCV (см. .claude/rules/gpu_and_pools.md)."""
    import threadpoolctl

    try:
        np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def _rotation(final: dict) -> int:
    """Поворот полосы, с которым её разбирали (из вердикта ориентации, если он применён)."""
    orientation = final.get("orientation") or {}
    return orientation.get("rotate_cw", 0) if orientation.get("apply") else 0


def _loose_rule(item: dict) -> LooseRule:
    """Сирота из итогового JSON (пиксели скана)."""
    return LooseRule(tuple(tuple(point) for point in item["points"]), item["horizontal"], item["thickness_px"])


def page_features(key: str, pack_dir: Path, sharpened_dir: Path) -> list[dict]:
    """Признаки всех сирот полосы.

    Args:
        key: Полоса ``год/выпуск/полоса``.
        pack_dir: Разбор пака (``pages/*.json``).
        sharpened_dir: Заострённые сканы — по ним шёл разбор.

    Returns:
        Строки CSV: ``rule_id`` (``полоса#номер``), полоса, номер в ``loose_rules``, ось, папка и признаки.
    """
    final = json.loads((pack_dir / "pages" / f"{page_key(key)}.json").read_text(encoding="utf-8"))
    if not final["loose_rules"]:
        return []
    image = load_image(PageTask(key, sharpened_dir / f"{key}.jpg"), _rotation(final))
    ink = page_ink(image.gray_at(WORK_DPI))
    scale = WORK_DPI / final["dpi"]
    formulas = formula_boxes(final["objects"], scale)
    rows = []
    for index, item in enumerate(final["loose_rules"]):
        features = rule_features(ink, _loose_rule(item).scaled(scale), WORK_DPI, formulas)
        row = {
            "rule_id": f"{key}#{index}",
            "page": key,
            "index": index,
            "horizontal": int(item["horizontal"]),
            "folder": final.get("folder", ""),
        }
        row.update(features.to_row())
        rows.append(row)
    return rows


def _features_one(args: tuple) -> list[dict]:
    """Обёртка для пула: ошибка одной полосы не рушит прогон."""
    try:
        return page_features(*args)
    except Exception:  # noqa: BLE001 — полоса с ошибкой уходит в лог, прогон идёт дальше
        logger.exception("Полоса %s: ошибка признаков", args[0])
        return []


@click.group()
def main() -> None:
    """Стенд «ложные линейки-сироты»."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@main.command()
@click.option("--pack-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--sharpened-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--jobs", default=16, show_default=True, help="Воркеров пула.")
def features(pack_dir: Path, sharpened_dir: Path, out_dir: Path, jobs: int) -> None:
    """Признаки всех сирот пака: ``<out-dir>/features.csv`` (строка на линейку)."""
    keys = []
    for path in sorted((pack_dir / "pages").glob("*.json")):
        final = json.loads(path.read_text(encoding="utf-8"))
        if final["loose_rules"]:
            keys.append(final["page"])
    logger.info("Полос с сиротами: %d", len(keys))
    tasks = [(key, pack_dir, sharpened_dir) for key in keys]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        for done, found in enumerate(pool.map(_features_one, tasks, chunksize=4), start=1):
            rows += found
            if done % 500 == 0:
                logger.info("Полос %d / %d, линеек %d", done, len(tasks), len(rows))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Линеек %d → %s", len(rows), out_dir / "features.csv")


def tile(key: str, index: int, caption: str, pack_dir: Path, sharpened_dir: Path) -> np.ndarray:
    """Плитка листа: вырезка вокруг сироты в разрешении скана, трасса сбоку от штриха, подпись над вырезкой.

    Args:
        key: Полоса.
        index: Номер линейки в ``loose_rules``.
        caption: Подпись плитки (латиница и цифры — шрифт OpenCV).
        pack_dir: Разбор пака.
        sharpened_dir: Заострённые сканы.

    Returns:
        Плитка BGR ``TILE_SIZE + TILE_CAPTION`` × ``TILE_SIZE``.
    """
    final = json.loads((pack_dir / "pages" / f"{page_key(key)}.json").read_text(encoding="utf-8"))
    item = final["loose_rules"][index]
    gray = load_image(PageTask(key, sharpened_dir / f"{key}.jpg"), _rotation(final)).gray
    points = np.asarray(item["points"], dtype=np.float64)
    pad = TILE_PAD_MM * final["dpi"] / 25.4
    x0, y0 = np.maximum(0, points.min(axis=0) - pad).astype(int)
    x1 = int(min(gray.shape[1], points[:, 0].max() + pad))
    y1 = int(min(gray.shape[0], points[:, 1].max() + pad))
    crop = cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    # Трасса сдвинута вбок (ниже горизонтали, правее вертикали) на ``TRACE_SHIFT_MM``.
    shift = TRACE_SHIFT_MM * final["dpi"] / 25.4
    offset = np.array([0.0, shift]) if item["horizontal"] else np.array([shift, 0.0])
    trace = (points - [x0, y0] + offset).astype(np.int32)
    cv2.polylines(crop, [trace], False, COLOUR_TRACE, max(2, int(final["dpi"] / 100)))
    scale = TILE_SIZE / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    result = np.full((TILE_SIZE + TILE_CAPTION, TILE_SIZE, 3), 255, np.uint8)
    result[TILE_CAPTION : TILE_CAPTION + crop.shape[0], : crop.shape[1]] = crop
    cv2.putText(result, caption, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return result


def _tile_one(args: tuple) -> np.ndarray:
    """Обёртка для пула: ошибка плитки — пустая плитка."""
    try:
        return tile(*args)
    except Exception:  # noqa: BLE001 — плитка с ошибкой остаётся пустой
        logger.exception("Плитка %s#%s: ошибка", args[0], args[1])
        return np.full((TILE_SIZE + TILE_CAPTION, TILE_SIZE, 3), 255, np.uint8)


@main.command()
@click.option("--pack-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--sharpened-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--features-csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--query", default="", help="Отбор строк признаков (pandas.DataFrame.query), пусто — все.")
@click.option("--ids", "ids_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--count", default=30, show_default=True, help="Сколько линеек в выборке.")
@click.option("--seed", default=0, show_default=True)
@click.option("--per-sheet", default=30, show_default=True)
@click.option("--name", default="sheet", show_default=True, help="Префикс файлов листов.")
@click.option("--jobs", default=16, show_default=True)
def sheets(
    pack_dir: Path,
    sharpened_dir: Path,
    features_csv: Path,
    out_dir: Path,
    query: str,
    ids_file: Path | None,
    count: int,
    seed: int,
    per_sheet: int,
    name: str,
    jobs: int,
) -> None:
    """Листы глазами: случайная выборка по ``--query`` (или список ``--ids``); ``<name>_NN.jpg`` и ``<name>_index.csv``."""
    import pandas as pd

    table = pd.read_csv(features_csv)
    if ids_file is not None:
        wanted = [line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        chosen = table.set_index("rule_id").loc[wanted].reset_index()
    else:
        chosen = table.query(query) if query else table
        chosen = chosen.sample(n=min(count, len(chosen)), random_state=seed)
    logger.info("Отобрано %d линеек (из %d под отбором)", len(chosen), len(table.query(query)) if query else len(table))
    out_dir.mkdir(parents=True, exist_ok=True)
    records = chosen.to_dict("records")
    tasks = [
        (
            row["page"],
            int(row["index"]),
            f"{number} {row['page'][:4]}/{row['page'][8:]} {row['length_mm']:.0f}mm g{row['glyph_share_20']:.2f}",
            pack_dir,
            sharpened_dir,
        )
        for number, row in enumerate(records)
    ]
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        tiles = list(pool.map(_tile_one, tasks))
    legend = [LegendEntry("трасса сироты (сдвинута вбок от штриха)", COLOUR_TRACE)]
    index_rows = []
    for sheet, start in enumerate(range(0, len(tiles), per_sheet)):
        part = tiles[start : start + per_sheet]
        while len(part) % SHEET_COLUMNS:
            part.append(np.full_like(part[0], 255))
        grid = np.vstack([np.hstack(part[i : i + SHEET_COLUMNS]) for i in range(0, len(part), SHEET_COLUMNS)])
        title = [f"{name}_{sheet:02d}: {query or ids_file}", "подпись: номер, год/полоса, длина, glyph_share при 2.0"]
        cv2.imwrite(
            str(out_dir / f"{name}_{sheet:02d}.jpg"), framed(grid, title, legend), [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        for number in range(start, min(start + per_sheet, len(records))):
            index_rows.append({"sheet": sheet, "tile": number, "rule_id": records[number]["rule_id"]})
    with (out_dir / f"{name}_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sheet", "tile", "rule_id"])
        writer.writeheader()
        writer.writerows(index_rows)
    logger.info("Листов %d → %s", (len(tiles) + per_sheet - 1) // per_sheet, out_dir)


@main.command()
@click.option("--features-csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--glyph-share", default=Thresholds.glyph_share, show_default=True)
@click.option("--rule-run-mm", default=Thresholds.rule_run_mm, show_default=True)
@click.option("--fraction-side-ink", default=Thresholds.fraction_side_ink, show_default=True)
@click.option("--edge-core-gray", default=Thresholds.edge_core_gray, show_default=True)
def evaluate(
    features_csv: Path,
    out_dir: Path,
    glyph_share: float,
    rule_run_mm: float,
    fraction_side_ink: float,
    edge_core_gray: float,
) -> None:
    """Правила на разметке (``sets/labels.csv``) и по паку: ``<out-dir>/verdicts.csv`` (причина на линейку), сводка в лог."""
    import pandas as pd

    limits = Thresholds(glyph_share, rule_run_mm, fraction_side_ink, edge_core_gray)
    table = pd.read_csv(features_csv)
    table["reason"] = reasons(table, limits)
    labels = read_labels()
    labelled = table[table.rule_id.isin(labels)].copy()
    labelled["label"] = labelled.rule_id.map(lambda rule_id: labels[rule_id].value)
    logger.info("Пороги: %s", limits)
    logger.info("Разметка (%d): %s", len(labelled), confusion(labelled, labelled.reason))
    # Ошибки на разметке — поимённо, чтобы их можно было открыть листом (``sheets --ids``).
    for _, row in labelled.iterrows():
        wrong_drop = row.label == RuleLabel.RULE.value and row.reason
        missed = row.label != RuleLabel.RULE.value and not row.reason
        if wrong_drop or missed:
            logger.info(
                "  %s %s: %s → %s", "ПОТЕРЯ" if wrong_drop else "пропуск", row.rule_id, row.label, row.reason or "-"
            )
    # По паку: сколько отброшено по каждой причине и оси.
    for horizontal, part in table.groupby("horizontal"):
        counts = part.reason.replace("", "kept").value_counts().to_dict()
        logger.info("Пак, %s (%d): %s", "горизонтали" if horizontal else "вертикали", len(part), counts)
    out_dir.mkdir(parents=True, exist_ok=True)
    table[["rule_id", "page", "index", "horizontal", "reason"]].to_csv(out_dir / "verdicts.csv", index=False)
    logger.info("Полос с отброшенными сиротами: %d → %s", table[table.reason != ""].page.nunique(), out_dir)


def _compare_one(args: tuple) -> dict | None:
    """Обёртка для пула: ошибка одной полосы не рушит прогон."""
    try:
        return compare_page(*args)
    except Exception:  # noqa: BLE001 — полоса с ошибкой уходит в лог
        logger.exception("Полоса %s: ошибка пересчёта блоков", args[0])
        return None


@main.command()
@click.option("--pack-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--sharpened-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--verdicts", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--limit", default=0, show_default=True, help="Не больше стольких полос (0 — все с отброшенными).")
@click.option("--jobs", default=16, show_default=True)
def compare(pack_dir: Path, sharpened_dir: Path, verdicts: Path, out_dir: Path, limit: int, jobs: int) -> None:
    """Текстовые блоки полос с отброшенными сиротами «было | стало»: ``<out-dir>/pages/*.jpg`` и ``compare.csv``."""
    import pandas as pd

    table = pd.read_csv(verdicts).fillna("")
    dropped = table[table.reason != ""].groupby("page")["index"].apply(lambda values: set(int(v) for v in values))
    keys = sorted(dropped.index)[: limit or None]
    tasks = [(key, dropped[key], pack_dir, sharpened_dir, out_dir) for key in keys]
    rows = []
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        for done, row in enumerate(pool.map(_compare_one, tasks, chunksize=2), start=1):
            if row is not None:
                rows.append(row)
            if done % 100 == 0:
                logger.info("Полос %d / %d", done, len(tasks))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "compare.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    changed = sum(1 for r in rows if r["axes_before"] != r["axes_after"] or r["blocks_before"] != r["blocks_after"])
    logger.info("Полос %d, изменились оси или блоки у %d → %s", len(rows), changed, out_dir)
