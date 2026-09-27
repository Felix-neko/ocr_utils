"""Команды стенда: ``detect`` — line art по паку и вырезки, ``ocr`` — tesseract по вырезкам, ``features`` — признаки, ``sheets`` — контактные листы по поясам."""

from __future__ import annotations

import csv
import json
import logging
import random
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import cv2

from research.line_art_titles.detect import CROP_DPI, detect_page, init_worker, load_jobs
from research.line_art_titles.features import region_features
from research.line_art_titles.ocr import read_region
from research.line_art_titles.sheets import write_sheets

logger = logging.getLogger(__name__)

# Пояса главного признака ``text_ink_share`` для контактных листов: [нижняя, верхняя).
BANDS = ((0.0, 0.05), (0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01))


def _detect_one(job, cache_root: Path, crops_dir: Path) -> tuple[str, list[dict], str | None]:
    """Обёртка для пула: одна битая полоса не валит прогон."""
    try:
        return job.name, detect_page(job, cache_root, crops_dir), None
    except Exception as error:  # noqa: BLE001
        return job.name, [], f"{type(error).__name__}: {error}"


def _ocr_one(row: dict, crops_dir: Path) -> dict:
    """Обёртка для пула: слова tesseract одной вырезки."""
    return {"id": row["id"], **read_region(str(crops_dir / row["crop"]))}


def _read_jsonl(path: Path) -> list[dict]:
    """Строки JSONL-файла; недописанная последняя строка (файл ещё пишет воркер) пропускается."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("%s: недописанная строка пропущена", path.name)
    return rows


def _pool(jobs: int) -> ProcessPoolExecutor:
    """Пул процессов forkserver с инициализатором (hugepage и потоки BLAS выключены)."""
    return ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker)


@click.group()
def main() -> None:
    """Стенд «стилизованный заголовок или рисунок» по находкам детектора line art."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@main.command("detect")
@click.option("--db", "db_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--sharpened-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--layout-cache", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--jobs", default=16, show_default=True, type=int, help="Воркеров (CPU; SSD диск не держит).")
@click.option("--limit", default=None, type=int, help="Только первые N полос (случайная выборка с семенем 0).")
@click.option(
    "--pages",
    "pages_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Только эти полосы (файл: по имени «год/выпуск/полоса» на строку) → regions_pages.jsonl, а не regions.jsonl",
)
def detect_command(
    db_path: Path,
    sharpened_dir: Path,
    layout_cache: Path,
    out_dir: Path,
    jobs: int,
    limit: int | None,
    pages_file: Path | None,
):
    """Line art по всем полосам пака: ``regions.jsonl``, вырезки в ``crops/``, ошибки в ``errors.txt``."""
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_jobs(db_path, sharpened_dir)
    if limit is not None:
        tasks = random.Random(0).sample(tasks, min(limit, len(tasks)))
    if pages_file is not None:
        wanted = {line.strip() for line in pages_file.read_text().splitlines() if line.strip()}
        tasks = [job for job in tasks if job.name in wanted]
    logger.info("Полос: %d, воркеров: %d", len(tasks), jobs)
    regions = errors = 0
    # Частичный прогон (--pages) пишет в свой файл: полный regions.jsonl им перезаписывать нельзя.
    regions_path = out_dir / ("regions_pages.jsonl" if pages_file else "regions.jsonl")
    with _pool(jobs) as pool, regions_path.open("w") as out, (out_dir / "errors.txt").open("w") as err:
        futures = [pool.submit(_detect_one, job, layout_cache, crops_dir) for job in tasks]
        for done, future in enumerate(futures, 1):
            name, rows, error = future.result()
            if error:
                errors += 1
                err.write(f"{name}\t{error}\n")
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            regions += len(rows)
            if done % 500 == 0:
                logger.info("%d/%d полос, областей %d, ошибок %d", done, len(tasks), regions, errors)
    logger.info("Готово: полос %d, областей %d, ошибок %d", len(tasks), regions, errors)


def _recrop_one(page: str, rows: list[dict], sharpened_dir: Path, crops_dir: Path, pad_mm: float, side_pad_mm: float):
    """Обёртка для пула: вырезки одной полосы с новым полем."""
    from research.line_art_titles.detect import recrop_page

    return recrop_page(page, rows, sharpened_dir, crops_dir, pad_mm, side_pad_mm)


@main.command("recrop")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--sharpened-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--pad-mm", default=2.0, show_default=True, type=float, help="Поле сверху и снизу, мм")
@click.option("--side-pad-mm", default=2.0, show_default=True, type=float, help="Поле слева и справа, мм")
@click.option("--jobs", default=16, show_default=True, type=int)
def recrop_command(out_dir: Path, sharpened_dir: Path, pad_mm: float, side_pad_mm: float, jobs: int) -> None:
    """Пересоздать вырезки всех областей ``regions.jsonl`` с другим полем (рамки и id те же) и обновить ``crop_inner``.

    Всё, что читало вырезки (``ocr.jsonl``, ``deepseek*_*.jsonl``, признаки, оверлеи), после этого
    надо пересчитать.
    """
    rows = _read_jsonl(out_dir / "regions.jsonl")
    by_page: dict[str, list[dict]] = {}
    for row in rows:
        by_page.setdefault(row["page"], []).append(row)
    pages = sorted(by_page)
    with _pool(jobs) as pool:
        results = pool.map(
            _recrop_one,
            pages,
            [by_page[p] for p in pages],
            [sharpened_dir] * len(pages),
            [out_dir / "crops"] * len(pages),
            [pad_mm] * len(pages),
            [side_pad_mm] * len(pages),
        )
        updated = {row["id"]: row for page_rows in results for row in page_rows}
    with (out_dir / "regions.jsonl").open("w") as out:
        for row in rows:
            out.write(json.dumps(updated[row["id"]], ensure_ascii=False) + "\n")
    logger.info(
        "Вырезки: %d областей на %d полосах, поле %.1f мм сверху/снизу, %.1f мм по бокам",
        len(rows),
        len(pages),
        pad_mm,
        side_pad_mm,
    )


def _hints_one(job, cache_root: Path) -> dict:
    """Обёртка для пула: вход и выход детектора на одной полосе; ошибка не валит прогон."""
    from research.line_art_titles.detect import page_hints

    try:
        return page_hints(job, cache_root)
    except Exception as error:  # noqa: BLE001
        return {"page": job.name, "error": f"{type(error).__name__}: {error}"}


@main.command("hints")
@click.option("--db", "db_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--sharpened-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--layout-cache", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--jobs", default=16, show_default=True, type=int)
def hints_command(db_path: Path, sharpened_dir: Path, layout_cache: Path, out_dir: Path, jobs: int) -> None:
    """Подсказки и исключения, которые детектор line art получил на каждой полосе пака → ``hints.jsonl`` и сводка."""
    from collections import Counter

    tasks = load_jobs(db_path, sharpened_dir)
    by_source: Counter = Counter()
    agreed: Counter = Counter()
    pages_with: Counter = Counter()
    errors = 0
    with _pool(jobs) as pool, (out_dir / "hints.jsonl").open("w") as out:
        for record in pool.map(_hints_one, tasks, [layout_cache] * len(tasks), chunksize=4):
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if "error" in record:
                errors += 1
                continue
            for hint in record["hints"]:
                by_source[hint["source"]] += 1
                agreed[hint["source"]] += hint["agrees"]
            for source in {hint["source"] for hint in record["hints"]}:
                pages_with[source] += 1
    total = sum(by_source.values())
    click.echo(
        f"Полос {len(tasks)}, ошибок {errors}; подсказок всего {total}, легли на итоговую область {sum(agreed.values())}"
    )
    for source, count in by_source.most_common():
        click.echo(f"  {source:16s} {count:6d} (на {pages_with[source]} полосах), легли на область {agreed[source]}")


def _formula_crops_one(record: dict, sharpened_dir: Path, crops_dir: Path) -> list[dict]:
    """Обёртка для пула: вырезки всех формул одной полосы (рамка после достройки, поле 2 мм)."""
    from ocr_utils.page_layout.geometry import Box, iou
    from ocr_utils.page_layout.image import PageImage, Variant
    from research.line_art_titles.detect import crop_region

    image = PageImage.from_file(
        sharpened_dir / f"{record['page']}.jpg", Variant.SHARPENED, record["page"], default_dpi=600
    )
    gray = image.gray
    raw = [Box(*e["box"]) for e in record.get("equations_surya", [])]
    rows = []
    for index, formula in enumerate(record["formulas"]):
        box = Box(*formula["box"])
        region_id = f"{record['page'].replace('/', '_')}_f{index}"
        inner = crop_region(gray, box, image.dpi, crops_dir / f"{region_id}.png")
        # Исходная рамка surya — ближайшая по IoU (порядок блоков тот же, но надёжнее сверить).
        surya_box = max(raw, key=lambda b: iou(b, box)) if raw else box
        rows.append(
            {
                "id": region_id,
                "page": record["page"],
                "box": list(box.as_tuple()),
                "surya_box": list(surya_box.as_tuple()),
                "confidence": formula.get("confidence"),
                "page_size": record["page_size"],
                "crop": f"{region_id}.png",
                "crop_inner": inner,
            }
        )
    return rows


@main.command("formula-crops")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--sharpened-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--jobs", default=16, show_default=True, type=int)
def formula_crops_command(out_dir: Path, sharpened_dir: Path, jobs: int) -> None:
    """Формулы детектора из ``hints.jsonl`` → ``formulas/regions.jsonl`` и вырезки ``formulas/crops/`` (для DeepSeek)."""
    records = [r for r in _read_jsonl(out_dir / "hints.jsonl") if r.get("formulas")]
    root = out_dir / "formulas"
    (root / "crops").mkdir(parents=True, exist_ok=True)
    with _pool(jobs) as pool:
        results = pool.map(_formula_crops_one, records, [sharpened_dir] * len(records), [root / "crops"] * len(records))
        rows = [row for page_rows in results for row in page_rows]
    with (root / "regions.jsonl").open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Формулы: %d вырезок на %d полосах → %s", len(rows), len(records), root)


def _formula_overlays_one(page: str, rows: list[dict], markdown: dict, labels: dict, sharpened_dir: Path, root: Path):
    """Обёртка для пула: оверлеи всех формул одной полосы; возвращает пары (id, вердикт)."""
    from ocr_utils.page_layout.image import PageImage, Variant
    from research.line_art_titles.formula_overlay import draw_formula, formula_path, verdict_of

    image = PageImage.from_file(sharpened_dir / f"{page}.jpg", Variant.SHARPENED, page, default_dpi=600)
    gray = image.gray
    done = []
    for row in rows:
        record = markdown.get(row["id"], {})
        elements, raw = record.get("elements", []), record.get("raw", "")
        picture = draw_formula(gray, image.dpi, row, elements, raw, labels.get(row["id"]))
        verdict = verdict_of(elements, raw)
        cv2.imwrite(str(formula_path(root, row, verdict)), picture, [cv2.IMWRITE_JPEG_QUALITY, 90])
        done.append((row["id"], verdict.value))
    return done


@main.command("formula-overlays")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--sharpened-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--jobs", default=16, show_default=True, type=int)
def formula_overlays_command(out_dir: Path, sharpened_dir: Path, jobs: int) -> None:
    """Оверлеи всех формул детектора по вердикту DeepSeek → ``formulas/вход/{equation,текст_с_формулой,без_формулы}/``."""
    import shutil

    from research.line_art_titles.formula_overlay import FormulaVerdict

    root = out_dir / "formulas" / "вход"
    shutil.rmtree(root, ignore_errors=True)
    for verdict in FormulaVerdict:
        (root / verdict.value).mkdir(parents=True)
    rows = _read_jsonl(out_dir / "formulas" / "regions.jsonl")
    markdown = {r["id"]: r for r in _read_jsonl(out_dir / "formulas" / "deepseek_vllm_markdown.jsonl")}
    labels_path = out_dir / "formulas" / "labels_formulas.csv"
    labels = {r["id"]: r["label"] for r in csv.DictReader(labels_path.open())} if labels_path.is_file() else {}
    by_page: dict[str, list[dict]] = {}
    for row in rows:
        by_page.setdefault(row["page"], []).append(row)
    pages = sorted(by_page)
    with _pool(jobs) as pool:
        results = pool.map(
            _formula_overlays_one,
            pages,
            [by_page[p] for p in pages],
            [{r["id"]: markdown.get(r["id"], {}) for r in by_page[p]} for p in pages],
            [labels] * len(pages),
            [sharpened_dir] * len(pages),
            [root] * len(pages),
        )
        pairs = [pair for page_pairs in results for pair in page_pairs]
    with (root / "index.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "verdict"])
        writer.writerows(sorted(pairs))
    from collections import Counter

    logger.info("Оверлеи формул: %s → %s", dict(Counter(v for _, v in pairs)), root)


def _merge_one(
    page: str,
    rows: list[dict],
    features: dict,
    blocks: dict,
    barriers: list,
    labels: dict,
    sharpened_dir: Path,
    root: Path,
):
    """Обёртка для пула: слияние и оверлеи всех областей одной полосы."""
    from ocr_utils.page_layout.geometry import Box
    from ocr_utils.page_layout.image import PageImage, Variant
    from research.line_art_titles.merge import draw_merged, folder_of, merge_region

    image = PageImage.from_file(sharpened_dir / f"{page}.jpg", Variant.SHARPENED, page, default_dpi=600)
    ink = image.bitonal_at(image.dpi) == 0
    gray = image.gray
    walls = [Box(x0, y0, max(x0, x1), max(y0, y1)) for x0, y0, x1, y1, _ in barriers]
    results = []
    for row in rows:
        own = blocks.get(row["id"], [])
        result = merge_region(row, features[row["id"]], own, ink, walls)
        folder = root / folder_of(result)
        folder.mkdir(parents=True, exist_ok=True)
        picture = draw_merged(gray, image.dpi, row, own, result, labels.get(row["id"]))
        cv2.imwrite(str(folder / f"{row['id']}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
        results.append(result)
    return results


@main.command("merge")
@click.option("--db", "db_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--sharpened-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--prefix", default="deepseek_vllm", show_default=True, help="Префикс вывода DeepSeek и признаков")
@click.option("--jobs", default=16, show_default=True, type=int)
def merge_command(db_path: Path, out_dir: Path, sharpened_dir: Path, prefix: str, jobs: int) -> None:
    """Слияние областей line art с вердиктом DeepSeek → ``merged.jsonl`` и оверлеи ``merged/<класс>/``."""
    import shutil
    from collections import Counter

    rows = _read_jsonl(out_dir / "regions.jsonl")
    features = {r["id"]: r for r in csv.DictReader((out_dir / f"features_{prefix}.csv").open())}
    blocks = {r["id"]: r.get("elements", []) for r in _read_jsonl(out_dir / f"{prefix}_markdown.jsonl")}
    jobs_by_page = {job.name: job for job in load_jobs(db_path, sharpened_dir)}
    labels = _labels(out_dir)
    root = out_dir / "merged"
    shutil.rmtree(root, ignore_errors=True)
    by_page: dict[str, list[dict]] = {}
    for row in rows:
        by_page.setdefault(row["page"], []).append(row)
    pages = sorted(by_page)
    with _pool(jobs) as pool:
        results = pool.map(
            _merge_one,
            pages,
            [by_page[p] for p in pages],
            [{r["id"]: features[r["id"]] for r in by_page[p]} for p in pages],
            [{r["id"]: blocks.get(r["id"], []) for r in by_page[p]} for p in pages],
            [jobs_by_page[p].known_raster + jobs_by_page[p].known_tables for p in pages],
            [labels] * len(pages),
            [sharpened_dir] * len(pages),
            [root] * len(pages),
        )
        merged = [r for page_results in results for r in page_results]
    with (out_dir / "merged.jsonl").open("w") as out:
        for result in merged:
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
    from research.line_art_titles.merge import folder_of

    logger.info("Слияние: %s → %s", dict(Counter(folder_of(r) for r in merged)), root)


def _pass2_prepare_one(row: dict, words: list[dict], src: Path, dst: Path) -> list:
    """Обёртка для пула: залитая вырезка одной области; возвращает залитые рамки слов."""
    from research.line_art_titles.pass2 import fill_words

    gray = cv2.imread(str(src / row["crop"]), cv2.IMREAD_GRAYSCALE)
    binary, boxes = fill_words(gray, row["crop_inner"], words)
    cv2.imwrite(str(dst / row["crop"]), binary)
    return [list(b) for b in boxes]


@main.command("pass2-prepare")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--prefix", default="deepseek_vllm", show_default=True)
@click.option("--jobs", default=16, show_default=True, type=int)
def pass2_prepare_command(out_dir: Path, prefix: str, jobs: int) -> None:
    """Области, где у DeepSeek только текст (итог слияния «надпись»/«неясно»): залить слова → ``pass2/crops/``, ``pass2/regions.jsonl``."""
    merged = {r["id"]: r for r in _read_jsonl(out_dir / "merged.jsonl")}
    words = {r["id"]: r.get("elements", []) for r in _read_jsonl(out_dir / f"{prefix}_ocr.jsonl")}
    rows = [r for r in _read_jsonl(out_dir / "regions.jsonl") if merged[r["id"]]["verdict"] is not None]
    root = out_dir / "pass2"
    (root / "crops").mkdir(parents=True, exist_ok=True)
    with _pool(jobs) as pool:
        filled = list(
            pool.map(
                _pass2_prepare_one,
                rows,
                [words.get(r["id"], []) for r in rows],
                [out_dir / "crops"] * len(rows),
                [root / "crops"] * len(rows),
                chunksize=8,
            )
        )
    with (root / "regions.jsonl").open("w") as out:
        for row, boxes in zip(rows, filled):
            out.write(json.dumps({**row, "filled_words": boxes}, ensure_ascii=False) + "\n")
    logger.info("Второй проход: %d областей залито → %s", len(rows), root)


def _pass2_finish_one(row: dict, blocks: list[dict], label: str | None, out_dir: Path) -> dict:
    """Обёртка для пула: классика, вердикт и оверлей второго прохода по одной области."""
    from research.line_art_titles.pass2 import classic_boxes, draw_pass2, verdict_pass2

    gray = cv2.imread(str(out_dir / "crops" / row["crop"]), cv2.IMREAD_GRAYSCALE)
    binary = cv2.imread(str(out_dir / "pass2" / "crops" / row["crop"]), cv2.IMREAD_GRAYSCALE)
    classic = classic_boxes(binary, row["crop_inner"])
    result = verdict_pass2(binary, row["crop_inner"], blocks, classic)
    picture = draw_pass2(gray, binary, row, row["filled_words"], blocks, classic, result, label)
    cv2.imwrite(
        str(out_dir / "pass2" / "overlays" / result["verdict"] / f"{row['id']}.jpg"),
        picture,
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )
    return {"id": row["id"], **result, "classic": [list(c) for c in classic], "blocks": [b["label"] for b in blocks]}


@main.command("pass2-finish")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--jobs", default=16, show_default=True, type=int)
def pass2_finish_command(out_dir: Path, jobs: int) -> None:
    """Классика + вердикт по выводу DeepSeek на залитых вырезках → ``pass2/result.jsonl`` и ``pass2/overlays/{объект,надпись}/``."""
    import shutil
    from collections import Counter

    from research.line_art_titles.pass2 import Pass2Verdict

    root = out_dir / "pass2"
    shutil.rmtree(root / "overlays", ignore_errors=True)
    for verdict in Pass2Verdict:
        (root / "overlays" / verdict.value).mkdir(parents=True)
    rows = _read_jsonl(root / "regions.jsonl")
    blocks = {r["id"]: r.get("elements", []) for r in _read_jsonl(root / "deepseek_vllm_markdown.jsonl")}
    labels = _labels(out_dir)
    with _pool(jobs) as pool:
        results = list(
            pool.map(
                _pass2_finish_one,
                rows,
                [blocks.get(r["id"], []) for r in rows],
                [labels.get(r["id"]) for r in rows],
                [out_dir] * len(rows),
                chunksize=8,
            )
        )
    with (root / "result.jsonl").open("w") as out:
        for result in results:
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info("Второй проход: %s", dict(Counter(r["verdict"] for r in results)))


@main.command("ocr")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--jobs", default=16, show_default=True, type=int)
def ocr_command(out_dir: Path, jobs: int) -> None:
    """Tesseract по всем вырезкам ``regions.jsonl`` → ``ocr.jsonl``."""
    rows = _read_jsonl(out_dir / "regions.jsonl")
    with _pool(jobs) as pool, (out_dir / "ocr.jsonl").open("w") as out:
        for done, result in enumerate(pool.map(_ocr_one, rows, [out_dir / "crops"] * len(rows), chunksize=4), 1):
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            if done % 200 == 0:
                logger.info("%d/%d вырезок", done, len(rows))
    logger.info("OCR: %d вырезок", len(rows))


@main.command("features")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def features_command(out_dir: Path) -> None:
    """Признаки по каждой области → ``features.csv``."""
    ocr = {r["id"]: r for r in _read_jsonl(out_dir / "ocr.jsonl")}
    table = []
    for row in _read_jsonl(out_dir / "regions.jsonl"):
        gray = cv2.imread(str(out_dir / "crops" / row["crop"]), cv2.IMREAD_GRAYSCALE)
        words = {key: value for key, value in ocr[row["id"]].items() if key.startswith("psm")}
        table.append(region_features(row, words, gray, CROP_DPI))
    with (out_dir / "features.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    logger.info("Признаки: %d областей → %s", len(table), out_dir / "features.csv")


@main.command("sheets")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--per-band", default=60, show_default=True, type=int, help="Сколько областей на пояс (случайно).")
def sheets_command(out_dir: Path, per_band: int) -> None:
    """Контактные листы по поясам ``text_ink_share`` → ``sheets/``, опись клеток — ``sheets/index.csv``."""
    features = {r["id"]: r for r in csv.DictReader((out_dir / "features.csv").open())}
    ocr = {r["id"]: r for r in _read_jsonl(out_dir / "ocr.jsonl")}
    regions = {r["id"]: r for r in _read_jsonl(out_dir / "regions.jsonl")}
    sheets_dir = out_dir / "sheets"
    index = []
    for low, high in BANDS:
        ids = sorted(i for i, f in features.items() if low <= float(f["text_ink_share"]) < high)
        chosen = sorted(random.Random(0).sample(ids, min(per_band, len(ids))))
        items = []
        for number, region_id in enumerate(chosen, 1):
            f = features[region_id]
            gray = cv2.imread(str(out_dir / "crops" / regions[region_id]["crop"]), cv2.IMREAD_GRAYSCALE)
            caption = f"текст {float(f['text_ink_share']):.2f} линии {float(f['long_stroke_share']):.2f} {f['kind']}"
            items.append((gray, regions[region_id], ocr[region_id][f["best_psm"]], caption))
            index.append({"band": f"{low:.2f}-{min(high, 1):.2f}", "cell": number, "id": region_id})
        prefix = f"текст_{low:.2f}-{min(high, 1):.2f}"
        paths = write_sheets(items, sheets_dir, prefix) if items else []
        logger.info("Пояс %s: областей %d, на листах %d, листов %d", prefix, len(ids), len(items), len(paths))
    with (sheets_dir / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["band", "cell", "id"])
        writer.writeheader()
        writer.writerows(index)


# Классы ручной разметки: надписи — T (только текст) и B (рубрика-вензель: текст + буквица,
# линейка, рамка); прочее — C (схема, график, рисунок с подписями), D (рисунок без текста),
# F (бланк), X (таблица, формула, тень, пустая рамка, обрывок линии или фото).
TITLE_LABELS = ("T", "B")


class Engine(str, Enum):
    """Движок распознавания, по словам которого судит правило «надпись»."""

    TESSERACT = "tesseract"
    DEEPSEEK = "deepseek"  # DeepSeek-OCR-2 через HF remote code, по одной картинке
    DEEPSEEK_VLLM = "deepseek_vllm"  # то же через vLLM батчем


# Файлы признаков и правило по движку.
FEATURES_FILE = {
    Engine.TESSERACT: "features.csv",
    Engine.DEEPSEEK: "features_deepseek.csv",
    Engine.DEEPSEEK_VLLM: "features_deepseek_vllm.csv",
}

# Для DeepSeek: префикс файлов вывода модели, папка оверлеев и имя движка в шапке и легенде.
DEEPSEEK_RUNS = {
    Engine.DEEPSEEK: ("deepseek", "overlays_deepseek", "DeepSeek-OCR-2 (HF)"),
    Engine.DEEPSEEK_VLLM: ("deepseek_vllm", "overlays_deepseek_vllm", "DeepSeek-OCR-2 (vLLM)"),
}


def _rule(engine: Engine):
    """Функция-правило «надпись» для движка."""
    from research.line_art_titles.features import is_title_like, is_title_like_deepseek

    return is_title_like if engine is Engine.TESSERACT else is_title_like_deepseek


def _labels(out_dir: Path) -> dict[str, str]:
    """Ручная разметка: ``labels.csv`` (выборка по поясам) и ``labels_holdout.csv`` (отложенная), если есть."""
    labels: dict[str, str] = {}
    for name in ("labels.csv", "labels_holdout.csv"):
        path = out_dir / name
        if path.is_file():
            labels.update({r["id"]: r["label"] for r in csv.DictReader(path.open())})
    return labels


@main.command("deepseek-features")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--prefix",
    default="deepseek",
    show_default=True,
    help="Начало имён входов (<prefix>_markdown.jsonl, <prefix>_ocr.jsonl) и выхода features_<prefix>.csv",
)
def deepseek_features_command(out_dir: Path, prefix: str) -> None:
    """Признаки tesseract-прогона + признаки DeepSeek-OCR-2 (``<prefix>_markdown.jsonl``, ``<prefix>_ocr.jsonl``) → ``features_<prefix>.csv``."""
    from research.line_art_titles.features import deepseek_features, ink_blobs

    base = {r["id"]: r for r in csv.DictReader((out_dir / "features.csv").open())}
    markdown = {r["id"]: r.get("elements", []) for r in _read_jsonl(out_dir / f"{prefix}_markdown.jsonl")}
    words = {r["id"]: r.get("elements", []) for r in _read_jsonl(out_dir / f"{prefix}_ocr.jsonl")}
    table = []
    for row in _read_jsonl(out_dir / "regions.jsonl"):
        gray = cv2.imread(str(out_dir / "crops" / row["crop"]), cv2.IMREAD_GRAYSCALE)
        blobs = ink_blobs(gray, row["crop_inner"], CROP_DPI)
        extra = deepseek_features(blobs, markdown.get(row["id"], []), words.get(row["id"], []), row["crop_inner"])
        table.append({**base[row["id"]], **extra})
    with (out_dir / f"features_{prefix}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    missing = sum(1 for r in _read_jsonl(out_dir / "regions.jsonl") if r["id"] not in words or r["id"] not in markdown)
    logger.info("Признаки %s: %d областей, без вывода модели %d → features_%s.csv", prefix, len(table), missing, prefix)


@main.command("evaluate")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--engine", type=click.Choice([e.value for e in Engine]), default=Engine.TESSERACT.value, show_default=True
)
def evaluate_command(out_dir: Path, engine: str) -> None:
    """Правило «надпись» движка против ручной разметки (выборка по поясам и отложенная) и оценка по паку."""
    from collections import Counter

    rule = _rule(Engine(engine))
    features = {r["id"]: r for r in csv.DictReader((out_dir / FEATURES_FILE[Engine(engine)]).open())}
    for name in ("labels.csv", "labels_holdout.csv"):
        path = out_dir / name
        if not path.is_file():
            continue
        labels = list(csv.DictReader(path.open()))
        matrix: Counter = Counter()
        for row in labels:
            matrix[(row["label"], rule(features[row["id"]]))] += 1
        tp = sum(v for (label, said), v in matrix.items() if label in TITLE_LABELS and said)
        fp = sum(v for (label, said), v in matrix.items() if label not in TITLE_LABELS and said)
        fn = sum(v for (label, said), v in matrix.items() if label in TITLE_LABELS and not said)
        click.echo(
            f"{name} ({len(labels)}): TP {tp}, FP {fp}, FN {fn}, "
            f"точность {tp / max(1, tp + fp):.3f}, полнота {tp / max(1, tp + fn):.3f}"
        )
        for label in sorted({label for label, _ in matrix}):
            click.echo(f"  {label}: надпись {matrix[(label, True)]}, не надпись {matrix[(label, False)]}")
    click.echo(f"Пак: областей {len(features)}, правило «надпись» — {sum(rule(f) for f in features.values())}")


def _common_notes(features: dict) -> str:
    """Справочная строка шапки оверлея: пятна, ширина, вид и источники детектора."""
    return (
        f"пятен {features['cc_count']}, ширина {float(features['width_mm']):.0f} мм, "
        f"вид «{features['kind']}», источники: {features['sources']}"
    )


def _tesseract_explained(gray, row: dict, features: dict, ocr: dict) -> tuple[list[dict], list, list[str]]:
    """Слова, условия и справочные строки оверлея для правила tesseract (:func:`features.is_title_like`).

    Args:
        gray: Серая вырезка.
        row: Строка ``regions.jsonl``.
        features: Строка ``features.csv``.
        ocr: Слова tesseract по режимам.

    Returns:
        (слова из букв лучшего режима, условия (подпись, выполнено), справочные строки).
    """
    from research.line_art_titles.features import (
        FEW_BLOBS,
        TITLE_MAX_HEIGHT_MM,
        TITLE_MIN_LETTER_SHARE,
        TITLE_MIN_WEAK_TEXT_SHARE,
        ink_blobs,
        title_letters,
        weak_words,
    )

    _, words = weak_words(ink_blobs(gray, row["crop_inner"], CROP_DPI), ocr)
    letters, height, weak = (
        title_letters(features),
        float(features["height_mm"]),
        float(features["weak_text_ink_share"]),
    )
    which = "по всем пятнам (их ≤ 3)" if int(features["cc_count"]) <= FEW_BLOBS else "без крупнейшего пятна"
    conditions = [
        (f"буквы {which} {letters:.2f} (порог ≥ {TITLE_MIN_LETTER_SHARE})", letters >= TITLE_MIN_LETTER_SHARE),
        (f"высота {height:.1f} мм (порог ≤ {TITLE_MAX_HEIGHT_MM:.0f})", height <= TITLE_MAX_HEIGHT_MM),
        (
            f"краска под словами из букв {weak:.2f} (порог ≥ {TITLE_MIN_WEAK_TEXT_SHARE})",
            weak >= TITLE_MIN_WEAK_TEXT_SHARE,
        ),
    ]
    return words, conditions, [_common_notes(features)]


def _deepseek_explained(features: dict) -> tuple[list, list[str]]:
    """Условия и справочные строки оверлея для блочного правила DeepSeek (:func:`features.is_title_like_deepseek`).

    Args:
        features: Строка ``features_deepseek.csv``.

    Returns:
        (условия (подпись, выполнено), справочные строки: буквы и слова DeepSeek, текст блоков).
    """
    from research.line_art_titles.features import TITLE_MAX_HEIGHT_MM, title_letters

    labels = features["ds_block_labels"] or "—"
    text_share = float(features["ds_text_block_ink_share"])
    height = float(features["height_mm"])
    conditions = [
        (f"нет блоков image / table / equation (блоки: {labels})", int(features["ds_has_non_text_block"]) == 0),
        (f"есть текстовый блок над краской (под ним {text_share:.2f} краски)", text_share > 0),
        (f"высота {height:.1f} мм (порог ≤ {TITLE_MAX_HEIGHT_MM:.0f})", height <= TITLE_MAX_HEIGHT_MM),
    ]
    notes = [
        f"справочно: буквы {title_letters(features):.2f}, краска под словами DeepSeek "
        f"{float(features['ds_word_ink_share']):.2f} (слов {features['ds_words']})",
        _common_notes(features),
    ]
    if features.get("ds_text"):
        notes.append(f"DeepSeek: {features['ds_text'][:110]}")
    return conditions, notes


def _overlay_one(
    row: dict,
    features: dict,
    ocr: dict,
    markdown: list[dict],
    label: str | None,
    out_dir: Path,
    engine: str,
    root: Path,
) -> tuple[str, bool]:
    """Обёртка для пула: оверлей одной области движка, запись в ``root/<вердикт>/``."""
    from research.line_art_titles.overlay import draw_region, overlay_path

    gray = cv2.imread(str(out_dir / "crops" / row["crop"]), cv2.IMREAD_GRAYSCALE)
    title = _rule(Engine(engine))(features)
    if Engine(engine) is Engine.TESSERACT:
        words, conditions, notes = _tesseract_explained(gray, row, features, ocr)
        blocks, name = [], "tesseract"
    else:
        words = ocr.get("elements", [])
        conditions, notes = _deepseek_explained(features)
        blocks, name = markdown, DEEPSEEK_RUNS[Engine(engine)][2]
    image = draw_region(gray, row, features, words, CROP_DPI, label, title, conditions, name, blocks, notes)
    cv2.imwrite(str(overlay_path(root, row, title)), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return row["id"], title


@main.command("overlays")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--engine", type=click.Choice([e.value for e in Engine]), default=Engine.TESSERACT.value, show_default=True
)
@click.option("--jobs", default=16, show_default=True, type=int)
@click.option(
    "--partial",
    is_flag=True,
    help="Только области, по которым DeepSeek уже отработал оба промпта (прогон ещё идёт) → overlays_deepseek_partial/",
)
def overlays_command(out_dir: Path, engine: str, jobs: int, partial: bool) -> None:
    """Оверлей каждой области → ``overlays[_deepseek]/надпись`` и ``…/не_надпись``; опись — ``index.csv`` там же."""
    import shutil

    chosen = Engine(engine)
    features = {r["id"]: r for r in csv.DictReader((out_dir / FEATURES_FILE[chosen]).open())}
    if chosen is Engine.TESSERACT:
        ocr = {r["id"]: {k: v for k, v in r.items() if k.startswith("psm")} for r in _read_jsonl(out_dir / "ocr.jsonl")}
        markdown: dict[str, list] = {}
        root = out_dir / "overlays"
    else:
        prefix, folder, _ = DEEPSEEK_RUNS[chosen]
        ocr = {r["id"]: r for r in _read_jsonl(out_dir / f"{prefix}_ocr.jsonl")}
        markdown = {r["id"]: r.get("elements", []) for r in _read_jsonl(out_dir / f"{prefix}_markdown.jsonl")}
        root = out_dir / (f"{folder}_partial" if partial else folder)
    labels = _labels(out_dir)
    rows = _read_jsonl(out_dir / "regions.jsonl")
    if partial:
        rows = [row for row in rows if row["id"] in ocr and row["id"] in markdown]
    # Папки пересоздаются: иначе после смены правила в старой папке остались бы чужие картинки.
    shutil.rmtree(root, ignore_errors=True)
    for folder in ("надпись", "не_надпись"):
        (root / folder).mkdir(parents=True)
    tasks = [
        (
            row,
            features[row["id"]],
            ocr.get(row["id"], {}),
            markdown.get(row["id"], []),
            labels.get(row["id"]),
            out_dir,
            engine,
            root,
        )
        for row in rows
    ]
    with _pool(jobs) as pool:
        results = list(pool.map(_overlay_one, *zip(*tasks), chunksize=8))
    with (root / "index.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "page", "verdict", "manual_label"])
        for (region_id, title), row in zip(results, rows):
            writer.writerow([region_id, row["page"], "надпись" if title else "не_надпись", labels.get(region_id, "")])
    titles = sum(title for _, title in results)
    logger.info(
        "Оверлеи %s: %d областей — надпись %d, не надпись %d → %s",
        engine,
        len(results),
        titles,
        len(results) - titles,
        root,
    )


def _box_iou(first: dict, second: dict) -> float:
    """IoU двух рамок-словарей ``x0, y0, x1, y1``."""
    width = min(first["x1"], second["x1"]) - max(first["x0"], second["x0"])
    height = min(first["y1"], second["y1"]) - max(first["y0"], second["y0"])
    if width <= 0 or height <= 0:
        return 0.0
    common = width * height
    area = lambda b: max(0, b["x1"] - b["x0"]) * max(0, b["y1"] - b["y0"])  # noqa: E731
    return common / max(1, area(first) + area(second) - common)


@main.command("compare-deepseek")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--other", default="deepseek_vllm", show_default=True, help="Префикс второго прогона")
def compare_deepseek_command(out_dir: Path, other: str) -> None:
    """Сверка двух прогонов DeepSeek-OCR-2 (HF по одной картинке против ``<other>``): вывод, рамки, признаки, вердикты."""
    import statistics

    from research.line_art_titles.features import is_title_like_deepseek

    for prompt in ("markdown", "ocr"):
        first = {r["id"]: r for r in _read_jsonl(out_dir / f"deepseek_{prompt}.jsonl")}
        second = {r["id"]: r for r in _read_jsonl(out_dir / f"{other}_{prompt}.jsonl")}
        common = sorted(set(first) & set(second))
        same_raw = sum(first[i].get("raw") == second[i].get("raw") for i in common)
        same_count = sum(len(first[i].get("elements", [])) == len(second[i].get("elements", [])) for i in common)
        # Рамки: каждой рамке первого прогона — лучшая по IoU рамка второго.
        ious = [
            max((_box_iou(a, b) for b in second[i].get("elements", [])), default=0.0)
            for i in common
            for a in first[i].get("elements", [])
        ]
        texts = sum(
            [e["text"] for e in first[i].get("elements", [])] == [e["text"] for e in second[i].get("elements", [])]
            for i in common
        )
        click.echo(
            f"{prompt}: областей {len(common)}; сырой вывод совпал {same_raw}, число рамок совпало {same_count}, "
            f"тексты рамок совпали {texts}; IoU рамок: медиана {statistics.median(ious) if ious else 0:.3f}, "
            f"доля ≥ 0.9 {sum(v >= 0.9 for v in ious) / max(1, len(ious)):.3f} из {len(ious)}"
        )
    base = {r["id"]: r for r in csv.DictReader((out_dir / "features_deepseek.csv").open())}
    alt = {r["id"]: r for r in csv.DictReader((out_dir / f"features_{other}.csv").open())}
    diffs = [abs(float(base[i]["ds_word_ink_share"]) - float(alt[i]["ds_word_ink_share"])) for i in base if i in alt]
    flips = [i for i in base if i in alt and is_title_like_deepseek(base[i]) != is_title_like_deepseek(alt[i])]
    click.echo(
        f"ds_word_ink_share: |разница| медиана {statistics.median(diffs):.4f}, > 0.1 у {sum(d > 0.1 for d in diffs)}; "
        f"смен вердикта «надпись» {len(flips)} из {len(diffs)}"
    )
    (out_dir / f"flips_{other}.txt").write_text("\n".join(flips))
