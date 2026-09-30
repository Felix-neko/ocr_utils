"""CLI стенда v17–v18: ``measure`` (пул по страницам, кэш), ``report`` (вердикты, распределения, сравнение с v14/v16 и эталоном), ``sheets`` (оверлеи с метриками и порогами), ``diff`` (смены вердикта между прогонами), ``export`` (выгрузки по вердиктам и поясам)."""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import numpy as np

from ocr_utils.page_layout.pack_analysis.stages import init_worker
from ocr_utils.geometry_regression.quality.measure import load_cached, measure_cached
from ocr_utils.geometry_regression.quality.scoring import Group, Thresholds17, Verdict
from ocr_utils.geometry_regression.quality.sources import PageRef, list_pages

logger = logging.getLogger(__name__)

# Метрики, распределение которых печатается в отчёте (новые меры и их выигрыши).
REPORT_METRICS = (
    "line_quality_mm",
    "line_quality_gain_mm",
    "lines_quality_gain_mm",
    "edge_quality_mm",
    "edge_quality_raw_mm",
    "edge_quality_gain_mm",
)
PERCENTILES = (50, 90, 95, 98, 99, 99.5)


def _measure_one(args: tuple) -> dict:
    """Мера страницы в воркере (кэш стенда)."""
    ref, layout_root, v16_dir, run_dir, redo, pdf_dirs = args
    return measure_cached(ref, layout_root, v16_dir, run_dir, redo, pdf_dirs)


@click.group()
@click.option("--log-level", default="INFO", show_default=True)
def main(log_level: str) -> None:
    """Стенд v17 детектора порчи геометрии FineReader: качество строк и краёв по text_blocks, группы и совокупность."""
    logging.basicConfig(level=log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option(
    "--layout-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Корень разбора page_layout обоих вариантов (geo/, nogeo/).",
)
@click.option(
    "--v16-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Прогон v16 (кэш метрик и поля смещений).",
)
@click.option(
    "--run-dir", required=True, type=click.Path(file_okay=False, path_type=Path), help="Каталог прогона стенда."
)
@click.option("--pdfs", default=None, help="Только эти PDF через запятую (full_1966_01,...).")
@click.option(
    "--geo-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="PDF с коррекцией: вместе с --nogeo-dir включает меры line art по плотному полю.",
)
@click.option(
    "--nogeo-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="PDF без коррекции.",
)
@click.option("--jobs", default=16, show_default=True, type=int, help="Воркеров пула.")
@click.option("--redo", is_flag=True, help="Пересчитать и страницы с готовым кэшем.")
def measure(
    layout_root: Path,
    v16_dir: Path,
    run_dir: Path,
    pdfs: str | None,
    geo_dir: Path | None,
    nogeo_dir: Path | None,
    jobs: int,
    redo: bool,
) -> None:
    """Мера всех страниц с разбором обоих вариантов (кэш: <run-dir>/cache)."""
    refs = list_pages(layout_root, set(pdfs.split(",")) if pdfs else None)
    logger.info("Страниц с разбором обоих вариантов: %d", len(refs))
    errors = 0
    pdf_dirs = (geo_dir, nogeo_dir) if geo_dir is not None and nogeo_dir is not None else None
    tasks = [(ref, layout_root, v16_dir, run_dir, redo, pdf_dirs) for ref in refs]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for done, payload in enumerate(pool.map(_measure_one, tasks, chunksize=8), 1):
            if "error" in payload:
                errors += 1
                logger.error("%s с.%s: %s", payload["pdf"], payload["page"], payload["error"])
            if done % 1000 == 0:
                logger.info("мера: %d/%d", done, len(tasks))
    logger.info("Готово: %d страниц, ошибок %d", len(tasks), errors)


def _thresholds(
    thr: tuple[str, ...], hard: tuple[str, ...], total: float, min_gain: float, ratio: float
) -> Thresholds17:
    return Thresholds17(Thresholds17.parse_pairs(thr), Thresholds17.parse_pairs(hard), total, min_gain, ratio)


def threshold_options(function):
    """Общие опции порогов у report и sheets."""
    for decorator in reversed(
        [
            click.option("--thr", multiple=True, help="Порог метрики: имя=число (порчи или выигрыша)."),
            click.option("--hard", multiple=True, help="Жёсткий порог метрики порчи: имя=число."),
            click.option("--total", default=2.5, show_default=True, type=float, help="Порог суммы групп S."),
            click.option("--min-gain", default=1.0, show_default=True, type=float),
            click.option("--ratio", default=0.75, show_default=True, type=float),
        ]
    ):
        function = decorator(function)
    return function


def load_run(run_dir: Path) -> list[dict]:
    """Все кэши прогона стенда (без ошибок)."""
    out = []
    for path in sorted((run_dir / "cache").glob("*/p*.json")):
        pdf, page = path.parent.name, int(path.stem[1:])
        payload = load_cached(run_dir, PageRef(pdf, page))
        if payload is not None:
            out.append(payload)
    return out


def load_verdicts(csv_path: Path) -> dict[tuple[str, int], str]:
    """Вердикты прежней версии из ``metrics.csv`` её прогона."""
    if not csv_path.is_file():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return {(row["pdf"], int(row["page"])): row["verdict"] for row in csv.DictReader(handle)}


def load_labels(path: Path) -> dict[tuple[str, int], tuple[str, str]]:
    """Эталон стенда v16 (``research.geometry_regression.report.load_labels``)."""
    from research.geometry_regression.report import load_labels as load

    return load(path) if path.exists() else {}


def _percentiles(values: list[float]) -> str:
    if not values:
        return "—"
    return " ".join(f"p{p:g} {np.percentile(values, p):.2f}" for p in PERCENTILES)


@main.command()
@click.option("--run-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--v14-csv", default=None, type=click.Path(path_type=Path), help="metrics.csv прогона v14.")
@click.option("--v16-csv", default=None, type=click.Path(path_type=Path), help="metrics.csv прогона v16.")
@click.option(
    "--labels", "labels_dir", default=None, type=click.Path(path_type=Path), help="Папка эталона (TSV bad/good)."
)
@threshold_options
def report(run_dir, v14_csv, v16_csv, labels_dir, thr, hard, total, min_gain, ratio) -> None:
    """Вердикты v17 по кэшу прогона: verdicts.csv, report.md; распределения новых мер, смены против v14/v16, эталон."""
    thresholds = _thresholds(thr, hard, total, min_gain, ratio)
    pages = load_run(run_dir)
    v14 = load_verdicts(v14_csv) if v14_csv else {}
    v16 = load_verdicts(v16_csv) if v16_csv else {}
    labels = load_labels(labels_dir) if labels_dir else {}
    rows = []
    for payload in pages:
        key = (payload["pdf"], payload["page"])
        verdict = thresholds.assess(payload["metrics"])
        rows.append((key, payload, verdict))
    lines = ["# Детектор порчи геометрии v17: сводка прогона", "", f"Страниц: {len(rows)}", ""]
    counts = Counter(v.verdict for _, _, v in rows)
    share = 100.0 * counts[Verdict.BAD] / max(1, len(rows))
    lines += [
        f"Вердикты v17: bad {counts[Verdict.BAD]} ({share:.1f} %), mixed {counts[Verdict.MIXED]}, ok {counts[Verdict.OK]}",
        "",
    ]
    rules = Counter(v.rule for _, _, v in rows if v.verdict is not Verdict.OK)
    lines += ["Правила, давшие bad/mixed:", ""] + [f"- {rule.value}: {n}" for rule, n in rules.most_common()] + [""]
    culprits = Counter(v.culprit for _, _, v in rows if v.verdict is Verdict.BAD)
    lines += (
        ["Виновники bad (метрика или группа):", ""] + [f"- {name}: {n}" for name, n in culprits.most_common(25)] + [""]
    )
    lines += ["## Распределения новых мер (по страницам, где мера ненулевая)", ""]
    for name in REPORT_METRICS:
        values = [float(p["metrics"].get(name, 0.0) or 0.0) for _, p, _ in rows]
        nonzero = [v for v in values if v > 0]
        lines.append(f"- `{name}`: ненулевых {len(nonzero)}; {_percentiles(nonzero)}")
    lines += ["", "## Суммы групп", "", f"- Σ: {_percentiles([v.total for _, _, v in rows])}"]
    for group in Group:
        lines.append(f"- {group.value}: {_percentiles([v.groups[group] for _, _, v in rows if v.groups[group] > 0])}")
    for name, old in (("v14", v14), ("v16", v16)):
        if not old:
            continue
        pairs = Counter((old.get(key, "—"), v.verdict.value) for key, _, v in rows)
        lines += [
            "",
            f"## Смены вердикта {name} → v17",
            "",
            "| было \\ стало | ok | mixed | bad |",
            "|---|---|---|---|",
        ]
        for before in ("ok", "mixed", "bad"):
            lines.append(
                f"| {before} | " + " | ".join(str(pairs[(before, after)]) for after in ("ok", "mixed", "bad")) + " |"
            )
    if labels:
        lines += [
            "",
            "## Эталон",
            "",
            "| страница | метка | v17 | правило | виновник | Σ | выигрыш |",
            "|---|---|---|---|---|---|---|",
        ]
        right = 0
        seen = 0
        for key, payload, verdict in rows:
            if key not in labels:
                continue
            seen += 1
            label = labels[key][0]
            ok = (verdict.verdict is Verdict.BAD) == (label == "bad")
            right += ok
            lines.append(
                f"| {key[0]} с.{key[1]} | {label} | {verdict.verdict.value}{'' if ok else ' ✗'} | {verdict.rule.value} | "
                f"{verdict.culprit} | {verdict.total:.2f} | {verdict.gain:.2f} |"
            )
        lines.insert(
            lines.index("## Эталон") + 1, f"\nВерно: {right} из {seen} (эталона в прогоне: {seen} из {len(labels)})\n"
        )
    lines += ["", "## Пороги", "", "```", thresholds.describe(), "```"]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (run_dir / "verdicts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pdf", "page", "verdict", "rule", "culprit", "total", "damage", "worst", "gain", "gain_reason", "v14", "v16",
                         "label", *REPORT_METRICS])  # fmt: skip
        for key, payload, v in rows:
            writer.writerow([*key, v.verdict.value, v.rule.value, v.culprit, f"{v.total:.3f}", f"{v.damage:.3f}", f"{v.worst:.3f}",
                             f"{v.gain:.3f}", v.gain_reason, v14.get(key, ""), v16.get(key, ""), labels.get(key, ("",))[0],
                             *(f"{float(payload['metrics'].get(n, 0.0) or 0.0):.3f}" for n in REPORT_METRICS)])  # fmt: skip
    click.echo("\n".join(lines[:12]))
    click.echo(f"Отчёт: {run_dir / 'report.md'}")


SELECTIONS = ("labels", "changed", "belts", "pages", "lineart")
# Папка вердикта в выгрузке по line art: ok — «good» (так говорит пользователь), mixed и bad — как есть.
VERDICT_FOLDER = {"ok": "good", "mixed": "mixed", "bad": "bad"}


def has_lineart(payload: dict, layout_root: Path) -> bool:
    """Есть ли на странице line art по разбору ``page_layout`` без коррекции: объекты «рисунок» или «неясно».

    Рамки line art v16 сюда не идут: v16 принимал за рисунок и подчёркнутую строку текста (1966/05 с.29 — рамка
    32×6 мм на «объём потребления.» при двух таблицах на полосе), таких страниц в выгрузке было 560 из 1437.

    Args:
        payload: Кэш страницы стенда.
        layout_root: Корень разбора v6 (``nogeo/pages``).

    Returns:
        ``True``, если в разборе B есть объект класса ``measure.LINEART_CLASSES``.
    """
    from ocr_utils.geometry_regression.quality.measure import LINEART_CLASSES

    path = layout_root / "nogeo" / "pages" / f"{payload['pdf']}_p{payload['page'] - 1:04d}.json"
    objects = json.loads(path.read_text(encoding="utf-8")).get("objects", []) if path.is_file() else []
    return any(obj["class"] in LINEART_CLASSES for obj in objects)


def page_classes(payload: dict, layout_root: Path) -> set[str]:
    """Классы объектов разбора ``page_layout`` без коррекции на странице (пустое множество, если разбора нет)."""
    path = layout_root / "nogeo" / "pages" / f"{payload['pdf']}_p{payload['page'] - 1:04d}.json"
    objects = json.loads(path.read_text(encoding="utf-8")).get("objects", []) if path.is_file() else []
    return {obj["class"] for obj in objects}


# Пояса score худшей метрики в выгрузке bad/mixed страниц без line art (как в ``sheets --select belts``).
DAMAGE_BELTS = (0.0, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, float("inf"))
# Метрики в описи выгрузки: вердикт объясняют они и выигрыши.
EXPORT_METRICS = (
    "line_quality_mm", "line_quality_transferred_mm", "edge_quality_mm", "lineart_aad_mm", "lineart_seg_rel_mm",
    "parallel_spread_lineart", "lineart_axis_delta_deg", "lineart_bend_mm", "lineart_shape_mm",
    "lineart_part_turn_deg", "formula_bar_ptp_mm", "formula_skew_mm", "fraction_tilt_mean_delta_deg",
    "lines_quality_gain_mm", "line_quality_gain_mm", "edge_quality_gain_mm",
)  # fmt: skip


def worst_metric(assessment) -> tuple[str, float]:
    """Метрика порчи с наибольшим score (с мягкими) и её score: по ней страница раскладывается в пояс."""
    item = max(assessment.scores.values(), key=lambda score: score.score)
    return item.name, item.score


def _belt(score: float) -> str:
    """Имя пояса score: ``score_1-1.5``, ``score_5-inf``."""
    for lo, hi in zip(DAMAGE_BELTS, DAMAGE_BELTS[1:]):
        if lo <= score < hi:
            return f"score_{lo:g}-{hi:g}"
    return f"score_{DAMAGE_BELTS[-2]:g}-inf"


def _draw_one(args: tuple) -> str:
    """Одна картинка «было | стало» в воркере пула; возвращает путь."""
    from research.geometry_quality.overlay import draw_pair
    import cv2

    payload, thresholds, geo_pdf, nogeo_pdf, target = args
    assessment = thresholds.assess(payload["metrics"])
    picture = draw_pair(payload, assessment, thresholds, geo_pdf, nogeo_pdf)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return str(target)


@main.command()
@click.option("--run-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--geo-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--nogeo-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--select", "selection", default="labels", show_default=True, type=click.Choice(SELECTIONS),
              help="labels — эталон; changed — смены вердикта против --v16-csv; belts — пояса score метрики --metric; "
                   "pages — страницы из --pages; lineart — все страницы с line art, по папкам вердикта и годам.")  # fmt: skip
@click.option("--labels", "labels_dir", default=None, type=click.Path(path_type=Path))
@click.option("--v16-csv", default=None, type=click.Path(path_type=Path))
@click.option("--metric", default="line_quality_mm", show_default=True, help="Метрика поясов (--select belts).")
@click.option("--per-belt", default=8, show_default=True, type=int, help="Страниц на пояс (--select belts).")
@click.option("--limit", default=60, show_default=True, type=int, help="Не больше стольких картинок.")
@click.option("--pages", "pages_text", default="", help="Страницы через запятую: full_1967_01:38,...")
@click.option("--seed", default=0, show_default=True, type=int)
@click.option(
    "--jobs", default=8, show_default=True, type=int, help="Воркеров отрисовки (рендер двух PDF на страницу)."
)
@click.option(
    "--layout-root",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Корень разбора v6 — для отбора --select lineart (рисунки по разбору без коррекции).",
)
@threshold_options
def sheets(run_dir, geo_dir, nogeo_dir, out_dir, selection, labels_dir, v16_csv, metric, per_belt, limit, pages_text, seed,
           jobs, layout_root, thr, hard, total, min_gain, ratio) -> None:  # fmt: skip
    """Оверлеи «было | стало» с метриками и порогами в шапке — по эталону, сменам вердикта, поясам score или line art."""
    thresholds = _thresholds(thr, hard, total, min_gain, ratio)
    pages = {(p["pdf"], p["page"]): p for p in load_run(run_dir)}
    labels = load_labels(labels_dir) if labels_dir else {}
    v16 = load_verdicts(v16_csv) if v16_csv else {}
    chosen: list[tuple[str, tuple[str, int]]] = []  # (подпапка, страница)
    if selection == "labels":
        chosen = [(labels[key][0], key) for key in sorted(labels) if key in pages]
    elif selection == "changed":
        for key, payload in sorted(pages.items()):
            new = thresholds.assess(payload["metrics"]).verdict.value
            old = v16.get(key)
            if old and (old == "bad") != (new == "bad"):
                chosen.append((f"v16_{old}__v17_{new}", key))
    elif selection == "belts":
        spec = thresholds.specs[metric]
        edges = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, float("inf")]
        rng = random.Random(seed)
        for lo, hi in zip(edges, edges[1:]):
            keys = [
                k for k, p in pages.items() if lo <= float(p["metrics"].get(metric, 0.0) or 0.0) / spec.threshold < hi
            ]
            rng.shuffle(keys)
            chosen += [(f"{metric}_score_{lo:g}-{hi:g}", key) for key in sorted(keys[:per_belt])]
    elif selection == "lineart":
        for key, payload in sorted(pages.items()):
            if has_lineart(payload, layout_root):
                verdict = thresholds.assess(payload["metrics"]).verdict.value
                chosen.append((f"{VERDICT_FOLDER[verdict]}/{key[0].split('_')[1]}", key))
    else:
        for item in pages_text.split(","):
            pdf, _, page = item.strip().partition(":")
            if (pdf, int(page)) in pages:
                chosen.append(("pages", (pdf, int(page))))
    chosen = chosen[:limit]
    logger.info("Картинок: %d", len(chosen))
    tasks = []
    for folder, key in chosen:
        payload = dict(pages[key])
        payload["metrics"] = dict(payload["metrics"], v16_verdict=v16.get(key, "—"))
        assessment = thresholds.assess(payload["metrics"])
        culprit = assessment.culprit.replace(" ", "_") if assessment.culprit else "none"
        name = f"{key[0]}_p{key[1]:03d}_{assessment.verdict.value}_{assessment.rule.name.lower()}_{culprit}.jpg"
        tasks.append(
            (payload, thresholds, geo_dir / f"{key[0]}.pdf", nogeo_dir / f"{key[0]}.pdf", out_dir / folder / name)
        )
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for done, _ in enumerate(pool.map(_draw_one, tasks, chunksize=4), 1):
            if done % 200 == 0:
                logger.info("картинок: %d/%d", done, len(tasks))
    logger.info("Оверлеи: %s", out_dir)


@main.command("aad-sheets")
@click.option("--layout-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--v16-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--geo-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--nogeo-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--run-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Прогон стенда: вердикт страницы v17 в шапку.",
)
@click.option(
    "--labels",
    "labels_dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Эталон: листы по всем его страницам с рамками line art (подпапка — метка).",
)
@click.option("--pages", "pages_text", default="", help="Страницы через запятую: full_1967_10:74,...")
@click.option("--mode", default="local", show_default=True, type=click.Choice(["affine", "homography", "local"]))
def aad_sheets(layout_root, v16_dir, geo_dir, nogeo_dir, out_dir, run_dir, labels_dir, pages_text, mode) -> None:
    """Листы AAD по рамкам line art: «B | A в кадре B | карта отклонений», меры, режим выравнивания и порог в шапке."""
    import cv2
    import fitz

    from ocr_utils.geometry_regression.render import render_gray, to_work
    from research.geometry_quality.aad_overlay import aad_sheet
    from ocr_utils.geometry_regression.quality.lineart_flow import Align, align_page, measure_box, merge_boxes
    from ocr_utils.geometry_regression.quality.measure import LINEART_CLASSES
    from ocr_utils.geometry_regression.quality.scoring import METRICS
    from ocr_utils.geometry_regression.quality.sources import load_pair, object_boxes

    threshold = METRICS["lineart_aad_mm"].threshold
    align = Align(mode)
    labels = load_labels(labels_dir) if labels_dir else {}
    chosen = [(labels[key][0], key) for key in sorted(labels)]
    for item in filter(None, (t.strip() for t in pages_text.split(","))):
        pdf, _, page = item.partition(":")
        chosen.append(("pages", (pdf, int(page))))
    thresholds = Thresholds17()
    written = 0
    for folder, (pdf, page) in chosen:
        pair = load_pair(PageRef(pdf, page), layout_root, v16_dir)
        boxes = [list(box) for box in pair.v16_raw.get("lineart", [])] + object_boxes(pair.b, LINEART_CLASSES)
        if not boxes:
            continue
        with fitz.open(str(nogeo_dir / f"{pdf}.pdf")) as nogeo, fitz.open(str(geo_dir / f"{pdf}.pdf")) as geo:
            gray_b, gray_a = to_work(render_gray(nogeo, page - 1)), to_work(render_gray(geo, page - 1))
        aligned, resid = align_page(gray_b, gray_a, pair.field, align)
        verdict = ""
        if run_dir is not None:
            cached = load_cached(run_dir, PageRef(pdf, page))
            if cached is not None:
                result = thresholds.assess(cached["metrics"])
                verdict = (
                    f"{result.verdict.value} ({result.rule.value}{', ' + result.culprit if result.culprit else ''})"
                )
        for index, box in enumerate(merge_boxes(boxes)):
            measure = measure_box(gray_b, aligned, box, pair.field, resid, align is Align.LOCAL, keep_maps=True)
            if measure is None or measure.crop_b is None:
                continue
            title = f"{pdf} с.{page}" + (f" (эталон: {folder})" if folder in ("bad", "good") else "")
            picture = aad_sheet(
                measure, title, threshold, align.value, verdict, METRICS["lineart_seg_rel_mm"].threshold
            )
            target = out_dir / folder / f"{pdf}_p{page:03d}_box{index}_aad{measure.aad_mean_mm:.3f}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), picture, [cv2.IMWRITE_JPEG_QUALITY, 90])
            written += 1
    logger.info("Листов AAD: %d → %s", written, out_dir)


def load_run_any(run_dir: Path) -> dict[tuple[str, int], dict]:
    """Все кэши прогона стенда любой версии (для сравнения прогонов разных версий)."""
    out = {}
    for path in sorted((run_dir / "cache").glob("*/p*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "error" not in payload and "metrics" in payload:
            out[(payload["pdf"], payload["page"])] = payload
    return out


@main.command()
@click.option("--old-run", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Прежний прогон стенда (любой версии кэша).")  # fmt: skip
@click.option("--new-run", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--geo-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--nogeo-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--layout-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Корень разбора v6: страницы с line art по разбору без коррекции.")  # fmt: skip
@click.option("--labels", "labels_dir", default=None, type=click.Path(path_type=Path), help="Эталон — в сводку.")
@click.option("--jobs", default=8, show_default=True, type=int, help="Воркеров отрисовки.")
@threshold_options
def diff(
    old_run, new_run, geo_dir, nogeo_dir, out_dir, layout_root, labels_dir, jobs, thr, hard, total, min_gain, ratio
) -> None:
    """Смены вердикта «bad / не bad» между двумя прогонами при одних порогах: оверлеи нового прогона и diff.csv.

    Оверлеи — в ``<out-dir>/<было>_to_<стало>/<lineart|other>/`` (line art — по разбору v6 без коррекции), в шапке —
    прежний вердикт с правилом и виновником. ``diff.csv`` — все смены, ``diff.md`` — таблица переходов и эталон.
    """
    thresholds = _thresholds(thr, hard, total, min_gain, ratio)
    old_pages, new_pages = load_run_any(old_run), load_run_any(new_run)
    labels = load_labels(labels_dir) if labels_dir else {}
    rows, tasks = [], []
    transitions: Counter = Counter()
    for key, payload in sorted(new_pages.items()):
        if key not in old_pages:
            continue
        before = thresholds.assess(old_pages[key]["metrics"])
        after = thresholds.assess(payload["metrics"])
        lineart = has_lineart(payload, layout_root)
        transitions[("lineart" if lineart else "other", before.verdict.value, after.verdict.value)] += 1
        if (before.verdict is Verdict.BAD) == (after.verdict is Verdict.BAD):
            continue
        folder = f"{before.verdict.value}_to_{after.verdict.value}/{'lineart' if lineart else 'other'}"
        culprit = after.culprit.replace(" ", "_") if after.culprit else "none"
        target = (
            out_dir / folder / f"{key[0]}_p{key[1]:03d}_{after.verdict.value}_{after.rule.name.lower()}_{culprit}.jpg"
        )
        shown = dict(payload)
        shown["previous"] = (f"{old_pages[key].get('version', '?')}: {before.verdict.value} ({before.rule.value}"
                             f"{', ' + before.culprit if before.culprit else ''})")  # fmt: skip
        tasks.append((shown, thresholds, geo_dir / f"{key[0]}.pdf", nogeo_dir / f"{key[0]}.pdf", target))
        rows.append([*key, "lineart" if lineart else "other", before.verdict.value, after.verdict.value, after.rule.value,
                     after.culprit, labels.get(key, ("",))[0], str(target)])  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "diff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pdf", "page", "kind", "old", "new", "rule", "culprit", "label", "picture"])
        writer.writerows(rows)
    lines = [
        f"# Смены вердикта: {old_run.name} → {new_run.name}",
        "",
        "| страницы | было | стало | число |",
        "|---|---|---|---|",
    ]
    for (kind, before, after), count in sorted(transitions.items()):
        if before != after:
            lines.append(f"| {kind} | {before} | {after} | {count} |")
    if labels:
        for name, pages in (("прежний", old_pages), ("новый", new_pages)):
            seen = [k for k in labels if k in pages]
            right = sum(
                (thresholds.assess(pages[k]["metrics"]).verdict is Verdict.BAD) == (labels[k][0] == "bad") for k in seen
            )
            lines.append(f"\nЭталон, {name} прогон: верно {right} из {len(seen)}")
    (out_dir / "diff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Смен bad/не bad: %d; рисую оверлеи", len(tasks))
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        list(pool.map(_draw_one, tasks, chunksize=2))
    click.echo("\n".join(lines))
    logger.info("Сравнение: %s", out_dir)


@main.command()
@click.option("--run-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--geo-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--nogeo-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--layout-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Корень разбора page_layout: line art и формулы — по разбору без коррекции.")  # fmt: skip
@click.option("--mode", required=True, type=click.Choice(["lineart_formula", "damage_belts"]),
              help="lineart_formula — все страницы с line art или формулами, по папкам вердикта и годам; "
                   "damage_belts — страницы без line art с вердиктом bad и mixed, по худшей метрике и поясам score.")  # fmt: skip
@click.option("--jobs", default=8, show_default=True, type=int, help="Воркеров отрисовки.")
@threshold_options
def export(run_dir, geo_dir, nogeo_dir, out_dir, layout_root, mode, jobs, thr, hard, total, min_gain, ratio) -> None:
    """Выгрузка оверлеев «было | стало» по вердиктам с описью index.csv.

    ``lineart_formula``: ``<out>/<good|mixed|bad>/<год>/``, имя файла — страница, вид (``lineart``, ``formula``,
    ``lineart+formula``), вердикт, правило, виновник. ``damage_belts``: ``<out>/<bad|mixed>/<метрика>/<пояс score>/``
    — худшая метрика порчи страницы и пояс её score (``DAMAGE_BELTS``), как пояса ``sheets --select belts``.
    """
    from ocr_utils.geometry_regression.quality.measure import FORMULA_CLASSES, LINEART_CLASSES

    thresholds = _thresholds(thr, hard, total, min_gain, ratio)
    pages = load_run(run_dir)
    rows, tasks = [], []
    for payload in pages:
        key = (payload["pdf"], payload["page"])
        classes = page_classes(payload, layout_root)
        lineart, formula = bool(classes & LINEART_CLASSES), bool(classes & FORMULA_CLASSES)
        assessment = thresholds.assess(payload["metrics"])
        verdict = assessment.verdict.value
        metric, score = worst_metric(assessment)
        if mode == "lineart_formula":
            if not (lineart or formula):
                continue
            kind = "+".join(name for name, flag in (("lineart", lineart), ("formula", formula)) if flag)
            folder = f"{VERDICT_FOLDER[verdict]}/{key[0].split('_')[1]}"
        else:
            if lineart or verdict == "ok":
                continue
            kind = "formula" if formula else "text"
            folder = f"{verdict}/{metric}/{_belt(score)}"
        culprit = assessment.culprit.replace(" ", "_") if assessment.culprit else "none"
        name = f"{key[0]}_p{key[1]:03d}_{kind}_{verdict}_{assessment.rule.name.lower()}_{culprit}.jpg"
        target = out_dir / folder / name
        tasks.append((payload, thresholds, geo_dir / f"{key[0]}.pdf", nogeo_dir / f"{key[0]}.pdf", target))
        rows.append([*key, kind, verdict, assessment.rule.value, assessment.culprit, metric, f"{score:.2f}",
                     f"{assessment.total:.2f}", f"{assessment.gain:.2f}", f"{assessment.gain_other:.2f}",
                     *(f"{float(payload['metrics'].get(n, 0.0) or 0.0):.3f}" for n in EXPORT_METRICS),
                     str(target.relative_to(out_dir))])  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pdf", "page", "kind", "verdict", "rule", "culprit", "worst_metric", "worst_score", "total",
                         "gain", "gain_non_text", *EXPORT_METRICS, "picture"])  # fmt: skip
        writer.writerows(rows)
    counts = Counter((row[2], row[3]) for row in rows)
    logger.info("Страниц: %d; по виду и вердикту: %s", len(rows), dict(sorted(counts.items())))
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for done, _ in enumerate(pool.map(_draw_one, tasks, chunksize=4), 1):
            if done % 500 == 0:
                logger.info("картинок: %d/%d", done, len(tasks))
    logger.info("Выгрузка: %s", out_dir)


# Выборка на разметку (``review``): метрики поясов score, пояса, страниц на пояс; смены вердикта и сколько брать.
REVIEW_METRICS = (
    "edge_quality_mm",
    "line_quality_mm",
    "lineart_seg_rel_mm",
    "lineart_aad_mm",
    "formula_bar_ptp_mm",
    "formula_skew_mm",
)
REVIEW_BELTS = ((0.75, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, float("inf")))
REVIEW_PER_BELT = 3
REVIEW_TOTAL_BELTS = ((2.5, 4.0), (4.0, float("inf")))
REVIEW_CHANGES = (
    ("v14", "ok", "bad", 20),
    ("v14", "bad", "ok", 15),
    ("v16", "ok", "bad", 15),
    ("v16", "bad", "ok", 15),
)


@main.command()
@click.option("--run-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--geo-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--nogeo-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--v14-csv", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--v16-csv", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--labels", "labels_dir", default=None, type=click.Path(path_type=Path), help="Эталон: его страницы не берутся."
)
@click.option("--seed", default=0, show_default=True, type=int)
@threshold_options
def review(
    run_dir, geo_dir, nogeo_dir, out_dir, v14_csv, v16_csv, labels_dir, seed, thr, hard, total, min_gain, ratio
) -> None:
    """Выборка на разметку: пояса score главных мер и смены вердикта v14/v16 → v17; оверлеи и заготовка review.tsv.

    В TSV — формат эталона (``название<TAB>путь<TAB>аннотация``) плюс колонки причины отбора и вердиктов; метку
    (bad/good) пользователь ставит в колонку ``метка``, заметку — в ``аннотация``.
    """
    import cv2

    from research.geometry_quality.overlay import draw_pair

    thresholds = _thresholds(thr, hard, total, min_gain, ratio)
    pages = {(p["pdf"], p["page"]): p for p in load_run(run_dir)}
    labels = load_labels(labels_dir) if labels_dir else {}
    old = {"v14": load_verdicts(v14_csv), "v16": load_verdicts(v16_csv)}
    verdicts = {key: thresholds.assess(p["metrics"]) for key, p in pages.items()}
    rng = random.Random(seed)
    chosen: dict[tuple[str, int], str] = {}

    def take(reason: str, keys: list, count: int) -> None:
        pool = sorted(k for k in keys if k not in chosen and k not in labels)
        rng.shuffle(pool)
        for key in pool[:count]:
            chosen[key] = reason

    for name in REVIEW_METRICS:
        threshold = thresholds.specs[name].threshold
        for lo, hi in REVIEW_BELTS:
            keys = [k for k, p in pages.items() if lo <= float(p["metrics"].get(name, 0.0) or 0.0) / threshold < hi]
            take(f"{name}_score_{lo:g}-{hi:g}", keys, REVIEW_PER_BELT)
    for lo, hi in REVIEW_TOTAL_BELTS:
        keys = [k for k, v in verdicts.items() if lo <= v.total < hi and v.worst < 1.0]
        take(f"сумма_групп_{lo:g}-{hi:g}_без_метрики_выше_порога", keys, REVIEW_PER_BELT)
    for version, before, after, count in REVIEW_CHANGES:
        keys = [
            k for k, v in verdicts.items()
            if old[version].get(k) == before and (v.verdict is Verdict.BAD) == (after == "bad")
        ]  # fmt: skip
        take(f"{version}_{before}__v17_{after}", keys, count)
    logger.info("Страниц на разметку: %d", len(chosen))
    rows = []
    for key, reason in sorted(chosen.items(), key=lambda item: (item[1], item[0])):
        payload = dict(pages[key])
        payload["metrics"] = dict(payload["metrics"], v16_verdict=old["v16"].get(key, "—"))
        assessment = verdicts[key]
        picture = draw_pair(payload, assessment, thresholds, geo_dir / f"{key[0]}.pdf", nogeo_dir / f"{key[0]}.pdf")
        target = (
            out_dir / reason / f"{key[0]}_p{key[1]:03d}_{assessment.verdict.value}_{assessment.rule.name.lower()}.jpg"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append([f"{key[0]} с.{key[1]}", str(target), "", "", reason, old["v14"].get(key, ""), old["v16"].get(key, ""),
                     assessment.verdict.value, assessment.rule.value, assessment.culprit])  # fmt: skip
    with (out_dir / "review.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["название", "путь", "аннотация", "метка", "причина отбора", "v14", "v16", "v17", "правило", "виновник"]
        )
        writer.writerows(rows)
    logger.info("Оверлеи и review.tsv: %s", out_dir)


__all__ = ["main"]
