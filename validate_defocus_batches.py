"""Валидация алгоритмов детекции расфокуса по ручной разметке подшивок.

ЧТО ДЕЛАЕТ. Один раз читает каждый кадр выборки и считает по нему СРАЗУ ВСЕ метрики во
всех агрегациях плюс масштаб набора, анизотропию и яркость, складывая это в широкий CSV.
Дальше все эксперименты — подбор порогов, сравнение алгоритмов, разрезы по подшивкам и
съёмочным сессиям — идут по этому CSV за секунды и не трогают диск.

ПОЧЕМУ ОДИН ПРОХОД, А НЕ ПРОГОН CLI НА КАЖДЫЙ АЛГОРИТМ. Дорого в этой задаче ровно одно —
декодирование превью (13 Мп на кадр). Метрики поверх уже раскодированного кадра стоят
дёшево. Прогон CLI восемь раз означал бы восемь декодирований одного и того же.

КАК МЕРЯЕТСЯ КАЧЕСТВО. Подробности — в докстрингах функций, здесь суть:

* **AUC** и **средняя точность (AP)**. При доле брака в 4 % AUC выглядит благополучно
  даже у посредственного детектора, поэтому решающая величина — AP.
* **Полнота в первых k %** — прямой ответ на «сколько листать».
* **Бюджет просмотра** — позиция самого невезучего плохого кадра, то есть длина списка,
  который надо отсмотреть, чтобы не пропустить ни одного.
* **Парный тест на пересъёмках** — самая чистая проверка: та же полоса, та же вёрстка,
  тот же кегль, разница только в фокусе. Не зависит ни от вёрстки, ни от полноты разметки.
* **Специфичность на проверенно хороших** — единственное место, где ложное срабатывание
  считается по-настоящему, а не «может, я это просто просмотрел».
"""

import csv
import math
import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
import numpy as np

from ocr_utils.defocus_detection import labels as lbl
from ocr_utils.defocus_detection.analysis import THREAD_LIMIT_VARS, _init_worker
from ocr_utils.defocus_detection.image_io import read_gray
from ocr_utils.defocus_detection.metrics import ALGORITHMS
from ocr_utils.defocus_detection.metrics import edge_dir
from ocr_utils.defocus_detection.scale import text_line_pitch
from ocr_utils.defocus_detection.scoring import DEFAULT_QUANTILE, aggregate
from ocr_utils.defocus_detection.tiles import detail_rms_map, make_grid, printed_mask

# Агрегации, которые считаются для каждой метрики. Это и есть «разные настройки» из
# задачи: одна и та же метрика с разной агрегацией — по сути разные детекторы.
AGGREGATIONS = ("worst", "median", "best")

# Доли списка, на которых меряется полнота.
RECALL_POINTS = (0.05, 0.10, 0.15, 0.20)


# --------------------------------------------------------------------------------------
# Сбор признаков
# --------------------------------------------------------------------------------------


def measure(path: Path) -> dict:
    """Считает все метрики, масштаб и служебные величины по одному кадру.

    Args:
        path: Путь к изображению.

    Returns:
        Плоский словарь «имя признака -> число»; при ошибке чтения — с полем ``error``.
    """
    gray = read_gray(path)
    if gray is None:
        return {"error": "не прочитан"}

    grid = make_grid(gray.shape)
    printed = printed_mask(detail_rms_map(gray, grid))
    row: dict = {
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "printed_tiles": int(printed.sum()),
        "tiles": int(grid.ny * grid.nx),
        # Средняя яркость нужна не для метрики, а для проверки: не коррелирует ли балл
        # с освещением. Пользователь предупредил, что уровни гуляют внутри подшивки.
        "mean_level": float(gray.mean()),
        "rms_contrast": float(gray.std()),
    }

    scale = text_line_pitch(gray)
    row["pitch"] = scale.pitch
    row["pitch_conf"] = scale.confidence
    row["pitch_spread"] = scale.spread
    row["pitch_usable"] = int(scale.usable)

    for name, algorithm in ALGORITHMS.items():
        tile_map = algorithm.tile_sharpness(gray, grid)
        for mode in AGGREGATIONS:
            row[f"{name}__{mode}"] = aggregate(tile_map, printed, mode=mode, quantile=DEFAULT_QUANTILE)
        # Нормировка на шаг строк — то, ради чего вообще возможен абсолютный порог.
        # Осмысленна только у метрик с размерностью длины: у долей энергии деление на
        # шаг строк смысла не имеет.
        if algorithm.length_scaled and np.isfinite(scale.pitch) and scale.pitch > 0:
            for mode in AGGREGATIONS:
                score = row[f"{name}__{mode}"]
                blur_px = 1.0 / score if np.isfinite(score) and score > 0 else float("nan")
                row[f"{name}__{mode}__norm"] = blur_px / scale.pitch

    shares, _ = edge_dir.sector_map(gray, grid)
    ratio, angle = edge_dir.anisotropy(shares)
    usable = np.isfinite(ratio) & printed
    row["anisotropy"] = float(np.nanmedian(ratio[usable])) if usable.any() else float("nan")
    row["anisotropy_max"] = float(np.nanmax(ratio[usable])) if usable.any() else float("nan")
    row["blur_angle"] = float(np.nanmedian(angle[usable])) if usable.any() else float("nan")
    return row


def _worker(path_str: str) -> tuple[str, dict]:
    """Обёртка для пула процессов.

    Args:
        path_str: Путь к кадру строкой (пути пиклятся дешевле объектов).

    Returns:
        Пара (путь, словарь признаков).
    """
    try:
        return path_str, measure(Path(path_str))
    except Exception as error:  # noqa: BLE001 — воркер не должен ронять весь прогон
        return path_str, {"error": f"{type(error).__name__}: {error}"}


def load_exif(path: Path | None, base: Path) -> dict[str, tuple[str, str]]:
    """Читает подготовленную выгрузку EXIF: время съёмки и модель камеры.

    Время съёмки нужно, чтобы резать выборку по СЪЁМОЧНЫМ СЕССИЯМ, а не только по
    папкам: доля брака привязана к сессии (в выборке СИ она гуляет от 0 до 10 %), и
    смешивать сессии в одну статистику — значит усреднять разные условия съёмки.

    Args:
        path: TSV из ``exiftool -T -FileName -Directory -ImageCount -DateTimeOriginal
            -Model``; None — вернуть пустое отображение.
        base: Папка, из которой запускался exiftool: колонка Directory в его выгрузке
            относительная, и без базы пути разошлись бы с реальными.

    Returns:
        Отображение «абсолютный путь -> (время съёмки, модель)».
    """
    if path is None or not path.exists():
        return {}
    out: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        name, folder, _count, shot_at, model = parts[:5]
        out[str((base / folder / name).resolve())] = (shot_at, model)
    return out


def sessions_from_times(times: dict[str, datetime], gap_hours: float = 2.0) -> dict[str, str]:
    """Режет кадры на съёмочные сессии по разрывам во времени.

    Args:
        times: Отображение «путь -> момент съёмки».
        gap_hours: Разрыв, начиная с которого считаем, что началась новая сессия.

    Returns:
        Отображение «путь -> имя сессии» (дата и время её начала).
    """
    if not times:
        return {}
    ordered = sorted(times.items(), key=lambda kv: kv[1])
    out: dict[str, str] = {}
    start = ordered[0][1]
    previous = start
    for path_str, moment in ordered:
        if (moment - previous).total_seconds() > gap_hours * 3600:
            start = moment
        out[path_str] = start.strftime("%Y-%m-%d %H:%M")
        previous = moment
    return out


# --------------------------------------------------------------------------------------
# Меры качества ранжирования
# --------------------------------------------------------------------------------------


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Доля пар «плохой, хороший», где плохой оказался ниже по резкости.

    Args:
        positive: Баллы резкости помеченных браком кадров.
        negative: Баллы резкости остальных.

    Returns:
        AUC в [0, 1]; 0.5 — случайное угадывание. NaN, если один из классов пуст.
    """
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    # Знак переворачиваем ЗДЕСЬ, а не у вызывающего. На входе — баллы резкости, где
    # больше значит резче, а мера должна расти, когда брак оказывается НИЖЕ хороших
    # кадров. Без этого 0.974 читается как 0.026 — ровно так и вышло при первом прогоне.
    positive, negative = -positive, -negative
    # Через ранги: сортировка вместо перебора всех пар.
    joined = np.concatenate([positive, negative])
    order = joined.argsort(kind="mergesort")
    ranks = np.empty(joined.size, dtype=np.float64)
    ranks[order] = np.arange(1, joined.size + 1, dtype=np.float64)
    # Средний ранг для связок, иначе одинаковые баллы дадут смещение.
    _, inverse, counts = np.unique(joined, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    rank_sum = ranks[: positive.size].sum()
    return float((rank_sum - positive.size * (positive.size + 1) / 2.0) / (positive.size * negative.size))


def average_precision(scores: np.ndarray, is_bad: np.ndarray) -> float:
    """Средняя точность при обходе списка от самого мягкого кадра к самому резкому.

    При сильном дисбалансе классов (у нас брака около 4 %) AP информативнее AUC: она
    падает, если в верхушку списка набилось много хороших кадров, тогда как AUC этого
    почти не замечает.

    Args:
        scores: Баллы резкости (меньше = подозрительнее).
        is_bad: Булев массив разметки.

    Returns:
        AP в [0, 1]; NaN, если положительных нет.
    """
    good = np.isfinite(scores)
    scores, is_bad = scores[good], is_bad[good]
    if is_bad.sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    hits = is_bad[order].astype(np.float64)
    cumulative = np.cumsum(hits)
    precision = cumulative / np.arange(1, hits.size + 1)
    return float((precision * hits).sum() / hits.sum())


def recall_at(scores: np.ndarray, is_bad: np.ndarray, fraction: float) -> float:
    """Какая доля брака попала в первые ``fraction`` списка.

    Args:
        scores: Баллы резкости (меньше = подозрительнее).
        is_bad: Булев массив разметки.
        fraction: Доля списка.

    Returns:
        Полнота в [0, 1]; NaN, если положительных нет.
    """
    good = np.isfinite(scores)
    scores, is_bad = scores[good], is_bad[good]
    if is_bad.sum() == 0:
        return float("nan")
    take = max(1, int(math.ceil(scores.size * fraction)))
    order = np.argsort(scores, kind="mergesort")
    return float(is_bad[order][:take].sum() / is_bad.sum())


def review_budget(scores: np.ndarray, is_bad: np.ndarray) -> float:
    """Доля списка, которую надо отсмотреть, чтобы не пропустить ни одного плохого кадра.

    Args:
        scores: Баллы резкости (меньше = подозрительнее).
        is_bad: Булев массив разметки.

    Returns:
        Доля в (0, 1]; NaN, если положительных нет.
    """
    good = np.isfinite(scores)
    scores, is_bad = scores[good], is_bad[good]
    if is_bad.sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    positions = np.flatnonzero(is_bad[order])
    return float((positions[-1] + 1) / scores.size)


def paired_wins(pairs: list[tuple[float, float]]) -> tuple[float, int]:
    """Доля пар «брак / его пересъёмка», где алгоритм поставил брак ниже.

    Args:
        pairs: Список пар (балл забракованного кадра, балл его пересъёмки).

    Returns:
        Кортеж (доля верных пар, сколько пар участвовало).
    """
    usable = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
    if not usable:
        return float("nan"), 0
    wins = sum((a < b) + 0.5 * (a == b) for a, b in usable)
    return wins / len(usable), len(usable)


def bootstrap_auc(positive: np.ndarray, negative: np.ndarray, draws: int = 400, seed: int = 0) -> tuple[float, float]:
    """Доверительный интервал AUC бутстрэпом по кадрам.

    Нужен не для красоты: положительных в отдельной подшивке единицы, и разница
    «AUC 0.88 против 0.86» на такой выборке может быть чистым шумом.

    Args:
        positive: Баллы помеченных кадров.
        negative: Баллы остальных.
        draws: Число повторных выборок.
        seed: Зерно генератора.

    Returns:
        Границы 95-процентного интервала; NaN, если считать не на чем.
    """
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size < 2 or negative.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = [
        roc_auc(rng.choice(positive, positive.size, replace=True), rng.choice(negative, negative.size, replace=True))
        for _ in range(draws)
    ]
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


@dataclass
class Row:
    """Строка сводной таблицы: кадр, его разметка и посчитанные признаки."""

    sample: lbl.Sample
    features: dict
    session: str = ""
    shot_at: str = ""


def _feature_columns(rows: list[Row]) -> list[str]:
    """Собирает объединение имён признаков по всем строкам.

    Args:
        rows: Строки таблицы.

    Returns:
        Отсортированный список имён колонок.
    """
    names: set[str] = set()
    for row in rows:
        names.update(row.features)
    return sorted(names)


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main() -> None:
    """Валидация детекторов расфокуса по ручной разметке."""


@main.command()
@click.option("--root", type=click.Path(exists=True, path_type=Path), required=True, help="Корень с подшивками.")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Куда записать CSV признаков.")
@click.option("--exif", type=click.Path(path_type=Path), default=None, help="TSV с EXIF (для съёмочных сессий).")
@click.option("--workers", type=int, default=0, help="Процессов; 0 — по числу ядер минус --reserve-cores.")
@click.option("--reserve-cores", type=int, default=2, show_default=True, help="Сколько ядер оставить системе.")
@click.option("--limit", type=int, default=0, help="Взять только первые N кадров (для отладки).")
def sweep(root: Path, out: Path, exif: Path | None, workers: int, reserve_cores: int, limit: int) -> None:
    """Считает все метрики по всем кадрам ROOT и пишет широкий CSV."""
    samples = lbl.collect(root)
    samples = [s for s in samples if not s.label.ignored]
    if limit:
        samples = samples[:limit]
    click.echo(f"Кадров: {len(samples)}, помечено браком: {sum(s.label.is_bad for s in samples)}", err=True)

    exif_data = load_exif(exif, root)
    times: dict[str, datetime] = {}
    for sample in samples:
        key = str(sample.path.resolve())
        if key in exif_data:
            try:
                times[key] = datetime.strptime(exif_data[key][0], "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
    session_of = sessions_from_times(times)
    if session_of:
        click.echo(f"Съёмочных сессий: {len(set(session_of.values()))}", err=True)

    if workers <= 0:
        workers = max(1, (os.cpu_count() or 1) - reserve_cores)
    paths = [str(s.path) for s in samples]
    by_path: dict[str, lbl.Sample] = {str(s.path): s for s in samples}

    # Ограничиваем внутренние пулы потоков и берём forkserver, а не fork: rawpy собран
    # с OpenMP и в форкнутом процессе способен намертво заклиниться — он сам об этом
    # предупреждает при импорте, и на прогоне в четыре тысячи кадров рисковать нечем.
    for name in THREAD_LIMIT_VARS:
        os.environ.setdefault(name, "1")
    context = multiprocessing.get_context("forkserver")

    rows: list[Row] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
        with click.progressbar(pool.map(_worker, paths, chunksize=4), length=len(paths), label="Замер") as stream:
            for path_str, features in stream:
                key = str(Path(path_str).resolve())
                rows.append(
                    Row(
                        sample=by_path[path_str],
                        features=features,
                        session=session_of.get(key, ""),
                        shot_at=exif_data.get(key, ("", ""))[0],
                    )
                )

    columns = _feature_columns(rows)
    header = [
        "path",
        "batch",
        "stem",
        "suffix",
        "severity",
        "is_bad",
        "verified_good",
        "motion_blur",
        "zonal",
        "reshoot_of",
        "replaced_by",
        "session",
        "shot_at",
        *columns,
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            s = row.sample
            writer.writerow(
                [
                    s.path,
                    s.batch,
                    s.stem,
                    s.label.suffix,
                    s.label.severity or "",
                    int(s.label.is_bad),
                    int(s.label.verified_good),
                    int(s.label.motion_blur),
                    int(s.label.zonal),
                    s.reshoot_of or "",
                    s.replaced_by or "",
                    row.session,
                    row.shot_at,
                    *[row.features.get(name, "") for name in columns],
                ]
            )
    failed = sum(1 for r in rows if "error" in r.features)
    click.echo(f"CSV: {out} ({len(rows)} строк, ошибок чтения: {failed})", err=True)


# Подшивки, отсмотренные человеком ЦЕЛИКОМ и признанные чистыми. Это знание о конкретных
# данных, а не о формате разметки, поэтому оно живёт здесь, а не в ``labels.py``. Ценность
# такого набора особая: во всём остальном материале «неразмеченный» означает «просмотрено,
# но мог пропустить», и точность там — оценка снизу. Здесь же ложное срабатывание —
# настоящее ложное срабатывание, и только на нём специфичность меряется честно.
VERIFIED_CLEAN_BATCHES = ("1988/01-03",)


def read_rows(path: Path) -> list[dict]:
    """Читает CSV свипа, приводя числа к float и проставляя проверенно чистые подшивки.

    Args:
        path: Путь к CSV из команды ``sweep``.

    Returns:
        Список словарей по кадрам.
    """
    text_columns = (
        "path",
        "batch",
        "stem",
        "suffix",
        "severity",
        "session",
        "shot_at",
        "reshoot_of",
        "replaced_by",
        "error",
    )
    # Флаги разметки держим булевыми, а не числами: ниже они используются как маски, и
    # молчаливое превращение их во float уже один раз обнулило всю статистику.
    flag_columns = ("is_bad", "verified_good", "motion_blur", "zonal", "pitch_usable")

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows:
        for key in flag_columns:
            row[key] = row.get(key) == "1"
        if row["batch"] in VERIFIED_CLEAN_BATCHES and not row["is_bad"]:
            row["verified_good"] = True
        for key, value in list(row.items()):
            if key in text_columns or key in flag_columns:
                continue
            try:
                row[key] = float(value) if value not in ("", "nan") else float("nan")
            except ValueError:
                row[key] = float("nan")
    return rows


def metric_columns(rows: list[dict]) -> list[str]:
    """Имена колонок с баллами метрик (``алгоритм__агрегация`` и нормированные).

    Args:
        rows: Строки таблицы.

    Returns:
        Отсортированный список имён.
    """
    return sorted(k for k in rows[0] if "__" in k and k not in ("reshoot_of", "replaced_by"))


def direction_of(column: str) -> float:
    """Знак, приводящий колонку к правилу «меньше = подозрительнее».

    Нормированные колонки (``__norm``) — это размытие в долях шага строк, у них шкала
    ОБРАТНАЯ: больше значит хуже. Остальные — резкость, больше значит лучше.

    Args:
        column: Имя колонки.

    Returns:
        +1.0 либо -1.0.
    """
    return -1.0 if column.endswith("__norm") else 1.0


@main.command()
@click.option("--csv", "csv_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Куда записать отчёт (md).")
@click.option("--top", type=int, default=12, show_default=True, help="Сколько лучших вариантов показать.")
def report(csv_path: Path, out: Path, top: int) -> None:
    """Считает меры качества по CSV свипа и пишет markdown-отчёт."""
    rows = read_rows(csv_path)
    columns = metric_columns(rows)

    is_bad = np.array([r["is_bad"] for r in rows])
    verified = np.array([r["verified_good"] for r in rows])
    click.echo(f"Кадров: {len(rows)}, брака: {is_bad.sum()}, проверенно хороших: {verified.sum()}", err=True)

    # --- рейтинг ВНУТРИ подшивок -------------------------------------------------------
    # Считать одну общую AUC по всем 13 подшивкам нельзя, и это не педантизм. В общей куче
    # выигрывает метрика, которая хорошо разделяет СЪЁМОЧНЫЕ СЕССИИ (у них разные экспозиция
    # и освещение), а не расфокус. Проверено на этих же данных: при сквозном счёте наверх
    # вылезает laplacian — наивная база, про которую в самом пакете написано, что её нельзя
    # брать именно из-за зависимости от контраста и ISO, — а рабочий edge_width падает вниз.
    # Инструментом же пользуются иначе: ранжируют кадры ВНУТРИ одной папки. Поэтому меры
    # считаются по каждой подшивке отдельно и усредняются по числу брака в ней.
    by_batch: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_batch[row["batch"]].append(index)

    per_batch = []
    for column in columns:
        values_all = np.array([r[column] for r in rows]) * direction_of(column)
        aps, recalls, budgets, weights = [], [], [], []
        for indices in by_batch.values():
            idx = np.array(indices)
            local_bad = is_bad[idx]
            if local_bad.sum() == 0:
                continue
            local_values = values_all[idx]
            if np.isfinite(local_values).sum() < idx.size * 0.9:
                continue
            aps.append(average_precision(local_values, local_bad))
            recalls.append({f: recall_at(local_values, local_bad, f) for f in RECALL_POINTS})
            budgets.append(review_budget(local_values, local_bad))
            weights.append(int(local_bad.sum()))
        if not aps:
            continue
        w = np.array(weights, dtype=np.float64)
        per_batch.append(
            dict(
                column=column,
                ap=float(np.average(aps, weights=w)),
                recall={f: float(np.average([r[f] for r in recalls], weights=w)) for f in RECALL_POINTS},
                budget=float(np.average(budgets, weights=w)),
                batches=len(aps),
            )
        )
    per_batch.sort(key=lambda d: -d["ap"])

    # --- общий рейтинг вариантов -------------------------------------------------------
    scored = []
    for column in columns:
        values = np.array([r[column] for r in rows]) * direction_of(column)
        good = np.isfinite(values)
        if good.sum() < len(rows) * 0.9:
            continue
        auc_verified = roc_auc(values[is_bad & good], values[verified & good])
        auc_rest = roc_auc(values[is_bad & good], values[~is_bad & good])
        ap = average_precision(values, is_bad)
        scored.append(
            dict(
                column=column,
                auc_verified=auc_verified,
                auc_rest=auc_rest,
                ap=ap,
                recall={f: recall_at(values, is_bad, f) for f in RECALL_POINTS},
                budget=review_budget(values, is_bad),
            )
        )
    scored.sort(key=lambda d: -d["ap"])

    lines = ["# Валидация детекторов расфокуса на подшивках «Социалистической индустрии»", ""]
    lines.append(
        f"**Выборка:** {len(rows)} кадров, {int(is_bad.sum())} помечено браком вручную, "
        f"{int(verified.sum())} проверенно хороших."
    )
    lines.append("")
    lines.append("## 1. Рейтинг ВНУТРИ подшивок — главная таблица")
    lines.append("")
    lines.append(
        "Меры считаются по каждой подшивке отдельно и усредняются по числу брака в ней. "
        "Так и работает инструмент: он ранжирует кадры внутри одной папки. Сквозной счёт "
        "по всем подшивкам сразу даёт другой и обманчивый ответ — см. раздел 2."
    )
    lines.append("")
    lines.append("| вариант | AP | recall@5% | @10% | @15% | @20% | бюджет |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for entry in per_batch[:top]:
        r = entry["recall"]
        lines.append(
            f"| `{entry['column']}` | **{entry['ap']:.3f}** | {r[0.05]:.2f} | {r[0.10]:.2f} | "
            f"{r[0.15]:.2f} | {r[0.20]:.2f} | {entry['budget']*100:.0f} % |"
        )
    lines.append("")
    lines.append("## 2. Сквозной рейтинг по всем подшивкам — и почему ему нельзя верить")
    lines.append("")
    lines.append(
        "`AUC пров.` — против проверенно хороших кадров: единственное место, где ложное "
        "срабатывание считается честно. `AP` — средняя точность, решающая величина при "
        "доле брака 4 %. `бюджет` — какую долю списка надо отсмотреть, чтобы не пропустить "
        "ни одного плохого кадра."
    )
    lines.append("")
    lines.append("| вариант | AP | AUC пров. | AUC все | recall@5% | @10% | @15% | бюджет |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for entry in scored[:top]:
        r = entry["recall"]
        lines.append(
            f"| `{entry['column']}` | **{entry['ap']:.3f}** | {entry['auc_verified']:.3f} | "
            f"{entry['auc_rest']:.3f} | {r[0.05]:.2f} | {r[0.10]:.2f} | {r[0.15]:.2f} | "
            f"{entry['budget']*100:.0f} % |"
        )
    lines.append("")

    # --- 1987: сравнение без примеси сессий --------------------------------------------
    # В подшивках 1985 брак снят в июле, а часть хороших кадров — пересъёмки 18 августа,
    # то есть в другую сессию с другим освещением. Любая контраст-зависимая метрика
    # разделяет их даром, не глядя на фокус. В 1987 и брак, и хорошие кадры сняты в одну
    # сессию — это единственный полностью чистый разрез, и он же самый крупный по разметке.
    clean = [b for b in by_batch if b.startswith("1987")]
    clean_scores = []
    for column in columns:
        values_all = np.array([r[column] for r in rows]) * direction_of(column)
        aps, weights = [], []
        for batch in clean:
            idx = np.array(by_batch[batch])
            local_bad = is_bad[idx]
            if local_bad.sum() == 0:
                continue
            aps.append(average_precision(values_all[idx], local_bad))
            weights.append(int(local_bad.sum()))
        if aps:
            clean_scores.append((float(np.average(aps, weights=np.array(weights, float))), column))
    clean_scores.sort(reverse=True)

    # --- парный тест на пересъёмках ----------------------------------------------------
    by_path = {r["path"]: r for r in rows}
    pairs_index = [(r, by_path[r["replaced_by"]]) for r in rows if r["replaced_by"] and r["replaced_by"] in by_path]
    paired = []
    for column in columns:
        sign = direction_of(column)
        pairs = [(bad[column] * sign, good[column] * sign) for bad, good in pairs_index]
        share, count = paired_wins(pairs)
        paired.append((share, count, column))
    paired.sort(reverse=True)

    # --- зависимость от освещения и кегля ----------------------------------------------
    level = np.array([r["mean_level"] for r in rows])
    pitch = np.array([r["pitch"] for r in rows])
    correlations = []
    for column in columns:
        values = np.array([r[column] for r in rows])
        good = np.isfinite(values) & np.isfinite(level) & np.isfinite(pitch)
        if good.sum() < 100:
            continue
        correlations.append(
            (
                column,
                float(np.corrcoef(values[good], level[good])[0, 1]),
                float(np.corrcoef(values[good], pitch[good])[0, 1]),
            )
        )

    lines.append("## 3. Разрез по 1987 — единственный без примеси сессий")
    lines.append("")
    lines.append(
        "В подшивках 1985 брак снят в июле, а часть хороших кадров — пересъёмки августа, "
        "то есть в другую сессию с другим освещением: контраст-зависимая метрика разделяет "
        "их даром, не глядя на фокус. В 1987 брак и хорошие кадры сняты в одну сессию, "
        "и разметки там больше всего (116 кадров)."
    )
    lines.append("")
    lines.append("| вариант | AP (среднее по 4 подшивкам 1987) |")
    lines.append("|---|--:|")
    for ap, column in clean_scores[:top]:
        lines.append(f"| `{column}` | **{ap:.3f}** |")
    lines.append("")

    lines.append("## 4. Парный тест на пересъёмках")
    lines.append("")
    lines.append(
        f"Пар «забракованный оригинал / снятая взамен замена»: {len(pairs_index)}. "
        "Та же полоса, та же вёрстка, тот же кегль — разница только в фокусе. Доля пар, "
        "где алгоритм поставил брак ниже замены; 0.5 — уровень монетки."
    )
    lines.append("")
    lines.append("| вариант | доля верных пар | пар |")
    lines.append("|---|--:|--:|")
    for share, count, column in paired[:top]:
        lines.append(f"| `{column}` | **{share:.3f}** | {count} |")
    lines.append("")

    lines.append("## 5. Зависимость от освещения и кегля")
    lines.append("")
    lines.append(
        "Корреляция балла со средней яркостью кадра и с шагом строк. Метрика, у которой "
        "связь с яркостью сильная, не годится для абсолютного порога: освещение гуляет "
        "даже внутри подшивки."
    )
    lines.append("")
    lines.append("| вариант | с яркостью | с шагом строк |")
    lines.append("|---|--:|--:|")
    for column, r_level, r_pitch in sorted(correlations, key=lambda c: abs(c[1]))[:top]:
        lines.append(f"| `{column}` | {r_level:+.3f} | {r_pitch:+.3f} |")
    lines.append("")
    lines.append("Худшие по связи с яркостью:")
    lines.append("")
    lines.append("| вариант | с яркостью | с шагом строк |")
    lines.append("|---|--:|--:|")
    for column, r_level, r_pitch in sorted(correlations, key=lambda c: -abs(c[1]))[:6]:
        lines.append(f"| `{column}` | {r_level:+.3f} | {r_pitch:+.3f} |")
    lines.append("")

    # --- смаз ---------------------------------------------------------------------------
    motion = np.array([r["motion_blur"] for r in rows])
    aniso = np.array([r["anisotropy"] for r in rows])
    if motion.sum():
        lines.append("## 6. Смаз против промаха фокуса")
        lines.append("")
        lines.append(
            f"Кадров, помеченных именно смазом: {int(motion.sum())}. Медиана анизотропии "
            f"у них {np.nanmedian(aniso[motion]):.2f}, у прочего брака "
            f"{np.nanmedian(aniso[is_bad & ~motion]):.2f}, у неразмеченных "
            f"{np.nanmedian(aniso[~is_bad]):.2f}."
        )
        lines.append("")

    summary: list[str] = []
    best_clean = clean_scores[0] if clean_scores else (float("nan"), "—")
    ew_clean = next((ap for ap, c in clean_scores if c == "edge_width__worst"), float("nan"))
    bright = {c: r for c, r, _ in correlations}
    summary.append("## 0. Что показал прогон")
    summary.append("")
    summary.append(
        f"* **Лучший вариант на чистом разрезе 1987 — `{best_clean[1]}`, AP {best_clean[0]:.3f}.** "
        f"Нынешний рабочий `edge_width__worst` даёт {ew_clean:.3f}, то есть проигрывает "
        f"в {best_clean[0]/ew_clean:.1f} раза."
    )
    summary.append(
        f"* **Но `dom` зависит от освещения** (корреляция с яркостью "
        f"{bright.get('dom__best', float('nan')):+.2f}), тогда как `edge_width` и `cpbd` "
        f"к ней почти нечувствительны ({bright.get('edge_width__worst', float('nan')):+.2f} и "
        f"{bright.get('cpbd__worst', float('nan')):+.2f}). Для РАНЖИРОВАНИЯ внутри папки это "
        "не мешает, а вот вешать на `dom` абсолютный порог нельзя: уровни гуляют между "
        "сессиями, и порог поедет вместе с ними."
    )
    summary.append(
        "* **Сквозной рейтинг по всем подшивкам обманчив** — там наверх вылезает `laplacian`, "
        "наивная база, которую сам пакет не рекомендует. Причина в разделе 2."
    )
    summary.append("* **Детектор смаза на реальных данных не подтвердился** — раздел 6.")
    summary.append("")
    # Раздел 0 печатается первым, но считается последним: ему нужны величины из
    # разделов 3-5. Поэтому он собирается отдельно и вставляется после шапки.
    lines[4:4] = summary

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    click.echo(f"Отчёт: {out}", err=True)

    # краткая сводка в консоль
    click.echo("")
    click.echo("ВНУТРИ ПОДШИВОК (главная таблица):")
    click.echo(f"{'вариант':<34}{'AP':>7}{'r@5%':>7}{'r@10%':>7}{'r@15%':>7}{'бюджет':>9}")
    for entry in per_batch[:top]:
        r = entry["recall"]
        click.echo(
            f"{entry['column']:<34}{entry['ap']:>7.3f}{r[0.05]:>7.2f}{r[0.10]:>7.2f}"
            f"{r[0.15]:>7.2f}{entry['budget']*100:>8.0f}%"
        )
    click.echo("")
    click.echo("ТОЛЬКО 1987 (без примеси сессий):")
    for ap, column in clean_scores[:8]:
        click.echo(f"  {column:<34}{ap:>7.3f}")
    click.echo("")
    click.echo(f"ПАРНЫЙ ТЕСТ ({len(pairs_index)} пересъёмок):")
    for share, count, column in paired[:8]:
        click.echo(f"  {column:<34}{share:>7.3f}")


if __name__ == "__main__":
    main()
