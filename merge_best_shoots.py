"""Собирает одну папку из двух съёмок одной подшивки: по каждому развороту — снимок с лучшим фокусом."""

import csv
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import click
import cv2
import numpy as np
import threadpoolctl
from tqdm import tqdm

from select_best_raws import read_raf_image


class Shoot(str, Enum):
    """Съёмка, из которой взят кадр; значение — постфикс в имени выходного файла."""

    OLD = "старый"
    NEW = "новый"


@dataclass
class Frame:
    """Кадр одной съёмки: исходный RAF, AKAZE-признаки всего кадра и балл фокуса DOM."""

    # Исходный RAF-файл
    path: Path
    # Координаты ключевых точек (n, 2) float32 — в масштабе уменьшенного кадра
    points: np.ndarray
    # Бинарные AKAZE-дескрипторы (n, 61) uint8; пустой массив, если точек нет
    descriptors: np.ndarray
    # Балл DOM из CSV defocus_detection; None — файла в CSV нет
    dom: float | None


@dataclass
class Unit:
    """Один разворот на выходе: группы кадров старой и новой съёмки (любая может быть пустой)."""

    # Индексы групп дублей старой и новой съёмки (None — разворота в этой съёмке нет)
    old_group: int | None
    new_group: int | None
    # Сходство пары групп (доля RANSAC-инлайеров); None — разворот без пары
    similarity: float | None


def init_worker() -> None:
    """Инициализатор воркера: без hugepage и без внутренней многопоточности OpenCV/BLAS.

    Returns:
        None.
    """
    # Выделения больших массивов с hugepage в воркерах в разы медленнее (замер в ocr_utils/dewarp)
    np._core.multiarray._set_madvise_hugepage(False)
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def frame_features(path: Path, width: int, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Считает AKAZE-признаки всего кадра (без центрального кропа: кадрирование съёмок разное).

    Args:
        path: RAF-файл; читается встроенное JPEG-превью (read_raf_image).
        width: Ширина, до которой уменьшается кадр перед детекцией (быстрее, текст ещё различим).
        max_points: Сколько самых сильных точек оставить (ограничивает время попарного сравнения).

    Returns:
        Кортеж (точки (n, 2) float32, дескрипторы (n, 61) uint8); при отсутствии точек — пустые массивы.
    """
    rgb = read_raf_image(path)
    h, w = rgb.shape[:2]
    # Уменьшаем кадр до заданной ширины с сохранением пропорций
    small = cv2.resize(rgb, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.AKAZE_create().detectAndCompute(gray, None)
    if descriptors is None or not keypoints:
        return np.zeros((0, 2), np.float32), np.zeros((0, 61), np.uint8)
    # Оставляем max_points самых сильных точек по отклику детектора
    order = np.argsort([-kp.response for kp in keypoints])[:max_points]
    points = np.float32([keypoints[i].pt for i in order])
    return points, descriptors[order]


def pair_similarity(
    points_a: np.ndarray,
    desc_a: np.ndarray,
    points_b: np.ndarray,
    desc_b: np.ndarray,
    max_scale: float,
    ratio: float = 0.75,
) -> float:
    """Сходство двух кадров: доля RANSAC-инлайеров гомографии от меньшего числа точек пары.

    Args:
        points_a: Точки первого кадра (n, 2).
        desc_a: Дескрипторы первого кадра (n, 61).
        points_b: Точки второго кадра (m, 2).
        desc_b: Дескрипторы второго кадра (m, 61).
        max_scale: Допустимое изменение масштаба гомографии: [1/max_scale, max_scale].
        ratio: Порог Lowe's ratio test.

    Returns:
        Число от 0 до 1: 0 — общего нет (мало совпадений, гомографии нет или масштаб вне допуска).
    """
    if len(desc_a) < 2 or len(desc_b) < 2:
        return 0.0
    # Однозначные совпадения: лучшее заметно ближе второго
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    if len(good) < 10:
        return 0.0
    src = points_a[[m.queryIdx for m in good]]
    dst = points_b[[m.trainIdx for m in good]]
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if homography is None:
        return 0.0
    # Линейный масштаб гомографии: разворот против разворота даёт ≈1, съёмки с разным
    # расстоянием до подшивки — в пределах max_scale
    scale = float(np.sqrt(abs(np.linalg.det(homography[:2, :2]))))
    if not (1 / max_scale <= scale <= max_scale):
        return 0.0
    return float(mask.sum()) / min(len(desc_a), len(desc_b))


def similarity_task(args: tuple) -> tuple[int, int, float]:
    """Обёртка pair_similarity для пула: принимает и возвращает индексы пары.

    Args:
        args: Кортеж (i, j, points_a, desc_a, points_b, desc_b, max_scale).

    Returns:
        Кортеж (i, j, сходство).
    """
    i, j, points_a, desc_a, points_b, desc_b, max_scale = args
    return i, j, pair_similarity(points_a, desc_a, points_b, desc_b, max_scale)


def features_task(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Обёртка frame_features для пула.

    Args:
        args: Кортеж (path, width, max_points).

    Returns:
        Кортеж (точки, дескрипторы) — как у frame_features.
    """
    path, width, max_points = args
    return frame_features(path, width, max_points)


def load_features(
    paths: list[Path], cache_dir: Path, width: int, max_points: int, jobs: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Признаки всех кадров с дисковым кэшем (повторный прогон не перечитывает RAF с NTFS).

    Args:
        paths: RAF-файлы в порядке съёмки.
        cache_dir: Папка кэша; ключ — имя файла, его размер и параметры детекции.
        width: Ширина кадра для детекции (см. frame_features).
        max_points: Предел числа точек (см. frame_features).
        jobs: Число процессов пула.

    Returns:
        Список (точки, дескрипторы) в порядке paths.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Имя кэша зависит от размера файла и параметров, чтобы подмена файла или смена
    # параметров не подсовывала старые признаки
    cache_files = [cache_dir / f"{p.name}.{p.stat().st_size}.w{width}.n{max_points}.npz" for p in paths]
    missing = [i for i, c in enumerate(cache_files) if not c.exists()]
    if missing:
        with ProcessPoolExecutor(jobs, mp_context=mp.get_context("forkserver"), initializer=init_worker) as pool:
            tasks = [(paths[i], width, max_points) for i in missing]
            results = pool.map(features_task, tasks)
            for i, (points, descriptors) in tqdm(zip(missing, results), total=len(missing), desc="Признаки"):
                np.savez(cache_files[i], points=points, descriptors=descriptors)
    loaded = []
    for c in cache_files:
        with np.load(c) as data:
            loaded.append((data["points"], data["descriptors"]))
    return loaded


def similarity_pairs(
    pairs: list[tuple[int, int]], frames_a: list[Frame], frames_b: list[Frame], max_scale: float, jobs: int, desc: str
) -> dict[tuple[int, int], float]:
    """Считает сходство заданных пар кадров в пуле процессов.

    Args:
        pairs: Пары индексов (i в frames_a, j в frames_b).
        frames_a: Кадры первой последовательности.
        frames_b: Кадры второй последовательности.
        max_scale: Допуск масштаба гомографии (см. pair_similarity).
        jobs: Число процессов пула.
        desc: Подпись полосы прогресса.

    Returns:
        Словарь {(i, j): сходство}.
    """
    tasks = [
        (i, j, frames_a[i].points, frames_a[i].descriptors, frames_b[j].points, frames_b[j].descriptors, max_scale)
        for i, j in pairs
    ]
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("forkserver"), initializer=init_worker) as pool:
        results = list(tqdm(pool.map(similarity_task, tasks, chunksize=8), total=len(tasks), desc=desc))
    return {(i, j): s for i, j, s in results}


def group_duplicates(frames: list[Frame], dup_threshold: float, max_scale: float, jobs: int) -> list[list[int]]:
    """Склеивает подряд идущие пересъёмки одного разворота внутри съёмки в группы.

    Сравниваются кадры на расстоянии 1 и 2 позиции; связь — сходство ≥ dup_threshold.

    Args:
        frames: Кадры одной съёмки в порядке съёмки.
        dup_threshold: Порог сходства «тот же разворот».
        max_scale: Допуск масштаба гомографии.
        jobs: Число процессов пула.

    Returns:
        Группы индексов кадров в порядке съёмки; одиночный кадр — группа из одного.
    """
    pairs = [(i, j) for i in range(len(frames)) for j in (i + 1, i + 2) if j < len(frames)]
    sims = similarity_pairs(pairs, frames, frames, max_scale, jobs, "Дубли внутри съёмки")
    # Объединение через «родителя»: кадр j присоединяется к группе кадра i при сходстве выше порога
    parent = list(range(len(frames)))
    for (i, j), s in sorted(sims.items()):
        if s >= dup_threshold:
            parent[j] = parent[i]
    groups: dict[int, list[int]] = {}
    for idx, root in enumerate(parent):
        groups.setdefault(root, []).append(idx)
    return sorted(groups.values(), key=min)


def align(similarity: np.ndarray, min_sim: float) -> list[tuple[int | None, int | None]]:
    """Монотонное выравнивание двух последовательностей разворотов (Needleman–Wunsch).

    Пара (i, j) приносит similarity[i, j] − min_sim, пропуск — 0: пара берётся, только если
    её сходство выше min_sim, и порядок разворотов сохраняется.

    Args:
        similarity: Матрица сходства (n, m); вне полосы сравнения — 0.
        min_sim: Порог сходства, ниже которого выгоднее считать развороты разными.

    Returns:
        Последовательность шагов: (i, j) — пара, (i, None) / (None, j) — разворот только в одной съёмке.
    """
    n, m = similarity.shape
    gain = similarity - min_sim
    # score[i, j] — лучшая сумма для префиксов длины i и j
    score = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i, j] = max(score[i - 1, j], score[i, j - 1], score[i - 1, j - 1] + gain[i - 1, j - 1])
    # Обратный проход от конца к началу
    steps: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and gain[i - 1, j - 1] > 0
            and np.isclose(score[i, j], score[i - 1, j - 1] + gain[i - 1, j - 1])
        ):
            steps.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or np.isclose(score[i, j], score[i - 1, j])):
            steps.append((i - 1, None))
            i -= 1
        else:
            steps.append((None, j - 1))
            j -= 1
    return steps[::-1]


def load_dom(csv_path: Path) -> dict[str, float]:
    """Читает баллы DOM из CSV defocus_detection.

    Args:
        csv_path: CSV, записанный опцией --csv модуля ocr_utils.defocus_detection.

    Returns:
        Словарь {имя файла: балл}; кадры с ошибкой счёта пропускаются.
    """
    with csv_path.open(encoding="utf-8") as handle:
        return {Path(r["file"]).name: float(r["score"]) for r in csv.DictReader(handle) if r["score"]}


def read_shoot(folder: Path, dom_csv: Path, cache_dir: Path, width: int, max_points: int, jobs: int) -> list[Frame]:
    """Собирает кадры съёмки: RAF папки по порядку имён, признаки и балл DOM.

    Args:
        folder: Папка съёмки.
        dom_csv: CSV defocus_detection по этой папке.
        cache_dir: Кэш признаков.
        width: Ширина кадра для детекции.
        max_points: Предел числа точек.
        jobs: Число процессов пула.

    Returns:
        Кадры в порядке имён файлов (порядок съёмки).
    """
    paths = sorted(p for p in folder.glob("*.[Rr][Aa][Ff]") if p.is_file())
    dom = load_dom(dom_csv)
    features = load_features(paths, cache_dir, width, max_points, jobs)
    return [Frame(p, pts, desc, dom.get(p.name)) for p, (pts, desc) in zip(paths, features)]


def output_name(number: int, frame: Frame, shoot: Shoot, sub: int | None = None) -> str:
    """Имя файла на выходе: номер разворота, исходное имя (с пометками .blurry.*) и съёмка.

    Args:
        number: Порядковый номер разворота (с 1).
        frame: Выбранный кадр.
        shoot: Съёмка, из которой он взят.
        sub: Подындекс, когда у разворота на выходе несколько кадров (--keep-both): 1, 2, …
            «117.1_» и «117.2_» сортируются по имени после «116_» и перед «118_».

    Returns:
        Например «014_DSCF0022.blurry.defocus_medium.старый.RAF» или «117.1_DSCF0122.старый.RAF».
    """
    prefix = f"{number:03d}" if sub is None else f"{number:03d}.{sub}"
    return f"{prefix}_{frame.path.stem}.{shoot.value}{frame.path.suffix}"


def choose(candidates: list[tuple[Shoot, Frame]], tie: float) -> tuple[Shoot, Frame, str]:
    """Выбирает кадр с наибольшим баллом DOM; при разнице меньше tie предпочитает новую съёмку.

    Args:
        candidates: Все кадры разворота из обеих съёмок.
        tie: Разница баллов, ниже которой кадры считаются равными по фокусу.

    Returns:
        Кортеж (съёмка, кадр, причина выбора для манифеста).
    """
    scored = [(s, f) for s, f in candidates if f.dom is not None]
    if not scored:
        # Баллов нет — берём первый новый кадр, иначе первый старый
        shoot, frame = next(((s, f) for s, f in candidates if s is Shoot.NEW), candidates[0])
        return shoot, frame, "нет балла DOM"
    best_shoot, best = max(scored, key=lambda sf: sf[1].dom)
    if best_shoot is Shoot.OLD:
        # Старый берём, только если он резче лучшего нового больше чем на tie
        new = [f for s, f in scored if s is Shoot.NEW]
        if new and best.dom - max(f.dom for f in new) < tie:
            frame = max(new, key=lambda f: f.dom)
            return Shoot.NEW, frame, f"равны (разница < {tie}), берём новый"
    if len({s for s, _ in candidates}) == 1:
        return best_shoot, best, "без пары"
    return best_shoot, best, "выше DOM"


@click.command()
@click.argument("old_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("old_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("new_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("new_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Кэш признаков")
@click.option("--width", default=1536, show_default=True, help="Ширина кадра для AKAZE")
@click.option("--max-points", default=4000, show_default=True, help="Предел числа AKAZE-точек на кадр")
@click.option("--band", default=15, show_default=True, help="Полоса сравнения вокруг диагонали, в кадрах")
@click.option("--max-scale", default=1.5, show_default=True, help="Допуск масштаба гомографии между съёмками")
@click.option("--min-pair-sim", default=0.05, show_default=True, help="Порог сходства «тот же разворот» между съёмками")
@click.option("--dup-sim", default=0.15, show_default=True, help="Порог сходства пересъёмки внутри одной съёмки")
@click.option("--tie", default=0.005, show_default=True, help="Разница DOM, ниже которой берётся новый кадр")
@click.option("--jobs", default=8, show_default=True, help="Процессов; чтение RAF упирается в NTFS, 8 хватает")
@click.option(
    "--force-pick",
    multiple=True,
    help="Кадр «съёмка:имя» (старый:DSCF0126.RAF), который выбирается для своего разворота вместо "
    "кадра с лучшим DOM. Можно повторять",
)
@click.option(
    "--keep-both",
    multiple=True,
    help="Кадр «съёмка:имя» (старый:DSCF0122.RAF), который кладётся на выход, даже если не выбран; "
    "кадры разворота идут подряд: сначала старый, потом новый. Можно повторять",
)
@click.option("--dry-run", is_flag=True, help="Только manifest.csv, без копирования RAF")
@click.option("--overwrite", is_flag=True, help="Разрешить запись в непустую папку")
def main(
    old_dir: Path,
    old_csv: Path,
    new_dir: Path,
    new_csv: Path,
    out_dir: Path,
    cache_dir: Path,
    width: int,
    max_points: int,
    band: int,
    max_scale: float,
    min_pair_sim: float,
    dup_sim: float,
    tie: float,
    jobs: int,
    force_pick: tuple[str, ...],
    keep_both: tuple[str, ...],
    dry_run: bool,
    overwrite: bool,
) -> None:
    """Сопоставляет развороты двух съёмок и копирует по каждому самый резкий снимок.

    Args:
        old_dir: Папка старой съёмки (RAF).
        old_csv: CSV defocus_detection по старой съёмке (балл DOM).
        new_dir: Папка новой съёмки (RAF).
        new_csv: CSV defocus_detection по новой съёмке.
        out_dir: Куда класть выбранные RAF и manifest.csv.
        cache_dir: Кэш AKAZE-признаков.
        width: Ширина кадра для AKAZE.
        max_points: Предел числа точек на кадр.
        band: Полоса сравнения вокруг диагонали (в кадрах).
        max_scale: Допуск масштаба гомографии.
        min_pair_sim: Порог сходства пары разворотов между съёмками.
        dup_sim: Порог сходства пересъёмки внутри съёмки.
        tie: Разница DOM, ниже которой предпочитается новый кадр.
        jobs: Число процессов пула.
        force_pick: Кадры «съёмка:имя», которые выбираются вместо кадра с лучшим DOM (ручное исключение).
        keep_both: Невыбранные кадры «съёмка:имя» (например «старый:DSCF0122.RAF»), которые всё равно
            кладутся на выход рядом с выбранным.
        dry_run: Не копировать RAF, только записать manifest.csv.
        overwrite: Разрешить запись в непустую папку.

    Returns:
        None.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(p.suffix.upper() == ".RAF" for p in out_dir.iterdir()) and not overwrite:
        raise click.ClickException(f"В {out_dir} уже есть RAF; повторный запуск — с --overwrite")

    old = read_shoot(old_dir, old_csv, cache_dir, width, max_points, jobs)
    new = read_shoot(new_dir, new_csv, cache_dir, width, max_points, jobs)
    click.echo(f"Старая съёмка: {len(old)} кадров, новая: {len(new)}")
    # Имя из --keep-both, которого нет ни в одной съёмке, — почти наверняка опечатка: останавливаемся до копирования
    # Ключ кадра «съёмка:имя»: имена DSCF в двух съёмках повторяются
    known = {f"{Shoot.OLD.value}:{f.path.name}" for f in old} | {f"{Shoot.NEW.value}:{f.path.name}" for f in new}
    unknown = (set(keep_both) | set(force_pick)) - known
    if unknown:
        raise click.ClickException(
            f"--keep-both/--force-pick: нет таких кадров ни в одной съёмке: {', '.join(sorted(unknown))}"
        )

    # Шаг 1: пересъёмки внутри каждой съёмки — в группы
    old_groups = group_duplicates(old, dup_sim, max_scale, jobs)
    new_groups = group_duplicates(new, dup_sim, max_scale, jobs)
    click.echo(f"Разворотов (групп): старая {len(old_groups)}, новая {len(new_groups)}")

    # Шаг 2: сходство групп между съёмками в полосе вокруг диагонали (порядок разворотов один)
    n, m = len(old_groups), len(new_groups)
    pairs_g = [(gi, gj) for gi in range(n) for gj in range(m) if abs(gi * m / n - gj) <= band]
    frame_pairs = sorted({(i, j) for gi, gj in pairs_g for i in old_groups[gi] for j in new_groups[gj]})
    frame_sims = similarity_pairs(frame_pairs, old, new, max_scale, jobs, "Старая × новая")
    similarity = np.zeros((n, m))
    for gi, gj in pairs_g:
        similarity[gi, gj] = max(frame_sims[(i, j)] for i in old_groups[gi] for j in new_groups[gj])

    # Шаг 3: монотонное выравнивание
    units = [
        Unit(gi, gj, float(similarity[gi, gj]) if gi is not None and gj is not None else None)
        for gi, gj in align(similarity, min_pair_sim)
    ]

    # Шаг 4: выбор кадра по каждому развороту, копирование, манифест
    rows = []
    found_keep: set[str] = set()
    for number, unit in enumerate(tqdm(units, desc="Копируем" if not dry_run else "Выбираем"), 1):
        candidates = [(Shoot.OLD, old[i]) for i in (old_groups[unit.old_group] if unit.old_group is not None else [])]
        candidates += [(Shoot.NEW, new[j]) for j in (new_groups[unit.new_group] if unit.new_group is not None else [])]
        shoot, frame, reason = choose(candidates, tie)
        # Ручное исключение --force-pick: указанный кадр заменяет выбор по DOM
        forced = [(s, f) for s, f in candidates if f"{s.value}:{f.path.name}" in force_pick]
        if forced:
            shoot, frame = forced[0]
            reason = f"выбран вручную (--force-pick), по DOM был бы {choose(candidates, tie)[1].path.name}"
        found_keep |= {f"{s.value}:{f.path.name}" for s, f in candidates if f"{s.value}:{f.path.name}" in keep_both}
        # Выбранный кадр плюс исключения --keep-both: невыбранные кадры разворота, которые надо оставить тоже
        outputs = [(shoot, frame, reason)]
        outputs += [
            (s, f, "оставлен вдобавок (--keep-both)")
            for s, f in candidates
            if f is not frame and f"{s.value}:{f.path.name}" in keep_both
        ]
        # Несколько кадров одного разворота — сначала старый, потом новый, внутри съёмки — в порядке
        # съёмки (по имени файла); подындексы 1, 2, …
        outputs.sort(key=lambda item: (item[0] is Shoot.NEW, item[1].path.name))
        for sub, (out_shoot, out_frame, out_reason) in enumerate(outputs, 1):
            name = output_name(number, out_frame, out_shoot, sub if len(outputs) > 1 else None)
            target = out_dir / name
            # Уже скопированный файл того же размера не копируем заново: повторный прогон с
            # --overwrite не гонит 14 ГБ с NTFS ради одного добавленного кадра
            if not dry_run and not (target.exists() and target.stat().st_size == out_frame.path.stat().st_size):
                shutil.copy2(out_frame.path, target)
            rows.append(
                {
                    "number": number,
                    "output": name,
                    "chosen": out_shoot.value,
                    "reason": out_reason,
                    "similarity": "" if unit.similarity is None else f"{unit.similarity:.3f}",
                    "old_files": " ".join(f.path.name for s, f in candidates if s is Shoot.OLD),
                    "old_dom": " ".join(f"{f.dom:.3f}" for s, f in candidates if s is Shoot.OLD and f.dom is not None),
                    "new_files": " ".join(f.path.name for s, f in candidates if s is Shoot.NEW),
                    "new_dom": " ".join(f"{f.dom:.3f}" for s, f in candidates if s is Shoot.NEW and f.dom is not None),
                }
            )
    # Исключение, не нашедшее своего кадра, — почти наверняка опечатка в имени: молча не пропускаем
    # Исключение --keep-both должно было найтись в каком-то развороте (имя проверено до копирования)
    assert found_keep == set(keep_both), set(keep_both) - found_keep
    # RAF прошлых прогонов, которых нет в этом манифесте (например, разворот с --keep-both получил
    # подындекс и сменил имя), — убираем из выходной папки, иначе на выходе окажутся лишние кадры
    produced = {r["output"] for r in rows}
    stale = sorted(p for p in out_dir.glob("*.RAF") if p.name not in produced)
    for p in stale:
        click.echo(f"Удаляю устаревший файл прошлого прогона: {p.name}" + (" (dry-run — нет)" if dry_run else ""))
        if not dry_run:
            p.unlink()
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Сводка
    paired = [u for u in units if u.similarity is not None]
    click.echo(
        f"Разворотов на выходе: {len(units)}; пар: {len(paired)}, только старая: "
        f"{sum(u.new_group is None for u in units)}, только новая: {sum(u.old_group is None for u in units)}"
    )
    click.echo(
        f"Выбрано: новых {sum(r['chosen'] == Shoot.NEW.value for r in rows)}, "
        f"старых {sum(r['chosen'] == Shoot.OLD.value for r in rows)}"
    )
    if paired:
        sims = np.array([u.similarity for u in paired])
        click.echo(
            f"Сходство пар: min {sims.min():.3f}, p10 {np.quantile(sims, 0.1):.3f}, медиана {np.median(sims):.3f}"
        )
    click.echo(f"Манифест: {out_dir / 'manifest.csv'}" + (" (dry-run, RAF не копировались)" if dry_run else ""))


if __name__ == "__main__":
    main()
