"""Похожесть соседних кадров папки (AKAZE + RANSAC): кандидаты в повторную съёмку разворота и подсказка, какая половина сдвинулась.

Повторная съёмка разворота с поднятием страниц даёт два соседних кадра одного разворота: на
одном поднята левая страница, на другом правая. У такой пары много общих точек, но общая
гомография хорошо ложится только на ту половину, что в обоих кадрах лежит одинаково.
Поэтому кроме общей доли инлаеров считается доля инлаеров отдельно в левой и правой половине
первого кадра: заметно меньшая доля у одной половины — подсказка, что эту страницу в одном из
кадров поднимали. Это только подсказка для отсмотра глазами, а не решение.
"""

import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

# Ширина копии кадра для поиска точек: текст ещё различим, а AKAZE на ней — доли секунды.
WORK_WIDTH = 1400
# Сколько самых сильных точек кадра оставлять (ограничивает время сопоставления).
MAX_POINTS = 4000
# Порог Lowe's ratio test для однозначных совпадений.
RATIO = 0.75
# Порог RANSAC гомографии на копии шириной WORK_WIDTH, пиксели.
RANSAC_PX = 5.0
# Меньше стольких однозначных совпадений — кадры считаются разными.
MIN_MATCHES = 10
# На каком расстоянии по порядку сравнивать кадры (1 — соседние, 2 — через один).
DEFAULT_GAPS = (1, 2)


@dataclass(frozen=True)
class PairScore:
    """Сходство пары кадров: общая доля инлаеров и доли по половинам первого кадра."""

    frame_a: str  # имя первого кадра
    frame_b: str  # имя второго кадра (позже по порядку)
    matches: int  # однозначных совпадений
    inliers: int  # инлаеров общей гомографии
    share: float  # инлаеры / меньшее число точек пары: 0 — общего нет
    share_left: float  # инлаеры / совпадения в левой половине кадра A
    share_right: float  # инлаеры / совпадения в правой половине кадра A
    shift_px: float  # сдвиг центра кадра гомографией, px копии шириной WORK_WIDTH


def init_worker() -> None:
    """Инициализатор воркера: один поток OpenCV и BLAS, без пометки hugepage (см. gpu_and_pools)."""
    cv2.setNumThreads(1)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # noqa: BLE001 — без threadpoolctl остаётся только окружение
        pass
    try:
        np._core.multiarray._set_madvise_hugepage(False)
    except AttributeError:
        pass


def frame_features(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """AKAZE-точки кадра на уменьшенной серой копии.

    Args:
        path: Файл кадра (JPG/PNG/TIFF).

    Returns:
        Кортеж (точки (n, 2) float32, дескрипторы (n, 61) uint8, (ширина, высота) копии).
        Нечитаемый кадр или кадр без точек — пустые массивы.
    """
    gray = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if gray is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 61), np.uint8), (WORK_WIDTH, 1)
    h, w = gray.shape
    size = (WORK_WIDTH, max(1, int(h * WORK_WIDTH / w)))
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    keypoints, descriptors = cv2.AKAZE_create().detectAndCompute(small, None)
    if descriptors is None or not keypoints:
        return np.zeros((0, 2), np.float32), np.zeros((0, 61), np.uint8), size
    # Оставляем самые сильные точки по отклику детектора.
    order = np.argsort([-kp.response for kp in keypoints])[:MAX_POINTS]
    return np.float32([keypoints[i].pt for i in order]), descriptors[order], size


def pair_score(name_a: str, feat_a: tuple, name_b: str, feat_b: tuple) -> PairScore:
    """Сходство двух кадров по их AKAZE-точкам.

    Args:
        name_a: Имя первого кадра.
        feat_a: Результат ``frame_features`` первого кадра.
        name_b: Имя второго кадра.
        feat_b: Результат ``frame_features`` второго кадра.

    Returns:
        ``PairScore``; при недостатке совпадений или без гомографии — нули.
    """
    pts_a, desc_a, (width, height) = feat_a
    pts_b, desc_b, _ = feat_b
    zero = PairScore(name_a, name_b, 0, 0, 0.0, 0.0, 0.0, 0.0)
    if len(desc_a) < 2 or len(desc_b) < 2:
        return zero
    # Однозначные совпадения: лучшее заметно ближе второго.
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < RATIO * m[1].distance]
    if len(good) < MIN_MATCHES:
        return PairScore(name_a, name_b, len(good), 0, 0.0, 0.0, 0.0, 0.0)
    src = pts_a[[m.queryIdx for m in good]]
    dst = pts_b[[m.trainIdx for m in good]]
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if homography is None:
        return PairScore(name_a, name_b, len(good), 0, 0.0, 0.0, 0.0, 0.0)
    inl = mask.ravel().astype(bool)
    # Доли инлаеров по половинам кадра A: страница, которую в одном из кадров поднимали,
    # не ложится на общую гомографию и даёт меньшую долю.
    left = src[:, 0] < width / 2
    share_left = float(inl[left].mean()) if left.any() else 0.0
    share_right = float(inl[~left].mean()) if (~left).any() else 0.0
    # Сдвиг центра кадра гомографией — грубая мера «камера/книга сдвинулась».
    centre = np.float32([[[width / 2, height / 2]]])
    moved = cv2.perspectiveTransform(centre, homography)[0, 0]
    shift = float(np.hypot(*(moved - centre[0, 0])))
    share = float(inl.sum()) / min(len(desc_a), len(desc_b))
    return PairScore(name_a, name_b, len(good), int(inl.sum()), share, share_left, share_right, shift)


def _pair_task(args: tuple) -> PairScore:
    """Обёртка ``pair_score`` для пула: аргументы одним кортежем."""
    return pair_score(*args)


def neighbor_scores(paths: list[Path], jobs: int, gaps: tuple[int, ...] = DEFAULT_GAPS) -> list[PairScore]:
    """Сходство кадров на расстоянии ``gaps`` по порядку имён.

    Args:
        paths: Кадры папки в порядке съёмки (по имени).
        jobs: Число процессов пула.
        gaps: Расстояния по порядку, на которых сравнивать кадры.

    Returns:
        Оценки пар в порядке (первый кадр, расстояние).
    """
    ctx = mp.get_context("forkserver")
    with ProcessPoolExecutor(jobs, mp_context=ctx, initializer=init_worker) as pool:
        feats = list(pool.map(frame_features, paths, chunksize=2))
        tasks = [
            (paths[i].name, feats[i], paths[i + g].name, feats[i + g])
            for i in range(len(paths))
            for g in gaps
            if i + g < len(paths)
        ]
        return list(pool.map(_pair_task, tasks, chunksize=4))


def write_scores(scores: list[PairScore], out_csv: Path) -> None:
    """Пишет оценки пар в CSV (доли — 3 знака).

    Args:
        scores: Оценки пар.
        out_csv: Путь CSV; папка создаётся.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([f.name for f in fields(PairScore)])
        for s in scores:
            writer.writerow([round(v, 3) if isinstance(v, float) else v for v in astuple(s)])
