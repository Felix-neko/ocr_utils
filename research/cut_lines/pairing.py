"""Пары кадров двух фотосетов одной подшивки: SIFT по уменьшенным кадрам, доля инлайеров гомографии, монотонное выравнивание."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

from merge_best_shoots import align
from research.cut_lines.frames import init_worker, read_bgr

log = logging.getLogger(__name__)

# Ширина уменьшенного кадра для поиска пар, px: текст ещё различим, SIFT быстрый.
PAIR_WIDTH = 1400
# Сколько самых сильных точек SIFT оставлять на кадр.
PAIR_POINTS = 6000
# Допуск масштаба гомографии между сетами (разные камеры и расстояние до подшивки).
MAX_SCALE = 1.6
# Порог сходства «тот же разворот» для выравнивания последовательностей (доля инлайеров от меньшего числа точек).
MIN_SIM = 0.02


@dataclass(frozen=True)
class FramePair:
    """Пара кадров: кадр основы (1) и кадр-донор (2) с тем же разворотом.

    Поля:
        base: Путь кадра (1).
        donor: Путь кадра (2).
        similarity: Доля инлайеров гомографии.
    """

    base: Path
    donor: Path
    similarity: float


def sift_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Точки и дескрипторы SIFT кадра, уменьшенного до :data:`PAIR_WIDTH`.

    Args:
        path: Путь кадра.

    Returns:
        ``(точки (n, 2) float32, дескрипторы (n, 128) float32)``; точки — в долях ширины кадра (масштаб сетов разный).
    """
    bgr = read_bgr(path)
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (PAIR_WIDTH, int(h * PAIR_WIDTH / w)), interpolation=cv2.INTER_AREA)
    kp, desc = cv2.SIFT_create(nfeatures=PAIR_POINTS).detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None)
    if desc is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.float32([k.pt for k in kp]) / PAIR_WIDTH, desc


def similarity(fa: tuple[np.ndarray, np.ndarray], fb: tuple[np.ndarray, np.ndarray]) -> float:
    """Сходство двух кадров: доля RANSAC-инлайеров гомографии от меньшего числа точек.

    Args:
        fa: Признаки первого кадра (``sift_features``).
        fb: Признаки второго кадра.

    Returns:
        Число от 0 до 1; 0 — мало совпадений, гомографии нет или масштаб вне допуска.
    """
    (pa, da), (pb, db) = fa, fb
    if len(da) < 10 or len(db) < 10:
        return 0.0
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 12:
        return 0.0
    src = pa[[m.queryIdx for m in good]] * PAIR_WIDTH
    dst = pb[[m.trainIdx for m in good]] * PAIR_WIDTH
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    if homography is None:
        return 0.0
    scale = float(np.sqrt(abs(np.linalg.det(homography[:2, :2]))))
    if not (1 / MAX_SCALE <= scale <= MAX_SCALE):
        return 0.0
    return float(mask.sum()) / min(len(da), len(db))


def _similarity_task(args: tuple[int, int, tuple, tuple]) -> tuple[int, int, float]:
    """Задача пула: сходство пары кадров по индексам.

    Args:
        args: ``(i, j, признаки i, признаки j)``.

    Returns:
        ``(i, j, сходство)``.
    """
    i, j, fa, fb = args
    return i, j, similarity(fa, fb)


def pair_frames(base: list[Path], donor: list[Path], jobs: int) -> list[FramePair]:
    """Пары «кадр (1) — кадр (2)» с тем же разворотом.

    Сходство считается для всех пар (кадров немного), пары выбираются монотонным выравниванием
    (``merge_best_shoots.align``): порядок разворотов в обоих сетах один, пересъёмки и пропуски остаются без пары.

    Args:
        base: Кадры (1) в порядке съёмки.
        donor: Кадры (2) в порядке съёмки.
        jobs: Число процессов.

    Returns:
        Пары с ненулевым сходством.
    """
    ctx = get_context("forkserver")
    with ProcessPoolExecutor(jobs, mp_context=ctx, initializer=init_worker) as pool:
        feats_a = list(pool.map(sift_features, base))
        feats_b = list(pool.map(sift_features, donor))
        tasks = [(i, j, feats_a[i], feats_b[j]) for i in range(len(base)) for j in range(len(donor))]
        sim = np.zeros((len(base), len(donor)))
        for i, j, s in pool.map(_similarity_task, tasks, chunksize=16):
            sim[i, j] = s
    steps = align(sim, MIN_SIM)
    pairs = [FramePair(base[i], donor[j], float(sim[i, j])) for i, j in steps if i is not None and j is not None]
    # Пересъёмки в (1) (дубли разворота) выравнивание оставляет без пары — добираем лучшим кадром (2).
    paired = {p.base for p in pairs}
    for i, path in enumerate(base):
        if path not in paired and sim[i].max() >= MIN_SIM * 2:
            j = int(sim[i].argmax())
            pairs.append(FramePair(path, donor[j], float(sim[i, j])))
            log.info("%s: без пары в выравнивании, взят лучший кадр (2) %s (сходство %.3f)", path.name, donor[j].name,
                     sim[i, j])  # fmt: skip
    return sorted(pairs, key=lambda p: p.base.name)
