"""Пул процессов стенда: forkserver, один поток BLAS и OpenCV на воркер, без hugepage (``.claude/rules/gpu_and_pools.md``)."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

# Пользователю нужна машина: по умолчанию два воркера (разбор держит ~0.4 ГБ на процесс, упор — в CPU).
DEFAULT_JOBS = 2


def init_worker() -> None:
    """Инициализатор воркера: один поток у OpenCV и BLAS, без пометки hugepage, пониженный приоритет."""
    cv2.setNumThreads(1)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # noqa: BLE001 — без threadpoolctl остаётся окружение
        pass
    try:
        np._core.multiarray._set_madvise_hugepage(False)
    except AttributeError:
        pass
    try:
        os.nice(10)
    except OSError:
        pass


def make_pool(jobs: int) -> ProcessPoolExecutor:
    """Пул на ``jobs`` воркеров (forkserver, :func:`init_worker`).

    Args:
        jobs: Число воркеров.

    Returns:
        Пул.
    """
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    return ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("forkserver"), initializer=init_worker)


__all__ = ["DEFAULT_JOBS", "init_worker", "make_pool"]
