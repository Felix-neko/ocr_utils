"""Отладочный оверлей: что каждый детектор увидел на полосе.

Одна картинка на полосу, панель на детектор: копия 150 dpi, поверх неё сырые измерения
(тайлы с углами, центр-линии строк, ленты сдвигов, полигоны Surya) и подпись с метриками.
Смотреть глазами оверлей, а не CSV, — единственный способ понять, ПОЧЕМУ детектор
сработал: на кривой строке или на слипшейся с линейкой таблице.

Сырые данные берутся из кэша измерений (``--cache-dir``) или из самой меры, если прогон
шёл с ``keep_raw``; полоса перечитывается заново — оверлеи пишутся постфактум, по
находкам, и хранить ради них кадры всего пака нельзя.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ocr_utils.scan_markup.curved_lines.cache import PageCache
from ocr_utils.scan_markup.curved_lines.detectors import DETECTORS, Detector
from ocr_utils.page_layout.orientation.image_io import read_frame

logger = logging.getLogger(__name__)

CAPTION_LINES = 3
CAPTION_LINE_PX = 18
JPEG_QUALITY = 85


def _caption(name: str, info: dict | None) -> list[str]:
    """Подпись панели: имя, score, флаг и до четырёх метрик. Только латиница — шрифт OpenCV."""
    if info is None:
        return [name, "no data", ""]
    if info.get("silent"):
        return [f"{name}: silent", str(info.get("note", ""))[:60], ""]
    flag = "FLAG" if info.get("flag") else "ok"
    head = f"{name}: {flag} score={info.get('score', 0.0):.2f}"
    metrics = info.get("metrics", {})
    keys = list(info.get("keys", []))[:4]
    body = "  ".join(f"{key}={metrics[key]:.3f}" for key in keys if key in metrics)
    rest = "  ".join(f"{key}={value:.3f}" for key, value in metrics.items() if key not in keys)[:90]
    return [head, body, rest]


def render(
    gray150: np.ndarray, raws: dict[str, dict | None], detectors: Sequence[Detector], infos: dict[str, dict]
) -> np.ndarray:
    base = cv2.cvtColor(gray150, cv2.COLOR_GRAY2BGR)
    panels = []
    for detector in detectors:
        canvas = base.copy()
        raw = raws.get(detector.name)
        if raw and detector.draw is not None:
            scale = gray150.shape[1] / float(raw.get("w") or gray150.shape[1])
            try:
                detector.draw(canvas, raw, scale)
            except Exception as error:  # рисовалка не должна ронять прогон
                cv2.putText(
                    canvas, f"draw failed: {error}"[:80], (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                )
        top = CAPTION_LINES * CAPTION_LINE_PX + 6
        panel = cv2.copyMakeBorder(canvas, top, 0, 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for index, text in enumerate(_caption(detector.name, infos.get(detector.name))):
            cv2.putText(
                panel,
                text,
                (4, 14 + index * CAPTION_LINE_PX),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        panels.append(panel)
    return np.hstack(panels) if panels else base


@dataclass(frozen=True)
class OverlayJob:
    path: Path
    rel_path: str
    out_path: Path
    default_dpi: int
    names: tuple[str, ...]
    cache_root: Path | None
    raws: dict[str, dict | None]  # если сырые данные уже в памяти (прогон с keep_raw)
    infos: dict[str, dict]


def _init_worker() -> None:
    cv2.setNumThreads(1)
    try:
        os.nice(10)
    except OSError:
        pass
    try:  # см. analysis._init_worker: без этого дюжина воркеров толкается в ядре на больших массивах
        np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
    except Exception:
        pass


def _write_one(job: OverlayJob) -> str:
    try:
        frame, _ = read_frame(job.path, job.rel_path, default_dpi=job.default_dpi, gpu_side=0)
    except Exception as error:
        return f"{job.rel_path}: чтение: {error}"
    raws = dict(job.raws)
    if job.cache_root is not None:
        cache = PageCache(job.cache_root)
        for name in job.names:
            if raws.get(name) is None:
                payload = cache.load(job.path, name, DETECTORS[name].cache_key)
                if payload is not None:
                    raws[name] = payload.get("raw")
    image = render(frame.gray150, raws, [DETECTORS[name] for name in job.names], job.infos)
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(job.out_path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return ""


def write_many(jobs: Sequence[OverlayJob], workers: int, progress: bool = True) -> list[str]:
    """Пишет оверлеи в пуле; возвращает список ошибок."""
    errors: list[str] = []
    if not jobs:
        return errors
    if workers <= 1 or len(jobs) <= 1:
        outcomes = (_write_one(job) for job in jobs)
    else:
        context = multiprocessing.get_context("forkserver")
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker)
        outcomes = pool.map(_write_one, jobs)
    if progress:
        from tqdm import tqdm

        outcomes = tqdm(outcomes, total=len(jobs), desc="оверлеи", unit="полоса")
    for outcome in outcomes:
        if outcome:
            errors.append(outcome)
    logger.info("Оверлеев: %d, ошибок: %d", len(jobs) - len(errors), len(errors))
    return errors
