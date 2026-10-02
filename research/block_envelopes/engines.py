"""Чужие движки на тестовом множестве: регионы текста pero, kraken, eynollah, chronicling и блоки surya layout из кэша — в JSON и оверлеи рядом с нашими границами."""

from __future__ import annotations

import csv
import json
import time
from enum import Enum
from pathlib import Path

import numpy as np

from ocr_utils.page_layout.pack_analysis.stages import PageTask, load_image
from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI

# Кэш surya layout заострённых сканов пака-1 (``page_layout prefill-surya``).
SURYA_CACHE = Path("/mnt/hotstore/scan_processing/mts/pack1_page_layout/sharpened")


class EngineKind(str, Enum):
    """Чужой движок, отдающий регионы (блоки) текста."""

    SURYA = "surya_layout"  # боксы разметки surya из кэша, без пересчёта
    PERO = "pero"
    KRAKEN = "kraken"
    EYNOLLAH = "eynollah"
    CHRONICLING = "chronicling"


def _engine(kind: EngineKind):
    """Адаптер движка из ``text_blocks.engines`` (импорт по требованию: окружения у всех свои)."""
    if kind is EngineKind.PERO:
        from ocr_utils.page_layout.text_blocks.engines.pero import PeroEngine

        return PeroEngine()
    if kind is EngineKind.KRAKEN:
        from ocr_utils.page_layout.text_blocks.engines.kraken import KrakenEngine

        return KrakenEngine()
    if kind is EngineKind.EYNOLLAH:
        from ocr_utils.page_layout.text_blocks.engines.eynollah import EynollahEngine

        return EynollahEngine()
    if kind is EngineKind.CHRONICLING:
        from ocr_utils.page_layout.text_blocks.engines.catalog import WorkerEngineName, spec_of
        from ocr_utils.page_layout.text_blocks.engines.generic import WorkerEngine

        return WorkerEngine(spec_of(WorkerEngineName.CHRONICLING))
    raise ValueError(f"У движка {kind.value} нет адаптера строк")


def surya_regions(key: str) -> tuple[list[np.ndarray], list[str]]:
    """Боксы surya layout полосы из кэша — прямоугольники в пикселях рабочей копии.

    Args:
        key: Полоса ``год/выпуск/полоса``.

    Returns:
        ``(контуры, метки)``; кэша нет — пустые списки.
    """
    path = SURYA_CACHE / f"{key}.json"
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text(encoding="utf-8"))
    scale = WORK_DPI / float(payload["frame"]["dpi"])
    polygons, labels = [], []
    for block in payload["blocks"]:
        x0, y0, x1, y1 = (value * scale for value in block["box"])
        polygons.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64))
        labels.append(block["label"])
    return polygons, labels


def rotation_of(key: str, pack_dir: Path) -> int:
    """Поворот полосы в разборе пака — по ``index.csv`` (он не переписывается при пересборке ``work/``).

    Args:
        key: Полоса.
        pack_dir: Выход разбора пака.

    Returns:
        Поворот по часовой, градусы; полосы нет в индексе — 0.
    """
    with (pack_dir / "index.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["page"] == key:
                return int(row["rotate_cw"])
    return 0


def run_engine(kind: EngineKind, key: str, pack_dir: Path, sharpened_dir: Path) -> dict:
    """Регионы и строки движка на полосе в пикселях рабочей копии.

    Args:
        kind: Движок.
        key: Полоса.
        pack_dir: Выход разбора пака (оттуда поворот полосы).
        sharpened_dir: Заострённые сканы.

    Returns:
        Словарь ``{"key", "engine", "seconds", "regions": [[[x, y], …]], "labels", "lines": [[[x, y], …]]}``.
    """
    started = time.monotonic()
    if kind is EngineKind.SURYA:
        regions, labels = surya_regions(key)
        return {
            "key": key,
            "engine": kind.value,
            "seconds": time.monotonic() - started,
            "regions": [region.tolist() for region in regions],
            "labels": labels,
            "lines": [],
        }
    image = load_image(PageTask(key, sharpened_dir / f"{key}.jpg"), rotation_of(key, pack_dir))
    result = _engine(kind).segment(image.gray_at(RENDER_DPI), WORK_DPI)
    return {
        "key": key,
        "engine": kind.value,
        "seconds": time.monotonic() - started,
        "regions": [np.asarray(region, dtype=np.float64).tolist() for region in result.regions],
        "labels": [kind.value] * len(result.regions),
        "lines": [np.asarray(line.points, dtype=np.float64).tolist() for line in result.lines],
    }


__all__ = ["EngineKind", "run_engine", "surya_regions"]
