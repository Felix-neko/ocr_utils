"""Адаптер eynollah: сегментация SBB с точными контурами кривых строк (PAGE-XML)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ocr_utils.curved_layout import RENDER_DPI, WORK_DPI
from ocr_utils.curved_layout.engines.base import EngineLine, EngineResult
from ocr_utils.curved_layout.engines.external import (
    ENGINES_ROOT,
    centre_from_polygon,
    polygon_height,
    polyline,
    run_worker,
)

# Базовый каталог моделей: сам eynollah дописывает к нему подкаталог ``models_eynollah``
# (модели скачаны с Zenodo, ``models_inference_layout_v0_9_1.zip``).
DEFAULT_MODELS = ENGINES_ROOT / "models"


class EynollahEngine:
    """Строки от eynollah в режиме ``--curved-line``: контур строки, базовая линия — если есть."""

    name = "eynollah"

    def __init__(self, python: Path | None = None, models: Path | None = None) -> None:
        """Args:
        python: Интерпретатор окружения eynollah; по умолчанию ``ENGINES_ROOT/eynollah/bin/python``.
        models: Каталог моделей eynollah.
        """
        self.python = Path(python) if python else ENGINES_ROOT / "eynollah" / "bin" / "python"
        self.models = Path(models) if models else DEFAULT_MODELS

    def segment(self, gray300: np.ndarray, dpi: float = WORK_DPI) -> EngineResult:
        """Строки страницы в пикселях рабочей копии ``dpi``."""
        started = time.monotonic()
        data = run_worker(self.python, "eynollah_worker.py", gray300, extra=[str(self.models)], timeout=1800)
        scale = dpi / RENDER_DPI
        lines = []
        for item in data["lines"]:
            boundary = polyline(item["boundary"], scale) if item.get("boundary") else None
            points = polyline(item["baseline"], scale) if item.get("baseline") else None
            is_baseline = points is not None and len(points) >= 2
            if not is_baseline:
                # Часть строк приходит только контуром: ось берём серединой его толщины.
                points = centre_from_polygon(boundary) if boundary is not None else None
            if points is None or len(points) < 2:
                continue
            height = polygon_height(boundary) if boundary is not None else 0.0
            lines.append(EngineLine(points=points, polygon=boundary, height=float(height), baseline=is_baseline))
        regions = [polyline(region, scale) for region in data.get("regions", []) if region]
        return EngineResult(
            lines=lines,
            regions=regions,
            engine=self.name,
            seconds=time.monotonic() - started,
            note=f"строк {len(lines)}",
        )


__all__ = ["EynollahEngine"]
