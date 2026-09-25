"""Адаптер kraken: нейросетевая сегментация blla (базовые линии и полигоны строк)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ocr_utils.curved_layout import RENDER_DPI, WORK_DPI
from ocr_utils.curved_layout.engines.base import EngineLine, EngineResult
from ocr_utils.curved_layout.engines.external import ENGINES_ROOT, polygon_height, polyline, run_worker


class KrakenEngine:
    """Строки от kraken ``blla``: базовая линия плюс полигон строки, модель идёт в пакете."""

    name = "kraken"

    def __init__(self, python: Path | None = None) -> None:
        """Args:
        python: Интерпретатор окружения kraken; по умолчанию ``ENGINES_ROOT/kraken/bin/python``.
        """
        self.python = Path(python) if python else ENGINES_ROOT / "kraken" / "bin" / "python"

    def segment(self, gray300: np.ndarray, dpi: float = WORK_DPI) -> EngineResult:
        """Строки страницы в пикселях рабочей копии ``dpi``."""
        started = time.monotonic()
        data = run_worker(self.python, "kraken_worker.py", gray300)
        scale = dpi / RENDER_DPI
        lines = []
        for item in data["lines"]:
            baseline = polyline(item["baseline"], scale)
            boundary = polyline(item["boundary"], scale) if item.get("boundary") else None
            height = polygon_height(boundary) if boundary is not None else 0.0
            lines.append(EngineLine(points=baseline, polygon=boundary, height=float(height), baseline=True))
        regions = [polyline(region, scale) for region in data.get("regions", []) if region]
        return EngineResult(
            lines=lines,
            regions=regions,
            engine=self.name,
            seconds=time.monotonic() - started,
            note=f"строк {len(lines)}",
        )


__all__ = ["KrakenEngine"]
