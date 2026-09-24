"""Адаптер pero: нейросетевой ParseNet (базовые линии, полигоны строк и регионы)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ocr_utils.curved_layout import RENDER_DPI, WORK_DPI
from ocr_utils.curved_layout.engines.base import EngineLine, EngineResult
from ocr_utils.curved_layout.engines.external import ENGINES_ROOT, polygon_height, polyline, run_worker

# Общая европейская модель печатных газет: она ближе всего к нашему материалу.
DEFAULT_CONFIG = ENGINES_ROOT / "pero_model" / "pero_eu_cz_print_newspapers_2022-09-26" / "config.ini"


class PeroEngine:
    """Строки от pero: базовая линия, полигон строки и высоты над/под линией."""

    name = "pero"

    def __init__(self, python: Path | None = None, config: Path | None = None) -> None:
        """Args:
        python: Интерпретатор окружения pero; по умолчанию ``ENGINES_ROOT/pero/bin/python``.
        config: ``config.ini`` модели; по умолчанию газетная европейская модель.
        """
        self.python = Path(python) if python else ENGINES_ROOT / "pero" / "bin" / "python"
        self.config = Path(config) if config else DEFAULT_CONFIG

    def segment(self, gray300: np.ndarray, dpi: float = WORK_DPI) -> EngineResult:
        """Строки страницы в пикселях рабочей копии ``dpi``."""
        started = time.monotonic()
        data = run_worker(self.python, "pero_worker.py", gray300, extra=[str(self.config)])
        scale = dpi / RENDER_DPI
        lines = []
        for item in data["lines"]:
            baseline = polyline(item["baseline"], scale)
            boundary = polyline(item["boundary"], scale) if item.get("boundary") else None
            # Высоту берём из пары (над линией, под линией), а если её нет — из полигона.
            height = float(item.get("height", 0.0)) * scale
            if height <= 0.0 and boundary is not None:
                height = polygon_height(boundary)
            lines.append(EngineLine(points=baseline, polygon=boundary, height=height, baseline=True))
        regions = [polyline(region, scale) for region in data.get("regions", []) if region]
        return EngineResult(
            lines=lines,
            regions=regions,
            engine=self.name,
            seconds=time.monotonic() - started,
            note=f"строк {len(lines)}",
        )


__all__ = ["PeroEngine"]
