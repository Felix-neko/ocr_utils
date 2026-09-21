"""Движки стенда: ядро (v14) и v15 — измерение страницы, пороги, версия кэша.

Стенд (``research.geometry_regression.cli``) выбирает движок опцией ``--engine``; всё
остальное — пул, кэш surya, CSV, картинки — общее. Кэш измерений у движков раздельный:
JSON с другой ``version`` не читается и переписывается.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import fitz

from ocr_utils.geometry_regression import VERSION
from ocr_utils.geometry_regression.cache import measure_page as core_measure_page
from ocr_utils.geometry_regression.cache import surya_source_for
from ocr_utils.geometry_regression.metrics import PageMeasure, Params
from ocr_utils.geometry_regression.regions import layout_regions
from ocr_utils.geometry_regression.render import render_gray
from ocr_utils.geometry_regression.scoring import Thresholds
from research.geometry_regression.v15 import ENGINE_VERSION
from research.geometry_regression.v15.scoring import Thresholds15


def measure_page_v15(
    geo_doc: fitz.Document, nogeo_doc: fitz.Document, page: int, params: Params, surya=None
) -> PageMeasure:
    """Измерить одну пару страниц движком v15 (интерфейс как у ``cache.measure_page`` ядра)."""
    from research.geometry_regression.v15.measure import measure_pair

    started = time.time()
    if surya is None:
        surya = surya_source_for(params)
    regions = layout_regions(nogeo_doc, page - 1, params.dpi, surya)
    measure = measure_pair(
        render_gray(nogeo_doc, page - 1),
        render_gray(geo_doc, page - 1),
        params,
        regions.drawings,
        regions.tables,
        regions.raster,
    )
    measure.metrics["seconds"] = round(time.time() - started, 2)
    measure.metrics["layout_surya"] = float(surya is not None)
    return measure


@dataclass(frozen=True)
class Engine:
    """Движок: имя, версия кэша, измерение страницы и класс порогов."""

    name: str
    version: object
    measure_page: Callable
    thresholds: type

    def load_page_cache(self, path: Path) -> dict | None:
        """Метрики страницы из JSON, если он есть и той же версии движка."""
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != self.version or "metrics" not in payload:
            return None
        return payload

    def save_page_cache(self, path: Path, measure: PageMeasure) -> None:
        """Записать измерение в JSON версии движка (поля те же, что у ядра)."""
        from ocr_utils.page_layout import version_tag

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "page_layout": version_tag(),
            "metrics": measure.metrics,
            "culprits": measure.culprits,
            "raw": measure.raw,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


ENGINES: dict[str, Engine] = {
    "core": Engine("core", VERSION, core_measure_page, Thresholds),
    "v15": Engine("v15", ENGINE_VERSION, measure_page_v15, Thresholds15),
}


def get_engine(name: str) -> Engine:
    """Движок по имени (``core`` | ``v15``)."""
    if name not in ENGINES:
        raise KeyError(f"неизвестный движок {name!r}; есть: {', '.join(ENGINES)}")
    return ENGINES[name]


__all__ = ["Engine", "ENGINES", "get_engine", "measure_page_v15"]
