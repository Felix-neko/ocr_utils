"""Мера страницы движком v16 и его кэш JSON: поле смещений, штрихи, line art, фото — вход мер v17–v18 (``quality``)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import fitz

from ocr_utils.geometry_regression.cache import surya_source_for
from ocr_utils.geometry_regression.metrics import PageMeasure, Params
from ocr_utils.geometry_regression.regions import layout_regions
from ocr_utils.geometry_regression.render import render_gray
from ocr_utils.geometry_regression.v16 import ENGINE_VERSION


def measure_page(
    geo_doc: fitz.Document, nogeo_doc: fitz.Document, page: int, params: Params, surya=None
) -> PageMeasure:
    """Измерить одну пару страниц движком v16 (интерфейс как у ``cache.measure_page`` ядра v14).

    Args:
        geo_doc: Открытый PDF с коррекцией геометрии («стало», A).
        nogeo_doc: Открытый PDF без коррекции («было», B).
        page: Номер страницы, с единицы.
        params: Параметры измерения (размеры, привязанные к бумаге; кэш surya).
        surya: Источник блоков surya для рамок таблиц, рисунков и растра; ``None`` — собрать из ``params``
            (кэш только для чтения), а без кэша — рамки без surya.

    Returns:
        :class:`PageMeasure` с ``metrics["seconds"]`` и ``metrics["layout_surya"]``.
    """
    from ocr_utils.geometry_regression.v16.measure import measure_pair

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


def cache_path(run_dir: Path, pdf_stem: str, page: int) -> Path:
    """JSON страницы в прогоне v16: ``<run_dir>/cache/<pdf>/pNNN.json`` (номер с единицы)."""
    return Path(run_dir) / "cache" / pdf_stem / f"p{page:03d}.json"


def load_cache(path: Path) -> dict | None:
    """JSON страницы, если он есть и версии движка v16; иначе ``None``."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != ENGINE_VERSION or "metrics" not in payload:
        return None
    return payload


def save_cache(path: Path, measure: PageMeasure) -> None:
    """Записать измерение v16 в JSON (поля — как у кэша ядра: версия, тег разбора, метрики, виновники, сырьё)."""
    from ocr_utils.page_layout import version_tag

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": ENGINE_VERSION,
        "page_layout": version_tag(),
        "metrics": measure.metrics,
        "culprits": measure.culprits,
        "raw": measure.raw,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


__all__ = ["cache_path", "load_cache", "measure_page", "save_cache"]
