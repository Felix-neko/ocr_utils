"""Кэш измерений по страницам: JSON на страницу и вердикт «с пересчётом при промахе».

Один и тот же кэш пишет стенд ``research.geometry_regression run`` (по всему паку) и читает
сборщик финальных PDF: если для страницы уже есть JSON текущей версии детектора — вердикт
получается за миллисекунды, иначе пара страниц рендерится и меряется на месте (~3 с) и JSON
дописывается, чтобы следующий запуск его нашёл. Вердикт в JSON не хранится: он зависит от
порогов и всегда вычисляется по метрикам через :class:`Thresholds`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import fitz

from ocr_utils.geometry_regression import VERSION
from ocr_utils.geometry_regression.metrics import PageMeasure, Params, measure_pair
from ocr_utils.geometry_regression.regions import layout_regions
from ocr_utils.geometry_regression.render import render_gray
from ocr_utils.geometry_regression.scoring import Thresholds, Verdict


def cache_path(run_dir: Path, pdf_stem: str, page: int) -> Path:
    """Путь JSON страницы в каталоге прогона.

    Args:
        run_dir: Каталог прогона (в нём лежит ``cache/``).
        pdf_stem: Имя PDF без расширения (``full_1966_01``).
        page: Номер страницы, с единицы.

    Returns:
        ``<run_dir>/cache/<pdf_stem>/pNNN.json``.
    """
    return Path(run_dir) / "cache" / pdf_stem / f"p{page:03d}.json"


def load_page_cache(path: Path) -> dict | None:
    """Метрики страницы из JSON, если он есть и той же версии детектора.

    Args:
        path: Путь JSON (:func:`cache_path`).

    Returns:
        Содержимое JSON (``version``, ``metrics``, ``culprits``, ``raw``) или ``None``, если
        файла нет, он другой версии или без метрик.
    """
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != VERSION or "metrics" not in payload:
        return None
    return payload


def save_page_cache(path: Path, measure: PageMeasure) -> None:
    """Записать измерение страницы в JSON текущей версии.

    Args:
        path: Путь JSON (:func:`cache_path`).
        measure: Результат :func:`measure_pair`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    from ocr_utils.page_layout import version_tag

    payload = {
        "version": VERSION,
        "page_layout": version_tag(),
        "metrics": measure.metrics,
        "culprits": measure.culprits,
        "raw": measure.raw,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@dataclass(frozen=True)
class PageVerdict:
    """Вердикт по странице и откуда он взялся."""

    verdict: Verdict
    metrics: dict[str, float]
    cached: bool  # True — метрики прочитаны из кэша, False — измерены сейчас
    seconds: float  # время измерения (0 при попадании в кэш)


def surya_source_for(params: Params, model=None):
    """Источник surya по параметрам: кэш из ``params.layout_cache_dir`` (+ модель в родителе) или ``None``."""
    from ocr_utils.page_layout.surya.source import OnMiss, SuryaSourceConfig

    if params.layout_cache_dir is None:
        return None
    return SuryaSourceConfig(Path(params.layout_cache_dir), OnMiss(params.layout_on_miss)).open(model)


def measure_page(
    geo_doc: fitz.Document, nogeo_doc: fitz.Document, page: int, params: Params, surya=None
) -> PageMeasure:
    """Измерить одну пару страниц (без коррекции → с коррекцией).

    Args:
        geo_doc: Открытый PDF с коррекцией геометрии («стало», A).
        nogeo_doc: Открытый PDF без коррекции («было», B).
        page: Номер страницы, с единицы (одинаковый в обоих PDF).
        params: Параметры детектора.
        surya: Источник блоков surya для разбора ``page_layout``; ``None`` — собрать из ``params``
            (кэш только для чтения), а если кэша нет — разбор без surya (детектор таблиц + пятна).

    Returns:
        :class:`PageMeasure` с полем ``metrics["seconds"]``.
    """
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


def verdict_for_page(
    run_dir: Path | None,
    pdf_stem: str,
    page: int,
    geo_doc: fitz.Document,
    nogeo_doc: fitz.Document,
    thresholds: Thresholds | None = None,
    params: Params | None = None,
    surya=None,
) -> PageVerdict:
    """Вердикт «испортил ли FineReader геометрию страницы»: из кэша, при промахе — измерение.

    Args:
        run_dir: Каталог прогона с ``cache/``; ``None`` — без кэша, всегда мерить.
        pdf_stem: Имя PDF без расширения.
        page: Номер страницы, с единицы.
        geo_doc: Открытый PDF с коррекцией геометрии.
        nogeo_doc: Открытый PDF без коррекции.
        thresholds: Пороги вердикта (по умолчанию — из кода детектора).
        params: Параметры измерения (по умолчанию — для пака-1).

    Returns:
        :class:`PageVerdict`: вердикт ``bad``/``mixed``/``ok`` с метриками.
    """
    thresholds = thresholds or Thresholds()
    params = params or Params()
    path = cache_path(run_dir, pdf_stem, page) if run_dir is not None else None
    cached = load_page_cache(path) if path is not None else None
    if cached is not None:
        metrics = cached["metrics"]
        return PageVerdict(thresholds.apply(metrics), metrics, True, 0.0)
    measure = measure_page(geo_doc, nogeo_doc, page, params, surya)
    if path is not None:
        save_page_cache(path, measure)
    return PageVerdict(thresholds.apply(measure.metrics), measure.metrics, False, measure.metrics["seconds"])
