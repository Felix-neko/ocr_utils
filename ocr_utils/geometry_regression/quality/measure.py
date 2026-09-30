"""Мера страницы v17–v18: метрики v16 из кэша плюс качество строк и краёв блоков по разбору обоих вариантов, меры по плотному полю (line art, формулы, строки без пары в A); кэш стенда JSON на страницу."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ocr_utils.geometry_regression.quality import ENGINE_VERSION
from ocr_utils.geometry_regression.quality.edges import EdgePair, edge_metrics, match_edges
from ocr_utils.geometry_regression.quality.lines import LinePair, line_metrics, match_lines, transfer_lines
from ocr_utils.geometry_regression.quality.sources import PageRef, load_pair, object_boxes

# Классы объектов разбора, строки в которых — только выигрыш (таблицы, рисунки, формулы, неясное).
OBJECT_CLASSES = {"таблица", "рисунок", "формула", "неясно"}
# Рамки формул разбора B (surya Equation и DeepSeek): своя группа мер по плотному полю (``formula_*``).
FORMULA_CLASSES = {"формула"}
# Классы растра в разборе B: строки в них не переносятся полем (``lines.transfer_lines``).
RASTER_CLASSES = {"цветной_растр", "серый_растр"}
# Классы line art в разборе B: разошедшиеся параллели внутри них — непрощаемая порча рисунка.
LINEART_CLASSES = {"рисунок", "неясно"}


def _centre_inside(box: list[float], boxes: list) -> bool:
    """Центр рамки внутри одной из рамок."""
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in boxes)


def parallel_in_lineart(metrics: dict, culprits: dict, lineart_boxes: list) -> dict[str, float]:
    """Разбить разброс параллельных линий v16 по месту: внутри line art или вне его.

    Параллели схемы, графика, номограммы, переставшие быть параллельными, — порча рисунка, которую выигрыш не
    прощает (решение пользователя 2026-09-29: 1967/10 с.74 — перекорёженная оргсхема, 1968/05 с.55 — сетка
    графика). Разошедшиеся линейки таблиц и дробные черты формул (1968/05 с.72: «формулы покорёжило незначительно»,
    эталон good) остаются обычной порчей линеек. Место — центр рамки худшей группы параллельных в B.

    Args:
        metrics: Метрики v16 страницы.
        culprits: Виновники v16.
        lineart_boxes: Рамки line art на B (150 dpi): v16 и классы ``LINEART_CLASSES`` разбора v6.

    Returns:
        ``{"parallel_spread_lineart": …, "parallel_spread_other": …}`` — одна из двух равна метрике v16, другая ноль.
    """
    value = float(metrics.get("parallel_spread_delta_max", 0.0) or 0.0)
    where = (culprits.get("parallel_spread_delta_max") or {}).get("b")
    inside = bool(where) and _centre_inside(where, lineart_boxes)
    return {"parallel_spread_lineart": value if inside else 0.0, "parallel_spread_other": 0.0 if inside else value}


def _line_json(pair: LinePair) -> dict:
    """Пара строк для кэша (оверлей подписывает у строки её меры)."""
    return {
        "b": list(pair.box_b), "a": list(pair.box_a), "q_b": round(pair.q_b, 3), "q_a": round(pair.q_a, 3),
        "w": round(pair.weight, 3), "len": round(pair.length_mm, 1), "in_obj": pair.inside_objects,
        "tr": pair.transferred,
    }  # fmt: skip


def _edge_json(pair: EdgePair) -> dict:
    """Пара сторон для кэша: линии на общем отрезке и меры."""
    return {
        "side": pair.side.name.lower(), "rows": pair.rows, "e_b": round(pair.e_b, 3), "e_a": round(pair.e_a, 3),
        "raw_b": round(pair.e_raw_b, 3), "raw_a": round(pair.e_raw_a, 3), "w": round(pair.weight, 3),
        "len": round(pair.length_mm, 1), "patched": round(pair.patched_b, 3),
        "points_b": pair.points_b.round(1).tolist(), "points_a": pair.points_a.round(1).tolist(),
    }  # fmt: skip


def measure_page(ref: PageRef, layout_root: Path, v16_dir: Path, pdf_dirs: tuple[Path, Path] | None = None) -> dict:
    """Все метрики страницы v17.

    Args:
        ref: Страница.
        layout_root: Корень разбора v6 (``geo``/``nogeo``).
        v16_dir: Каталог прогона v16 (кэш метрик и поля).
        pdf_dirs: ``(PDF с коррекцией, PDF без коррекции)`` — для мер по плотному полю (:mod:`lineart_flow`):
            line art, формулы и строки без пары в A; ``None`` — эти меры не считаются.

    Returns:
        ``{"version", "pdf", "page", "metrics", "culprits", "lines", "edges", "raw"}``: метрики — плоский словарь
        (v16 как есть плюс новые), линии и стороны — пары для оверлея, ``raw`` — рамки объектов B.
    """
    started = time.time()
    pair = load_pair(ref, layout_root, v16_dir)
    objects_b = object_boxes(pair.b, OBJECT_CLASSES)
    lines = match_lines(pair.b, pair.a, pair.field, objects_b)
    edges = match_edges(pair.b, pair.a, pair.field)
    metrics = dict(pair.v16)
    lineart = [list(box) for box in pair.v16_raw.get("lineart", [])] + object_boxes(pair.b, LINEART_CLASSES)
    metrics.update(parallel_in_lineart(pair.v16, pair.v16_culprits, lineart))
    culprits: dict = {}
    if "parallel_spread_delta_max" in pair.v16_culprits:
        culprits["parallel_spread_delta_max"] = {
            k: v for k, v in pair.v16_culprits["parallel_spread_delta_max"].items() if k in ("a", "b")
        }
    formulas = object_boxes(pair.b, FORMULA_CLASSES)
    if pdf_dirs is not None:
        # Рендеры обоих PDF и выравнивание страницы — один раз: для строк без пары в A (перенос оси полем) и для
        # мер line art и формул по плотному полю.
        flow_results, moved = _flow_metrics(ref, pdf_dirs, pair, lines, objects_b, lineart, formulas)
        lines = lines + moved
        for values, where in flow_results:
            metrics.update(values)
            culprits.update(where)
    for values, where in (line_metrics(lines), edge_metrics(edges)):
        metrics.update(values)
        culprits.update(where)
    metrics["v17_seconds"] = round(time.time() - started, 3)
    return {
        "version": ENGINE_VERSION,
        "pdf": ref.pdf,
        "page": ref.page,
        "metrics": metrics,
        "culprits": culprits,
        "lines": [_line_json(p) for p in lines],
        "edges": [_edge_json(p) for p in edges],
        "raw": {"objects_b": [[round(v, 1) for v in box] for box in objects_b], "v16": pair.v16_raw},
    }


def _flow_metrics(
    ref: PageRef,
    pdf_dirs: tuple[Path, Path],
    pair,
    lines: list[LinePair],
    objects_b: list,
    lineart: list,
    formulas: list,
) -> tuple[list[tuple[dict, dict]], list[LinePair]]:
    """Меры по плотному полю: строки без пары в A, рамки line art и рамки формул; рендеры 150 dpi и выравнивание — один раз.

    Args:
        ref: Страница.
        pdf_dirs: ``(PDF с коррекцией, PDF без коррекции)``.
        pair: Пара разборов страницы (:func:`sources.load_pair`): разбор B и поле v16.
        lines: Пары строк, уже найденные по разбору A.
        objects_b: Рамки таблиц, рисунков и формул на B.
        lineart: Рамки line art на B.
        formulas: Рамки формул на B.

    Returns:
        ``(пары (метрики, виновники) line art и формул, пары строк, перенесённых полем)``. У пустого набора рамок —
        нули (метрики есть на каждой странице с рамками хоть одного вида); без рамок — пустой список.
    """
    import fitz

    from ocr_utils.geometry_regression.render import render_gray, to_work
    from ocr_utils.geometry_regression.quality.lineart_flow import (
        FORMULA_MIN_LONG_MM,
        FORMULA_MIN_SIDE_MM,
        lineart_flow_metrics,
        page_alignment,
    )

    geo_dir, nogeo_dir = pdf_dirs
    with fitz.open(str(nogeo_dir / f"{ref.pdf}.pdf")) as nogeo, fitz.open(str(geo_dir / f"{ref.pdf}.pdf")) as geo:
        gray_b, gray_a = to_work(render_gray(nogeo, ref.page - 1)), to_work(render_gray(geo, ref.page - 1))
    alignment = page_alignment(gray_b, gray_a, pair.field)
    aligned, _, matrix = alignment
    # Растр — только по разбору v6: v16 принимал штриховой рисунок с подписью за растр (1972/10 с.79), и подпись,
    # наклонённую FineReader, не мерили бы.
    raster = object_boxes(pair.b, RASTER_CLASSES)
    moved = transfer_lines(pair.b, lines, gray_b, aligned, matrix, objects_b, raster)
    results = []
    if lineart or formulas:
        results = [
            lineart_flow_metrics(gray_b, gray_a, pair.field, lineart, prefix="lineart", alignment=alignment),
            lineart_flow_metrics(gray_b, gray_a, pair.field, formulas, prefix="formula", min_side_mm=FORMULA_MIN_SIDE_MM,
                                 min_long_mm=FORMULA_MIN_LONG_MM, alignment=alignment),  # fmt: skip
        ]
    return results, moved


def cache_path(run_dir: Path, ref: PageRef) -> Path:
    """Файл кэша стенда: ``<run>/cache/<pdf>/pNNN.json`` (номер с единицы)."""
    return run_dir / "cache" / ref.pdf / f"p{ref.page:03d}.json"


def load_cached(run_dir: Path, ref: PageRef) -> dict | None:
    """Кэш страницы той же версии или ``None``."""
    path = cache_path(run_dir, ref)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("version") == ENGINE_VERSION else None


def measure_cached(
    ref: PageRef,
    layout_root: Path,
    v16_dir: Path,
    run_dir: Path,
    redo: bool = False,
    pdf_dirs: tuple[Path, Path] | None = None,
) -> dict:
    """Мера страницы из кэша стенда или заново (с записью в кэш).

    Args:
        ref: Страница.
        layout_root: Корень разбора v6.
        v16_dir: Каталог прогона v16.
        run_dir: Каталог прогона стенда.
        redo: Пересчитать, даже если кэш есть.
        pdf_dirs: PDF обоих вариантов для мер line art по плотному полю (см. :func:`measure_page`).

    Returns:
        Словарь :func:`measure_page`; при ошибке — ``{"pdf", "page", "error"}``.
    """
    if not redo:
        cached = load_cached(run_dir, ref)
        if cached is not None:
            return cached
    try:
        payload = measure_page(ref, layout_root, v16_dir, pdf_dirs)
    except Exception as error:  # noqa: BLE001 — одна битая страница не валит прогон
        return {"pdf": ref.pdf, "page": ref.page, "error": f"{type(error).__name__}: {error}"}
    path = cache_path(run_dir, ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


__all__ = [
    "LINEART_CLASSES",
    "OBJECT_CLASSES",
    "parallel_in_lineart",
    "cache_path",
    "load_cached",
    "measure_cached",
    "measure_page",
]
