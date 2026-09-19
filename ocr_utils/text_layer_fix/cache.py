"""Кэш разбора страниц: JSON на страницу (зоны, вердикты слов, чтения) и его чтение.

JSON пишет :func:`pipeline.process_page` через ``PageResult.to_json()``; читают стадия второго
мнения surya (дописывает чтения) и правка PDF (:mod:`fixer`). Запись другой ``VERSION`` или с
ошибкой считается отсутствующей — страница разбирается заново.
"""

from __future__ import annotations

import json
from pathlib import Path

from ocr_utils.text_layer_fix import VERSION


def cache_path(cache_dir: Path, pdf_name: str, index: int) -> Path:
    """Путь JSON страницы.

    Args:
        cache_dir: Корень кэша (``<прогон>/cache``).
        pdf_name: Имя файла PDF (с расширением) — подпапка кэша.
        index: Номер страницы с нуля.

    Returns:
        ``<cache_dir>/<pdf_name>/pNNNN.json``.
    """
    return Path(cache_dir) / pdf_name / f"p{index:04d}.json"


def load_page(path: Path) -> dict | None:
    """JSON страницы, если он есть, текущей версии и без ошибки разбора.

    Args:
        path: Путь JSON (:func:`cache_path`).

    Returns:
        Словарь страницы или ``None``.
    """
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != VERSION or payload.get("error"):
        return None
    return payload


def save_page(path: Path, payload: dict) -> None:
    """Записать JSON страницы, создав подпапку.

    Args:
        path: Путь JSON (:func:`cache_path`).
        payload: Словарь страницы (``PageResult.to_json()`` с любыми добавками).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
