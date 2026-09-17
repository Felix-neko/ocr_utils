"""Реестр движков выпрямления страниц.

Фабрики ленивые: модуль движка (и его тяжёлые импорты, клон репозитория) подтягивается
только когда движок реально выбран. Порядок задаёт последовательность для ``all``:
сперва классика на CPU, потом нейросети на GPU.
"""

from __future__ import annotations

import importlib
from typing import Callable

from ocr_utils.dewarp.engines.base import DewarpEngine


def _factory(module: str, cls: str) -> Callable[[], DewarpEngine]:
    def make() -> DewarpEngine:
        mod = importlib.import_module(f"ocr_utils.dewarp.engines.{module}")
        return getattr(mod, cls)()

    return make


ENGINES: dict[str, Callable[[], DewarpEngine]] = {
    "textline": _factory("textline", "TextLineEngine"),
    "pagedewarp": _factory("pagedewarp", "PageDewarpEngine"),
    "docscanner": _factory("docscanner", "DocScannerEngine"),
    "uvdoc": _factory("uvdoc", "UVDocEngine"),
    "doctr": _factory("doctr", "DocTrEngine"),
    "doctr_plus": _factory("doctr_plus", "DocTrPlusEngine"),
    "dewarpnet": _factory("dewarpnet", "DewarpNetEngine"),
}

# Движки, которые считают на CPU и потому едут в пул процессов; остальные — GPU, только
# в родителе (видеопамять одна на всех, CLAUDE.md).
CPU_ENGINES = ("textline", "pagedewarp")


def get_engine(name: str) -> DewarpEngine:
    """Создаёт экземпляр движка по имени."""
    if name not in ENGINES:
        raise KeyError(f"Неизвестный движок: {name}. Доступны: {', '.join(ENGINES)}")
    return ENGINES[name]()
