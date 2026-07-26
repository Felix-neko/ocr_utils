"""Реестр движков выпрямления страниц.

Фабрики ленивые: модуль движка (и его тяжёлые импорты/клон репозитория) подтягивается
только когда движок реально выбран. Порядок задаёт последовательность для ``--method all``.
"""

import importlib
from typing import Callable

from ocr_utils.legacy.dewarp.engines.base import DewarpEngine


def _factory(module: str, cls: str) -> Callable[[], DewarpEngine]:
    def make() -> DewarpEngine:
        mod = importlib.import_module(f"ocr_utils.legacy.dewarp.engines.{module}")
        return getattr(mod, cls)()

    return make


# Порядок важен для --method all (основной движок — первым)
ENGINES: dict[str, Callable[[], DewarpEngine]] = {
    "docscanner": _factory("docscanner", "DocScannerEngine"),
    "doctr": _factory("doctr", "DocTrEngine"),
    "doctr_plus": _factory("doctr_plus", "DocTrPlusEngine"),
    "uvdoc": _factory("uvdoc", "UVDocEngine"),
    "dewarpnet": _factory("dewarpnet", "DewarpNetEngine"),
    "pagedewarp": _factory("pagedewarp", "PageDewarpEngine"),
}


def get_engine(name: str) -> DewarpEngine:
    """Создаёт экземпляр движка по имени."""
    if name not in ENGINES:
        raise KeyError(f"Неизвестный движок: {name}. Доступны: {', '.join(ENGINES)}")
    return ENGINES[name]()
