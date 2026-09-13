"""Дисковый кэш результатов: тонкая обёртка над ``curved_lines.cache.PageCache``.

Своего хранилища нет намеренно. Схема «один JSON на картинку и алгоритм, ключ — путь плюс
размер плюс mtime плюс версия» уже отлажена в двух пакетах проекта; второй экземпляр той же
схемы разошёлся бы с первым на первой же правке.

Здесь добавлено ровно одно: в ключ версии подмешана ``TABLE_PROCESSING_VERSION``, поэтому
правка любого детектора обесценивает кэш всего пакета разом. Это грубо, зато не бывает так,
что отчёт показывает вчерашние числа для одного алгоритма и сегодняшние для другого.
"""

from __future__ import annotations

from pathlib import Path

from ocr_utils.scan_markup.curved_lines.cache import PageCache

from research.legacy.table_processing import TABLE_PROCESSING_VERSION


class ResultCache:
    def __init__(self, root: Path) -> None:
        self._cache = PageCache(Path(root))

    @property
    def root(self) -> Path:
        return self._cache.root

    @staticmethod
    def _key(version: int | str) -> str:
        return f"v{TABLE_PROCESSING_VERSION}.{version}"

    def load(self, image: Path, name: str, version: int | str = 1) -> dict | None:
        payload = self._cache.load(image, name, self._key(version))
        return payload.get("metrics") if payload else None

    def store(self, image: Path, name: str, payload: dict, version: int | str = 1) -> None:
        # PageCache считает валидной только запись с ключом "metrics" — кладём весь результат
        # туда, а не заводим свой формат: разбирать два формата в отчёте не за что.
        self._cache.store(image, name, self._key(version), {"metrics": payload})
