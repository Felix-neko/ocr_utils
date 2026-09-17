"""Дисковый кэш измерений: один JSON на полосу и детектор.

ЗАЧЕМ. Метрики калибруются итеративно: прогон по паку идёт часами (surya), а порог
меняется за секунду. Кэш хранит РЕЗУЛЬТАТ детектора (метрики и сырые измерения для
оверлея), а не картинку, поэтому повторный ``run`` с другими порогами не трогает ни диск
с полосами, ни GPU, и по паку проходит за минуты.

Ключ — путь к файлу, его размер и mtime плюс ключ версии детектора: правка файла или
алгоритма делают старую запись невидимой, чистить руками не нужно. Схема — та же, что
у ``defocus_detection.lines.detect.DetectCache``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PageCache:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    @staticmethod
    def _digest(image: Path) -> str:
        try:
            stat = image.stat()
            stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            stamp = "?"
        return hashlib.sha1(f"{image.resolve()}|{stamp}".encode("utf-8")).hexdigest()

    def path_for(self, image: Path, name: str, key: str) -> Path:
        """Файл кэша для полосы и детектора; двухсимвольный подкаталог — чтобы в одной папке
        не копились десятки тысяч файлов."""
        digest = self._digest(image)
        return self._root / name / key / digest[:2] / f"{digest}.json"

    def has(self, image: Path, name: str, key: str) -> bool:
        return self.path_for(image, name, key).is_file()

    def load(self, image: Path, name: str, key: str) -> dict | None:
        """Сохранённая запись либо None: битая или отсутствующая запись — повод посчитать
        заново, а не падать."""
        path = self.path_for(image, name, key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) and "metrics" in payload else None

    def store(self, image: Path, name: str, key: str, payload: dict) -> None:
        path = self.path_for(image, name, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Через временный файл: прерванный прогон не должен оставить обрубок, который
            # потом прочитается как валидный кэш.
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            temporary.replace(path)
        except OSError as error:
            logger.warning("Не удалось записать кэш %s: %s", path, error)
