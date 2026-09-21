"""Кэш ответов surya layout на диске: JSON на страницу и вариант картинки, ключ — отпечаток источника.

ЗАЧЕМ. Surya стоит около 0.7 с GPU на страницу — часы на пак, — а нужна она всем детекторам
сразу и на каждом прогоне. Ответ кладётся на диск, и любой следующий прогон (в том числе
воркеры пула, где torch не импортируется) берёт его оттуда.

КЛЮЧ. ``<корень>/<вариант>/<имя страницы>.json``: вариант картинки (скан, заострённая,
битональная страница FineReader с коррекцией и без — surya на них отвечает по-разному) плюс имя
страницы (``1966/01/IMG_0003_2R`` у сканов, ``full_1967_01/p0079`` у страниц PDF). Внутри —
отпечаток файла-источника (размер, mtime, номер страницы): совпал — попадание, и картинку
читать незачем; не совпал или источника нет — сверяется дайджест кадра surya. Старый кэш
(pickle по ``rel_path`` без всего этого) импортируется как ``legacy`` и принимается на веру.

ПРАВИЛО ЧТЕНИЯ прежнее: файл есть и цел — берём; нет, битый, чужой или другой версии — промах,
считаем заново и ПЕРЕЗАПИСЫВАЕМ. Любое исключение при чтении — промах, а не ошибка прогона.
Запись атомарная (``.part`` → ``replace``): прерванный прогон не оставляет обрезка.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ocr_utils.page_layout.image import PageImage, SourceStat, Variant
from ocr_utils.page_layout.surya.blocks import LayoutBlocks

logger = logging.getLogger(__name__)

# Версия формата записи. Поднимать при смене того, что лежит внутри; старые записи — промах.
CACHE_VERSION = 1

# Модули, чьи классы при импорте старого pickle восстанавливаются по-настоящему. Всё остальное
# (surya, pydantic, torch) — заглушка: сырой ответ модели там не нужен, а импортировать surya
# ради него нельзя.
_TRUSTED_MODULES = ("builtins", "collections", "copyreg", "datetime", "numpy", "_codecs", "pathlib")


@dataclass(frozen=True)
class CacheEntry:
    """Одна запись кэша: блоки в пикселях кадра surya ``frame_width`` × ``frame_height`` при ``frame_dpi``."""

    variant: Variant
    cache_name: str
    blocks: LayoutBlocks
    frame_dpi: int
    source: SourceStat | None = None
    digest: str | None = None
    legacy: bool = False
    model: str = ""

    def to_json(self) -> dict:
        return {
            "cache_version": CACHE_VERSION,
            "variant": self.variant.value,
            "cache_name": self.cache_name,
            "source": self.source.to_json() if self.source is not None else None,
            "frame": {"width": self.blocks.width, "height": self.blocks.height, "dpi": self.frame_dpi},
            "digest": self.digest,
            "legacy": self.legacy,
            "model": self.model,
            "blocks": [b.to_json() for b in self.blocks.blocks],
        }

    @classmethod
    def from_json(cls, payload: dict) -> "CacheEntry":
        if int(payload.get("cache_version", -1)) != CACHE_VERSION:
            raise ValueError(f"версия записи {payload.get('cache_version')!r}, ожидалась {CACHE_VERSION}")
        frame = payload["frame"]
        blocks = LayoutBlocks.from_json(
            {"width": frame["width"], "height": frame["height"], "blocks": payload["blocks"]}
        )
        if blocks.width <= 0 or blocks.height <= 0:
            raise ValueError(f"размер кадра {blocks.width}x{blocks.height}")
        source = payload.get("source")
        return cls(
            Variant(payload["variant"]),
            str(payload["cache_name"]),
            blocks,
            int(frame.get("dpi") or 0),
            SourceStat.from_json(source) if source else None,
            payload.get("digest"),
            bool(payload.get("legacy", False)),
            str(payload.get("model", "")),
        )

    def matches(self, image: PageImage) -> bool:
        """Про эту ли картинку запись: по отпечатку источника, иначе по дайджесту кадра, иначе legacy."""
        if self.source is not None and image.source is not None:
            if (self.source.size, self.source.page_index) == (image.source.size, image.source.page_index) and abs(
                self.source.mtime - image.source.mtime
            ) < 1e-3:
                return True
        if self.digest is not None:
            return self.digest == image.surya_frame_digest()
        return self.legacy


def scan_cache_name(rel_path: str) -> str:
    """Имя полосы скана в кэше: путь внутри пака без расширения (``1966/01/IMG_0003_2R``)."""
    return Path(rel_path).with_suffix("").as_posix()


class SuryaCache:
    """Кэш на диске под корнем ``root``; ``readonly`` запрещает запись (воркеры пула)."""

    def __init__(self, root: Path, readonly: bool = False) -> None:
        self.root = Path(root)
        self.readonly = readonly

    def path(self, variant: Variant, cache_name: str) -> Path:
        return self.root / Variant(variant).value / f"{cache_name}.json"

    def has(self, variant: Variant, cache_name: str) -> bool:
        return self.path(variant, cache_name).is_file()

    def read(self, variant: Variant, cache_name: str) -> CacheEntry | None:
        """Запись как есть, без сверки с картинкой; ``None`` — нет, битая или другой версии."""
        path = self.path(variant, cache_name)
        if not path.is_file():
            return None
        try:
            entry = CacheEntry.from_json(json.loads(path.read_text(encoding="utf-8")))
            if entry.cache_name != cache_name or entry.variant != Variant(variant):
                raise ValueError(f"внутри записаны {entry.variant.value}/{entry.cache_name!r}")
            return entry
        except Exception as error:  # noqa: BLE001 — промах кэша, а не ошибка прогона
            logger.warning(
                "%s: кэш surya не читается (%s: %s), страница будет размечена заново", path, type(error).__name__, error
            )
            return None

    def blocks_of(self, variant: Variant, cache_name: str) -> LayoutBlocks | None:
        """Блоки записи как есть, без сверки с картинкой — для потребителей, у которых картинки нет."""
        entry = self.read(variant, cache_name)
        return entry.blocks if entry is not None else None

    def load(self, image: PageImage) -> LayoutBlocks | None:
        """Блоки для этой картинки (в пикселях её кадра surya) или ``None`` — промах."""
        if image.cache_name is None:
            return None
        entry = self.read(image.variant, image.cache_name)
        if entry is None or not entry.matches(image):
            return None
        # Размер кадра surya известен из заголовка: при попадании по отпечатку пиксели не читаются.
        width, height = image.size_at(image.surya_dpi)
        return entry.blocks.scaled_to(width, height)

    def save(self, image: PageImage, blocks: LayoutBlocks, model: str = "") -> Path | None:
        """Записать ответ модели для картинки; ``None`` — кэш только для чтения или имени нет."""
        if self.readonly or image.cache_name is None:
            return None
        entry = CacheEntry(
            image.variant,
            image.cache_name,
            blocks,
            image.surya_dpi,
            image.source,
            image.surya_frame_digest(),
            False,
            model,
        )
        return self.write(entry)

    def write(self, entry: CacheEntry) -> Path:
        if self.readonly:
            raise PermissionError("кэш открыт только для чтения")
        path = self.path(entry.variant, entry.cache_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_text(json.dumps(entry.to_json(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path

    def iter_names(self, variant: Variant) -> Iterator[str]:
        """Имена страниц, записанных под вариантом."""
        base = self.root / Variant(variant).value
        if not base.is_dir():
            return
        for path in sorted(base.rglob("*.json")):
            yield path.relative_to(base).with_suffix("").as_posix()


# --- Импорт старого кэша (pickle по rel_path) ----------------------------------------


class _Stub:
    """Заглушка вместо класса чужого пакета: принимает любое состояние и ничего не делает."""

    def __new__(cls, *args, **kwargs):  # noqa: ARG003 — сигнатура задаётся pickle
        return object.__new__(cls)

    def __setstate__(self, state) -> None:
        self.state = state


class _LenientUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.split(".")[0] in _TRUSTED_MODULES:
            return super().find_class(module, name)
        return _Stub


def read_legacy_pickle(path: Path) -> tuple[str, int, LayoutBlocks]:
    """``(scan_rel_path, dpi, блоки)`` из pickle старого формата (``scan_markup.detection.layout_cache``).

    Raises:
        ValueError: Файл не того формата.
    """
    with Path(path).open("rb") as handle:
        payload = _LenientUnpickler(handle).load()
    if not isinstance(payload, dict) or "layout" not in payload:
        raise ValueError(f"{path}: не словарь старого кэша")
    blocks = LayoutBlocks.from_json(payload["layout"])
    return str(payload.get("scan_rel_path", "")), int(payload.get("dpi") or 0), blocks


def import_legacy(src_dir: Path, variant: Variant, cache: SuryaCache, overwrite: bool = False) -> tuple[int, int, int]:
    """Перенести старый кэш pickle в новый как записи ``legacy`` (без отпечатка и дайджеста).

    Args:
        src_dir: Корень старого кэша: ``{год}/{выпуск}/{основа}.pkl``.
        variant: Под каким вариантом класть.
        cache: Куда класть.
        overwrite: Перезаписывать ли уже существующие записи.

    Returns:
        ``(перенесено, пропущено как уже существующие, битых)``.
    """
    done = skipped = broken = 0
    for path in sorted(Path(src_dir).rglob("*.pkl")):
        cache_name = path.relative_to(src_dir).with_suffix("").as_posix()
        if not overwrite and cache.has(variant, cache_name):
            skipped += 1
            continue
        try:
            _, dpi, blocks = read_legacy_pickle(path)
        except Exception as error:  # noqa: BLE001 — один битый файл не должен ронять импорт
            logger.warning("%s: не импортирован (%s: %s)", path, type(error).__name__, error)
            broken += 1
            continue
        cache.write(CacheEntry(variant, cache_name, blocks, dpi, None, None, True, "legacy pickle"))
        done += 1
    return done, skipped, broken


__all__ = ["CACHE_VERSION", "CacheEntry", "SuryaCache", "import_legacy", "read_legacy_pickle", "scan_cache_name"]
