"""Surya layout для стенда: блоки полосы с видом и их кэш на диске.

ТИПЫ И РАЗБОР ОТВЕТА ПЕРЕЕХАЛИ в ``ocr_utils.page_layout.surya.blocks``, кэш — в
``ocr_utils.page_layout.surya.cache`` (JSON по варианту картинки; старые pickle читаются, а
набивать кэш штатно — ``page_layout prefill-surya``, команда ``layout-pack`` устарела). Здесь —
реэкспорт, обёртки кэша со старой сигнатурой и ``Predictor`` для ``layout-pack``.

ЗАЧЕМ. Детектор по линейкам точен в геометрии, но не знает, ЧТО обвёл: кусок блок-схемы
для него — маленькая таблица. Surya смотрит на полосу целиком и отвечает на другой вопрос —
где здесь таблица, где рисунок, где текст. Эксперимент на 190 размеченных полосах: все
двенадцать «кусков блок-схемы» и все диаграммы получили целый Figure/Form-блок, на
перекошенных таблицах Table-блок шире нашей рамки ровно там, где нам не хватало графы, на
«тексте рядом» он кончается на 15–60 мм выше. Слабости тоже замерены: 10 из 38 полос «не
таблицы» помечены Table, бланк целиком бывает одним Form без таблицы внутри, границы ±1–2 мм.
Поэтому surya — источник ВИДА и подсказки протяжённости, а не точных границ.

ЦЕНА. Около секунды на полосу на GPU в 150 dpi; по паку это три с половиной часа. Отсюда
кэш: JSON на полосу в ``<каталог>/{год}/{выпуск}/{основа}.json``, и два режима команды
``layout-pack``: все полосы или только полосы-кандидаты, где детектор по линейкам что-то нашёл.

GPU ТОЛЬКО В РОДИТЕЛЕ. Модель грузится лениво, по первому вызову ``predict``, и в пул процессов
не заворачивается (правило проекта: видеопамять одна на всех).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ocr_utils.page_layout.image import Variant
from ocr_utils.page_layout.surya.cache import CacheEntry, SuryaCache, read_legacy_pickle, scan_cache_name
from ocr_utils.page_layout.surya.model import BATCH  # noqa: F401 — реэкспорт для стенда
from ocr_utils.page_layout.surya.blocks import (  # noqa: F401 — реэкспорт для стенда
    FIGURE_LABELS,
    FORM_LABELS,
    TABLE_LABELS,
    TEXT_LABELS,
    MIN_CONFIDENCE,
    Block,
    LayoutBlocks,
    from_surya_result,
)


def cache_path(cache_dir: Path, scan_rel_path: str) -> Path:
    """Файл старого кэша полосы (pickle): та же структура подпапок, что у копий полос."""
    return Path(cache_dir) / Path(scan_rel_path).with_suffix(".pkl")


def load(cache_dir: "Path | None", scan_rel_path: str) -> "LayoutBlocks | None":
    """Разметка полосы из кэша или ``None``, если её там нет (тогда детектор идёт без неё).

    Понимает оба формата: старый pickle (``<dir>/<rel>.pkl``) и новый JSON ``page_layout``
    (``<dir>`` — папка варианта внутри корня кэша, например ``pack1_page_layout/sharpened``).
    """
    if cache_dir is None:
        return None
    cache_dir = Path(cache_dir)
    pickled = cache_path(cache_dir, scan_rel_path)
    if pickled.is_file():
        try:
            return read_legacy_pickle(pickled)[2]
        except Exception:  # noqa: BLE001 — битый файл = промах
            return None
    try:
        variant = Variant(cache_dir.name)
    except ValueError:
        return None
    return SuryaCache(cache_dir.parent, readonly=True).blocks_of(variant, scan_cache_name(scan_rel_path))


def save(cache_dir: Path, scan_rel_path: str, layout: LayoutBlocks, raw: object = None, dpi: int = 0) -> Path:
    """Сохранить разметку полосы в НОВЫЙ кэш (``<dir>`` — папка варианта корня ``page_layout``).

    Сырой ответ surya больше не хранится: набивать кэш штатно — ``page_layout prefill-surya``.
    """
    cache_dir = Path(cache_dir)
    variant = Variant(cache_dir.name)
    entry = CacheEntry(variant, scan_cache_name(scan_rel_path), layout, dpi, None, None, True, "legacy layout-pack")
    return SuryaCache(cache_dir.parent).write(entry)


class Predictor:
    """Ленивая обёртка над ``surya.layout.LayoutPredictor``: модель грузится при первом вызове."""

    def __init__(self) -> None:
        self._predictor = None

    def _load(self):
        if self._predictor is None:
            from surya.foundation import FoundationPredictor
            from surya.layout import LayoutPredictor
            from surya.settings import settings

            self._predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
        return self._predictor

    def predict(self, grays: Sequence[np.ndarray]) -> list[LayoutBlocks]:
        """Разметка серых полос; координаты — в пикселях поданных картинок."""
        return [layout for layout, _ in self.predict_raw(grays)]

    def predict_raw(self, grays: Sequence[np.ndarray]) -> list[tuple[LayoutBlocks, object]]:
        """То же, плюс сырой ответ surya на каждую полосу — для кэша."""
        from PIL import Image

        predictor = self._load()
        results: list[tuple[LayoutBlocks, object]] = []
        for start in range(0, len(grays), BATCH):
            chunk = grays[start : start + BATCH]
            images = [Image.fromarray(gray).convert("RGB") for gray in chunk]
            for gray, result in zip(chunk, predictor(images)):
                height, width = gray.shape[:2]
                results.append((from_surya_result(result, width, height), result))
        return results


__all__ = ["Block", "LayoutBlocks", "Predictor", "cache_path", "load", "save"]
