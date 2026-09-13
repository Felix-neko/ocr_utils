"""Распознаватели боковых ячеек: реестр и набор по умолчанию.

Импорты внутри функции: surya тянет torch, а команде, которой surya не заказана, ждать
его загрузки незачем.
"""

from __future__ import annotations

from research.legacy.table_processing.ocr.base import Engine, OcrResult, Recognizer, prepare  # noqa: F401

REGISTRY_ORDER = ("tesseract", "surya", "paddle")

# По умолчанию — все, которые доступны: команда ``ocr`` для того и есть, чтобы сравнить.
DEFAULT_SET = ("tesseract", "surya")


def load(name: str) -> Engine:
    if name == "tesseract":
        from research.legacy.table_processing.ocr.tesseract_engine import ALGORITHM
    elif name == "surya":
        from research.legacy.table_processing.ocr.surya_engine import ALGORITHM
    elif name == "paddle":
        from research.legacy.table_processing.ocr.paddle_engine import ALGORITHM
    else:
        raise KeyError(f"нет движка {name!r}; есть {', '.join(REGISTRY_ORDER)}")
    return ALGORITHM


def resolve(names: "tuple[str, ...] | None") -> list[Engine]:
    chosen = names or DEFAULT_SET
    return [engine for engine in (load(name) for name in chosen) if engine.available()]
