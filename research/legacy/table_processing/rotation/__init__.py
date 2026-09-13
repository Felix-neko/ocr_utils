"""Детекторы поворота ячейки: реестр и набор по умолчанию.

Импорты внутри функций, а не наверху модуля: GPU-детекторы тянут за собой torch и surya,
а команда, которой они не заказаны, не должна ждать их загрузки.
"""

from __future__ import annotations

from research.legacy.table_processing.rotation.base import CellCrop, CellDetector, Verdict, unknown  # noqa: F401

# Порядок — как в отчёте: сперва дешёвая классика, потом сеть, потом арбитр.
REGISTRY_ORDER = ("glyph_aspect", "ink_axis", "profile", "surya_lines", "doctr", "ocr_vote")

# Набор по умолчанию: победитель по оси, второй по оси для контроля и арбитр по стороне.
# Замер, на котором стоит выбор, — в README пакета.
DEFAULT_SET = ("glyph_aspect", "ink_axis", "ocr_vote")


def load(name: str) -> CellDetector:
    if name == "glyph_aspect":
        from research.legacy.table_processing.rotation.glyph_aspect import ALGORITHM
    elif name == "ink_axis":
        from research.legacy.table_processing.rotation.ink_axis_cell import ALGORITHM
    elif name == "profile":
        from research.legacy.table_processing.rotation.profile_cell import ALGORITHM
    elif name == "surya_lines":
        from research.legacy.table_processing.rotation.surya_axis_cell import ALGORITHM
    elif name == "doctr":
        from research.legacy.table_processing.rotation.doctr_crop import ALGORITHM
    elif name == "ocr_vote":
        from research.legacy.table_processing.rotation.ocr_vote_cell import ALGORITHM
    else:
        raise KeyError(f"нет детектора {name!r}; есть {', '.join(REGISTRY_ORDER)}")
    return ALGORITHM


def resolve(names: "tuple[str, ...] | None") -> list[CellDetector]:
    """Детекторы по именам; недоступные (нет весов, нет tesseract) отбрасываются."""
    chosen = names or DEFAULT_SET
    return [detector for detector in (load(name) for name in chosen) if detector.available()]
