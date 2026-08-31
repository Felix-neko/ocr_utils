"""Проверка «починен ли дефект» по найденным на полосе областям."""

from ocr_utils.scan_markup.db.models import KIND_GRAYSCALE
from ocr_utils.scan_markup.validation.cases import (
    COLOR_ON_GRAY,
    DOT_LEADERS,
    DOT_LEADERS_TABLE,
    FALSE_POSITIVE,
    LINEART,
    MERGED,
    SPLIT,
)


def expectation_holds(defect_key: str, regions) -> bool:
    """Выполнено ли ожидание по этому типу дефекта."""
    if defect_key == COLOR_ON_GRAY:
        return bool(regions) and all(region.kind == KIND_GRAYSCALE for region in regions)
    if defect_key in (LINEART, FALSE_POSITIVE, DOT_LEADERS, DOT_LEADERS_TABLE):
        return not regions
    if defect_key == MERGED:
        return len(regions) >= 2
    if defect_key == SPLIT:
        return len(regions) == 1
    return True  # дефекты вне зачёта проверять нечем


def describe(defect_key: str, regions) -> str:
    """Человекочитаемое «что получилось» — то, что печатается против непочиненной полосы."""
    if not regions:
        return "областей нет"
    kinds = ", ".join(sorted({region.kind for region in regions}))
    if defect_key == COLOR_ON_GRAY:
        return f"областей {len(regions)}, типы: {kinds}"
    return f"областей {len(regions)} ({kinds})"
