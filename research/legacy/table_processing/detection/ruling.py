"""Таблица по линейкам: морфология вместо сети (первая версия детектора).

ЖИВОЙ КОД ПЕРЕЕХАЛ в ``ocr_utils.page_layout.tables.ruling``: здесь остался реэкспорт, чтобы стенд исследований
(сравнение версий, добыча, отчёты) мерил тот же детектор, что стоит в конвейере.
Здесь остался только ``detect`` первой и второй версий и его ``ALGORITHM``.

ЗАЧЕМ ИМЕННО ЭТО ПЕРВЫМ. Таблицы в этих журналах ЛИНОВАНЫ: колонки разделены вертикальными
линейками, шапка отбита двойной горизонтальной, а вот внешней рамки слева и справа часто
нет вовсе. Значит, самый прямой признак таблицы — не «сеть узнала таблицу», а собственно
линейки, и они же сразу дают сетку ячеек, за которой иначе пришлось бы идти ко второй
модели. Это тот же приём, на котором стоит img2table, и он не требует ни GPU, ни весов.

КАК. Бинаризация Otsu, затем открытие длинным горизонтальным и длинным вертикальным ядром —
остаётся только то, что тянется на много миллиметров подряд, то есть линейки. Дальше
компоненты связности дают отрезки, а объединение отрезков, стоящих рядом, — таблицу.

МАСШТАБ. Всё считается на копии 150 dpi. Замер на трёх известных полосах: разжатие через
``Image.draft`` занимает 0.1 с, линейки при ядре 40-60 px (7-10 мм) находятся все, число
компонент устойчиво (страница 28: 9 горизонтальных и 9 вертикальных при ядрах 40 и 60).
На 600 dpi та же морфология стоила бы шестнадцатикратно дороже и не дала бы ничего нового:
рамку таблицы не нужно знать точнее полумиллиметра, её всё равно берут с полями.

РАЗМЕРЫ — В МИЛЛИМЕТРАХ, а не в долях кадра. Полосы пака различаются по размеру (после
распрямления от 3589x6445 до 3830x5892 px), и доля кадра означала бы разный физический
порог на разных полосах. Пересчёт в пиксели — единственное место, где живёт dpi.
"""

from __future__ import annotations

import logging

import numpy as np

from ocr_utils.page_layout.tables.ruling import (  # noqa: F401 — реэкспорт для стенда
    WORK_DPI,
    MIN_RULE_MM,
    MAX_RULE_THICKNESS_MM,
    CLUSTER_GAP_MM,
    MIN_TABLE_SIDE_MM,
    MIN_HORIZONTAL,
    MIN_VERTICAL,
    LONG_RULE_SPAN,
    MIN_LONG_RULES,
    MAX_INK_SHARE,
    CURVED_SPREAD_DEG,
    mm_to_px,
    Segment,
    Lines,
    binarize,
    _axis_mask,
    _segments,
    _angle_of,
    find_lines,
    skew_of,
    ClusterPolicy,
    DEFAULT_POLICY,
    Cluster,
    _cluster_box,
    long_rules_of,
    metrics_of,
    cluster_tables,
    merge_clusters,
    _center_inside,
)
from research.legacy.table_processing.detection.base import TableDetector
from research.legacy.table_processing.geometry import TableBox

logger = logging.getLogger(__name__)


def detect(gray: np.ndarray, dpi: int = WORK_DPI, verify_findings: bool = True) -> list[TableBox]:
    """Таблицы на серой копии; координаты — в пикселях поданного изображения.

    ``verify_findings`` — вторая ступень: каждая находка проверяется на то, таблица ли это
    вообще (``detection.verify``). Линейки бывают не только у таблиц, и без проверки в
    находки попадают рамки объявлений, чертежи и координатные сетки — замерено на 27
    находках, размеченных глазами: без проверки 12 из 27 оказались не таблицами.

    Импорт ``verify`` отложен внутрь функции: он тянет разбор сетки, который сам импортирует
    этот модуль.
    """
    found = cluster_tables(find_lines(gray, dpi), dpi, binarize(gray))
    if not verify_findings or not found:
        return found

    from research.legacy.table_processing.detection import verify as verification

    height, width = gray.shape[:2]
    kept: list[TableBox] = []
    for table in found:
        box = table.box.clipped(width, height)
        if box.width < 8 or box.height < 8:
            continue
        crop = gray[box.slice]
        signs = verification.features(crop, find_lines(crop, dpi), dpi)
        good, reason = verification.is_table(signs)
        metrics = {**table.metrics, **{k: float(v) for k, v in signs.as_row().items()}}
        metrics["verified"] = float(good)
        if good:
            kept.append(TableBox(table.box, table.score, table.source, table.origin, table.skew_deg, metrics))
        else:
            logger.debug("находка %s отклонена: %s", table.box.as_tuple(), reason)
    return kept


ALGORITHM = TableDetector(
    name="ruling", summary="классика: морфология по линейкам, кластеры отрезков (CPU)", stage="cpu", run=detect
)
