"""Ось текста в ячейке по строкам, найденным детектором Surya.

НЕЙРОСЕТЕВОЙ АНАЛОГ трёх пиксельных мер: строки ищет обученная сеть, а ось считается по
форме найденных рамок — вытянутых поперёк таблицы или вдоль неё. Мера та же, что у
``orientation.detectors.surya_lines`` (там она замерена на полосах и разделяет чисто),
и функция счёта оттуда же переиспользуется.

ДЕТЕКЦИЯ ИДЁТ ПО ТАБЛИЦЕ ЦЕЛИКОМ, а не по каждой ячейке отдельно. Причина в том, как
устроен детектор строк: он ищет строки в контексте страницы, и на вырезке 100x230 px без
соседей путается. Найденные рамки потом раскладываются по ячейкам по центру рамки.

ЧЕГО НЕ ДАЁТ. Стороны: строка остаётся строкой и вверх ногами. Детектор всегда ``axis_only``.

В поле ``vertical_lines`` заглядывать бесполезно — в surya 0.17 его нет вовсе
(``surya/detection/schema.py``: только ``bboxes``, ``heatmap``, ``affinity_map``), ось
считается по форме рамок.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ocr_utils.scan_markup.orientation.detectors.surya_lines import (
    AXIS_MARGIN_THR,
    shared_detection_predictor,
    surya_available,
)

from research.legacy.table_processing.geometry import Box
from research.legacy.table_processing.rotation.base import BatchCellDetector, CellCrop, CellDetector, Verdict, unknown

# Меньше этого числа рамок внутри ячейки — судить не по чему.
MIN_BOXES = 1

# Максимальная сторона таблицы, подаваемой в сеть. Больше не нужно: детектор всё равно
# ресайзит вход под свой процессор, а на вырезке 3000x1300 одна только конвертация в RGB
# стоит дольше самой детекции.
WORK_SIDE = 1600


class SuryaCellAxis(BatchCellDetector):
    def __init__(self) -> None:
        self._predictor = None
        self._cache: dict[int, list[tuple[Box, float]]] = {}

    def _load(self):
        if self._predictor is None:
            self._predictor = shared_detection_predictor()
        return self._predictor

    def _boxes(self, table: np.ndarray) -> list[tuple[Box, float]]:
        """Рамки строк таблицы в её собственных координатах; считается один раз на таблицу."""
        key = id(table)
        if key in self._cache:
            return self._cache[key]

        import cv2
        from PIL import Image

        height, width = table.shape[:2]
        scale = min(1.0, WORK_SIDE / max(height, width))
        small = cv2.resize(table, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else table
        rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
        try:
            result = self._load()([Image.fromarray(rgb)])[0]
        except Exception:
            self._cache[key] = []
            return []
        found: list[tuple[Box, float]] = []
        for line in result.bboxes:
            x0, y0, x1, y1 = (value / scale for value in line.bbox)
            box = Box(int(x0), int(y0), int(x1), int(y1))
            found.append((box, float(box.width) / max(1.0, float(box.height))))
        self._cache[key] = found
        return found

    def __call__(self, crops: Sequence[CellCrop]) -> list[Verdict]:
        verdicts: list[Verdict] = []
        for crop in crops:
            if crop.table is None:
                verdicts.append(unknown("нет таблицы целиком, детектор строк её требует"))
                continue
            wide = tall = 0.0
            count = 0
            for box, ratio in self._boxes(crop.table):
                center_x, center_y = (box.x0 + box.x1) // 2, (box.y0 + box.y1) // 2
                if not (crop.cell.box.x0 <= center_x <= crop.cell.box.x1):
                    continue
                if not (crop.cell.box.y0 <= center_y <= crop.cell.box.y1):
                    continue
                count += 1
                if ratio >= 1.0:
                    wide += box.area
                else:
                    tall += box.area
            if count < MIN_BOXES or wide + tall <= 0.0:
                verdicts.append(unknown("строк в ячейке не нашлось"))
                continue
            axis_score = (wide - tall) / (wide + tall)
            metrics = {"axis_score": axis_score, "lines": float(count)}
            if abs(axis_score) < AXIS_MARGIN_THR:
                verdicts.append(Verdict(0, 0.0, metrics=metrics, note="ось не различается"))
                continue
            verdicts.append(
                Verdict(0 if axis_score > 0 else 90, min(1.0, abs(axis_score)), axis_only=True, metrics=metrics)
            )
        return verdicts


ALGORITHM = CellDetector(
    name="surya_lines",
    summary="ось по форме строк, найденных детектором Surya (GPU)",
    stage="gpu",
    gives_sign=False,
    make_batch=SuryaCellAxis,
    available=surya_available,
)
