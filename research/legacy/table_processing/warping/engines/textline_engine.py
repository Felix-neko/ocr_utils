"""Готовый выпрямитель проекта по СТРОКАМ ТЕКСТА, применённый к вырезке таблицы.

ЗАЧЕМ ОН ЗДЕСЬ. Он ведёт не по линейкам, а по центрлиниям строк, и потому работает там, где
линеек нет вовсе — а такие таблицы в паке есть. По собственному замеру проекта
(``reports/dewarp_report.md``) это единственный из шести движков, который на плоских
журнальных сканах улучшает, а не портит: медианное отношение сагитты p90 после/до 0.41,
улучшил 96% страниц из 580.

ОГРАНИЧЕНИЕ, записанное там же: движок исправляет только ВЕРТИКАЛЬНОЕ смещение и не трогает
ни поворот, ни горизонтальную составляющую.
"""

from __future__ import annotations

import cv2
import numpy as np

from research.legacy.table_processing.warping.engines.base import Warped, Warper


def available() -> bool:
    try:
        from ocr_utils.dewarp.engines.textline import measure_field  # noqa: F401
    except Exception:
        return False
    return True


def run(gray: np.ndarray, dpi: int) -> Warped:
    from ocr_utils.dewarp.engines.textline import SEG_DPI, apply_field, measure_field

    height, width = gray.shape[:2]
    scale = SEG_DPI / dpi
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    fine_scale = 2 * SEG_DPI / dpi
    fine = (
        cv2.resize(gray, None, fx=fine_scale, fy=fine_scale, interpolation=cv2.INTER_AREA) if fine_scale < 1.0 else gray
    )

    try:
        field = measure_field(small, fine)
    except Exception as error:
        return Warped(gray, note=f"строки не измерились: {error}")
    if field is None:
        return Warped(gray, note="строк для измерения не хватило")

    fixed = apply_field(gray, field, dpi / SEG_DPI)
    if fixed.shape[:2] != (height, width):
        fixed = cv2.resize(fixed, (width, height), interpolation=cv2.INTER_CUBIC)
    return Warped(fixed, None, note=f"по {field.lines} строкам", changed=True)


ALGORITHM = Warper(
    name="textline", summary="готовый движок проекта: поле по центрлиниям строк", run=run, available=available
)
