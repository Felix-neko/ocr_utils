"""Лист стенда зоны обрезки: страница с рамками всех способов (без заливки), строками-сором и закрасом пальцев; шапка с мерами и легенда — в полях листа."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from ocr_utils.scan_markup.content_box.content import Box, Item
from ocr_utils.scan_markup.content_box.methods import Method, TrimMeasures

# Высота страницы на листе, px.
SHEET_HEIGHT = 1600
# Цвета рамок способов (BGR) и толщина, px листа.
COLOUR_OF = {
    Method.FIXED: (170, 170, 170),  # серый: справочный способ
    Method.CONTENT: (220, 90, 20),  # синий
    Method.INK: (40, 170, 40),  # зелёный
    Method.HYBRID: (200, 60, 200),  # фиолетовый
}
BOX_PX = 2
COLOUR_STRAY = (0, 0, 220)  # строка-сор — красная рамка
COLOUR_FINGER = (87, 0, 245)  # закрас пальцев — малиновый контур (как метка CVAT #F50057)


def trim_sheet(
    page: np.ndarray,
    boxes: dict[Method, Box],
    stray: list[Item],
    fingers: list[list[list[int]]],
    header: list[str],
    measures: dict[Method, TrimMeasures],
) -> np.ndarray:
    """Лист одной страницы.

    Args:
        page: Страница BGR.
        boxes: Рамки способов, px страницы.
        stray: Строки-сор.
        fingers: Контуры закраса пальцев из паспорта, px страницы.
        header: Строки шапки.
        measures: Меры способов (строка в шапке на способ).

    Returns:
        Лист BGR.
    """
    f = SHEET_HEIGHT / page.shape[0]
    canvas = cv2.resize(page, (int(page.shape[1] * f), SHEET_HEIGHT), interpolation=cv2.INTER_AREA)
    for poly in fingers:
        cv2.polylines(canvas, [np.round(np.array(poly) * f).astype(np.int32)], True, COLOUR_FINGER, 2, cv2.LINE_AA)
    for item in stray:
        x0, y0, x1, y1 = (int(round(v * f)) for v in item.box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOUR_STRAY, 2)
    for method, (x0, y0, x1, y1) in boxes.items():
        cv2.rectangle(canvas, (int(x0 * f), int(y0 * f)), (int(x1 * f) - 1, int(y1 * f) - 1), COLOUR_OF[method], BOX_PX)
    lines = header + [
        f"{m.value}: потеряно элементов {r.lost_items} ({r.lost_mm2} мм²), мусор у краёв {r.junk_mm2} мм², "
        f"сора внутри {r.strays_in}, срез в/н/л/п {r.cut_mm} мм"
        for m, r in measures.items()
    ]
    legend = [LegendEntry(f"рамка {m.value}", c) for m, c in COLOUR_OF.items()] + [
        LegendEntry("строка-сор (вне тела страницы)", COLOUR_STRAY, style=SampleStyle.BOX),
        LegendEntry("закрас пальцев (паспорт страницы)", COLOUR_FINGER),
    ]
    return framed(canvas, lines, legend)
