"""Достройка рамки до краёв пятен краски, классы подсказок и отдельный выход формул — на синтетике."""

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box, HintBox
from ocr_utils.page_layout.line_art.classes import ObjectClass
from ocr_utils.page_layout.line_art.detector import (
    LineArtInputs,
    detect_formulas,
    detect_line_art,
    hints_from_surya,
    seeds_from,
)
from ocr_utils.page_layout.line_art.expand import grow_to_components
from ocr_utils.page_layout.surya.blocks import from_boxes

DPI = 300
MM = DPI / 25.4


def page(height: int = 1200, width: int = 1000) -> np.ndarray:
    """Белая полоса: 255 — бумага."""
    return np.full((height, width), 255, np.uint8)


def test_рамка_режущая_букву_снизу_дорастает_до_её_края() -> None:
    gray = page()
    cv2.rectangle(gray, (300, 400), (340, 470), 0, -1)  # «буква» 40×70 px
    grown = grow_to_components(Box(280, 380, 360, 440), gray == 0, DPI, 3.0)
    assert grown.box.y1 >= 471 and grown.failed == ()


def test_линейка_колонки_не_тянет_рамку_дальше_потолка() -> None:
    gray = page()
    cv2.line(gray, (100, 500), (900, 500), 0, 3)  # линейка на всю ширину
    start = Box(400, 450, 500, 510)
    grown = grow_to_components(start, gray == 0, DPI, 3.0)
    assert grown.box.x0 >= start.x0 - int(3 * MM) - 1 and grown.box.x1 <= start.x1 + int(3 * MM) + 1
    assert "слева" in grown.failed and "справа" in grown.failed


def test_преграда_не_пересекается() -> None:
    gray = page()
    cv2.rectangle(gray, (300, 400), (340, 480), 0, -1)
    barrier = Box(250, 460, 400, 600)
    grown = grow_to_components(Box(280, 380, 360, 440), gray == 0, DPI, 6.0, [barrier])
    assert grown.box.y1 <= barrier.y0 and "снизу" in grown.failed


def test_классы_подсказок_surya_и_формулы_вне_затравок_line_art() -> None:
    gray = page()
    cv2.rectangle(gray, (100, 100), (400, 140), 0, -1)
    blocks = from_boxes(
        [("Equation", 0.9, Box(90, 95, 410, 130)), ("Figure", 0.8, Box(500, 500, 900, 900))], 1000, 1200
    )
    hints = hints_from_surya(blocks, gray == 0, DPI)
    by_source = {h.source: h for h in hints}
    assert by_source["surya:Equation"].likely == (ObjectClass.FORMULA,)
    assert by_source["surya:Figure"].likely == (ObjectClass.DIAGRAM, ObjectClass.DRAWING)
    assert by_source["surya:Equation"].y1 >= 141, "срезанный низ формулы достроен"
    inputs = LineArtInputs(gray, DPI, [], hints)
    assert all(label != "surya:Equation" for _, label in seeds_from(inputs))
    formulas = detect_formulas(inputs)
    assert len(formulas) == 1 and formulas[0].kind.value == "formula"
    assert all("surya:Equation" not in r.info["sources"] for r in detect_line_art(inputs))


def test_формула_под_исключением_и_пустая_отбрасываются() -> None:
    gray = page()
    cv2.rectangle(gray, (100, 100), (400, 140), 0, -1)
    formula = HintBox(90, 95, 410, 145, source="surya:Equation", likely=(ObjectClass.FORMULA,))
    empty = HintBox(600, 600, 900, 650, source="surya:Equation", likely=(ObjectClass.FORMULA,))
    assert detect_formulas(LineArtInputs(gray, DPI, [Box(0, 0, 500, 200)], [formula])) == []
    assert detect_formulas(LineArtInputs(gray, DPI, [], [empty])) == []
