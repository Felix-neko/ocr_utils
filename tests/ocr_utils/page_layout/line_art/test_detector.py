"""Вход единого детектора line art: исключения и подсказки списками рамок, на синтетике."""

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box, HintBox
from ocr_utils.page_layout.line_art.detector import LineArtInputs, detect_line_art, seeds_from

PAGE = (6733, 4165)  # страница пака-1 при 600 dpi
DPI = 600

# Область рисунка на полосе.
DRAWING = Box(600, 2000, 3400, 5600)


def blank() -> np.ndarray:
    """Чистая белая полоса."""
    return np.full(PAGE, 255, np.uint8)


def with_drawing(page: np.ndarray) -> np.ndarray:
    """Связный штриховой рисунок (ломаная толщиной 7 px) в рамке ``DRAWING``."""
    rng = np.random.default_rng(0)
    points = rng.integers([DRAWING.x0, DRAWING.y0], [DRAWING.x1, DRAWING.y1], size=(60, 2))
    cv2.polylines(page, [points.reshape(-1, 1, 2).astype(np.int32)], False, 0, 7)
    return page


def with_scattered_diagram(page: np.ndarray) -> np.ndarray:
    """Схема из несвязных коробок: ни одно пятно не проходит ворота по размеру само по себе."""
    for row in range(8):
        for col in range(6):
            left, top = DRAWING.x0 + col * 460, DRAWING.y0 + row * 450
            cv2.rectangle(page, (left, top), (left + 150, top + 100), 0, 5)
            cv2.line(page, (left + 75, top + 115), (left + 75, top + 430), 0, 5)
    return page


def overlap(box: Box, other: Box) -> float:
    """Доля ``other``, накрытая ``box``."""
    width = min(box.x1, other.x1) - max(box.x0, other.x0)
    height = min(box.y1, other.y1) - max(box.y0, other.y0)
    return max(0, width) * max(0, height) / other.area


def test_без_подсказок_и_исключений_рисунок_находится_пикселями() -> None:
    regions = detect_line_art(LineArtInputs(with_drawing(blank()), DPI, [], []))
    assert any(overlap(r.box, DRAWING) > 0.8 for r in regions)
    assert all(r.info["sources"] == ["ink"] for r in regions)


def test_исключение_гасит_кандидата() -> None:
    regions = detect_line_art(LineArtInputs(with_drawing(blank()), DPI, [DRAWING.padded(20)], []))
    assert regions == []


def test_голая_рамка_подсказки_даёт_область_без_крупного_пятна() -> None:
    page = with_scattered_diagram(blank())
    assert detect_line_art(LineArtInputs(page, DPI, [], [])) == [], "пиксели сами схему не видят"
    regions = detect_line_art(LineArtInputs(page, DPI, [], [DRAWING]))
    assert len(regions) == 1 and overlap(regions[0].box, DRAWING) > 0.8
    assert regions[0].info["sources"] == ["hint"]


def test_метка_и_уверенность_подсказки_доезжают_до_info() -> None:
    hint = HintBox(*DRAWING.as_tuple(), source="surya:Figure", confidence=0.87)
    regions = detect_line_art(LineArtInputs(with_drawing(blank()), DPI, [], [hint]))
    art = max(regions, key=lambda r: overlap(r.box, DRAWING))
    assert "surya:Figure" in art.info["sources"] and "ink" in art.info["sources"]
    assert art.info["surya_conf"] == 0.87
    assert art.confidence == round(2 / 3, 4)


def test_затравки_в_порядке_подсказок() -> None:
    hints = [Box(0, 0, 10, 10), HintBox(5, 5, 20, 20, source="tables:схема")]
    inputs = LineArtInputs(blank(), DPI, [], hints)
    assert seeds_from(inputs) == [((0, 0, 10, 10), "hint"), ((5, 5, 20, 20), "tables:схема")]
