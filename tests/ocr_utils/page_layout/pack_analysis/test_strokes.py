"""Одиночные черты — не line art: рамки рисунков и «неясно» тоньше 5 мм снимаются с объектов полосы (``pack_analysis.final.split_strokes``)."""

from __future__ import annotations

from ocr_utils.page_layout.line_art.deepseek.decide import STROKE_MAX_MM, is_thin_stroke
from ocr_utils.page_layout.pack_analysis.final import split_strokes

# Полосы пака-1 в родном разрешении — 600 dpi: 1 мм ≈ 23.6 px.
DPI = 600.0
MM = DPI / 25.4


def _box(width_mm: float, height_mm: float) -> list[int]:
    """Рамка заданного размера в миллиметрах от угла (100, 100)."""
    return [100, 100, 100 + round(width_mm * MM), 100 + round(height_mm * MM)]


def test_thin_frame_is_a_stroke_in_either_orientation():
    """Концевая черта 36 × 3 мм (1966/05 с.86) и линейка колонки 2 × 45 мм — черты; рисунок 40 × 30 мм — нет."""
    assert is_thin_stroke(_box(36, 3.2), DPI)
    assert is_thin_stroke(_box(2, 45), DPI)
    assert not is_thin_stroke(_box(40, 30), DPI)
    assert not is_thin_stroke(_box(STROKE_MAX_MM + 0.5, 60), DPI)


def test_split_takes_only_thin_drawings_and_unclear():
    """Снимаются тонкие «рисунок» и «неясно»; тонкая таблица, формула и толстый рисунок остаются объектами."""
    objects = [
        {"class": "рисунок", "box": _box(36, 3), "source": "line art: детектор"},
        {"class": "неясно", "box": _box(3, 40), "source": "line art"},
        {"class": "рисунок", "box": _box(60, 40), "source": "line art: детектор"},
        {"class": "таблица", "box": _box(80, 4), "source": "таблицы"},
        {"class": "формула", "box": _box(30, 4), "source": "surya Equation"},
    ]
    kept, strokes = split_strokes(objects, DPI)
    assert [o["class"] for o in kept] == ["рисунок", "таблица", "формула"]
    assert strokes == [_box(36, 3), _box(3, 40)]
