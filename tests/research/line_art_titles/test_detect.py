"""Поворот рамок из базы вслед за повёрнутой заострённой копией."""

import pytest

from research.line_art_titles.detect import rotate_box

WIDTH, HEIGHT = 3448, 5963  # скан 1974/07 с.16


def test_поворот_на_90_совпадает_с_находкой_детектора_таблиц_на_1974_07_с16() -> None:
    assert rotate_box((332, 460, 3196, 5736, "table"), WIDTH, HEIGHT, 90) == (227, 332, 5503, 3196, "table")


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_четыре_поворота_возвращают_рамку_на_место(angle: int) -> None:
    box = (100, 200, 400, 900, "table")
    width, height = WIDTH, HEIGHT
    current = box
    for _ in range(4):
        current = rotate_box(current, width, height, angle)
        if angle in (90, 270):
            width, height = height, width
    assert current == box
