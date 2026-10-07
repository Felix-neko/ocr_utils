"""Тесты вариантов линии сгиба: складка над фото, изогнутая складка, прищепка, пересчёт рамок разметки в кадр."""

import numpy as np
import pytest

from research.spread_defects.fold_line import Assembly, Candidate, Variant, band_pieces, find_line, piece_box_to_frame

WIDTH, HEIGHT = 2000, 800


def _frame(fold_rows: np.ndarray) -> np.ndarray:
    """Синтетический кадр: загиб соседней полосы (150) выше складки, тонкая тёмная складка (80), бумага (200)."""
    gray = np.full((HEIGHT, WIDTH), 200, np.uint8)
    rows = np.arange(HEIGHT)[:, None]
    gray[rows < fold_rows[None, :] - 4] = 150
    gray[np.abs(rows - fold_rows[None, :]) <= 4] = 80
    return np.repeat(gray[..., None], 3, axis=2)


def test_ridge_path_ignores_photo_below_fold():
    """Тёмное фото под складкой не утягивает линию хребта (C) с путём: линия остаётся на складке."""
    rgb = _frame(np.full(WIDTH, 120))
    rgb[170:300, 700:1300] = 60  # фото у сгиба
    line = find_line(rgb, Variant(Candidate.RIDGE, False, False, False, Assembly.PATH))
    assert np.abs(line.rows[700:1300] - 120).max() <= 6


def test_path_follows_tilted_fold():
    """Наклонная складка (как на 0180_2: 150 → 80 по ширине) — путь идёт вдоль неё."""
    fold = np.linspace(150, 80, WIDTH)
    line = find_line(_frame(fold), Variant(Candidate.RIDGE, False, False, False, Assembly.PATH))
    assert np.abs(line.rows[50:-50] - fold[50:-50]).max() <= 6


def test_clip_columns_bridged():
    """Под прищепкой (зелёный прямоугольник) с F линия продолжает складку соседних столбцов."""
    rgb = _frame(np.full(WIDTH, 120))
    rgb[0:300, 400:600] = (20, 140, 40)
    line = find_line(rgb, Variant(Candidate.MINIMUM, False, False, True, Assembly.PATH))
    assert np.abs(line.rows[400:600] - 120).max() <= 6


def test_piece_box_maps_back_to_frame():
    """Рамка вокруг метки в прямом куске ленты пересчитывается в рамку той же метки в кадре (обе стороны)."""
    for side in ("L", "R"):
        rgb = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        rgb[100:110, 1700:1720] = 255  # метка: строки 100–109, столбцы 1700–1719 кадра
        for piece in band_pieces(rgb, side, band=600):
            ys, xs = np.nonzero(piece.image[..., 0])
            if len(ys):
                x0, y0, x1, y1 = piece_box_to_frame(piece, xs.min(), ys.min(), xs.max(), ys.max())
                assert (round(x0), round(y0), round(x1), round(y1)) == (1700, 100, 1719, 109)
                break
        else:
            raise AssertionError("метка не попала ни в один кусок")


# --- Ориентация: сторона корешка на кадре по EXIF ---------------------------------------------------------------

from research.spread_defects.fold_line import (  # noqa: E402
    SpineSide,
    boxes_to_spine_top,
    points_from_spine_top,
    points_to_spine_top,
    spine_side_from_exif,
    to_spine_top,
)


@pytest.mark.parametrize("side", list(SpineSide))
def test_spine_top_roundtrip_points(side: SpineSide) -> None:
    """Точки кадра по EXIF → канон → обратно возвращаются на место; пиксель под точкой тот же."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (40, 70), np.uint8)
    canon = to_spine_top(img, side)
    xs, ys = rng.integers(0, 70, 20).astype(float), rng.integers(0, 40, 20).astype(float)
    cx, cy = points_to_spine_top(xs, ys, side, img.shape)
    assert np.array_equal(canon[cy.astype(int), cx.astype(int)], img[ys.astype(int), xs.astype(int)])
    bx, by = points_from_spine_top(cx, cy, side, img.shape)
    assert np.allclose(bx, xs) and np.allclose(by, ys)


def test_spine_side_moves_to_top() -> None:
    """Тёмная полоса у стороны корешка после поворота оказывается у верхнего края."""
    for side, sl in (
        (SpineSide.LEFT, np.s_[:, :5]),
        (SpineSide.RIGHT, np.s_[:, -5:]),
        (SpineSide.BOTTOM, np.s_[-5:, :]),
        (SpineSide.TOP, np.s_[:5, :]),
    ):
        img = np.full((40, 70), 200, np.uint8)
        img[sl] = 0
        canon = to_spine_top(img, side)
        assert canon[:5].max() == 0 and canon[10:].min() == 200


def test_spine_side_from_exif_eg_1985() -> None:
    """ЭГ 1985: корешок у верха снимка; Orientation 8 — слева, 6 — справа, 3 — снизу, 1 — сверху."""
    assert spine_side_from_exif(8) is SpineSide.LEFT
    assert spine_side_from_exif(6) is SpineSide.RIGHT
    assert spine_side_from_exif(3) is SpineSide.BOTTOM
    assert spine_side_from_exif(1) is SpineSide.TOP
    # Корешок справа на снимке «как снято» при Orientation 8 (поворот против часовой) уходит наверх.
    assert spine_side_from_exif(8, SpineSide.RIGHT) is SpineSide.TOP


def test_boxes_to_spine_top_ordered() -> None:
    """Рамка после поворота остаётся упорядоченной и того же размера (с переменой сторон)."""
    ((x0, y0, x1, y1),) = boxes_to_spine_top([(10, 5, 30, 8)], SpineSide.LEFT, (40, 70))
    assert x0 < x1 and y0 < y1
    assert (x1 - x0, y1 - y0) == (3, 20)
