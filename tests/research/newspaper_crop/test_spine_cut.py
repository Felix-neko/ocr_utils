"""Тесты отрезки за складкой корешка: прямая стороны по складке, маска «за складкой», перенос стороны, поиск складки."""

from __future__ import annotations

import numpy as np

from research.newspaper_crop.spine_cut import SideFold, _move_side, beyond_mask, find_side_fold, straight_cut
from research.spread_defects.fold_line import SpineSide


def _fold(side: SpineSide, points: np.ndarray) -> SideFold:
    """Принятая складка с заданными точками (меры не важны).

    Args:
        side: Сторона.
        points: (N, 2) точки линии.

    Returns:
        ``SideFold``.
    """
    return SideFold(side, True, "", 0, 0, 1, 1, 0, 0, 1.2, 0.2, points)


def test_straight_cut_passes_outer_point() -> None:
    """Прямая по правой складке проходит по самой внешней (правой) точке выгнутой линии — наш лист не режется."""
    ys = np.arange(0, 400, dtype=float)
    xs = 280 + 10 * np.sin(np.linspace(0, np.pi, len(ys)))  # выгиб наружу до x = 290
    p0, p1 = straight_cut(_fold(SpineSide.RIGHT, np.stack([xs, ys], 1)), (400, 300))
    assert abs(p0[0] - 290) < 1.5 and abs(p1[0] - 290) < 1.5
    assert p0[1] == 0 and p1[1] == 399


def test_beyond_mask_right_side() -> None:
    """Маска «за складкой» у правой стороны — полоса между линией и правым краем."""
    ys = np.arange(0, 100, dtype=float)
    pts = np.stack([np.full_like(ys, 70.0), ys], 1)
    m = beyond_mask((100, 90), _fold(SpineSide.RIGHT, pts))
    assert m[:, 75:].all() and not m[:, :65].any()


def test_beyond_mask_top_side() -> None:
    """Маска «за складкой» у верхней стороны — полоса над линией."""
    xs = np.arange(0, 90, dtype=float)
    pts = np.stack([xs, np.full_like(xs, 20.0)], 1)
    m = beyond_mask((100, 90), _fold(SpineSide.TOP, pts))
    assert m[:15].all() and not m[25:].any()


def test_move_side_left() -> None:
    """Перенос левой стороны двигает только левые углы."""
    corners = {"tl": np.array([0.0, 0]), "tr": np.array([9.0, 0]), "br": np.array([9.0, 19]), "bl": np.array([0.0, 19])}
    out = _move_side(corners, SpineSide.LEFT, np.array([3.0, 0]), np.array([4.0, 19]))
    assert np.allclose(out["tl"], [3, 0]) and np.allclose(out["bl"], [4, 19])
    assert np.allclose(out["tr"], [9, 0]) and np.allclose(corners["tl"], [0, 0])


def test_find_side_fold_on_synthetic_spine() -> None:
    """Синтетика: справа за тёмной складкой на 60 px — светлая чужая полоса с «строками»; складка найдена и принята,
    у левой стороны (чистое поле) — нет."""
    h, w = 600, 400
    rgb = np.full((h, w, 3), (205, 200, 185), np.uint8)
    # Наш текст: тёмные штрихи-«строки» в середине.
    for y in range(80, 520, 16):
        rgb[y : y + 5, 60:300] = (40, 40, 40)
    # Складка: тёмная полоса с тенью вдоль правой стороны, за ней — чужая полоса светлее и со «строками».
    rgb[:, w - 60 : w - 56] = (90, 85, 75)
    rgb[:, w - 56 :] = (225, 220, 205)
    for y in range(20, h - 20, 14):
        rgb[y : y + 4, w - 50 : w - 10] = (50, 50, 50)
    right = find_side_fold(rgb, SpineSide.RIGHT, [])
    assert right.accepted, right.reason
    assert 50 <= np.median(w - 1 - right.points[:, 0]) <= 70
    left = find_side_fold(rgb, SpineSide.LEFT, [])
    assert not left.accepted


def test_neighbor_beyond_rules() -> None:
    """Отрезать: освещённая чужая полоса с печатью, широкая затенённая с густой печатью; не отрезать: узкий наш завал."""
    from research.newspaper_crop.spine_cut import neighbor_beyond

    assert neighbor_beyond(1.2, 0.15, 40)  # освещённая чужая полоса (2_2060)
    assert neighbor_beyond(0.95, 0.125, 109)  # затенённая широкая (2_3190)
    assert not neighbor_beyond(0.944, 0.133, 67)  # наш завал с печатью (DSCF0078)
    assert not neighbor_beyond(1.1, 0.003, 32)  # освещённое пустое поле (DSCF0018)


def test_deep_line_needs_dense_print() -> None:
    """Глубокая линия с редкой печатью за ней (линейка внутри нашей полосы) не режется; с густой — режется."""
    from research.newspaper_crop.spine_cut import neighbor_beyond

    assert not neighbor_beyond(1.0, 0.045, 236)  # 2_0850 справа
    assert neighbor_beyond(1.2, 0.18, 202)  # 1_0720 слева, соседняя полоса


def test_only_side_spines() -> None:
    """Складки ищутся только у боковых сторон."""
    from research.newspaper_crop.spine_cut import side_folds

    rgb = np.full((900, 800, 3), 200, np.uint8)
    assert {f.side for f in side_folds(rgb)} == {SpineSide.LEFT, SpineSide.RIGHT}
