"""Тесты прямых разворота: разделитель страниц, края книги, ограничители внешних краёв (книга в кадре и за кадром)."""

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.page_pose import PageFeatures, PagePose, PageQuad
from ocr_utils.scan_cropping.v2.side_fit import Side
from ocr_utils.scan_cropping.v2.spread_lines import (
    EdgeKind,
    Line,
    SpineMethod,
    auto_source,
    book_edge,
    finger_hands,
    mirror_k,
    VirtualRule,
    spine_candidates,
    spread_lines,
)

H, W = 900, 1300
FEATURES = PageFeatures(fill=1.0, keystone=0.0, width_share=1.0, light_share=1.0, finger=0.0, curl=0.0)


def poly_mask(pts: list[list[float]]) -> np.ndarray:
    """Маска многоугольника на кадре ``H × W``."""
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(pts)).astype(np.int32)], 1)
    return mask > 0


def page(pts: list[list[float]]) -> PageQuad:
    """Страница-четырёхугольник с трапецией по его вершинам (по часовой стрелке от левого верхнего)."""
    return PageQuad(poly_mask(pts), np.asarray(pts, np.float64), FEATURES, PagePose.FLAT)


def spread(dx_book: float = 0.0) -> tuple[list[PageQuad], np.ndarray]:
    """Разворот с наклонённым разделителем (низ правее на 40 px) и книгой шире страниц на 30 px с боков.

    Args:
        dx_book: Сдвиг всей книги по горизонтали (отрицательный — книга уходит за левую рамку).

    Returns:
        ``(страницы, маска книги)``.
    """
    s = dx_book
    left = page([[200 + s, 100], [640 + s, 100], [680 + s, 800], [220 + s, 800]])
    right = page([[640 + s, 100], [1060 + s, 100], [1120 + s, 800], [680 + s, 800]])
    book = poly_mask([[170 + s, 80], [1090 + s, 80], [1150 + s, 820], [190 + s, 820]])
    return [left, right], book


def test_spine_candidates_follow_inner_sides():
    """Все способы разделителя ядра проходят по общей стороне страниц: x = 640 вверху, 680 внизу (±1.5 px)."""
    pages, _ = spread()
    for method, line in spine_candidates(pages).items():
        assert abs(line.x_at(100) - 640) <= 1.5, method
        assert abs(line.x_at(800) - 680) <= 1.5, method


def test_bounds_parallel_to_book_in_frame():
    """Книга целиком в кадре: ограничители параллельны краям книги и стоят по внешней кромке страниц."""
    pages, book = spread()
    lines = spread_lines(pages, spine_candidates(pages)[SpineMethod.QUAD_MID], SpineMethod.QUAD_MID, book, book & False)
    assert lines.left_kind is EdgeKind.BOOK and lines.right_kind is EdgeKind.BOOK
    assert abs(lines.left.k - 20 / 740) < 0.01
    assert abs(lines.right.k - 60 / 740) < 0.01
    assert abs(lines.left.x_at(450) - 210) <= 3
    assert abs(lines.right.x_at(450) - 1090) <= 3


def test_virtual_bound_mirrors_opposite_edge():
    """Книга упирается в левую рамку: левый ограничитель виртуальный, наклон зеркален правому краю книги."""
    pages, book = spread(dx_book=-195)
    spine = spine_candidates(pages)[SpineMethod.QUAD_MID]
    lines = spread_lines(pages, spine, SpineMethod.QUAD_MID, book, book & False, VirtualRule.MIRROR)
    assert not lines.book_left.visible and lines.book_right.visible
    assert lines.left_kind is EdgeKind.VIRTUAL and lines.right_kind is EdgeKind.BOOK
    assert abs(lines.left.k - mirror_k(spine, lines.book_right.line)) < 1e-9
    # Зеркало: 2·атан(40/700) − атан(60/740) ≈ 0.0333 (низ правее верха).
    assert abs(lines.left.k - np.tan(2 * np.arctan(40 / 700) - np.arctan(60 / 740))) < 0.01


def test_finger_at_book_edge_dropped():
    """Палец, выходящий за край книги, не сдвигает прямую края: строки у руки отброшены."""
    _, book = spread()
    hands = np.zeros_like(book)
    hands[400:500, 100:260] = True
    edge = book_edge(book | hands, hands, Side.LEFT, (100.0, 800.0))
    assert len(edge.dropped) > 0
    assert abs(edge.line.x_at(450) - (170 + 20 * (450 - 80) / 740)) <= 2


def test_cover_coloured_like_skin_not_a_hand():
    """Кожа по цвету без связи с пальцем (красный переплёт) — не рука; кожа, касающаяся закраса пальца, — рука."""
    skin = np.zeros((H, W), bool)
    skin[100:800, 150:175] = True  # «переплёт» вдоль края
    skin[400:450, 1100:1200] = True  # палец
    paint = np.zeros_like(skin)
    paint[420:430, 1150:1160] = True
    hands = finger_hands(skin, paint)
    assert not hands[300, 160] and hands[410, 1110]


def test_virtual_bound_parallel_to_spine_by_default():
    """По умолчанию виртуальный ограничитель параллелен разделителю."""
    pages, book = spread(dx_book=-195)
    spine = spine_candidates(pages)[SpineMethod.QUAD_MID]
    lines = spread_lines(pages, spine, SpineMethod.QUAD_MID, book, book & False)
    assert lines.left_kind is EdgeKind.VIRTUAL and abs(lines.left.k - spine.k) < 1e-9


def test_cover_under_finger_is_hand_only_near_paint():
    """Переплёт цвета кожи, на котором лежит палец: рукой считается только его часть у закраса (≤ 15 мм)."""
    skin = np.zeros((H, W), bool)
    skin[100:800, 150:175] = True  # переплёт вдоль края во всю высоту
    skin[700:760, 150:260] = True  # палец на переплёте внизу
    paint = np.zeros_like(skin)
    paint[720:740, 200:240] = True
    hands = finger_hands(skin, paint)
    assert hands[720, 160] and not hands[300, 160]


def test_auto_spine_falls_back_to_quad_mid():
    """AUTO — gap_mid, пока его наклон близок к quad_mid; при разнице больше 1° — quad_mid."""
    pages, _ = spread()
    candidates = spine_candidates(pages)
    assert auto_source(candidates) is SpineMethod.GAP_MID
    assert candidates[SpineMethod.AUTO] is candidates[SpineMethod.GAP_MID]
    quad = candidates[SpineMethod.QUAD_MID]
    torn = Line(quad.x0, quad.y0, quad.k + np.tan(np.radians(2.0)), quad.y_top, quad.y_bot)
    assert auto_source({**candidates, SpineMethod.GAP_MID: torn}) is SpineMethod.QUAD_MID
