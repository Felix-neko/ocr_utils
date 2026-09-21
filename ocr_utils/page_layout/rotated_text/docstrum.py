"""Повёрнутый текст вне таблиц: направление на ближайших соседей глифа (Docstrum).

ПОЧЕМУ НЕ RLSA. Смыкание краски вдоль оси (``orientation.detectors.ink_axis.line_boxes``)
ловит длинные строки и пропускает боковые подписи в одно-три слова на чертежах — как раз
то, что здесь нужно. У Docstrum (O'Gorman, 1993) единица — не строка, а пара «глиф —
ближайший сосед»: у прямого текста ближайшие соседи буквы лежат слева и справа на
расстоянии в доли высоты глифа, у бокового — сверху и снизу. Межстрочный интервал в
полтора-два кегля дальше, чем межбуквенный, поэтому порог по расстоянию отсекает соседей
из другой строки, и вертикальные цепочки собираются только там, где буквы стоят друг над
другом вплотную — то есть в повёрнутом слове.

ГРАНИЦЫ. Компоненты размера глифа (0.8–7 мм) — линейки, рамки и штрихи чертежа не
участвуют; цепочка от трёх глифов; соседние вертикальные цепочки на расстоянии ширины
глифа склеиваются в одну зону (многострочная боковая подпись). Сторона (90 или 270)
здесь не решается — её называет tesseract по числу прочитанных букв (``zones.side_of``).

ПРЯМОЙ ТЕКСТ — тем же ходом по транспонированному растру: горизонтальная цепочка глифов
есть вертикальная цепочка в координатах ``(y, x)``. Так ищутся прямые подписи на схемах и
чертежах, для которых FineReader слоя не оставил (``cluster_lines(..., vertical=False)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from ocr_utils.page_layout.geometry import Box

from ocr_utils.text_layer_fix import mm_to_px

# Компонента размера глифа: высота и ширина в этих пределах (мм бумаги). 0.8 мм — точка над
# «й» и пыль отсекаются, 7 мм — заголовок, но не рамка таблицы и не штрих чертежа.
GLYPH_MIN_MM = 0.8
GLYPH_MAX_MM = 7.0

# Заполнение габарита краской: буква — от 12 %, длинная диагональ чертежа — меньше.
GLYPH_MIN_FILL = 0.12

# Сосед считается «в той же строке», если расстояние между центрами не больше стольких
# высот глифа (по большему из пары): межбуквенный шаг 0.6–1.0 h, межстрочный 1.3 h и больше.
NEIGHBOR_REACH = 1.15

# Порог соседства для ПРЯМЫХ строк: шаг букв в строке 1.2–1.5 высоты строчной (замер на
# синтетике DejaVu: 13–15 px при x-высоте 12), а межстрочный шаг вдвое-втрое больше и
# отсекается правилом «нет более близкого соседа поперёк». У боковых цепочек порог строже
# (1.15): столбик одиночных цифр списка с шагом строки иначе собрался бы в «боковое слово».
UPRIGHT_REACH = 1.6

# Угол вектора к соседу считается вертикальным, если он ближе к 90°, чем на столько градусов.
VERTICAL_TOLERANCE_DEG = 30.0

# Столько ближайших соседей смотрится у каждого глифа.
NEIGHBORS = 3

# Минимум глифов в вертикальной цепочке, чтобы считать её боковым словом.
MIN_CHAIN = 3

# Соседние вертикальные цепочки склеиваются в зону, если по горизонтали между ними не больше
# стольких медианных ширин глифа, а по вертикали они перекрываются хотя бы на эту долю.
MERGE_GAP_GLYPHS = 1.3
MERGE_OVERLAP = 0.4

# Куски одной вертикальной строки, разорванной пробелами между словами, склеиваются, если
# они лежат на одной вертикали (перекрытие по x от этой доли) и зазор между ними не больше
# стольких медианных высот глифа (пробел — 0.3–0.6 em, «с» между двумя пробелами — до 2.5).
ALONG_OVERLAP = 0.5
ALONG_GAP_GLYPHS = 3.0
# Зазор длиннее, но в нём стоит одиночный глиф (слово из одной буквы: «с», «и», «в»), —
# тоже склеивается, пока зазор не длиннее стольких высот глифа.
BRIDGED_GAP_GLYPHS = 6.0

# Зона должна быть вытянута вертикально хотя бы во столько раз, иначе это пятно, а не строка.
MIN_ELONGATION = 1.6


@dataclass
class GlyphStats:
    """Компоненты размера глифа: центры, габариты, высоты (всё в пикселях поданного растра)."""

    centers: np.ndarray  # (n, 2) — x, y
    boxes: np.ndarray  # (n, 4) — x0, y0, x1, y1
    heights: np.ndarray  # (n,)
    widths: np.ndarray  # (n,)

    def __len__(self) -> int:
        return int(len(self.centers))


def glyph_components(gray: np.ndarray, dpi: int, exclude: "list[Box] | None" = None) -> GlyphStats:
    """Компоненты связности размера глифа на битональном растре.

    Args:
        gray: Серый растр (0 — краска).
        dpi: Его разрешение, для перевода миллиметров в пиксели.
        exclude: Прямоугольники (таблицы, растровые области), чьи компоненты не нужны.

    Returns:
        Статистика компонент, прошедших отбор по размеру и заполнению.
    """
    ink = (gray < 128).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(ink, 8)
    if count <= 1:
        empty = np.empty((0, 2))
        return GlyphStats(empty, np.empty((0, 4)), np.empty(0), np.empty(0))
    stats, centroids = stats[1:], centroids[1:]
    width = stats[:, cv2.CC_STAT_WIDTH].astype(float)
    height = stats[:, cv2.CC_STAT_HEIGHT].astype(float)
    area = stats[:, cv2.CC_STAT_AREA].astype(float)
    low, high = mm_to_px(GLYPH_MIN_MM, dpi), mm_to_px(GLYPH_MAX_MM, dpi)
    keep = (
        (height >= low) & (height <= high) & (width >= 2) & (width <= high) & (area >= GLYPH_MIN_FILL * width * height)
    )
    if exclude:
        cx, cy = centroids[:, 0], centroids[:, 1]
        for box in exclude:
            keep &= ~((cx >= box.x0) & (cx <= box.x1) & (cy >= box.y0) & (cy <= box.y1))
    left, top = stats[keep, cv2.CC_STAT_LEFT], stats[keep, cv2.CC_STAT_TOP]
    boxes = np.stack([left, top, left + width[keep], top + height[keep]], axis=1).astype(float)
    return GlyphStats(centroids[keep].astype(float), boxes, height[keep], width[keep])


def vertical_edges(stats: GlyphStats, reach: float = NEIGHBOR_REACH, k: int = NEIGHBORS) -> np.ndarray:
    """Пары глифов, стоящих друг над другом вплотную (кандидаты в боковое слово).

    Пара принимается, только если ни у одного из двух глифов нет более близкого соседа по
    горизонтали: у буквы прямого текста ближайший сосед — соседняя буква строки, и стоящая
    под ней буква следующей строки (первые буквы строк, колонка цифр) пары не образует.

    Args:
        stats: Компоненты страницы.
        reach: Порог расстояния между центрами в высотах глифа.
        k: Сколько ближайших соседей смотреть.

    Returns:
        Массив ``(m, 2)`` индексов пар ``i < j``.
    """
    n = len(stats)
    if n < 2:
        return np.empty((0, 2), int)
    tree = cKDTree(stats.centers)
    distances, indices = tree.query(stats.centers, k=min(k + 1, n))
    tolerance = np.tan(np.radians(90.0 - VERTICAL_TOLERANCE_DEG))
    # Ближайший горизонтальный сосед каждого глифа (любой дальности среди k ближайших).
    nearest_horizontal = np.full(n, np.inf)
    vertical: dict[tuple[int, int], float] = {}
    for i in range(n):
        for distance, j in zip(distances[i][1:], indices[i][1:]):
            if j >= n or j == i:
                continue
            dx = abs(stats.centers[j, 0] - stats.centers[i, 0])
            dy = abs(stats.centers[j, 1] - stats.centers[i, 1])
            if dy >= tolerance * max(dx, 1e-6):
                size = max(stats.heights[i], stats.heights[j], stats.widths[i], stats.widths[j])
                if distance <= reach * size:
                    vertical[(min(i, j), max(i, j))] = float(distance)
            elif dx >= tolerance * max(dy, 1e-6):
                nearest_horizontal[i] = min(nearest_horizontal[i], float(distance))
    pairs = [
        pair
        for pair, distance in vertical.items()
        if distance < nearest_horizontal[pair[0]] and distance < nearest_horizontal[pair[1]]
    ]
    return np.array(sorted(pairs), int).reshape(-1, 2)


def _components(n: int, edges: np.ndarray) -> list[list[int]]:
    """Связные компоненты графа на ``n`` вершинах (union-find)."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _glyph_in_gap(stats: GlyphStats, first: Box, second: Box) -> bool:
    """Есть ли центр глифа в промежутке между двумя кусками одной вертикальной строки."""
    x0, x1 = max(first.x0, second.x0), min(first.x1, second.x1)
    y0, y1 = (first.y1, second.y0) if second.y0 >= first.y1 else (second.y1, first.y0)
    cx, cy = stats.centers[:, 0], stats.centers[:, 1]
    return bool(np.any((cx >= x0) & (cx <= x1) & (cy > y0) & (cy < y1)))


def cluster_rotated(stats: GlyphStats, dpi: int, reach: float = NEIGHBOR_REACH) -> list[Box]:
    """Зоны бокового текста: вертикальные цепочки глифов, склеенные в многострочные подписи.

    Args:
        stats: Компоненты страницы (:func:`glyph_components`).
        dpi: Разрешение растра.
        reach: Порог соседства в высотах глифа.

    Returns:
        Рамки зон в пикселях растра, отсортированные сверху вниз и слева направо.
    """
    edges = vertical_edges(stats, reach)
    if len(edges) == 0:
        return []
    chains = [group for group in _components(len(stats), edges) if len(group) >= MIN_CHAIN]
    chain_boxes: list[Box] = []
    for group in chains:
        boxes = stats.boxes[group]
        box = Box(
            int(boxes[:, 0].min()),
            int(boxes[:, 1].min()),
            int(np.ceil(boxes[:, 2].max())),
            int(np.ceil(boxes[:, 3].max())),
        )
        if box.height < MIN_ELONGATION * max(box.width, 1):
            continue
        chain_boxes.append(box)
    if not chain_boxes:
        return []
    glyph_width = float(np.median(stats.widths)) if len(stats) else mm_to_px(2.0, dpi)
    glyph_height = float(np.median(stats.heights)) if len(stats) else mm_to_px(2.0, dpi)
    gap = MERGE_GAP_GLYPHS * glyph_width
    along_gap = ALONG_GAP_GLYPHS * glyph_height
    # Склейка цепочек: куски одной строки через пробел (на одной вертикали, зазор мал) и
    # соседние строки одной боковой подписи (бок о бок, перекрываются по высоте).
    merge_edges = []
    for a in range(len(chain_boxes)):
        for b in range(a + 1, len(chain_boxes)):
            first, second = chain_boxes[a], chain_boxes[b]
            horizontal_gap = max(second.x0 - first.x1, first.x0 - second.x1)
            overlap = min(first.y1, second.y1) - max(first.y0, second.y0)
            if horizontal_gap <= gap and overlap >= MERGE_OVERLAP * min(first.height, second.height):
                merge_edges.append((a, b))
                continue
            x_overlap = min(first.x1, second.x1) - max(first.x0, second.x0)
            vertical_gap = max(second.y0 - first.y1, first.y0 - second.y1)
            if x_overlap < ALONG_OVERLAP * min(first.width, second.width):
                continue
            if vertical_gap <= along_gap:
                merge_edges.append((a, b))
            elif vertical_gap <= BRIDGED_GAP_GLYPHS * glyph_height and _glyph_in_gap(stats, first, second):
                merge_edges.append((a, b))
    zones: list[Box] = []
    for group in _components(len(chain_boxes), np.array(merge_edges, int).reshape(-1, 2)):
        members = [chain_boxes[i] for i in group]
        zones.append(
            Box(
                min(m.x0 for m in members),
                min(m.y0 for m in members),
                max(m.x1 for m in members),
                max(m.y1 for m in members),
            )
        )
    return sorted(zones, key=lambda z: (z.y0, z.x0))


def transposed(stats: GlyphStats) -> GlyphStats:
    """Те же компоненты в координатах ``(y, x)``: горизонтальные строки становятся вертикальными.

    Args:
        stats: Компоненты страницы.

    Returns:
        Статистика с переставленными осями (центры, рамки, ширины и высоты).
    """
    centers = stats.centers[:, ::-1].copy() if len(stats) else stats.centers
    boxes = stats.boxes[:, [1, 0, 3, 2]].copy() if len(stats) else stats.boxes
    return GlyphStats(centers, boxes, stats.widths.copy(), stats.heights.copy())


def cluster_lines(stats: GlyphStats, dpi: int, vertical: bool = True, reach: "float | None" = None) -> list[Box]:
    """Зоны текста вдоль одной оси: боковые (``vertical``) или прямые строки и подписи.

    Args:
        stats: Компоненты страницы (:func:`glyph_components`).
        dpi: Разрешение растра.
        vertical: True — цепочки букв друг над другом (боковой текст); False — слева направо (прямой).
        reach: Порог соседства в высотах глифа; None — ``NEIGHBOR_REACH`` для боковых, ``UPRIGHT_REACH`` для прямых.

    Returns:
        Рамки зон в пикселях растра.
    """
    if vertical:
        return cluster_rotated(stats, dpi, NEIGHBOR_REACH if reach is None else reach)
    reach = UPRIGHT_REACH if reach is None else reach
    return sorted(
        (Box(b.y0, b.x0, b.y1, b.x1) for b in cluster_rotated(transposed(stats), dpi, reach)),
        key=lambda z: (z.y0, z.x0),
    )
