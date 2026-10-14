"""Защитный зазор вокруг печати своей полосы: четырёхугольник его охватывает, вне листа он закрашивается ровным цветом бумаги (Telea не тянет краску букв)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.glyphs import CLOSE_PX, FINGER_MIN_GLYPHS, GUARD_PX, INK_BLOCK, INK_OFFSET
from ocr_utils.scan_cropping.v2.glyphs import finger_protect, guard_mask  # noqa: F401 — ядро, ре-экспорт для стенда
from research.newspaper_crop.ink_guard import text_lines
from research.newspaper_crop.zone import Zone, quad_mask, zone_of

# Глубина от края листа, глубже которой краска считается своей печатью, px рабочего масштаба: у самого края
# адаптивный порог ловит тень кромки и торец листа, а не буквы.
EDGE_PX = 3
# Окрестность строк набора, в которой краска считается печатью, px рабочего масштаба: боксы строк не доходят до
# букв у самой кромки загиба (их режет край листа), а бахрома переплёта лежит дальше.
LINE_REACH_PX = 8
# Наименьшая площадь пятна краски, px рабочего масштаба: мельче — сор и шум JPEG.
MIN_SPECK = 4


@dataclass(frozen=True)
class Guard:
    """Итог защиты кадра: зона с охватом зазора, маска «зазор вне листа» и сколько своей печати лежало в дорисовке."""

    zone: Zone
    keep: np.ndarray
    print_fill_before: float
    expand_px: float
    suspect_fill: float


def own_print(rgb: np.ndarray, paper: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Краска своей полосы: тёмные пятна в строках набора внутри листа и четырёхугольника, глубже :data:`EDGE_PX`
    от края листа.

    Чужой лист и соседняя полоса лежат вне маски листа или вне четырёхугольника (после отрезки за складкой), поэтому
    их печать сюда не попадает. Строки (``ink_guard.text_lines``: глифы по размеру, вытянутые сгустки) отсекают
    бахрому и тень у переплёта — крупные неровные пятна, которые адаптивный порог тоже видит тёмными.

    Args:
        rgb: Кадр RGB (любой масштаб — приводится к размеру маски).
        paper: Булева маска листа на рабочем масштабе.
        quad: Четырёхугольник зоны (пиксели маски).

    Returns:
        Булева маска своей печати на рабочем масштабе.
    """
    h, w = paper.shape
    gray = cv2.cvtColor(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    ink = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, INK_BLOCK, INK_OFFSET)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EDGE_PX + 1, 2 * EDGE_PX + 1))
    inner = cv2.erode(paper.astype(np.uint8), k) > 0
    reach = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * LINE_REACH_PX + 1, 2 * LINE_REACH_PX + 1))
    near_lines = cv2.dilate(text_lines(rgb, (h, w)).astype(np.uint8), reach) > 0
    ink = (ink > 0) & inner & quad_mask(quad, (h, w)) & near_lines
    # Сор и шум — мелкие пятна — не охраняются.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= MIN_SPECK
    keep[0] = False
    return keep[labels]






def expand_quad(quad: np.ndarray, pts: np.ndarray, frozen: tuple[int, ...] = ()) -> tuple[np.ndarray, float]:
    """Сдвигает стороны четырёхугольника наружу параллельно себе, пока все точки не окажутся внутри.

    Args:
        quad: Вершины (4, 2) по кругу.
        pts: Точки (N, 2), которые надо охватить.
        frozen: Номера сторон (сторона i — от вершины i к i+1), которые не двигаются.

    Returns:
        (новые вершины, наибольший сдвиг стороны в px).
    """
    centre = quad.mean(axis=0)
    lines, shift = [], 0.0
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = (b - a) / (np.linalg.norm(b - a) + 1e-12)
        line = np.array([-d[1], d[0], 0.0])
        line[2] = -float(line[:2] @ a)
        # Нормаль — наружу (от центра).
        if line[:2] @ centre + line[2] > 0:
            line = -line
        if len(pts) and i not in frozen:
            out = float((pts @ line[:2] + line[2]).max())
            if out > 0:
                line[2] -= out
                shift = max(shift, out)
        lines.append(line)
    corners = []
    for i in range(4):
        p = np.cross(lines[i - 1], lines[i])
        corners.append(p[:2] / p[2])
    # Порядок вершин: вершина i — пересечение сторон i−1 и i (сторона i идёт от вершины i к i+1).
    return np.array(corners), shift


def side_index(quad: np.ndarray, name: str) -> int:
    """Номер стороны четырёхугольника по имени ``left`` / ``right`` / ``top`` / ``bottom`` (по положению середины).

    Args:
        quad: Вершины (4, 2) по кругу.
        name: Имя стороны (как ``SpineSide.value``).

    Returns:
        Номер стороны (от вершины i к i+1).
    """
    mids = np.array([(quad[i] + quad[(i + 1) % 4]) / 2 for i in range(4)])
    pick = {"left": lambda: np.argmin(mids[:, 0]), "right": lambda: np.argmax(mids[:, 0]),
            "top": lambda: np.argmin(mids[:, 1]), "bottom": lambda: np.argmax(mids[:, 1])}  # fmt: skip
    return int(pick[name]())


def protect(rgb: np.ndarray, zone: Zone, guard_px: int = GUARD_PX, frozen: tuple[int, ...] = ()) -> Guard:
    """Защита своей печати в зоне: четырёхугольник охватывает зазор, зазор вне листа исключается из дорисовки.

    Args:
        rgb: Кадр RGB.
        zone: Зона на рабочем масштабе (после отрезки за складкой).
        guard_px: Ширина зазора, px рабочего масштаба.
        frozen: Стороны, отрезанные по складке корешка (:func:`side_index`): их охват не двигает — за ними соседняя
            полоса, её печать у стороны не своя.

    Returns:
        :class:`Guard`; ``keep`` — зазор вне листа внутри четырёхугольника (исходные пиксели, не дорисовка).
    """
    if zone.quad is None:
        return Guard(zone, np.zeros_like(zone.paper), 0.0, 0.0, 0.0)
    own = own_print(rgb, zone.paper, zone.quad)
    guard = guard_mask(own, guard_px)
    h, w = zone.paper.shape
    # Мера «было»: доля своей печати, к которой дорисовка подходит ближе зазора.
    fill_before = quad_mask(zone.quad, (h, w)) & ~zone.paper
    near = guard_mask(fill_before, guard_px, 0) & own
    total = int(own.sum())
    before = float(near.sum() / total) if total else 0.0
    # Подозрение «свои строки ушли в дорисовку»: краска дорисовки внутри сомкнутых колонок своей печати (без зазора) —
    # движок не взял кусок своей бумаги с текстом (или рядом торчит чужой лист — разбирать глазами).
    ink_all = (
        cv2.adaptiveThreshold(
            cv2.cvtColor(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY),
            1,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            INK_BLOCK,
            INK_OFFSET,
        )
        > 0
    )
    columns = guard_mask(own, 0)
    suspect = float((ink_all & fill_before & columns).sum() / total) if total else 0.0
    # Охват: точки зазора внутри кадра, которые вылезли за четырёхугольник.
    outside = guard & ~quad_mask(zone.quad, (h, w))
    pts = np.argwhere(outside)[:, ::-1].astype(np.float64)
    # Своя печать внутри четырёхугольника, поэтому сдвиг стороны охватом не больше зазора (плюс огрехи смыкания).
    quad, shift = expand_quad(zone.quad, pts, frozen) if len(pts) else (zone.quad, 0.0)
    keep = guard & ~zone.paper & quad_mask(quad, (h, w))
    return Guard(zone_of(zone.paper, quad), keep, before, shift, suspect)


def paint_margin(bgr: np.ndarray, region: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """Закрашивает зазор вне листа ровным цветом бумаги рядом: своей печати в нём нет (она всегда в листе), а тень и
    чужой лист у кромки, оставленные как есть или высветленные, дают пятна; Telea, начатая от букв, тянет краску
    в заливку — ровное поле этого не даёт.

    Цвет — медиана светлой половины пикселей листа в кольце у зазора (буквы и тень отбрасываются).

    Args:
        bgr: Картинка BGR.
        region: Булева маска зазора вне листа того же размера.
        paper: Булева маска листа того же размера.

    Returns:
        Картинка BGR; вне ``region`` — без изменений.
    """
    if not region.any():
        return bgr
    ring = cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0
    near = ring & paper & ~region
    pool = bgr[near] if near.sum() > 50 else bgr[paper]
    light = pool[pool.sum(axis=1) >= np.median(pool.sum(axis=1))]
    out = bgr.copy()
    out[region] = np.median(light, axis=0).astype(np.uint8)
    return out
