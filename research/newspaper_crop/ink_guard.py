"""Строки печати по краске (рецепт ``page_layout.orientation.detectors.ink_axis``): движок «оболочка печати» и мера «строки в дорисовке»."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.orientation.detectors.ink_axis import glyph_mask, line_boxes
from research.newspaper_crop.zone import Zone, quad_mask, work_shape_of

# Окно и сдвиг адаптивного порога краски, px рабочего масштаба (~120 dpi у превью ЭГ) и уровни серого:
# фон кадра (пол, мрамор крышки) неоднороден, глобальный Оцу из ink_axis на кадре с фоном не годится.
ADAPTIVE_BLOCK = 31
ADAPTIVE_OFFSET = 18
# Смыкание строк в «печать полосы»: зазор между колонками газеты на рабочем масштабе ~15–25 px,
# 41 px склеивает колонки одной полосы, но не перескакивает поле и корешок до соседней полосы.
CLUSTER_CLOSE_PX = 41
# Наименьший сгусток печати относительно крупнейшего: блоки полосы, разделённые широкой линейкой или
# заголовком, остаются, мелкие обрывки (номер страницы соседней полосы, сор) — нет.
MIN_CLUSTER_SHARE = 0.1
# Поле вокруг оболочки печати у движка «текст-первым», px рабочего масштаба (~8 мм бумаги).
HULL_MARGIN_PX = 40


def text_lines(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Маска «строк» печати на рабочем масштабе: глифы по размеру, смыкание вдоль и поперёк, вытянутые сгустки.

    Args:
        rgb: Кадр RGB.
        shape: Рабочий (H, W).

    Returns:
        Булева маска прямоугольников строк (обе ориентации набора).
    """
    gray = cv2.cvtColor(cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK, ADAPTIVE_OFFSET
    )
    # glyph_mask сам бинаризует Оцу по серому: подаём ему уже чистую краску (инвертированную в «чёрное на белом»).
    glyphs = glyph_mask(255 - ink)
    out = np.zeros(shape, bool)
    for horizontal in (True, False):
        boxes, _ = line_boxes(glyphs, horizontal)
        for x, y, w, h in boxes:
            out[y : y + h, x : x + w] = True
    return out


def print_cluster(lines: np.ndarray) -> np.ndarray:
    """Печать полосы: сгустки строк после смыкания колонок, не меньше :data:`MIN_CLUSTER_SHARE` крупнейшего.

    Строки, касающиеся рамки кадра, отбрасываются: это тени и торцы края листа, а не набор.

    Args:
        lines: Маска строк.

    Returns:
        Булева маска строк печати.
    """
    h, w = lines.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lines.astype(np.uint8), connectivity=8)
    x0, y0 = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    x1, y1 = x0 + stats[:, cv2.CC_STAT_WIDTH], y0 + stats[:, cv2.CC_STAT_HEIGHT]
    border = (x0 <= 1) | (y0 <= 1) | (x1 >= w - 1) | (y1 >= h - 1)
    border[0] = True
    lines = lines & ~border[labels]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (CLUSTER_CLOSE_PX, CLUSTER_CLOSE_PX))
    merged = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE, k)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    if count <= 1:
        return np.zeros_like(lines)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(count, bool)
    keep[1:] = areas >= MIN_CLUSTER_SHARE * areas.max()
    return keep[labels] & lines


def ink_hull_mask(rgb: np.ndarray) -> np.ndarray:
    """Движок «текст-первым»: выпуклая оболочка печати полосы с полем :data:`HULL_MARGIN_PX`.

    Args:
        rgb: Кадр превью RGB.

    Returns:
        Булева маска на рабочем масштабе (это зона печати, а не бумаги: край листа не ищется).
    """
    shape = work_shape_of(rgb)
    cluster = print_cluster(text_lines(rgb, shape))
    out = np.zeros(shape, np.uint8)
    pts = np.argwhere(cluster)[:, ::-1].astype(np.int32)
    if len(pts) < 3:
        return out.astype(bool)
    cv2.fillPoly(out, [cv2.convexHull(pts)], 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HULL_MARGIN_PX + 1, 2 * HULL_MARGIN_PX + 1))
    return cv2.dilate(out, k) > 0


def print_guard(zone: Zone, cluster: np.ndarray) -> tuple[float, float]:
    """Страж печати: доли строк печати полосы вне четырёхугольника и в зоне дорисовки.

    Args:
        zone: Зона кадра.
        cluster: Маска печати полосы (:func:`print_cluster`).

    Returns:
        (строки вне четырёхугольника — срез печати, строки в дорисовке — печать, которую затрёт заливка);
        доли от всей печати сгустка.
    """
    total = int(cluster.sum())
    if not total or zone.quad is None:
        return 0.0, 0.0
    qm = quad_mask(zone.quad, zone.paper.shape)
    fill = qm & ~zone.paper
    return float((cluster & ~qm).sum() / total), float((cluster & fill).sum() / total)
