"""Доводка края маски бумаги SAM 3 наружу до настоящей кромки там, где за кромкой сразу фон (``snap_bg``)."""

from __future__ import annotations

import cv2
import numpy as np

# Полоса поиска кромки наружу от границы маски, px масштаба сегментации (замер на 11/IMG_0647: сдвиг 5–9 px).
SNAP_BAND = 12
# Пиксель полосы — ещё бумага (ободок края), если он не темнее медианы бумаги у края больше чем на SNAP_DROP уровней
# серого и близок к ней по оттенку (ab Lab) не дальше SNAP_AB.
SNAP_DROP = 35
SNAP_AB = 8.0
# Опорная бумага: кольцо внутри маски на этой глубине от края, px.
REF_NEAR, REF_FAR = 10, 30
# Окно усреднения опорной бумаги вдоль края, px.
REF_WINDOW = 151
# Снап «только у фона»: точка края двигается, только если в её секторе фон начинается не дальше этого, px
# (за белой стопкой и торцом фона рядом нет — там край не трогаем).
SNAP_BG_REACH = 15
# Окно медианы, сглаживающей дальность роста вдоль края, px (нечётное).
REACH_SMOOTH = 11


def snap_outward(
    bgr: np.ndarray,
    paper: np.ndarray,
    book: np.ndarray,
    bg: np.ndarray,
    band: int = SNAP_BAND,
    bg_reach: int | None = None,
) -> np.ndarray:
    """Доводка по кромке: каждая точка края переносится наружу до самой дальней «бумаги» в полосе ``band`` px.

    Для каждого пикселя кольца вокруг маски известна ближайшая точка края (метки преобразования расстояний). Для точки
    края берётся наибольшее расстояние, на котором в её секторе ещё есть бумага (светлый пиксель того же оттенка, не
    фон); все пиксели сектора ближе этого расстояния добавляются — так тёмная полоса тени у края, лежащая между маской
    и светлым ободком, тоже входит.

    Args:
        bgr: Кадр BGR (масштаб сегментации).
        paper: Маска бумаги.
        book: Маска книги.
        bg: Маска фона.
        band: Ширина полосы поиска, px.
        bg_reach: Если задан — двигаются только точки края, в секторе которых фон (``bg`` или вне книги) начинается
            не дальше ``bg_reach`` px; остальные остаются на месте. ``None`` — без ограничения.

    Returns:
        Маска бумаги, доведённая до кромки.
    """
    lab = cv2.cvtColor(cv2.medianBlur(bgr, 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inside = cv2.distanceTransform(paper.astype(np.uint8), cv2.DIST_L2, 5)
    ring_in = paper & (inside >= REF_NEAR) & (inside <= REF_FAR)
    # Опора — локальная: средние по окну только по кольцу бумаги у края.
    k = (REF_WINDOW, REF_WINDOW)
    cnt = cv2.boxFilter(ring_in.astype(np.float32), -1, k, normalize=False)
    ref_g = cv2.boxFilter(gray * ring_in, -1, k, normalize=False) / np.maximum(cnt, 1)
    ref_a = cv2.boxFilter(lab[..., 1] * ring_in, -1, k, normalize=False) / np.maximum(cnt, 1)
    ref_b = cv2.boxFilter(lab[..., 2] * ring_in, -1, k, normalize=False) / np.maximum(cnt, 1)
    light = (gray >= ref_g - SNAP_DROP) & (np.hypot(lab[..., 1] - ref_a, lab[..., 2] - ref_b) <= SNAP_AB) & (cnt > 20)
    # Расстояние до маски и метка ближайшей точки края для каждого пикселя снаружи.
    dist, labels = cv2.distanceTransformWithLabels(
        (~paper).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    ring = (~paper) & (dist <= band) & book & ~bg
    cand = ring & light
    # Дальность роста точки края — самая дальняя бумага в её секторе.
    reach = np.zeros(labels.max() + 1, np.float32)
    np.maximum.at(reach, labels[cand], dist[cand])
    if bg_reach is not None:
        # Для каждой точки края — ближайший фон в её секторе; точки без фона рядом не двигаются.
        outside = (~paper) & (dist <= bg_reach) & (bg | ~book)
        near_bg = np.full(labels.max() + 1, np.inf, np.float32)
        np.minimum.at(near_bg, labels[outside], dist[outside])
        reach = np.where(np.isfinite(near_bg), reach, 0)
    # Дальность роста соседних точек края разная (шум яркости, текстура ободка) — край выходил зубчатым и с петлями.
    # Сглаживание медианой: карта дальности ближайшей точки края, затем медиана.
    reach_map = np.clip(reach[labels], 0, 255).astype(np.uint8)
    reach_map = cv2.medianBlur(reach_map, REACH_SMOOTH)
    grow = ring & (dist <= reach_map)
    # Рост — только связный с маской: оторванные островки не берём.
    out = paper | grow
    _, comp = cv2.connectedComponents(out.astype(np.uint8))
    keep = np.unique(comp[paper])
    return np.isin(comp, keep[keep > 0])
