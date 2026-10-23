"""Нежёсткое совмещение страницы кадра-донора (2) с кадром основы (1): гомография по SIFT, DIS-поток, продление TPS за край видимой бумаги."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

# Число точек SIFT на кадр (масштаб совмещения ½ кадра (1)).
SIFT_POINTS = 30000
# Порог Лоу и допуск RANSAC гомографии, px: страница изогнута, гомография — только грубое начало.
LOWE_RATIO = 0.8
RANSAC_PX = 25.0
# Меньше инлайеров — пара страниц не совмещается (чужой разворот, пустая страница).
MIN_INLIERS = 40
# Масштабы запасного поиска совпадений для расфокусированной страницы (доли масштаба совмещения).
FALLBACK_SCALES = (0.5, 0.25)
# Размытие перед DIS, px; число проходов DIS (второй проход уточняет после первого).
FLOW_BLUR = 1.5
FLOW_PASSES = 2
# Надёжность поля: окно нормированной корреляции, порог корреляции, минимальная дисперсия (есть краска), отступ
# от края видимой бумаги, px.
NCC_WINDOW = 21
NCC_MIN = 0.6
VAR_MIN = 100.0
EDGE_MARGIN = 7
# Продление поля: шаг выборки надёжных точек, их предел, сглаживание TPS (координаты в сотнях px), шаг сетки
# вычисления; переход от измеренного поля к продлённому на этой глубине от края надёжной зоны, px.
SAMPLE_STEP = 12
SAMPLE_MAX = 3000
TPS_SMOOTHING = 30.0
GRID_STEP = 8
BLEND_PX = 20.0
# Поле вокруг видимой бумаги, где ещё считается поток, px.
FLOW_PAD = 15


@dataclass(frozen=True)
class PageMap:
    """Отображение страницы: для каждого пикселя (1) — координаты в (2) на масштабе совмещения.

    Поля:
        map_x: Абсцисса в (2), float32 (H, W) масштаба (1).
        map_y: Ордината в (2).
        reliable: Где поле измерено надёжно (видимая бумага, совпадение по корреляции).
        inliers: Число инлайеров гомографии.
        ncc: Медиана корреляции серого в надёжной зоне.
    """

    map_x: np.ndarray
    map_y: np.ndarray
    reliable: np.ndarray
    inliers: int
    ncc: float


def local_ncc(a: np.ndarray, b: np.ndarray, window: int = NCC_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Нормированная корреляция двух серых картинок в скользящем окне и дисперсия первой.

    Args:
        a: Первая картинка (uint8 или float).
        b: Вторая картинка того же размера.
        window: Сторона окна, px.

    Returns:
        ``(корреляция, дисперсия a)`` — карты float32.
    """
    a, b = a.astype(np.float32), b.astype(np.float32)
    k = (window, window)
    ma, mb = cv2.blur(a, k), cv2.blur(b, k)
    cov = cv2.blur(a * b, k) - ma * mb
    va, vb = cv2.blur(a * a, k) - ma**2, cv2.blur(b * b, k) - mb**2
    return cov / np.sqrt(np.maximum(va * vb, 1e-3)), va


def match_homography(
    gray_a: np.ndarray, mask_a: np.ndarray, points_b: np.ndarray, desc_b: np.ndarray, ransac_px: float
) -> tuple[np.ndarray, int] | None:
    """Гомография по совпадениям SIFT точек (1) внутри маски с готовыми признаками (2).

    Args:
        gray_a: Серое (1).
        mask_a: Где искать точки (bool).
        points_b: Точки (2) (n, 2).
        desc_b: Дескрипторы (2).
        ransac_px: Допуск RANSAC, px.

    Returns:
        ``(матрица 3×3, число инлайеров)`` или None.
    """
    kp_a, desc_a = cv2.SIFT_create(nfeatures=SIFT_POINTS).detectAndCompute(gray_a, mask_a.astype(np.uint8))
    if desc_a is None or len(kp_a) < 10 or desc_b is None:
        return None
    matches = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < LOWE_RATIO * m[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    pa = np.float32([kp_a[m.queryIdx].pt for m in good])
    pb = np.float32([points_b[m.trainIdx] for m in good])
    hom, mask = cv2.findHomography(pa, pb, cv2.RANSAC, ransac_px)
    if hom is None or int(mask.sum()) < MIN_INLIERS:
        return None
    return hom, int(mask.sum())


def homography_map(
    gray_a: np.ndarray, gray_b: np.ndarray, mask_a: np.ndarray, sift_b: tuple
) -> tuple[np.ndarray, int] | None:
    """Гомография (1)→(2) по совпадениям SIFT внутри маски страницы (1), с запасным путём для расфокуса.

    Кадры (1) сняты «по страницам»: в фокусе одна страница разворота, вторая размыта, и SIFT на ней почти не находит
    совпадений с резким донором. Тогда совпадения ищутся на уменьшенных копиях обоих кадров
    (:data:`FALLBACK_SCALES`), где расфокус почти не виден, и гомография пересчитывается в масштаб совмещения.

    Args:
        gray_a: Серое (1).
        gray_b: Серое (2).
        mask_a: Маска страницы (1), где искать точки (bool).
        sift_b: ``(точки, дескрипторы)`` SIFT кадра (2) на масштабе совмещения.

    Returns:
        ``(матрица 3×3, число инлайеров)`` или None, если совпадений мало на всех масштабах.
    """
    found = match_homography(gray_a, mask_a, sift_b[0], sift_b[1], RANSAC_PX)
    if found is not None:
        return found
    for scale in FALLBACK_SCALES:
        small_a = cv2.resize(gray_a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small_m = cv2.resize(
            mask_a.astype(np.uint8), (small_a.shape[1], small_a.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        small_b = cv2.resize(gray_b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        kp_b, desc_b = cv2.SIFT_create(nfeatures=SIFT_POINTS).detectAndCompute(small_b, None)
        if desc_b is None:
            continue
        found = match_homography(small_a, small_m > 0, np.float32([k.pt for k in kp_b]), desc_b, RANSAC_PX * scale)
        if found is not None:
            # H_small: a·s → b·s; в масштабе совмещения: S⁻¹·H_small·S.
            to_small = np.diag([scale, scale, 1.0])
            return np.linalg.inv(to_small) @ found[0] @ to_small, found[1]
    return None


def flow_refine(
    gray_a: np.ndarray, gray_b: np.ndarray, page: np.ndarray, map_x: np.ndarray, map_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Уточнение отображения DIS-потоком между (1) и выровненным (2) в пределах страницы.

    Args:
        gray_a: Серое (1).
        gray_b: Серое (2).
        page: Маска страницы (1); вне неё обе картинки белые (поток не тянется к чужому).
        map_x: Текущее отображение по x.
        map_y: Текущее отображение по y.

    Returns:
        Уточнённые ``(map_x, map_y)``.
    """
    h, w = gray_a.shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fa = cv2.GaussianBlur(np.where(page, gray_a, 255).astype(np.uint8), (0, 0), FLOW_BLUR)
    for _ in range(FLOW_PASSES):
        warped = cv2.remap(gray_b, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)
        fb = cv2.GaussianBlur(np.where(page, warped, 255).astype(np.uint8), (0, 0), FLOW_BLUR)
        flow = dis.calc(fa, fb, None)
        # Композиция: пиксель (1) → выровненное (2) (поток) → (2) (прежнее отображение в смещённой точке).
        sx, sy = gx + flow[..., 0], gy + flow[..., 1]
        map_x = cv2.remap(map_x, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        map_y = cv2.remap(map_y, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return map_x, map_y


def extend_field(
    base_x: np.ndarray, base_y: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, reliable: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Продление отображения за край надёжной зоны гладким полем TPS.

    Поправка к гомографии (``map − base``) выбирается по сетке в надёжной зоне, приближается TPS со сглаживанием и
    вычисляется на всей странице. Внутри надёжной зоны остаётся измеренное поле, к её краю оно плавно переходит в
    продлённое: срезанная строка за кромкой (1) изгибается и тянется так же, как соседние видимые строки.

    Args:
        base_x: Гомография по x на сетке пикселей (1).
        base_y: Гомография по y.
        map_x: Измеренное отображение по x.
        map_y: Измеренное отображение по y.
        reliable: Надёжная зона (bool).

    Returns:
        ``(map_x, map_y)`` с продлением.
    """
    h, w = reliable.shape
    dx, dy = map_x - base_x, map_y - base_y
    ys, xs = np.nonzero(reliable[::SAMPLE_STEP, ::SAMPLE_STEP])
    ys, xs = ys * SAMPLE_STEP, xs * SAMPLE_STEP
    if len(xs) < 20:
        return map_x, map_y
    pts = np.c_[xs, ys].astype(np.float64)
    val = np.c_[dx[ys, xs], dy[ys, xs]].astype(np.float64)
    if len(pts) > SAMPLE_MAX:
        sel = np.random.default_rng(0).choice(len(pts), SAMPLE_MAX, replace=False)
        pts, val = pts[sel], val[sel]
    # Координаты в сотнях px: сглаживание TPS не зависит от размера кадра так сильно.
    tps = RBFInterpolator(pts / 100.0, val, kernel="thin_plate_spline", smoothing=TPS_SMOOTHING)
    sy, sx = np.mgrid[0:h:GRID_STEP, 0:w:GRID_STEP]
    fit = tps(np.c_[sx.ravel(), sy.ravel()] / 100.0)
    ex = cv2.resize(fit[:, 0].reshape(sx.shape).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    ey = cv2.resize(fit[:, 1].reshape(sx.shape).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    # Вес измеренного поля: 1 в глубине надёжной зоны, 0 вне её.
    wgt = np.clip(cv2.distanceTransform(reliable.astype(np.uint8), cv2.DIST_L2, 5) / BLEND_PX, 0, 1)
    out_x = base_x + wgt * dx + (1 - wgt) * ex
    out_y = base_y + wgt * dy + (1 - wgt) * ey
    return out_x.astype(np.float32), out_y.astype(np.float32)


def register_page(
    gray_a: np.ndarray, gray_b: np.ndarray, page: np.ndarray, visible: np.ndarray, sift_b: tuple
) -> PageMap | None:
    """Совмещение страницы (2) с (1): гомография, DIS-поток, продление за край видимой бумаги.

    Args:
        gray_a: Серое (1) на масштабе совмещения.
        gray_b: Серое (2) на масштабе совмещения.
        page: Маска страницы (1) (bool).
        visible: Видимая бумага этой страницы (1) (bool).
        sift_b: Признаки SIFT кадра (2).

    Returns:
        Отображение или None, если страница не совмещается.
    """
    found = homography_map(gray_a, gray_b, visible, sift_b)
    if found is None:
        return None
    hom, inliers = found
    h, w = gray_a.shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = cv2.perspectiveTransform(np.dstack([gx, gy]).reshape(-1, 1, 2), hom).reshape(h, w, 2)
    base_x, base_y = base[..., 0].astype(np.float32), base[..., 1].astype(np.float32)
    # Поток — только по видимой бумаге страницы (с небольшим полем): палец и рука в основе, которых нет в доноре,
    # иначе тянут поток к себе, и строки у них разъезжаются (IMG_0639).
    flow_area = cv2.dilate(visible.astype(np.uint8), np.ones((FLOW_PAD, FLOW_PAD), np.uint8)) > 0
    map_x, map_y = flow_refine(gray_a, gray_b, flow_area, base_x, base_y)
    # Надёжно — внутри видимой бумаги (с отступом от кромки), где серое совпало и есть краска.
    warped = cv2.remap(gray_b, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)
    ncc, var = local_ncc(gray_a, warped)
    inner = cv2.erode(visible.astype(np.uint8), np.ones((2 * EDGE_MARGIN + 1,) * 2, np.uint8)) > 0
    reliable = inner & (ncc > NCC_MIN) & (var > VAR_MIN)
    map_x, map_y = extend_field(base_x, base_y, map_x, map_y, reliable)
    score = float(np.median(ncc[inner & (var > VAR_MIN)])) if (inner & (var > VAR_MIN)).any() else 0.0
    return PageMap(map_x, map_y, reliable, inliers, score)
