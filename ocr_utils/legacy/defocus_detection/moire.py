"""Основной метод детектора — энергия растрового муара (+ нормировка A, гейт B, зона C).

ИДЕЯ. Газетный полутоновый растр при нормальном фокусе порождает сильный муар, если
уменьшить кадр БЕЗ сглаживания (именно так дефект ловят глазами в XNView). Меряем муар
как std разницы между уменьшением методом NEAREST (даёт алиасинг/муар) и AREA (усредняет,
муара нет): структура текста/фото есть в обоих уменьшениях и сокращается — остаётся чистая
энергия растрового муара. Растр есть везде, где лежит краска (включая фото), поэтому в
фокусе даже фото-тайлы «горячие»; низкий муар на ПЕЧАТНОМ тайле = расфокус.

Доработки (см. defocus_moire_improvement_plan.md):
- A (gradient_tile_map + нормировка в pipeline): делим муар тайла на меру «сколько в тайле
  резких переходов», чтобы метрика отражала фокус, а не количество краски на полосе.
- B (raster_edge_density): отсев обложек/пустых листов без типографского растра.
- C (find_defocus_zone): подозрение по связной 2D-зоне мягких тайлов, а не по всей полосе.
"""

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.legacy.defocus_detection.grid import tile_bounds


def center_std(gray: np.ndarray) -> float:
    """Вычисляет std центрального кропа (50%) изображения.

    Используется для нормировки метрик на общий динамический диапазон изображения:
    сканы с разной экспозицией/контрастом имеют разный базовый уровень муара даже
    при одинаковом фокусе.

    Args:
        gray: Полутоновое изображение.

    Returns:
        Std центрального кропа (50% от размера по каждой оси).
    """
    h, w = gray.shape
    crop_h, crop_w = h // 4, w // 4
    center_crop = gray[crop_h : 3 * crop_h, crop_w : 3 * crop_w]
    return float(center_crop.std())


def moire_tile_maps(gray: np.ndarray, factor: float, grid_x: int, grid_y: int) -> tuple[np.ndarray, np.ndarray]:
    """Считает по сетке карту энергии муара и карту контраста (наличия краски).

    Args:
        gray: Полутоновое изображение.
        factor: Во сколько раз уменьшать кадр перед измерением муара.
        grid_x: Число тайлов по горизонтали.
        grid_y: Число тайлов по вертикали.

    Returns:
        Кортеж (moire, structure) — два массива shape (grid_y, grid_x):
        moire — std разницы NEAREST−AREA в тайле (энергия муара),
        structure — std AREA-уменьшения в тайле (мера наличия печатного контента).
    """
    g = gray.astype(np.float32)
    h, w = g.shape
    nw, nh = max(1, int(w / factor)), max(1, int(h / factor))
    nn = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ar = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    diff = nn - ar

    moire = np.zeros((grid_y, grid_x), dtype=np.float64)
    structure = np.zeros((grid_y, grid_x), dtype=np.float64)
    for ry in range(grid_y):
        y1, y2 = tile_bounds(nh, grid_y, ry)
        for rx in range(grid_x):
            x1, x2 = tile_bounds(nw, grid_x, rx)
            moire[ry, rx] = diff[y1:y2, x1:x2].std()
            structure[ry, rx] = ar[y1:y2, x1:x2].std()
    return moire, structure


def gradient_tile_map(gray: np.ndarray, grid_x: int, grid_y: int) -> np.ndarray:
    """Карта полноразмерной энергии градиента (Tenengrad, RMS) по тайлам.

    Отражает, «сколько в тайле было резких переходов» на нативном превью ДО уменьшения.
    Используется двояко: (A2) как знаменатель нормировки муара; (C) как гейт «реально ли
    тайл размыт» — у резкого НЕрастрового края (тёмная полоса/сгиб/граница страница-стол)
    низкий муар, но ВЫСОКИЙ градиент, и это не расфокус.

    Args:
        gray: Полутоновое изображение (нативное превью, без уменьшения).
        grid_x: Число тайлов по горизонтали.
        grid_y: Число тайлов по вертикали.

    Returns:
        Массив shape (grid_y, grid_x): RMS градиента Собеля в каждом тайле.
    """
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy
    h, w = g.shape
    grad = np.zeros((grid_y, grid_x), dtype=np.float64)
    for ry in range(grid_y):
        y1, y2 = tile_bounds(h, grid_y, ry)
        for rx in range(grid_x):
            x1, x2 = tile_bounds(w, grid_x, rx)
            grad[ry, rx] = float(np.sqrt(energy[y1:y2, x1:x2].mean()))
    return grad


def raster_edge_density(gray: np.ndarray) -> float:
    """Доля краёв (Canny) на изображении — прокси наличия типографского растра (этап B).

    Газетная полоса (даже малотекстовая или расфокусная) усыпана краями текста/растра,
    тогда как обложка/картонный переплёт/пустой лист (дерево, картон, рукописная этикетка)
    почти гладкие. Очень низкая плотность краёв → это не полоса, а обложка/пусто, и такой
    файл нельзя ранжировать вместе с полосами (отсекается гейтом обложек в pipeline/CLI).

    Args:
        gray: Полутоновое изображение (нативное превью).

    Returns:
        Доля пикселей-краёв в [0, 1].
    """
    return float(cv2.Canny(gray, 50, 150).mean() / 255.0)


def find_defocus_zone(
    ratio: np.ndarray,
    structure: np.ndarray,
    grad: np.ndarray,
    min_structure: float,
    margin: int,
    k_abs: float,
    k_rel: float,
    g_rel: float,
    min_rows: int,
    min_cols: int,
) -> tuple[dict | None, float]:
    """Ищет связную 2D-зону расфокуса по карте нормированного муара (этап C).

    Тайл считается «мягким» (кандидат в зону расфокуса), если одновременно:
      * ratio < k_abs — абсолютный обвал нормированного муара;
      * ratio < k_rel · медиана(ratio полосы) — обвал относительно своей же полосы;
      * grad < g_rel · медиана(grad полосы) — тайл реально размыт (низкий градиент),
        а не резкий НЕрастровый край (тёмная полоса/сгиб/граница страницы дают низкий
        ratio при высоком градиенте — это не расфокус).
    Краевые тайлы (поля шириной `margin`) исключаются: там лежит граница страница/стол.
    Подозрением считается лишь СВЯЗНАЯ компонента мягких тайлов размером ≥ min_rows строк
    И ≥ min_cols столбцов — это отсекает одно-строчные/столбцовые артефакты контента
    (крупные заголовки, фото, сгиб).

    Args:
        ratio: Карта нормированного муара (grid_y, grid_x).
        structure: Карта контраста (мера наличия печатного контента).
        grad: Карта RMS градиента по тайлам.
        min_structure: Порог контраста, ниже — пустое поле (тайл не рассматривается).
        margin: Сколько крайних рядов/столбцов тайлов исключить (поля страницы).
        k_abs: Абсолютный порог «мягкости» нормированного муара.
        k_rel: Доля медианы полосы, ниже которой тайл «мягкий».
        g_rel: Доля медианы градиента полосы, ниже которой тайл считается размытым.
        min_rows: Минимальная высота зоны в тайлах.
        min_cols: Минимальная ширина зоны в тайлах.

    Returns:
        Кортеж (zone, inner_median):
        zone — словарь найденной зоны (depth, size, rows, cols, bbox) или None;
        inner_median — медиана ratio по внутренним печатным тайлам («здоровый» уровень полосы).
    """
    gy, gx = ratio.shape
    inner = np.zeros((gy, gx), dtype=bool)
    inner[margin : gy - margin, margin : gx - margin] = True
    valid = (structure > min_structure) & inner
    if valid.sum() < 10:
        return None, float("nan")

    med = float(np.median(ratio[valid]))
    gmed = float(np.median(grad[valid]))
    soft = valid & (ratio < k_abs) & (ratio < k_rel * med) & (grad < g_rel * gmed)
    if not soft.any():
        return None, med

    labels, n = ndimage.label(soft)
    best: dict | None = None
    for i in range(1, n + 1):
        comp = labels == i
        ys, xs = np.where(comp)
        rows = int(ys.max() - ys.min() + 1)
        cols = int(xs.max() - xs.min() + 1)
        if rows >= min_rows and cols >= min_cols:
            cand = dict(
                depth=float(ratio[comp].mean()),
                size=int(comp.sum()),
                rows=rows,
                cols=cols,
                bbox=(int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
            )
            # Берём самую крупную подходящую зону.
            if best is None or cand["size"] > best["size"]:
                best = cand
    return best, med
