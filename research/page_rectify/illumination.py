"""Выравнивание освещения страницы — перенос Equalize illumination из ScanTailor Advanced (``EstimateBackground.cpp``, ``OutputGenerator.cpp`` ``RaiseAboveBackground``, ``AdjustBrightness.cpp``): фон — полином 8×5 по маске на копии 300 px, пиксель поднимается над фоном, цвет — сдвигом яркости."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import reconstruction

# Длинная сторона рабочей копии оценки фона, px (``reducedSize.scale(300, 300)``).
WORK_SIDE = 300
# Кирпич «закрытия» остатков букв на копии 300 px: 1 в ширину, 20 в высоту (``openGray(QSize(1, 20))``).
CLOSE_BRICK = (1, 20)
# Выбор способа предобработки: доля пикселей, где разность способов отходит от полинома 3×3 больше чем на 10 уровней;
# меньше 1 % — разность считается тенью (способ 1), иначе — рисунком (способ 2).
CHOICE_DEVIATION = 10
CHOICE_SHARE = 0.01
# Степени полиномов отсева: по столбцу — 2, по строке — 4; пиксель темнее подгонки больше чем на 30 уровней — не фон.
COLUMN_DEGREE = 2
ROW_DEGREE = 4
DARKER_BY = 30
# Строка или столбец, где годных пикселей меньше этой доли, выбрасываются целиком.
LINE_MIN_SHARE = 0.25
# Итоговая поверхность фона: степень по X и по Y.
SURFACE_DEGREES = (8, 5)
# Добавка к диагонали нормальных уравнений (``fixSquareMatrixRankDeficiency``).
RIDGE = 1e-5
# Веса яркости ScanTailor: R, G, B из 32 (``qGray`` в Qt — 11, 16, 5).
LUMA_WEIGHTS = np.array([5, 16, 11], np.float32) / 32  # порядок B, G, R


@dataclass(frozen=True)
class Background:
    """Оценка фона страницы.

    Поля:
        surface: Фон в размере входа, float32 0..255.
        work: Предобработанная копия 300 px, uint8 (для оверлея).
        mask: Пиксели копии, по которым подогнан полином (bool).
        method: Способ предобработки: 1 — восстановление от рамки (тень), 2 — по столбцам сверху и снизу (рисунок).
    """

    surface: np.ndarray
    work: np.ndarray
    mask: np.ndarray
    method: int


def luma(bgr: np.ndarray) -> np.ndarray:
    """Яркость ScanTailor ``(11R + 16G + 5B) / 32``, float32."""
    return bgr.astype(np.float32) @ LUMA_WEIGHTS


def poly_basis(width: int, height: int, deg_x: int, deg_y: int) -> np.ndarray:
    """Базис ``x^j · y^i`` на сетке ``width × height`` с координатами, сведёнными к ``[0, 1]``.

    Args:
        width: Ширина сетки.
        height: Высота сетки.
        deg_x: Степень по X.
        deg_y: Степень по Y.

    Returns:
        Матрица (height·width, (deg_y+1)·(deg_x+1)); порядок членов — Y снаружи, X внутри, как в ScanTailor.
    """
    xs = np.arange(width) / max(width - 1, 1)
    ys = np.arange(height) / max(height - 1, 1)
    px = xs[None, :] ** np.arange(deg_x + 1)[:, None]  # (deg_x+1, W)
    py = ys[None, :] ** np.arange(deg_y + 1)[:, None]  # (deg_y+1, H)
    # Член (i, j) на пикселе (y, x) = py[i, y] · px[j, x].
    terms = py[:, None, :, None] * px[None, :, None, :]  # (deg_y+1, deg_x+1, H, W)
    return terms.reshape((deg_y + 1) * (deg_x + 1), height * width).T


def fit_surface(values: np.ndarray, mask: np.ndarray | None, deg_x: int, deg_y: int) -> np.ndarray:
    """МНК-полином ``deg_x × deg_y`` по пикселям маски (``PolynomialSurface``), значения в долях 0..1.

    Если точек меньше, чем членов, степени снижаются по очереди, начиная с большей (``maybeReduceDegrees``); без
    точек — нулевая поверхность.

    Args:
        values: Картинка (H, W), уровни 0..255.
        mask: Пиксели подгонки или None — все.
        deg_x: Степень по X.
        deg_y: Степень по Y.

    Returns:
        Поверхность на сетке ``values`` (float64, уровни 0..255, без обрезки).
    """
    h, w = values.shape
    coeffs, deg_x, deg_y = surface_coeffs(values, mask, deg_x, deg_y)
    return (poly_basis(w, h, deg_x, deg_y) @ coeffs).reshape(h, w) * 255.0


def surface_coeffs(values: np.ndarray, mask: np.ndarray | None, deg_x: int, deg_y: int) -> tuple[np.ndarray, int, int]:
    """Коэффициенты МНК-полинома (см. :func:`fit_surface`) и степени после возможного снижения.

    Args:
        values: Картинка (H, W), уровни 0..255.
        mask: Пиксели подгонки или None — все.
        deg_x: Степень по X.
        deg_y: Степень по Y.

    Returns:
        ``(коэффициенты, deg_x, deg_y)``.
    """
    h, w = values.shape
    sel = np.ones(h * w, bool) if mask is None else mask.ravel()
    n = int(sel.sum())
    while (deg_x + 1) * (deg_y + 1) > n and (deg_x > 0 or deg_y > 0):
        if deg_x >= deg_y:
            deg_x -= 1
        else:
            deg_y -= 1
    terms = (deg_x + 1) * (deg_y + 1)
    if n == 0:
        return np.zeros(terms), deg_x, deg_y
    a = poly_basis(w, h, deg_x, deg_y)[sel]
    b = values.ravel()[sel].astype(np.float64) / 255.0
    ata = a.T @ a + RIDGE * np.eye(terms)
    return np.linalg.solve(ata, a.T @ b), deg_x, deg_y


def render_surface(coeffs: np.ndarray, deg_x: int, deg_y: int, width: int, height: int) -> np.ndarray:
    """Поверхность по коэффициентам на сетке ``width × height`` (координаты снова ``[0, 1]``), float32 0..255.

    Рендер идёт по строкам-блокам: базис всего кадра 4000×3500 × 54 члена — 6 ГБ.

    Args:
        coeffs: Коэффициенты (:func:`surface_coeffs`).
        deg_x: Степень по X.
        deg_y: Степень по Y.
        width: Ширина.
        height: Высота.

    Returns:
        Поверхность (height, width) float32, обрезанная в 0..255.
    """
    xs = np.arange(width) / max(width - 1, 1)
    ys = np.arange(height) / max(height - 1, 1)
    px = xs[None, :] ** np.arange(deg_x + 1)[:, None]  # (deg_x+1, W)
    py = ys[:, None] ** np.arange(deg_y + 1)[None, :]  # (H, deg_y+1)
    c = coeffs.reshape(deg_y + 1, deg_x + 1)
    # Σ_i Σ_j c[i,j]·y^i·x^j = py @ (c @ px) — без полного базиса.
    surface = py @ (c @ px) * 255.0
    return np.clip(surface, 0, 255).astype(np.float32)


def line_fit(values: np.ndarray, degree: int) -> np.ndarray:
    """МНК-полином степени ``degree`` по каждому столбцу ``values`` (``PolynomialLine``), уровни 0..255.

    Args:
        values: (N, M) — подгонка вдоль оси 0 для каждого из M столбцов.
        degree: Степень.

    Returns:
        Подгонка того же размера, округлённая и обрезанная в 0..255.
    """
    n = values.shape[0]
    t = np.arange(n) / max(n - 1, 1)
    a = t[:, None] ** np.arange(degree + 1)[None, :]
    ata = a.T @ a + RIDGE * np.eye(degree + 1)
    coeffs = np.linalg.solve(ata, a.T @ (values.astype(np.float64) / 255.0))
    return np.clip(np.round(a @ coeffs * 255.0), 0, 255)


def top_bottom_fill(img: np.ndarray) -> np.ndarray:
    """Способ 2: по каждому столбцу затравка — тёмные строки сверху и снизу, цвет растекается только по вертикали.

    Args:
        img: uint8 (H, W).

    Returns:
        uint8 (H, W): тёмное, не связанное по столбцу с верхом или низом, поднято до окружения.
    """
    top = np.maximum.accumulate(img, axis=0)
    out = np.empty_like(img)
    prev = np.zeros(img.shape[1], np.uint8)
    for y in range(img.shape[0] - 1, -1, -1):
        prev = np.maximum(img[y], np.minimum(top[y], prev))
        out[y] = prev
    return out


def preprocess(img: np.ndarray) -> tuple[np.ndarray, int]:
    """Убирает с копии 300 px буквы и рисунки двумя способами и выбирает подходящий (``morphologicalPreprocessing``).

    Args:
        img: Копия 300 px, uint8.

    Returns:
        Предобработанная копия и номер способа.
    """
    # Способ 1: тёмные области, не связанные с рамкой, заливаются уровнем окружения (восстановление эрозией от рамки).
    seed = np.full_like(img, 255)
    seed[0, :] = seed[-1, :] = seed[:, 0] = seed[:, -1] = 0
    seed = np.maximum(seed, img)
    method1 = reconstruction(seed, img, method="erosion").astype(np.uint8)
    # Остатки букв: max-фильтр, затем min-фильтр кирпичом 1×20 при чёрном окружении (``openGray(..., 0x00)``).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_BRICK)
    method1 = cv2.dilate(method1, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    method1 = cv2.erode(method1, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    method2 = top_bottom_fill(img)
    # Разность способов (uint8 с переполнением, как в ScanTailor) и её отход от гладкого полинома 3×3.
    diff = (method2.astype(np.int16) - method1.astype(np.int16)) % 256
    approx = np.clip(np.round(fit_surface(diff.astype(np.float64), None, 3, 3)), 0, 255)
    deviation = np.abs(diff - approx)
    if (deviation > CHOICE_DEVIATION).mean() < CHOICE_SHARE:
        return method1, 1
    return method2, 2


def estimate_background(gray: np.ndarray, area: np.ndarray | None) -> Background:
    """Фон страницы полиномом 8×5 (``estimateBackground``).

    Args:
        gray: Яркость (H, W), uint8 или float.
        area: Где оценивать фон (bool, H×W) или None — везде.

    Returns:
        Фон в размере входа и промежуточные данные.
    """
    h, w = gray.shape
    scale = WORK_SIDE / max(h, w)
    size = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
    small = cv2.resize(np.clip(gray, 0, 255).astype(np.uint8), size, interpolation=cv2.INTER_AREA)
    work, method = preprocess(small)
    mask = np.ones(work.shape, bool)
    if area is not None:
        mask &= cv2.resize(area.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    # Отсев по столбцам (степень 2) и по строкам (степень 4): пиксель темнее гладкой подгонки на 30+ — не фон.
    mask &= ~(work.astype(np.float64) + DARKER_BY < line_fit(work, COLUMN_DEGREE))
    mask &= ~(work.astype(np.float64) + DARKER_BY < line_fit(work.T, ROW_DEGREE).T)
    # Эрозия годного кирпичом 3×3 (за краем — годно, как у ``erodeBrick`` по умолчанию).
    mask = (
        cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1) > 0
    )
    # Строки, потом столбцы, где годного меньше четверти, выбрасываются целиком.
    mask[mask.sum(axis=1) < mask.shape[1] * LINE_MIN_SHARE, :] = False
    mask[:, mask.sum(axis=0) < mask.shape[0] * LINE_MIN_SHARE] = False
    coeffs, deg_x, deg_y = surface_coeffs(work.astype(np.float64), mask, *SURFACE_DEGREES)
    return Background(render_surface(coeffs, deg_x, deg_y, w, h), work, mask, method)


def raise_above(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Яркость, поднятая над фоном (``RaiseAboveBackground``): не темнее фона → 255, иначе ``round(Y·255/фон)``.

    Args:
        gray: Яркость float32.
        background: Фон float32 того же размера.

    Returns:
        Новая яркость float32 0..255.
    """
    out = np.round(gray * 255.0 / np.maximum(background, 1.0))
    return np.where(background - gray < 1.0, 255.0, np.clip(out, 0, 255)).astype(np.float32)


def brightness_shift(bgr: np.ndarray, area: np.ndarray) -> tuple[np.ndarray, Background]:
    """Сдвиг яркости страницы ``Y' − Y`` по выравниванию освещения (на каналы он ложится одинаково — AdjustBrightness).

    Args:
        bgr: Картинка BGR uint8.
        area: Маска страницы (bool) — по ней оценивается фон.

    Returns:
        ``(сдвиг float32 размера картинки, фон)``.
    """
    y = luma(bgr)
    background = estimate_background(y, area)
    return raise_above(y, background.surface) - y, background


def apply_shift(bgr: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Картинка со сдвигом яркости по каналам, округление и обрезка в 0..255."""
    return np.clip(np.round(bgr.astype(np.float32) + shift[..., None]), 0, 255).astype(np.uint8)


def equalize(bgr: np.ndarray, area: np.ndarray) -> tuple[np.ndarray, Background]:
    """Выравнивание освещения одной страницы: правка внутри ``area``.

    Args:
        bgr: Картинка BGR uint8.
        area: Маска страницы (bool).

    Returns:
        ``(картинка, фон)``.
    """
    shift, background = brightness_shift(bgr, area)
    return apply_shift(bgr, shift * area), background
