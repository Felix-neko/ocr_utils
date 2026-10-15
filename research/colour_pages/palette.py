"""Палитра страницы по уверенному центру и рост бумаги по ней у края: принимается цвет краски или бумаги страницы, незнакомый цвет (обложка, крышка, коврик) и тень кромки листа останавливают рост."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.side_fit import EDGE_MIN_SHARE, EDGE_TOL_MM, edge_dips, mm, ransac_line
from ocr_utils.scan_cropping.v2.trapezoid import quad_mask

# Уверенный центр страницы: трапеция, ужатая к центру до этой доли по каждой оси (подсказка пользователя: человек
# смотрит, какие краски и бумага на центральных 80 % страницы).
CENTRE_SHARE = 0.8
# Корзины гистограммы Lab (уровни OpenCV): L, a*, b*.
BIN_L, BIN_AB = 8, 4
# Корзина входит в палитру, если в ней не меньше этой доли центра.
PALETTE_MIN_SHARE = 0.001
# Тень и освещение у края: цвет принимается и темнее палитры на столько уровней L (при тех же a*, b*), и светлее — на
# PALETTE_L_LIGHT (засвет у края, IMG_1113: проплешины в поле слева сверху).
PALETTE_L_SHADE = 25
PALETTE_L_LIGHT = 8
# Рост: не дальше этого от маски страницы, мм; в трапеции, расширенной на столько, мм.
REACH_MM = 25.0
QUAD_GROW_MM = 10.0
# Рост только вверх и вниз: по горизонтали — между боковыми сторонами трапеции с этим запасом, мм. За боковыми
# сторонами — белый торец книжного блока: белое есть и в палитре страницы (поле вокруг плашки, белый круг), а тени
# кромки у торца нет (МТС 1990/09 IMG_0310 справа, РИСК 1991/02 IMG_0642 слева). Боковые стороны — за
# ``side_fit.snap_sheet_edges`` с проверкой «край рваный».
LATERAL_MARGIN_MM = 2.0


def centre_quad(quad: np.ndarray, share: float = CENTRE_SHARE) -> np.ndarray:
    """Трапеция, ужатая к своему центру до ``share`` (вершины — к центроиду)."""
    centre = quad.mean(axis=0)
    return centre + (quad - centre) * share


def palette_of(lab: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Палитра — занятые корзины Lab уверенного центра, с допуском тени ``PALETTE_L_SHADE`` вниз по L.

    Args:
        lab: Кадр Lab float32 (``paper.smooth_lab``).
        centre: Уверенный центр (bool).

    Returns:
        Таблица принятых корзин (bool, [L, a, b]).
    """
    bins = (256 // BIN_L, 256 // BIN_AB, 256 // BIN_AB)
    idx = (lab[centre] // np.array([BIN_L, BIN_AB, BIN_AB], np.float32)).astype(int)
    hist = np.zeros(bins, np.int64)
    np.add.at(hist, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)
    occupied = hist >= max(1, PALETTE_MIN_SHARE * len(idx))
    # Тень: корзина принимается, если та же a*, b* светлее на 0…PALETTE_L_SHADE есть в палитре.
    accepted = occupied.copy()
    for step in range(1, PALETTE_L_SHADE // BIN_L + 1):
        accepted[:-step] |= occupied[step:]
    for step in range(1, PALETTE_L_LIGHT // BIN_L + 1):
        accepted[step:] |= occupied[:-step]
    return accepted


def in_palette(lab: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    """Пиксели, цвет которых есть в палитре (bool)."""
    idx = (lab // np.array([BIN_L, BIN_AB, BIN_AB], np.float32)).astype(int)
    return accepted[idx[..., 0], idx[..., 1], idx[..., 2]]


def between_sides(quad: np.ndarray, shape: tuple[int, int], margin: float) -> np.ndarray:
    """Пиксели между боковыми сторонами трапеции (продолженными вверх и вниз) с запасом ``margin`` px.

    Args:
        quad: Вершины (4, 2) по кругу.
        shape: ``(H, W)``.
        margin: Запас наружу, px.

    Returns:
        Маска (bool).
    """
    order = np.argsort(quad[:, 0])
    left, right = quad[order[:2]], quad[order[2:]]
    rows = np.arange(shape[0], dtype=np.float64)

    def side_x(pair: np.ndarray) -> np.ndarray:
        (x0, y0), (x1, y1) = pair[np.argsort(pair[:, 1])]
        return x0 + (rows - y0) * (x1 - x0) / max(y1 - y0, 1e-6)

    cols = np.arange(shape[1])[None, :]
    return (cols >= side_x(left)[:, None] - margin) & (cols <= side_x(right)[:, None] + margin)


def edge_barrier(light: np.ndarray, page: np.ndarray) -> np.ndarray:
    """Преграда «за кромкой листа»: по каждой стороне страницы, где найдена тень кромки (провалы L на прямой RANSAC
    не меньше ``EDGE_MIN_SHARE`` строк), — полуплоскость за прямой (+1 мм).

    Args:
        light: Светлота L (0..255) float32.
        page: Маска страницы (bool).

    Returns:
        Маска «за кромкой» (bool).
    """
    h, w = page.shape
    barrier = np.zeros((h, w), bool)
    for outward, transposed in ((-1, False), (1, False), (-1, True), (1, True)):
        mask = page.T if transposed else page
        lum = light.T if transposed else light
        rows, xs, count = edge_dips(lum, mask, outward)
        fit = ransac_line(rows, xs, mm(EDGE_TOL_MM))
        if fit is None or fit[1].sum() < EDGE_MIN_SHARE * count:
            continue
        (a, b), _ = fit
        line = a * np.arange(mask.shape[0]) + b
        cols = np.arange(mask.shape[1])[None, :]
        beyond = cols < line[:, None] - mm(1.0) if outward < 0 else cols > line[:, None] + mm(1.0)
        barrier |= beyond.T if transposed else beyond
    return barrier


def grow_by_palette(
    lab: np.ndarray,
    light: np.ndarray,
    page: np.ndarray,
    quad: np.ndarray,
    paper: np.ndarray,
    book: np.ndarray,
    bg: np.ndarray,
) -> np.ndarray:
    """Бумага страницы, наращённая у края по её палитре: связные куски из палитры, касающиеся страницы.

    Args:
        lab: Кадр Lab float32 (``paper.smooth_lab``).
        light: Светлота L без медианы (гаусс 3×3, как в ``side_fit``) — для тени кромки: медиана ``smooth_lab`` стирает
            тонкую линию тени (IMG_1113: прямая слева по 39 % строк против 79 %).
        page: Маска страницы (bool).
        quad: Трапеция страницы (4, 2).
        paper: Бумага кадра (bool) — не кандидаты.
        book: Грубая маска книги (bool).
        bg: Фон (bool).

    Returns:
        Добавка (bool).
    """
    shape = page.shape
    centre = quad_mask(centre_quad(quad), shape) & page
    if centre.sum() < 1000:
        return np.zeros(shape, bool)
    accepted = palette_of(lab, centre)
    reach = cv2.distanceTransform((~page).astype(np.uint8), cv2.DIST_L2, 5) <= REACH_MM * SEG_PX_PER_MM
    k = 2 * int(round(QUAD_GROW_MM * SEG_PX_PER_MM)) + 1
    near_quad = cv2.dilate(
        quad_mask(quad, shape).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    candidates = in_palette(lab, accepted) & book & ~bg & ~paper & reach & (near_quad > 0)
    candidates &= between_sides(quad, shape, LATERAL_MARGIN_MM * SEG_PX_PER_MM)
    candidates &= ~edge_barrier(light, page)
    count, labels = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=8)
    touch = np.unique(labels[cv2.dilate(page.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0])
    grown = np.isin(labels, touch[touch > 0])
    # Дыры, замкнутые страницей и ростом (краска и пятна вне палитры внутри поля), — тоже бумага, если не фон.
    closed = ndimage.binary_fill_holes(page | grown) & ~(page | grown) & ~bg & book
    return grown | closed


def edge_light(bgr: np.ndarray) -> np.ndarray:
    """Светлота L (0..255) для поиска тени кромки: гаусс 3×3, как в ``side_fit.snap_sheet_edges``."""
    return cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32), (3, 3), 0)
