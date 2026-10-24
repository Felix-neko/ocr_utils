"""Цветные страницы: уровень цветности по маске страницы и рост бумаги у края цветной страницы по палитре её уверенного центра (краска и бумага страницы — да, незнакомый цвет обложки, крышки, коврика и тень кромки листа — нет); защита контента цветных страниц при любой позе."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM, PagePose, PageQuad
from ocr_utils.scan_cropping.v2.paper import smooth_lab
from ocr_utils.scan_cropping.v2.side_fit import (
    EDGE_MAX_KEYSTONE,
    EDGE_MIN_SHARE,
    EDGE_TOL_MM,
    edge_dips,
    mm,
    ransac_line,
)
from ocr_utils.scan_cropping.v2.trapezoid import quad_mask

# Подсказка пользователя (2026-10-15): у цветной страницы важнее не потерять контент на краях, чем не захватить кусок
# периферии; человек видит, какие краски и бумага на центральных 80 % страницы, и доводит край ими, а незнакомый цвет
# на краю — кусок обложки или фона. Стенд и замеры — research/colour_pages.

# Целиком цветная страница: медиана хромы (a*, b* OpenCV от 128) по маске страницы не меньше этого. Пак-6, 4921
# страница: медиана в два горба — 0–12 (4809 страниц) и 28+ (112), между ними пусто; порог где угодно в 12…25.
FULL_MIN_MEDIAN = 20.0
# Насыщенный пиксель — хрома больше этого; доли насыщенного для уровней «цветные элементы» и «следы цвета».
SATURATED = 30.0
ELEMENTS_MIN_SHARE = 0.03
TRACE_MIN_SHARE = 0.01


class Colour(Enum):
    """Уровень цветности страницы."""

    FULL = "full"  # целиком на цветном фоне (обложки и их обороты)
    ELEMENTS = "elements"  # белая страница с цветными плашками
    TRACE = "trace"  # тонкие цветные рамки, линии, мелкие логотипы
    PLAIN = "plain"  # обычная


@dataclass(frozen=True)
class PageColour:
    """Цветность страницы.

    Поля:
        median_chroma: Медиана хромы по маске страницы.
        saturated_share: Доля пикселей с хромой больше ``SATURATED``.
        level: Уровень.
    """

    median_chroma: float
    saturated_share: float
    level: Colour


def chroma_of(bgr: np.ndarray) -> np.ndarray:
    """Хрома Lab (OpenCV, a* и b* от 128) по пикселям, float32."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)


def page_colour(chroma: np.ndarray, page: np.ndarray) -> PageColour:
    """Цветность страницы по маске.

    Args:
        chroma: Хрома кадра (``chroma_of``).
        page: Маска страницы (bool).

    Returns:
        Цветность; пустая маска — обычная.
    """
    if not page.any():
        return PageColour(0.0, 0.0, Colour.PLAIN)
    values = chroma[page]
    median, share = float(np.median(values)), float((values > SATURATED).mean())
    if median >= FULL_MIN_MEDIAN:
        level = Colour.FULL
    elif share >= ELEMENTS_MIN_SHARE:
        level = Colour.ELEMENTS
    elif share >= TRACE_MIN_SHARE:
        level = Colour.TRACE
    else:
        level = Colour.PLAIN
    return PageColour(median, share, level)


# Тонированная страница: заметная доля фона (не краски) другого цвета, чем бумага страницы, — оливковые, синие, бежевые
# поля обложек с белыми плашками. Хрома их не отличает: у обложки ПХ 1974/01 IMG_0003 медиана 10.2, а у белой страницы
# IMG_0087 — 9.5. Выравнивание освещения поднимает такой фон до белого (обложка ПХ осветлялась целиком).
# Пиксель тонированный — сдвиг (a*, b*) от цвета бумаги страницы больше TINT_DELTA_AB. Замер 2026-10-23 по половинам итогов
# v2 (ПХ пак-4 1974/01 и 1975, 2222 половины; ХиП 1991, 912): обложки ПХ 0.55–0.81, обложки ХиП 0.36–0.57; текстовые
# страницы до 0.17 (синяя печать, карточки учёта), полоса с фото 0.28. Признак по ΔE (с яркостью) отвергнут: тень
# и виньетирование текстовых страниц ХиП давали 164 срабатывания из 912.
TINT_DELTA_AB = 8.0
TINT_MIN_SHARE = 0.3
# Плотность рабочей копии признака, px/мм: фон без букв различим, счёт — миллисекунды.
TINT_WORK_PX_PER_MM = 2.0
# Краска — пиксели темнее этого перцентиля L страницы; «бумага» — пиксели светлее PAPER_PERCENTILE.
TINT_INK_PERCENTILE = 20.0
TINT_PAPER_PERCENTILE = 85.0


def tinted_share(bgr: np.ndarray, page: np.ndarray, px_per_mm: float) -> float:
    """Доля фона страницы, отличного по цвету от её бумаги (признак тонированной страницы — обложки).

    «Бумага» — медиана (a*, b*) самых светлых пикселей страницы (светлее :data:`TINT_PAPER_PERCENTILE`). Фон — пиксели
    светлее :data:`TINT_INK_PERCENTILE` (без краски). Тонированный пиксель фона — сдвиг (a*, b*) от бумаги больше
    :data:`TINT_DELTA_AB`.

    Args:
        bgr: Картинка BGR (итог v2).
        page: Маска страницы того же размера (bool).
        px_per_mm: Плотность картинки, px/мм.

    Returns:
        Доля тонированного фона 0..1; пустая маска — 0.
    """
    if not page.any():
        return 0.0
    # Рабочая копия ~2 px/мм по рамке страницы: признак — про поля цвета, буквы ему не нужны.
    ys, xs = np.nonzero(page)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    factor = min(1.0, TINT_WORK_PX_PER_MM / px_per_mm)
    size = (max(1, int((x1 - x0) * factor)), max(1, int((y1 - y0) * factor)))
    small = cv2.resize(bgr[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(page[y0:y1, x0:x1].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    if not mask.any():
        return 0.0
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[mask].astype(np.float32)
    lightness, a, b = lab[:, 0], lab[:, 1] - 128.0, lab[:, 2] - 128.0
    # ">=": на ровном поле (ничьи по L) перцентиль совпадает с самим полем, строгое «>» выкинуло бы его целиком.
    background = lightness >= np.percentile(lightness, TINT_INK_PERCENTILE)
    paper = lightness >= np.percentile(lightness, TINT_PAPER_PERCENTILE)
    if not background.any():
        return 0.0
    # Цвет бумаги страницы — по самым светлым пикселям; сдвиг цвета фона от него.
    shift = np.hypot(a - np.median(a[paper]), b - np.median(b[paper]))
    return float((shift[background] > TINT_DELTA_AB).mean())


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
    cols = np.arange(shape[1])[None, :]
    return (cols >= side_x(left, rows)[:, None] - margin) & (cols <= side_x(right, rows)[:, None] + margin)


def side_x(pair: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Абсцисса боковой стороны (прямой через две вершины) на строках ``rows``, продолженной вверх и вниз.

    Args:
        pair: Две вершины стороны (2, 2).
        rows: Строки.

    Returns:
        Абсциссы, float64.
    """
    (x0, y0), (x1, y1) = pair[np.argsort(pair[:, 1])]
    return x0 + (rows - y0) * (x1 - x0) / max(y1 - y0, 1e-6)


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


def colour_pages_growth(
    img: np.ndarray, pages: list[PageQuad], paper: np.ndarray, book: np.ndarray, bg: np.ndarray
) -> np.ndarray:
    """Рост бумаги у края целиком цветных страниц кадра по их палитрам.

    Args:
        img: Кадр сегментации BGR.
        pages: Страницы кадра.
        paper: Бумага кадра (bool).
        book: Грубая маска книги (bool).
        bg: Фон (bool).

    Returns:
        Добавка (bool); без цветных страниц — пустая.
    """
    chroma = chroma_of(img)
    added = np.zeros_like(paper)
    lab = light = None
    for page in pages:
        if page.quad is None or page_colour(chroma, page.mask).level is not Colour.FULL:
            continue
        if lab is None:
            lab, light = smooth_lab(img), edge_light(img)
        added |= grow_by_palette(lab, light, page.mask, page.quad, paper, book, bg)
    return added


def colour_pages_flat(img: np.ndarray, pages: list[PageQuad]) -> list[PageQuad]:
    """Страницы для защиты контента: целиком цветные со сходом до ``EDGE_MAX_KEYSTONE`` считаются лежащими плоско.

    Рваная маска цветной страницы делает её «приподнятой» (МТС 1990/04 IMG_1113), и защита контента отключалась как раз
    там, где нужна.

    Args:
        img: Кадр сегментации BGR.
        pages: Страницы кадра.

    Returns:
        Страницы с позой «плоско» у цветных.
    """
    chroma = chroma_of(img)
    return [
        (
            replace(p, pose=PagePose.FLAT)
            if p.quad is not None
            and p.features.keystone <= EDGE_MAX_KEYSTONE
            and page_colour(chroma, p.mask).level is Colour.FULL
            else p
        )
        for p in pages
    ]
