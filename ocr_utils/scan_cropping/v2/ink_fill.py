"""Кадр для второго прохода SAM 3 без крупной краски: рамки, линейки, плашки в окружении бумаги закрашены Telea (напечатанная рамка не край листа)."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.background import chroma
from ocr_utils.scan_cropping.v2.book import disk

# Окно локального фона бумаги (закрытие серого), px сегментации (~6.9 px/мм): шире самой широкой полосы краски, которую
# надо найти (рамка IMG_0250 — ~5 мм ≈ 35 px), иначе закрытие не «перешагнёт» полосу и фон в ней будет тёмным.
PAPER_BG_R = 30
# Краска: темнее локального фона бумаги хотя бы на столько уровней серого.
INK_DROP = 30
# Краска малонасыщенная (хрома Lab, 8 бит OpenCV): цветная обложка и кожа — не краска.
INK_MAX_CHROMA = 25.0
# Открытие маски краски, px: остаются полосы шире 2·R+1 ≈ 1 мм (штрихи букв 0.3–0.6 мм уходят).
BOLD_OPEN_R = 3
# Длинные тонкие линии (рамки, линейки, контуры рисунков), которые открытие выбрасывает вместе с буквами: компонента
# краски длиной не меньше RULE_MIN_LEN px (~15 мм) и либо тонкая (короткая сторона повёрнутой рамки ≤ RULE_MAX_WIDTH px,
# ~1.2 мм), либо почти пустая внутри (контур: доля заполнения рамки ≤ RULE_MAX_FILL). Слова и строки не проходят: их
# короткая сторона — высота букв (≥ 12 px), заполнение плотное. Внутренняя тонкая линия двойной рамки IMG_0663 — 3–4 px.
RULE_MIN_LEN = 100
RULE_MAX_WIDTH = 8
RULE_MAX_FILL = 0.15
# Запас вокруг найденной краски перед закрасом, px (полутоновый край полосы).
FILL_MARGIN = 3
# Радиус Telea, px.
TELEA_R = 9
# Кольцо вокруг пятна краски, px (~2 мм), и доля его, которая должна быть светлой бумагой (не краска, не фон, внутри
# книги): у рамки, линейки, плашки бумага с обеих сторон, а у щели у корешка, кармана коврика, тёмного торца по краю —
# обложка, коврик, торец.
RING_PX = 14
RING_PAPER_SHARE = 0.8
# Одиночные полосы и линии не закрашиваются: удлинение повёрнутой рамки компоненты ≥ STRIP_MIN_ASPECT. Так выглядят тень
# и тонкая линия края страницы на стыке с торцом, линии торца, тень у корешка, одиночные линейки: их закрас сливал торец
# со страницей, и второй проход SAM 3 брал торец (МТС 1990/03 IMG_1025, IMG_1034, МТС 1991/05 IMG_0182, РИСК 1991/05
# IMG_0619). Рамка, которая останавливает SAM 3, — замкнутый контур вокруг текста, удлинение ~1–2. Условие на
# заполнение рамки отвергнуто: у чуть изогнутой линии края (IMG_0182) рамка шире линии, заполнение < 0.5.
STRIP_MIN_ASPECT = 8.0
# Печать насыщенная: 90-й перцентиль «темнее бумаги» у пятна печати не меньше этого (рамки, плашки — 63…250, тонкие
# линии рамки IMG_0663 — от 160), у серой тени края страницы — 35…56 (МТС 1991/05 IMG_0182: рваная тень у внешнего
# края — короткие куски, фильтр полос её не ловит).
INK_MIN_P90 = 60
# Добавка второго прохода принимается кусками, касающимися закрашенной краски (с допуском, px): бумага за рамкой
# начинается от рамки. Куски, не касающиеся краски, — колебания SAM 3 на кадре без краски: торец приподнятой страницы
# (МТС 1990/03 IMG_1084: +495 тыс. px), поле у края (IMG_1081: +19 тыс.).
GAIN_TOUCH_PX = 3
# Бумага в кольце ещё и светлая: не темнее этой доли типичной яркости бумаги кадра (медиана серого внутри оболочек
# печати). «Темнее локального фона» на широкой тёмной области (крышка переплёта) не срабатывает — локальный фон там
# тоже тёмный, и без этого порога крышка считалась бумагой: край переплёта закрашивался, второй проход SAM 3 брал
# торец (МТС 1990/03 IMG_1025, РИСК 1991/05 IMG_0619).
PAPER_MIN_SHARE = 0.7


def ink_mask(bgr: np.ndarray, book: np.ndarray) -> np.ndarray:
    """Вся краска внутри книги: темнее локального фона бумаги и малонасыщенная.

    Args:
        bgr: Кадр BGR (масштаб сегментации).
        book: Грубая маска книги.

    Returns:
        Булева маска.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    local = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, disk(PAPER_BG_R))
    drop = local.astype(np.int16) - gray.astype(np.int16)
    dark = drop > INK_DROP
    return dark & (chroma(lab) < INK_MAX_CHROMA) & book


def long_rules(ink: np.ndarray) -> np.ndarray:
    """Длинные тонкие линии и контуры среди всей краски (см. ``RULE_*``).

    Args:
        ink: Вся краска (bool).

    Returns:
        Булева маска.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    for i in np.nonzero(np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= RULE_MIN_LEN)[0]:
        if i == 0:
            continue
        x, y, w, h, area = stats[i]
        ys, xs = np.nonzero(labels[y : y + h, x : x + w] == i)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        long_side, short_side = max(rw, rh), max(min(rw, rh), 1.0)
        if long_side >= RULE_MIN_LEN and (
            short_side <= RULE_MAX_WIDTH or area / (long_side * short_side) <= RULE_MAX_FILL
        ):
            keep[i] = True
    return keep[labels]


def drop_straight_strips(bold: np.ndarray) -> np.ndarray:
    """Крупная краска без одиночных прямых полос (см. ``STRIP_*``).

    Args:
        bold: Крупная краска (bool).

    Returns:
        Булева маска без прямых полос.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bold.astype(np.uint8), connectivity=8)
    keep = np.ones(count, bool)
    keep[0] = False
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if max(w, h) < RULE_MIN_LEN:
            continue
        ys, xs = np.nonzero(labels[y : y + h, x : x + w] == i)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        long_side, short_side = max(rw, rh), max(min(rw, rh), 1.0)
        if long_side / short_side >= STRIP_MIN_ASPECT:
            keep[i] = False
    return keep[labels]


def drop_faint(bold: np.ndarray, drop: np.ndarray) -> np.ndarray:
    """Крупная краска без бледных пятен (тени): 90-й перцентиль «темнее бумаги» меньше ``INK_MIN_P90``.

    Args:
        bold: Крупная краска (bool).
        drop: Насколько темнее локального фона бумаги, уровни серого (int16).

    Returns:
        Булева маска.
    """
    count, labels = cv2.connectedComponents(bold.astype(np.uint8), connectivity=8)
    if count <= 1:
        return bold
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(flat, minlength=count))]
    values = drop.ravel()[order]
    keep = np.zeros(count, bool)
    for i in range(1, count):
        part = values[bounds[i] : bounds[i + 1]]
        keep[i] = len(part) > 0 and np.percentile(part, 90) >= INK_MIN_P90
    return keep[labels]


def surrounded_by_paper(bold: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """Пятна крупной краски, вокруг которых бумага: кольцо ``RING_PX`` почти целиком бумага.

    Args:
        bold: Крупная краска (bool).
        paper: Светлая бумага (bool): в книге, не фон, не тёмное, не темнее ``PAPER_MIN_SHARE`` типичной бумаги.

    Returns:
        Булева маска оставленных пятен.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bold.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    h, w = bold.shape
    for i in range(1, count):
        x, y, bw, bh = stats[i, :4]
        x0, y0 = max(0, x - RING_PX), max(0, y - RING_PX)
        x1, y1 = min(w, x + bw + RING_PX), min(h, y + bh + RING_PX)
        comp = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = (cv2.dilate(comp, disk(RING_PX)) > 0) & (comp == 0) & ~bold[y0:y1, x0:x1]
        if ring.any() and paper[y0:y1, x0:x1][ring].mean() >= RING_PAPER_SHARE:
            keep[i] = True
    return keep[labels]


def bold_ink(bgr: np.ndarray, book: np.ndarray, bg: np.ndarray, hulls: np.ndarray | None = None) -> np.ndarray:
    """Крупная краска в окружении бумаги: рамки (и тонкие), линейки, плашки, тёмные иллюстрации — без строк текста.

    Args:
        bgr: Кадр BGR (масштаб сегментации).
        book: Грубая маска книги.
        bg: Маска фона (щели и карманы коврика не закрашиваются: вокруг них не бумага).
        hulls: Оболочки своей печати — по ним типичная яркость бумаги; None — вся книга.

    Returns:
        Булева маска с запасом ``FILL_MARGIN``.
    """
    ink = ink_mask(bgr, book)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    local = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, disk(PAPER_BG_R))
    drop = local.astype(np.int16) - gray.astype(np.int16)
    dark = drop > INK_DROP
    bold = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, disk(BOLD_OPEN_R)) > 0
    bold |= long_rules(ink)
    bold = drop_faint(drop_straight_strips(bold), drop)
    inside = (hulls & book) if hulls is not None and (hulls & book).any() else book
    light = gray >= PAPER_MIN_SHARE * float(np.median(gray[inside])) if inside.any() else np.ones_like(book)
    bold = surrounded_by_paper(bold, book & ~bg & ~dark & light)
    return (cv2.dilate(bold.astype(np.uint8), disk(FILL_MARGIN)) > 0) & book


def touching_ink(added: np.ndarray, ink: np.ndarray, gap: int = GAIN_TOUCH_PX) -> np.ndarray:
    """Куски добавки второго прохода SAM 3, касающиеся закрашенной краски (остальное — колебания SAM 3).

    Args:
        added: Добавка второго прохода (bool).
        ink: Закрашенная краска (bool).
        gap: Допуск касания, px.

    Returns:
        Булева маска принятых кусков.
    """
    if not added.any() or not ink.any():
        return np.zeros_like(added)
    near = cv2.dilate(ink.astype(np.uint8), disk(gap)) > 0
    count, labels = cv2.connectedComponents(added.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    keep[np.unique(labels[near & added])] = True
    keep[0] = False
    return keep[labels]


def fill_bold_ink(
    bgr: np.ndarray, book: np.ndarray, bg: np.ndarray, hulls: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Кадр для SAM 3: крупная краска закрашена Telea по соседней бумаге.

    Args:
        bgr: Кадр BGR (масштаб сегментации).
        book: Грубая маска книги.
        bg: Маска фона.
        hulls: Оболочки своей печати (типичная яркость бумаги) или None.

    Returns:
        ``(закрашенный кадр, маска закрашенного)``.
    """
    mask = bold_ink(bgr, book, bg, hulls)
    if not mask.any():
        return bgr, mask
    return cv2.inpaint(bgr, mask.astype(np.uint8), TELEA_R, cv2.INPAINT_TELEA), mask
