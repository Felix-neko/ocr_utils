"""Строки своей печати, срезанные краем листа: глифы, упёршиеся в кромку бумаги (коврик, крышка, ступенька бумаги)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

# Все размеры ниже — в px рабочего масштаба: ½ кадра (``cv2.IMREAD_REDUCED_COLOR_2``), для кадров пака-6
# 5687×4268 это 2844×2134, ~6.9 px/мм на развороте (замер ``ocr_utils/scan_cropping/v2/background.py``).

# Краска: адаптивный порог (окно ~7 мм, сдвиг в уровнях серого) или просто очень тёмное (доля яркой бумаги p95).
INK_BLOCK, INK_OFFSET = 51, 15
DARK_SHARE = 0.4
# «Длинные тёмные структуры» (коврик, крышка, тень, линейки): тёмные пробеги не короче этого по одной из четырёх
# осей (0°, 90°, ±45°), px (~9 мм — длиннее любого глифа обычного набора).
STRUCT_RUN_PX = 61
# Толстые структуры (коврик, крышка, иллюстрации): переживают раскрытие квадратом этой стороны (~1.3 мм, толще
# штриха текста); тонкие линейки и подчёркивания (2–6 px) — нет, их граница краем листа не считается.
STRUCT_THICK_PX = 9
# Толстая структура — край листа, только если она задевает полосу у рамки кадра (коврик, крышка уходят за кадр);
# логотипы, плашки и иллюстрации внутри страницы — нет, и ступеньки бумаги в STEP_NEAR_INNER_PX от них — тоже нет.
STEP_NEAR_INNER_PX = 7
# Срез вдоль (низ, верх): за краем под касающимися глифами (полоса ALONG_BEYOND_PX) снаружи — коврик, крышка — не
# меньше этой доли (серые плашки и растр иллюстраций — не «снаружи»).
ALONG_BEYOND_PX = (4, 40)
ALONG_OUTSIDE_SHARE = 0.5
# Бумага без печати: закрытие серого кругом этого диаметра (~3.6 мм, толще штриха любого заголовка).
PAPER_CLOSE_PX = 25
# Тонкая вертикальная тёмная линия (край листа над корешком): вертикальный тёмный пробег не короче, px (как у
# длинных структур — тогда линия не входит в глифы и не слипается с упёршейся в неё буквой).
VERTICAL_RUN_PX = STRUCT_RUN_PX
# Ступенька бумаги (край верхнего листа над нижним, лист над тенью корешка): морфологический градиент 3×3
# закрытой бумаги не меньше этого, уровни серого (край листа у корешка IMG_0542: 159 → 211; плавная тень < 10).
PAPER_STEP = 40
# Запас вокруг глифа, в котором берётся ход края для стороны касания, px.
CONTACT_PAD_PX = 10
# Краска строки нейтральная: медианная хрома пикселей глифов строки не больше (фактура голубой крышки — нет).
LINE_INK_MAX_CHROMA = 12.0
# Бумага под строкой белая: медиана серого не тёмных и не глифовых пикселей рамки строки не ниже доли p95 кадра
# (растр фотографий, серые плашки, логотипы — серее).
LINE_WHITE_SHARE = 0.72
# Касание глифа и края: раздутие края квадратом этой стороны (5 → до 2 px, ~0.3 мм).
TOUCH_PX = 5
# Компонента-глиф: высота, ширина и площадь, px.
GLYPH_MIN_H, GLYPH_MAX_H, GLYPH_MAX_W, GLYPH_MIN_AREA = 5, 60, 90, 12
# Смыкание глифов в строку вдоль, px (~3.6 мм: межсловный пробел, не межколонник).
LINE_GAP_PX = 25
# Строка: не меньше глифов; вытянута (длина ≥ LINE_MIN_ASPECT высот и ≥ LINE_MIN_LEN_PX); медианная высота
# глифа в пределах обычного набора, px.
LINE_MIN_GLYPHS = 3
LINE_MIN_ASPECT, LINE_MIN_LEN_PX = 2.0, 30
LINE_GLYPH_H = (7.0, 40.0)
# Срез вдоль строки (низ, верх): касаются не меньше стольких глифов и не меньше этой доли глифов строки.
ALONG_MIN_TOUCH, ALONG_MIN_SHARE = 3, 0.25
# Срез конца строки: строка не выше END_MAX_LINE_HEIGHT медианных высот глифа (не слипшиеся строки), крайний
# глиф не ниже END_MIN_GLYPH_SHARE медианной высоты (буква, а не крап обреза у кромки).
END_MAX_LINE_HEIGHT = 2.5
END_MIN_GLYPH_SHARE = 0.5
# Бумага под строкой: медиана закрытой бумаги не темнее доли p95 кадра, медианная хрома не больше (обложка,
# крышка, свитер — нет).
LINE_PAPER_SHARE = 0.55
LINE_MAX_CHROMA = 30.0
# Маска листа: светлое (≥ PAGE_LIGHT_SHARE·p95) и малоцветное, раскрытое квадратом PAGE_OPEN_PX (рвёт мелкую
# фактуру свитера, ковра), компоненты не меньше PAGE_MIN_AREA_SHARE кадра, смыкание PAGE_CLOSE_PX (печать внутри
# страницы) и заливка дыр. Строка должна лежать на листе долей рамки не меньше LINE_ON_PAGE_SHARE.
PAGE_LIGHT_SHARE = 0.55
# Синева бумаги не больше (128 − b Lab): светлая бумага пака b ∈ [−4, 4], голубая крышка подшивки — синее.
PAGE_MAX_BLUE = 7.0
PAGE_OPEN_PX = 9
PAGE_MIN_AREA_SHARE = 0.015
PAGE_CLOSE_PX = 41
LINE_ON_PAGE_SHARE = 0.6
# Кожа (YCrCb): пальцы срезом не считаются; Cr ≥ 146 отделяет палец (Cr 152–154) от жёлтой бумаги в тени
# корешка (Cr 141, IMG_0542), остальное — как ``scan_cropping/v2/background.hand_mask``.
SKIN_CR, SKIN_CB, SKIN_MIN_Y, SKIN_MIN_CR_MINUS_CB = (146, 185), (80, 130), 60, 19
# Зона пальца: раздутие кожи, px (~6 мм: тень и ноготь).
SKIN_ZONE_PX = 41
# Полоса у рамки кадра, где касания не считаются (срез рамкой снимка — не край листа), px.
FRAME_BORDER_PX = 20
# Что за краем по ходу строки (конец строки): полоса этой длины, px (~6 см: за щелью корешка шириной до ~1 см
# лежит другая страница, за наружным краем — обрез стопки 1–3 см, крышка и коврик), отступ от глифа, px.
BEYOND_LEN_PX, BEYOND_GAP_PX = 400, 12
# За краем «наружу» (коврик, крышка, вне кадра): доля тёмного (< MAT_SHARE·p95) или цветного тёмного не меньше этой
# (щель корешка IMG_0644 даёт ~0.1).
OUTER_MIN_SHARE = 0.3
MAT_SHARE = 0.3
# Цветное тёмное (крышка подшивки) — темнее этой доли p95.
COVER_DARK_SHARE = 0.7
# Обложка: доля цветного (хрома > COVER_CHROMA) среди не тёмных пикселей кадра.
COVER_CHROMA = 30.0
COVER_MIN_SHARE = 0.12


class Side(str, Enum):
    """Сторона листа, краем которой срезана строка."""

    BOTTOM = "bottom"
    TOP = "top"
    OUTER = "outer"
    SPINE = "spine"


class Contact(str, Enum):
    """Где край относительно глифа: ниже, выше, справа, слева."""

    BELOW = "below"
    ABOVE = "above"
    RIGHT = "right"
    LEFT = "left"


@dataclass(frozen=True)
class CutLine:
    """Срезанная строка: сторона, рамка ``(x, y, w, h)`` на ½ кадра, глифов в строке и касающихся края."""

    side: Side
    box: tuple[int, int, int, int]
    glyphs: int
    touching: int


@dataclass
class FrameMasks:
    """Маски разбора кадра для оверлея (½ кадра): край листа, глифы, касающиеся края, глифы срезанных строк."""

    edge: np.ndarray
    touching: np.ndarray
    cut: np.ndarray


@dataclass
class FrameResult:
    """Разбор кадра: срезанные строки, число глифов, доля цветного (обложка), маски (по запросу)."""

    lines: list[CutLine]
    glyphs: int
    colour_share: float
    masks: FrameMasks | None = field(default=None, repr=False)

    def count(self, side: Side) -> int:
        """Число срезанных строк стороны ``side``.

        Args:
            side: Сторона.

        Returns:
            Число строк.
        """
        return sum(line.side is side for line in self.lines)

    @property
    def is_cover(self) -> bool:
        """Кадр с обложкой: цветного больше :data:`COVER_MIN_SHARE` светлой части кадра."""
        return self.colour_share >= COVER_MIN_SHARE


def read_half(path: Path) -> np.ndarray:
    """Читает кадр сразу в ½ разрешения (декодер JPEG уменьшает при разборе — вдвое быстрее полного чтения).

    Args:
        path: Путь JPEG.

    Returns:
        Картинка BGR uint8.

    Raises:
        OSError: Файл не читается.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
    if bgr is None:
        raise OSError(f"не читается: {path}")
    return bgr


def chroma_map(bgr: np.ndarray) -> np.ndarray:
    """Хрома Lab (8 бит OpenCV): расстояние (a, b) от нейтрали.

    Args:
        bgr: Картинка BGR.

    Returns:
        Карта float32.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)


def dark_mask(gray: np.ndarray, paper_level: float) -> np.ndarray:
    """Тёмное: краска по адаптивному порогу и всё очень тёмное (коврик вдали от бумаги).

    Args:
        gray: Серое uint8.
        paper_level: Яркость бумаги кадра (p95 серого).

    Returns:
        Маска uint8 0/1.
    """
    ink = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, INK_BLOCK, INK_OFFSET)
    return (ink.astype(bool) | (gray < DARK_SHARE * paper_level)).astype(np.uint8)


def line_kernels(length: int) -> list[np.ndarray]:
    """Линейные ядра длины ``length`` по осям 0°, 90°, 45°, 135°.

    Args:
        length: Длина, px.

    Returns:
        Четыре ядра uint8.
    """
    diagonal = np.eye(length, dtype=np.uint8)
    return [np.ones((1, length), np.uint8), np.ones((length, 1), np.uint8), diagonal, np.fliplr(diagonal).copy()]


def dark_structures(
    dark: np.ndarray, run_px: int = STRUCT_RUN_PX, thick_px: int = STRUCT_THICK_PX
) -> tuple[np.ndarray, np.ndarray]:
    """Тёмные структуры не размера буквы: толстые области и длинные тонкие линии.

    Толстое (коврик, крышка, заливки) — раскрытие квадратом ``thick_px``; тонкое (линейки, тени, край стопки) —
    пробег не короче ``run_px`` вдоль одной из осей по тёмному **вне толстого**: иначе буква, упёршаяся в коврик,
    лежала бы на длинном вертикальном пробеге «буква + коврик» и пропадала бы из глифов.

    Args:
        dark: Тёмное (uint8 0/1).
        run_px: Длина пробега тонкой линии, px.
        thick_px: Сторона квадрата толстого, px.

    Returns:
        Кортеж ``(thick, thin)`` масок uint8 0/1.
    """
    thick = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((thick_px, thick_px), np.uint8))
    rest = dark & (1 - thick)
    thin = np.zeros_like(dark)
    for kernel in line_kernels(run_px):
        thin |= cv2.morphologyEx(rest, cv2.MORPH_OPEN, kernel)
    return thick, thin


def skin_zone(bgr: np.ndarray) -> np.ndarray:
    """Зона пальцев: кожа по YCrCb, раздутая на :data:`SKIN_ZONE_PX`.

    Args:
        bgr: Кадр BGR.

    Returns:
        Булева маска.
    """
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.int16)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    skin = (cr >= SKIN_CR[0]) & (cr <= SKIN_CR[1]) & (cb >= SKIN_CB[0]) & (cb <= SKIN_CB[1]) & (y >= SKIN_MIN_Y)
    skin &= (cr - cb) >= SKIN_MIN_CR_MINUS_CB
    # Раскрытие убирает одиночные «кожаные» пиксели бумаги.
    skin8 = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.dilate(skin8, np.ones((SKIN_ZONE_PX, SKIN_ZONE_PX), np.uint8)) > 0


def edge_mask(
    gray: np.ndarray, thick: np.ndarray, thin: np.ndarray, excluded: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Край листа: ступеньки бумаги без печати и граница толстых тёмных структур у рамки; для концов строк ещё и
    тонкие вертикальные тёмные линии (тень у края верхнего листа над корешком), без пальцев и рамки кадра.

    Тонкие линии годятся только для среза конца строки: горизонтальные линейки и подчёркивания под строкой
    срезом вдоль не считаются, а узкая полоса тени у края листа над корешком (IMG_0542: ~20 px) крупным закрытием
    бумаги заливается, и ступеньки там нет — край виден только как тёмная линия.

    Args:
        gray: Серое uint8.
        thick: Толстые тёмные структуры (uint8 0/1).
        thin: Тонкие длинные тёмные линии (uint8 0/1).
        excluded: Где край не ищется (пальцы, полоса у рамки), bool.

    Returns:
        Кортеж ``(along, end, paper)``: край для среза вдоль строки и для среза конца (uint8 0/1), закрытая бумага
        (uint8, серое без печати).
    """
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PAPER_CLOSE_PX, PAPER_CLOSE_PX))
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, disk)
    step = cv2.morphologyEx(paper, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) >= PAPER_STEP
    # Край — только граница толстых структур (линейки и подчёркивания — нет: текст над линейкой не срезан),
    # задевающих рамку кадра (коврик, крышка).
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
    keep = np.zeros(count, bool)
    keep[np.unique(labels[frame_band(thick.shape, FRAME_BORDER_PX + 2)])] = True
    keep[0] = False
    inner = (~keep[labels]) & (labels > 0)
    near_inner = cv2.dilate(inner.astype(np.uint8), np.ones((STEP_NEAR_INNER_PX, STEP_NEAR_INNER_PX), np.uint8)) > 0
    thick = keep[labels].astype(np.uint8)
    border = thick - cv2.erode(thick, np.ones((3, 3), np.uint8))
    along = ((step & ~near_inner) | (border > 0)) & ~excluded
    # Тонкие вертикальные тёмные линии (вне толстых структур).
    vertical = cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((VERTICAL_RUN_PX, 1), np.uint8))
    vertical = (vertical > 0) & ~excluded
    end = along | vertical
    return along.astype(np.uint8), end.astype(np.uint8), paper


def page_mask(paper: np.ndarray, chroma: np.ndarray, blue: np.ndarray, paper_level: float) -> np.ndarray:
    """Маска листов книги: крупные светлые малоцветные области с печатью внутри (свитер, ковёр, руки — нет).

    Args:
        paper: Закрытая бумага (серое без печати).
        chroma: Хрома.
        blue: Синева: ``128 − b`` Lab (8 бит), сглаженная.
        paper_level: Яркость бумаги кадра (p95).

    Returns:
        Булева маска.
    """
    light = (paper >= PAGE_LIGHT_SHARE * paper_level) & (chroma < LINE_MAX_CHROMA) & (blue < PAGE_MAX_BLUE)
    light = light.astype(np.uint8)
    light = cv2.morphologyEx(light, cv2.MORPH_OPEN, np.ones((PAGE_OPEN_PX, PAGE_OPEN_PX), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=4)
    big = stats[:, cv2.CC_STAT_AREA] >= PAGE_MIN_AREA_SHARE * light.size
    big[0] = False
    page = big[labels].astype(np.uint8)
    page = cv2.morphologyEx(page, cv2.MORPH_CLOSE, np.ones((PAGE_CLOSE_PX, PAGE_CLOSE_PX), np.uint8))
    # Заливка дыр: всё, что не связано с рамкой через «не лист», — лист.
    holes = 1 - page
    flood = holes.copy()
    mask = np.zeros((page.shape[0] + 2, page.shape[1] + 2), np.uint8)
    for x, y in ((0, 0), (page.shape[1] - 1, 0), (0, page.shape[0] - 1), (page.shape[1] - 1, page.shape[0] - 1)):
        if flood[y, x]:
            cv2.floodFill(flood, mask, (x, y), 2)
    return (page > 0) | (flood == 1)


def frame_band(shape: tuple[int, int], width: int = FRAME_BORDER_PX) -> np.ndarray:
    """Полоса шириной ``width`` вдоль рамки кадра.

    Args:
        shape: ``(H, W)``.
        width: Ширина, px.

    Returns:
        Булева маска.
    """
    band = np.zeros(shape, bool)
    band[:width] = band[-width:] = True
    band[:, :width] = band[:, -width:] = True
    return band


def contact_of(edge: np.ndarray, box: np.ndarray, centre: np.ndarray) -> Contact:
    """С какой стороны глифа край: по ходу края в окрестности глифа и по положению края относительно центра.

    Край в рамке глифа с запасом раскладывается по главным осям: вытянут по горизонтали — касание снизу или сверху
    (срез вдоль строки, в том числе конец изогнутой строки у нижнего края), по вертикали — слева или справа.

    Args:
        edge: Край (uint8 0/1).
        box: Рамка глифа ``(x, y, w, h)``.
        centre: Центр глифа ``(x, y)``.

    Returns:
        Сторона касания.
    """
    pad = CONTACT_PAD_PX
    x, y, w, h = (int(v) for v in box)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    ys, xs = np.nonzero(edge[y0 : y + h + pad, x0 : x + w + pad])
    if not len(xs):
        return Contact.BELOW
    dx = xs.mean() + x0 - centre[0]
    dy = ys.mean() + y0 - centre[1]
    horizontal = np.var(xs) >= np.var(ys)
    if horizontal:
        return Contact.BELOW if dy > 0 else Contact.ABOVE
    return Contact.RIGHT if dx > 0 else Contact.LEFT


def outside_region(thick: np.ndarray, gray: np.ndarray, chroma: np.ndarray, paper_level: float) -> np.ndarray:
    """«Снаружи листа»: толстые тёмные структуры, задевающие рамку кадра (коврик, крышка), и цветное тёмное (крышка).

    Тёмная печать соседней страницы (жирный заголовок, плашка) рамки не задевает и «снаружи» не считается — так
    край над корешком (соседняя страница с печатью за краем) отличается от наружного края (коврик за краем).

    Args:
        thick: Толстые тёмные структуры (uint8 0/1).
        gray: Серое.
        chroma: Хрома.
        paper_level: Яркость бумаги кадра (p95).

    Returns:
        Булева маска.
    """
    count, labels = cv2.connectedComponents(thick, connectivity=8)
    keep = np.zeros(count, bool)
    keep[np.unique(labels[frame_band(thick.shape, FRAME_BORDER_PX + 2)])] = True
    keep[0] = False
    return keep[labels] | ((chroma > COVER_CHROMA) & (gray < COVER_DARK_SHARE * paper_level))


def beyond_is_outside(outside: np.ndarray, glyph_box: np.ndarray, line_box: np.ndarray, to_right: bool) -> bool:
    """За краем по ходу строки коврик или крышка (наружный край), а не другая страница (корешок).

    Args:
        outside: «Снаружи листа» (:func:`outside_region`).
        glyph_box: Рамка крайнего глифа ``(x, y, w, h)``.
        line_box: Рамка строки ``(x, y, w, h)``.
        to_right: Край справа от строки (иначе слева).

    Returns:
        True — наружный край листа.
    """
    gx, _, gw, _ = (int(v) for v in glyph_box)
    _, ly, _, lh = (int(v) for v in line_box)
    height, width = outside.shape
    if to_right:
        xa, xb = gx + gw + BEYOND_GAP_PX, gx + gw + BEYOND_LEN_PX
    else:
        xa, xb = gx - BEYOND_LEN_PX, gx - BEYOND_GAP_PX
    xa, xb = max(0, xa), min(width, xb)
    ya, yb = max(0, ly - 20), min(height, ly + lh + 20)
    # Часть полосы за рамкой кадра считается «снаружи».
    missing = BEYOND_LEN_PX - BEYOND_GAP_PX - max(0, xb - xa)
    if xb <= xa:
        return True
    share = (outside[ya:yb, xa:xb].mean(axis=0).sum() + missing) / (BEYOND_LEN_PX - BEYOND_GAP_PX)
    return bool(share >= OUTER_MIN_SHARE)


def along_outside_share(outside: np.ndarray, boxes: np.ndarray, below: bool) -> float:
    """Доля «снаружи» в полосе за краем под (над) глифами, касающимися края вдоль строки.

    Args:
        outside: «Снаружи листа» (:func:`outside_region`).
        boxes: Рамки касающихся глифов ``(N, 4)``.
        below: Край ниже строки (иначе выше).

    Returns:
        Доля пикселей полосы, лежащих снаружи листа (0 — полоса пуста).
    """
    height = outside.shape[0]
    total, count = 0, 0
    for x, y, w, h in (tuple(int(v) for v in box) for box in boxes):
        if below:
            ya, yb = y + h + ALONG_BEYOND_PX[0], y + h + ALONG_BEYOND_PX[1]
        else:
            ya, yb = y - ALONG_BEYOND_PX[1], y - ALONG_BEYOND_PX[0]
        ya, yb = max(0, ya), min(height, yb)
        if yb <= ya:
            continue
        part = outside[ya:yb, x : x + w]
        total += part.size
        count += int(part.sum())
    return count / total if total else 0.0


def analyze(bgr: np.ndarray, keep_masks: bool = False) -> FrameResult:
    """Ищет на кадре строки своей печати, срезанные краем листа.

    Шаги: тёмное → длинные структуры (коврик, тени, крышка) → край листа (граница толстых структур и ступеньки
    закрытой бумаги) → глифы (тёмное вне структур, размера буквы) → строки (смыкание глифов вдоль, разрыв на краю)
    → касания глифов с краем и сторона касания → вердикт строки: срез вдоль (низ или верх: касаются многие глифы)
    или срез конца (крайний глиф строки упирается в край по ходу строки; наружу или к корешку).

    Args:
        bgr: Кадр BGR в ½ разрешения.
        keep_masks: Вернуть маски для оверлея.

    Returns:
        Разбор кадра.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    chroma = chroma_map(bgr)
    height, width = gray.shape
    paper_level = float(np.percentile(gray, 95))
    # Доля цветного среди не тёмного: голубая обложка РИСК и МТС.
    light = gray > MAT_SHARE * paper_level
    colour_share = float((chroma[light] > COVER_CHROMA).mean()) if light.any() else 0.0

    dark = dark_mask(gray, paper_level)
    thick, thin = dark_structures(dark)
    structures = thick | thin
    excluded = skin_zone(bgr) | frame_band(gray.shape)
    edge_along, edge, paper = edge_mask(gray, thick, thin, excluded)
    # Синева (128 − b Lab): бумага пака белая или желтоватая (b ≥ −4), голубая крышка подшивки — синее.
    blue = 128.0 - cv2.blur(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 2], (9, 9)).astype(np.float32)
    page = page_mask(paper, chroma, blue, paper_level)
    outside = outside_region(thick, gray, chroma, paper_level)

    # Глифы: тёмное вне длинных структур, размера буквы.
    count, labels, stats, centres = cv2.connectedComponentsWithStats(dark & (1 - structures), connectivity=8)
    w, h, area = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    is_glyph = (h >= GLYPH_MIN_H) & (h <= GLYPH_MAX_H) & (w <= GLYPH_MAX_W) & (area >= GLYPH_MIN_AREA)
    is_glyph[0] = False
    glyph = is_glyph[labels]
    near_edge = cv2.dilate(edge, np.ones((TOUCH_PX, TOUCH_PX), np.uint8)) > 0
    near_along = cv2.dilate(edge_along, np.ones((TOUCH_PX, TOUCH_PX), np.uint8)) > 0
    # Глифы, касающиеся края «вдоль» (без тонких вертикальных линий).
    along_ids = set(np.unique(labels[glyph & near_along]).tolist())
    touching_ids = np.unique(labels[glyph & near_edge])

    # Строки: смыкание глифов вдоль, но не через край листа (строки двух страниц у корешка не сливаются).
    smeared = cv2.morphologyEx(glyph.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, LINE_GAP_PX), np.uint8))
    smeared[near_edge & ~glyph] = 0
    _, line_labels, line_stats, _ = cv2.connectedComponentsWithStats(smeared, connectivity=8)
    cx = np.clip(np.round(centres[:, 0]).astype(int), 0, width - 1)
    cy = np.clip(np.round(centres[:, 1]).astype(int), 0, height - 1)
    line_of = np.where(is_glyph, line_labels[cy, cx], 0)
    per_line = np.bincount(line_of[is_glyph], minlength=len(line_stats))

    # Касающиеся глифы по строкам.
    by_line: dict[int, list[int]] = {}
    for gid in touching_ids:
        if line_of[gid] and per_line[line_of[gid]] >= LINE_MIN_GLYPHS:
            by_line.setdefault(int(line_of[gid]), []).append(int(gid))

    lines: list[CutLine] = []
    cut_ids: list[int] = []
    for lid, gids in by_line.items():
        side = classify_line(
            lid,
            gids,
            along_ids,
            line_of,
            stats,
            centres,
            line_stats,
            edge,
            paper,
            page,
            glyph,
            outside,
            gray,
            chroma,
            paper_level,
        )
        if side is None:
            continue
        lx, ly, lw, lh = (int(v) for v in line_stats[lid, :4])
        lines.append(CutLine(side, (lx, ly, lw, lh), int(per_line[lid]), len(gids)))
        cut_ids.extend(gids)

    masks = None
    if keep_masks:
        masks = FrameMasks(
            edge=edge.astype(bool),
            touching=np.isin(labels, touching_ids) & glyph,
            cut=np.isin(labels, np.array(cut_ids, int)) & glyph,
        )
    return FrameResult(lines=lines, glyphs=int(is_glyph.sum()), colour_share=colour_share, masks=masks)


def classify_line(
    lid: int,
    gids: list[int],
    along_ids: set[int],
    line_of: np.ndarray,
    stats: np.ndarray,
    centres: np.ndarray,
    line_stats: np.ndarray,
    edge: np.ndarray,
    paper: np.ndarray,
    page: np.ndarray,
    glyph_pixels: np.ndarray,
    outside: np.ndarray,
    gray: np.ndarray,
    chroma: np.ndarray,
    paper_level: float,
) -> Side | None:
    """Вердикт строки с касаниями: срезана ли она краем листа и каким.

    Args:
        lid: Метка строки.
        gids: Метки глифов строки, касающихся края.
        along_ids: Метки глифов, касающихся края без тонких вертикальных линий (годятся для среза вдоль).
        line_of: Строка каждой компоненты (0 — не глиф или вне строк).
        stats: Статистика компонент (``connectedComponentsWithStats``).
        centres: Центры компонент.
        line_stats: Статистика строк.
        edge: Край листа (для концов строк, с тонкими вертикальными линиями).
        paper: Закрытая бумага (серое без печати).
        page: Маска листов (:func:`page_mask`).
        glyph_pixels: Пиксели глифов (bool).
        outside: «Снаружи листа» (:func:`outside_region`).
        gray: Серое.
        chroma: Хрома.
        paper_level: Яркость бумаги кадра (p95).

    Returns:
        Сторона среза или None — строка не срезана (или это не строка текста на бумаге).
    """
    lx, ly, lw, lh = (int(v) for v in line_stats[lid, :4])
    # Строка текста вытянута вдоль; столбик сора у торца блока — нет.
    if lw < LINE_MIN_ASPECT * lh or lw < LINE_MIN_LEN_PX:
        return None
    members = np.nonzero(line_of == lid)[0]
    # Высота глифов строки — по не касающимся края (у срезанных краем видна только часть высоты); если почти все
    # касаются, нижняя граница не проверяется.
    whole = np.setdiff1d(members, gids)
    heights = stats[whole if len(whole) >= LINE_MIN_GLYPHS else members, cv2.CC_STAT_HEIGHT]
    median_h = float(np.median(heights))
    low = LINE_GLYPH_H[0] if len(whole) >= LINE_MIN_GLYPHS else GLYPH_MIN_H
    if not low <= median_h <= LINE_GLYPH_H[1]:
        return None
    # Строка лежит на бумаге: не на обложке, крышке, свитере.
    box = (slice(ly, ly + lh), slice(lx, lx + lw))
    # Бумага вокруг строки — по рамке, расширенной на высоту строки вверх и вниз, и по верхнему квартилю: у строки,
    # срезанной краем, половина окрестности — коврик.
    around = (slice(max(0, ly - lh), ly + 2 * lh), box[1])
    if np.percentile(paper[around], 75) < LINE_PAPER_SHARE * paper_level:
        return None
    if np.percentile(chroma[around], 25) > LINE_MAX_CHROMA:
        return None
    if page[box].mean() < LINE_ON_PAGE_SHARE:
        return None
    ink = glyph_pixels[box]
    if ink.any() and np.median(chroma[box][ink]) > LINE_INK_MAX_CHROMA:
        return None
    # Фон строки: не глифы и не коврик (у срезанной строки часть рамки — коврик за краем).
    background = gray[box][~ink & (gray[box] >= MAT_SHARE * paper_level)]
    if background.size == 0 or np.median(background) < LINE_WHITE_SHARE * paper_level:
        return None
    contacts = {gid: contact_of(edge, stats[gid, :4], centres[gid]) for gid in gids}
    along = [gid for gid, c in contacts.items() if c in (Contact.BELOW, Contact.ABOVE) and gid in along_ids]
    if len(along) >= ALONG_MIN_TOUCH and len(along) >= ALONG_MIN_SHARE * len(members):
        below = sum(contacts[gid] is Contact.BELOW for gid in along) * 2 >= len(along)
        if along_outside_share(outside, stats[along, :4], below) >= ALONG_OUTSIDE_SHARE:
            return Side.BOTTOM if below else Side.TOP
        return None
    # Срез конца: крайний глиф строки касается края по ходу строки.
    xs = centres[members, 0]
    # Строка одна (не слипшиеся соседние строки) — иначе «крайний глиф» бывает соринкой у края.
    if lh > END_MAX_LINE_HEIGHT * median_h:
        return None
    for gid, c in contacts.items():
        first = centres[gid, 0] <= xs.min() + 1 and c is Contact.LEFT
        last = centres[gid, 0] >= xs.max() - 1 and c is Contact.RIGHT
        # Крайний глиф — буква строки, а не соринка, крап обреза или пятнышко у кромки.
        if stats[gid, cv2.CC_STAT_HEIGHT] < END_MIN_GLYPH_SHARE * median_h:
            continue
        if first or last:
            is_outer = beyond_is_outside(outside, stats[gid, :4], line_stats[lid, :4], bool(last))
            return Side.OUTER if is_outer else Side.SPINE
    return None
