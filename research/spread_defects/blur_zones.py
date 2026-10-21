"""Карта смаза по всему кадру: зоны плиток с полутоновым ореолом или двоением штриха, а не одна точка пика.

Зачем. Оценка ``motion_score`` даёт одну плитку пика кепстра, и отсмотр по крупу вокруг неё пропускает смаз в другом
месте кадра. На «Хозяйстве и праве» 1982/12 IMG_1208 (2026-10-21) пик кепстра пришёлся на середину правой полосы, а
нечитаемый смаз сидел в её верхнем углу. Кепстр там почти молчит: смаз не двоит штрих, а размывает его. Зато доля
полутонов среди краски там 0.40–0.70 против 0.23 по кадру. Кроме того, угол — край текста, а ``interior_mask``
отбрасывает краевые плитки.

Как считается. Маска плиток ослаблена: текстовая плитка, у которой текстовых не меньше 3 из 4 соседей, с долей
краски от ``ZONE_INK``, без длинных прямых и не в столбце сгиба. Доля полутонов плитки делится на медиану таких
плиток по всему выпуску, а не по кадру: так виден и смаз всего кадра. Карта — медиана в окне 3×3 по плиткам маски.
Зона — 8-связная группа плиток, где карта ореола не ниже ``MID_ZONE_REL`` или кепстр двоит штрих (``agree`` из
``smoothed_maps`` не ниже ``AGREE_ZONE`` при ореоле не ниже ``AGREE_MID_REL``, правило пака-4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from research.spread_defects.motion import BODY_LINES, TILE, MidtoneTiles, MotionTiles, smoothed_maps

# Плитка карты: доля краски не ниже этой. Ниже основного набора (``BODY_INK`` = 0.11): угол полосы и короткие
# строки абзаца дают 0.07–0.10, а смаз там виден не хуже. Поля страницы и обрез блока — 0.03–0.06.
ZONE_INK = 0.07
# Плитка карты: текстовых соседей (из 4) не меньше этого. Внешний край блока (обрез, стопка листов) — 2 соседа.
ZONE_NEIGHBOURS = 3
# Окно 3×3 карты: плиток маски не меньше этого (в углу текста окно неполное).
ZONE_WINDOW = 4
# Порог карты ореола (медиана окна к медиане выпуска) для зоны. Калибровка — «Хозяйство и право» 1981–1982,
# 1 186 кадров (2026-10-21): медиана пика по кадрам 1.27, p75 1.35; пять тяжёлых смазов 2.1–2.7.
MID_ZONE_REL = 1.6
# Зона по кепстру: ``agree`` окна не ниже этого при ореоле не ниже ``AGREE_MID_REL`` (правило пака-4 для кадра).
AGREE_ZONE, AGREE_MID_REL = 6.0, 1.15
# Зона из меньшего числа плиток не выдаётся: одиночная плитка — заголовок или фото.
ZONE_MIN_TILES = 2

# Отсев полиграфии и заголовков, которые дают ореол без смаза (отсмотр 37 кадров с зонами, «Хозяйство и право» 1981–1982,
# 2026-10-21). Растр фото и репродукций: доля краски в мелких точках (связные компоненты меньше ``DOT_AREA`` px)
# у растра 0.21–1.00, у смазанного текста 0.01–0.08. Цветные плашки и карикатуры: доля светлых насыщенных пикселей
# у плашек 0.07–0.66, у текста на бумаге (в том числе смазанного) 0–0.02.
DOT_AREA = 60  # px полного кадра при 434–450 dpi: точка растра 5–30 px, буква 150+ px
DOT_SHARE = 0.15
COLOUR_S, COLOUR_V = 70, 110  # насыщенность и яркость HSV (0..255) «светлого цветного» пикселя
COLOUR_SHARE = 0.05
# Краска для поиска точек — темнее бумаги (p90) на эту долю размаха «бумага − ядро (p3)».
DOT_INK = 0.35
# Заголовки: крупный жирный кегль даёт полутоновый край у букв без всякого смаза (1991/02 IMG_1100, IMG_1117 —
# ложные зоны, отсмотр пользователя 2026-10-21). Зона — заголовок, если медианная высота букв в ней не ниже этой
# доли медианы по кольцу плиток вокруг зоны на той же полосе: заголовок выделяется скачком кегля против соседнего
# набора. Сравнение с медианой всего кадра не годится: на поднятой странице буквы плавно крупнее на 10–30 % из-за
# перспективы (1981/02 IMG_1243, 1265, 1981/10 IMG_0414 — настоящий смаз, отсеивался бы как заголовок).
HEADING_HEIGHT = 1.2
# Ширина кольца вокруг зоны, плиток; в кольце должно быть не меньше стольких плиток с буквами.
HEADING_RING, HEADING_RING_MIN = 2, 4
# Второй признак — кегль зоны к основному (мелкому) набору полосы: нижняя четверть высот букв по плиткам полосы.
# Нужен, когда крупным кеглем набрана половина полосы и кольцо целиком лежит на нём (1991/02 IMG_1100: постановление,
# 1.46). Поднятая страница даёт до 1.22 (рост кегля к краю листа), поэтому порог выше, чем у кольца.
HEADING_BODY, BODY_QUANTILE = 1.35, 25
# Буква для высоты: компонента не мельче точки растра и ниже доли плитки (длинные линейки и рамки не в счёт);
# медиана — если таких компонент в плитке не меньше этого.
LETTER_MAX_SHARE, LETTER_MIN_COUNT = 0.6, 4


class Side(Enum):
    """Полоса разворота относительно сгиба."""

    LEFT = "L"
    RIGHT = "R"


class Band(Enum):
    """Пояс высоты кадра, где лежит центр зоны (подпись для отсмотра)."""

    TOP = "верх"
    MIDDLE = "середина"
    BOTTOM = "низ"


@dataclass(frozen=True)
class ZoneMaps:
    """Карты кадра в сетке плиток ``TILE``.

    Attributes:
        mask: bool[gy, gx] — плитки карты (ослабленная маска текста).
        mid_rel: float32[gy, gx] — медиана доли полутонов в окне 3×3 к медиане выпуска (NaN вне маски или при
            неполном окне).
        agree: float32[gy, gx] — согласованная оценка кепстра ``smoothed_maps`` (0 вне основного набора).
        frame_rel: Медиана доли полутонов плиток карты кадра к медиане выпуска (смаз всего кадра, обложки).
    """

    mask: np.ndarray
    mid_rel: np.ndarray
    agree: np.ndarray
    frame_rel: float


@dataclass(frozen=True)
class BlurZone:
    """Одна зона смаза кадра.

    Attributes:
        row0, row1, col0, col1: Охват зоны в плитках (включительно).
        tiles: Число плиток зоны.
        side: Полоса, где больше плиток зоны.
        band: Пояс высоты по центру зоны.
        mid_peak: Максимум карты ореола в зоне.
        mid_median: Медиана карты ореола по зоне.
        agree_peak: Максимум оценки кепстра в зоне (0 — кепстр молчит).
        by_midtone: В зоне есть плитки по ореолу.
        by_cepstrum: В зоне есть плитки по двоению (кепстр).
        peak_row, peak_col: Плитка с максимумом ореола (центр крупа для отсмотра).
    """

    row0: int
    row1: int
    col0: int
    col1: int
    tiles: int
    side: Side
    band: Band
    mid_peak: float
    mid_median: float
    agree_peak: float
    by_midtone: bool
    by_cepstrum: bool
    peak_row: int
    peak_col: int

    def box_px(self) -> tuple[int, int, int, int]:
        """Охват зоны в px полного кадра: (x0, y0, x1, y1)."""
        return self.col0 * TILE, self.row0 * TILE, (self.col1 + 1) * TILE, (self.row1 + 1) * TILE


@dataclass(frozen=True)
class PrintFlags:
    """Признаки полиграфии плитки, дающей ореол без смаза.

    Attributes:
        dots: Доля краски в мелких связных компонентах (растр).
        colour: Доля светлых насыщенных пикселей (цветная плашка, цветной рисунок).
        letter_height: Медианная высота букв плитки, px (NaN — букв мало); сравнивается с медианой кадра.
    """

    dots: float
    colour: float
    letter_height: float = float("nan")

    def is_print(self) -> bool:
        """Плитка — растр или цвет, а не текст на бумаге."""
        return self.dots >= DOT_SHARE or self.colour >= COLOUR_SHARE


def print_flags(rgb_tile: np.ndarray) -> PrintFlags:
    """Признаки полиграфии одной плитки.

    Args:
        rgb_tile: Плитка RGB uint8 полного разрешения.

    Returns:
        ``PrintFlags`` плитки.
    """
    gray = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2HSV)
    paper, core = np.percentile(gray, (90, 3))
    ink = (gray < paper - DOT_INK * max(float(paper - core), 1.0)).astype(np.uint8)
    # Площади связных компонент краски: у растра почти вся краска в точках, у текста — в буквах.
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    dots = float(areas[areas < DOT_AREA].sum()) / max(int(areas.sum()), 1)
    colour = float(((hsv[..., 1] >= COLOUR_S) & (hsv[..., 2] >= COLOUR_V)).mean())
    # Высота букв: компоненты не мельче точки растра и не длиннее доли плитки (линейки, рамки).
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    letters = heights[(areas >= DOT_AREA) & (heights < LETTER_MAX_SHARE * rgb_tile.shape[0])]
    letter_height = float(np.median(letters)) if len(letters) >= LETTER_MIN_COUNT else float("nan")
    return PrintFlags(dots, colour, letter_height)


def print_mask(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Плитки маски, которые оказались растром или цветом, и высота букв по плиткам (читаются из полного кадра).

    Args:
        rgb: Кадр RGB полного разрешения.
        mask: bool-сетка плиток, которые проверять.

    Returns:
        Пара сеток той же формы: bool — плитка полиграфии, из карты её убрать; float32 — медианная высота букв
        плитки, px (NaN вне маски и где букв мало) — для ``heading_ratio``.
    """
    out = np.zeros(mask.shape, bool)
    heights = np.full(mask.shape, np.nan, np.float32)
    for row, col in zip(*np.nonzero(mask)):
        tile = rgb[row * TILE : (row + 1) * TILE, col * TILE : (col + 1) * TILE]
        flags = print_flags(tile)
        out[row, col] = flags.is_print()
        heights[row, col] = flags.letter_height
    return out, heights


def heading_ratio(zone: BlurZone, heights: np.ndarray, fold_x: float) -> tuple[float, float]:
    """Кегль зоны к кеглю набора вокруг неё и к основному набору той же полосы.

    Args:
        zone: Зона смаза.
        heights: Высота букв по плиткам (``print_mask``), px.
        fold_x: x сгиба, px полного кадра (кольцо не переходит на другую полосу; NaN — без ограничения).

    Returns:
        Пара: медиана высоты букв в охвате зоны к медиане по кольцу ``HEADING_RING`` плиток вокруг охвата (NaN, если
        в кольце букв меньше ``HEADING_RING_MIN``) и к ``BODY_QUANTILE``-му перцентилю высот по полосе. Без букв в
        зоне — (NaN, NaN).
    """
    gy, gx = heights.shape
    rows, cols = slice(zone.row0, zone.row1 + 1), slice(zone.col0, zone.col1 + 1)
    inside = heights[rows, cols]
    ring = np.zeros(heights.shape, bool)
    ring[
        max(zone.row0 - HEADING_RING, 0) : zone.row1 + HEADING_RING + 1,
        max(zone.col0 - HEADING_RING, 0) : zone.col1 + HEADING_RING + 1,
    ] = True
    ring[rows, cols] = False
    # Только та полоса, где лежит зона: у соседней свой масштаб (поднятая страница, перспектива).
    same = np.ones(gx, bool)
    if np.isfinite(fold_x):
        centers = (np.arange(gx) + 0.5) * TILE
        same = centers < fold_x if zone.side is Side.LEFT else centers > fold_x
    ring &= same[None, :]
    if not np.isfinite(inside).any():
        return float("nan"), float("nan")
    level = float(np.nanmedian(inside))
    around = heights[ring & np.isfinite(heights)]
    to_ring = level / float(np.median(around)) if len(around) >= HEADING_RING_MIN else float("nan")
    page = heights[:, same]
    page = page[np.isfinite(page)]
    to_body = level / float(np.percentile(page, BODY_QUANTILE)) if len(page) else float("nan")
    return to_ring, to_body


def is_heading(ratios: tuple[float, float]) -> bool:
    """Зона — заголовок или крупный набор: скачок кегля против кольца или против основного набора полосы.

    Args:
        ratios: Пара из ``heading_ratio``.

    Returns:
        True — зону не считать смазом.
    """
    to_ring, to_body = ratios
    return bool(to_ring >= HEADING_HEIGHT or to_body >= HEADING_BODY)


def zone_mask(mid: MidtoneTiles, fold_x: float) -> np.ndarray:
    """Ослабленная маска текста для карты смаза.

    Args:
        mid: Сетка полутонов кадра (``midtone_tiles`` или npz второго прохода).
        fold_x: x сгиба посередине высоты, px полного кадра; NaN — столбец сгиба не исключается.

    Returns:
        bool-сетка плиток: текстовая, не меньше ``ZONE_NEIGHBOURS`` текстовых соседей из 4, краски не меньше
        ``ZONE_INK``, на длинных прямых не больше ``BODY_LINES``, центр дальше плитки от сгиба.
    """
    cross = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
    # Число текстовых соседей по кресту; за краем сетки соседей нет.
    neighbours = cv2.filter2D(mid.text.astype(np.float32), -1, cross, borderType=cv2.BORDER_CONSTANT)
    centers = (np.arange(mid.text.shape[1]) + 0.5) * TILE
    # Столбец сгиба: тень корешка и изгиб листа дают полутона без всякого смаза.
    away = np.abs(centers - fold_x) > TILE if np.isfinite(fold_x) else np.ones_like(centers, bool)
    return (
        mid.text & (neighbours >= ZONE_NEIGHBOURS) & (mid.ink >= ZONE_INK) & (mid.lines <= BODY_LINES) & away[None, :]
    )


def issue_midtone_level(mids: list[MidtoneTiles], masks: list[np.ndarray]) -> float:
    """Медиана доли полутонов по плиткам карты всех кадров выпуска (печать и свет у выпуска общие).

    Args:
        mids: Сетки полутонов кадров выпуска.
        masks: Маски ``zone_mask`` тех же кадров в том же порядке.

    Returns:
        Медиана или NaN, если плиток нет.
    """
    values = [m.mid[k] for m, k in zip(mids, masks) if k.any()]
    return float(np.median(np.concatenate(values))) if values else float("nan")


def window_median_map(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Медиана сетки по плиткам маски в окне 3×3 вокруг каждой плитки маски.

    Args:
        values: Сетка значений.
        mask: bool-сетка плиток, которые идут в медиану и получают значение.

    Returns:
        float32-сетка; NaN вне маски и там, где в окне меньше ``ZONE_WINDOW`` плиток маски.
    """
    out = np.full(values.shape, np.nan, np.float32)
    for row, col in zip(*np.nonzero(mask)):
        rows, cols = slice(max(row - 1, 0), row + 2), slice(max(col - 1, 0), col + 2)
        chosen = values[rows, cols][mask[rows, cols]]
        if len(chosen) >= ZONE_WINDOW:
            out[row, col] = float(np.median(chosen))
    return out


def zone_maps(
    tiles: MotionTiles, mid: MidtoneTiles, fold_x: float, issue_level: float, excluded: np.ndarray | None = None
) -> ZoneMaps:
    """Карты ореола и кепстра кадра.

    Args:
        tiles: Сетка кепстра кадра (первый проход).
        mid: Сетка полутонов кадра (второй проход).
        fold_x: x сгиба, px полного кадра.
        issue_level: Медиана доли полутонов по выпуску (``issue_midtone_level``).
        excluded: bool-сетка плиток полиграфии (``print_mask``), которые убрать из карты; None — ничего.

    Returns:
        ``ZoneMaps``; без плиток карты или без уровня выпуска карта ореола — сплошь NaN.
    """
    mask = zone_mask(mid, fold_x)
    _, _, agree = smoothed_maps(tiles)
    if excluded is not None:
        mask = mask & ~excluded
        agree = np.where(excluded, 0.0, agree).astype(np.float32)
    if not mask.any() or not np.isfinite(issue_level) or issue_level <= 0:
        return ZoneMaps(mask, np.full(mask.shape, np.nan, np.float32), agree, float("nan"))
    relative = mid.mid / issue_level
    return ZoneMaps(mask, window_median_map(relative, mask), agree, float(np.median(relative[mask])))


def blur_zones(maps: ZoneMaps, fold_x: float) -> list[BlurZone]:
    """Зоны смаза кадра по картам.

    Args:
        maps: Карты кадра.
        fold_x: x сгиба, px полного кадра (делит полосы; NaN — середина кадра).

    Returns:
        Зоны по убыванию ``mid_peak``; пустой список, если зон нет.
    """
    mid_rel = np.nan_to_num(maps.mid_rel, nan=0.0)
    by_mid = mid_rel >= MID_ZONE_REL
    by_cep = (maps.agree >= AGREE_ZONE) & (mid_rel >= AGREE_MID_REL)
    hit = (by_mid | by_cep).astype(np.uint8)
    count, labels = cv2.connectedComponents(hit, connectivity=8)
    gy, gx = hit.shape
    fold = fold_x if np.isfinite(fold_x) else gx * TILE / 2
    zones = []
    for label in range(1, count):
        part = labels == label
        if part.sum() < ZONE_MIN_TILES:
            continue
        rows, cols = np.nonzero(part)
        # Плитка максимума ореола — центр крупа на листе отсмотра.
        best = int(np.argmax(mid_rel[rows, cols]))
        left = int(((cols + 0.5) * TILE < fold).sum())
        center = (rows.mean() + 0.5) / gy
        zones.append(
            BlurZone(
                row0=int(rows.min()),
                row1=int(rows.max()),
                col0=int(cols.min()),
                col1=int(cols.max()),
                tiles=int(part.sum()),
                side=Side.LEFT if left * 2 >= len(cols) else Side.RIGHT,
                band=Band.TOP if center < 1 / 3 else Band.MIDDLE if center < 2 / 3 else Band.BOTTOM,
                mid_peak=float(mid_rel[rows, cols].max()),
                mid_median=float(np.median(mid_rel[rows, cols])),
                agree_peak=float(maps.agree[rows, cols].max()),
                by_midtone=bool(by_mid[part].any()),
                by_cepstrum=bool(by_cep[part].any()),
                peak_row=int(rows[best]),
                peak_col=int(cols[best]),
            )
        )
    return sorted(zones, key=lambda z: z.mid_peak, reverse=True)
