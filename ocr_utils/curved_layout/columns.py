"""Колонки страницы: ЛОКАЛЬНЫЕ межколонники — вертикальные пустоты, живущие на части высоты.

Межколонник на нашем материале не обязан идти через всю страницу: врезка «50 лет / ЦИФРЫ
РОСТА» (1967/10 с.63) занимает правую четверть только вверху, а ниже текст идёт на всю ширину.
Поэтому межколонники ищутся по лентам высоты и собираются в вертикальные отрезки
``(x0, x1, y0, y1)``; колонка строки — промежуток между теми межколонниками, которые живы на её
высоте.

Голосование по лентам, а не профиль всей страницы: заголовок во всю ширину иначе заклеивает
межколонник (готовый ``line_fit.column_separators_banded`` на 1973/06 с.65 голосов не набирал).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.curved_layout import WORK_DPI
from ocr_utils.page_layout import mm_to_px, px_to_mm
from ocr_utils.page_layout.orientation.detectors.ink_axis import _smear, glyph_mask

# Лента: высота и шаг (мм бумаги). 40 мм — 8–12 строк корпуса, 20 мм — половинное перекрытие.
BAND_MM = 40.0
BAND_STEP_MM = 20.0
# Лента без краски в этой доле столбцов не голосует (пустой низ страницы).
BAND_MIN_INK_SHARE = 0.05
# Межколонник уже этого — не межколонник (в наборе журнала межколонник 3–6 мм).
MIN_GUTTER_MM = 2.5
# Колонка уже этого — не колонка (узкая врезка курсивом).
MIN_COLUMN_MM = 20.0
# Межколонник должен быть виден не меньше чем в стольких лентах подряд: одна лента — случайное
# совпадение пустот в паре строк.
MIN_GUTTER_BANDS = 2
# Пустота шире этого — поле страницы, а не межколонник.
MAX_GUTTER_MM = 25.0
# Доля строк с краской по каждую сторону межколонника в его диапазоне высот и минимальная
# абсолютная высота текста с каждой стороны (мм): у поля страницы справа только номер полосы.
SIDE_ROWS_SHARE = 0.25
SIDE_TEXT_MM = 12.0
# Зона вёрстки короче этого сливается с соседней: событие «межколонник кончился» на 5 мм — шум.
MIN_ZONE_MM = 15.0
# Межколонники соседних лент считаются одним и тем же, если их середины ближе этого (мм).
GUTTER_MATCH_MM = 4.0
# Строка, заходящая в межколонник на эту долю его ширины, набрана ЧЕРЕЗ него (заголовок во всю
# ширину). Правило «кусок упирается в межколонник» отвергнуто: в плотном двухколоночном наборе
# строки соседних колонок стоят на одной высоте и начинаются сразу за межколонником — помечались
# все подряд.
GUTTER_OVERLAP_SHARE = 0.3


@dataclass(frozen=True)
class Gutter:
    """Межколонник как ЛОМАНАЯ: пустота ``(x0, x1)`` на каждой своей высоте.

    Прямой межколонник на трапеции не годится: при сильном наклоне или изгибе колонка «уплывает»
    вбок, и вертикальная граница режет не там, где надо, — в блок попадает то, что ниже уехало под
    него (номер полосы, колонтитул). Поэтому у межколонника хранится ход по лентам: ``points`` —
    ``(y, x0, x1)`` от ленты к ленте, а ``x0_at``/``x1_at`` дают границу на нужной высоте.
    """

    points: tuple[tuple[float, float, float], ...]  # (y, x0, x1), снизу вверх по y

    @property
    def y0(self) -> float:
        return self.points[0][0]

    @property
    def y1(self) -> float:
        return self.points[-1][0]

    @property
    def x0(self) -> float:
        """Левая граница в середине межколонника (для сводок и подписей)."""
        return self.x0_at((self.y0 + self.y1) / 2.0)

    @property
    def x1(self) -> float:
        """Правая граница в середине межколонника."""
        return self.x1_at((self.y0 + self.y1) / 2.0)

    @property
    def centre(self) -> float:
        return (self.x0 + self.x1) / 2.0

    def x0_at(self, y: float) -> float:
        """Левая граница межколонника на высоте ``y`` (за концами — концевое значение)."""
        ys = [point[0] for point in self.points]
        return float(np.interp(y, ys, [point[1] for point in self.points]))

    def x1_at(self, y: float) -> float:
        """Правая граница межколонника на высоте ``y``."""
        ys = [point[0] for point in self.points]
        return float(np.interp(y, ys, [point[2] for point in self.points]))

    def alive_at(self, y: float) -> bool:
        """Жив ли межколонник на высоте ``y``."""
        return self.y0 <= y <= self.y1


def _bands(height: int, dpi: float) -> list[tuple[int, int]]:
    """Ленты высоты ``(y0, y1)`` с половинным перекрытием."""
    band = mm_to_px(BAND_MM, dpi)
    step = max(1, mm_to_px(BAND_STEP_MM, dpi))
    if height <= band:
        return [(0, height)]
    bands = [(y0, y0 + band) for y0 in range(0, height - band + 1, step)]
    if bands[-1][1] < height:
        bands.append((height - band, height))
    return bands


def _runs(flags: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """Полосы подряд идущих ``True`` длиной не меньше ``min_length``."""
    out: list[tuple[int, int]] = []
    start = None
    for x, flag in enumerate(flags):
        if flag and start is None:
            start = x
        if not flag and start is not None:
            if x - start >= min_length:
                out.append((start, x))
            start = None
    if start is not None and len(flags) - start >= min_length:
        out.append((start, len(flags)))
    return out


def text_mask(gray: np.ndarray) -> np.ndarray:
    """Маска текста рабочей копии: глифы, сомкнутые по горизонтали (RLSA)."""
    return _smear(glyph_mask(gray), horizontal=True) > 0


def gutters_of(gray: np.ndarray, dpi: float = WORK_DPI) -> list[Gutter]:
    """Локальные межколонники страницы.

    Args:
        gray: Серая рабочая копия страницы.
        dpi: Её разрешение.

    Returns:
        Межколонники с диапазоном высот, слева направо. Поля страницы сюда не входят: пустоты
        ищутся только между первой и последней краской ленты.
    """
    mask = text_mask(gray)
    height, width = mask.shape
    min_gutter = mm_to_px(MIN_GUTTER_MM, dpi)
    found: list[tuple[int, int, int, int]] = []  # (x0, x1, y0, y1) по одной ленте
    for y0, y1 in _bands(height, dpi):
        columns = mask[y0:y1].any(axis=0)
        if columns.mean() < BAND_MIN_INK_SHARE:
            continue
        inked = np.nonzero(columns)[0]
        if inked.size == 0:
            continue
        inside = np.zeros(width, dtype=bool)
        inside[inked[0] : inked[-1] + 1] = True
        empty = (~columns) & inside
        for gx0, gx1 in _runs(empty, min_gutter):
            found.append((gx0, gx1, y0, y1))
    # Сначала подтверждение текстом по обе стороны (по «рабочему» диапазону лент), и только потом
    # продление по пустой краске: иначе пустой хвост над колонкой рушит долю строк с текстом.
    return [_extend(gutter, mask) for gutter in _confirm(_merge_gutters(found, dpi), mask, dpi)]


def _merge_gutters(found: list[tuple[int, int, int, int]], dpi: float) -> list[Gutter]:
    """Слить межколонники соседних лент в вертикальные отрезки."""
    tolerance = mm_to_px(GUTTER_MATCH_MM, dpi)
    groups: list[list[tuple[int, int, int, int]]] = []
    for item in sorted(found, key=lambda entry: ((entry[0] + entry[1]) / 2.0, entry[2])):
        centre = (item[0] + item[1]) / 2.0
        placed = False
        for group in groups:
            last = group[-1]
            same_x = abs(centre - (last[0] + last[1]) / 2.0) <= tolerance
            # Ленты идут внахлёст, поэтому «соседняя» — та, что начинается не позже конца прошлой.
            if same_x and item[2] <= last[3]:
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
    out: list[Gutter] = []
    for group in groups:
        if len(group) < MIN_GUTTER_BANDS:
            continue
        # Ломаная по лентам: у каждой ленты своя пара границ, узел ставится в её середину; концы
        # продлеваются до верха первой и низа последней ленты.
        nodes = sorted(((item[2] + item[3]) / 2.0, float(item[0]), float(item[1])) for item in group)
        top, bottom = min(item[2] for item in group), max(item[3] for item in group)
        points = [(float(top), nodes[0][1], nodes[0][2])]
        points += [node for node in nodes if top < node[0] < bottom]
        points.append((float(bottom), nodes[-1][1], nodes[-1][2]))
        out.append(Gutter(points=tuple(points)))
    return sorted(out, key=lambda gutter: gutter.centre)


@dataclass(frozen=True)
class Zone:
    """Зона вёрстки: полоса высоты ``(y0, y1)``, внутри которой набор колонок постоянен."""

    y0: int
    y1: int
    columns: tuple[tuple[int, int], ...]


def zones_of(gutters: list[Gutter], height: int, width: int, dpi: float = WORK_DPI) -> list[Zone]:
    """Зоны вёрстки страницы: границы там, где межколонник появляется или кончается.

    Врезка живёт на части высоты (1967/10 с.63: «50 лет / ЦИФРЫ РОСТА» — правая четверть только
    вверху), поэтому число колонок меняется по высоте. Группировать строки по их собственным
    границам нельзя — границы скачут от строки к строке и дробят колонку; зона же одна на все
    строки своей полосы.

    Args:
        gutters: Локальные межколонники.
        height, width: Размеры рабочей копии.
        dpi: Её разрешение.

    Returns:
        Зоны сверху вниз; у каждой — свои колонки ``(x0, x1)``.
    """
    events = {0, height}
    for gutter in gutters:
        events.update((gutter.y0, gutter.y1))
    edges = sorted(events)
    merged = [edges[0]]
    min_zone = mm_to_px(MIN_ZONE_MM, dpi)
    for edge in edges[1:]:
        if edge - merged[-1] >= min_zone:
            merged.append(edge)
    if merged[-1] != height:
        merged[-1] = height
    zones: list[Zone] = []
    min_column = mm_to_px(MIN_COLUMN_MM, dpi)
    for y0, y1 in zip(merged[:-1], merged[1:]):
        middle = (y0 + y1) / 2.0
        alive = sorted((g for g in gutters if g.alive_at(middle)), key=lambda g: g.x0_at(middle))
        bounds = [0.0] + [x for g in alive for x in (g.x0_at(middle), g.x1_at(middle))] + [float(width)]
        columns = tuple(
            (int(bounds[i]), int(bounds[i + 1]))
            for i in range(0, len(bounds) - 1, 2)
            if bounds[i + 1] - bounds[i] >= min_column
        )
        if columns:
            zones.append(Zone(y0=y0, y1=y1, columns=columns))
    return zones


def _extend(gutter: Gutter, mask: np.ndarray) -> Gutter:
    """Продлить межколонник вверх и вниз, пока полоса пуста.

    Ленты голосования грубые (40 мм), и межколонник «начинается» ниже, чем на самом деле: первые
    строки колонок попадают в зону над ним и собираются в один блок во всю ширину страницы
    (1973/07 с.77). Продление идёт по самой краске: пока в полосе межколонника на этой высоте
    пусто — межколонник жив. Останавливает его первая же строка, набранная через него
    (колонтитул, заголовок).
    """
    height = mask.shape[0]
    top = int(gutter.y0)
    while top > 0 and not _row_blocked(mask, gutter, top - 1):
        top -= 1
    bottom = int(gutter.y1)
    while bottom < height - 1 and not _row_blocked(mask, gutter, bottom + 1):
        bottom += 1
    points = list(gutter.points)
    if top < points[0][0]:
        points.insert(0, (float(top), points[0][1], points[0][2]))
    if bottom > points[-1][0]:
        points.append((float(bottom), points[-1][1], points[-1][2]))
    return Gutter(points=tuple(points))


def _row_blocked(mask: np.ndarray, gutter: Gutter, y: int) -> bool:
    """Есть ли краска в полосе межколонника на высоте ``y``."""
    x0 = int(max(0, min(mask.shape[1] - 1, gutter.x0_at(y))))
    x1 = int(max(x0 + 1, min(mask.shape[1], gutter.x1_at(y))))
    return bool(mask[y, x0:x1].any())


def _confirm(gutters: list[Gutter], mask: np.ndarray, dpi: float) -> list[Gutter]:
    """Оставить те межколонники, у которых по обе стороны есть текст на их же высоте.

    Пустое поле справа от текста (там лишь номер страницы) и пустота внутри врезки иначе
    становятся «межколонниками» и режут блок пополам (1967/10 с.63). Проверка: в диапазоне
    высот межколонника доля строк с краской слева и справа — не меньше ``SIDE_ROWS_SHARE``.
    Слишком широкая пустота (шире ``MAX_GUTTER_MM``) — это поле, а не межколонник.
    """
    out: list[Gutter] = []
    max_gutter = mm_to_px(MAX_GUTTER_MM, dpi)
    for gutter in gutters:
        if gutter.x1 - gutter.x0 > max_gutter:
            continue
        y0, y1 = int(gutter.y0), int(gutter.y1)
        band = mask[y0:y1]
        if band.size == 0:
            continue
        # Текст считается по обе стороны от ЛОМАНОЙ, а не от прямой: на наклонной странице прямая
        # уезжает от межколонника и «видит» текст не с той стороны.
        left_rows = np.zeros(band.shape[0], bool)
        right_rows = np.zeros(band.shape[0], bool)
        for index in range(band.shape[0]):
            y = y0 + index
            left = int(max(0, min(mask.shape[1], gutter.x0_at(y))))
            right = int(max(0, min(mask.shape[1], gutter.x1_at(y))))
            left_rows[index] = band[index, :left].any()
            right_rows[index] = band[index, right:].any()
        enough = mm_to_px(SIDE_TEXT_MM, dpi)
        sides_ok = (
            left_rows.mean() >= SIDE_ROWS_SHARE
            and right_rows.mean() >= SIDE_ROWS_SHARE
            and left_rows.sum() >= enough
            and right_rows.sum() >= enough
        )
        if sides_ok:
            out.append(gutter)
    return out


def bounds_at(gutters: list[Gutter], x0: float, x1: float, y: float, width: int) -> tuple[float, float]:
    """Границы колонки для строки ``[x0, x1]`` НА ЕЁ ВЫСОТЕ: ближайшие живые межколонники."""
    left, right = 0.0, float(width)
    for gutter in gutters:
        if not gutter.alive_at(y):
            continue
        gx0, gx1 = gutter.x0_at(y), gutter.x1_at(y)
        if gx1 <= x0 + 1:
            left = max(left, gx1)
        elif gx0 >= x1 - 1:
            right = min(right, gx0)
    return left, right


def mark_cut_lines(axes: list, gutters: list[Gutter], dpi: float = WORK_DPI) -> list[bool]:
    """Какие оси — куски широкой строки, набранной ЧЕРЕЗ межколонник (заголовок во всю ширину).

    Args:
        axes: Оси строк страницы (``lines.LineAxis``).
        gutters: Локальные межколонники.
        dpi: Разрешение рабочей копии.

    Returns:
        Список признаков по осям в том же порядке.
    """
    out = [False] * len(axes)
    for gutter in gutters:
        for i, axis in enumerate(axes):
            if not gutter.alive_at(axis.cy):
                continue
            gx0, gx1 = gutter.x0_at(axis.cy), gutter.x1_at(axis.cy)
            overlap = min(axis.x1, gx1) - max(axis.x0, gx0)
            if overlap >= GUTTER_OVERLAP_SHARE * max(1.0, gx1 - gx0):
                out[i] = True
    return out


def separators_for_segmentation(gutters: list[Gutter]) -> list[tuple[int, int]]:
    """Межколонники полосами ``(x0, x1)`` для сегментации строк: ``link_spans`` про высоту не знает.

    Берётся ОБЩАЯ часть ломаной (максимум левых границ и минимум правых): полоса заведомо внутри
    межколонника на всех его высотах, и сборка кусков строки через него не перескочит.
    """
    out: list[tuple[int, int]] = []
    for gutter in gutters:
        x0 = max(point[1] for point in gutter.points)
        x1 = min(point[2] for point in gutter.points)
        if x1 > x0:
            out.append((int(x0), int(x1)))
    return out


def column_width_mm(span: tuple[int, int], dpi: float = WORK_DPI) -> float:
    """Ширина колонки в мм."""
    return px_to_mm(span[1] - span[0], dpi)


__all__ = [
    "Gutter",
    "Zone",
    "zones_of",
    "bounds_at",
    "column_width_mm",
    "gutters_of",
    "mark_cut_lines",
    "separators_for_segmentation",
    "text_mask",
]
