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
from enum import Enum

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks import WORK_DPI
from ocr_utils.page_layout import mm_to_px, px_to_mm
from ocr_utils.page_layout.tables.ruling import find_lines
from ocr_utils.page_layout.orientation.detectors.ink_axis import _smear, glyph_mask

# Лента: высота и шаг (мм бумаги). 40 мм — 8–12 строк корпуса, 20 мм — половинное перекрытие.
BAND_MM = 40.0
BAND_STEP_MM = 20.0
# Лента без краски в этой доле столбцов не голосует (пустой низ страницы).
BAND_MIN_INK_SHARE = 0.05
# Межколонник уже этого — не межколонник (в наборе журнала межколонник 3–6 мм).
MIN_GUTTER_MM = 2.5
# Колонка уже этого — не колонка. 12 мм: в таблицах есть узкие колонки чисел («1000 кг»),
# и без них часть текста оставалась вне блоков (1971/10 с.93).
MIN_COLUMN_MM = 12.0
# Межколонник должен быть виден не меньше чем в стольких лентах подряд: одна лента — случайное
# совпадение пустот в паре строк.
MIN_GUTTER_BANDS = 2
# Доля краски в полосе межколонника, выше которой он считается перегороженным.
BLOCK_INK_SHARE = 0.3
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
# Доля столбцов межколонника с краской НА ВЫСОТЕ строки, выше которой через него идёт краска самой
# строки (отточия таблицы, заголовок через межколонник), и запреты межколонника на неё не
# распространяются. Замер по паку: обычные страницы p99 = 0.23, страница с отточиями — 0.69.
DOT_FILL_SHARE = 0.4
# Столько отточий, идущих сквозь полосу-кандидат, превращают её из межколонника в поле точек.
DOT_GUTTER_LEADERS = 3
# По скольким высотам меряется ширина межколонника внутри зоны.
ZONE_BOUND_SAMPLES = 24
# Минимальная ширина полосы запрета по вертикальной линейке (px рабочей копии).
RULE_SEPARATOR_MIN_PX = 3
# Минимальная полуширина окна пробы по вертикали (px рабочей копии): у тонких строк высота мала.
MIN_PROBE_HEIGHT_PX = 4.0
# Крупный набор (:func:`heading_mask`): сгусток маски текста выше медианы страницы во столько раз
# (как ``segment.BODY_HEIGHT_RATIO``). Межколонник новых режимов не продлевается сквозь строку, где
# такой набор подходит к его полосе с обеих сторон ближе ``HEADING_REACH_PX`` (px рабочей копии, ~2 мм).
HEADING_HEIGHT_RATIO = 1.6
HEADING_REACH_PX = 12
# Короткие межколонники (:func:`short_gutters`, режим ``GutterMode.SHORT``). Лента мелкая — три
# строки корпуса (шаг строк пака 3.3–4 мм), шаг — строка: межколонник в 3 строки должен целиком
# лечь хотя бы в одну ленту, не задев заголовка над ним.
SHORT_BAND_MM = 12.0
SHORT_BAND_STEP_MM = 3.0
# Подтверждение короткого межколонника — не высотой, а ВЫРОВНЕННОЙ КРОМКОЙ колонки за ним (табулятор
# Tesseract, кромка текста у kraken ``compute_colseps_conv``): начала строк справа от пустоты (или
# концы слева) стоят на одной x. У «реки» пробелов в наборе по формату и у пробела заголовка слова
# за пустотой начинаются где попало.
SHORT_MIN_LINES = 3  # строк текста с каждой стороны
SHORT_ALIGN_MM = 0.8  # допуск выравнивания кромки (``alignment.ALIGN_TOL_MM``; у Tesseract ≈ 0.8 мм)
SHORT_ALIGN_SHARE = 0.75  # доля строк, чья кромка у медианы (абзацный отступ — не больше четверти)
SHORT_EDGE_REACH_MM = 10.0  # кромку ищем не дальше этого от края пустоты
SHORT_ROW_MIN_PX = 3  # строка текста в маске — не ниже этого (пиксели рабочей копии)
# Кандидаты соседних мелких лент — один межколонник, только если их середины ближе этого (мм): с
# допуском длинных (``GUTTER_MATCH_MM`` = 4 мм) цепочка ползёт по «рекам» пробелов через всю полосу.
SHORT_MATCH_MM = 1.5
# Короткий межколонник — только продолжение найденного: середина его пустоты не дальше этого (мм) от
# середины длинного межколонника на ближнем конце.
SHORT_PRECEDENT_MM = 3.0
# Кромка колонки прямая с наклоном не больше этого (dx/dy, ≈ 2°): наклон полосы пака — до 1.5°.
SHORT_MAX_SLOPE = 0.035
# Отрезок ломаной межколонника, у концов которого нет общей пустоты, делится пополам, пока не
# станет ниже этого (px рабочей копии): ниже полоса запрета уже не нужна.
MIN_SEPARATOR_HEIGHT_PX = 4


class GutterMode(Enum):
    """Как межколонники превращаются в запреты сцепки и какие межколонники ищутся.

    * ``LEGACY`` — прежний ход: один прямоугольник на межколонник по ОБЩЕЙ части ломаной по x. На
      наклонной полосе общая часть пуста, и межколонник молча выпадает из запретов — строки
      сшиваются через него (1966/02 IMG_0076_1L: наклон ~1°, ломаная 426..451 → 452..479);
    * ``SEGMENTED`` — прямоугольник на каждый отрезок ломаной между соседними узлами;
    * ``SHORT`` — как ``SEGMENTED``, плюс КОРОТКИЕ межколонники (:func:`short_gutters`): двухколонный
      фрагмент в 3–8 строк под заголовком во всю ширину или над ним короче двух лент и прежним ходом
      не находится вовсе (1969/03 IMG_0147_1L: семь строк под «Реорганизация помогла»).
    """

    LEGACY = "legacy"
    SEGMENTED = "segmented"
    SHORT = "short"


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


def gutter_filled(ink: np.ndarray, gx0: float, gx1: float, cy: float, height: float, k: float) -> float:
    """Доля столбцов межколонника, в которых есть краска НА ВЫСОТЕ строки.

    Признак «через межколонник идёт краска самой строки»: так выглядят отточия таблицы
    (1971/10 с.93 — точки заполняют больше половины столбцов) и заголовок, набранный через
    межколонник. У обычного двухколоночного набора на высоте строки межколонник пуст: замер по
    паку дал p99 = 0.23 против 0.69 на странице с отточиями.

    Args:
        ink: Краска рендера ``RENDER_DPI`` (ненулевое — краска).
        gx0, gx1: Границы межколонника на этой высоте, пиксели рабочей копии.
        cy: Ордината середины строки, пиксели рабочей копии.
        height: Высота строки, пиксели рабочей копии.
        k: Во сколько раз рендер крупнее рабочей копии.

    Returns:
        Доля от 0 до 1; 0, если полоса пуста или вырождена.
    """
    half = max(height, MIN_PROBE_HEIGHT_PX)
    y0, y1 = int((cy - half) * k), int((cy + half) * k) + 1
    x0, x1 = int(gx0 * k), int(gx1 * k) + 1
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(ink.shape[0], y1), min(ink.shape[1], x1)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    band = ink[y0:y1, x0:x1] > 0
    return float(band.any(axis=0).mean())


def gutters_of(
    gray: np.ndarray, dpi: float = WORK_DPI, leaders: list | None = None, mode: GutterMode = GutterMode.SHORT
) -> list[Gutter]:
    """Локальные межколонники страницы.

    Args:
        gray: Серая рабочая копия страницы.
        dpi: Её разрешение.
        leaders: Отточия страницы: полоса, сквозь которую они идут, межколонником не считается.
        mode: ``GutterMode.SHORT`` — искать ещё и короткие межколонники (:func:`short_gutters`);
            остальные режимы ищут межколонники одинаково.

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
    # В новых режимах продление останавливается на строке крупного набора, подходящей к полосе с обеих
    # сторон: пробел заголовка над межколонником — не межколонник (см. :func:`heading_mask`).
    headings = heading_mask(mask, dpi) if mode is not GutterMode.LEGACY else None
    extended = [_extend(gutter, mask, headings=headings) for gutter in _confirm(_merge_gutters(found, dpi), mask, dpi)]
    if mode is GutterMode.SHORT:
        extended = sorted(extended + short_gutters(mask, dpi, extended, headings), key=lambda gutter: gutter.centre)
    # Отбраковка поля отточий — ПОСЛЕ продления: до него диапазон высот кандидата ещё короткий, и
    # отточия в него не попадают (на 1971/10 с.93 ложный межколонник видел 0 отточий вместо 15).
    return [gutter for gutter in extended if not leaders or _dotted(gutter, leaders) < DOT_GUTTER_LEADERS]


def _band_runs(mask: np.ndarray, bands: list[tuple[int, int]], min_gutter: int) -> list[tuple[int, int, int, int]]:
    """Пустые полосы шире ``min_gutter`` в каждой ленте, между первой и последней краской ленты.

    Args:
        mask: Маска текста рабочей копии.
        bands: Ленты ``(y0, y1)``.
        min_gutter: Минимальная ширина пустоты (пиксели).

    Returns:
        Находки ``(x0, x1, y0, y1)`` по одной на пустоту ленты.
    """
    width = mask.shape[1]
    found: list[tuple[int, int, int, int]] = []
    for y0, y1 in bands:
        columns = mask[y0:y1].any(axis=0)
        if columns.mean() < BAND_MIN_INK_SHARE:
            continue
        inked = np.nonzero(columns)[0]
        if inked.size == 0:
            continue
        inside = np.zeros(width, dtype=bool)
        inside[inked[0] : inked[-1] + 1] = True
        for gx0, gx1 in _runs((~columns) & inside, min_gutter):
            found.append((gx0, gx1, y0, y1))
    return found


def _fine_bands(height: int, dpi: float) -> list[tuple[int, int]]:
    """Мелкие ленты ``(y0, y1)`` для коротких межколонников: ``SHORT_BAND_MM`` с шагом ``SHORT_BAND_STEP_MM``."""
    band = mm_to_px(SHORT_BAND_MM, dpi)
    step = max(1, mm_to_px(SHORT_BAND_STEP_MM, dpi))
    if height <= band:
        return [(0, height)]
    bands = [(y0, y0 + band) for y0 in range(0, height - band + 1, step)]
    if bands[-1][1] < height:
        bands.append((height - band, height))
    return bands


def _text_lines(rows: np.ndarray) -> list[tuple[int, int]]:
    """Строки текста по профилю строк пикселей: пробеги ``True`` не ниже ``SHORT_ROW_MIN_PX``."""
    return _runs(rows, SHORT_ROW_MIN_PX)


def _edge_aligned(mask: np.ndarray, gutter: Gutter, dpi: float, right_side: bool) -> tuple[int, bool]:
    """Кромка колонки у межколонника: сколько строк текста подходит к нему с одной стороны и выровнена ли она.

    Для каждой строки текста (пробег строк пикселей маски с краской в полосе ``SHORT_EDGE_REACH_MM``
    у края пустоты) берётся ближайшая к межколоннику краска: справа — начало строки, слева — конец.
    Кромка выровнена, если у ``SHORT_ALIGN_SHARE`` строк она не дальше ``SHORT_ALIGN_MM`` от ПРЯМОЙ
    с наклоном не больше ``SHORT_MAX_SLOPE``. Мерить от самой ломаной кандидата нельзя: она
    повторяет изгибы «реки» пробелов, и кромка реки относительно неё выходит ровной.

    Args:
        mask: Маска текста рабочей копии.
        gutter: Кандидат в межколонники.
        dpi: Разрешение рабочей копии.
        right_side: ``True`` — правая колонка (начала строк), ``False`` — левая (концы строк).

    Returns:
        Пара: число строк текста с этой стороны и выровнена ли их кромка.
    """
    reach = mm_to_px(SHORT_EDGE_REACH_MM, dpi)
    y0, y1 = int(gutter.y0), int(gutter.y1) + 1
    height, width = mask.shape
    y0, y1 = max(0, y0), min(height, y1)
    # Абсолютная x ближайшей к межколоннику краски в каждой строке пикселей; nan — краски в полосе нет.
    xs = np.full(y1 - y0, np.nan)
    for index, y in enumerate(range(y0, y1)):
        if right_side:
            edge = max(0, int(round(gutter.x1_at(y))))
            inked = np.nonzero(mask[y, edge : min(width, edge + reach)])[0]
            if inked.size:
                xs[index] = edge + inked[0]
        else:
            edge = min(width, int(round(gutter.x0_at(y))))
            start = max(0, edge - reach)
            inked = np.nonzero(mask[y, start:edge])[0]
            if inked.size:
                xs[index] = start + inked[-1]
    lines = _text_lines(~np.isnan(xs))
    if not lines:
        return 0, False
    # Кромка строки — ближайшая к межколоннику краска по всей её высоте: справа минимум, слева максимум.
    pick = np.nanmin if right_side else np.nanmax
    edges = np.array([pick(xs[a:b]) for a, b in lines])
    middles = np.array([(a + b) / 2.0 for a, b in lines]) + y0
    tolerance = mm_to_px(SHORT_ALIGN_MM, dpi)
    return len(lines), _aligned(middles, edges, tolerance)


def _aligned(ys: np.ndarray, xs: np.ndarray, tolerance: float) -> bool:
    """Лежат ли ``SHORT_ALIGN_SHARE`` точек кромки не дальше ``tolerance`` от одной прямой ``x = a + b·y``.

    Прямая — лучшая по числу попавших точек среди прямых через пары точек с наклоном не больше
    ``SHORT_MAX_SLOPE`` (точек — строки фрагмента, их единицы и десятки, перебор дешёвый); плюс
    вертикаль через медиану. Абзацный отступ и короткая концевая строка в допуск не попадают, и
    доля ``SHORT_ALIGN_SHARE`` их прощает.

    Args:
        ys: Ординаты строк.
        xs: Кромка каждой строки.
        tolerance: Допуск (пиксели).

    Returns:
        ``True``, если кромка выровнена.
    """
    best = int((np.abs(xs - np.median(xs)) <= tolerance).sum())
    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            if ys[j] == ys[i]:
                continue
            slope = (xs[j] - xs[i]) / (ys[j] - ys[i])
            if abs(slope) > SHORT_MAX_SLOPE:
                continue
            best = max(best, int((np.abs(xs - (xs[i] + slope * (ys - ys[i]))) <= tolerance).sum()))
    return best >= SHORT_ALIGN_SHARE * len(ys)


def _continues(item: tuple[int, int, int, int], known: list[Gutter], tolerance: float) -> bool:
    """Продолжает ли пустота ленты ``(x0, x1, y0, y1)`` уже найденный межколонник за его концом.

    Середина пустоты должна лежать не дальше ``tolerance`` от середины межколонника на его ближнем
    конце (``x0_at``/``x1_at`` за концами отдают концевые значения; наклон полосы пака на десятках
    миллиметров сдвигает межколонник на миллиметр — в допуск укладывается), а сама лента — вне
    высот, где межколонник жив: там пустота уже найдена прежним ходом.

    Args:
        item: Пустота мелкой ленты.
        known: Межколонники прежнего хода.
        tolerance: Допуск по x (пиксели).

    Returns:
        ``True``, если пустота — продолжение какого-то из них.
    """
    x0, x1, y0, y1 = item
    middle_y = (y0 + y1) / 2.0
    centre = (x0 + x1) / 2.0
    for gutter in known:
        if gutter.alive_at(middle_y):
            continue
        if abs(centre - (gutter.x0_at(middle_y) + gutter.x1_at(middle_y)) / 2.0) <= tolerance:
            return True
    return False


def _behind_obstacle(candidate: Gutter, known: list[Gutter], mask: np.ndarray, tolerance: float) -> bool:
    """Отделён ли кандидат от длинного межколонника, который он продолжает, преградой поперёк полосы.

    Короткий межколонник — это продолжение сетки колонок ЗА заголовком или линейкой, которые
    перегородили длинный. Кандидат, вплотную примыкающий к длинному или заходящий на его высоты,
    — не продолжение, а пустота рядом с ним: пробел заголовка ровно над межколонником (1967/07
    IMG_0051_1L — заголовок резался пополам). Преграда — хоть одна строка пикселей между ними, где
    полоса ДЛИННОГО межколонника перегорожена (:func:`_row_blocked`).

    Args:
        candidate: Короткий межколонник.
        known: Межколонники прежнего хода.
        mask: Маска текста рабочей копии.
        tolerance: Допуск по x для «продолжает» (пиксели).

    Returns:
        ``True``, если нашёлся длинный межколонник на той же x, отделённый от кандидата преградой.
    """
    centre = candidate.centre
    # Кандидат, заходящий на высоты хоть одного длинного межколонника на той же x, — пустота рядом с
    # ним, а не продолжение (даже если с другой стороны его отделяет преграда от другого длинного).
    for gutter in known:
        overlaps = candidate.y0 <= gutter.y1 and gutter.y0 <= candidate.y1
        middle = (max(candidate.y0, gutter.y0) + min(candidate.y1, gutter.y1)) / 2.0
        if overlaps and abs(centre - (gutter.x0_at(middle) + gutter.x1_at(middle)) / 2.0) <= tolerance:
            return False
    for gutter in known:
        if candidate.y0 > gutter.y1:
            gap = range(int(gutter.y1) + 1, int(candidate.y0))
        elif candidate.y1 < gutter.y0:
            gap = range(int(candidate.y1) + 1, int(gutter.y0))
        else:
            continue  # заходит на высоты длинного межколонника
        end = gutter.y1 if candidate.y0 > gutter.y1 else gutter.y0
        if abs(centre - (gutter.x0_at(end) + gutter.x1_at(end)) / 2.0) > tolerance:
            continue
        if any(_row_blocked(mask, gutter, y) for y in gap):
            return True
    return False


def short_gutters(
    mask: np.ndarray, dpi: float, known: list[Gutter], headings: np.ndarray | None = None
) -> list[Gutter]:
    """Короткие межколонники — продолжения найденных за заголовком или линейкой, с выровненной кромкой колонки.

    Прежний ход затравливает межколонник только с двух лент по 40 мм: двухколонный фрагмент в 3–8
    строк под заголовком во всю ширину (или над ним) туда не ложится, межколонник не находится, и
    строки обеих колонок сшиваются через него (1969/03 IMG_0147_1L — семь строк под «Реорганизация
    помогла», 1967/05 IMG_0095_2R, 1975/12 IMG_0141_2R). Здесь ленты мелкие (``SHORT_BAND_MM``).

    Кандидат берётся только на продолжении уже найденного межколонника (``SHORT_PRECEDENT_MM``):
    конец статьи над заголовком и начало следующей под ним держат сетку колонок полосы. Без этого
    условия «реки» пробелов в наборе по формату проходят проверку кромки (замер на 1976/04
    IMG_0036_1L и 1967/05 IMG_0095_2R: десятки ложных межколонников). Кандидаты соседних лент
    сливаются :func:`_merge_gutters` с узким допуском ``SHORT_MATCH_MM`` и подтверждаются не высотой
    текста, а числом строк (``SHORT_MIN_LINES`` с каждой стороны) и выровненной кромкой хотя бы
    одной из колонок — так Tesseract находит табуляторы.

    Args:
        mask: Маска текста рабочей копии (:func:`text_mask`).
        dpi: Её разрешение.
        known: Межколонники прежнего хода.
        headings: Маска крупного набора (:func:`heading_mask`): продление на ней останавливается.

    Кандидат должен быть отделён от длинного межколонника преградой (:func:`_behind_obstacle`) и
    продлевается только по совсем пустой полосе.

    Returns:
        Новые межколонники, продлённые по пустой краске (:func:`_extend`).
    """
    if not known:
        return []
    found = _band_runs(mask, _fine_bands(mask.shape[0], dpi), mm_to_px(MIN_GUTTER_MM, dpi))
    tolerance = mm_to_px(SHORT_PRECEDENT_MM, dpi)
    found = [item for item in found if _continues(item, known, tolerance)]
    max_gutter = mm_to_px(MAX_GUTTER_MM, dpi)
    out: list[Gutter] = []
    for candidate in _merge_gutters(found, dpi, min_bands=1, match_mm=SHORT_MATCH_MM):
        if candidate.x1 - candidate.x0 > max_gutter:
            continue
        left_lines, left_aligned = _edge_aligned(mask, candidate, dpi, right_side=False)
        right_lines, right_aligned = _edge_aligned(mask, candidate, dpi, right_side=True)
        if min(left_lines, right_lines) < SHORT_MIN_LINES or not (left_aligned or right_aligned):
            continue
        # Продление — только по СОВСЕМ пустой полосе: пробел заголовка над межколонником прежний
        # допуск (``BLOCK_INK_SHARE``) пропускал, и короткий межколонник резал заголовок.
        extended = _extend(candidate, mask, share=0.0, headings=headings)
        if _behind_obstacle(extended, known, mask, tolerance):
            out.append(extended)
    return out


def _merge_gutters(
    found: list[tuple[int, int, int, int]],
    dpi: float,
    min_bands: int = MIN_GUTTER_BANDS,
    match_mm: float = GUTTER_MATCH_MM,
) -> list[Gutter]:
    """Слить межколонники соседних лент в вертикальные отрезки.

    Args:
        found: Пустоты лент ``(x0, x1, y0, y1)``.
        dpi: Разрешение рабочей копии.
        min_bands: Группа из меньшего числа лент отбрасывается.
        match_mm: Пустоты соседних лент — один межколонник, если их середины ближе этого (мм).

    Returns:
        Межколонники-ломаные слева направо.
    """
    tolerance = mm_to_px(match_mm, dpi)
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
        if len(group) < min_bands:
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


def _zone_bounds(gutter: Gutter, y0: float, y1: float, middle: float) -> tuple[float, float]:
    """Границы колонок по межколоннику на всю высоту зоны: самые широкие колонки без потери текста.

    Брать ширину межколонника в СЕРЕДИНЕ зоны нельзя: она меняется по высоте. На 1971/10 с.93
    межколонник широк против таблицы (x 606–726) и узок ниже неё, где справа стоит текст в две
    графы шириной (x 609–636). Зона одна на всю высоту, и по середине правая колонка начиналась с
    x 725 — одиннадцать строк, начинающихся с x 641, не помещались ни в одну колонку и терялись
    вместе со своим блоком.

    Поэтому левая колонка тянется до САМОЙ ПРАВОЙ левой границы межколонника по высоте зоны, а
    правая начинается с САМОЙ ЛЕВОЙ правой границы: так ни одна строка зоны не обрезается. Если
    межколонник так сильно уходит вбок, что границы переворачиваются (трапеция), берутся значения
    в середине зоны — как раньше.

    Args:
        gutter: Межколонник.
        y0, y1: Границы зоны по высоте.
        middle: Середина зоны (запасной вариант).

    Returns:
        Пара ``(конец левой колонки, начало правой)``.
    """
    top, bottom = max(y0, gutter.y0), min(y1, gutter.y1)
    if bottom <= top:
        return gutter.x0_at(middle), gutter.x1_at(middle)
    ys = np.linspace(top, bottom, ZONE_BOUND_SAMPLES)
    left = max(gutter.x0_at(y) for y in ys)
    right = min(gutter.x1_at(y) for y in ys)
    if right <= left:
        return gutter.x0_at(middle), gutter.x1_at(middle)
    return left, right


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
        bounds = [0.0] + [x for g in alive for x in _zone_bounds(g, y0, y1, middle)] + [float(width)]
        columns = tuple(
            (int(bounds[i]), int(bounds[i + 1]))
            for i in range(0, len(bounds) - 1, 2)
            if bounds[i + 1] - bounds[i] >= min_column
        )
        if columns:
            zones.append(Zone(y0=y0, y1=y1, columns=columns))
    return zones


def _extend(
    gutter: Gutter, mask: np.ndarray, share: float = BLOCK_INK_SHARE, headings: np.ndarray | None = None
) -> Gutter:
    """Продлить межколонник вверх и вниз, пока полоса пуста.

    Ленты голосования грубые (40 мм), и межколонник «начинается» ниже, чем на самом деле: первые
    строки колонок попадают в зону над ним и собираются в один блок во всю ширину страницы
    (1973/07 с.77). Продление идёт по самой краске: пока в полосе межколонника на этой высоте
    пусто — межколонник жив. Останавливает его первая же строка, набранная через него
    (колонтитул, заголовок).

    Args:
        gutter: Межколонник.
        mask: Маска текста рабочей копии.
        share: Доля краски в полосе, с которой строка пикселей её перегораживает
            (:func:`_row_blocked`); 0 — любая краска.
        headings: Маска крупного набора (:func:`heading_mask`) или ``None``: строка, где крупный
            набор подходит к полосе с обеих сторон, тоже её перегораживает (:func:`_heading_across`).

    Returns:
        Продлённый межколонник.
    """
    height = mask.shape[0]
    top = int(gutter.y0)
    while top > 0 and not _stops(mask, gutter, top - 1, share, headings):
        top -= 1
    bottom = int(gutter.y1)
    while bottom < height - 1 and not _stops(mask, gutter, bottom + 1, share, headings):
        bottom += 1
    points = list(gutter.points)
    if top < points[0][0]:
        points.insert(0, (float(top), points[0][1], points[0][2]))
    if bottom > points[-1][0]:
        points.append((float(bottom), points[-1][1], points[-1][2]))
    return Gutter(points=tuple(points))


def _stops(mask: np.ndarray, gutter: Gutter, y: int, share: float, headings: np.ndarray | None) -> bool:
    """Останавливает ли строка пикселей ``y`` продление межколонника: полоса перегорожена или через неё идёт заголовок."""
    if _row_blocked(mask, gutter, y, share):
        return True
    return headings is not None and _heading_across(headings, gutter, y)


def heading_mask(mask: np.ndarray, dpi: float) -> np.ndarray:
    """Маска сгустков крупного набора: выше медианы сгустков страницы в ``HEADING_HEIGHT_RATIO`` раз.

    Сгусток маски текста (буквы, сомкнутые RLSA) — слово или кусок строки; у заголовка он выше, чем у
    корпуса. Используется, чтобы межколонник не продлевался сквозь пробел заголовка, стоящий ровно
    над ним (1968/04 IMG_0050_1L «Сборник статей | о снабжении», 1974/10 IMG_0041_2R «Полезное |
    пособие»): прежнее условие «полоса пуста» такой пробел пропускает.

    Args:
        mask: Маска текста рабочей копии (:func:`text_mask`).
        dpi: Её разрешение.

    Returns:
        Булева маска того же размера.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    # Медиана — по сгусткам не уже миллиметра: точки и запятые её не тянут вниз.
    usable = heights[widths >= mm_to_px(1.0, dpi)]
    if count <= 1 or usable.size == 0:
        return np.zeros(mask.shape, dtype=bool)
    tall = np.zeros(count, dtype=bool)
    tall[1:] = heights >= HEADING_HEIGHT_RATIO * float(np.median(usable))
    return tall[labels]


def _heading_across(headings: np.ndarray, gutter: Gutter, y: int) -> bool:
    """Подходит ли крупный набор к полосе межколонника на высоте ``y`` с обеих сторон (ближе ``HEADING_REACH_PX``)."""
    width = headings.shape[1]
    x0 = int(max(0, min(width, gutter.x0_at(y))))
    x1 = int(max(0, min(width, gutter.x1_at(y))))
    left = headings[y, max(0, x0 - HEADING_REACH_PX) : x0]
    right = headings[y, x1 : min(width, x1 + HEADING_REACH_PX)]
    return bool(left.any() and right.any())


def _row_blocked(mask: np.ndarray, gutter: Gutter, y: int, share: float = BLOCK_INK_SHARE) -> bool:
    """Перегорожена ли полоса межколонника на высоте ``y``.

    Не «есть ли хоть пиксель», а «есть ли краска шире ``BLOCK_INK_SHARE`` полосы»: одиночная
    точка, дефект бумаги или свисающая запятая соседней колонки не должны обрывать межколонник
    (иначе верхние строки колонок попадают в зону над ним и собираются в блок во всю ширину —
    1973/07 с.88).
    """
    x0 = int(max(0, min(mask.shape[1] - 1, gutter.x0_at(y))))
    x1 = int(max(x0 + 1, min(mask.shape[1], gutter.x1_at(y))))
    return bool(mask[y, x0:x1].mean() > share)


def _dotted(gutter: Gutter, leaders: list) -> int:
    """Сколько отточий задевает полосу межколонника на его высоте.

    Поле точек «. . . . .» пустое по краске, и его середина выглядит как межколонник: на 1971/10
    с.93 таких ложных межколонников четыре (ширина 11–21 мм), и таблица рассыпалась по колонкам на
    девятнадцать блоков. Отточия же там идут насквозь. Замер по паку: ложные задевают 9–13
    отточий, настоящие межколонники — ни одного.

    Args:
        gutter: Кандидат в межколонники.
        leaders: Отточия страницы (``leaders.Leader``).

    Returns:
        Число отточий, пересекающих полосу кандидата на его высоте.
    """
    count = 0
    for leader in leaders:
        if not gutter.alive_at(leader.y):
            continue
        if leader.x1 > gutter.x0_at(leader.y) and leader.x0 < gutter.x1_at(leader.y):
            count += 1
    return count


def _confirm(gutters: list[Gutter], mask: np.ndarray, dpi: float) -> list[Gutter]:
    """Оставить те межколонники, у которых по обе стороны есть текст на их же высоте.

    Пустое поле справа от текста (там лишь номер страницы) и пустота внутри врезки иначе
    становятся «межколонниками» и режут блок пополам (1967/10 с.63). Проверка: в диапазоне
    высот межколонника доля строк с краской слева и справа — не меньше ``SIDE_ROWS_SHARE``.
    Слишком широкая пустота (шире ``MAX_GUTTER_MM``) — это поле, а не межколонник. Полоса, сквозь
    которую идут отточия, отбраковывается отдельно, уже после продления (см. :func:`_dotted`).

    Args:
        gutters: Кандидаты в межколонники.
        mask: Маска текста рабочей копии.
        dpi: Разрешение рабочей копии.

    Returns:
        Подтверждённые межколонники.
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
    centre = (x0 + x1) / 2.0
    for gutter in gutters:
        if not gutter.alive_at(y):
            continue
        gx0, gx1 = gutter.x0_at(y), gutter.x1_at(y)
        # Сравнение по СЕРЕДИНЕ строки, а не по её концам: последняя буква или дефис нередко
        # заходят в межколонник, и по концам он переставал считаться границей — окно поиска края
        # уходило в соседнюю колонку и брало её край (1973/07 с.88).
        if gx1 <= centre:
            left = max(left, gx1)
        elif gx0 >= centre:
            right = min(right, gx0)
    return left, right


def inside_gutter(gutters: list[Gutter], x0: float, x1: float, y: float) -> bool:
    """Лежит ли отрезок ``[x0, x1]`` ЦЕЛИКОМ внутри живого на высоте ``y`` межколонника.

    Такой кусок не принадлежит ни одной колонке: по геометрии он ровно в пустоте между ними.
    Сам по себе он бывает и сором, и началом строки соседней графы, заехавшим в межколонник
    (1971/10 с.93: межколонник 605..726, а строка правой графы начинается с 641), — решает это
    уже разбор блоков, здесь только признак.

    Args:
        gutters: Локальные межколонники-ломаные.
        x0, x1: Концы отрезка (пиксели рабочей копии).
        y: Ордината, на которой смотрим.

    Returns:
        ``True``, если нашёлся живой межколонник, накрывающий отрезок целиком.
    """
    return any(gutter.alive_at(y) and gutter.x0_at(y) <= x0 and x1 <= gutter.x1_at(y) for gutter in gutters)


def mark_cut_lines(
    axes: list, gutters: list[Gutter], dpi: float = WORK_DPI, ink: np.ndarray | None = None, k: float = 2.0
) -> list[bool]:
    """Какие оси — куски широкой строки, набранной ЧЕРЕЗ межколонник (заголовок во всю ширину).

    Args:
        axes: Оси строк страницы (``lines.LineAxis``).
        gutters: Локальные межколонники.
        dpi: Разрешение рабочей копии.
        ink: Краска рендера ``RENDER_DPI``; нужна, чтобы отличить отточия от настоящего разреза.
        k: Во сколько раз рендер крупнее рабочей копии.

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
            # Строка набрана ЧЕРЕЗ межколонник, только если она выходит за него с ОБЕИХ сторон.
            # Иначе последняя буква строки, заехавшая в межколонник на пару пикселей, объявляла
            # обычную строку колонки «широкой», и та уходила в отдельный блок во всю ширину
            # страницы поверх обеих колонок (1973/08 с.85).
            through = axis.x0 < gx0 and axis.x1 > gx1
            # Через межколонник может идти краска самой строки — отточия таблицы. Такая строка
            # не «разрезанная»: она целиком принадлежит своей колонке (1971/10 с.93).
            dotted = ink is not None and gutter_filled(ink, gx0, gx1, axis.cy, axis.height, k) >= DOT_FILL_SHARE
            if through and not dotted and overlap >= GUTTER_OVERLAP_SHARE * max(1.0, gx1 - gx0):
                out[i] = True
    return out


def separators_for_segmentation(
    gutters: list[Gutter], mode: GutterMode = GutterMode.SHORT
) -> list[tuple[int, int, int, int]]:
    """Межколонники полосами ``(x0, x1, y0, y1)`` для сегментации строк.

    По x берётся часть, заведомо лежащая внутри межколонника на всех высотах полосы, — сборка
    кусков строки через неё не перескочит. По y отдаётся ВЫСОТА, на которой межколонник живёт:
    заголовок над колонками идёт выше него, и резать заголовок по чужой пустоте нельзя
    (1973/06 с.65: «ОСНОВЫ ЭКОНОМИКИ…» рвалось пополам там, где ниже начинался межколонник
    основного текста).

    Args:
        gutters: Межколонники-ломаные.
        mode: ``LEGACY`` — одна полоса на межколонник по общей части ВСЕЙ ломаной (на наклонной
            полосе она пуста, и межколонник выпадает); ``SEGMENTED`` — по полосе на каждый отрезок
            ломаной (см. :func:`_segment_separators`).

    Returns:
        Полосы запрета в пикселях рабочей копии.
    """
    out: list[tuple[int, int, int, int]] = []
    for gutter in gutters:
        if mode is GutterMode.LEGACY:
            x0 = max(point[1] for point in gutter.points)
            x1 = min(point[2] for point in gutter.points)
            if x1 > x0:
                out.append((int(x0), int(x1), int(gutter.y0), int(gutter.y1)))
            continue
        out += _segment_separators(gutter)
    return out


def _segment_separators(gutter: Gutter) -> list[tuple[int, int, int, int]]:
    """Полосы запрета по отрезкам ломаной межколонника.

    Между соседними узлами граница межколонника идёт линейно, поэтому общая часть двух узлов по x
    лежит внутри межколонника на всей высоте отрезка. Если у концов отрезка общей части нет
    (межколонник сдвинулся вбок больше своей ширины), отрезок делится пополам по высоте — концы
    половинок берутся с ломаной (``x0_at`` / ``x1_at``).

    Args:
        gutter: Межколонник-ломаная.

    Returns:
        Полосы ``(x0, x1, y0, y1)`` сверху вниз; соседние стыкуются по y.
    """
    out: list[tuple[int, int, int, int]] = []
    # Стек отрезков по высоте: у каждого берутся границы межколонника на его концах.
    pending = [(a[0], b[0]) for a, b in zip(gutter.points[:-1], gutter.points[1:]) if b[0] > a[0]]
    if not pending:
        pending = [(gutter.y0, gutter.y1)]
    while pending:
        top, bottom = pending.pop(0)
        x0 = max(gutter.x0_at(top), gutter.x0_at(bottom))
        x1 = min(gutter.x1_at(top), gutter.x1_at(bottom))
        if int(x1) > int(np.ceil(x0)):
            out.append((int(np.ceil(x0)), int(x1), int(top), int(np.ceil(bottom))))
        elif bottom - top >= 2 * MIN_SEPARATOR_HEIGHT_PX:
            middle = (top + bottom) / 2.0
            pending[:0] = [(top, middle), (middle, bottom)]
    return out


def rule_separators(gray: np.ndarray, dpi: float = WORK_DPI) -> list[tuple[int, int, int, int]]:
    """Вертикальные линейки полосами ``(x0, x1, y0, y1)`` — такие же запреты сцепки, как межколонник.

    Графы таблицы разделены не пустотой, а ЛИНЕЙКОЙ, и пустого межколонника между ними может не
    быть вовсе: числа графы стоят вплотную к черте. Поэтому детектор межколонников такую границу
    не видит, а сборка строки спокойно перешагивает через неё и склеивает текст левой графы с
    числами правой (1971/10 с.93). Линейку же видно прямо — тем же морфологическим детектором,
    которым таблицы находит ``page_layout``.

    Args:
        gray: Серая рабочая копия страницы.
        dpi: Её разрешение.

    Returns:
        Полосы запрета по каждой вертикальной линейке, в пикселях рабочей копии.
    """
    lines = find_lines(gray, int(round(dpi)))
    out: list[tuple[int, int, int, int]] = []
    for item in lines.vertical:
        box = item.box
        # Линейка тонкая, поэтому полоса расширяется до минимальной ширины: иначе разрыв между
        # кусками, стоящими вплотную к черте, может её «перепрыгнуть».
        pad = max(0, (RULE_SEPARATOR_MIN_PX - (box.x1 - box.x0)) // 2)
        out.append((int(box.x0 - pad), int(box.x1 + pad), int(box.y0), int(box.y1)))
    return out


def column_width_mm(span: tuple[int, int], dpi: float = WORK_DPI) -> float:
    """Ширина колонки в мм."""
    return px_to_mm(span[1] - span[0], dpi)


__all__ = [
    "Gutter",
    "GutterMode",
    "Zone",
    "zones_of",
    "bounds_at",
    "inside_gutter",
    "column_width_mm",
    "gutters_of",
    "rule_separators",
    "mark_cut_lines",
    "gutter_filled",
    "separators_for_segmentation",
    "text_mask",
]
