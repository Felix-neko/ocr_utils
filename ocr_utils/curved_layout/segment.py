"""Сегментация строк по краске в ДВУХ масштабах: корпус и крупный набор (заголовки, подписи).

Готовый конвейер ``curved_lines.detectors.line_fit`` настроен на корпус: маска глифов берёт
компоненты 5–42 px (копия 150 dpi), смыкание RLSA — 8 px. Крупный заголовок туда не попадает
совсем (буква выше 42 px), а разрядка между словами шире зазора смыкания, поэтому строка
логотипа «ЭКОНОМИЧЕСКОЕ ОБРАЗОВАНИЕ КАДРОВ» (1973/06 с.65) распадалась и теряла ось. Здесь тот
же конвейер прогоняется двумя наборами размеров, и результаты сливаются: крупный сегмент
принимается, если корпусные его не покрыли.

Межколонники приходят снаружи (:mod:`ocr_utils.curved_layout.columns`): иначе строка двух
колонок сшивается через межколонник, ведь ``link_spans`` пускает разрыв до 2.5 высот.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.curved_layout import LINKING_DEFAULT, LINKING_ZONES, RENDER_DPI, WORK_DPI
from ocr_utils.curved_layout.columns import DOT_FILL_SHARE, gutter_filled
from ocr_utils.curved_layout.leaders import flatten_axis, inside_spans, leaders_of, spans_at
from ocr_utils.curved_layout.pieces import is_low_mark
from ocr_utils.page_layout import mm_to_px
from ocr_utils.scan_markup.curved_lines.fitting import centreline, smooth_median

# Окно медианного сглаживания центр-линии в высотах строки (как в ``line_fit``).
SMOOTH_HEIGHTS = 2.0
# Доля краски в боксе, выше которой «строка» — на самом деле сплошная черта (линейка сноски,
# подчёркивание колонтитула): у набора заполнение 0.15–0.45, у черты — выше 0.55 при высоте 1–2 px.
RULE_MAX_HEIGHT_MM = 1.2
RULE_MIN_FILL = 0.55
# Волнистое подчёркивание колонтитула по высоте бокса от строки не отличишь (бокс 2 мм, а самой
# краски 1–2 px), и заполнение у него даже НИЖЕ, чем у набора (0.31 против 0.4–0.7). Отличает его
# ТОЛЩИНА краски в столбце: у черты это доли миллиметра, у строки — больше половины её высоты.
# Замер на 1975/05 с.97: жирная линейка под колонтитулом имеет толщину 0.42–0.45 мм и при пороге
# 0.4 не находилась вовсе, отчего колонтитул и заголовок рубрики попадали в один блок.
RULE_MAX_THICKNESS_MM = 0.6
# Косая или волнистая черта укладывается в такой бокс по высоте (сама краска всё так же тонкая).
RULE_WAVY_HEIGHT_MM = 3.0
# Меньше стольких глифов — не строка, а сор: порог заменяет прежние жёсткие требования к длине
# и отношению сторон, из-за которых выпадали короткие строки («500 м», «хозах.»). Три буквы —
# решение пользователя: «строка от 3 букв — уже строка».
MIN_GLYPHS = 3
SEGMENT_RULE_MIN_ASPECT = 6.0
# Отточие принимается за продолжение строки, если начинается не дальше LEADER_BACK_MM за её концом
# и лежит в полосе от LEADER_ABOVE_HEIGHTS высот выше центра строки до LEADER_BELOW_HEIGHTS ниже
# (точки стоят на базовой линии). LEADER_AXIS_STEP_MM — шаг, которым ось достраивается по отточию.
LEADER_BACK_MM = 25.0
LEADER_ABOVE_HEIGHTS = 0.2
LEADER_BELOW_HEIGHTS = 0.9
LEADER_AXIS_STEP_MM = 1.0
# Крупный сегмент, накрытый корпусными на эту долю длины, — дубль и не берётся.
COVER_SHARE = 0.6
# Крупная строка считается «той же самой», если она выше обрывка во столько раз и накрывает
# его по x на такую долю длины.
SWALLOW_HEIGHT_RATIO = 1.5
SWALLOW_SHARE = 0.8
# Ход цепочки кусков строки предсказывается прямой по её последним звеньям: сколько звеньев нужно,
# чтобы прямую строить, и по скольким последним она строится.
# Кандидат может начинаться левее конца цепочки на такую долю своей ширины (разрядка, засечки).
X_OVERLAP_SHARE = 0.5
# Полуширина полосы второго прохода центр-линии, в высотах строки: хвост соседней строки, попавший
# в тот же сгусток, за неё не выходит.
CENTRE_WINDOW_HEIGHTS = 0.55
# Зона на КОНЦЕ строки, где центр масс столбца ненадёжен, — в высотах строки. У курсива нет
# вертикальных штрихов: крайний столбец строки держит не всю букву, а только её угол — слева
# нижний, справа верхний, — и центр масс уезжает вниз и вверх соответственно (1971/10 с.87,
# подпись «В. ГУЛЕНКО / учёный секретарь экспертной / комиссии ВАК»: на «В.» ось ныряла на 3.5 px
# рабочей копии, на «ВАК» поднималась на 2.4 при высоте строчной 10).
END_FLAT_HEIGHTS = 0.6
# Насколько ордината в этой зоне может отойти от продолжения строки. Больше — это уже угол буквы,
# и ордината прижимается к продолжению. Меньше высоты полустрочной: настоящий изгиб строки на
# отрезке в полвысоты столько не набирает.
END_FLAT_TOLERANCE_HEIGHTS = 0.08
# По скольким высотам строки сразу за зоной строится опорная прямая.
END_FIT_HEIGHTS = 1.5
# Меньше стольких столбцов с краской — прижимать концы не по чему.
MIN_END_COLUMNS = 3
# Окно медианы, которым сглаживается опора перед подгонкой прямой (в высотах строки).
END_FIT_SMOOTH_HEIGHTS = 0.5
# Столбец НЕПОЛНЫЙ, если краски в нём меньше этой доли медианной по строке: он пересёк не всю
# букву, а только её угол. Замер на подписи 1971/10 с.87: у столбцов-углов 0.27–0.50 медианного
# веса, у полных столбцов той же зоны — 0.75–1.5.
END_THIN_SHARE = 0.55
# Корпусному масштабу оставляем строки не выше этой доли от медианы страницы; медиана берётся,
# только если корпусных строк набралось хотя бы столько.
BODY_HEIGHT_RATIO = 1.6
BODY_MIN_SEGMENTS = 10
# Заполнение бокса краской, ниже которого длинный компонент — слипшиеся буквы, а не черта.
LETTER_MAX_FILL = 0.55
# Мостик между строками: компонента выше стольких медианных высот страницы И вытянутая по
# вертикали (высота больше стольких ширин). Замер: на 1975/05 с.97 (плотный набор) мостик
# 14 × 29 px — это 2.9 медианы при p99 = 1.8; на 1973/08 с.85 компонент выше 1.4 шага нет вовсе.
# Буква заголовка тоже бывает высокой, но она и широкая, поэтому нужны оба признака.
BRIDGE_HEIGHT_RATIO = 2.2
BRIDGE_ASPECT = 1.6
# Черта короче этого (мм) разделителем не считается: это дефис или тире. Разделитель сноски в
# журнале — черта примерно в 6–8 мм (1973/06 с.65), поэтому порог низкий.
RULE_MIN_LENGTH_MM = 5.0
# Плотность краски в боксе черты: на бинарном рендере сплошная линейка даёт 0.5–0.9.
RULE_MIN_DENSITY = 0.4


@dataclass(frozen=True)
class Scale:
    """Набор размеров одного масштаба сегментации (пиксели копии ``WORK_DPI``).

    Args:
        name: Имя масштаба для отладки.
        min_height: Компонента ниже — не буква (пыль, точки).
        max_height: Компонента выше — не буква (росчерк логотипа, рамка, рисунок).
        gap: Зазор смыкания RLSA: у крупного набора пробел шире.
        aspect: Строка должна быть во столько раз длиннее своей высоты.
        min_length_mm: И не короче этого.
        max_height_mm: Медианная высота сгустков выше этой — не строка текста.
    """

    name: str
    min_height: int
    max_height: int
    gap: int
    aspect: float
    min_length_mm: float
    max_height_mm: float
    max_thickness_mm: float  # сгусток толще — не строка (рисунок, рамка)
    link_gap_heights: float  # разрыв между кусками строки, в высотах
    link_dy_heights: float  # расхождение центров кусков, в высотах
    link_height_ratio: float = 1.8  # во столько раз могут различаться высоты соседних кусков


# Корпус — числа готового конвейера (``ink_axis``); крупный набор — заголовки, логотипы рубрик,
# подписи авторов: буквы до 25 мм, разрядка между словами до 4 мм, длина от 12 мм.
SCALES = (
    Scale(
        "корпус",
        min_height=3,
        max_height=42,
        gap=8,
        aspect=1.5,
        min_length_mm=4.0,
        max_height_mm=8.0,
        max_thickness_mm=7.6,
        link_gap_heights=2.5,
        link_dy_heights=0.35,
    ),
    Scale(
        "крупный",
        min_height=16,
        max_height=150,
        gap=24,
        aspect=2.2,
        min_length_mm=12.0,
        max_height_mm=25.0,
        max_thickness_mm=30.0,
        link_gap_heights=2.0,
        link_dy_heights=0.7,
        # В крупном наборе рядом стоят буквы разного кегля: «50 лет» (1967/10 с.63) — цифры вдвое
        # выше слова, и при 1.8 строка не собиралась вовсе.
        link_height_ratio=3.0,
    ),
)


def _crosses(
    separators: list[tuple[int, int, int, int]],
    left: float,
    right: float,
    cy: float,
    height: float,
    ink300: np.ndarray,
    k: float,
) -> bool:
    """Лежит ли между ``left`` и ``right`` межколонник, ЖИВОЙ на высоте ``cy`` и ПУСТОЙ на ней.

    Высота важна: межколонник основного текста не должен запрещать сборку заголовка, который идёт
    выше начала колонок. А вот «межколонник залит отточиями — значит это продолжение строки»
    отвергнуто замером: отточия и так продлевают строку отдельным ходом
    (:func:`extend_with_leaders`), а послабление склеивало текст левой колонки с графой чисел
    правой (1971/10 с.93, низ).

    Args:
        separators: Межколонники ``(x0, x1, y0, y1)`` в пикселях рабочей копии.
        left, right: Разрыв между соседними кусками строки по x.
        cy: Ордината середины строки.
        height: Высота строки (не используется, оставлено для единообразия вызова).
        ink300: Краска рендера (не используется).
        k: Во сколько раз рендер крупнее рабочей копии (не используется).

    Returns:
        ``True``, если сцеплять куски нельзя.
    """
    return any(gy0 <= cy <= gy1 and gx0 < right and gx1 > left for gx0, gx1, gy0, gy1 in separators)


def candidates_of(stats: np.ndarray, scale: Scale, dpi: float) -> list[int]:
    """Сгустки, годные в куски строки этого масштаба, слева направо.

    Args:
        stats: Статистика компонент после смыкания.
        scale: Масштаб (корпус или крупный набор).
        dpi: Разрешение рабочей копии.

    Returns:
        Индексы компонент, отсортированные по левому краю.
    """
    max_thickness = mm_to_px(scale.max_thickness_mm, dpi)
    out = [
        index
        for index in range(1, stats.shape[0])
        if scale.min_height <= stats[index, cv2.CC_STAT_HEIGHT] <= max_thickness
        and stats[index, cv2.CC_STAT_WIDTH] >= 0.5 * stats[index, cv2.CC_STAT_HEIGHT]
    ]
    out.sort(key=lambda index: stats[index, cv2.CC_STAT_LEFT])
    return out


@dataclass(frozen=True)
class Rule:
    """Сплошная горизонтальная черта: разделитель сноски, подчёркивание, линейка колонтитула."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass(frozen=True)
class Segment:
    """Строка после сегментации: бокс на рабочей копии и центр-линия на копии ``RENDER_DPI``."""

    x0: int
    y0: int
    x1: int
    y1: int
    height: float
    xs: np.ndarray
    ys: np.ndarray
    weights: np.ndarray
    scale: str = "корпус"
    # Отрезки по x, занятые НИЗКИМИ МЕТКАМИ — точками и запятыми. Они сидят на базовой линии, и
    # центр масс краски в их столбцах лежит на полвысоты строчной ниже оси строки: ось там
    # провисает. Участки помечаются, чтобы не сбивать меры наклона и формы строки.
    mark_spans: tuple[tuple[float, float], ...] = ()

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


def component_mask(work: np.ndarray, scale: Scale) -> np.ndarray:
    """Маска компонент, похожих на буквы этого масштаба (пиксели копии ``work``).

    Длинный компонент — либо линейка, либо слипшиеся жирные буквы заголовка. Различает их
    ЗАПОЛНЕНИЕ бокса краской: у черты оно около единицы, у слова из букв — 0.3–0.5. Без этого
    «ОБРАЗОВАНИЕ КАДРОВ» (слиплось в один компонент 700 × 26 px) выпадало целиком.

    Отдельно выбрасывается МОСТИК между строками — две буквы соседних строк, соприкоснувшиеся по
    вертикали. На плотном наборе (1975/05 с.97: шаг строк 18.4 px при высоте строчной 10) такая
    компонента бывает 29 px высотой и проходит по пределу масштаба (42 px). Через неё смыкание
    RLSA сшивает две строки в один сгусток, у сгустка получается левый конец на одной строке, а
    правый на другой, и ось взбирается со строки на строку — перескок, который никакой сцепкой
    уже не исправить. Узнаётся мостик по двум признакам сразу: он много выше медианной буквы
    страницы И вытянут по вертикали (буква заголовка тоже высокая, но она и широкая).
    """
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros(count, dtype=bool)
    for index in range(1, count):
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        if not (scale.min_height <= height <= scale.max_height):
            continue
        fill = stats[index, cv2.CC_STAT_AREA] / max(1, width * height)
        if width <= 5 * max(height, 1) or fill < LETTER_MAX_FILL:
            keep[index] = True
    heights = stats[1:, cv2.CC_STAT_HEIGHT][keep[1:]]
    if heights.size:
        limit = BRIDGE_HEIGHT_RATIO * float(np.median(heights))
        for index in np.nonzero(keep)[0]:
            height = float(stats[index, cv2.CC_STAT_HEIGHT])
            width = float(stats[index, cv2.CC_STAT_WIDTH])
            if height > limit and height > BRIDGE_ASPECT * max(width, 1e-6):
                keep[index] = False
    return keep[labels].astype(np.uint8)


def rules_of(work: np.ndarray, dpi: float) -> list[Rule]:
    """Сплошные горизонтальные черты страницы: разделитель сноски, подчёркивание, линейка.

    Ищутся отдельно от строк: в маску букв они не попадают (слишком длинные при малой высоте),
    а блоку нужны как граница — текст под разделителем сноски к блоку не относится.
    """
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    max_height = mm_to_px(RULE_MAX_HEIGHT_MM, dpi)
    wavy_height = mm_to_px(RULE_WAVY_HEIGHT_MM, dpi)
    min_width = mm_to_px(RULE_MIN_LENGTH_MM, dpi)
    out: list[Rule] = []
    for index in range(1, count):
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        area = int(stats[index, cv2.CC_STAT_AREA])
        fill = area / max(1, width * height)
        if width < min_width or width < 6 * height:
            continue
        # Прямая черта узнаётся по высоте и плотности: у разделителя сноски на бинарном рендере
        # заполнение падает до 0.5, поэтому порог ниже, чем у «строки-черты». Косая или волнистая
        # черта (скан кривой бумаги) в свой бокс не укладывается — её выдаёт ТОЛЩИНА: площадь на
        # единицу длины меньше полумиллиметра (1973/08 с.85: разделитель сноски терялся).
        straight = height <= max_height and fill >= RULE_MIN_DENSITY
        wavy = height <= wavy_height and area / max(1, width) <= mm_to_px(RULE_MAX_THICKNESS_MM, dpi)
        if straight or wavy:
            x0 = int(stats[index, cv2.CC_STAT_LEFT])
            y0 = int(stats[index, cv2.CC_STAT_TOP])
            out.append(Rule(x0=x0, y0=y0, x1=x0 + width, y1=y0 + height))
    return out


def _glyph_count(mask: np.ndarray, labels: np.ndarray, span: list[int], x0: int, y0: int, x1: int, y1: int) -> int:
    """Сколько отдельных глифов в сегменте (компоненты ДО смыкания RLSA).

    Порог по длине и отношению сторон опущен, чтобы в блок попадали короткие строки таблиц
    («500 м», 9 мм, отношение 3.7 — 1971/10 с.93) и однословные концы абзацев. Чтобы вместе с
    ними не набежал сор, у строки требуется хотя бы ``MIN_GLYPHS`` букв.

    Args:
        mask: Маска глифов рабочей копии.
        labels: Карта компонент ПОСЛЕ смыкания.
        span: Индексы компонент сегмента.
        x0, y0, x1, y1: Бокс сегмента на рабочей копии.

    Returns:
        Число связных компонент маски глифов внутри сегмента.
    """
    own = np.isin(labels[y0:y1, x0:x1], span)
    glyphs = (mask[y0:y1, x0:x1] > 0) & own
    if not glyphs.any():
        return 0
    return int(cv2.connectedComponents(glyphs.astype(np.uint8), 8)[0]) - 1


def _thin_rule(ink: np.ndarray, k: float, spans: list[tuple[float, float]] | None = None) -> bool:
    """Черта ли это: медианная толщина краски по столбцам меньше ``RULE_MAX_THICKNESS_MM``.

    Args:
        ink: Краска сегмента (пиксели копии ``RENDER_DPI``).
        k: Во сколько раз ``RENDER_DPI`` крупнее рабочей копии.
        spans: Отрезки отточий внутри сегмента (пиксели ``RENDER_DPI`` от его левого края). Точки
            отточия тонкие, и строка «текст плюс длинное отточие» иначе принималась за волнистую
            черту и выбрасывалась целиком (1971/10 с.93).

    Returns:
        ``True``, если краска в столбцах лежит тонким слоем — так идёт черта, а не набор.
    """
    thickness = ink.sum(axis=0).astype(np.float64)
    if spans:
        keep = ~inside_spans(np.arange(thickness.size, dtype=np.float64), spans)
        if keep.any():
            thickness = thickness[keep]
    thickness = thickness[thickness > 0]
    if thickness.size == 0:
        return False
    return float(np.median(thickness)) <= mm_to_px(RULE_MAX_THICKNESS_MM, RENDER_DPI)


def _segments_at_scale(
    gray300: np.ndarray,
    work: np.ndarray,
    ink300: np.ndarray,
    separators: list[tuple[int, int, int, int]],
    scale: Scale,
    dpi: float,
    leaders: list | None = None,
    rules: list["Rule"] | None = None,
    linking: str = LINKING_DEFAULT,
) -> list[Segment]:
    """Строки одного масштаба; ``leaders`` — отточия страницы (по ним выравнивается ось),
    ``rules`` — сплошные черты (через ребро ячейки строка не сшивается), ``linking`` — способ
    сцепки кусков (``zones`` — по зонам поиска, ``greedy`` — прежняя жадная цепочка)."""
    mask = component_mask(work, scale)
    smeared = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, scale.gap), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(smeared, 8)
    if count <= 1:
        return []
    k = RENDER_DPI / dpi
    rule_height = mm_to_px(RULE_MAX_HEIGHT_MM, dpi)
    min_length = mm_to_px(scale.min_length_mm, dpi)
    max_height = mm_to_px(scale.max_height_mm, dpi)
    heights = [
        int(stats[index, cv2.CC_STAT_HEIGHT])
        for index in range(1, stats.shape[0])
        if stats[index, cv2.CC_STAT_WIDTH] >= 5 and stats[index, cv2.CC_STAT_HEIGHT] >= scale.min_height
    ]
    # Типичная высота строки страницы: по ней ограничивается окно второго прохода центр-линии,
    # иначе у сгустка из двух слипшихся строк окно накрывает обе и ось идёт между ними.
    typical_height = float(np.median(heights)) if heights else 0.0

    def build(spans: list[list[int]]) -> list[Segment]:
        """Собрать строки из готовых наборов кусков (общий хвост обоих проходов)."""
        result: list[Segment] = []
        for whole in spans:
            for span in _split_at_gutters(whole, stats, separators, ink300, k):
                segment = _segment_of(
                    span,
                    stats,
                    labels,
                    mask,
                    ink300,
                    scale,
                    k,
                    dpi,
                    leaders,
                    typical_height,
                    rule_height,
                    min_length,
                    max_height,
                )
                if segment is not None:
                    result.append(segment)
        return result

    if linking == LINKING_ZONES:
        # Сцепка по зонам поиска: у каждого куска своя ось, от её концов выпущены «колбаски», и
        # соединяются только куски, чьи зоны встретились под малым углом. Копить ошибке нечего —
        # каждое соединение проверяется от исходного куска, а не от конца растущей цепочки.
        from ocr_utils.curved_layout.zones import link_by_zones

        return build(link_by_zones(stats, labels, mask, separators, scale, dpi, ink300, k, leaders, rules))
    # Прежний ход: жадная цепочка, по её длинным строкам поле, затем та же цепочка с полем как
    # ограничителем и сшивание обрывков одного уровня. Подробности и замеры — в legacy_linking.
    from ocr_utils.curved_layout.legacy_linking.baselines import field_of
    from ocr_utils.curved_layout.legacy_linking.linking import link_spans, merge_by_field

    first = build(link_spans(stats, separators, scale, dpi, ink300, k))
    field = field_of(first, work.shape, k, dpi)
    if field is None:
        return first
    guarded = link_spans(stats, separators, scale, dpi, ink300, k, field=field)
    return build(merge_by_field(guarded, stats, field, separators, scale, ink300, k, leaders, rules))


def _mark_spans(
    mask: np.ndarray, labels: np.ndarray, span: list[int], x0: int, y0: int, x1: int, y1: int
) -> tuple[tuple[float, float], ...]:
    """Отрезки по x, занятые низкими метками сегмента — точками и запятыми.

    Точка и запятая стоят на БАЗОВОЙ линии, а не на оси строки: центр масс краски в их столбцах
    на полвысоты строчной ниже оси, и ось в конце строки провисает («Энгельс Ф.» в сноске
    1973/08 с.85). Такие участки помечаются, чтобы исключать их из мер наклона и формы строки и
    рисовать на оверлее отдельно.

    Args:
        mask: Маска глифов рабочей копии.
        labels: Карта компонент после смыкания RLSA.
        span: Индексы сгустков сегмента.
        x0, y0, x1, y1: Бокс сегмента на рабочей копии.

    Returns:
        Отрезки ``(x0, x1)`` в пикселях рабочей копии, слева направо.
    """
    own = np.isin(labels[y0:y1, x0:x1], span)
    glyphs = ((mask[y0:y1, x0:x1] > 0) & own).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(glyphs, 8)
    if count <= 1:
        return ()
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    x_h = float(np.median(heights))
    out = [
        (float(x0 + stats[index + 1, cv2.CC_STAT_LEFT]), float(x0 + stats[index + 1, cv2.CC_STAT_LEFT] + widths[index]))
        for index in range(count - 1)
        if is_low_mark(widths[index], heights[index], x_h)
    ]
    return tuple(sorted(out))


def _segment_of(
    span: list[int],
    stats: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    ink300: np.ndarray,
    scale: Scale,
    k: float,
    dpi: float,
    leaders: list | None,
    typical_height: float,
    rule_height: float,
    min_length: float,
    max_height: float,
) -> Segment | None:
    """Строка из набора кусков: отбор по размерам, центр-линия по краске, сглаживание.

    Returns:
        :class:`Segment` или ``None``, если набор не похож на строку (короткий, черта, сор).
    """
    members = stats[span]
    x0 = int(members[:, cv2.CC_STAT_LEFT].min())
    y0 = int(members[:, cv2.CC_STAT_TOP].min())
    x1 = int((members[:, cv2.CC_STAT_LEFT] + members[:, cv2.CC_STAT_WIDTH]).max())
    y1 = int((members[:, cv2.CC_STAT_TOP] + members[:, cv2.CC_STAT_HEIGHT]).max())
    width = x1 - x0
    h_line = float(np.median(members[:, cv2.CC_STAT_HEIGHT]))
    if h_line > max_height or h_line < scale.min_height:
        return None
    if width < scale.aspect * h_line or width < min_length:
        return None
    if _glyph_count(mask, labels, span, x0, y0, x1, y1) < MIN_GLYPHS:
        return None  # один компонент — не строка: сор, точка, обрывок рамки
    own = np.isin(labels[y0:y1, x0:x1], span).astype(np.uint8)
    crop = ink300[int(y0 * k) : int(y1 * k), int(x0 * k) : int(x1 * k)]
    own300 = cv2.resize(own, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    ink = crop & (own300 > 0)
    own_spans = [
        ((x0_span - x0) * k, (x1_span - x0) * k)
        for x0_span, x1_span in spans_at(leaders or [], (y0 + y1) / 2.0, h_line)
    ]
    if h_line <= rule_height and ink.mean() >= RULE_MIN_FILL:
        return None  # сплошная черта: она ищется отдельно (``rules_of``)
    if width >= SEGMENT_RULE_MIN_ASPECT * h_line and _thin_rule(ink, k, own_spans):
        return None  # волнистая черта (подчёркивание колонтитула): краски в столбце на волос
    xs, ys, weights = refined_centreline(ink, min(h_line, typical_height or h_line) * k)
    if xs.size == 0:
        return None
    # Ось идёт ПО отточию и ПО точкам с запятыми, но не ныряет к базовой линии: ординаты в их
    # столбцах берутся интерполяцией по соседним буквам (см. ``leaders.flatten_axis``). Точка и
    # запятая стоят на базовой линии, то есть на полвысоты строчной ниже оси, и без этого ось
    # провисала над каждым знаком препинания, а в конце строки — уходила вниз совсем.
    marks = _mark_spans(mask, labels, span, x0, y0, x1, y1)
    flat = own_spans + [((mark_x0 - x0) * k, (mark_x1 - x0) * k) for mark_x0, mark_x1 in marks]
    xs, ys, weights = flatten_axis(xs, ys, weights, flat)
    ys = _flatten_ends(xs, ys, weights, h_line * k)
    ys = smooth_median(ys, int(SMOOTH_HEIGHTS * h_line * k))
    return Segment(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        height=h_line,
        xs=xs / k + x0,
        ys=ys / k + y0,
        weights=weights,
        scale=scale.name,
        mark_spans=marks,
    )


def _split_at_gutters(
    span: list[int], stats: np.ndarray, separators: list[tuple[int, int, int, int]], ink300: np.ndarray, k: float
) -> list[list[int]]:
    """Разрезать строку по межколонникам, в полосе которых нет краски.

    Сборка кусков строки (``link_spans``) иногда сшивает соседние колонки: между последним
    сгустком одной колонки и первым сгустком другой попадает случайная краска (надстрочный знак,
    точка), и строка тянется через межколонник (1975/05 с.97). Настоящий заголовок через
    межколонник отличается тем, что краска в межколоннике ЕСТЬ, — такую строку не режем, её
    пометит :func:`columns.mark_cut_lines`.
    """
    members = stats[span]
    y0 = int(members[:, cv2.CC_STAT_TOP].min())
    y1 = int((members[:, cv2.CC_STAT_TOP] + members[:, cv2.CC_STAT_HEIGHT]).max())
    centres = members[:, cv2.CC_STAT_LEFT] + members[:, cv2.CC_STAT_WIDTH] / 2.0
    cuts: list[float] = []
    cy = (y0 + y1) / 2.0
    for gx0, gx1, gy0, gy1 in separators:
        if gx0 <= 0 or not (centres.min() < gx0 and centres.max() > gx1):
            continue
        if not gy0 <= cy <= gy1:
            continue  # межколонник живёт ниже или выше этой строки — резать по нему нечего
        # Та же мера, что и при сцепке: доля СТОЛБЦОВ межколонника с краской на высоте строки.
        # По площади мера была слишком чувствительной — одна точка давала «краска есть».
        if gutter_filled(ink300, gx0, gx1, cy, (y1 - y0) / 2.0, k) >= DOT_FILL_SHARE:
            continue  # краска в межколоннике: строка и правда набрана через него (заголовок, отточия)
        cuts.append((gx0 + gx1) / 2.0)
    if not cuts:
        return [span]
    groups: dict[int, list[int]] = {}
    for index, centre in zip(span, centres):
        key = int(np.searchsorted(sorted(cuts), centre))
        groups.setdefault(key, []).append(index)
    return list(groups.values())


def _covered(segment: Segment, others: list[Segment]) -> bool:
    """Накрыт ли сегмент другими: та же полоса по y и больше ``COVER_SHARE`` длины."""
    covered = 0.0
    for other in others:
        if min(segment.y1, other.y1) - max(segment.y0, other.y0) <= 0.4 * (segment.y1 - segment.y0):
            continue
        covered += max(0, min(segment.x1, other.x1) - max(segment.x0, other.x0))
    return covered >= COVER_SHARE * max(1, segment.x1 - segment.x0)


def _swallowed(small: Segment, big: Segment, height_ratio: float = SWALLOW_HEIGHT_RATIO) -> bool:
    """Лежит ли корпусный обрывок целиком внутри крупной строки (тот же заголовок, но огрызком).

    Корпусный масштаб на жирном заголовке цепляет отдельные буквы: получается короткая ось на
    своей высоте, и блок заголовка раздваивался (1973/06 с.65, «ОСНОВЫ ЭКОНОМИКИ…» и «Тема 11…»).
    Обрывок отбрасывается, если крупная строка выше его как минимум в ``height_ratio`` раз и
    перекрывает его по x почти целиком.

    Args:
        small: Проверяемая строка корпусного масштаба.
        big: Строка крупного масштаба.
        height_ratio: Во сколько раз крупная строка должна быть выше (1.0 — высота не важна).

    Returns:
        ``True``, если ``small`` — часть той же строки, что и ``big``.
    """
    if big.height < height_ratio * max(small.height, 1e-6):
        return False
    if min(small.y1, big.y1) - max(small.y0, big.y0) <= 0.5 * (small.y1 - small.y0):
        return False
    overlap = max(0, min(small.x1, big.x1) - max(small.x0, big.x0))
    return overlap >= SWALLOW_SHARE * max(1, small.x1 - small.x0)


def _body_sized(body: list[Segment], large: list[Segment]) -> list[Segment]:
    """Оставить корпусному масштабу только строки основного кегля.

    Корпусный масштаб цепляет и заголовок, но собирает его ХУЖЕ: у него узкий допуск на перепад
    высот соседних слов (``link_dy_heights`` 0.35), и сильно наклонённый заголовок рвётся посреди
    строки на два куска (1973/06 с.65, «ОСНОВЫ ЭКОНОМИКИ…»). Такие куски дальше выигрывали у
    цельной крупной строки, и заголовок становился двумя блоками. Поэтому строка, которая заметно
    выше медианы страницы, отдаётся крупному масштабу — но только если он её действительно нашёл.

    Args:
        body: Строки корпусного масштаба.
        large: Строки крупного масштаба (ещё не отфильтрованные).

    Returns:
        Строки корпусного масштаба без тех, что явно крупнее основного кегля и накрыты крупной.
    """
    if len(body) < BODY_MIN_SEGMENTS:
        return body
    median_height = float(np.median([segment.height for segment in body]))
    out = []
    for segment in body:
        if segment.height <= BODY_HEIGHT_RATIO * median_height:
            out.append(segment)
            continue
        if not any(_swallowed(segment, big, height_ratio=1.0) for big in large):
            out.append(segment)  # крупный масштаб эту строку не нашёл — своё не отдаём
    return out


def refined_centreline(ink: np.ndarray, height_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Центр-линия строки, устойчивая к слипанию соседних строк.

    Первый проход — обычный центр масс краски по столбцам (``fitting.centreline``). Если в
    сгусток попали хвосты соседней строки (на сильно искажённой бумаге они слипаются), центр масс
    в таких столбцах уезжает, и ось идёт по диагонали через две строки — на 1973/08 с.85 так
    получались наклоны до 10° при норме два. Второй проход считает центр масс заново, но только
    в полосе ``±CENTRE_WINDOW_HEIGHTS`` высоты вокруг сглаженной линии первого прохода.

    Args:
        ink: Краска сегмента (пиксели ``RENDER_DPI``).
        height_px: Высота строки в тех же пикселях.

    Returns:
        ``(xs, ys, weights)`` — как у ``fitting.centreline``.
    """
    xs, ys, weights = centreline(ink)
    if xs.size < 3 or height_px <= 0:
        return xs, ys, weights
    anchor = smooth_median(ys, max(3, int(SMOOTH_HEIGHTS * height_px) | 1))
    half = max(2.0, CENTRE_WINDOW_HEIGHTS * height_px)
    rows = np.arange(ink.shape[0], dtype=np.float64)[:, None]
    mask = ink > 0
    out_y, out_w = np.empty(xs.size), np.empty(xs.size)
    for index, (x, centre) in enumerate(zip(xs.astype(int), anchor)):
        column = mask[:, x]
        near = column & (np.abs(rows[:, 0] - centre) <= half)
        count = near.sum()
        if count == 0:
            out_y[index], out_w[index] = ys[index], weights[index]
            continue
        out_y[index] = float((rows[:, 0] * near).sum() / count)
        out_w[index] = float(count)
    return xs, out_y, out_w


def _flatten_ends(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray, height_px: float) -> np.ndarray:
    """Не давать оси нырять на концах строки: там столбец держит только УГОЛ буквы.

    Ось — центр масс краски по столбцам, и это верная оценка уровня строки, пока столбец
    пересекает букву во всю её высоту. На концах строки это не так, и особенно у курсива:
    вертикальных штрихов в нём нет, каждый штрих наклонён, поэтому дальше всего слева выступает
    НИЖНИЙ-левый угол первой буквы, а справа — ВЕРХНИЙ-правый угол последней. Центр масс в этих
    столбцах уходит вниз и вверх соответственно, и ось на концах загибается.

    Лечится прижатием: по опорному окну сразу за зоной конца строится прямая (по медианно
    сглаженным ординатам, иначе её развернёт форма отдельной буквы), и ордината в зоне зажимается
    вокруг её продолжения. Честный изгиб строки на полвысоты допуска не выбирает, а угол буквы
    выбирает вдвое.

    Делается это ДО общего сглаживания оси. После — поздно: окно медианы у самого конца строки
    одностороннее, доля столбцов-углов в нём вырастает, и медиана тянется за ними. На той же
    подписи 1971/10 с.87 четыре столбца угла «К» утащили за собой девять пикселей рабочей копии.

    Прижимаются не все столбцы зоны, а только НЕПОЛНЫЕ — те, где краски в столбце заметно меньше
    обычного: столбец, пересекающий букву во всю высоту, даёт верный центр масс и трогать его
    незачем. Замер на подписи 1971/10 с.87: у столбцов-углов вес 0.27–0.50 медианного, у
    остальных столбцов той же зоны — 0.75–1.5.

    Args:
        xs: Абсциссы центр-линии (пиксели рендера, начало — левый край сегмента).
        ys: Ординаты в тех же координатах, ещё не сглаженные.
        weights: Вес столбца — сколько в нём краски.
        height_px: Высота строки в тех же пикселях.

    Returns:
        Те же ординаты с прижатыми концами.
    """
    if xs.size < MIN_END_COLUMNS or height_px <= 0:
        return ys
    reach = END_FLAT_HEIGHTS * height_px
    limit = END_FLAT_TOLERANCE_HEIGHTS * height_px
    if float(xs[-1] - xs[0]) <= 2.0 * reach:
        return ys  # строка короче двух зон: прижимать не от чего
    out = ys.copy()
    thin = weights < END_THIN_SHARE * max(float(np.median(weights)), 1e-6)
    if not thin.any():
        return out
    for at_start in (True, False):
        if at_start:
            edge = float(xs[0])
            zone = xs <= edge + reach
            base = (xs > edge + reach) & (xs <= edge + reach + END_FIT_HEIGHTS * height_px)
        else:
            edge = float(xs[-1])
            zone = xs >= edge - reach
            base = (xs < edge - reach) & (xs >= edge - reach - END_FIT_HEIGHTS * height_px)
        zone &= thin
        if int(base.sum()) < MIN_END_COLUMNS or not zone.any():
            continue
        # Прямая строится по СГЛАЖЕННЫМ ординатам опоры: на сырых её разворачивает форма
        # отдельной буквы (разброс центра масс по столбцам — полвысоты строчной).
        steady = smooth_median(out[base], max(3, int(END_FIT_SMOOTH_HEIGHTS * height_px) | 1))
        slope, intercept = np.polyfit(xs[base], steady, 1)
        predicted = slope * xs[zone] + intercept
        out[zone] = np.clip(out[zone], predicted - limit, predicted + limit)
    return out


def extend_with_leaders(
    segments: list[Segment], leaders: list, dpi: float, separators: list[tuple[int, int, int, int]] | None = None
) -> list[Segment]:
    """Продлить строки по отточиям, идущим за ними.

    Отточие — продолжение своей строки, но собрать его вместе с ней не удаётся: между последним
    словом и первой точкой бывает полтора сантиметра, а сами точки стоят на базовой линии, то есть
    ниже центра строки. Поэтому строки собираются как обычно, а потом каждая продлевается вправо
    по отточиям, которые начинаются за её концом и лежат в полосе от центра строки до базовой
    линии. Ось при этом продолжается ПО ОТТОЧИЮ, но по наклону самой строки, а не ныряет к точкам.

    Args:
        segments: Строки масштаба корпуса.
        leaders: Отточия страницы (``leaders.Leader``).
        dpi: Разрешение рабочей копии.
        separators: Межколонники ``(x0, x1, y0, y1)``: дальше своей колонки строка не продлевается.

    Returns:
        Те же строки; у продлённых изменены ``x1`` и хвост центр-линии.
    """
    if not leaders or not segments:
        return segments
    k = RENDER_DPI / dpi
    step = mm_to_px(LEADER_AXIS_STEP_MM, dpi)
    out: list[Segment] = []
    for segment in segments:
        own = [
            leader
            for leader in leaders
            if leader.x0 >= segment.x1 - mm_to_px(LEADER_BACK_MM, dpi)
            and segment.cy - LEADER_ABOVE_HEIGHTS * segment.height <= leader.y
            and leader.y <= segment.cy + LEADER_BELOW_HEIGHTS * segment.height
        ]
        if not own:
            out.append(segment)
            continue
        finish = max(leader.x1 for leader in own)
        # Не дальше своей колонки: отточие доходит до межколонника, и продлённая по нему строка
        # переставала помещаться в колонку — блок собирался во всю ширину страницы и мешался с
        # соседней графой (1971/10 с.93, низ).
        for gx0, _, gy0, gy1 in separators or []:
            if gy0 <= segment.cy <= gy1 and gx0 >= segment.x1:
                finish = min(finish, float(gx0))
        if finish <= segment.x1:
            out.append(segment)
            continue
        # Ось продолжается ПО САМИМ ТОЧКАМ, поднятым до уровня строки: точки стоят на базовой
        # линии, и вести ось по наклону хвоста нельзя — на десятке сантиметров она уезжает вниз.
        xs, ys = segment.xs, segment.ys
        dots = np.array(sorted(point for leader in own for point in leader.points), dtype=np.float64)
        if dots.shape[0] < 2:
            out.append(segment)
            continue
        grid = np.arange(segment.x1, finish + step, step)
        dotted = np.interp(grid * 1.0, dots[:, 0] * k, dots[:, 1] * k)
        # Смещение выбирается так, чтобы продолжение начиналось ровно с конца оси строки.
        added_y = dotted + (ys[-1] - float(np.interp(xs[-1], dots[:, 0] * k, dots[:, 1] * k)))
        out.append(
            Segment(
                x0=segment.x0,
                y0=segment.y0,
                x1=int(round(finish)),
                y1=segment.y1,
                height=segment.height,
                xs=np.concatenate([xs, grid]),
                ys=np.concatenate([ys, added_y]),
                weights=np.concatenate([segment.weights, np.full(grid.shape, float(np.min(segment.weights)))]),
                scale=segment.scale,
                mark_spans=segment.mark_spans,
            )
        )
    return out


def segments_of(
    gray300: np.ndarray,
    separators: list[tuple[int, int, int, int]],
    dpi: float = WORK_DPI,
    leaders: list | None = None,
    linking: str = LINKING_DEFAULT,
) -> tuple[list[Segment], list[Rule]]:
    """Строки страницы с центр-линиями по краске, в обоих масштабах, и сплошные черты.

    Args:
        gray300: Серый рендер страницы в ``RENDER_DPI``.
        separators: Межколонники ``(x0, x1)`` в пикселях рабочей копии ``dpi``.
        dpi: Разрешение рабочей копии (боксы и высоты отдаются в нём).
        leaders: Готовые отточия страницы; ``None`` — посчитать самим.
        linking: Способ сцепки кусков: ``zones`` (по зонам поиска) или ``greedy`` (прежний ход).

    Returns:
        Строки сверху вниз (сначала корпус, затем крупные строки, не накрытые корпусными) и
        сплошные черты страницы.
    """
    work = cv2.resize(
        gray300,
        (max(1, round(gray300.shape[1] * dpi / RENDER_DPI)), max(1, round(gray300.shape[0] * dpi / RENDER_DPI))),
        interpolation=cv2.INTER_AREA,
    )
    threshold, _ = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink300 = gray300 <= threshold
    leaders = leaders_of(work, dpi)[0] if leaders is None else leaders
    # Черты нужны сборке (через ребро ячейки строка не сшивается), поэтому ищутся заранее.
    rules = rules_of(work, dpi)
    body = extend_with_leaders(
        _segments_at_scale(gray300, work, ink300, separators, SCALES[0], dpi, leaders, rules, linking), leaders, dpi
    )
    large_all = _segments_at_scale(gray300, work, ink300, separators, SCALES[1], dpi, None, rules, linking)
    body = _body_sized(body, large_all)
    # Крупный масштаб добавляет только то, чего корпус не нашёл (заголовки, логотип).
    large = [segment for segment in large_all if not _covered(segment, body)]
    # Обратный ход: корпусные обрывки внутри найденной крупной строки — это её же буквы.
    out = [segment for segment in body if not any(_swallowed(segment, big) for big in large)]
    out.extend(large)
    return sorted(out, key=lambda item: item.cy), rules


__all__ = ["SCALES", "Rule", "Scale", "Segment", "component_mask", "rules_of", "segments_of"]
