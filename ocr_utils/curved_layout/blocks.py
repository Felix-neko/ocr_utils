"""Текстовый блок = колонка: ряды строк, края рядов по краске и гладкая огибающая блока.

Огибающая строится не по крайним точкам краски (тогда её тянут абзацные отступы и висячие
строки), а КВАНТИЛЬНЫМ ТРЕНДОМ: в скользящем окне шириной в несколько межстрочных интервалов
берётся квантиль краёв рядов (слева нижний, справа верхний) и сглаживается. Получается плавная
кривая по телу блока — то, что нельзя описать прямой на изогнутой бумаге.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
from scipy.signal import savgol_filter

from ocr_utils.curved_layout import RENDER_DPI, WORK_DPI
from ocr_utils.curved_layout.columns import DOT_FILL_SHARE, bounds_at, gutter_filled
from ocr_utils.curved_layout.leaders import inside_spans, spans_at
from ocr_utils.curved_layout.lines import LineAxis, with_column
from ocr_utils.page_layout import mm_to_px, px_to_mm

# Край ряда ищется в пределах колонки с этим припуском: выносные элементы и кавычки вылезают за
# границу колонки, найденную по профилю.
COLUMN_PAD_MM = 5.0
# Столбец считается краем, если в полосе ряда в нём не меньше стольких пикселей краски и краска
# есть ещё не меньше чем в EDGE_CONFIRM_MIN из следующих EDGE_CONFIRM_COLS столбцов (пыль не край).
EDGE_MIN_INK_PX = 2
EDGE_CONFIRM_COLS = 3
EDGE_CONFIRM_MIN = 2
# Полоса ряда расширяется на эту долю высоты: край буквы с выносным элементом иначе теряется.
ROW_PAD_HEIGHTS = 0.15
# Сегменты с центрами ближе этой доли высоты — один ряд.
ROW_TOL_HEIGHTS = 0.5
# Доля точек оси с каждого конца, по которой меряется её наклон на этом конце.
AXIS_END_SHARE = 0.2
# Насколько далеко за свой конец ось продолжается по наклону конца (мм бумаги).
EXTEND_REACH_MM = 6.0
# И не круче этого: наклон конца короткой оси — шум, а перекос полосы не превышает пары градусов.
EXTEND_SLOPE_LIMIT_DEG = 8.0
# Окно огибающей по умолчанию — в межстрочных интервалах (решение пользователя: 2–3, настраивается).
SMOOTH_PITCHES = 2.5
# Во сколько раз крупнее окно у «крупной» огибающей, с которой сравнивается основная.
COARSE_FACTOR = 3.0
# Квантили краёв в окне: слева берём нижний, справа верхний — отступ абзаца и короткий конец
# строки остаются за кривой и не тянут её внутрь блока.
EDGE_QUANTILE = 0.25
# Шаг сетки огибающей по y.
GRID_STEP_MM = 1.0
# Блок короче стольких рядов не строится. Один ряд — это тоже блок (заголовок в одну строку,
# колонтитул, подпись автора): его огибающая идёт по оси строки с полем в полвысоты.
MIN_BLOCK_ROWS = 1
# Насколько край ряда может выйти за границу колонки (мм): дальше — чужая колонка.
OVERHANG_MM = 1.5
# Насколько край ряда может отстоять от крайней оси строки (мм): дальше — не буква, а мусор.
AXIS_REACH_MM = 6.0
# Ряд, вылезший за колонку, остаётся в ней, если она накрывает такую долю его длины; иначе он
# уходит в отдельный кусок во всю ширину страницы.
SPILL_KEEP_SHARE = 0.7
# Доля длины строки, которая должна лежать в колонке, чтобы строка считалась её строкой.
INSIDE_SHARE = 0.9
# Заход края ряда внутрь блока дальше этого (мм) — абзацный отступ или короткий конец строки:
# в тренд кромки такой ряд не идёт (но в мерах выключки считается отдельно).
TRIM_MM = 2.5
# Рядов в окне локальной прямой не меньше этого: по двум-трём кромку не провести.
MIN_TREND_ROWS = 5
# Широким (и потому способным забрать свой хвост из колонки) считается кусок не уже этой доли
# страницы; хвост забирается, пока идёт с шагом блока (не больше стольких шагов) и тем же кеглем.
CONTINUATION_WIDE_SHARE = 0.8
CONTINUATION_GAP_PITCHES = 1.6
CONTINUATION_HEIGHT_RATIO = 1.4
# Хвост абзаца — это одна-две строки, и его левый край совпадает с краем блока.
CONTINUATION_MAX_ROWS = 2
CONTINUATION_EDGE_MM = 6.0
# Строка выше медианы страницы во столько раз — крупный набор: у него разрыв между блоками
# меряется в высотах самих строк, а не в межстрочных интервалах группы.
LARGE_TYPE_RATIO = 1.3
GAP_HEIGHTS = 2.4
# Разрыв длиннее стольких шагов делит блок, если вдобавок меняется набор (см. _soft_style_break):
# по паку внутри блоков разрывы почти всегда около одного шага (p98 = 1.26), а 1.6–2.2 шага — это
# либо пропавшая строка, либо настоящая граница заголовка.
SOFT_GAP_PITCHES = 1.6
SOFT_STYLE_RATIO = 1.15
# Размер символа: компоненты краски от GLYPH_MIN_MM до GLYPH_MAX_MM (как в ``docstrum``), ширина —
# не больше GLYPH_MAX_WIDTH_RATIO высот (иначе это слипшееся слово); меньше MIN_GLYPHS_FOR_SIZE
# компонент — размер не меряем.
# Полуширина окна, в котором отточие считается принадлежащим ряду (пиксели рабочей копии).
MIN_LEADER_PROBE_PX = 6.0
GLYPH_MIN_MM = 0.8
GLYPH_MAX_MM = 7.0
GLYPH_MAX_WIDTH_RATIO = 3.0
MIN_GLYPHS_FOR_SIZE = 8
# На какую долю размера символа раздувается граница блока, поле растра под дилатацию и допуск
# упрощения снятого контура.
DILATE_GLYPHS = 0.5
RASTER_PAD_MM = 3.0
APPROX_EPS_MM = 0.3
# Если глифы намерить не удалось, размер символа оценивается от высоты ряда: ширина — такая доля,
# высота — такая (у корпуса строка 14 px, символ около 8 × 9 px).
FALLBACK_ASPECT = 0.55
FALLBACK_HEIGHT = 0.65
# Соседние ряды одного блока перекрываются по x не меньше чем на эту долю короткого ряда;
# проверка идёт только между рядами не короче LONG_ROW_SHARE от самого длинного ряда группы.
MIN_ROW_OVERLAP = 0.35
LONG_ROW_SHARE = 0.5
# Короткий ряд, отодвинутый от обеих кромок длинного соседа дальше этого (мм), — не конец абзаца,
# а отдельный блок: подпись автора, заголовок по центру. Абзацный отступ здесь 4–5 мм.
OFFSET_MM = 8.0
# Если центры короткого и длинного рядов совпадают в пределах этого (мм), они набраны по одному
# центру и делить их нельзя.
CENTRE_TOL_MM = 4.0
# Вертикальный разрыв больше стольких межстрочных интервалов делит колонку на два блока.
# 2.5, а не 3: в шапке статьи логотип рубрики, заголовок и подзаголовок идут с разными
# интервалами, и при 3 они слипались в один блок с пустотами внутри (1973/06 с.65). В корпусе
# разрыв между абзацами здесь меньше полутора интервалов, так что абзацы не делятся.
BLOCK_GAP_PITCHES = 2.5
# Медианные высоты по HEIGHT_WINDOW рядам с обеих сторон границы различаются во столько раз —
# другой кегль, другой блок (заголовок над корпусом, подпись автора под ним).
# 1.6 и окно в четыре ряда: у корпуса высота ряда гуляет 10–15 px (прописные, выносные), и при
# 1.45 колонка дробилась на куски (1975/05 с.97), а заголовок выше корпуса минимум в 1.7 раза.
HEIGHT_BREAK_RATIO = 1.6
HEIGHT_WINDOW = 4
# Толщина штриха по одну сторону границы больше, чем по другую, во столько раз — другой набор.
STROKE_BREAK_RATIO = 1.5
# На короткой группе (окно меньше HEIGHT_WINDOW) хватает меньшей разницы, но сразу по обоим
# признакам: и по высоте, и по штриху.
SHORT_BREAK_RATIO = 1.3
# Совместная мера смены набора на короткой группе: произведение отношений кегля и штриха.
STYLE_PRODUCT_RATIO = 1.9
# Кегль СОВПАЛ, если медианные размеры глифа с обеих сторон границы отличаются не больше чем в
# столько раз. Высота ряда — мера бокса, а бокс раздувают наклон строки, слипшиеся сгустки и
# прописная в начале абзаца: на 1971/10 с.87 nogeo три верхние строки корпуса дали высоты
# 26.5, 37.0 и 21.0 при 14–18 ниже (отношение 1.56) и штрих 2.50 против 2.00 (1.25, а штрих
# квантован шагом 0.25 px) — произведение 1.95 при пороге 1.9, и колонка рвалась пополам.
# Медианная высота компоненты краски при этом одинакова с точностью до сотых: 10.0 и 10.0 px
# при ширине 7.5 и 7.5. Такой границы нет, и признак кегля на ней гасится.
GLYPH_SAME_RATIO = 1.15
# Меньше стольких рядов с каждой стороны границы — признаки не меряем.
MIN_BREAK_WINDOW = 2
# Сплошная черта делит блок, если она перекрывает колонку не меньше чем на эту долю ширины:
# разделитель сноски в журнале — короткая черта примерно в четверть колонки.
RULE_OVERLAP_SHARE = 0.15
# Обрывок короче MIN_BLOCK_ROWS приклеивается к соседней группе, если разрыв не больше этого.
MERGE_GAP_PITCHES = 6.0
# Куски колонки из соседних зон считаются одной колонкой при таком перекрытии по ширине и
# совпадении левого края (мм).
MERGE_OVERLAP = 0.6
MERGE_EDGE_MM = 3.0
MERGE_WIDTH_RATIO = 1.5
# Блок шире типичного во столько раз и короче стольких рядов — кандидат в колонтитул.
WIDE_BLOCK_RATIO = 1.6
WIDE_BLOCK_ROWS = 5
# И выбрасывается, если такая доля его рядов слиплась из нескольких осей (строки соседних колонок).
GLUED_ROWS_SHARE = 0.5
# Столько раз повторяется «тренд → отбор рядов тела → тренд».
TREND_PASSES = 2
# Запас, на который кромка отодвигается наружу от самого дальнего края ряда (мм), и сколько раз
# повторяется «прижать к требованию → сгладить».
OUTWARD_MARGIN_MM = 0.3
OUTWARD_PASSES = 3
# Доля точек профиля с краю, по которой берётся наклон для продления кромки за её конец, и
# ограничения на само продление: сколько миллиметров оно живёт по x и насколько уводит по y.
END_SLOPE_SHARE = 0.3
END_EXTEND_MM = 8.0
END_EXTEND_MAX_MM = 2.0
# Шаг профиля краски ряда по x (мм) и окно его сглаживания: мельче кегля, но крупнее буквы.
PROFILE_STEP_MM = 1.0
PROFILE_SMOOTH_MM = 6.0
# Полуширина окна поиска краски вокруг оси строки, в её высотах: 0.9 берёт выносные элементы и
# прописные, но не достаёт до соседней строки (шаг строк — около 1.6 высоты).
PROFILE_WINDOW_HEIGHTS = 0.9
# Насколько кромке позволено выйти за диапазон краёв рядов (мм): дальше — экстраполяция.
TREND_CLIP_MM = 3.0


@dataclass(frozen=True)
class Row:
    """Ряд блока: одна строка (или слитые сегменты одной y-полосы) с краями по краске."""

    y: float  # ордината середины ряда, пиксели рабочей копии
    height: float
    x0: float  # край краски слева (None-строки сюда не попадают)
    x1: float
    axes: tuple[LineAxis, ...]
    stroke: float = 0.0  # толщина штриха, пиксели рабочей копии
    glyph_w: float = 0.0  # медианная ширина символа ряда, пиксели рабочей копии
    glyph_h: float = 0.0  # медианная высота символа ряда
    # Профили краски ряда ``(N, 2)``: верхняя и нижняя граница краски по столбцам. По ним идут
    # верхняя и нижняя кромки блока — они повторяют изгиб строки и не режут её.
    top_edge: np.ndarray | None = None
    bottom_edge: np.ndarray | None = None

    @property
    def axis(self) -> LineAxis:
        """Самая длинная ось ряда — она представляет ряд в верхней и нижней кромках блока."""
        return max(self.axes, key=lambda item: item.x1 - item.x0)


@dataclass(frozen=True)
class BlockEnvelope:
    """Огибающая блока: четыре кривые и замкнутый контур по ним (пиксели рабочей копии).

    ``left``/``right`` идут СНАРУЖИ всех строк: пользователь требует, чтобы в границы блока
    попадали все его строки целиком. Меры выключки считаются не от них, а от ``core_left`` и
    ``core_right`` — гладкого тренда по телу блока, который не прыгает за одиночным выносом.

    ``polygon`` — НЕ просто обход четырёх кромок: он ещё и проглатывает краску рядов там, где
    рамка её режет (:func:`_swallow_rows`), поэтому в него гарантированно попадают все буквы всех
    рядов блока. Четыре кромки остаются гладкой рамкой и от этой правки не меняются.
    """

    left: np.ndarray
    right: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    polygon: np.ndarray
    smooth_pitches: float
    core_left: np.ndarray | None = None
    core_right: np.ndarray | None = None
    # Граница, раздутая на долю размера символа: контур ровнее (впадины между выносными элементами
    # заполнены) и с запасом вокруг текста. Исходный ``polygon`` остаётся нетронутым.
    polygon_dilated: np.ndarray | None = None
    dilate_px: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class TextBlock:
    """Текстовый блок: ряды колонки, шаг строк и две огибающие разного масштаба сглаживания.

    Блок — колонка целиком (решение пользователя), но колонка, разорванная по вертикали большим
    пробелом (заголовок между двумя заметками — 1975/05 с.97), делится на блоки по этому пробелу:
    иначе огибающая тянется через пустоту в двадцать строк.
    """

    column: int
    index: int  # номер блока внутри колонки, сверху вниз
    span: tuple[int, int]  # границы колонки по x
    rows: tuple[Row, ...]
    pitch_px: float
    dpi: float
    envelope: BlockEnvelope
    envelope_coarse: BlockEnvelope

    @property
    def glyph_size(self) -> tuple[float, float]:
        """Средний размер символа блока: медианы ширины и высоты по рядам (пиксели рабочей копии)."""
        widths = [row.glyph_w for row in self.rows if row.glyph_w > 0]
        heights = [row.glyph_h for row in self.rows if row.glyph_h > 0]
        if not widths:
            # Глифов не намерили (короткий заголовок, разрядка) — оцениваем по высоте рядов.
            height = float(np.median([row.height for row in self.rows])) if self.rows else 0.0
            return height * FALLBACK_ASPECT, height * FALLBACK_HEIGHT
        return float(np.median(widths)), float(np.median(heights))

    @property
    def pitch_mm(self) -> float:
        return px_to_mm(self.pitch_px, self.dpi)

    @property
    def lines(self) -> int:
        return len(self.rows)


def ink_edge(ink: np.ndarray, x_lo: int, x_hi: int, y0: int, y1: int, side: str) -> float | None:
    """Край краски в полосе ``[y0, y1)`` между столбцами ``x_lo`` и ``x_hi`` (пиксели ``ink``).

    Слева — первый столбец с краской, подтверждённой соседями справа, справа — наоборот.
    Тот же приём, что у кромок блока в детекторе геометрии (v16): одиночная точка (пыль,
    пробившаяся засечка) краем не становится.
    """
    height, width = ink.shape
    x_lo, x_hi, y0, y1 = max(0, x_lo), min(width, x_hi), max(0, y0), min(height, y1)
    if x_hi - x_lo < EDGE_CONFIRM_COLS + 1 or y1 <= y0:
        return None
    counts = (ink[y0:y1, x_lo:x_hi] > 0).sum(axis=0)
    filled = counts >= EDGE_MIN_INK_PX
    order = range(len(filled)) if side == "left" else range(len(filled) - 1, -1, -1)
    for i in order:
        if not filled[i]:
            continue
        window = (
            filled[i + 1 : i + 1 + EDGE_CONFIRM_COLS] if side == "left" else filled[max(0, i - EDGE_CONFIRM_COLS) : i]
        )
        if window.sum() >= EDGE_CONFIRM_MIN:
            return float(x_lo + i)
    return None


def _axis_slope(axis: LineAxis, at_start: bool) -> float:
    """Наклон оси у её конца: по крайней доле ``AXIS_END_SHARE`` точек, но не круче правдоподобного.

    Ограничение обязательно: у короткой оси в конце стоит два-три узла, и её «наклон» — это шум
    формы буквы. Замер на 1973/08 с.85: у обрывка «с. 421» длиной 7 мм наклон начала вышел +0.696
    (35 градусов) при перекосе полосы ±1.2 градуса, и продолжение оси навстречу соседнему обрывку
    уезжало на 24 px — ряд не собирался.

    Args:
        axis: Ось строки.
        at_start: Считать наклон у левого конца (иначе у правого).

    Returns:
        Тангенс угла наклона, ограниченный ``EXTEND_SLOPE_LIMIT_DEG``.
    """
    points = axis.points
    if points.shape[0] < 3:
        return 0.0
    count = max(2, int(round(points.shape[0] * AXIS_END_SHARE)))
    part = points[:count] if at_start else points[-count:]
    run = float(part[-1, 0] - part[0, 0])
    if run <= 0:
        return 0.0
    limit = math.tan(math.radians(EXTEND_SLOPE_LIMIT_DEG))
    return float(np.clip((part[-1, 1] - part[0, 1]) / run, -limit, limit))


def _same_row(group: list[LineAxis], axis: LineAxis) -> bool:
    """Продолжает ли ось ряд, уже набранный в ``group``.

    Сравнивать медианы ординат (``LineAxis.cy``) нельзя: у наклонной строки левая и правая
    половины законно стоят на разной высоте, и на сильно искажённой бумаге разница доходит до
    полушага (1973/11 с.79, низ левой колонки: половинки x111..342 и x290..476 разнесены на 11 px
    при шаге 20). Тогда половинки не сливаются в ряд, и колонку режет пополам правило «слабое
    перекрытие по x» — те самые рваные границы блоков. Поэтому оси сводятся в ТОЧКЕ ВСТРЕЧИ:
    каждая продолжается от своего ближнего конца с собственным наклоном.

    Args:
        group: Оси, уже набранные в ряд.
        axis: Очередная ось (она правее или ниже: список отсортирован по ``cy``).

    Returns:
        ``True``, если ось — продолжение того же ряда.
    """
    tolerance = ROW_TOL_HEIGHTS * axis.height
    # Сравнивается только БЛИЖАЙШАЯ ось ряда: ряд растёт слева направо, и продолжать его должен
    # сосед, а не дальний кусок через полполосы.
    nearest = min(group, key=lambda item: _x_distance(item, axis))
    # Точка встречи — середина между ближними концами; если оси перекрываются по x, это середина
    # перекрытия.
    meeting = (max(nearest.x0, axis.x0) + min(nearest.x1, axis.x1)) / 2.0
    return abs(_level_at(nearest, meeting) - _level_at(axis, meeting)) <= tolerance


def _x_distance(item: LineAxis, axis: LineAxis) -> float:
    """Расстояние между осями по x: ноль, если они перекрываются."""
    return max(0.0, max(item.x0, axis.x0) - min(item.x1, axis.x1))


def _end_window(axis: LineAxis, at_start: bool) -> tuple[float, float]:
    """Опорная точка конца оси: медианы абсцисс и ординат крайних ``AXIS_END_SHARE`` узлов.

    Медиана, а не сам крайний узел: на коротком обрывке крайний узел сидит на запятой или выносном
    элементе и уводит отсчёт на полвысоты.
    """
    points = axis.points
    count = max(2, int(round(points.shape[0] * AXIS_END_SHARE)))
    part = points[:count] if at_start else points[-count:]
    return float(np.median(part[:, 0])), float(np.median(part[:, 1]))


def _level_at(axis: LineAxis, x: float) -> float:
    """Ордината оси в точке ``x``: внутри — интерполяция, за концом — продолжение по наклону конца.

    Вылет за конец ограничен ``EXTEND_REACH_MM``, а наклон — ``EXTEND_SLOPE_LIMIT_DEG``: обе
    величины местные, и на длинном пролёте они разъезжаются. Замер на 1973/08 с.85: обрывки одной
    строки сноски «ч. II,» (x 520–610) и «с. 421» (x 827–902) стоят на ординатах 1396 и 1397, но
    продолжение их осей навстречу через 108 px (18 мм) расходилось на 24 px при допуске 8 — ряд не
    собирался, огибающая блока обходила левый обрывок стороной, и ось оказывалась ВНЕ блока.

    Args:
        axis: Ось строки.
        x: Абсцисса, в которой нужна ордината (пиксели рабочей копии).

    Returns:
        Ордината оси или её продолжения.
    """
    if axis.x0 <= x <= axis.x1:
        return axis.y_at(x)
    at_start = x < axis.x0
    x_ref, y_ref = _end_window(axis, at_start)
    reach = mm_to_px(EXTEND_REACH_MM, axis.dpi)
    return y_ref + _axis_slope(axis, at_start) * float(np.clip(x - x_ref, -reach, reach))


def rows_of(
    axes: list[LineAxis],
    ink: np.ndarray,
    span: tuple[int, int],
    dpi: float,
    gutters: list | None = None,
    width: int | None = None,
    pad: float | None = None,
    leaders: list | None = None,
    fallback: bool = False,
) -> list[Row]:
    """Ряды колонки: оси с близкими ординатами сливаются, края берутся по краске рендера.

    Args:
        axes: Оси строк колонки.
        ink: Краска рендера ``RENDER_DPI`` (ненулевое — краска).
        span: Границы колонки ``(x0, x1)`` в пикселях рабочей копии (границы зоны).
        dpi: Разрешение рабочей копии.
        gutters: Межколонники-ломаные: окно поиска края берётся по ним НА ВЫСОТЕ РЯДА, чтобы на
            трапеции не заехать в соседнюю колонку (при межколоннике в 4 мм припуск в 5 мм уводил
            правый край ряда на букву соседней колонки — 1975/05 с.97).
        width: Ширина рабочей копии (нужна вместе с ``gutters``).
        pad: Припуск к границам колонки; по умолчанию ``COLUMN_PAD_MM``.
        fallback: Ряды ЗАПАСНОГО куска — тех строк, которым колонки не нашлось. У них границы
            колонки не проверяются: строка уже не поместилась ни в одну колонку, и мерить её
            снова по границе, взятой из межколонника, значит выбросить её совсем. Так терялись
            одиннадцать строк правой части 1971/10 с.93: межколонник там по высоте ряда
            интерполируется широким (x 606–726), текст начинается с x 641, и ряд не проходил
            проверку вылета. Слияние осей в ряд через живой межколонник запрещено и здесь.

    Returns:
        Ряды сверху вниз; ряды, у которых край не нашёлся, выбрасываются.
    """
    k = RENDER_DPI / dpi
    pad = mm_to_px(COLUMN_PAD_MM, dpi) if pad is None else pad
    groups: list[list[LineAxis]] = []
    for axis in sorted(axes, key=lambda item: item.cy):
        close = groups and _same_row(groups[-1], axis)
        # Оси по разные стороны живого межколонника в один ряд не сливаются: иначе строка левой
        # колонки и число правой графы (они на одной высоте) дают ряд во всю ширину страницы, а он
        # потом не помещается ни в одну колонку (1971/10 с.93, низ).
        if close and gutters is not None and _split_by_gutter(groups[-1], axis, gutters):
            close = False
        if close:
            groups[-1].append(axis)
        else:
            groups.append([axis])
    rows: list[Row] = []
    for group in groups:
        height = float(np.median([item.height for item in group]))
        y = float(np.median([item.cy for item in group]))
        row_x0 = float(min(item.x0 for item in group))
        row_x1 = float(max(item.x1 for item in group))
        margin = ROW_PAD_HEIGHTS * height
        y0 = int((y - height / 2.0 - margin) * k)
        y1 = int((y + height / 2.0 + margin) * k) + 1
        # Окно поиска края — границы колонки на высоте ряда плюс припуск, но не дальше середины
        # межколонника: иначе крайняя буква соседней колонки становится краем этого ряда.
        if gutters is not None and width is not None and not fallback:
            left_bound, right_bound = bounds_at(gutters, row_x0, row_x1, y, width)
        else:
            left_bound, right_bound = float(span[0]), float(span[1])
        x_lo = max(0.0, left_bound - pad)
        x_hi = min(float(ink.shape[1] / k), right_bound + pad)
        # Край ряда ищется рядом с его СТРОКАМИ: дальше живёт мусор (пылинка, след от прокола),
        # через который ось не проходит, но кромка блока к нему уезжала (1971/10 с.95, заголовок
        # «к Особым условиям» — контур уходил на два сантиметра влево).
        reach = mm_to_px(AXIS_REACH_MM, dpi)
        x_lo = max(x_lo, row_x0 - reach)
        x_hi = min(x_hi, row_x1 + reach)
        if gutters is not None and not fallback:
            for gutter in gutters:
                if not gutter.alive_at(y):
                    continue
                gx0, gx1 = gutter.x0_at(y), gutter.x1_at(y)
                # Раньше межколонник, залитый точками, край ряда не ограничивал — но отточия
                # теперь и так входят в краску текста, а послабление уводило правый край ряда
                # в соседнюю графу (1971/10 с.93, низ).
                if gx1 <= left_bound + 1:
                    x_lo = max(x_lo, (gx0 + gx1) / 2.0)
                elif gx0 >= right_bound - 1:
                    x_hi = min(x_hi, (gx0 + gx1) / 2.0)
        left = ink_edge(ink, int(x_lo * k), int(x_hi * k), y0, y1, "left")
        right = ink_edge(ink, int(x_lo * k), int(x_hi * k), y0, y1, "right")
        if left is None or right is None:
            continue
        band = ink[max(0, y0) : y1, int(left) : int(right) + 1]
        # Точки отточия толще штриха корпуса вдвое и проходят нижний порог глифа: если их не
        # исключить, строка таблицы выглядит «другим набором» и блок дробится.
        skip = _leader_columns(leaders, y, height, left, band.shape[1], k)
        stroke = stroke_width(band, skip) / k
        glyph_w, glyph_h = glyph_metrics(band, k, dpi, skip)
        left, right = left / k, right / k
        axis_points = np.vstack([np.asarray(item.points, dtype=np.float64) for item in group])
        axis_points = axis_points[np.argsort(axis_points[:, 0])]
        # Профиль краски берётся по ВСЕЙ длине осей ряда, а не только между краями, найденными
        # ``ink_edge``: край ищется по заполненности столбца и отбрасывает тонкие элементы —
        # буквицу-украшение заголовка (1973/11 с.79: край ряда 163 при оси от 111). Ось же
        # проходит через них, и без этого расширения кромка блока их срезала.
        profile_x0 = min(left, float(axis_points[0, 0]))
        profile_x1 = max(right, float(axis_points[-1, 0]))
        top_edge, bottom_edge = edge_profiles(ink, axis_points, profile_x0, profile_x1, height, k, dpi)
        # Край, вылезший за колонку дальше OVERHANG_MM, — не край этого ряда: так выглядит
        # строка, сшитая через межколонник (1973/07 с.88), и колонтитул во всю ширину.
        overhang = mm_to_px(OVERHANG_MM, dpi)
        if left < left_bound - overhang or right > right_bound + overhang:
            continue
        rows.append(
            Row(
                y=y,
                height=height,
                x0=left,
                x1=right,
                axes=tuple(group),
                stroke=stroke,
                glyph_w=glyph_w,
                glyph_h=glyph_h,
                top_edge=top_edge,
                bottom_edge=bottom_edge,
            )
        )
    return rows


def stroke_width(band: np.ndarray, skip: np.ndarray | None = None) -> float:
    """Толщина штриха ряда: медиана длин горизонтальных пробегов краски (пиксели того же масштаба).

    Признак кегля, не зависящий от высоты бокса: у корпуса журнала штрих около 0.25 мм, у жирного
    заголовка — вдвое-втрое толще. Считается по строкам пикселей разом: пробеги ищутся по местам,
    где краска начинается и кончается.

    Args:
        band: Полоса краски ряда (``True`` — краска).

    Returns:
        Медианная длина пробега; 0.0, если краски нет.
    """
    if band.size == 0 or not band.any():
        return 0.0
    if skip is not None and skip.any() and not skip.all():
        band = band[:, ~skip]
        if band.size == 0 or not band.any():
            return 0.0
    padded = np.pad(band.astype(np.int8), ((0, 0), (1, 1)))
    changes = np.diff(padded, axis=1)
    starts = np.argwhere(changes == 1)
    ends = np.argwhere(changes == -1)
    if starts.shape[0] == 0 or starts.shape[0] != ends.shape[0]:
        return 0.0
    lengths = ends[:, 1] - starts[:, 1]
    return float(np.median(lengths))


def edge_profiles(
    ink: np.ndarray, axis_points: np.ndarray, x0: float, x1: float, height: float, k: float, dpi: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Верхний и нижний профили краски ряда вдоль его ОСИ, по столбцам с шагом ``PROFILE_STEP_MM``.

    По ним строятся верхняя и нижняя кромки блока. Искать краску в прямоугольной полосе ряда
    нельзя: наклонная строка в неё не влезает (на 1970/02 с.90 строка уходит по вертикали на
    12 px при полосе в 22 px), профиль упирается в края полосы и кромка получается ПЛОСКОЙ, хотя
    строка изгибается. Поэтому окно поиска едет вдоль оси строки: в столбце ``x`` краска ищется
    в ``±PROFILE_WINDOW_HEIGHTS`` высоты вокруг оси, а за концами осей ось продлевается по наклону
    крайнего участка (ограниченно, как и сама кромка).

    Args:
        ink: Краска текста рендера ``RENDER_DPI``.
        axis_points: Точки осей ряда ``(M, 2)`` в пикселях рабочей копии.
        x0, x1: Края ряда в пикселях рабочей копии.
        height: Высота ряда в пикселях рабочей копии.
        k: Во сколько раз рендер крупнее рабочей копии.
        dpi: Разрешение рабочей копии (в нём отдаются профили).

    Returns:
        Пара ``(верх, низ)``: ломаные ``(N, 2)`` в пикселях рабочей копии или ``None``, если
        краски не нашлось.
    """
    if axis_points is None or len(axis_points) < 2 or x1 - x0 < 1:
        return None, None
    rows_count, cols_count = ink.shape
    grid = np.arange(x0, x1 + mm_to_px(PROFILE_STEP_MM, dpi), mm_to_px(PROFILE_STEP_MM, dpi))
    if grid.size < 2:
        return None, None
    # Центр окна — ось строки; за её концами ось продлевается по наклону крайнего участка.
    centres = _extend_ends(grid, np.interp(grid, axis_points[:, 0], axis_points[:, 1]), axis_points, dpi)
    half = max(2.0, PROFILE_WINDOW_HEIGHTS * height)
    xs: list[float] = []
    tops: list[float] = []
    bottoms: list[float] = []
    step = max(2, int(round(mm_to_px(PROFILE_STEP_MM, dpi) * k)))
    for x, centre in zip(grid, centres):
        left = int(x * k)
        top_row = max(0, int((centre - half) * k))
        bottom_row = min(rows_count, int((centre + half) * k) + 1)
        right = min(cols_count, left + step)
        if right <= left or bottom_row <= top_row:
            continue
        chunk = ink[top_row:bottom_row, left:right] > 0
        rows_ink = np.nonzero(chunk.any(axis=1))[0]
        if rows_ink.size == 0:
            continue
        xs.append(float(x))
        tops.append((top_row + float(rows_ink[0])) / k)
        bottoms.append((top_row + float(rows_ink[-1])) / k)
    if len(xs) < 2:
        return None, None
    # Кромка должна быть СНАРУЖИ всех символов и при этом гладкой: сначала берётся крайнее
    # значение в окне (выносные элементы и прописные оказываются внутри), затем окно сглаживается.
    # Без этого кромка зубчатая — она ныряет между буквами.
    window = max(3, int(round(PROFILE_SMOOTH_MM / max(PROFILE_STEP_MM, 1e-6))) | 1)
    top = _smooth(grey_erosion(np.asarray(tops, dtype=np.float64), size=window, mode="nearest"), window)
    bottom = _smooth(grey_dilation(np.asarray(bottoms, dtype=np.float64), size=window, mode="nearest"), window)
    own = np.asarray(xs, dtype=np.float64)
    return np.column_stack([own, top]), np.column_stack([own, bottom])


def glyph_metrics(band: np.ndarray, k: float, dpi: float, skip: np.ndarray | None = None) -> tuple[float, float]:
    """Медианные ширина и высота символа в полосе ряда (пиксели рабочей копии).

    Размер символа нужен, чтобы раздуть границу блока на его половину: кромка идёт по самым
    дальним точкам краски, и выносные элементы («р», «д») с прописными делают её заметно более
    неровной, чем оси строк.

    Отбор компонент — как в ``page_layout.rotated_text.docstrum``: от ``GLYPH_MIN_MM`` до
    ``GLYPH_MAX_MM``. Нижний порог заодно выбрасывает точки отточий (0.5 мм), иначе они тянут
    медиану вниз.

    Args:
        band: Краска полосы ряда (пиксели рендера).
        k: Во сколько раз рендер крупнее рабочей копии.
        dpi: Разрешение рабочей копии.

    Returns:
        Пара ``(ширина, высота)``; ``(0.0, 0.0)``, если глифов не нашлось.
    """
    if band.size == 0 or not band.any():
        return 0.0, 0.0
    if skip is not None and skip.any() and not skip.all():
        band = band.copy()
        band[:, skip] = False
        if not band.any():
            return 0.0, 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), 8)
    low, high = mm_to_px(GLYPH_MIN_MM, dpi) * k, mm_to_px(GLYPH_MAX_MM, dpi) * k
    widths, heights = [], []
    for index in range(1, count):
        width = float(stats[index, cv2.CC_STAT_WIDTH])
        height = float(stats[index, cv2.CC_STAT_HEIGHT])
        if not (low <= height <= high and low <= width <= GLYPH_MAX_WIDTH_RATIO * high):
            continue
        widths.append(width / k)
        heights.append(height / k)
    if len(widths) < MIN_GLYPHS_FOR_SIZE:
        return 0.0, 0.0
    return float(np.median(widths)), float(np.median(heights))


def dilate_polygon(polygon: np.ndarray, dx: float, dy: float, dpi: float) -> np.ndarray | None:
    """Морфологическая дилатация замкнутого контура эллиптическим ядром ``(2·dx+1, 2·dy+1)``.

    Контур растеризуется, раздувается и снимается заново. Побочный эффект — он и нужен: впадины
    уже двух радиусов заполняются, и граница блока становится ровнее, а не только шире.

    Args:
        polygon: Замкнутый контур ``(N, 2)`` в пикселях рабочей копии.
        dx, dy: Полуоси ядра в тех же пикселях.
        dpi: Разрешение рабочей копии (для допуска упрощения контура).

    Returns:
        Новый контур ``(M, 2)`` или ``None``, если исходный вырожден.
    """
    if polygon is None or len(polygon) < 3 or dx < 0.5 or dy < 0.5:
        return None
    pad = int(max(dx, dy)) + mm_to_px(RASTER_PAD_MM, dpi)
    origin = polygon.min(axis=0) - pad
    size = np.ceil(polygon.max(axis=0) + pad - origin).astype(int)[::-1]
    if size.min() <= 2:
        return None
    mask = np.zeros(tuple(size), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon - origin).astype(np.int32)], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(round(dx)) + 1, 2 * int(round(dy)) + 1))
    contours, _ = cv2.findContours(cv2.dilate(mask, kernel), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, mm_to_px(APPROX_EPS_MM, dpi), True)
    return contour.reshape(-1, 2).astype(np.float64) + origin


def _leader_columns(
    leaders: list | None, y: float, height: float, left: float, width: int, k: float
) -> np.ndarray | None:
    """Булев вектор по столбцам полосы ряда: какие из них заняты отточием.

    Args:
        leaders: Отточия страницы или ``None``.
        y: Ордината середины ряда (пиксели рабочей копии).
        height: Высота ряда.
        left: Левый край полосы в пикселях рендера.
        width: Ширина полосы в столбцах рендера.
        k: Во сколько раз рендер крупнее рабочей копии.

    Returns:
        Вектор длиной ``width`` или ``None``, если отточий нет.
    """
    if not leaders or width <= 0:
        return None
    spans = spans_at(leaders, y, max(height, MIN_LEADER_PROBE_PX))
    if not spans:
        return None
    return inside_spans(np.arange(width, dtype=np.float64) + left, [(x0 * k, x1 * k) for x0, x1 in spans])


def _split_by_gutter(group: list[LineAxis], axis: LineAxis, gutters: list) -> bool:
    """Разделён ли кандидат с рядом живым межколонником на их общей высоте."""
    for gutter in gutters:
        if not gutter.alive_at(axis.cy):
            continue
        left, right = gutter.x0_at(axis.cy), gutter.x1_at(axis.cy)
        for item in group:
            if (item.x1 <= left and axis.x0 >= right) or (axis.x1 <= left and item.x0 >= right):
                return True
    return False


def row_gap(previous: Row, row: Row) -> float:
    """Расстояние между соседними рядами ВДОЛЬ НАКЛОНА: по ординатам осей в общей точке по x.

    Разница ординат центров на наклонных строках врёт: у двух рядов одной колонки центры могут
    разойтись на высоту строки просто из-за наклона, а чужой ряд (номер полосы, уехавший под
    блок) — наоборот, оказаться рядом. В общей точке по x такого не происходит.
    """
    x_lo = max(previous.axis.x0, row.axis.x0)
    x_hi = min(previous.axis.x1, row.axis.x1)
    if x_hi <= x_lo:
        return row.y - previous.y
    x = (x_lo + x_hi) / 2.0
    return row.axis.y_at(x) - previous.axis.y_at(x)


def _soft_style_break(previous: Row, row: Row) -> bool:
    """Слабая смена набора между двумя соседними рядами: штрих или кегль отличаются заметно.

    Признак работает только В ПАРЕ с увеличенным разрывом (``SOFT_GAP_PITCHES``). Порог здесь
    ниже, чем у :func:`_style_break`, и сравниваются сами ряды, а не окна: заголовок статьи в две
    строки («Инструкция о порядке и сроках возврата тары…», 1971/10 с.95) от соседей по кеглю
    почти не отличается, а по штриху — в 1.2 раза, и вместе с разрывом в 1.8–2.2 шага этого
    достаточно. Замер по паку: из 14 разрывов длиннее 1.6 шага внутри блоков смена набора есть у
    пяти — и это как раз границы заголовков.

    Args:
        previous: Верхний ряд.
        row: Следующий за ним ряд.

    Returns:
        ``True``, если ряды набраны по-разному.
    """
    strokes = sorted((previous.stroke, row.stroke))
    heights = sorted((previous.height, row.height))
    bold = strokes[0] > 0 and strokes[1] > SOFT_STYLE_RATIO * strokes[0]
    # Тот же предохранитель, что у ``_style_break``: высота бокса врёт на наклонных строках.
    tall = not _same_glyph_size([previous], [row]) and heights[1] > SOFT_STYLE_RATIO * max(heights[0], 1e-6)
    return bold or tall


def _split_by_style(rows: list[Row]) -> list[list[Row]]:
    """Разделить группу рядов там, где меняется набор (кегль и толщина штриха)."""
    if len(rows) < 2:
        return [rows]
    out: list[list[Row]] = [[rows[0]]]
    for index in range(len(rows) - 1):
        if _style_break(rows, index):
            out.append([rows[index + 1]])
        else:
            out[-1].append(rows[index + 1])
    return out


def _style_break(rows: list[Row], index: int) -> bool:
    """Устойчивая смена НАБОРА на границе после ряда ``index``: кегль и толщина штриха.

    Сравниваются МЕДИАНЫ по нескольким рядам с каждой стороны, а не сами соседи: высота
    отдельного ряда корпуса скачет на выносных элементах и прописных (13 против 17 px), и по
    соседям колонка дробилась на куски. Заголовок же отличается от корпуса устойчиво — и по
    высоте, и по толщине штриха (у корпуса журнала штрих около 0.25 мм, у заголовка вдвое
    толще). Толщина добавлена потому, что одной высоты не хватало: набранный тем же кеглем, но
    жирный заголовок оставался внутри блока корпуса.

    На коротких группах (заголовок в две строки над корпусом) окно сужается, но тогда
    требуется СОГЛАСОВАННЫЙ скачок обоих признаков — одиночный выброс так границу не поставит.

    Args:
        rows: Ряды блока сверху вниз.
        index: Граница проверяется между ``rows[index]`` и ``rows[index + 1]``.

    Returns:
        ``True``, если по обе стороны границы разный набор.
    """
    # Окно в один ряд не годится: высота и штрих отдельной строки скачут (прописная в начале
    # абзаца, разрядка), и первая строка вводки отрывалась в свой блок (1970/02 с.90).
    window = min(HEIGHT_WINDOW, index + 1, len(rows) - index - 1)
    # У группы ровно из двух рядов окна шире одного не бывает, и правило отключалось совсем:
    # колонтитул «Нам пишут» (высота 18 px, штрих 2.5) и заголовок рубрики «Нам пишут…»
    # (35.5 и 4.0) оставались одним блоком, когда между ними не находилась линейка
    # (1975/05 с.97). Для пары рядов окно в один ряд допускается, но решает только совместная
    # мера ниже — произведение отношений кегля и штриха (здесь 3.16 при пороге 1.9).
    if window < MIN_BREAK_WINDOW and len(rows) != 2:
        return False
    if window < 1:
        return False
    before = rows[index + 1 - window : index + 1]
    after = rows[index + 1 : index + 1 + window]
    heights = sorted(
        (float(np.median([row.height for row in before])), float(np.median([row.height for row in after])))
    )
    strokes = sorted(
        (float(np.median([row.stroke for row in before])), float(np.median([row.stroke for row in after])))
    )
    # Признак кегля гасится, если РАЗМЕР ГЛИФА с обеих сторон один и тот же: высота ряда мерит
    # бокс, а он раздувается от наклона строки и от одной прописной, тогда как медианный размер
    # компоненты краски — честная мера кегля. Штриха это не касается: заголовок того же кегля,
    # но жирный, границей остаётся.
    same_size = _same_glyph_size(before, after)
    tall = not same_size and heights[1] > HEIGHT_BREAK_RATIO * max(heights[0], 1e-6)
    bold = strokes[0] > 0 and strokes[1] > STROKE_BREAK_RATIO * strokes[0]
    if window >= HEIGHT_WINDOW:
        return tall or bold
    # Короткая группа: берётся СОВМЕСТНАЯ мера — произведение отношений кегля и штриха. Требовать
    # скачка каждого признака по отдельности оказалось слишком строго: заголовок «ОСНОВЫ
    # ЭКОНОМИКИ…» над подзаголовком «Тема 11…» даёт 1.30 по кеглю и 1.67 по штриху, и оба порога
    # он проходил впритык (1973/06 с.65). Замер по паку: внутри блоков произведение имеет
    # p99 = 1.62 при максимуме 2.16, а у этой границы — 2.17.
    if heights[0] <= 0 or strokes[0] <= 0:
        return False
    size_ratio = 1.0 if same_size else heights[1] / heights[0]
    return size_ratio * (strokes[1] / strokes[0]) > STYLE_PRODUCT_RATIO


def _same_glyph_size(before: list[Row], after: list[Row]) -> bool:
    """Одинаков ли КЕГЛЬ по обе стороны границы — по медианному размеру глифа, а не по боксу.

    Размер глифа (``glyph_w``/``glyph_h``, медиана компонент краски в полосе ряда) не зависит ни
    от наклона строки, ни от того, слиплись ли сгустки, ни от прописной в начале абзаца — а
    высота ряда зависит от всего этого сразу. Если глифы совпали по обоим измерениям, смены
    кегля нет, чего бы ни показывала высота бокса.

    Args:
        before: Ряды по верхнюю сторону границы.
        after: Ряды по нижнюю сторону.

    Returns:
        ``True``, если кегль одинаков. Если размер глифа где-то не померился (нули), — ``False``:
        молчащая мера ничего не отменяет.
    """
    heights = sorted(
        (float(np.median([row.glyph_h for row in before])), float(np.median([row.glyph_h for row in after])))
    )
    widths = sorted(
        (float(np.median([row.glyph_w for row in before])), float(np.median([row.glyph_w for row in after])))
    )
    if heights[0] <= 0 or widths[0] <= 0:
        return False
    return heights[1] <= GLYPH_SAME_RATIO * heights[0] and widths[1] <= GLYPH_SAME_RATIO * widths[0]


def _rule_between(previous: Row, row: Row, rules: list, span: tuple[int, int]) -> bool:
    """Проходит ли между рядами сплошная черта, перекрывающая колонку.

    Разделитель сноски — именно такая черта: текст под ней относится к сноске, а не к блоку
    (1973/06 с.65, правая колонка).
    """
    width = max(1, span[1] - span[0])
    for rule in rules:
        if not (previous.y < rule.cy < row.y):
            continue
        overlap = min(rule.x1, span[1]) - max(rule.x0, span[0])
        if overlap >= RULE_OVERLAP_SHARE * width:
            return True
    return False


def _weak_overlap(previous: Row, row: Row, reference: float) -> bool:
    """Расходятся ли соседние ряды по горизонтали настолько, что это разные блоки.

    Строки одного блока стоят друг под другом и перекрываются по x почти целиком. А логотип
    рубрики слева и заголовок заметки справа под ним (1975/05 с.97) почти не перекрываются — это
    два блока, и общая огибающая протянулась бы через пустоту наискосок.

    Проверка идёт только между ДЛИННЫМИ рядами (не короче ``LONG_ROW_SHARE`` от самого длинного
    ряда группы): короткий конец абзаца в одно слово перекрывается со следующей строкой чуть-чуть
    просто потому, что он короткий, и блок рвался посреди текста (1968/07 с.93).

    Args:
        previous: Верхний ряд.
        row: Следующий за ним ряд.
        reference: Длина самого длинного ряда группы.

    Returns:
        ``True``, если оба ряда длинные, а перекрытие меньше ``MIN_ROW_OVERLAP`` короткого из них.
    """
    widths = (previous.x1 - previous.x0, row.x1 - row.x0)
    if reference <= 0:
        return False
    if min(widths) >= LONG_ROW_SHARE * reference:
        overlap = min(previous.x1, row.x1) - max(previous.x0, row.x0)
        return overlap < MIN_ROW_OVERLAP * min(widths)
    # Один из рядов короткий. Конец абзаца стоит по левому краю блока, а подпись автора или
    # заголовок в подбор отодвинуты от ОБЕИХ кромок длинного соседа (1968/07 с.93: «А. ИВАНОВА»).
    short, long_row = (previous, row) if widths[0] < widths[1] else (row, previous)
    dpi = short.axis.dpi
    # Обе строки набраны по ОДНОМУ центру — это одна центрированная группа (первая строка
    # заголовка «Инструкция» над «о порядке и сроках возврата тары…», 1971/10 с.95), и делить её
    # нельзя. У подписи автора под колонкой центры расходятся на сантиметры.
    centres = ((short.x0 + short.x1) / 2.0, (long_row.x0 + long_row.x1) / 2.0)
    if abs(centres[0] - centres[1]) <= mm_to_px(CENTRE_TOL_MM, dpi):
        return False
    offset = mm_to_px(OFFSET_MM, dpi)
    return short.x0 - long_row.x0 > offset and long_row.x1 - short.x1 > offset


def split_blocks(
    rows: list[Row],
    pitch: float,
    rules: list | None = None,
    span: tuple[int, int] | None = None,
    body_height: float = 0.0,
) -> list[list[Row]]:
    """Разделить ряды колонки на блоки по вертикальным разрывам больше ``BLOCK_GAP_PITCHES`` шагов.

    Обрывок из одной-двух строк (пропала строка при сегментации — жирный или разрядка) к блоку
    не относится сам по себе: он приклеивается к соседней группе, если разрыв до неё меньше
    ``MERGE_GAP_PITCHES`` шагов. Настоящий разрыв между заметками (1975/05 с.97 — полтора десятка
    шагов) так не склеится.
    """
    if len(rows) < 2:
        return [rows]
    groups: list[list[Row]] = [[rows[0]]]
    limit = BLOCK_GAP_PITCHES * pitch
    reference = max(row.x1 - row.x0 for row in rows)
    rules = rules or []
    for index, (previous, row) in enumerate(zip(rows[:-1], rows[1:])):
        gap = row_gap(previous, row)
        # В шапке статьи строки идут с разным интервалом, и медианный шаг смешанной группы ни о
        # чём не говорит: логотип рубрики отделён от заголовка 4.7 высотами, а две строки самого
        # заголовка — полутора (1973/06 с.65). Поэтому у КРУПНОГО набора разрыв меряется в высотах
        # самих строк. У корпуса так мерить нельзя: пропавшая при сегментации строка оставляет
        # двойной интервал, и колонка порвалась бы посреди абзаца.
        smaller = min(previous.height, row.height)
        large_type = body_height > 0 and smaller > LARGE_TYPE_RATIO * body_height
        divided = (
            gap > limit
            or (large_type and gap > GAP_HEIGHTS * smaller)
            or (gap > SOFT_GAP_PITCHES * pitch and _soft_style_break(previous, row))
            or _weak_overlap(previous, row, reference)
            or (span is not None and _rule_between(previous, row, rules, span))
        )
        if divided:
            groups.append([row])
        else:
            groups[-1].append(row)
    # Смена набора ищется ВНУТРИ уже разделённых групп: иначе окно сравнения перешагивает через
    # готовую границу и в него попадает чужой кегль (логотип рубрики сверху — 1973/06 с.65,
    # заголовок над вводкой — 1970/02 с.90), отчего отрывалась первая строка следующей группы.
    groups = [piece for group in groups for piece in _split_by_style(group)]
    merged: list[list[Row]] = []
    for group in groups:
        gap = row_gap(merged[-1][-1], group[0]) if merged else None
        small = len(group) < MIN_BLOCK_ROWS or (merged and len(merged[-1]) < MIN_BLOCK_ROWS)
        joinable = (
            merged
            and small
            and gap is not None
            and gap <= MERGE_GAP_PITCHES * pitch
            and not _style_break([*merged[-1], *group], len(merged[-1]) - 1)
            and not (span is not None and _rule_between(merged[-1][-1], group[0], rules, span))
        )
        if joinable:
            merged[-1].extend(group)
        else:
            merged.append(group)
    return merged


def pitch_of(rows: list[Row]) -> float:
    """Межстрочный шаг блока: медиана расстояний между серединами соседних рядов (пиксели)."""
    if len(rows) < 2:
        return float(rows[0].height * 1.6) if rows else 0.0
    gaps = np.array([row_gap(previous, row) for previous, row in zip(rows[:-1], rows[1:])])
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return float(rows[0].height * 1.6)
    # Между абзацами и вокруг заголовка шаг больше: берём медиану, а не среднее.
    return float(np.median(gaps))


def _quantile_trend(ys: np.ndarray, xs: np.ndarray, grid: np.ndarray, window: float, quantile: float) -> np.ndarray:
    """Грубый квантильный тренд ``x(y)``: в окне ``window`` вокруг узла сетки — квантиль ``xs``.

    Нужен только как первое приближение, по которому отбираются ряды тела блока: сам по себе
    квантиль в скользящем окне прыгает там, где в окно попало два-три ряда.
    """
    out = np.empty(grid.shape, dtype=np.float64)
    half = window / 2.0
    for i, node in enumerate(grid):
        own = xs[np.abs(ys - node) <= half]
        if own.size == 0:
            # Окно пустое (разрыв между абзацами шире окна) — берём ближайшую точку.
            own = xs[np.argsort(np.abs(ys - node))[:1]]
        out[i] = float(np.quantile(own, quantile))
    return out


def _local_line(ys: np.ndarray, xs: np.ndarray, grid: np.ndarray, window: float) -> np.ndarray:
    """Локальная взвешенная прямая (LOWESS первой степени) ``x(y)`` в узлах сетки.

    Прямая в окне, а не среднее: на наклонном блоке среднее сдвигает кромку внутрь на полокна.
    Веса — трикубические по расстоянию до узла, поэтому кривая не дёргается, когда ряд входит в
    окно и выходит из него. Если в окно попало меньше ``MIN_TREND_ROWS`` рядов, окно
    расширяется до ближайших ``MIN_TREND_ROWS``: по двум-трём рядам кромку не провести, и
    именно на этом ломалась кромка у коротких концов абзацев (1973/07 с.77).
    """
    out = np.empty(grid.shape, dtype=np.float64)
    half = max(window / 2.0, 1e-6)
    if ys.size == 0:
        return out
    for i, node in enumerate(grid):
        distance = np.abs(ys - node)
        reach = half
        if (distance <= reach).sum() < MIN_TREND_ROWS:
            nearest = np.sort(distance)[: min(MIN_TREND_ROWS, ys.size)]
            reach = max(reach, float(nearest[-1]))
        own = distance <= reach
        if own.sum() == 0:
            out[i] = float(xs[np.argmin(distance)])
            continue
        weights = (1.0 - (distance[own] / reach) ** 3) ** 3
        if own.sum() >= 2 and weights.sum() > 0:
            slope, intercept = np.polyfit(ys[own], xs[own], 1, w=np.sqrt(weights))
            out[i] = slope * node + intercept
        else:
            out[i] = float(np.average(xs[own], weights=weights))
    return out


def _smooth(values: np.ndarray, window_nodes: int) -> np.ndarray:
    """Сглаживание кромки по сетке: Савицкий–Голей второй степени, окно в узлах."""
    window = max(3, int(window_nodes) | 1)
    if values.size <= window:
        return values
    return savgol_filter(values, window_length=window, polyorder=2, mode="nearest")


def _row_band(row: Row, margin: float) -> np.ndarray | None:
    """Замкнутый контур КРАСКИ одного ряда: верхний профиль слева направо, нижний — обратно.

    По этому контуру проверяется и чинится главное требование к границе блока: в неё должны
    попадать все буквы всех его рядов. Профили краски (:func:`edge_profiles`) идут вдоль оси
    строки и лежат снаружи всех её символов, так что контур ряда — это ровно его буквы.

    Args:
        row: Ряд блока.
        margin: Поле наружу (пиксели рабочей копии) по обеим осям.

    Returns:
        Контур ``(N, 2)`` или ``None``, если у ряда нет ни профилей краски, ни осей.
    """
    top, bottom = row.top_edge, row.bottom_edge
    if top is None or bottom is None or len(top) < 2 or len(bottom) < 2:
        if not row.axes:
            return None
        points = np.vstack([np.asarray(axis.points, dtype=np.float64) for axis in row.axes])
        points = points[np.argsort(points[:, 0])]
        top = np.column_stack([points[:, 0], points[:, 1] - row.height / 2.0])
        bottom = np.column_stack([points[:, 0], points[:, 1] + row.height / 2.0])
    upper = np.column_stack([top[:, 0], top[:, 1] - margin])
    lower = np.column_stack([bottom[:, 0], bottom[:, 1] + margin])
    # Поле по x: крайние столбцы профиля — это крайние столбцы краски, и без поля первая и
    # последняя буквы лежали бы ровно на контуре.
    upper = np.vstack([[upper[0, 0] - margin, upper[0, 1]], upper, [upper[-1, 0] + margin, upper[-1, 1]]])
    lower = np.vstack([[lower[0, 0] - margin, lower[0, 1]], lower, [lower[-1, 0] + margin, lower[-1, 1]]])
    return np.vstack([upper, lower[::-1]])


def _swallow_rows(polygon: np.ndarray, rows: list[Row], dpi: float) -> np.ndarray:
    """Дотянуть контур блока туда, где он режет краску своих же рядов.

    Гладкая рамка блока строится в осях «ордината → края»: боковые кромки — функции ``x(y)``, а
    верх и низ — профили краски крайних рядов, натянутые между кромками. Это верно, пока ряд
    занимает узкую горизонтальную полосу. Если СОБСТВЕННЫЙ изгиб строки больше межстрочного шага,
    модель рассыпается: на сноске 1973/08 с.85 ось первой строки идёт 1382 → 1368 → 1403 (размах
    35 px при шаге между рядами 20.5 px), ряды перекрываются по ординате целиком, и правая кромка
    обязана обрушиться с x = 903 до x = 612 за восемь пикселей высоты — эта диагональ и срезала
    «Соч., т. 25,» (конец оси оказался на 17.6 px снаружи).

    Чинится не рамка, а контур: к нему подмешивается краска рядов. Блоки, где резать нечего, не
    меняются вовсе — проверка идёт по растру рамки, и объединение делается, только если хоть одна
    точка ряда оказалась снаружи.

    Args:
        polygon: Контур блока ``(N, 2)``, собранный обходом четырёх кромок.
        rows: Ряды блока.
        dpi: Разрешение рабочей копии.

    Returns:
        Тот же контур, если он уже охватывает все ряды, иначе объединённый с их краской.
    """
    if polygon is None or len(polygon) < 3 or not rows:
        return polygon
    margin = mm_to_px(OUTWARD_MARGIN_MM, dpi)
    bands = [band for band in (_row_band(row, margin) for row in rows) if band is not None and len(band) >= 3]
    if not bands:
        return polygon
    pad = mm_to_px(RASTER_PAD_MM, dpi)
    points = np.vstack([polygon, *bands])
    origin = np.floor(points.min(axis=0)) - pad
    size = np.ceil(points.max(axis=0) + pad - origin).astype(int)[::-1]
    if size.min() <= 2:
        return polygon
    mask = np.zeros(tuple(size), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon - origin).astype(np.int32)], 255)
    # Проверка по растру, а не ``pointPolygonTest`` по каждой точке: точек в профилях тысячи, а
    # индексация массива стоит копейки.
    shifted = [np.round(band - origin).astype(np.int32) for band in bands]
    if all(bool(mask[band[:, 1], band[:, 0]].all()) for band in shifted):
        return polygon
    cv2.fillPoly(mask, shifted, 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return polygon
    contour = max(contours, key=cv2.contourArea)
    # Упрощение срезает углы на величину своего допуска, а обещание «все буквы внутри» должно
    # держаться и после него: упрощённый контур принимается, только если ряды по-прежнему внутри.
    simple = cv2.approxPolyDP(contour, mm_to_px(APPROX_EPS_MM, dpi), True)
    if simple.shape[0] >= 3:
        check = np.zeros(tuple(size), dtype=np.uint8)
        cv2.fillPoly(check, [simple], 255)
        if all(bool(check[band[:, 1], band[:, 0]].all()) for band in shifted):
            contour = simple
    if contour.shape[0] < 3:
        return polygon
    return contour.reshape(-1, 2).astype(np.float64) + origin


def _single_row_envelope(
    row: Row, smooth_pitches: float, dpi: float, glyph: tuple[float, float] | None, dilate: float
) -> BlockEnvelope:
    """Огибающая блока из одного ряда: по его оси, с полем в половину высоты.

    Тренд по одному ряду не построить, поэтому кромки — вертикальные отрезки по краям краски, а
    верх и низ — сама ось, поднятая и опущенная на полвысоты.
    """
    top = _cap(row, row.x0, row.x1, -1.0)
    bottom = _cap(row, row.x0, row.x1, 1.0)
    # Боковые кромки соединяют верх и низ ПО КРАСКЕ, а не отмеряют полвысоты ряда от оси: у
    # логотипа рубрики высота ряда 121 px при 76 px фактической краски, и из контура торчали
    # вертикальные «усы» на полсантиметра (1973/06 с.65).
    left = np.array([[row.x0, top[0, 1]], [row.x0, bottom[0, 1]]])
    right = np.array([[row.x1, top[-1, 1]], [row.x1, bottom[-1, 1]]])
    polygon = _swallow_rows(np.vstack([left, bottom, right[::-1], top[::-1]]), [row], dpi)
    return BlockEnvelope(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        polygon=polygon,
        smooth_pitches=float(smooth_pitches),
        polygon_dilated=_dilated(polygon, glyph, dilate, dpi),
        dilate_px=(dilate * glyph[0], dilate * glyph[1]) if glyph else (0.0, 0.0),
    )


def envelope_of(
    rows: list[Row],
    pitch: float,
    dpi: float,
    smooth_pitches: float,
    glyph: tuple[float, float] | None = None,
    dilate: float = DILATE_GLYPHS,
) -> BlockEnvelope:
    """Гладкая огибающая блока с окном ``smooth_pitches`` межстрочных интервалов.

    Левая и правая кромки — квантильный тренд краёв рядов по сетке шага ``GRID_STEP_MM``;
    верх и низ — оси крайних рядов, поднятые (опущенные) на половину высоты и продлённые до
    кромок. Контур собирается обходом: левая сверху вниз, низ слева направо, правая снизу
    вверх, верх справа налево.
    """
    if len(rows) == 1:
        return _single_row_envelope(rows[0], smooth_pitches, dpi, glyph, dilate)
    ys = np.array([row.y for row in rows], dtype=np.float64)
    xs_left = np.array([row.x0 for row in rows], dtype=np.float64)
    xs_right = np.array([row.x1 for row in rows], dtype=np.float64)
    step = mm_to_px(GRID_STEP_MM, dpi)
    top_row, bottom_row = rows[0], rows[-1]
    y_top = top_row.y - top_row.height / 2.0
    y_bottom = bottom_row.y + bottom_row.height / 2.0
    grid = np.arange(y_top, y_bottom + step, step)
    window = max(smooth_pitches * pitch, 2 * step)
    core_left = _side_trend(ys, xs_left, grid, window, dpi, inward=+1.0)
    core_right = _side_trend(ys, xs_right, grid, window, dpi, inward=-1.0)
    nodes = max(3, int(round(window / step)))
    core_left, core_right = _smooth(core_left, nodes), _smooth(core_right, nodes)
    # Кромка для контура — тот же тренд, отодвинутый наружу до самых дальних краёв рядов.
    left = _outward(core_left, ys, xs_left, grid, pitch, nodes, dpi, inward=+1.0)
    right = _outward(core_right, ys, xs_right, grid, pitch, nodes, dpi, inward=-1.0)
    top_curve = _cap(top_row, left[0], right[0], -1.0)
    bottom_curve = _cap(bottom_row, left[-1], right[-1], 1.0)
    # Кромки обрезаются верхней и нижней кривыми: иначе угол блока уходит вниз (или вверх) за
    # строку, которая на своём конце загнулась (1973/07 с.77, правый нижний угол).
    left_curve = _trim_to_caps(np.column_stack([left, grid]), top_curve, bottom_curve)
    right_curve = _trim_to_caps(np.column_stack([right, grid]), top_curve, bottom_curve)
    polygon = _swallow_rows(np.vstack([left_curve, bottom_curve, right_curve[::-1], top_curve[::-1]]), rows, dpi)
    return BlockEnvelope(
        left=left_curve,
        right=right_curve,
        top=top_curve,
        bottom=bottom_curve,
        polygon=polygon,
        smooth_pitches=float(smooth_pitches),
        core_left=np.column_stack([core_left, grid]),
        core_right=np.column_stack([core_right, grid]),
        polygon_dilated=_dilated(polygon, glyph, dilate, dpi),
        dilate_px=(dilate * glyph[0], dilate * glyph[1]) if glyph else (0.0, 0.0),
    )


def _dilated(polygon: np.ndarray, glyph: tuple[float, float] | None, dilate: float, dpi: float) -> np.ndarray | None:
    """Раздутый контур блока, если известен размер символа."""
    if not glyph or dilate <= 0:
        return None
    return dilate_polygon(polygon, dilate * glyph[0], dilate * glyph[1], dpi)


def _outward(
    trend: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    grid: np.ndarray,
    pitch: float,
    nodes: int,
    dpi: float,
    inward: float,
) -> np.ndarray:
    """Отодвинуть кромку наружу так, чтобы ВСЕ строки блока оказались внутри контура.

    Тренд по телу блока идёт по середине разброса краёв, и строки, выступающие наружу, остаются
    за контуром — пользователь требует обратного: в границы блока должны попадать все его строки
    целиком. Требование каждой строки размазывается на межстрочный интервал (кромка гладкая, и
    достаточно, чтобы она была снаружи на высоте самой строки), кромка прижимается к нему,
    сглаживается и прижимается ещё раз — последний проход и даёт гарантию.

    Args:
        trend: Кромка-тренд в узлах сетки.
        ys, xs: Ординаты рядов и их края с этой стороны.
        grid: Узлы сетки по y.
        pitch: Межстрочный шаг (px): на него размазывается требование одной строки.
        nodes: Окно сглаживания в узлах сетки.
        dpi: Разрешение рабочей копии.
        inward: ``+1`` для левой кромки, ``-1`` для правой.

    Returns:
        Кромка, снаружи которой не остаётся ни одного края ряда.
    """
    margin = inward * mm_to_px(OUTWARD_MARGIN_MM, dpi)
    half = max(pitch / 2.0, 1.0)
    need = np.empty(grid.shape, dtype=np.float64)
    for index, y in enumerate(grid):
        near = np.abs(ys - y) <= half
        own = xs[near] if near.any() else xs[np.argmin(np.abs(ys - y))][None]
        need[index] = (own.min() if inward > 0 else own.max()) - margin
    out = trend
    for _ in range(OUTWARD_PASSES):
        out = np.minimum(out, need) if inward > 0 else np.maximum(out, need)
        out = _smooth(out, nodes)
    return np.minimum(out, need) if inward > 0 else np.maximum(out, need)


def _trim_to_caps(curve: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Обрезать боковую кромку верхней и нижней кривыми блока.

    Верх и низ идут по осям крайних строк, а сетка боковых кромок — по прямоугольнику от самой
    высокой до самой низкой точки. На наклонной или загнутой строке угол контура из-за этого
    уходил за текст. Здесь узлы, вышедшие за кромку-крышку, подтягиваются к ней.

    Args:
        curve: Боковая кромка ``(N, 2)`` сверху вниз.
        top: Верхняя кривая блока ``(M, 2)``.
        bottom: Нижняя кривая блока.

    Returns:
        Та же кромка, зажатая между верхней и нижней кривыми.
    """
    out = curve.copy()
    top_y = np.interp(out[:, 0], top[:, 0], top[:, 1])
    bottom_y = np.interp(out[:, 0], bottom[:, 0], bottom[:, 1])
    out[:, 1] = np.clip(out[:, 1], np.minimum(top_y, bottom_y), np.maximum(top_y, bottom_y))
    return out


def _side_trend(
    ys: np.ndarray, xs: np.ndarray, grid: np.ndarray, window: float, dpi: float, inward: float
) -> np.ndarray:
    """Кромка блока: грубый квантиль → отбор рядов тела → локальная прямая по ним.

    Args:
        ys, xs: Ординаты рядов и их края с этой стороны.
        grid: Узлы сетки по y, на которых считается кромка.
        window: Окно сглаживания в пикселях.
        dpi: Разрешение рабочей копии (для перевода допуска из мм).
        inward: Куда от кромки смотрит внутренность блока: ``+1`` для левой стороны, ``-1`` для правой.

    Returns:
        Значения кромки в узлах сетки.
    """
    quantile = EDGE_QUANTILE if inward > 0 else 1.0 - EDGE_QUANTILE
    trend = _quantile_trend(ys, xs, grid, max(window * 2.0, window), quantile)
    trim = mm_to_px(TRIM_MM, dpi)
    keep = np.ones(ys.shape, dtype=bool)
    # Два прохода: по грубому тренду отбрасываются ряды, ушедшие внутрь блока (абзацный отступ,
    # короткий конец абзаца), затем кромка пересчитывается по оставшимся и отбор повторяется.
    for _ in range(TREND_PASSES):
        fitted = np.interp(ys, grid, trend)
        inside = inward * (xs - fitted) > trim
        keep = ~inside
        if keep.sum() < MIN_TREND_ROWS:
            keep = np.ones(ys.shape, dtype=bool)
        trend = _local_line(ys[keep], xs[keep], grid, window)
    # Кромка не должна уходить за пределы самих краёв рядов: на краях сетки локальная прямая
    # экстраполирует, и у коротких блоков огибающая улетала на десятки миллиметров.
    margin = mm_to_px(TREND_CLIP_MM, dpi)
    return np.clip(trend, xs[keep].min() - margin, xs[keep].max() + margin)


def _cap(row: Row, x_left: float, x_right: float, direction: float) -> np.ndarray:
    """Верхняя (``direction=-1``) или нижняя (``+1``) кромка блока по краске крайнего ряда.

    Берётся профиль краски ряда (:func:`edge_profiles`) — он повторяет изгиб строки и заведомо
    лежит снаружи всех её символов. Если профиля нет (ряд пришёл не из ``rows_of``), кромка
    строится по осям ряда со сдвигом на полвысоты, как раньше.

    За пределами краски кромка продлевается по наклону крайнего участка, но не дальше
    ``END_EXTEND_MM`` по x и не больше ``END_EXTEND_MAX_MM`` по y: короткая строка (колонтитул,
    обрывок) иначе уводит кромку через всю страницу.

    Args:
        row: Крайний ряд блока.
        x_left, x_right: Куда протянуть кромку по x.
        direction: ``-1`` — верх, ``+1`` — низ.

    Returns:
        Кривая ``(N, 2)`` слева направо.
    """
    profile = row.top_edge if direction < 0 else row.bottom_edge
    if profile is None or len(profile) < 2:
        points = np.vstack([np.asarray(axis.points, dtype=np.float64) for axis in row.axes])
        points = points[np.argsort(points[:, 0])]
        profile = np.column_stack([points[:, 0], points[:, 1] + direction * row.height / 2.0])
    dpi = row.axis.dpi
    margin = direction * mm_to_px(OUTWARD_MARGIN_MM, dpi)
    xs = np.linspace(x_left, x_right, num=max(2, int(abs(x_right - x_left) / 4) + 2))
    ys = np.interp(xs, profile[:, 0], profile[:, 1])
    ys = _extend_ends(xs, ys, profile, dpi)
    return np.column_stack([xs, ys + margin])


def _extend_ends(xs: np.ndarray, ys: np.ndarray, profile: np.ndarray, dpi: float) -> np.ndarray:
    """Продлить кромку за концы краски по наклону её крайнего участка, с ограничением.

    ``np.interp`` за концами отдаёт концевое значение, и кромка идёт горизонталью там, где строка
    продолжает загибаться. Но и продлевать без меры нельзя: у короткой строки (колонтитул
    «Нам пишут», 1975/05 с.97) наклон, продлённый через всю полосу, уводит кромку на сантиметры.
    Поэтому продление живёт ``END_EXTEND_MM`` миллиметров и ограничено ``END_EXTEND_MAX_MM``.

    Args:
        xs: Сетка по x, на которой построена кромка.
        ys: Значения кромки на этой сетке (интерполяция профиля).
        profile: Профиль краски ряда ``(M, 2)``, отсортированный по x.
        dpi: Разрешение рабочей копии.

    Returns:
        Те же значения с ограниченным продлением за концами профиля.
    """
    if profile.shape[0] < 3:
        return ys
    out = ys.copy()
    take = max(3, int(profile.shape[0] * END_SLOPE_SHARE))
    reach = mm_to_px(END_EXTEND_MM, dpi)
    limit = mm_to_px(END_EXTEND_MAX_MM, dpi)
    for left_side, mask in ((True, xs < profile[0, 0]), (False, xs > profile[-1, 0])):
        if not mask.any():
            continue
        own = profile[:take] if left_side else profile[-take:]
        slope = float(np.polyfit(own[:, 0], own[:, 1], 1)[0])
        anchor = profile[0] if left_side else profile[-1]
        distance = np.clip(xs[mask] - anchor[0], -reach, reach)
        out[mask] = anchor[1] + np.clip(slope * distance, -limit, limit)
    return out


def blocks_of(
    axes: list[LineAxis],
    zones: list,
    gutters: list,
    width: int,
    ink: np.ndarray,
    rules: list | None = None,
    dpi: float = WORK_DPI,
    smooth_pitches: float = SMOOTH_PITCHES,
    coarse_factor: float = COARSE_FACTOR,
    dilate: float = DILATE_GLYPHS,
    leaders: list | None = None,
) -> list[TextBlock]:
    """Блоки страницы: колонки зон вёрстки, склеенные по вертикали и разделённые разрывами.

    Зона вёрстки кончается там, где появляется или пропадает межколонник (врезка сверху справа —
    1967/10 с.63), но колонка основного текста при этом продолжается: соседние по вертикали
    куски одной колонки склеиваются обратно, если их границы совпадают (перекрытие не меньше
    ``MERGE_OVERLAP``) и разрыв между ними меньше ``BLOCK_GAP_PITCHES`` шагов строк.

    Args:
        axes: Оси всех строк страницы.
        zones: Зоны вёрстки (``columns.Zone``).
        gutters: Локальные межколонники-ломаные (``columns.Gutter``).
        width: Ширина рабочей копии.
        ink: Краска текста рендера ``RENDER_DPI``.
        rules: Сплошные черты страницы (``segment.Rule``): делят блок (сноска под линейкой).
        dpi: Разрешение рабочей копии.
        smooth_pitches: Окно основной огибающей в межстрочных интервалах.
        coarse_factor: Во сколько раз шире окно крупной огибающей.

    Returns:
        Блоки, слева направо и сверху вниз.
    """
    pad = mm_to_px(COLUMN_PAD_MM, dpi)
    pieces: list[tuple[tuple[int, int], list[Row]]] = []
    placed: set[tuple[int, int, int]] = set()
    for zone in zones:
        own_zone = [axis for axis in axes if zone.y0 <= axis.cy < zone.y1]
        for column, span in enumerate(zone.columns):
            own = [
                with_column(axis, column) for axis in own_zone if not axis.cross and _inside(axis, span, gutters, width)
            ]
            if not own:
                continue
            rows = rows_of(own, ink, span, dpi, gutters=gutters, width=width, pad=pad, leaders=leaders)
            # Пристроенной считается ось, реально вошедшая в РЯД, а не всякая, попавшая в колонку:
            # ``rows_of`` выбрасывает ряд, вылезающий за колонку дальше ``OVERHANG_MM``, и раньше
            # такие оси пропадали совсем. На 1971/10 с.93 так терялись одиннадцать строк правой
            # части: межколонник там сужается, текст начинается с x 641, а колонка зоны
            # начинается с 723. Теперь они попадают в кусок во всю ширину, как и всё, чему
            # колонки не нашлось.
            placed.update(_axis_key(axis) for row in rows for axis in row.axes)
            if rows:
                pieces.append((span, rows))
    # Строки, которым колонки не нашлось (набраны через межколонник, шире любой колонки —
    # колонтитул, заголовок во всю ширину), собираются в блоки ПО ВСЕЙ СТРАНИЦЕ, а не по зонам:
    # заголовок и его подзаголовок часто попадают в соседние зоны и иначе разъезжаются.
    rest = [axis for axis in axes if _axis_key(axis) not in placed]
    if rest:
        rows = rows_of(
            rest, ink, (0, width), dpi, gutters=gutters, width=width, pad=pad, leaders=leaders, fallback=True
        )
        if rows:
            pieces.append(((0, width), rows))
    pieces = _absorb_continuation(pieces, width, dpi)
    merged = _merge_pieces(pieces, dpi)
    widths = [span[1] - span[0] for span, _ in merged]
    typical = float(np.median(widths)) if widths else 0.0
    # Медианная высота ряда по всей странице — мера основного кегля: по ней крупный набор
    # (заголовки, логотипы) отличается от корпуса.
    all_heights = [row.height for _, rows in merged for row in rows]
    body_height = float(np.median(all_heights)) if all_heights else 0.0
    blocks: list[TextBlock] = []
    # Ряд, вылезающий за колонку своего куска, в неё не идёт: иначе широкий заголовок таблицы
    # попадает СРАЗУ В ДВА блока соседних колонок и их контуры пересекаются (1971/10 с.93). Такие
    # ряды собираются в отдельный кусок во всю ширину — терять их нельзя.
    merged, spilled = _spilled_rows(merged, dpi)
    if spilled:
        # Ряды-«проливы» кладутся в ТОТ ЖЕ кусок во всю ширину, что и строки, которым колонки не
        # нашлось: иначе куски одной широкой строки оказываются в разных кусках и никогда не
        # сравниваются между собой. На 1971/10 с.93 так делился заголовок «Минимальные нормы
        # заказа…»: его левая часть уходила в проливы, середина и правая — в запасной кусок, и
        # получались два блока на одну строку.
        full = next((index for index, (span, _) in enumerate(merged) if span == (0, width)), None)
        if full is None:
            merged.append(((0, width), spilled))
        else:
            span, rows = merged[full]
            merged[full] = (span, sorted([*rows, *spilled], key=lambda row: row.y))
    for column, (span, rows) in enumerate(merged):
        if len(rows) < MIN_BLOCK_ROWS:
            continue
        # Обрывок во всю ширину страницы из пары строк выбрасывается, только если его ряды
        # СЛИПЛИСЬ из нескольких строк соседних колонок (в ряду больше одной оси). Настоящий
        # заголовок над двумя колонками — одна ось на всю ширину (1975/05 с.97, «Актуальный
        # вопрос»), и он остаётся блоком.
        wide = typical and span[1] - span[0] > WIDE_BLOCK_RATIO * typical and len(rows) < WIDE_BLOCK_ROWS
        if wide and np.mean([len(row.axes) > 1 for row in rows]) >= GLUED_ROWS_SHARE:
            continue
        for number, group in enumerate(split_blocks(rows, pitch_of(rows), rules, span, body_height)):
            if len(group) < MIN_BLOCK_ROWS:
                continue
            pitch = pitch_of(group)
            widths = [row.glyph_w for row in group if row.glyph_w > 0]
            heights = [row.glyph_h for row in group if row.glyph_h > 0]
            if widths:
                glyph = (float(np.median(widths)), float(np.median(heights)))
            else:
                own_height = float(np.median([row.height for row in group]))
                glyph = (own_height * FALLBACK_ASPECT, own_height * FALLBACK_HEIGHT)
            blocks.append(
                TextBlock(
                    column=column,
                    index=number,
                    span=span,
                    rows=tuple(group),
                    pitch_px=pitch,
                    dpi=float(dpi),
                    envelope=envelope_of(group, pitch, dpi, smooth_pitches, glyph, dilate),
                    envelope_coarse=envelope_of(group, pitch, dpi, smooth_pitches * coarse_factor),
                )
            )
    return blocks


def _absorb_continuation(
    pieces: list[tuple[tuple[int, int], list[Row]]], width: int, dpi: float
) -> list[tuple[tuple[int, int], list[Row]]]:
    """Вернуть широкому блоку его же последние строки, уместившиеся в одну колонку.

    Вводка во всю ширину (1970/02 с.90: «В редакцию часто приходят письма…») кончается короткой
    строкой, которая целиком помещается в левую колонку, — и уходила в неё отдельным блоком из
    одной строки, а вводка теряла свой конец. Строка забирается обратно, если она идёт сразу под
    широким блоком с ЕГО шагом и того же кегля; первая строка колонки под заголовком так не
    уходит: до неё разрыв в несколько шагов.

    Args:
        pieces: Куски блоков ``(границы колонки, ряды)`` до склейки по зонам.
        width: Ширина рабочей копии.
        dpi: Разрешение рабочей копии.

    Returns:
        Те же куски; пустые (всё забрали) выброшены.
    """
    out = [(span, list(rows)) for span, rows in pieces]
    for index, (span, rows) in enumerate(out):
        if span[1] - span[0] < CONTINUATION_WIDE_SHARE * width or len(rows) < 2:
            continue
        pitch = pitch_of(rows)
        for other_index, (other_span, other_rows) in enumerate(out):
            if other_index == index or other_span[1] - other_span[0] >= CONTINUATION_WIDE_SHARE * width:
                continue
            taken = 0
            while taken < min(len(other_rows), CONTINUATION_MAX_ROWS):
                row, tail = other_rows[taken], out[index][1][-1]
                gap = row_gap(tail, row)
                if not 0 < gap <= CONTINUATION_GAP_PITCHES * pitch:
                    break
                if max(row.height, tail.height) > CONTINUATION_HEIGHT_RATIO * min(row.height, tail.height):
                    break
                # Левый край должен совпасть с краем широкого блока: у колонки под заголовком
                # он другой (заголовок набран по центру), и её строки так не уводятся.
                if abs(row.x0 - tail.x0) > mm_to_px(CONTINUATION_EDGE_MM, dpi):
                    break
                out[index][1].append(row)
                taken += 1
            if taken:
                out[other_index] = (other_span, other_rows[taken:])
    return [(span, rows) for span, rows in out if rows]


def _spilled_rows(
    pieces: list[tuple[tuple[int, int], list[Row]]], dpi: float
) -> tuple[list[tuple[tuple[int, int], list[Row]]], list[Row]]:
    """Развести ряды, вылезающие за колонку своего куска, по одному месту.

    Один и тот же ряд попадал сразу в несколько кусков (широкий заголовок таблицы — и в левую
    колонку, и в правую), их контуры пересекались и разбор выглядел хаотично (1971/10 с.93).
    Ряд остаётся в том куске, с которым перекрывается большей частью своей длины; если ни один
    кусок не накрывает его на ``SPILL_KEEP_SHARE``, ряд уходит в отдельный кусок во всю ширину.

    Args:
        pieces: Куски ``(границы колонки, ряды)``.
        dpi: Разрешение рабочей копии.

    Returns:
        Пара ``(куски, ряды для широкого куска)``.
    """
    overhang = mm_to_px(OVERHANG_MM, dpi)
    best: dict[tuple[int, int, int], tuple[float, int]] = {}
    for index, (span, rows) in enumerate(pieces):
        for row in rows:
            key = (round(row.y), round(row.x0), round(row.x1))
            covered = max(0.0, min(row.x1, span[1]) - max(row.x0, span[0])) / max(1.0, row.x1 - row.x0)
            if key not in best or covered > best[key][0]:
                best[key] = (covered, index)
    kept: list[tuple[tuple[int, int], list[Row]]] = []
    spilled: dict[tuple[int, int, int], Row] = {}
    for index, (span, rows) in enumerate(pieces):
        own = []
        for row in rows:
            if row.x0 >= span[0] - overhang and row.x1 <= span[1] + overhang:
                own.append(row)
                continue
            key = (round(row.y), round(row.x0), round(row.x1))
            covered, owner = best[key]
            if owner == index and covered >= SPILL_KEEP_SHARE:
                own.append(row)  # ряд в основном лежит в этой колонке — пусть в ней и остаётся
            elif owner == index:
                spilled.setdefault(key, row)
        kept.append((span, own))
    return kept, sorted(spilled.values(), key=lambda row: row.y)


def _merge_pieces(
    pieces: list[tuple[tuple[int, int], list[Row]]], dpi: float
) -> list[tuple[tuple[int, int], list[Row]]]:
    """Склеить куски одной колонки из соседних зон: тот же левый край и небольшой разрыв по высоте."""
    out: list[tuple[tuple[int, int], list[Row]]] = []
    for span, rows in sorted(pieces, key=lambda item: (item[0][0], item[1][0].y)):
        joined = False
        for index, (other_span, other_rows) in enumerate(out):
            overlap = min(span[1], other_span[1]) - max(span[0], other_span[0])
            # Одна колонка: левые края совпали (по ним равняется набор) и куски заметно
            # перекрываются по ширине. У верхнего куска колонки правый край бывает шире — там,
            # где сбоку ещё нет врезки (1967/10 с.63), и требовать совпадения обоих краёв нельзя.
            same_left = abs(span[0] - other_span[0]) <= mm_to_px(MERGE_EDGE_MM, dpi)
            widths = (span[1] - span[0], other_span[1] - other_span[0])
            # Ширины не должны различаться больше чем в MERGE_WIDTH_RATIO раз: иначе к колонке
            # приклеивается колонтитул во всю ширину страницы (1973/07 с.77 — кромка блока
            # уходила через всю страницу к номеру полосы).
            same_width = max(widths) <= MERGE_WIDTH_RATIO * min(widths)
            same = same_left and same_width and overlap >= MERGE_OVERLAP * min(widths)
            pitch = pitch_of(other_rows)
            gap = row_gap(other_rows[-1], rows[0])
            # Порог здесь щедрый (``MERGE_GAP_PITCHES``): куски одной колонки из соседних зон
            # бывают разделены заголовком или вводкой, а делить обратно — дело ``split_blocks``,
            # которое видит уже всю колонку и режет её по разрыву, кеглю и линейке.
            if same and 0 < gap <= MERGE_GAP_PITCHES * pitch:
                out[index] = ((min(span[0], other_span[0]), max(span[1], other_span[1])), other_rows + rows)
                joined = True
                break
        if not joined:
            out.append((span, list(rows)))
    return sorted(out, key=lambda item: (item[0][0], item[1][0].y))


def _fits_any(axis: LineAxis, zones: list, gutters: list, width: int) -> bool:
    """Есть ли зона и колонка в ней, в которую ось помещается целиком."""
    for zone in zones:
        if not (zone.y0 <= axis.cy < zone.y1):
            continue
        if any(_inside(axis, span, gutters, width) for span in zone.columns):
            return True
    return False


def _axis_key(axis: LineAxis) -> tuple[int, int, int]:
    """Ключ оси для сравнения между копиями: округлённые середина и концы."""
    return (round(axis.cy), round(axis.x0), round(axis.x1))


def _inside(axis: LineAxis, span: tuple[int, int], gutters: list, width: int) -> bool:
    """Ось лежит в колонке зоны — по границам колонки НА ВЫСОТЕ ЭТОЙ ОСИ.

    Границы берутся у межколонников-ломаных (:func:`columns.bounds_at`), а не у прямоугольника
    зоны: на трапеции колонка уезжает вбок, и прямая граница либо режет её, либо цепляет чужое
    (номер полосы, уехавший под блок).
    """
    left, right = bounds_at(gutters, axis.x0, axis.x1, axis.cy, width)
    # Колонка зоны и локальные границы должны быть той же колонкой: сравниваем по перекрытию.
    if min(right, span[1]) - max(left, span[0]) < 0.5 * (span[1] - span[0]):
        return False
    # Строка принадлежит колонке, если лежит в ней ОСНОВНОЙ своей частью: требовать «целиком
    # внутри» нельзя — у последней строки абзаца выносной элемент вылезает за кромку, и строка
    # уходила из колонки в блок-остаток (1970/02 с.90, «справочниках»).
    pad = 0.05 * (right - left)
    inside = min(axis.x1, right + pad) - max(axis.x0, left - pad)
    return inside >= INSIDE_SHARE * max(1.0, axis.x1 - axis.x0)


__all__ = [
    "row_gap",
    "BlockEnvelope",
    "Row",
    "TextBlock",
    "blocks_of",
    "envelope_of",
    "ink_edge",
    "pitch_of",
    "rows_of",
]
