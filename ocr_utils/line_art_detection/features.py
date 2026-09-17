"""Признаки связного пятна краски и правило «это крупный штриховой рисунок».

ЗАЧЕМ. FineReader с распрямлением строк иногда корёжит line art. Чтобы не просматривать
глазами девять тысяч страниц пака, нужен признак «на этой странице есть сколько-нибудь
КРУПНЫЙ штрих» — рисунок, гравюра, схема, карта, разлинованная таблица, график с осями.
Мелочь (виньетки, эмблемы рубрик, буквицы) не ловим: она и покорёженная никому не мешает.

НА ЧЁМ ДЕРЖИТСЯ РАЗДЕЛЕНИЕ. Текст распадается на пятна размером с букву, штриховой
рисунок даёт ОДНО длинное связное пятно на весь рисунок. Замер по эталону
``full_1967_01_bg_off_ori_off.pdf`` (площадь крупнейшего пятна страницы):

    страница 11, чистый текст            5021
    страница 68, плотный текст           2027
    страница 80, штриховой рисунок     412267

Разница в два порядка. То же самое замерено в ``scan_markup.detection.dots`` с другой
стороны и на другом материале: у полутоновых фотографий p99 площади пятна 100..4439,
у штриха 4761..550783. Отсюда и порог ``MIN_AREA_PX``.

КРАСКА БЕРЁТСЯ ГЛОБАЛЬНЫМ ПОРОГОМ, А НЕ ``adaptiveThreshold``. Соблазн взять готовый
``dots.ink_components`` велик и неверен: он строился под КАМЕРНЫЕ сканы, где яркость
бумаги плывёт по кадру, и берёт краску локальным порогом ``T = mean - C`` при ``C = -5``,
то есть ``mean + 5``. На уже битональной странице PDF в окне чистой бумаги ``mean = 255``,
порог 260, и вся бумага уезжает в краску. Замер долей краски на эталоне:

    страница                 g < 128      adaptiveThreshold(31, -5)
    11, текст                  0.096                          0.772
    68, текст плотный          0.146                          0.650
    80, штриховой рисунок      0.050                          0.875

Страница бинаризованного PDF имеет ровно две градации яркости при рендере 1:1, и никакой
локальный порог ей не нужен.

ЧЕГО ОСТЕРЕГАЕМСЯ. Три вещи выглядят как крупное пятно, но штрихом не являются:

* СПЛОШНАЯ ЧЁРНАЯ КЛЯКСА (край скана, тень разворота). Отличается заполнением рамки:
  у кляксы ``fill >= BLOB_FILL``, у штриха рамка почти пустая. Пятно, которое вдобавок
  касается края кадра, отбраковывается по более мягкому порогу — приём
  ``pixFindPageForeground`` из Leptonica.
* ПОЛУТОНОВАЯ ФОТОГРАФИЯ. В бинаризации она рассыпается в растровую сетку; в тенях точки
  смыкаются, и получается крупное пятно, изрешечённое ДЫРАМИ. Дыры и есть признак: у
  растра их сотни и все крошечные (просветы между точками), у рисунка они крупные
  (замкнутые области самого рисунка). Замер на ``full_1966_01.pdf`` стр. 78: крупнейшее
  пятно имеет 1640 дыр медианной площадью 58 px при 600 dpi.
* ТЕКСТ, СЛИПШИЙСЯ В СТРОКУ. Отсекается требованием ``max(w, h) >= MIN_SIDE_PX``:
  слипшаяся строка длинна, но её высота — одна буква, а площадь мала.

ЕДИНИЦЫ. Все размеры названы для 600 dpi и пересчитываются под другое разрешение
единственной функцией :func:`params_for_dpi` — чтобы точке пересчёта негде было
разъехаться (та же схема, что у ``dots.ScreenParams``).
"""

from dataclasses import dataclass, replace

import cv2
import numpy as np

from ocr_utils.scan_markup.detection.boxes import merge_boxes

# Разрешение, для которого названы все размеры ниже.
REFERENCE_DPI = 600

# Порог краски. Страница битональная, так что годится любое число между 0 и 255;
# 128 взято как середина.
INK_THRESHOLD = 128

# Шаг текстовых строк и толщина штриха пака-1 при 600 dpi (замер — ``ocr_utils.paper``).
# Через них выражены зазор слияния и длина «длинного прогона».
PITCH_PX = 89
STROKE_PX = 7

# Площадь, с которой пятно вообще рассматривается как штрих. Замер ``dots``: у штриха
# площади начинаются с 4761, у полутоновых фотографий кончаются на 4439.
MIN_AREA_PX = 4761

# Длинная сторона рамки пятна. Два шага строк: буква не бывает такой, а слипшаяся в одно
# пятно строка текста не проходит по площади.
MIN_SIDE_PX = 2 * PITCH_PX

# Заполнение рамки краской. Ниже нижнего — рамка пустая, это разрозненная пыль, попавшая
# в одно пятно по диагонали; выше верхнего — сплошная заливка, а не штрих.
MIN_FILL = 0.02
MAX_FILL = 0.60

# Сплошная клякса: рамка залита почти целиком и дыр в ней нет.
BLOB_FILL = 0.70
BLOB_MAX_HOLES = 2

# Пятно, касающееся края кадра, отбраковывается по более мягкому порогу заполнения:
# у края живут тень разворота и чёрная кайма скана, а рисунок в обрез — редкость.
BORDER_FILL = 0.50

# --- ЧЁРНАЯ КАЙМА СКАНА И ПРОЧАЯ СПЛОШНАЯ МАССА ------------------------------------
# Неосвещённое поле вокруг полосы при камерной съёмке бинаризуется в чёрную кайму по
# периметру скана. Рамка у неё во всю полосу, заполнение низкое, вытянутость около
# единицы — все ворота она проходит и выглядит гигантской находкой. На паке таких
# страниц сотни, и правило края их не берёт: промежуточный PDF собран с БЕЛЫМИ ПОЛЯМИ,
# и до края СТРАНИЦЫ кайма не достаёт (замер: начинается ровно с 288 px — ширины поля).
#
# РАЗДЕЛЯЕТ ИХ ПЛОТНОСТЬ. Приём взят у Leptonica: её ``pixGenerateHalftoneMask`` (та
# самая, которую зовёт Tesseract) — это детектор не картинок, а СПЛОШНОЙ МАССЫ; он
# оставляет только то, что переживает ранговое сжатие, то есть заведомо не видит тонкого
# штриха. Как детектор иллюстраций он нам бесполезен, а как ВЫЧИТАТЕЛЬ — ровно то, что
# нужно. У нас это одна эрозия диском в два штриха: штрих толщиной ``stroke_px`` исчезает
# целиком, сплошная клякса остаётся почти вся. Замер:
#
#     что это                                          плотность
#     кайма скана, 1967/07 стр.2 (титул, кайма толстая)  0.758, 0.860, 0.736
#     кайма скана, 1970/01 стр.19 (два куска)            0.644, 0.668
#     ------------------------------------------------------------
#     чертёж шкафа, 1966/01 стр.78                       0.000, 0.037
#     схема, эталон стр.80                               0.005, 0.000
#     чертёж крана, 1970/01 стр.19                       0.000
#     выворотная литера заголовка, 1967/07 стр.2         0.002
#
# Промежуток 0.037..0.644 огромен; порог поставлен ближе к штриху, чтобы график со
# сплошными столбиками или заливкой не уехал в кляксы.
#
# Пробовали и отвергли: «доля краски в поясе вдоль своей рамки» (кайма — кольцо, рисунок
# заполняет рамку внутри). На тонкой кайме это работает (0.94..1.00 против 0.15..0.59),
# но толстая кайма титульной полосы 1967/07 даёт всего 0.77 и правило обходит, а
# поднимать порог нельзя — под него уходит чертёж шкафа с его 0.59.
SOLID_ERODE_STROKES = 2
MAX_SOLID_FRAC = 0.35

# Дыры мельче этого не считаем вовсе: это ступеньки контура, а не просветы.
MIN_HOLE_AREA_PX = 4

# ГЛАВНЫЙ ОТСЕКАТЕЛЬ — ВЫТЯНУТОСТЬ РАМКИ. Рисунок двумерен, а типографская линейка,
# строка отточий в оглавлении, вертикальная линейка таблицы и чёрный потёк вдоль края
# скана — одномерны. Замер по паку (пятна, прошедшие ворота по площади):
#
#     что это                                              вытянутость  minside
#     схема, эталон стр.80                                         2.3     2170
#     чертёж шкафа, 1966/01 стр.78                                 1.1     2699
#     врезки в рамке, эталон стр.80                            5.2..7.2  116..332
#     ------------------------------------------------------------------------
#     линейка над колонтитулом, 1966/01 стр.76                   146.1       22
#     строка отточий в оглавлении, 1966/01 стр.95         65.6..119.6    16..17
#     вертикальные линейки таблицы, 1966/01 стр.28          32.6..53.2    27..51
#     чёрный потёк вдоль края, 1966/01 стр.95              61.5..166.4    34..93
#     сторона рекламной рамки, 1966/01 стр.97               43.1..69.8    36..44
#
# Промежуток между 10.4 (самая вытянутая настоящая находка) и 32.6 (самая компактная
# линейка) пуст, порог ставится посреди него.
MAX_ASPECT = 15.0

# Короткая сторона рамки. Полшага строк: рисунок ниже полустроки не бывает, а линейка,
# отточия и колонтитульная черта — это 15-25 px при 600 dpi (см. столбец minside выше).
MIN_MINOR_PX = PITCH_PX // 2

# --- Признаки растра: СЧИТАЮТСЯ ВСЕГДА, ОТБРАКОВЫВАЮТ ТОЛЬКО ПО ЯВНОЙ ПРОСЬБЕ --------
# Литература (Wang-Phillips-Haralick PR 2006) обещает, что полутоновую печать от штриха
# отделяют размер дыр (у растра это просветы между точками, то есть крошечные) и доля
# краски в длинных прогонах. НА ЭТОМ МАТЕРИАЛЕ НИ ТО НИ ДРУГОЕ НЕ РАБОТАЕТ, замерено:
#
#     пятно                                   дыр  медДыра  длПрогон
#     схема, эталон стр.80                    122       12      0.97
#     чертёж шкафа, 1966/01 стр.78           1640       58      0.97
#     рекламная рамка, 1966/01 стр.97           0        0      0.40
#
# У настоящего штриха дыры мельче, чем у якобы растра: у гравюры и чертежа это ячейки
# штриховки и полки шкафа. Доля длинных прогонов у всего подряд лежит в 0.86..0.99.
# Включённое отсечение по дырам стоило нам ровно той страницы, ради которой всё
# затевалось: схема на стр.80 эталона уезжала в «растр», и покрытие падало с 0.43 до
# 0.045 (ранг 1 -> 8 из 97).
#
# Поэтому полутоновые фотографии отсеиваются НЕ ПИКСЕЛЯМИ, А РУЧНОЙ РАЗМЕТКОЙ из базы
# (``--db``, см. ``markup.py``) — она для пака-1 уже выверена глазами. Признаки же
# по-прежнему считаются и пишутся в CSV: на другом паке разделение может и найтись,
# и тогда пороги ниже включаются ключом ``--reject-halftone``.
HALFTONE_MIN_HOLES = 50
HALFTONE_HOLE_AREA_PX = (2 * STROKE_PX) ** 2

# Длина «длинного прогона» краски. Три толщины штриха.
LONG_RUN_PX = 3 * STROKE_PX
MIN_LONG_RUN_FRAC = 0.15

# --- Разлинованная таблица: КЛАСТЕР ЛИНЕЕК, а не одно пятно ------------------------
# Сетка таблицы почти никогда не является одним связным пятном: линейки в этой печати
# друг друга не касаются. Замер по эталону стр.47 (крупная таблица во всю ширину набора):
# пятен, проходящих ворота по размеру и вытянутости, — НОЛЬ. Поэтому таблица ищется
# отдельно и не по пятну, а по СКОПЛЕНИЮ линеек — это же делает ``pixDecideIfTable`` в
# Leptonica, считая горизонтальные и вертикальные линии.
#
# Линейкой считается ровно то, что главный детектор выбросил как одномерное: длинное,
# тонкое и очень вытянутое.
#
# СЧИТАЮТСЯ ТОЛЬКО ВНУТРЕННИЕ ЛИНЕЙКИ — те, что не прижаты к краю скопления. Иначе
# таблицей оказывается любая ДЕКОРАТИВНАЯ РАМКА: у полосы выходных данных есть рамка
# вокруг набора и одна линейка между колонками, и внешне это те же «две горизонтали и
# две вертикали». Замер:
#
# И СЧИТАЮТСЯ НЕ ЛИНЕЙКИ, А ИХ РАЗЛИЧНЫЕ ПОЗИЦИИ. Разделитель колонок на полосе
# содержания перебит пересекающей горизонталью и распадается на два-четыре куска — по
# штукам это уже «несколько вертикалей», хотя стоят они все на одном и том же x. Сетку
# делает не число линеек, а число РАЗНЫХ колонок и строк. Замер (внутренних линеек и
# различных позиций среди них):
#
#     что это                             внутрГ (позиций)   внутрВ (позиций)
#     содержание, 1972/01 стр.3                2 (2)              2 (1)
#     содержание, 1975/05 стр.3                2 (2)              4 (1)
#     рамка выходных данных, 1974/04 стр.4     2 (2)              1 (1)
#     -----------------------------------------------------------------------
#     таблица, эталон стр.47                   5 (4)             14 (5)
#     таблицы, 1966/01 стр.28                  5 (3)              5 (2)
#                                              2 (2)              3 (3)
#     таблицы, 1966/04 стр.56                  5 (4)              8 (3)
#                                              3 (3)             10 (5)
#
# По РАЗЛИЧНЫМ внутренним вертикалям разделение полное: у рамки и содержания она всегда
# одна, у таблицы их две и больше. Требуются линейки ОБЕИХ ориентаций — это отделяет
# таблицу от полосы содержания и в том случае, когда вертикалей нет вовсе: строка отточий
# даёт сколько угодно горизонталей (замер: 1966/01 стр.95).
RULE_MAX_MINOR_PX = PITCH_PX // 2
TABLE_MIN_RULES = 2
# Насколько линейка должна отступать от края скопления, чтобы считаться внутренней.
TABLE_EDGE_TOLERANCE_PX = PITCH_PX

# Зазор слияния рамок в одну находку — шаг строк: разорванный на части рисунок должен
# собраться обратно, а соседняя колонка текста — не приклеиться.
MERGE_GAP_PX = PITCH_PX

# Минимальная доля площади полосы для ОДНОЙ находки. Мельче — виньетка или эмблема
# рубрики; заказчик просил такие не ловить.
MIN_REGION_FRAC = 0.002

# Доля площади полосы, с которой находка считается полосной иллюстрацией (обложка,
# вклейка). Такие идут в отчёт отдельной группой: их и так видно.
FULL_PAGE_FRAC = 0.75


@dataclass(frozen=True)
class LineArtParams:
    """Пороги детектора при КОНКРЕТНОМ разрешении страницы.

    Собирается только через :func:`params_for_dpi`.
    """

    dpi: int = REFERENCE_DPI
    ink_threshold: int = INK_THRESHOLD
    pitch_px: int = PITCH_PX
    stroke_px: int = STROKE_PX
    min_area_px: int = MIN_AREA_PX
    min_side_px: int = MIN_SIDE_PX
    min_minor_px: int = MIN_MINOR_PX
    max_aspect: float = MAX_ASPECT
    rule_max_minor_px: int = RULE_MAX_MINOR_PX
    table_min_rules: int = TABLE_MIN_RULES
    table_edge_tolerance_px: int = TABLE_EDGE_TOLERANCE_PX
    detect_tables: bool = True
    reject_halftone: bool = False
    min_fill: float = MIN_FILL
    max_fill: float = MAX_FILL
    blob_fill: float = BLOB_FILL
    blob_max_holes: int = BLOB_MAX_HOLES
    border_fill: float = BORDER_FILL
    solid_erode_strokes: int = SOLID_ERODE_STROKES
    max_solid_frac: float = MAX_SOLID_FRAC
    min_hole_area_px: int = MIN_HOLE_AREA_PX
    halftone_min_holes: int = HALFTONE_MIN_HOLES
    halftone_hole_area_px: int = HALFTONE_HOLE_AREA_PX
    long_run_px: int = LONG_RUN_PX
    min_long_run_frac: float = MIN_LONG_RUN_FRAC
    merge_gap_px: int = MERGE_GAP_PX
    min_region_frac: float = MIN_REGION_FRAC
    full_page_frac: float = FULL_PAGE_FRAC


def params_for_dpi(dpi: int | None, **overrides) -> LineArtParams:
    """Пороги под разрешение ``dpi``; ``overrides`` перекрывают уже пересчитанное.

    Длины масштабируются линейно, площади — квадратично. Доли не масштабируются вовсе.

    Args:
        dpi: Разрешение страницы; None — считать эталонным.
        **overrides: Поля :class:`LineArtParams`, задаваемые явно (из CLI).

    Returns:
        Готовый набор порогов.
    """
    dpi = int(dpi or REFERENCE_DPI)
    k = dpi / REFERENCE_DPI

    def length(value: float) -> int:
        return max(1, int(round(value * k)))

    def area(value: float) -> int:
        return max(1, int(round(value * k * k)))

    params = LineArtParams(
        dpi=dpi,
        pitch_px=length(PITCH_PX),
        stroke_px=length(STROKE_PX),
        min_area_px=area(MIN_AREA_PX),
        min_side_px=length(MIN_SIDE_PX),
        min_minor_px=length(MIN_MINOR_PX),
        rule_max_minor_px=length(RULE_MAX_MINOR_PX),
        table_edge_tolerance_px=length(TABLE_EDGE_TOLERANCE_PX),
        min_hole_area_px=area(MIN_HOLE_AREA_PX),
        halftone_hole_area_px=area(HALFTONE_HOLE_AREA_PX),
        long_run_px=length(LONG_RUN_PX),
        merge_gap_px=length(MERGE_GAP_PX),
    )
    return replace(params, **overrides) if overrides else params


@dataclass
class Candidate:
    """Одно пятно, признанное штрихом, вместе с признаками, по которым его признали."""

    box: tuple[int, int, int, int]
    area: int
    fill: float
    holes: int
    hole_median_px: float
    long_run_frac: float
    compactness: float
    source: str = "ink"


@dataclass
class PageFindings:
    """Итог по одной странице."""

    boxes: list[tuple[int, int, int, int]]
    coverage: float
    candidates: list[Candidate]
    max_cc_area: int
    n_components: int
    dropped: dict[str, int]
    full_page: bool


def ink_mask(gray: np.ndarray, params: LineArtParams) -> np.ndarray:
    """Маска краски (uint8 0/1) глобальным порогом — см. докстринг модуля."""
    return (gray < params.ink_threshold).astype(np.uint8)


def long_run_fraction(mask: np.ndarray, run_px: int) -> float:
    """Доля краски, лежащей в прогонах длиннее ``run_px`` хоть по одному направлению.

    Считается размыканием линейными структурными элементами: пиксель переживает
    размыкание элементом длины L тогда и только тогда, когда через него проходит
    целиком укладывающийся в краску отрезок этой длины — то есть ровно «пиксель лежит
    в прогоне не короче L».

    Направлений четыре, а не два. Штрих под 45 градусов даёт горизонтальные прогоны
    длиной всего в полторы толщины, и по двум осям такой рисунок выглядел бы россыпью
    точек, то есть растром.

    Args:
        mask: Битовая маска пятна (uint8 0/1).
        run_px: Минимальная длина прогона.

    Returns:
        Доля от 0 до 1; 0, если краски нет вовсе.
    """
    total = int(np.count_nonzero(mask))
    if total == 0:
        return 0.0

    horizontal = np.ones((1, run_px), np.uint8)
    vertical = np.ones((run_px, 1), np.uint8)
    diagonal = np.eye(run_px, dtype=np.uint8)
    kernels = (horizontal, vertical, diagonal, diagonal[::-1].copy())

    survived = np.zeros_like(mask)
    for kernel in kernels:
        np.bitwise_or(survived, cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), out=survived)
    return float(np.count_nonzero(survived)) / total


def solid_fraction(mask: np.ndarray, radius: int) -> float:
    """Доля краски пятна, пережившая эрозию диском радиуса ``radius``.

    Штрих тоньше диаметра диска исчезает целиком, сплошная масса остаётся почти вся.
    Это и отличает чёрную кайму скана от рисунка — см. докстринг ``MAX_SOLID_FRAC``.
    """
    total = int(np.count_nonzero(mask))
    if total == 0:
        return 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return float(np.count_nonzero(cv2.erode(mask, kernel))) / total


def shape_stats(mask: np.ndarray, params: LineArtParams) -> tuple[int, float, float]:
    """``(число дыр, медианная площадь дыры, компактность)`` для маски пятна.

    Дыры — контуры второго уровня при ``RETR_CCOMP``; мельче ``min_hole_area_px`` не
    считаются (это ступеньки контура, а не просветы). Компактность ``P^2 / (4 pi N)``
    равна единице у круга и растёт с изрезанностью: у штриха она в десятки раз больше,
    чем у залитой фигуры.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0, 0.0, 0.0

    hole_areas = [
        cv2.contourArea(contours[i])
        for i, row in enumerate(hierarchy[0])
        if row[3] != -1 and cv2.contourArea(contours[i]) >= params.min_hole_area_px
    ]
    perimeter = sum(cv2.arcLength(contours[i], True) for i, row in enumerate(hierarchy[0]) if row[3] == -1)
    filled = float(np.count_nonzero(mask))
    compactness = perimeter * perimeter / (4.0 * np.pi * filled) if filled else 0.0
    median = float(np.median(hole_areas)) if hole_areas else 0.0
    return len(hole_areas), median, compactness


def _overlaps_any(box: tuple[int, int, int, int], boxes, min_frac: float = 0.5) -> bool:
    """Перекрыт ли ``box`` объединением ``boxes`` больше чем на ``min_frac`` своей площади.

    Считается по сумме попарных пересечений, а не по настоящему объединению: исключающие
    прямоугольники (растровые области полосы) друг друга почти не перекрывают, а завышение
    здесь безопаснее занижения — лишний раз не покажем фотографию.
    """
    area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    covered = 0
    for other in boxes:
        width = min(box[2], other[2]) - max(box[0], other[0])
        height = min(box[3], other[3]) - max(box[1], other[1])
        if width > 0 and height > 0:
            covered += width * height
    return covered >= min_frac * area


def classify(mask: np.ndarray, box, area: int, params: LineArtParams, touches_border: bool, source: str = "ink"):
    """Разбор одного пятна: ``(Candidate, None)`` либо ``(None, причина отбраковки)``.

    ``mask`` — маска пятна В ЕГО РАМКЕ (uint8 0/1), ``box`` — сама рамка в координатах
    страницы, ``area`` — площадь краски в пикселях.
    """
    width, height = box[2] - box[0], box[3] - box[1]
    fill = area / max(1, width * height)

    if solid_fraction(mask, params.solid_erode_strokes * params.stroke_px) >= params.max_solid_frac:
        return None, "сплошная масса (кайма скана, клякса)"
    if touches_border and fill >= params.border_fill:
        return None, "клякса у края"

    holes, hole_median, compactness = shape_stats(mask, params)

    if fill >= params.blob_fill and holes <= params.blob_max_holes:
        return None, "сплошная клякса"

    long_run = long_run_fraction(mask, params.long_run_px)

    # Отсев растра по пикселям — только по явной просьбе: на паке-1 он вреден, см. модуль.
    if params.reject_halftone:
        if holes >= params.halftone_min_holes and hole_median < params.halftone_hole_area_px:
            return None, "растр (много мелких дыр)"
        if long_run < params.min_long_run_frac:
            return None, "растр (нет длинных прогонов)"

    return Candidate(box, area, fill, holes, hole_median, long_run, compactness, source), None


def analyse_gray(gray: np.ndarray, params: LineArtParams, exclude_boxes=(), extra_boxes=()) -> PageFindings:
    """Находит на странице крупный штрих и считает, какую долю полосы он занимает.

    Порядок ровно такой: сначала дешёвые ворота по размеру (их проходят единицы пятен из
    тысяч), и только потом дорогие признаки формы. Считать дыры и прогоны у каждой буквы
    страницы было бы на два порядка дороже и совершенно незачем.

    Args:
        gray: Полутоновый рендер страницы.
        params: Пороги под разрешение этого рендера.
        exclude_boxes: Прямоугольники, которые заведомо не штрих (растровые области из
            базы разметки, блоки ``Picture`` от Surya). Кандидат, перекрытый ими больше
            чем наполовину, выбрасывается.
        extra_boxes: Пары ``(рамка, метка)`` — предложения разметки страницы (таблицы,
            формулы). Связная статистика их не видит: у таблицы без линеек и у формулы
            длинных связных штрихов нет. Пиксельные проверки к ним применяются те же.

    Returns:
        Находки страницы вместе со счётчиком причин отбраковки.
    """
    height, width = gray.shape[:2]
    page_area = float(height * width)
    ink = ink_mask(gray, params)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)

    dropped: dict[str, int] = {}
    candidates: list[Candidate] = []
    max_cc_area = 0

    if count > 1:
        body = stats[1:]
        areas = body[:, cv2.CC_STAT_AREA]
        max_cc_area = int(areas.max())
        widths = body[:, cv2.CC_STAT_WIDTH]
        heights = body[:, cv2.CC_STAT_HEIGHT]
        lefts = body[:, cv2.CC_STAT_LEFT]
        tops = body[:, cv2.CC_STAT_TOP]
        fills = areas / np.maximum(widths * heights, 1)

        majors = np.maximum(widths, heights)
        minors = np.maximum(1, np.minimum(widths, heights))
        gate = (
            (areas >= params.min_area_px)
            & (majors >= params.min_side_px)
            & (minors >= params.min_minor_px)
            & (majors / minors <= params.max_aspect)
            & (fills >= params.min_fill)
            & (fills < params.max_fill)
        )
        for index in np.flatnonzero(gate):
            box = (
                int(lefts[index]),
                int(tops[index]),
                int(lefts[index] + widths[index]),
                int(tops[index] + heights[index]),
            )
            touches = box[0] <= 1 or box[1] <= 1 or box[2] >= width - 1 or box[3] >= height - 1
            mask = (labels[box[1] : box[3], box[0] : box[2]] == index + 1).astype(np.uint8)
            candidate, reason = classify(mask, box, int(areas[index]), params, touches)
            if candidate is None:
                dropped[reason] = dropped.get(reason, 0) + 1
            else:
                candidates.append(candidate)

    if params.detect_tables and count > 1:
        for cluster in rule_clusters(stats[1:], params):
            if any(_overlaps_any(cluster, [c.box], 0.8) for c in candidates):
                continue
            mask = ink[cluster[1] : cluster[3], cluster[0] : cluster[2]]
            candidate, reason = classify(mask, cluster, int(np.count_nonzero(mask)), params, False, "rules")
            if candidate is None:
                dropped[reason] = dropped.get(reason, 0) + 1
            else:
                candidates.append(candidate)

    for box, label in extra_boxes:
        box = (max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3]))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        if any(_overlaps_any(box, [c.box], 0.8) for c in candidates):
            continue  # эту область уже нашли пиксели, второй раз не считаем
        mask = ink[box[1] : box[3], box[0] : box[2]]
        candidate, reason = classify(mask, box, int(np.count_nonzero(mask)), params, False, f"surya:{label}")
        if candidate is None:
            dropped[reason] = dropped.get(reason, 0) + 1
        else:
            candidates.append(candidate)

    if exclude_boxes:
        kept = [c for c in candidates if not _overlaps_any(c.box, exclude_boxes)]
        if len(kept) != len(candidates):
            dropped["растр по разметке"] = dropped.get("растр по разметке", 0) + len(candidates) - len(kept)
        candidates = kept

    merged = merge_boxes([c.box for c in candidates], gap=params.merge_gap_px) if candidates else []
    min_area = params.min_region_frac * page_area
    boxes = [b for b in merged if (b[2] - b[0]) * (b[3] - b[1]) >= min_area]
    coverage = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / page_area if boxes else 0.0
    full_page = any((b[2] - b[0]) * (b[3] - b[1]) >= params.full_page_frac * page_area for b in boxes)

    return PageFindings(boxes, coverage, candidates, max_cc_area, max(0, count - 1), dropped, full_page)


def _distinct(positions: list[int], gap: int) -> int:
    """Сколько РАЗЛИЧНЫХ позиций в списке: значения ближе ``gap`` считаются одной.

    Перебитая пересечением линейка распадается на куски, и по штукам их несколько, а
    колонка таблицы при этом одна — считать надо позиции (см. ``TABLE_MIN_RULES``).
    """
    distinct = 0
    previous = None
    for value in sorted(positions):
        if previous is None or value - previous > gap:
            distinct += 1
            previous = value
    return distinct


def rule_clusters(stats: np.ndarray, params: LineArtParams) -> list[tuple[int, int, int, int]]:
    """Скопления типографских линеек — заготовки разлинованных таблиц.

    Линейка — пятно длинное, тонкое и очень вытянутое, то есть ровно то, что главный
    детектор отбрасывает как одномерное. Линейки сливаются в скопления с тем же зазором,
    что и находки, и скопление отдаётся наружу, только если в нём есть линейки ОБЕИХ
    ориентаций и каждой не меньше ``table_min_rules`` — причём считаются только ВНУТРЕННИЕ
    линейки (не прижатые к краю скопления) и только их РАЗЛИЧНЫЕ позиции. Иначе таблицей
    оказывается декоративная рамка с одним разделителем колонок — см. ``TABLE_MIN_RULES``.

    Args:
        stats: Массив ``stats`` из ``connectedComponentsWithStats`` БЕЗ строки фона.
        params: Пороги.

    Returns:
        Охватывающие прямоугольники скоплений.
    """
    if len(stats) == 0:
        return []

    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    majors = np.maximum(widths, heights)
    minors = np.maximum(1, np.minimum(widths, heights))
    is_rule = (
        (majors >= params.min_side_px) & (minors <= params.rule_max_minor_px) & (majors / minors > params.max_aspect)
    )
    if not is_rule.any():
        return []

    lefts = stats[:, cv2.CC_STAT_LEFT]
    tops = stats[:, cv2.CC_STAT_TOP]
    boxes = [
        (int(lefts[i]), int(tops[i]), int(lefts[i] + widths[i]), int(tops[i] + heights[i]))
        for i in np.flatnonzero(is_rule)
    ]
    horizontal = [widths[i] >= heights[i] for i in np.flatnonzero(is_rule)]

    clusters = []
    tolerance = params.table_edge_tolerance_px
    for cluster in merge_boxes(boxes, gap=params.merge_gap_px):
        rows: list[int] = []
        columns: list[int] = []
        for box, is_horizontal in zip(boxes, horizontal):
            if not (box[0] >= cluster[0] and box[1] >= cluster[1] and box[2] <= cluster[2] and box[3] <= cluster[3]):
                continue
            if is_horizontal:
                if box[1] > cluster[1] + tolerance and box[3] < cluster[3] - tolerance:
                    rows.append(box[1])
            elif box[0] > cluster[0] + tolerance and box[2] < cluster[2] - tolerance:
                columns.append(box[0])
        if _distinct(rows, params.pitch_px) >= params.table_min_rules:
            if _distinct(columns, params.pitch_px) >= params.table_min_rules:
                clusters.append(cluster)
    return clusters
