"""Поиск полутоновой (растровой) печати по статистике пятен краски на ПОЛНОМ кадре.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ПРЕДЫДУЩЕГО ДЕТЕКТОРА. Тот искал «средние тона 100..225» по копии
1/4 и путал полутоновую фотографию со штриховым рисунком: на копии 1/4 плотная штриховка
гравюры усредняется ровно в такой же серый. Замер по паку-1: 31 полоса со штриховыми
виньетками и эмблемами рубрик уехала в растр, причём одна и та же гравюра ловилась из
номера в номер (1970/04 IMG_0015..0021_1L — семь раз подряд).

НА ЧЁМ ДЕРЖИТСЯ РАЗДЕЛЕНИЕ. Полутоновая печать распадается на МНОЖЕСТВО МЕЛКИХ пятен
ограниченного размера, штрих даёт длинные связные компоненты. Замер по 67 областям с
настоящими фотографиями против 39 областей со штриховым рисунком (порог краски —
``adaptiveThreshold``, дальше связные компоненты и статистика их ПЛОЩАДЕЙ):

    признак                                фотографии      штрих            точность
    p99 площади компоненты                 100 .. 4439     4761 .. 550783   100 %
    доля компонент размером с точку        0.699 .. 0.993  0.128 .. 0.926    97 %
    p90 площади компоненты                 18 .. 669       50 .. 20732       95 %

ЧЕГО ДЕЛАТЬ НЕ НАДО (проверено, не работает). Буквальный подсчёт ТОЧЕК РАСТРА — плотность
точек на клетку, как это описано в патентах по сегментации сканов, — на этих сканах не
разделяет ничего: замер дал 1..11 точек на клетку 128 px и у фотографий, и у штриха, и у
чистого текста. Растровая сетка на камерных сканах смазана и как периодическая структура
не разрешается вовсе. Не работает и «доля пикселей крупных компонент»: краска на полосе
смыкается в одну сплошную сеть, и эта доля равна 0.5..0.7 везде. Работает именно СЧЁТНАЯ
статистика площадей — сколько компонент мелкие, а не сколько места они занимают.

ПОЧЕМУ ПО КЛЕТКАМ. Область надо не только опознать, но и обвести, а связная компонента
краски на роль иллюстрации не годится — она либо мельче зерна, либо смыкается с соседним
текстом. Поэтому кадр режется на клетки, каждая клетка получает свою статистику, и
иллюстрация собирается как связная область РАСТРОВЫХ КЛЕТОК. Побочная выгода: светлая
фотография перестаёт распадаться на куски (её светлые участки — те же мелкие пятна, просто
пореже), а это был отдельный дефект прошлого детектора.

КАК ОТКАЛИБРОВАНО. Перебором по валидационной выборке из шести папок-эталонов (см.
``scan_markup.validation``) плюс сорок случайных полос с уже найденными областями в качестве
контроля. Итог на выбранных значениях:

    фотографии, которые надо найти            62 / 63
    штриховой рисунок, который не надо        31 / 31
    полосы без картинок                        2 / 2
    две картинки не слиплись в одну            2 / 2
    одна картинка не рассыпалась                5 / 5

Единственная потеря — тёмный портрет 1969/12 IMG_0140_2R: печать плотная, точки растра в
тенях смыкаются в крупные пятна, и клетки не проходят по ``CELL_MAX_AREA_PX``. Ослабление
любого из порогов её возвращает, но ценой трёх лишних штриховых рисунков и одной слипшейся
пары фотографий — размен явно невыгодный, поэтому оставлено как есть.

ЧТО ОСТАЛОСЬ НЕПОЧИНЕННЫМ. Строка отточий в оглавлении — это ряд мелких круглых пятен
одинакового размера, то есть ровно та статистика, по которой опознаётся растр. На полосе
содержания детектор находит из-за них лишний прямоугольник (1969/09 IMG_0151_2R, 0.7% полосы).
Проверены и отброшены два способа его отсечь:

* по размеру — не выходит: настоящие фотографии в выборке доходят до 0.57% полосы;
* по заполнению прямоугольника растровыми клетками — тоже: у отточий 0.44, а у настоящих
  фотографий бывает и 0.09 (светлый снимок с большим белым полем).

Оставлено как есть сознательно: лишний прямоугольник разметчик снимает в CVAT одним щелчком,
а любой порог, который его убирает, режет и настоящие фотографии.

Из сорока контрольных полос область перестала находиться у шести — все шесть проверены
глазами, и все шесть были ошибками ПРОШЛОГО детектора: логотип журнала на полосе содержания
(1971/02 IMG_0053_2R, 1972/09 IMG_0105_2R и однотипные) и тень разворота у края
(1976/01 IMG_0008_1L).
"""

import logging
from dataclasses import dataclass, replace

import cv2
import numpy as np

from ocr_utils.scan_markup.detection.boxes import MERGE_GAP_PX, MIN_REGION_SIDE_PX

logger = logging.getLogger(__name__)

# Разрешение, при котором заданы все пиксельные константы ниже. Паки бывают 300, 450 и
# 600 dpi, поэтому пороги не константы, а функция от разрешения — см. :func:`params_for_dpi`.
REFERENCE_DPI = 600

# Окно локального порога краски. Порядка высоты строки: меньше — и внутренность крупной
# буквы начнёт считаться фоном, больше — и порог перестанет успевать за неравномерностью
# освещения камерного скана.
BLOCK_PX = 31

# Сдвиг локального порога от среднего по окну, в уровнях яркости. Отрицательный, то есть
# в сторону бумаги: краской считается то, что заметно темнее окрестности. На чистой бумаге
# это оставляет маску пустой, а не режет зерно пополам.
#
# КОМПЕНСАЦИЯ УРОВНЕЙ ПЕРЕД ПОРОГОМ ПРОВЕРЕНА И НЕ ПОМОГАЕТ. Сдвиг тут абсолютный, а в
# тёмной плотной печати динамический диапазон сжат, и напрашивается выровнять его заранее.
# Замер (клеток растра в области):
#
#                       тёмный портрет   чертёж   ЧИСТЫЙ ТЕКСТ
#     как сейчас           13 из 140     18/198      0 из 680
#     уровни 1..99%        11            17          0
#     CLAHE 2.0/8          33            52         28
#     CLAHE 4.0/16         48           104        108
#
# Глобальные уровни не дают ничего: ``adaptiveThreshold`` и так считает порог от локального
# среднего. CLAHE тёмный портрет вытаскивает, но ровно в той же пропорции вытаскивает
# штриховой чертёж и начинает видеть растр в ЧИСТОМ ТЕКСТЕ — то есть ломает то, на чём
# держится весь детектор. Перебор по силе (clipLimit 1.0..2.0, плитка 8..32) промежуточной
# настройки не нашёл: там, где портрет даёт область, её дают и чертежи, и текст.
THRESHOLD_BIAS = -5

# Пятно считается «точечным», если умещается в эти рамки. 300 px при 600 dpi — это круг
# диаметром 20 px, то есть 0.85 мм: крупнее самой жирной точки растра в тенях, но мельче
# буквы основного текста (у неё площадь несколько сотен px при высоте 60).
DOT_MAX_AREA_PX = 300
DOT_MAX_SIDE_PX = 24

# Сторона клетки, по которой собирается статистика. 128 px при 600 dpi — 5.4 мм: заметно
# больше шага растра (клетке есть что усреднять) и заметно меньше самой мелкой иллюстрации.
CELL_PX = 128

# Клетка растровая, если почти все её пятна — точечные...
DOT_FRAC_THR = 0.88
# ...и среди них нет ни одного крупного пятна. Порог взят по замеру выше: у фотографий p99
# площади не превысил 4439, у штриха начинался с 4761.
CELL_MAX_AREA_PX = 4400
# ...и пятен вообще достаточно, чтобы доля что-то значила. На пустой бумаге пятен единицы,
# и доля «все три пятна точечные» ничего не говорит.
MIN_DOTS_PER_CELL = 6

# Морфология по карте клеток (не по пикселям — на два порядка дешевле). Замыкание сшивает
# иллюстрацию через светлые провалы (4 клетки — 2.2 см оригинала: перекрывает и белое небо,
# и просвет между двумя половинами разворотной фотографии), размыкание убирает одиночные
# клетки, которыми штриховка усеивает полосу.
#
# Размыкание ровно 2, а не 3, и это осознанный размен. При 3 из выборки уходит одна лишняя
# полоса со штрихом, но теряется и одна настоящая фотография (1969/12 IMG_0140_2R), а у
# ошибок разная цена: лишний прямоугольник разметчик снимает в CVAT одним щелчком, потерянную
# фотографию он не увидит вовсе. Полосы с фотографиями, вырезанными по белому фону
# (1968/10 IMG_0035_1L — пятнадцать снимков ящиков в ряд), дают редкие разрозненные клетки и
# гибнут при размыкании 3 целиком.
CLOSE_CELLS = 4
OPEN_CELLS = 2

# Минимальный размер области в клетках. Считается по клеткам ДО морфологии — то есть по
# настоящему полутоновому содержимому, а не по раздутому замыканием пятну. Разница
# принципиальная: замыкание доводит пятно в две клетки до восьми и больше, и по площади
# ПОСЛЕ него сквозь порог проходит любая крапина — строка отточий в оглавлении даёт ровно
# такие же мелкие круглые пятна, что и растровая сетка (1969/09 IMG_0151_2R).
#
# Поэтому и само число мало: три клетки исходного растра. Крупность области дальше меряет
# ``MIN_REGION_FRAC`` в ``boxes``, уже по готовому прямоугольнику.
MIN_CELLS = 3


@dataclass(frozen=True)
class ScreenParams:
    """Пороги детектора при КОНКРЕТНОМ разрешении полосы.

    Собирается только через :func:`params_for_dpi` — там единственное место, где живёт
    пересчёт 300/450/600 dpi, и разъехаться ему больше негде.
    """

    block_px: int = BLOCK_PX
    threshold_bias: int = THRESHOLD_BIAS
    dot_max_area_px: int = DOT_MAX_AREA_PX
    dot_max_side_px: int = DOT_MAX_SIDE_PX
    cell_px: int = CELL_PX
    dot_frac_thr: float = DOT_FRAC_THR
    cell_max_area_px: int = CELL_MAX_AREA_PX
    min_dots_per_cell: int = MIN_DOTS_PER_CELL
    close_cells: int = CLOSE_CELLS
    open_cells: int = OPEN_CELLS
    min_cells: int = MIN_CELLS
    # Пороги над готовыми прямоугольниками — здесь же, чтобы точка пересчёта от dpi
    # осталась одна на всю детекцию (значения при 600 dpi — в ``boxes``).
    merge_gap_px: int = MERGE_GAP_PX
    min_region_side_px: int = MIN_REGION_SIDE_PX


def _odd(value: float) -> int:
    """Ближайшее нечётное не меньше 3 — ``adaptiveThreshold`` требует нечётного окна."""
    value = max(3, int(round(value)))
    return value if value % 2 else value + 1


def params_for_dpi(dpi: int | None, **overrides) -> ScreenParams:
    """Пороги, пересчитанные с ``REFERENCE_DPI`` на разрешение полосы.

    ДЛИНЫ масштабируются линейно, ПЛОЩАДИ — квадратом отношения, ДОЛИ и счётчики не
    масштабируются вовсе: доля точечных пятен — величина безразмерная, а число пятен в
    клетке не меняется, потому что вместе с dpi растёт и сама клетка.

    ``overrides`` (из опций CLI) применяются ПОСЛЕ пересчёта и задаются в пикселях этой
    самой полосы: если порог выставляют руками, спорить с ним масштабом неправильно.
    ``None`` среди значений игнорируются — так click передаёт «опция не задана».
    """
    scale = 1.0 if not dpi else float(dpi) / REFERENCE_DPI
    params = ScreenParams(
        block_px=_odd(BLOCK_PX * scale),
        dot_max_area_px=max(1, round(DOT_MAX_AREA_PX * scale * scale)),
        dot_max_side_px=max(1, round(DOT_MAX_SIDE_PX * scale)),
        cell_px=max(8, round(CELL_PX * scale)),
        cell_max_area_px=max(1, round(CELL_MAX_AREA_PX * scale * scale)),
        merge_gap_px=max(1, round(MERGE_GAP_PX * scale)),
        min_region_side_px=max(1, round(MIN_REGION_SIDE_PX * scale)),
    )
    given = {name: value for name, value in overrides.items() if value is not None}
    return replace(params, **given) if given else params


def ink_components(gray: np.ndarray, params: ScreenParams) -> tuple[np.ndarray, np.ndarray]:
    """Связные пятна краски на ПОЛНОМ кадре: ``(stats, centroids)`` без фона.

    Порог локальный, а не глобальный: у камерного скана яркость бумаги плывёт по кадру на
    десятки уровней, и один порог на всю полосу либо теряет светлый угол, либо заливает
    краской тёмный.
    """
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, params.block_px, params.threshold_bias
    )
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(ink, 8)
    if count <= 1:  # на кадре нет краски вовсе
        return np.empty((0, 5), np.int32), np.empty((0, 2), np.float64)
    return stats[1:], centroids[1:]


def is_dot(stats: np.ndarray, params: ScreenParams) -> np.ndarray:
    """Маска «пятно точечное» по строкам ``stats`` из :func:`ink_components`."""
    if len(stats) == 0:
        return np.zeros(0, bool)
    area = stats[:, cv2.CC_STAT_AREA]
    side = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    return (area <= params.dot_max_area_px) & (side <= params.dot_max_side_px)


@dataclass(frozen=True)
class CellMaps:
    """Статистика пятен, собранная по клеткам кадра.

    ``dots`` — сколько точечных пятен в клетке, ``total`` — сколько всего, ``biggest`` —
    площадь самого крупного. Три карты одного размера ``(gh, gw)``.
    """

    dots: np.ndarray
    total: np.ndarray
    biggest: np.ndarray
    cell_px: int


def cell_maps(stats: np.ndarray, centroids: np.ndarray, shape: tuple[int, int], params: ScreenParams) -> CellMaps:
    """Раскладывает пятна по клеткам кадра ``shape`` = ``(height, width)``.

    Пятно относится к клетке по своему ЦЕНТРОИДУ, а не по площади: длинный штрих
    пересекает десяток клеток, и разложи мы его по пикселям — он бы «испортил» ровно те
    клетки, по которым проходит, а не ту, где он есть. Нам же нужно обратное: одна клетка
    с одним крупным пятном должна выпасть из растровых целиком.
    """
    height, width = shape
    cell = params.cell_px
    grid_h, grid_w = max(1, height // cell), max(1, width // cell)

    dots = np.zeros((grid_h, grid_w), np.int32)
    total = np.zeros((grid_h, grid_w), np.int32)
    biggest = np.zeros((grid_h, grid_w), np.int64)
    if len(stats) == 0:
        return CellMaps(dots, total, biggest, cell)

    rows = np.clip((centroids[:, 1] // cell).astype(np.int64), 0, grid_h - 1)
    cols = np.clip((centroids[:, 0] // cell).astype(np.int64), 0, grid_w - 1)
    dot = is_dot(stats, params)

    np.add.at(total, (rows, cols), 1)
    np.add.at(dots, (rows[dot], cols[dot]), 1)
    np.maximum.at(biggest, (rows, cols), stats[:, cv2.CC_STAT_AREA].astype(np.int64))
    return CellMaps(dots, total, biggest, cell)


def raw_screen_cells(maps: CellMaps, params: ScreenParams) -> np.ndarray:
    """Карта 0/1 «клетка занята полутоновой печатью» ДО морфологии.

    Три условия сразу, и все три нужны: доля точечных пятен высока (иначе это текст или
    штрих), крупных пятен нет вовсе (иначе это штрих, у которого мелочи тоже хватает), и
    самих точечных пятен достаточно, чтобы доля что-то значила (иначе это чистая бумага).
    """
    frac = np.divide(maps.dots, maps.total, out=np.zeros(maps.dots.shape, np.float32), where=maps.total > 0)
    return (
        (frac >= params.dot_frac_thr)
        & (maps.biggest < params.cell_max_area_px)
        & (maps.dots >= params.min_dots_per_cell)
    ).astype(np.uint8)


def screen_cells(maps: CellMaps, params: ScreenParams) -> np.ndarray:
    """То же, но после морфологии: замыкание сшивает, размыкание чистит одиночные клетки."""
    mask = raw_screen_cells(maps, params)
    if params.close_cells > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((params.close_cells,) * 2, np.uint8))
    if params.open_cells > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((params.open_cells,) * 2, np.uint8))
    return mask


def boxes_from_cells(
    mask: np.ndarray, cell_px: int, shape: tuple[int, int], params: ScreenParams, raw: np.ndarray | None = None
) -> list[tuple]:
    """Связные области растровых клеток -> прямоугольники в координатах ОРИГИНАЛА.

    СВЯЗНОСТЬ решает ``mask`` (после морфологии), а ГРАНИЦЫ — ``raw`` (до неё). Разделение
    существенное: замыкание нужно, чтобы фотография не рассыпалась на куски по светлым
    провалам, но оно же раздувает прямоугольник на радиус ядра во все стороны, а ядро — это
    несколько клеток, то есть сантиметр оригинала. Взяв рамку по исходным клеткам внутри
    компоненты, получаем и сшивание, и тесную рамку.

    Правый и нижний края доводятся до размеров кадра, если упёрлись в последнюю клетку:
    кадр редко делится на клетку нацело, и остаток шириной меньше клетки принадлежит той же
    иллюстрации, просто в сетку не попал.
    """
    height, width = shape
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if raw is not None:
            rows, cols = np.nonzero(raw & (labels == label))
            # Порог размера меряется по ИСХОДНЫМ клеткам, а не по замкнутой компоненте.
            # Замыкание раздувает пятно в 2x2 клетки до восьми и больше, и по площади после
            # него сквозь порог проходит любая крапина: строка отточий в оглавлении даёт
            # ровно такие же мелкие круглые пятна, что и растровая сетка.
            if rows.size < params.min_cells:
                continue
            x, y = int(cols.min()), int(rows.min())
            w, h = int(cols.max()) - x + 1, int(rows.max()) - y + 1
        elif area < params.min_cells:
            continue
        x1, y1 = int(x) * cell_px, int(y) * cell_px
        x2, y2 = int(x + w) * cell_px, int(y + h) * cell_px
        if x + w >= mask.shape[1]:
            x2 = width
        if y + h >= mask.shape[0]:
            y2 = height
        boxes.append((x1, y1, min(x2, width), min(y2, height)))
    return boxes


def screen_boxes(gray: np.ndarray, params: ScreenParams) -> tuple[list[tuple], CellMaps]:
    """Полутоновые области полосы: прямоугольники ОРИГИНАЛА и карты клеток.

    Карты возвращаются вместе с боксами, потому что нужны дальше — по ним считается
    ``dot_frac`` области и проверяются блоки Surya, — а второй проход по 21-мегапиксельному
    кадру стоит секунду на полосу и двенадцать тысяч полос на пак.
    """
    maps = cell_maps(*ink_components(gray, params), gray.shape[:2], params)
    raw = raw_screen_cells(maps, params)
    mask = screen_cells(maps, params)
    return boxes_from_cells(mask, maps.cell_px, gray.shape[:2], params, raw), maps


def dot_fraction(maps: CellMaps, box: tuple[int, int, int, int]) -> float:
    """Доля точечных пятен внутри прямоугольника — то, что пишется в базу.

    Считается по уже готовым картам клеток, поэтому произвольный прямоугольник округляется
    до клеток. Для отбора это неважно (решение принято раньше и по клеткам же), а для
    перекалибровки порогов по базе точности клетки хватает с запасом.
    """
    cell = maps.cell_px
    grid_h, grid_w = maps.dots.shape
    x1 = np.clip(box[0] // cell, 0, grid_w - 1)
    y1 = np.clip(box[1] // cell, 0, grid_h - 1)
    x2 = np.clip(-(-box[2] // cell), x1 + 1, grid_w)
    y2 = np.clip(-(-box[3] // cell), y1 + 1, grid_h)
    dots = int(maps.dots[y1:y2, x1:x2].sum())
    total = int(maps.total[y1:y2, x1:x2].sum())
    return float(dots) / total if total else 0.0


@dataclass(frozen=True)
class ScreenRegions:
    """Связные полутоновые области полосы: карта меток по клеткам и их прямоугольники.

    Нужна там, где области не просто перечисляют, а СОПОСТАВЛЯЮТ с чужими прямоугольниками —
    блоками Surya. Блок садится на компоненту, и дальше границей служит компонента, а не он:
    замер по паку-1 показал, что блок Surya в медиане в 1.69 раза крупнее настоящей картинки
    (хватает бумагу вокруг), а область точек лежит внутри блока в 56 случаях из 76.

    ``labels`` — метки ПОСЛЕ морфологии (ими решается связность), ``boxes`` — прямоугольники
    по клеткам ДО неё (ими задаётся граница). Разделение то же, что в
    :func:`boxes_from_cells`, и по той же причине.
    """

    labels: np.ndarray
    boxes: dict[int, tuple[int, int, int, int]]
    maps: CellMaps


def screen_regions(gray: np.ndarray, params: ScreenParams) -> tuple[ScreenRegions, np.ndarray, np.ndarray]:
    """Полутоновые области полосы плюс сырые связные пятна краски.

    Пятна (``stats``, ``centroids``) возвращаются наружу не для красоты: по ним считается
    :func:`component_p99_in_box` для блоков Surya, а второй проход ``adaptiveThreshold`` по
    21-мегапиксельному кадру стоит секунду на полосу.
    """
    stats, centroids = ink_components(gray, params)
    maps = cell_maps(stats, centroids, gray.shape[:2], params)
    return regions_from_maps(maps, gray.shape[:2], params), stats, centroids


def regions_from_maps(maps: CellMaps, shape: tuple[int, int], params: ScreenParams) -> ScreenRegions:
    """Связные области по готовым картам клеток."""
    height, width = shape
    raw = raw_screen_cells(maps, params)
    mask = screen_cells(maps, params)
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)

    boxes: dict[int, tuple[int, int, int, int]] = {}
    for label in range(1, count):
        rows, cols = np.nonzero(raw & (labels == label))
        if rows.size < params.min_cells:
            continue
        x, y = int(cols.min()), int(rows.min())
        w, h = int(cols.max()) - x + 1, int(rows.max()) - y + 1
        x1, y1 = x * maps.cell_px, y * maps.cell_px
        x2 = width if x + w >= labels.shape[1] else (x + w) * maps.cell_px
        y2 = height if y + h >= labels.shape[0] else (y + h) * maps.cell_px
        boxes[label] = (x1, y1, min(x2, width), min(y2, height))
    return ScreenRegions(labels, boxes, maps)


def labels_touching(regions: ScreenRegions, box: tuple[int, int, int, int]) -> list[int]:
    """Метки областей, чьи клетки попадают в прямоугольник ``box`` (координаты ОРИГИНАЛА)."""
    cell = regions.maps.cell_px
    grid_h, grid_w = regions.labels.shape
    x1 = int(np.clip(box[0] // cell, 0, grid_w - 1))
    y1 = int(np.clip(box[1] // cell, 0, grid_h - 1))
    x2 = int(np.clip(-(-box[2] // cell), x1 + 1, grid_w))
    y2 = int(np.clip(-(-box[3] // cell), y1 + 1, grid_h))
    found = np.unique(regions.labels[y1:y2, x1:x2])
    return [int(label) for label in found if label and int(label) in regions.boxes]


def union_box(regions: ScreenRegions, labels: list[int]) -> tuple[int, int, int, int] | None:
    """Охватывающий прямоугольник объединения областей с этими метками."""
    boxes = [regions.boxes[label] for label in labels if label in regions.boxes]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def component_p99_in_box(stats: np.ndarray, centroids: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """p99 площади связных пятен краски, чей центроид лежит внутри ``box``.

    Признак «растр или штрих» для блока, под которым НЕ нашлось растровых клеток. Замер по
    67 областям с фотографиями против 39 со штриховым рисунком разделил их полностью:
    фотографии 100..4439, штрих 4761..550783 (см. докстринг модуля). Пятен меньше горстки —
    судить не по чему, возвращается 0.0, то есть «на штрих не похоже».
    """
    if len(stats) == 0:
        return 0.0
    inside = (
        (centroids[:, 0] >= box[0])
        & (centroids[:, 0] < box[2])
        & (centroids[:, 1] >= box[1])
        & (centroids[:, 1] < box[3])
    )
    areas = stats[inside][:, cv2.CC_STAT_AREA]
    if areas.size < 10:
        return 0.0
    return float(np.percentile(areas, 99))
