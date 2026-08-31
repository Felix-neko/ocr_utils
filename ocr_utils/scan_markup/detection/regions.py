"""Сборка растровых областей полосы: Surya предлагает, пиксели уточняют.

ПОРЯДОК. Первичный детектор — Surya: она размечает вёрстку и предлагает блоки ``Picture``.
Пиксельные проверки идут ПОСЛЕ и решают три вопроса, на которые модель ответить не может:
где настоящая граница картинки, растр это или штриховой рисунок, и цветная ли она.

ПОЧЕМУ ИМЕННО ТАК, А НЕ НАОБОРОТ. Прежде первичным был детектор точек, а Surya добавлялась
к найденному. Замер по 144 полосам пака-1 показал, что роли надо поменять:

    полосы с настоящими фотографиями       Surya даёт Picture на 63 из 63
    полоса с отточиями в оглавлении                          0 из 1
    полосы без картинок вовсе                                0 из 2
    полосы со штриховым рисунком                            28 из 31

Верхние три строки — то, чего не умеют пиксели. Строка отточий в оглавлении состоит из
мелких круглых пятен одинакового размера, то есть даёт ровно ту статистику, по которой
опознаётся растровая печать; отсечь её не вышло ни по размеру (настоящие фотографии доходят
до 0.57% полосы), ни по заполнению прямоугольника клетками (0.44 у отточий против 0.09 у
светлого снимка). Surya же видит там список, а не картинку.

Нижняя строка — то, чего не умеет Surya: штриховой рисунок она уверенно считает картинкой.
Поэтому её блоки обязаны проходить пиксельную проверку, и первичность роли этого не отменяет.

СТРАХОВКА. При чистой схеме всё, что Surya не предложила, терялось бы навсегда. Поэтому
области детектора точек, которых не коснулся ни один блок, остаются, если они КРУПНЫЕ (см.
``SAFETY_MIN_FRAC``). Отточия занимают 0.70% полосы и под порог не проходят, а самая мелкая
настоящая область, не подтверждённая Surya, — 19.78%.

ЧЕГО ЭТО СТОИТ, ЗАМЕРЕНО. Surya прогнана по всем 570 полосам пака-1, у которых прошлый
детектор нашёл хоть что-то (кроме обложек). Она молчит на 102 из них, и вот что на этих 102
видят пиксели:

    пиксели тоже ничего не находят      80   ошибки ПРОШЛОГО детектора: логотип журнала
                                             на полосе содержания, тень разворота, штрих
    область крупная, спасена страховкой  22
    область мелкая, потеряна бы           0

Ни одной полосы, где Surya молчит, пиксели что-то видят, а страховка не срабатывает, на паке
не нашлось. То есть на этом паке первичность Surya не стоит ни одной картинки — но проверять
это на новом паке придётся заново, командой ``validate``.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from ocr_utils.scan_markup.detection import cover as cover_module
from ocr_utils.scan_markup.detection.boxes import MIN_REGION_FRAC, keep_significant, merge_boxes
from ocr_utils.scan_markup.detection.dots import (
    ScreenParams,
    ScreenRegions,
    component_p99_in_box,
    dot_fraction,
    labels_touching,
    union_box,
)

logger = logging.getLogger(__name__)

# p99 площади связного пятна краски, с которой блок Surya считается ШТРИХОВЫМ рисунком, а не
# растром. Проверяется только у блоков, под которыми НЕ нашлось растровых клеток: если клетки
# есть, растр доказан ими, и спорить не о чем.
#
# Замер по НАСТОЯЩИМ границам картинок разделяет классы полностью: фотографии 100..4439,
# штрих 4761..550783. Порог взят из этого промежутка.
#
# ЧЕСТНАЯ ОГОВОРКА: на границах БЛОКОВ SURYA разделение хуже. Блок крупнее картинки и
# прихватывает окружающий текст, чьи длинные компоненты поднимают p99 у фотографий (по
# блокам: фотографии 96..90828, штрих 989..208344, лучший порог даёт 95%). Из-за этого три
# полосы со штрихом из папки-эталона остаются в разметке: 1968/12 IMG_0140_2R (p99 989),
# 1970/03 IMG_0104_2R (2728) и 1969/04 IMG_0045_1L (4220) выглядят «растровее», чем тёмный
# портрет 1969/12 IMG_0140_2R (4439), который терять нельзя. Порога, который делит эти
# четыре числа правильно, не существует, и подгонять его бессмысленно.
#
# Оставлено как есть осознанно. Размен по паку: ветка спасает тёмные фотографии, у которых
# точки растра слиплись, и стоит трёх лишних прямоугольников из тридцати одной полосы со
# штрихом. Лишний прямоугольник разметчик снимает в CVAT одним щелчком, потерянную
# фотографию он не увидит вовсе.
SURYA_LINEART_P99_PX = 4600

# Доля площади полосы, начиная с которой область детектора точек остаётся БЕЗ подтверждения
# Surya. Мотивировка и числа — в докстринге модуля.
SAFETY_MIN_FRAC = 0.03

# Доля площади полосы, покрытая краской, с которой полоса считается сплошным рисунком.
# Замер по копии 1/4: рисунок обложки 1970/04 IMG_0052_2R — 0.256, чёрный штрих из
# папки-эталона 0.142 медиана при 0.230 максимума, текстовые полосы 0.107..0.170. Порог 0.20
# отделяет рисунок от ТЕКСТА, и только от него: отделять цветной рисунок от чёрного он не
# обязан, это делает проверка цвета.
LINEART_FULL_PAGE_INK_FRAC = 0.20

# Доля площади полосы, с которой ЦВЕТНОЙ штриховой рисунок считается иллюстрацией, а не
# библиотечной печатью. Печать — это цветной штрих по определению (фиолетовая мастика), и
# отличить её от цветного рисунка можно только размером.
#
# Замер по паку-1: печати 1966/01 IMG_0004_2R и 1966/02 IMG_0055_2R занимают 2.12% и 2.15%
# полосы, синий рисунок обложки 1970/04 IMG_0052_2R — почти всю. Промежуток огромный, и
# порог поставлен ближе к печатям: цветной двухкрасочный чертёж на внутренней полосе вполне
# может занимать пятую часть кадра, и терять его из-за печатей неправильно.
#
# То, что не дотянуло, не выбрасывается, а помечается ``KIND_STAMP_SUSPECT``: печати всё
# равно предстоит закрашивать, и знать заранее, где они стоят, полезно.
LINEART_PICTURE_MIN_FRAC = 0.15

# Откуда взялась область — от этого зависит, как её помечать (см. ``page._classify_regions``).
SOURCE_SCREEN = "screen"  # растровая печать: клетки нашлись либо пятна мелкие
SOURCE_LINEART = "lineart"  # штриховой рисунок: цветной идёт в разметку, чёрный — нет


@dataclass
class Finding:
    """Одна найденная область: прямоугольник ОРИГИНАЛА и чем она оказалась."""

    box: tuple[int, int, int, int]
    source: str
    full_page: bool = False
    dot_frac: float | None = None
    # Нашлись ли под областью растровые клетки. Растр, доказанный клетками, — это точно
    # иллюстрация; без них область держится либо на статистике пятен, либо на страховке,
    # и мелкую цветную из них надо ещё отличить от библиотечной печати.
    has_cells: bool = True


@dataclass
class RasterFindings:
    """Что нашлось на полосе."""

    findings: list[Finding] = field(default_factory=list)
    cover: bool = False

    @property
    def boxes(self) -> list[tuple[int, int, int, int]]:
        return [finding.box for finding in self.findings]


def find_raster_boxes(
    regions: ScreenRegions,
    stats: np.ndarray,
    centroids: np.ndarray,
    work_gray: np.ndarray,
    shape: tuple[int, int],
    params: ScreenParams,
    surya_boxes: list[tuple[int, int, int, int]] | None = None,
    order_index: int = 0,
    first_page_is_cover: bool = False,
    min_region_frac: float = MIN_REGION_FRAC,
    merge_gap: int | None = None,
    min_side: int | None = None,
    lineart_ink_frac: float = LINEART_FULL_PAGE_INK_FRAC,
    lineart_p99: int = SURYA_LINEART_P99_PX,
    safety_min_frac: float = SAFETY_MIN_FRAC,
) -> RasterFindings:
    """Растровые области полосы. Все координаты — в пикселях ОРИГИНАЛА.

    ``shape`` — размеры ПОЛНОГО кадра ``(height, width)``; ``work_gray`` — его копия 1/4
    (нужна только детектору обложки и признаку сплошного рисунка). ``surya_boxes`` — сырые
    блоки ``Picture``, уже поднятые до координат оригинала, либо ``None``, если Surya не
    гонялась: тогда работают одни пиксели, и это законный режим для прогона без GPU.
    """
    height, width = shape
    merge_gap = params.merge_gap_px if merge_gap is None else merge_gap
    min_side = params.min_region_side_px if min_side is None else min_side

    if cover_module.cover_decision(order_index, work_gray, first_page_is_cover):
        return RasterFindings([Finding(cover_module.cover_region(width, height), SOURCE_SCREEN, True)], cover=True)

    findings: list[Finding] = []
    claimed: set[int] = set()

    for block in surya_boxes or []:
        labels = labels_touching(regions, block)
        if labels:
            # Растр доказан клетками. Область — ОБЪЕДИНЕНИЕ блока и этих клеток, и слово
            # «объединение» тут ключевое: заменять блок клетками нельзя.
            #
            # Соблазн заменить есть — блок в медиане в 1.69 раза крупнее картинки и хватает
            # бумагу вокруг. Но у карты клеток обратная беда: по краю картинки клетка
            # наполовину занята бумагой, точек в ней не набирается, и рамка уезжает ВНУТРЬ
            # фотографии. Замер на замене (1966/02 IMG_0095_2R и IMG_0096_1L, 1966/04
            # IMG_0030_2R, 1966/05 0210_1L): справа и снизу совпадение с блоком, а слева
            # срезано 228..348 px и сверху 184..300 px — то есть по сантиметру с лишним
            # настоящей картинки на двух сторонах.
            #
            # Цена ошибок несимметрична: лишняя бумага по краю не стоит ничего, срезанный
            # кусок фотографии теряется навсегда. То же правило и по той же причине записано
            # в ``background_smoothing.layout``: объединение не сжимает блок никогда.
            #
            # Слияние двух блоков над одной компонентой при этом сохраняется: оба объединения
            # включают общую компоненту, пересекаются и схлопываются в ``_merge_findings``.
            claimed.update(labels)
            cells = union_box(regions, labels)
            box = block if cells is None else _union(block, cells)
            findings.append(Finding(box, SOURCE_SCREEN, dot_frac=dot_fraction(regions.maps, box)))
            continue

        # Клеток под блоком нет: либо тёмный растр, у которого точки слиплись в крупные
        # пятна, либо штриховой рисунок. Границу в обоих случаях берём от блока — уточнять
        # её нечем.
        source = SOURCE_LINEART if component_p99_in_box(stats, centroids, block) >= lineart_p99 else SOURCE_SCREEN
        findings.append(Finding(block, source, dot_frac=dot_fraction(regions.maps, block), has_cells=False))

    # Страховка: крупные растровые области, которых Surya не заметила. Без Surya вовсе
    # (``surya_boxes is None``) остаются все — это режим прогона без GPU.
    threshold = 0.0 if surya_boxes is None else safety_min_frac
    for label, box in regions.boxes.items():
        if label in claimed:
            continue
        if (box[2] - box[0]) * (box[3] - box[1]) < threshold * width * height:
            continue
        findings.append(Finding(box, SOURCE_SCREEN, dot_frac=dot_fraction(regions.maps, box)))

    findings = _merge_findings(findings, merge_gap, width, height, min_region_frac, min_side)
    if findings:
        return RasterFindings(findings)

    # На полосе не нашлось ничего. Остался единственный случай, ради которого стоит смотреть
    # дальше: сплошной рисунок во весь кадр. Цветность проверит вызывающий — здесь нет ни
    # цвета полосы, ни цвета бумаги.
    if cover_module.lineart_side_ok(width, height) and cover_module.lineart_ink_frac(work_gray) >= lineart_ink_frac:
        return RasterFindings(
            [Finding(cover_module.cover_region(width, height), SOURCE_LINEART, True, has_cells=False)]
        )

    return RasterFindings()


def _merge_findings(
    findings: list[Finding], merge_gap: int, width: int, height: int, min_region_frac: float, min_side: int
) -> list[Finding]:
    """Слияние пересекающихся и отсев мелочи, с сохранением происхождения области.

    Слитая область считается растровой, если растровым был хоть один её кусок: найденные
    клетки — доказательство, а их отсутствие у соседа доказательством обратного не является.
    """
    if not findings:
        return []

    kept = keep_significant(merge_boxes([f.box for f in findings], merge_gap), width, height, min_region_frac, min_side)

    result = []
    for box in kept:
        parts = [f for f in findings if _inside(f.box, box)]
        fracs = [f.dot_frac for f in parts if f.dot_frac is not None]
        result.append(
            Finding(
                box=box,
                source=SOURCE_SCREEN if any(f.source == SOURCE_SCREEN for f in parts) else SOURCE_LINEART,
                full_page=any(f.full_page for f in parts),
                dot_frac=max(fracs) if fracs else None,
                has_cells=any(f.has_cells for f in parts),
            )
        )
    return result


def _union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Охватывающий прямоугольник двух — объединение, которое не сжимает ни один из них."""
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    """Лежит ли ``inner`` внутри ``outer`` — так части сопоставляются со слитой областью."""
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]
