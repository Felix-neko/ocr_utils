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

import cv2

from ocr_utils.background_smoothing.processing import HALFTONE_DOWNSCALE
from ocr_utils.scan_markup.detection import cover as cover_module
from ocr_utils.scan_markup.detection.boxes import FULL_PAGE_FRAC, MIN_REGION_FRAC, keep_significant, merge_boxes
from ocr_utils.scan_markup.detection.color_kind import CHROMA_SPREAD_THR
from ocr_utils.scan_markup.detection.tone import (
    LINEART_ENTROPY_THR,
    LINEART_MID_FRAC_THR,
    LINEART_SCREEN_PEAK_THR,
    ToneMaps,
    looks_like_line_art,
    tone_stats,
)
from ocr_utils.scan_markup.detection.dots import (
    CELL_PX,
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

# Доля площади полосы, с которой охватывающий прямоугольник ВСЕХ найденных областей
# считается «полоса занята картинкой целиком». Работает в паре с проверкой цвета — см.
# :func:`_fill_colour_page`.
FULL_PAGE_COLOR_FRAC = FULL_PAGE_FRAC

# Пороги «это отточия, а не растр». Отточия в таблице или оглавлении дают ровно ту
# статистику, по которой опознаётся растровая печать, и по размеру, плотности, расстоянию
# между точками и разбросу их площади от фотографии не отличаются — всё это замерено и
# перекрывается. Отличается СТРОЕНИЕ: точки стоят по базовым линиям текста, между линиями
# бумага. Замер по 6 областям с отточиями против 65 настоящих:
#
#     отточия                            пустых строк 0.477..0.651  период 0.595..0.891
#     настоящие с пустых строк >= 0.30   пустых строк до 0.696      период до 0.511
#
# Порознь ни один признак не делит, вместе делят полностью: правило отбрасывает 6 отточий
# из 6 и не теряет ни одной из 65 настоящих областей.
#
# Проверяется ТОЛЬКО у страховочных областей: все шесть ошибок оттуда, а области с блоком
# Surya моделью уже поручены, и сужать им ворота незачем.
LEADER_EMPTY_ROWS_THR = 0.30
LEADER_PERIODICITY_THR = 0.52

# ВТОРОЙ, НЕЗАВИСИМЫЙ признак отточий: размах яркостей после сильного уменьшения области.
#
# Смысл прямой: у фотографии есть светлые и тёмные места, и когда растр «расседается»
# усреднением, остаётся непрерывный тон с широким размахом. Колонка отточий усредняется в
# почти однородный светлый прямоугольник. Уменьшаем до ~7.5 dpi (от копии 1/4 это делитель
# 20 при 600 dpi) и берём p95 - p5.
#
# Замер по тем же 6 областям с отточиями против 65 настоящих, считая от копии 1/4:
#
#     отточия     9.2 .. 62.6
#     настоящие 101.2 .. 228.9
#
# Разделение полное, и зазор шире, чем у строкового признака (64 против 101, то есть в 1.6
# раза), — но признаки нарочно оставлены ОБА и работают через «и»: чтобы выбросить область,
# она должна провалить и строение, и тон. Настоящая фотография провалит оба разом с трудом,
# а цена ошибки несимметрична — лишний прямоугольник снимается щелчком, потерянная
# фотография не восстанавливается.
LEADER_TONE_SPREAD_THR = 80.0
LEADER_TONE_DIVISOR = 20

# Расширение страховочной рамки до бумаги. У области без блока Surya объединять не с чем, и
# граница берётся по сырым растровым клеткам, а они по краям снимка не добирают: светлые
# участки фотографии клеток не дают (1973/12 IMG_0271_1L). Поэтому каждая сторона растёт,
# пока прилегающая полоска заметно темнее бумаги.
#
# Брать вместо этого рамку по компоненте ПОСЛЕ замыкания проверено и отвергнуто: она растёт
# на 10..31%, но вниз, в подпись под фотографией, а левый край снимка всё равно срезан.
#
# Фотография отличается от бумаги тоном по всей площади, включая светлые участки, а подпись
# под ней — тонкий текст на белом, и средняя яркость полоски там почти бумажная, так что
# рост на подписи останавливается. Замер: рамка встаёт ровно по краю снимка на трёх полосах.
GROW_STEP_PX = 64
# Подрезка того, что клеточная сетка добавила СВЕРХ блока Surya. Край области округляется
# наружу до целой клетки (128 px при 600 dpi, около 5 мм), и в этот последний ряд попадает
# подпись под фотографией: замер по 1974/03 IMG_0140_1L — низ фотографии 3365, низ блока
# Surya 3396, а рамка кончалась на 3456.
#
# Подрезка идёт тем же признаком бумаги, что и рост, но внутрь, и НИКОГДА не заходит глубже
# блока Surya: блок — это то, что поручила модель, и спорить с ним пикселями нельзя, иначе
# вернётся срезание светлых краёв фотографии. Замер: 1974/03 IMG_0140_1L подрезалась ровно
# до блока, на прочих полосах ушло 0..36 px, две полосы с двумя фотографиями не задеты.
SHRINK_TO_BLOCK = True
GROW_PAPER_MARGIN = 25
GROW_PAPER_PERCENTILE = 97.0
GROW_MAX_STEPS = 60

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
    # Признаки «растр или штрих» (``detection.tone``). Пишутся в базу всегда, даже когда
    # правило не сработало: следующая итерация порогов тогда делается по measurements.csv,
    # без единого чтения TIFF.
    mid_frac: float | None = None
    tone_entropy: float | None = None
    screen_peak: float | None = None
    # Контраст «бумага минус краска». В решении «растр или штрих» не участвует: по нему
    # бледный оттиск печати отличается от чёрной виньетки (см. ``page._classify_regions``).
    ink_contrast: float | None = None


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
    page_chroma_spread: float | None = None,
    chroma_spread_thr: float = CHROMA_SPREAD_THR,
    full_page_color_frac: float = FULL_PAGE_COLOR_FRAC,
    leader_empty_rows_thr: float = LEADER_EMPTY_ROWS_THR,
    leader_periodicity_thr: float = LEADER_PERIODICITY_THR,
    leader_tone_spread_thr: float = LEADER_TONE_SPREAD_THR,
    grow_paper_margin: int = GROW_PAPER_MARGIN,
    tone: ToneMaps | None = None,
    lineart_mid_frac: float = LINEART_MID_FRAC_THR,
    lineart_entropy: float = LINEART_ENTROPY_THR,
    lineart_screen_peak: float = LINEART_SCREEN_PEAK_THR,
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
            box = shrink_to_paper(work_gray, box, block, grow_paper_margin)
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
        empty_rows, periodicity = regions.leader.get(label, (0.0, 0.0))
        looks_lined = empty_rows >= leader_empty_rows_thr and periodicity >= leader_periodicity_thr
        if looks_lined and tone_spread(work_gray, box, params) < leader_tone_spread_thr:
            continue  # отточия в таблице, а не растровая печать: и строение, и тон против
        box = grow_to_paper(work_gray, box, (height, width), grow_paper_margin)
        findings.append(Finding(box, SOURCE_SCREEN, dot_frac=dot_fraction(regions.maps, box)))

    findings = _merge_findings(findings, merge_gap, width, height, min_region_frac, min_side)
    # Растр это или штрих, решается ПОСЛЕ слияния: признаки объёмные, и мерить их надо по
    # окончательному прямоугольнику, а не по половинке картинки.
    findings = _mark_line_art(findings, tone, lineart_mid_frac, lineart_entropy, lineart_screen_peak)
    if findings:
        return RasterFindings(
            _fill_colour_page(findings, width, height, page_chroma_spread, chroma_spread_thr, full_page_color_frac)
        )

    # На полосе не нашлось ничего. Остался единственный случай, ради которого стоит смотреть
    # дальше: сплошной рисунок во весь кадр. Цветность проверит вызывающий — здесь нет ни
    # цвета полосы, ни цвета бумаги.
    if cover_module.lineart_side_ok(width, height) and cover_module.lineart_ink_frac(work_gray) >= lineart_ink_frac:
        return RasterFindings(
            [Finding(cover_module.cover_region(width, height), SOURCE_LINEART, True, has_cells=False)]
        )

    return RasterFindings()


def _mark_line_art(
    findings: list[Finding], tone: ToneMaps | None, mid_frac_thr: float, entropy_thr: float, screen_peak_thr: float
) -> list[Finding]:
    """Переводит находки, похожие на штрих, из ``SOURCE_SCREEN`` в ``SOURCE_LINEART``.

    Проверяются ВСЕ находки, а не только блоки Surya без растровых клеток: штриховка даёт
    ровно такие же мелкие пятна, что и растр, поэтому клетки под рисунком находятся исправно
    и до сих пор считались доказательством растра. Признаки и замеры — в ``detection.tone``.

    ``tone is None`` — карты не посчитаны (старый разбор из пула, тесты по кускам кадра):
    тогда правило молчит, и всё остаётся как было.
    """
    if tone is None:
        return findings
    for finding in findings:
        stats = tone_stats(tone, finding.box)
        finding.mid_frac = stats.mid_frac
        finding.tone_entropy = stats.entropy
        finding.screen_peak = stats.screen_peak
        finding.ink_contrast = stats.ink_contrast
        if looks_like_line_art(stats, mid_frac_thr, entropy_thr, screen_peak_thr):
            finding.source = SOURCE_LINEART
    return findings


def tone_spread(work_gray: np.ndarray, box: tuple[int, int, int, int], params: ScreenParams) -> float:
    """Размах яркостей области после сильного уменьшения: ``p95 - p5``. Мотивировка выше.

    Считается по копии 1/``HALFTONE_DOWNSCALE``, которая и так есть в родителе; делитель
    масштабируется от разрешения полосы, чтобы «около 7.5 dpi» означало одно и то же на
    паках 300, 450 и 600 dpi.
    """
    scale = params.cell_px / CELL_PX
    divisor = max(1, round(LEADER_TONE_DIVISOR * scale))
    x1, y1 = box[0] // HALFTONE_DOWNSCALE, box[1] // HALFTONE_DOWNSCALE
    x2, y2 = -(-box[2] // HALFTONE_DOWNSCALE), -(-box[3] // HALFTONE_DOWNSCALE)
    crop = work_gray[y1:y2, x1:x2]
    if crop.size < 16:
        return float("inf")  # судить не по чему — не мешаем области остаться

    small = cv2.resize(
        crop, (max(2, crop.shape[1] // divisor), max(2, crop.shape[0] // divisor)), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    return float(np.percentile(small, 95) - np.percentile(small, 5))


def shrink_to_paper(
    work_gray: np.ndarray,
    box: tuple[int, int, int, int],
    floor: tuple[int, int, int, int],
    margin: int = GROW_PAPER_MARGIN,
    step_px: int = GROW_STEP_PX,
    max_steps: int = GROW_MAX_STEPS,
) -> tuple[int, int, int, int]:
    """Убирает с краёв рамки бумагу, но не заходит внутрь ``floor``. Мотивировка выше.

    ``floor`` — блок Surya: подрезать глубже него нельзя, иначе вернётся срезание светлых
    краёв фотографии, ради борьбы с которым и делалось объединение.
    """
    if margin <= 0:
        return box
    work_h, work_w = work_gray.shape[:2]
    if work_h < 2 or work_w < 2:
        return box

    paper = float(np.percentile(work_gray, GROW_PAPER_PERCENTILE))
    step = max(1, step_px // HALFTONE_DOWNSCALE)
    x1, y1, x2, y2 = (v // HALFTONE_DOWNSCALE for v in box)
    f0, f1, f2, f3 = (v // HALFTONE_DOWNSCALE for v in floor)

    def paperish(strip: np.ndarray) -> bool:
        return strip.size > 0 and float(strip.mean()) >= paper - margin

    for _ in range(max_steps):
        moved = False
        edge = min(x1 + step, f0)
        if edge > x1 and paperish(work_gray[y1:y2, x1:edge]):
            x1, moved = edge, True
        edge = max(x2 - step, f2)
        if edge < x2 and paperish(work_gray[y1:y2, edge:x2]):
            x2, moved = edge, True
        edge = min(y1 + step, f1)
        if edge > y1 and paperish(work_gray[y1:edge, x1:x2]):
            y1, moved = edge, True
        edge = max(y2 - step, f3)
        if edge < y2 and paperish(work_gray[edge:y2, x1:x2]):
            y2, moved = edge, True
        if not moved:
            break

    scale = HALFTONE_DOWNSCALE
    return x1 * scale, y1 * scale, x2 * scale, y2 * scale


def grow_to_paper(
    work_gray: np.ndarray,
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    margin: int = GROW_PAPER_MARGIN,
    step_px: int = GROW_STEP_PX,
    max_steps: int = GROW_MAX_STEPS,
) -> tuple[int, int, int, int]:
    """Расширяет рамку, пока прилегающая полоска заметно темнее бумаги. Мотивировка выше.

    Считается по копии 1/``HALFTONE_DOWNSCALE`` — полного кадра здесь нет и быть не должно,
    а для сравнения средней яркости полоски разрешения копии хватает с запасом.
    """
    height, width = shape
    work_h, work_w = work_gray.shape[:2]
    if work_h < 2 or work_w < 2:
        return box

    paper = float(np.percentile(work_gray, GROW_PAPER_PERCENTILE))
    step = max(1, step_px // HALFTONE_DOWNSCALE)
    x1, y1, x2, y2 = (
        max(0, box[0] // HALFTONE_DOWNSCALE),
        max(0, box[1] // HALFTONE_DOWNSCALE),
        min(work_w, -(-box[2] // HALFTONE_DOWNSCALE)),
        min(work_h, -(-box[3] // HALFTONE_DOWNSCALE)),
    )
    if x2 - x1 < 1 or y2 - y1 < 1:
        return box

    def darker(strip: np.ndarray) -> bool:
        return strip.size > 0 and float(strip.mean()) < paper - margin

    for _ in range(max_steps):
        moved = False
        if x1 - step >= 0 and darker(work_gray[y1:y2, x1 - step : x1]):
            x1 -= step
            moved = True
        if x2 + step <= work_w and darker(work_gray[y1:y2, x2 : x2 + step]):
            x2 += step
            moved = True
        if y1 - step >= 0 and darker(work_gray[y1 - step : y1, x1:x2]):
            y1 -= step
            moved = True
        if y2 + step <= work_h and darker(work_gray[y2 : y2 + step, x1:x2]):
            y2 += step
            moved = True
        if not moved:
            break

    scale = HALFTONE_DOWNSCALE
    return max(0, x1 * scale), max(0, y1 * scale), min(x2 * scale, width), min(y2 * scale, height)


def _fill_colour_page(
    findings: list[Finding],
    width: int,
    height: int,
    page_chroma_spread: float | None,
    chroma_spread_thr: float,
    full_page_color_frac: float,
) -> list[Finding]:
    """Цветная полоса, занятая картинкой почти целиком, -> ОДНА область во весь кадр.

    ЗАЧЕМ. Обложку разрывает то, что само картинкой не является: сплошная плашка заголовка
    растровых клеток не даёт, компонента на ней рвётся, и полоса распадается на куски. Замер
    по 1967/11 IMG_0052_2R: две области, 4.3% и 68.4% полосы, между ними 588 px (2.5 см), а
    красная плашка между ними не попала в разметку вовсе.

    ПОЧЕМУ НЕ ПРОСТО УВЕЛИЧИТЬ ЗАЗОР СЛИЯНИЯ. Зазор в 2.5 см на внутренней полосе — это
    обычный просвет с подписью между двумя РАЗНЫМИ фотографиями, и слить их значило бы
    вернуть дефект из папки «две картинки детектированы как одна большая».

    РАЗДЕЛЯЕТ ЦВЕТ, а не зазор и не размер. Обложка отличается тем, что цветная ЦЕЛИКОМ.
    Замер разброса хроматичности по всей полосе (та же метрика, что решает color/grayscale
    у областей):

        1967/11 IMG_0052_2R   обложка, надо слить          33.7
        1966/03 IMG_0104_2R   обложка 1, контроль          26.4
        1969/12 IMG_0150_2R   обложка 3, две ч/б фото       4.1
        1969/11 IMG_0093_2R   две ч/б фотографии            4.8
        1970/03 IMG_0141_2R   две ч/б фотографии            5.1

    Порог берётся тот же, что у областей (``CHROMA_SPREAD_THR``), — не новая величина.

    Оба условия обязательны, и каждое закрывает свою ошибку. Без проверки площади цветная
    полоса с одной маленькой картинкой и текстом уехала бы в разметку целиком. Без проверки
    цвета слиплись бы две ч/б фотографии, разнесённые по полосе.
    """
    if page_chroma_spread is None or page_chroma_spread <= chroma_spread_thr:
        return findings

    covered = (
        min(f.box[0] for f in findings),
        min(f.box[1] for f in findings),
        max(f.box[2] for f in findings),
        max(f.box[3] for f in findings),
    )
    if (covered[2] - covered[0]) * (covered[3] - covered[1]) < full_page_color_frac * width * height:
        return findings

    fracs = [f.dot_frac for f in findings if f.dot_frac is not None]
    return [
        Finding(
            box=cover_module.cover_region(width, height),
            source=SOURCE_SCREEN if any(f.source == SOURCE_SCREEN for f in findings) else SOURCE_LINEART,
            full_page=True,
            dot_frac=max(fracs) if fracs else None,
            has_cells=any(f.has_cells for f in findings),
        )
    ]


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
