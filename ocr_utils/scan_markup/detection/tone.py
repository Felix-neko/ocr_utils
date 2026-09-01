"""Отличает полутоновую печать от штрихового рисунка по тонам и по растровой сетке.

ЗАЧЕМ ЕЩЁ ОДИН ПРИЗНАК. Статистика пятен краски (``dots``) отвечает на вопрос «где на полосе
мелкие однородные пятна», и этого хватает, чтобы ОБВЕСТИ иллюстрацию. Но штриховой рисунок с
плотной штриховкой даёт ровно такие же пятна, и на вопрос «фотография это или чертёж» пятна
не отвечают: единственный имевшийся признак ``component_p99_in_box`` неразрешим — у тёмного
портрета p99 = 4439 против 4220 у штриха.

НИ ОДИН ПРИЗНАК ПО ОТДЕЛЬНОСТИ НЕ РАЗДЕЛЯЕТ. Замер по 31 штриховой области (папка-эталон
«line art детектирован как растр» плюс три новых жалобы) против 75 областей с
подтверждёнными фотографиями, обложки исключены — они идут своим правилом:

    признак                        штрих            фотографии       перекрытие
    dot_frac (уже в базе)          0.60  .. 0.85    от 0.345         потеря 40 из 402
    доля средних тонов             0.032 .. 0.322   0.111 .. 0.568   есть
    энтропия гистограммы           4.245 .. 7.170   5.770 .. 7.721   есть
    выступ пика растровой сетки    1.100 .. 2.156   1.054 .. 49.796  есть

РАЗДЕЛЯЕТ ИХ КОНЪЮНКЦИЯ: каждый признак закрывает слепое пятно двух других.

* Пересвеченную фотографию (1969/01 IMG_0033_2R: средние тона 0.111, энтропия 5.77 — оба
  внутри штрихового диапазона) спасает СЕТКА: выступ пика 4.05, растр виден в спектре.
* Фотографию, у которой сетка смазана и пика нет вовсе (1969/09 IMG_0122_2R и соседние,
  выступ 1.05..1.19), спасают ТОНА: энтропия 6.87..7.08, печать непрерывная.
* Штрих проваливает все три сразу: краска одной плотности на бумаге, между ними почти
  ничего, и никакой периодики.

На выбранных порогах правило ловит 30 штриховых областей из 31 и не задевает ни одной из 75
фотографий. Каждый порог стоит посередине своего зазора: энтропия 6.5 при верхе штриха 6.37 и
ближайшей фотографии 6.66; выступ 3.0 при верхе штриха 2.16 и ближайшей фотографии 4.05.

Замер по всем 594 областям пака-1: правило снимает 52. Шесть — ровно те штриховые рисунки, на
которые жаловались, 33 — уже помеченные ``stamp_suspect``, 12 — серые библиотечные печати и
текстовые полосы, ошибочно взятые за картинки. Единственная «фотография» среди них —
1970/04 IMG_0052_2R, а это синий штриховой рисунок во всю полосу, который по правилам пометки
и должен остаться цветной картинкой.

ЧТО ПРОВЕРЕНО И НЕ РАБОТАЕТ.

* Энтропия ТОЛЬКО ПО КРАСКЕ (без бумаги) — разделяет наоборот: у плотных рисунков она выше
  (7.9), чем у фотографий (от 6.7), потому что вклад дают края штрихов.
* Гистограмма после сильного уменьшения (до 7.5 и до 30 dpi) — не разделяет ничем: ни
  разброс, ни энтропия, ни доля тёмного не расходятся у двух групп (замер по девяти полосам,
  все четыре величины перемешаны).
* Расширение полосы поиска пика до 4..24 px — фотографии без сетки так и не дают пика
  (1.22..1.33), зато штриховка начинает попадать в полосу и штрих поднимается до 2.25.
* Отношение мощности в полосе к медиане всего спектра без сглаживания — бессмысленно: спектр
  падает как 1/f, и полоса всегда ниже фона, отношение < 1 у всех без исключения.

ОДНОЙ ЭТОЙ СВЯЗКИ МАЛО, и это выяснилось на полном прогоне. Три портрета 1975/01
IMG_0048_1L (средние тона 0.179..0.215, энтропия 6.18..6.49, выступ 1.23..1.61) попали внутрь
штрихового облака по всем трём осям сразу. Их спасает четвёртое условие, которое живёт в
``regions.LINEART_MAX_DOT_FRAC``: доля растровых клеток в рамке. У этих портретов она
0.98..1.00, у всего пойманного штриха 0.60..0.85 — фотография заполняет свой прямоугольник
растром сплошь. Отдельным признаком доля клеток не работает (стоила бы 40 настоящих областей
из 402), но как ограничение поверх связки не отнимает ни одной находки.

ИЗВЕСТНЫЙ ПРЕДЕЛ. Плотный перспективный рисунок 1968/12 IMG_0139_2R (стеллаж с баллонами) не
ловится: средние тона 0.322 и энтропия 7.17 — внутри диапазона фотографий. Порог, который
достаёт его, начинает задевать настоящие снимки.

Литература: периодичность AM-растра в спектре — стандартный признак при descreening
(US7365882B2) и при сегментации сканов в MRC (US6360009).
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from ocr_utils.scan_markup.detection.dots import CELL_PX, ScreenParams

# Сторона плитки, по которой ищется пик растровой сетки. 256 px при 600 dpi — ровно две
# клетки статистики пятен, поэтому плитки и клетки нарезаются согласованно и долю краски в
# плитке можно взять из готовых гистограмм, не трогая пиксели второй раз.
TILE_PX = 2 * CELL_PX

# Полоса периодов, в которой ищется пик. 5.5..11 px при 600 dpi — это 55..110 lpi, то есть
# вся линиатура, какая встречается в паке (замер: у полос с явной сеткой пик стоит на
# 7.2..7.4 px, это 81..83 lpi). При другом разрешении полоса масштабируется вместе с ним:
# та же линиатура на скане 300 dpi даёт вдвое меньший период в пикселях.
PERIOD_MIN_PX = 5.5
PERIOD_MAX_PX = 11.0

# Ширина медианного сглаживания радиального профиля, в отсчётах радиуса. Профиль делится на
# своё же сглаживание — только так пик виден: сам спектр падает как 1/f, и без нормировки
# любая полоса высоких частот всегда ниже низкочастотного фона.
PROFILE_SMOOTH = 15

# Плитка, где краски меньше этого, в счёт не идёт: на чистой бумаге пика нет ни у растра, ни
# у штриха, и такая плитка только разбавляла бы медиану.
TILE_MIN_INK_FRAC = 0.02
# Насколько пиксель должен быть темнее бумаги, чтобы считаться краской при этом отборе.
TILE_INK_MARGIN = 25

# Уровни бумаги и краски внутри области берутся как процентили: крайние значения на скане
# даёт пыль и царапины, а не печать.
PAPER_PERCENTILE = 98.0
INK_PERCENTILE = 2.0
# Средними считаются тона в средней половине размаха «краска..бумага». Четверть с каждого
# края отдана самой краске и самой бумаге: у штриха там лежит почти всё.
MID_TONE_MARGIN = 0.25

# Контраст «бумага минус краска», НИЖЕ которого мелкий штрих считается оттиском библиотечной
# печати, а не виньеткой рубрики. Мастика бледная и выцветает, типографская краска чёрная:
# замер по паку-1 дал у 10 оттисков 79..220, у 33 виньеток 231..254.
STAMP_INK_CONTRAST_THR = 225

# Пороги правила «это штрих». Все три должны выполниться разом — см. замеры в шапке модуля.
LINEART_MID_FRAC_THR = 0.25
LINEART_ENTROPY_THR = 6.5
LINEART_SCREEN_PEAK_THR = 3.0


@dataclass(frozen=True)
class ToneMaps:
    """Карты тонов и растровой сетки по полосе. Едет из воркера в родителя.

    ``hist`` — гистограмма яркости на КЛЕТКУ, ``(gh, gw, 256)``; складывая клетки
    прямоугольника, получаем его точную гистограмму, не открывая файл заново.
    ``peak`` и ``ink`` — выступ пика сетки и доля краски на ПЛИТКУ, ``(th, tw)``.

    Размер: при 600 dpi это 48x27x256 чисел, около 1.3 МБ на полосу — против 4 МБ рабочей
    копии 1/4, которая едет через pickle уже сейчас.
    """

    hist: np.ndarray
    peak: np.ndarray
    ink: np.ndarray
    cell_px: int
    tile_px: int


@dataclass(frozen=True)
class ToneStats:
    """Признаки по одному прямоугольнику.

    Первые три решают «растр или штрих»; ``ink_contrast`` (бумага минус краска) в этом
    решении не участвует — он отличает бледный оттиск печати от чёрной виньетки уже ПОСЛЕ
    того, как область признана штрихом.
    """

    mid_frac: float
    entropy: float
    screen_peak: float
    ink_contrast: float = 0.0


def screen_peak(tile: np.ndarray, dpi_scale: float = 1.0) -> float:
    """Выступ пика растровой сетки над сглаженным спектром плитки.

    У полутоновой печати точки лежат периодической решёткой, и в двумерном спектре у неё
    дискретный пик; у текста и штриха его нет, есть только центральный максимум. Величина —
    во сколько раз мощность на частоте сетки выше сглаженного профиля: у явного растра 3..80,
    у штриха 1.1..1.6.

    ``dpi_scale`` — отношение разрешения полосы к ``REFERENCE_DPI``: полоса поиска задана в
    пикселях 600 dpi, а на скане 300 dpi та же линиатура даёт вдвое меньший период.
    """
    side = tile.shape[0]
    window = np.outer(np.hanning(side), np.hanning(side)).astype(np.float32)
    values = tile.astype(np.float32)
    values = values - values.mean()
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(values * window))) ** 2

    grid_y, grid_x = np.mgrid[-side // 2 : side // 2, -side // 2 : side // 2]
    radius = np.clip(np.hypot(grid_y, grid_x).astype(int), 0, side // 2 - 1)
    counts = np.bincount(radius.ravel(), minlength=side // 2)
    profile = np.bincount(radius.ravel(), spectrum.ravel(), side // 2) / np.maximum(counts, 1)

    relative = profile / np.maximum(median_filter(profile, size=PROFILE_SMOOTH, mode="nearest"), 1e-9)
    periods = side / (np.arange(side // 2) + 0.5)
    band = (periods >= PERIOD_MIN_PX * dpi_scale) & (periods <= PERIOD_MAX_PX * dpi_scale)
    return float(np.max(np.where(band, relative, 0.0)))


def tone_maps(gray: np.ndarray, params: ScreenParams) -> ToneMaps:
    """Карты тонов и сетки по ПОЛНОМУ кадру.

    Гистограммы собираются одним ``bincount`` по совмещённому индексу «клетка x тон»: цикл по
    двум тысячам клеток стоил бы дороже самого спектра. Замер на 1969/01 IMG_0029_2R
    (3532x6211): гистограммы 0.07 с, спектр 0.27 с при 0.11 с у нынешнего этапа пятен.
    """
    height, width = gray.shape[:2]
    cell = params.cell_px
    grid_h, grid_w = max(1, height // cell), max(1, width // cell)

    rows = np.clip(np.arange(height) // cell, 0, grid_h - 1)
    cols = np.clip(np.arange(width) // cell, 0, grid_w - 1)
    index = rows[:, None] * grid_w + cols[None, :]
    flat = (index.astype(np.int64) << 8) | gray
    hist = np.bincount(flat.ravel(), minlength=grid_h * grid_w * 256).reshape(grid_h, grid_w, 256)
    hist = hist.astype(np.uint32)

    tile = max(cell, params.cell_px * (TILE_PX // CELL_PX))
    tiles_h, tiles_w = max(1, height // tile), max(1, width // tile)
    peak = np.zeros((tiles_h, tiles_w), np.float32)
    ink = np.zeros((tiles_h, tiles_w), np.float32)

    paper = _percentile_from_hist(hist.sum(axis=(0, 1)), PAPER_PERCENTILE)
    dark = max(0, int(round(paper - TILE_INK_MARGIN)))
    for row in range(tiles_h):
        for col in range(tiles_w):
            y, x = row * tile, col * tile
            patch = gray[y : y + tile, x : x + tile]
            if patch.shape[0] != tile or patch.shape[1] != tile:
                continue
            ink[row, col] = float((patch < dark).mean())
            # Пустую плитку в спектр не гоняем: пика там нет ни у растра, ни у штриха, а
            # преобразование — самая дорогая часть всей карты.
            if ink[row, col] >= TILE_MIN_INK_FRAC:
                peak[row, col] = screen_peak(patch, cell / CELL_PX)
    return ToneMaps(hist, peak, ink, cell, tile)


def tone_stats(maps: ToneMaps, box: tuple[int, int, int, int]) -> ToneStats:
    """Три признака по прямоугольнику ``box`` в координатах ОРИГИНАЛА.

    Границы округляются до клетки и до плитки — карты собраны по ним. При клетке 128 px это
    сдвиг на полсантиметра оригинала, а признаки объёмные (доли и энтропия по всей области),
    и такой сдвиг им ничего не делает.
    """
    grid_h, grid_w = maps.hist.shape[:2]
    r0, r1, c0, c1 = _slice(box, maps.cell_px, grid_h, grid_w)
    hist = maps.hist[r0:r1, c0:c1].sum(axis=(0, 1)).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return ToneStats(0.0, 0.0, 0.0, 0.0)

    paper = _percentile_from_hist(hist, PAPER_PERCENTILE)
    ink = _percentile_from_hist(hist, INK_PERCENTILE)
    span = max(paper - ink, 1.0)
    lo = int(np.ceil(ink + MID_TONE_MARGIN * span))
    hi = int(np.floor(paper - MID_TONE_MARGIN * span))
    mid_frac = float(hist[lo : hi + 1].sum() / total) if hi >= lo else 0.0

    share = hist / total
    share = share[share > 0]
    entropy = float(-np.sum(share * np.log2(share)))

    tiles_h, tiles_w = maps.peak.shape
    t0, t1, u0, u1 = _slice(box, maps.tile_px, tiles_h, tiles_w)
    peaks = maps.peak[t0:t1, u0:u1]
    inked = maps.ink[t0:t1, u0:u1] >= TILE_MIN_INK_FRAC
    peak = float(np.median(peaks[inked])) if inked.any() else 0.0
    return ToneStats(mid_frac, entropy, peak, span)


def looks_like_line_art(
    stats: ToneStats,
    mid_frac_thr: float = LINEART_MID_FRAC_THR,
    entropy_thr: float = LINEART_ENTROPY_THR,
    screen_peak_thr: float = LINEART_SCREEN_PEAK_THR,
) -> bool:
    """Штрих ли это. Все три условия обязательны — по отдельности ни одно не разделяет."""
    return stats.mid_frac < mid_frac_thr and stats.entropy < entropy_thr and stats.screen_peak < screen_peak_thr


def _slice(box: tuple[int, int, int, int], step: int, rows: int, cols: int) -> tuple[int, int, int, int]:
    """Прямоугольник оригинала -> диапазон клеток (или плиток), не пустой никогда."""
    c0 = int(np.clip(box[0] // step, 0, cols - 1))
    r0 = int(np.clip(box[1] // step, 0, rows - 1))
    c1 = int(np.clip(-(-box[2] // step), c0 + 1, cols))
    r1 = int(np.clip(-(-box[3] // step), r0 + 1, rows))
    return r0, r1, c0, c1


def _percentile_from_hist(hist: np.ndarray, percentile: float) -> float:
    """Процентиль яркости по готовой гистограмме на 256 корзин."""
    total = hist.sum()
    if total <= 0:
        return 0.0
    cumulative = np.cumsum(hist)
    return float(np.searchsorted(cumulative, total * percentile / 100.0))
