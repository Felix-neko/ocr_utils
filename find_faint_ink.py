"""Поиск полос с пересвеченными буквами и бледными перемычками.

ЗАЧЕМ. Шапка ``run_scripts/scan_markup/pack1/run_7_cleanup.sh`` требует перед полным
прогоном посмотреть глазами, «что защитная маска накрывает бледные перемычки букв».
Проверки не было — пак-1 очищен целиком. Скрипт находит худшие полосы, чтобы было на
что смотреть.

ЧТО ТАКОЕ ПЕРЕСВЕТ И ПОЧЕМУ НЕ ЯРКОСТЬ. Всё считается по ОТРАЖЕНИЮ ``R = g / бумага``
(``ocr_utils.paper``), а не по яркости. Причина записана в докстринге ``paper.py``: на
пересвеченной полосе 1966/01 IMG_0047_2R краска имеет яркость 123-193 при бумаге 255,
то есть СВЕТЛЕЕ, чем просвет с оборота на другой полосе. Рабочая величина —
глубина краски ``D = 1 - R``: бумага 0, чёрный штрих около 1.

ПОЧЕМУ ПО ТАЙЛАМ. Пересвет почти всегда местный: вспышка легла неровно, и выгорел угол
или полстраницы. Балл, усреднённый по всей полосе, такой дефект растворяет. Поэтому
метрики считаются по тайлам, тайлы без набора выбрасываются, а балл полосы собирается
квантилем худших — та же схема, что в ``defocus_detection``.

ПРИЗНАКИ:

* ``ink_depth`` — глубина ТЕЛА штриха в тайле (краска после открытия диском радиуса
  3, то есть без тонких структур и без одиночных тёмных точек), 10-й перцентиль по
  полосе. Главный признак пересвета. Отражение нормировано по бумаге по построению,
  поэтому здесь, в отличие от lapvar, абсолютный порог законен.
* ``clip_frac`` — доля выбитых в 255 пикселей. Подтверждающий: пересвет с клиппингом
  необратим, без клиппинга — просто светлая съёмка.
* ``thin_ratio`` — контраст тонких структур к телу штриха через морфологию, и его
  остаток ``thin_residual`` после снятия зависимости от ``ink_depth``. Ранжируем по
  ОСТАТКУ, а не по самому отношению, — см. :func:`thin_residuals`.
* ``bridge_loss`` и ``break_ratio`` — справочные, в ранжировании НЕ участвуют. Задумывались
  как главные признаки перемычек (щедрая маска против РАБОЧЕЙ, той самой, которой работает
  конвейер очистки), но оба оказались негодны как балл, и это стоит знать, чтобы не
  переизобретать: у ``bridge_loss`` постоянный пол 0.12-0.19 от однопиксельной каймы по
  краям штрихов, которую щедрая маска берёт, а рабочая отсекает — на любой полосе, хоть
  на идеальной; у ``break_ratio`` на части полос текст при щедром пороге слипается в один
  блоб (замер: область в 80 016 px при окне глифа до 20 000), знаменатель схлопывается и
  величина завышается на ровном месте.

ГЛАВНЫЙ ВЫВОД ПО ПАКУ-1, из-за которого признаки устроены именно так. Бледность перемычек
здесь НЕ отдельный дефект: она следует из общей глубины краски. Проверено четырьмя разными
формулировками (отражение на хребте дистанционного преобразования; отношение площадей двух
масок; отношение числа связных областей; отношение «перемычка/тело» ВНУТРИ одной буквы) —
ни одна не дала сигнала, независимого от ``ink_depth``. У последней, самой прямой, отношение
держится в полосе 0.39-0.49 и монотонно растёт с глубиной краски. Физически это разумно:
связка тоньше пятна рассеяния объектива, и её глубина выходит фиксированной долей от тела
штриха. Поэтому список «самых бледных перемычек» — это в первую очередь список самых
пересвеченных полос, а собственный вклад перемычек виден только в остатке.

ЧЕГО ЗДЕСЬ НЕТ И ПОЧЕМУ. Отражения на хребте дистанционного преобразования. Хребет
щедрой маски при малом радиусе состоит из пикселей на самой границе порога, и медиана
``R`` там насыщается у порога: на четырёх пробных полосах разного качества получалось
0.891-0.897. Признак вырождается, мерить им нечего.
"""

import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np
from tqdm import tqdm

from ocr_utils.background_smoothing.processing import METHOD_SAUVOLA, primary_mask
from ocr_utils.defocus_detection.image_io import read_gray
from ocr_utils.defocus_detection.scoring import aggregate, rank_combine
from ocr_utils.defocus_detection.tiles import Grid, detail_rms_map, make_grid, printed_mask
from ocr_utils.paper import disk, reflectance
from ocr_utils.scan_cleanup.protect import is_full_page, rects_mask
from ocr_utils.scan_cleanup.source import PageMarkup, load_markup
from ocr_utils.db.models import KIND_COLOR, KIND_COLOR_TEXT

# --- Размеры набора: пак-1 при 600 dpi -------------------------------------------------
# ВСЕ РАЗМЕРЫ В ПИКСЕЛЯХ И ЗАДАНЫ НАБОРОМ, а не долей кадра. В «Готовое» попадают и
# неполные страницы (если текст занимал треть листа, пустые две трети обрезаны), и доля
# от высоты такого кадра дала бы втрое меньшее окно на том же самом наборе. Замер по
# паку-1 (``defocus_detection.scale``): толщина штриха 7 px, шаг строк 89 px.
# Для другого пака или разрешения ПЕРЕМЕРИТЬ и передать через CLI.

# Сторона тайла. Порядка шести строк набора: мельче — в тайл не попадает достаточно
# букв и перцентили скачут, крупнее — местный пересвет размазывается по тайлу.
TILE_SIZE = 512

# Радиус структурного элемента, который срезает тонкие структуры: всё уже 2*2 px.
# Перемычка при 600 dpi — 3-5 px, тело штриха — 7 px.
THIN_RADIUS = 2
# Радиус, который оставляет только тело штриха (штрих 7 px переживает радиус 3).
BODY_RADIUS = 3

# Доля от глубины ТЕЛА штриха, ниже которой пиксель в щедрую маску уже не берётся.
# Замер на 12 полосах (доли 0.10/0.15/0.20/0.30): при 0.30 щедрая маска оказывается
# СТРОЖЕ рабочей, и bridge_loss уходит в минус на всех полосах подряд — признак
# вырождается. Разброс появляется на 0.10: bridge_loss 0.12-0.25, break_ratio
# 1.00-1.88 при том, что на 0.20 это уже 1.00-1.22, а на 0.30 плоские 1.00-1.04.
GENEROUS_FRAC = 0.10

# Перцентиль «самой тёмной краски». Не максимум: одна пылинка или дырка в бумаге увела
# бы масштаб тайла куда угодно.
INK_PERCENTILE = 99.5

# Глубина тела штриха (в уровнях 8 бит), ниже которой в тайле мерить нечего: это не
# набор, а грязь на поле.
MIN_BODY_PEAK = 20

# Уровень, с которого пиксель считается выбитым в белое.
CLIP_LEVEL = 254

# --- Параметры рабочей маски: строго из run_7_cleanup.sh -------------------------------
# Это не «похожие» настройки, а буквально те, которыми очищен пак. Смысл признаков
# bridge_loss/break_ratio в том, что маска ТА ЖЕ САМАЯ: вопрос не «видна ли перемычка
# в принципе», а «видит ли её конвейер».
WORK_SAUVOLA_WINDOW = 101
WORK_SAUVOLA_K = 0.10
WORK_THRESHOLD_BIAS = 0.5
WORK_INK_LEVEL = 0.65
WORK_MIN_GLYPH_AREA = 34
WORK_SURE_GLYPH_AREA = 500
WORK_PAPER_DILATE_PX = 15
WORK_PAPER_BLUR_PX = 150

# Площадь связной области, при которой она считается глифом или его куском. Нижняя —
# та же, что у конвейера (p99 реального шума), верхняя отсекает плашки и линейки
# таблиц: они не буквы, и их дробление ничего не говорит о перемычках.
GLYPH_AREA_MIN = WORK_MIN_GLYPH_AREA
GLYPH_AREA_MAX = 20000

# Сколько глифов должна найти щедрая маска, чтобы дробление в тайле вообще считалось.
# На десятке областей отношение числа кусков квантуется шагом в десять процентов, и
# худшим тайлом полосы регулярно оказывается пустой угол с парой цифр колонтитула —
# смотреть там нечего, а в отчёт полоса из-за этого попадает.
MIN_GLYPHS = 30

# Доля площади тайла под щедрой маской, ниже которой мерить нечего. Абсолютный порог
# в пикселях (было 500 из 262 144, то есть 0.2%) пропускал тайлы, где «краска» — это
# горизонтальная линейка и крапины просвета с оборота: глубина такой «краски» мала,
# и полоса из-за пары таких тайлов уезжала наверх списка пересвета. Замер по семи
# полосам: у тайлов с настоящим набором доля 0.09-0.27, у линейки с просветом
# 0.024-0.045. Порог поставлен между ними.
MIN_INK_FRAC = 0.06
# Доля тайла, которую разрешено занимать вырезанным областям (растр, заплатки LaMa).
MAX_EXCLUDED_FRAC = 0.05

# Квантиль сведения тайлов в балл полосы: берём десятую часть худших.
TILE_QUANTILE = 0.90

# --- Пул -------------------------------------------------------------------------------
WORKER_NICE = 10
THREAD_LIMIT_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
DEFAULT_JOBS = 16

# Виды растровых областей, из-за которых полоса выбрасывается целиком: цветной растр.
COLOR_KINDS = (KIND_COLOR, KIND_COLOR_TEXT)


@dataclass
class PageResult:
    """Признаки одной полосы. Поля класса — ровно колонки CSV, в том же порядке."""

    rel_path: str
    ink_depth: float = float("nan")
    clip_frac: float = float("nan")
    bridge_loss: float = float("nan")
    break_ratio: float = float("nan")
    thin_ratio: float = float("nan")
    # Заполняется НЕ при обходе, а в отчёте: величина сравнительная, её нельзя посчитать
    # по одной полосе (см. :func:`thin_residuals`).
    thin_residual: float = float("nan")
    # Центр худшего тайла по каждому из двух главных признаков, в пикселях полосы:
    # по нему режется врезка 1:1 для листа проверки.
    exp_x: int = 0
    exp_y: int = 0
    brk_x: int = 0
    brk_y: int = 0
    printed_tiles: int = 0
    tiles: int = 0
    width: int = 0
    height: int = 0
    raster_kinds: str = ""
    mask_kinds: str = ""
    error: str = ""


def pct_u8(block: np.ndarray, q: float) -> float:
    """Перцентиль по массиву uint8 через гистограмму.

    ``np.percentile`` сортирует, и на 91 тайле по 262 тысячи пикселей это заметная доля
    времени полосы. Гистограмма даёт тот же ответ за один линейный проход.

    Args:
        block: Массив uint8.
        q: Перцентиль в процентах.

    Returns:
        Значение в уровнях 8 бит.
    """
    counts = np.bincount(block.ravel(), minlength=256)
    cumulative = np.cumsum(counts)
    return float(np.searchsorted(cumulative, cumulative[-1] * q / 100.0))


def _init_worker() -> None:
    """Инициализация воркера: одно ядро на процесс и пониженный приоритет.

    Без ограничения потоков шестнадцать процессов, каждый со своим OpenMP-пулом на
    шестнадцать нитей, дерутся за те же ядра и считают медленнее одного процесса.
    """
    for name in THREAD_LIMIT_VARS:
        os.environ[name] = "1"
    cv2.setNumThreads(1)
    try:
        os.nice(WORKER_NICE)
    except OSError:
        pass


def excluded_mask(shape: "tuple[int, int]", markup: PageMarkup) -> np.ndarray:
    """Области, которые нельзя мерить: растровые иллюстрации и заплатки LaMa.

    Растр не выбрасывает полосу, но из счёта уходит: растровая точка фотографии сама
    по себе тонкая структура и подделала бы признаки перемычек. Под масками закраса
    лежит искусственная бумага от LaMa — краски там нет по построению.

    Args:
        shape: Размер полосы (высота, ширина).
        markup: Разметка полосы из базы; координаты уже в пикселях оригинала.

    Returns:
        Маска uint8 0/255: 255 — мерить нельзя.
    """
    mask = rects_mask(shape, list(markup.regions))
    for m in markup.masks:
        y1, x1 = max(0, m.top), max(0, m.left)
        y2, x2 = min(shape[0], m.top + m.height), min(shape[1], m.left + m.width)
        if y2 > y1 and x2 > x1:
            mask[y1:y2, x1:x2] = 255
    return mask


def page_masks(
    gray: np.ndarray, d8: np.ndarray, body: np.ndarray, grid: Grid, roi: "np.ndarray | None"
) -> "tuple[np.ndarray, ...]":
    """Щедрая и рабочая маски краски.

    Щедрая строится ПО ТАЙЛАМ, каждый со своим масштабом краски: пересвет местный, и
    один порог на всю полосу означал бы, что в выгоревшем углу маска пуста просто
    потому, что где-то на другом краю полосы краска глубже.

    Рабочая — ровно та, которой работает конвейер очистки (параметры из
    ``run_7_cleanup.sh``), и строится по всей полосе сразу: Савола локальна, а
    глобальный порог считается по области анализа.

    Args:
        gray: Полутоновая полоса.
        d8: Глубина краски в уровнях 8 бит.
        body: Глубина краски после открытия крупным диском — только тело штриха.
        grid: Сетка тайлов.
        roi: Область, по которой считать глобальный порог; None — вся полоса.

    Returns:
        Пара масок uint8 0/255: (щедрая, рабочая).
    """
    generous = np.zeros(gray.shape, np.uint8)
    for iy in range(grid.ny):
        for ix in range(grid.nx):
            y1, y2, x1, x2 = grid.bounds(iy, ix)
            peak = pct_u8(body[y1:y2, x1:x2], INK_PERCENTILE)
            if peak < MIN_BODY_PEAK:
                continue
            generous[y1:y2, x1:x2] = (d8[y1:y2, x1:x2] > peak * GENEROUS_FRAC).astype(np.uint8) * 255

    working = primary_mask(
        gray,
        method=METHOD_SAUVOLA,
        bias=WORK_THRESHOLD_BIAS,
        sauvola_k=WORK_SAUVOLA_K,
        window=WORK_SAUVOLA_WINDOW,
        roi=roi,
        min_glyph_area=WORK_MIN_GLYPH_AREA,
        ink_level=WORK_INK_LEVEL,
        sure_glyph_area=WORK_SURE_GLYPH_AREA,
        paper_dilate_px=WORK_PAPER_DILATE_PX,
        paper_blur_px=WORK_PAPER_BLUR_PX,
        trust_strong=False,
    )
    return generous, working


def glyph_count(mask: np.ndarray) -> int:
    """Сколько в маске связных областей глифового размера.

    Плашки и линейки таблиц отсекаются по верхней границе площади: они не буквы, и их
    дробление ничего не говорит о перемычках.

    Args:
        mask: Маска uint8 0/255.

    Returns:
        Число областей.
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if count <= 1:
        return 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(np.count_nonzero((areas >= GLYPH_AREA_MIN) & (areas <= GLYPH_AREA_MAX)))


def analyze_page(path: Path, markup: PageMarkup, tile_size: int = TILE_SIZE) -> PageResult:
    """Считает все признаки одной полосы.

    Args:
        path: Файл очищенной полосы.
        markup: Разметка из базы (растровые области и маски закраса).
        tile_size: Сторона тайла в пикселях.

    Returns:
        Заполненный :class:`PageResult`; при ошибке — с текстом в ``error``.
    """
    result = PageResult(
        rel_path=markup.rel_path,
        raster_kinds="+".join(sorted({r.kind for r in markup.regions})),
        mask_kinds="+".join(sorted({m.kind for m in markup.masks})),
    )
    gray = read_gray(path)
    if gray is None:
        result.error = "не читается"
        return result
    result.height, result.width = gray.shape[:2]

    excluded = excluded_mask(gray.shape[:2], markup)
    roi = cv2.bitwise_not(excluded) if markup.regions else None

    depth = 1.0 - reflectance(gray, WORK_PAPER_DILATE_PX, WORK_PAPER_BLUR_PX)
    # Морфология по всей полосе разом: тонкое — то, что срезает открытие малым диском,
    # тело — то, что переживает открытие крупным.
    d8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
    body = cv2.morphologyEx(d8, cv2.MORPH_OPEN, disk(BODY_RADIUS))
    thin = cv2.subtract(d8, cv2.morphologyEx(d8, cv2.MORPH_OPEN, disk(THIN_RADIUS)))

    grid = make_grid(gray.shape[:2], tile_size)
    printed = printed_mask(detail_rms_map(gray, grid))
    result.tiles = grid.ny * grid.nx
    generous, working = page_masks(gray, d8, body, grid, roi)

    shape = (grid.ny, grid.nx)
    ink_map = np.full(shape, np.nan)
    clip_map = np.full(shape, np.nan)
    loss_map = np.full(shape, np.nan)
    break_map = np.full(shape, np.nan)
    thin_map = np.full(shape, np.nan)

    for iy in range(grid.ny):
        for ix in range(grid.nx):
            if not printed[iy, ix]:
                continue
            y1, y2, x1, x2 = grid.bounds(iy, ix)
            if excluded[y1:y2, x1:x2].mean() > MAX_EXCLUDED_FRAC * 255:
                printed[iy, ix] = False
                continue
            gen = generous[y1:y2, x1:x2]
            gen_area = int(np.count_nonzero(gen))
            if gen_area < MIN_INK_FRAC * gen.size:
                printed[iy, ix] = False
                continue
            work = working[y1:y2, x1:x2]

            # Глубина ТЕЛА штриха, а не сырой краски: перцентиль по сырой глубине берёт
            # самый тёмный пиксель тайла, а такой находится почти на любой полосе, и
            # признак садится в 0.94-1.00 на всём паке. Тело штриха даёт разброс
            # 0.48-0.90 на той же выборке.
            body_peak = pct_u8(body[y1:y2, x1:x2], INK_PERCENTILE)
            if body_peak < MIN_BODY_PEAK:
                printed[iy, ix] = False
                continue
            ink_map[iy, ix] = body_peak / 255.0
            clip_map[iy, ix] = float(np.mean(gray[y1:y2, x1:x2] >= CLIP_LEVEL))
            work_area = int(np.count_nonzero(work))
            if not work_area:
                continue
            loss_map[iy, ix] = 1.0 - work_area / gen_area
            n_gen = glyph_count(gen)
            break_map[iy, ix] = glyph_count(work) / n_gen if n_gen >= MIN_GLYPHS else np.nan
            thin_map[iy, ix] = pct_u8(thin[y1:y2, x1:x2], INK_PERCENTILE) / body_peak

    result.printed_tiles = int(np.count_nonzero(printed))
    if not result.printed_tiles:
        result.error = "нет тайлов с набором"
        return result

    # Пересвет и тонкость — чем МЕНЬШЕ, тем хуже, поэтому квантиль худших ("worst").
    # Потеря краски и дробление — чем БОЛЬШЕ, тем хуже, поэтому "best".
    result.ink_depth = aggregate(ink_map, printed, "worst", TILE_QUANTILE)
    result.clip_frac = aggregate(clip_map, printed, "median")
    result.bridge_loss = aggregate(loss_map, printed, "best", TILE_QUANTILE)
    result.break_ratio = aggregate(break_map, printed, "best", TILE_QUANTILE)
    result.thin_ratio = aggregate(thin_map, printed, "worst", TILE_QUANTILE)
    result.exp_y, result.exp_x = _worst_tile_centre(ink_map, printed, grid, biggest=False)
    result.brk_y, result.brk_x = _worst_tile_centre(break_map, printed, grid, biggest=True)
    return result


def _worst_tile_centre(tile_map: np.ndarray, printed: np.ndarray, grid: Grid, biggest: bool) -> "tuple[int, int]":
    """Центр худшего измеренного тайла в пикселях полосы — точка, вокруг которой резать врезку.

    Args:
        tile_map: Карта признака по тайлам.
        printed: Маска тайлов с набором.
        grid: Сетка тайлов.
        biggest: True — худший тот, где значение больше; False — где меньше.

    Returns:
        Пара (y, x); центр полосы, если измеренных тайлов нет.
    """
    usable = printed & np.isfinite(tile_map)
    if not usable.any():
        return grid.height // 2, grid.width // 2
    values = np.where(usable, tile_map, -np.inf if biggest else np.inf)
    iy, ix = np.unravel_index(int(np.argmax(values) if biggest else np.argmin(values)), values.shape)
    y1, y2, x1, x2 = grid.bounds(int(iy), int(ix))
    return (y1 + y2) // 2, (x1 + x2) // 2


def _worker(task: "tuple[str, PageMarkup, int]") -> PageResult:
    """Точка входа воркера: путь и разметка внутрь, признаки наружу.

    Картинки через границу процессов не ходят — воркер читает файл сам.
    """
    root, markup, tile_size = task
    try:
        return analyze_page(Path(root) / markup.rel_path, markup, tile_size)
    except Exception as exc:  # noqa: BLE001 — одна битая полоса не должна ронять прогон
        return PageResult(rel_path=markup.rel_path, error=f"{type(exc).__name__}: {exc}")


def select_pages(
    root: Path, db: Path, pack_name: str, report_csv: "Path | None", only_year: "str | None", limit: "int | None"
) -> "tuple[list[PageMarkup], dict[str, int]]":
    """Полосы, годные к замеру, и счётчик отброшенных по каждой причине.

    ЧТО ОТБРАСЫВАЕТСЯ ЦЕЛИКОМ:

    * цветной растр (``color``, ``color_text``) — на такой полосе меряется не набор,
      а иллюстрация, и в отчёте она бесполезна; это и есть просьба «проверить по базе,
      что цветного растра нет»;
    * полосная иллюстрация (``protect.is_full_page``) — мерить нечего, весь кадр
      содержимое;
    * всё, что шаг 7 не размывал (статус в его ``report.csv`` не ``ok``) — такие полосы
      скопированы как есть, и рабочая маска к ним отношения не имеет.

    Серый растр и маски закраса полосу НЕ отбрасывают: их прямоугольники вырезаются
    из области счёта (см. :func:`excluded_mask`).

    Args:
        root: Папка очищенных полос.
        db: База разметки.
        pack_name: Имя пака в базе.
        report_csv: ``report.csv`` шага 7; None — статусы не проверять.
        only_year: Ограничиться одним годом.
        limit: Взять не больше стольких полос (для пробы).

    Returns:
        Пара (список разметок, счётчик причин отбрасывания).
    """
    not_ok: "set[str]" = set()
    if report_csv and report_csv.exists():
        with report_csv.open(encoding="utf-8") as handle:
            not_ok = {row["rel_path"] for row in csv.DictReader(handle) if row["status"] != "ok"}

    dropped = {"цветной растр": 0, "полосная иллюстрация": 0, "не размывалась": 0, "нет файла": 0}
    kept: "list[PageMarkup]" = []
    for markup in load_markup(db, pack_name, only_year=only_year):
        if markup.regions_of(COLOR_KINDS):
            dropped["цветной растр"] += 1
            continue
        if is_full_page(markup):
            dropped["полосная иллюстрация"] += 1
            continue
        if markup.rel_path in not_ok:
            dropped["не размывалась"] += 1
            continue
        if not (root / markup.rel_path).exists():
            dropped["нет файла"] += 1
            continue
        kept.append(markup)
        if limit and len(kept) >= limit:
            break
    return kept, dropped


def write_results(path: Path, results: "list[PageResult]") -> None:
    """Пишет CSV по ВСЕМ полосам со всеми признаками.

    Широкий CSV по всему прогону, а не только по топу, — ради того же, ради чего он
    сделан в ``defocus_detection``: пересортировать и перерезать топ потом, ничего
    не пересчитывая. Полный проход стоит двадцать минут, отчёт — секунды.
    """
    names = [f.name for f in fields(PageResult)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for r in results:
            writer.writerow([getattr(r, n) for n in names])


def read_results(path: Path) -> "list[PageResult]":
    """Читает CSV, записанный :func:`write_results`."""
    types = {f.name: f.type for f in fields(PageResult)}
    out: "list[PageResult]" = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            kwargs = {}
            for name, raw in row.items():
                if name not in types:
                    continue
                if types[name] is float:
                    kwargs[name] = float(raw) if raw else float("nan")
                elif types[name] is int:
                    kwargs[name] = int(raw) if raw else 0
                else:
                    kwargs[name] = raw
            out.append(PageResult(**kwargs))
    return out


# --- Отчёт -----------------------------------------------------------------------------
# Размер врезки. Крупнее строки набора (шаг строк 89 px), но так, чтобы полтора десятка
# врезок влезли на один лист: 620 px по высоте — это примерно семь строк.
CROP_W, CROP_H = 420, 620
# Высота подписи под врезкой.
CAPTION_H = 34
# Наибольшая сторона оверлея: полосу целиком в 600 dpi открывать незачем, а увидеть,
# ГДЕ на полосе потери, по уменьшенной копии можно.
OVERLAY_MAX_SIDE = 2200

EXPOSURE, BRIDGES, COMBINED = "exposure", "bridges", "combined"
TABLE_TITLES = {
    EXPOSURE: "Пересвеченные буквы (глубина краски мала)",
    BRIDGES: "Бледные перемычки СВЕРХ объяснимого пересветом (остаток thin_ratio)",
    COMBINED: "Оба дефекта сразу (сводный ранг)",
}


def measurable(results: "list[PageResult]") -> "list[PageResult]":
    """Полосы, которые удалось измерить."""
    return [r for r in results if not r.error and math.isfinite(r.ink_depth)]


def thin_residuals(good: "list[PageResult]") -> None:
    """Проставляет ``thin_residual`` — бледность перемычек СВЕРХ объяснимой пересветом.

    ПОЧЕМУ НЕ ПРОСТО ``thin_ratio``. На этом паке контраст тонких структур к телу штриха
    почти целиком следует за общей глубиной краски: коэффициент корреляции -0.46, а по
    десяти полосам, размеченным вручную, отношение «перемычка/тело» внутри одной буквы
    держится в узкой полосе 0.39-0.49 и монотонно растёт с глубиной краски (0.588 -> 0.391,
    0.953 -> 0.463). Физически это разумно: связка тоньше пятна рассеяния объектива, и её
    глубина выходит фиксированной долей от тела штриха. Поэтому ``thin_ratio`` сам по себе
    ранжирует не бледность перемычек, а глубину краски — его верхушку занимают полосы с
    ЖИРНЫМ набором (thin 0.021 при ink 0.875).

    Отдельным дефектом перемычки становятся только в остатке: снимаем линейную зависимость
    от ``ink_depth`` и смотрим, у кого тонкие структуры слабее, чем предсказывает его же
    глубина краски. Отрицательный остаток и есть «перемычки хуже, чем должны быть».

    Args:
        good: Измеренные полосы; остаток проставляется им на месте.
    """
    x = np.array([r.ink_depth for r in good], dtype=np.float64)
    y = np.array([r.thin_ratio for r in good], dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 100:
        return
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    for r, xi, yi in zip(good, x, y):
        r.thin_residual = float(yi - (intercept + slope * xi)) if np.isfinite(xi) and np.isfinite(yi) else float("nan")


def rankings(results: "list[PageResult]") -> "dict[str, list[PageResult]]":
    """Три упорядоченных списка: пересвет, перемычки, оба сразу.

    Сводный ранг считается по ``defocus_detection.scoring.rank_combine``: шкалы у
    признаков несопоставимы (глубина краски в долях бумаги, дробление — отношение
    числа областей), а порядок полос — вполне.

    Args:
        results: Измеренные полосы.

    Returns:
        Отображение «имя таблицы -> список от худшего к лучшему».
    """
    good = measurable(results)
    thin_residuals(good)
    # Признаки приведены к «больше = лучше», чтобы сводный ранг читался единообразно.
    combo = rank_combine({"ink_depth": [r.ink_depth for r in good], "thin_residual": [r.thin_residual for r in good]})
    return {
        EXPOSURE: sorted(good, key=lambda r: r.ink_depth),
        # Тай-брейк по потере краски: у полос с одинаковым дроблением хуже та, где
        # рабочая маска потеряла больше.
        BRIDGES: sorted(good, key=lambda r: _last_if_nan(r.thin_residual)),
        COMBINED: [r for _, r in sorted(zip(combo, good), key=lambda pair: _last_if_nan(pair[0]))],
    }


def _last_if_nan(value: float, worst_is_small: bool = True) -> float:
    """Ключ сортировки, при котором неизмеренное уезжает в конец, а не в начало списка.

    Сортировка везде по возрастанию, а «худший» у разных признаков на разных концах
    шкалы, поэтому направление приходится называть явно: иначе полоса, которую нечем
    было измерить, оказывается во главе таблицы худших.

    Args:
        value: Значение признака.
        worst_is_small: True — худшее значение меньше (список идёт как есть);
            False — худшее больше, и вызывающий сам ставит минус перед результатом.

    Returns:
        Значение либо бесконечность нужного знака.
    """
    if math.isfinite(value):
        return value
    return float("inf") if worst_is_small else -float("inf")


def finalists(order: "dict[str, list[PageResult]]", top: int) -> "list[PageResult]":
    """Объединение трёх топов, без повторов, в порядке сводного ранга."""
    wanted = {r.rel_path for name in order for r in order[name][:top]}
    return [r for r in order[COMBINED] if r.rel_path in wanted]


def markdown_report(order: "dict[str, list[PageResult]]", top: int, total: int, root: Path) -> str:
    """Markdown с тремя таблицами.

    Args:
        order: Упорядоченные списки из :func:`rankings`.
        top: Сколько строк показывать в каждой таблице.
        total: Сколько полос всего измерено.
        root: Папка полос — для ссылок.

    Returns:
        Текст отчёта.
    """
    lines = [
        "# Пересвеченные буквы и бледные перемычки",
        "",
        f"Папка: `{root}`",
        "",
        f"Измерено полос: {total}. В каждой таблице — худшие {top}.",
        "",
        "**Признаки, по которым идёт ранжирование.** `ink_depth` — глубина тела штриха в",
        "долях бумаги, меньше = пересвет. `остаток` — на сколько контраст тонких структур",
        "ниже, чем предсказывает глубина краски этой же полосы; отрицательный остаток и",
        "есть «перемычки хуже, чем должны быть» (см. оговорку ниже).",
        "",
        "**Признаки-диагностика, по которым НЕ ранжируем.** `clip_frac` — доля выбитых в",
        "белое пикселей. `thin_ratio` — сырой контраст тонких структур к телу штриха; сам",
        "по себе он ранжирует не перемычки, а глубину краски (корреляция -0.46), поэтому",
        "в таблицу 2 идёт его остаток, а не он. `bridge_loss` — доля краски, которую не",
        "видит рабочая маска конвейера; у него есть постоянный пол в 0.12-0.19 от",
        "однопиксельной каймы по краям штрихов, и как балл он не годится. `break_ratio` —",
        "во сколько раз буквы дробятся при рабочем пороге; на части полос текст при щедром",
        "пороге слипается в один блоб (замер: одна область в 80 016 px при окне глифа до",
        "20 000), знаменатель схлопывается и величина завышается. Обе оставлены в CSV как",
        "справочные.",
        "",
    ]
    for number, name in enumerate((EXPOSURE, BRIDGES, COMBINED), start=1):
        lines += [
            f"## {number}. {TABLE_TITLES[name]}",
            "",
            "| # | ink_depth | clip_frac | thin_ratio | остаток | bridge_loss | break_ratio | растр | полоса |",
            "|--:|--:|--:|--:|--:|--:|--:|---|---|",
        ]
        for rank, r in enumerate(order[name][:top], start=1):
            link = f"[{r.rel_path}](file://{(root / r.rel_path).as_posix()})"
            lines.append(
                f"| {rank} | {r.ink_depth:.3f} | {r.clip_frac:.3f} | {r.thin_ratio:.3f} | "
                f"{r.thin_residual:+.3f} | {r.bridge_loss:.3f} | {r.break_ratio:.2f} | "
                f"{r.raster_kinds or '—'} | {link} |"
            )
        lines.append("")
    return "\n".join(lines)


def _crop(gray: "np.ndarray | None", cy: int, cx: int) -> np.ndarray:
    """Врезка 1:1 вокруг точки; серая заглушка, если картинки нет."""
    if gray is None:
        return np.full((CROP_H, CROP_W), 200, np.uint8)
    y = int(np.clip(cy - CROP_H // 2, 0, max(0, gray.shape[0] - CROP_H)))
    x = int(np.clip(cx - CROP_W // 2, 0, max(0, gray.shape[1] - CROP_W)))
    patch = gray[y : y + CROP_H, x : x + CROP_W]
    if patch.shape != (CROP_H, CROP_W):
        canvas = np.full((CROP_H, CROP_W), 255, np.uint8)
        canvas[: patch.shape[0], : patch.shape[1]] = patch
        return canvas
    return patch


def crop_centres(order: "dict[str, list[PageResult]]", chosen: "list[PageResult]") -> "dict[str, tuple[int, int]]":
    """Вокруг какого дефекта резать врезку каждому финалисту.

    Полоса попадает в финалисты по трём разным спискам, и худший тайл по пересвету —
    не тот же самый, что худший по дроблению. Режем вокруг того, по которому полоса
    стоит выше: иначе на листе оказывается врезка, не имеющая отношения к причине,
    по которой полосу вообще выбрали.

    Args:
        order: Упорядоченные списки из :func:`rankings`.
        chosen: Финалисты.

    Returns:
        Отображение «полоса -> (y, x) центра врезки».
    """
    last = len(order[EXPOSURE])
    exposure_rank = {r.rel_path: i for i, r in enumerate(order[EXPOSURE])}
    bridges_rank = {r.rel_path: i for i, r in enumerate(order[BRIDGES])}
    centres = {}
    for r in chosen:
        by_exposure = exposure_rank.get(r.rel_path, last) <= bridges_rank.get(r.rel_path, last)
        centres[r.rel_path] = (r.exp_y, r.exp_x) if by_exposure else (r.brk_y, r.brk_x)
    return centres


def contact_sheet(
    path: Path,
    results: "list[PageResult]",
    root: Path,
    originals: "Path | None",
    columns: int,
    centres: "dict[str, tuple[int, int]]",
) -> Path:
    """Лист врезок 1:1: по две колонки на полосу — оригинал и результат чистки.

    ПОЧЕМУ 1:1, А НЕ ПРЕВЬЮ ПОЛОСЫ. Перемычка при 600 dpi — это 3-5 пикселей. На
    превью целой полосы её не видно вообще, и смотреть там нечего.

    ПОЧЕМУ ПАРА, А НЕ ОДНА КАРТИНКА. Одна врезка отвечает только «плохо ли сейчас».
    Пара отвечает на вопрос, ради которого всё и затевалось: дефект был в съёмке или
    его доела чистка фона.

    Args:
        path: Куда писать PNG.
        results: Финалисты.
        root: Папка очищенных полос.
        originals: Папка оригиналов; None — колонка «до» не рисуется.
        columns: Сколько полос в ряд.
        centres: Центр врезки для каждой полосы (см. :func:`crop_centres`).

    Returns:
        Путь записанного листа.
    """
    pair_w = CROP_W * (2 if originals else 1) + 8
    cell_h = CROP_H + CAPTION_H
    rows = math.ceil(len(results) / columns)
    sheet = np.full((rows * cell_h, columns * pair_w, 3), 255, np.uint8)

    for index, r in enumerate(results):
        clean = read_gray(root / r.rel_path)
        before = read_gray(originals / r.rel_path) if originals else None
        cy, cx = centres[r.rel_path]
        cells = [_crop(before, cy, cx), _crop(clean, cy, cx)] if originals else [_crop(clean, cy, cx)]
        strip = np.hstack([cv2.cvtColor(c, cv2.COLOR_GRAY2BGR) for c in cells])
        iy, ix = divmod(index, columns)
        y0, x0 = iy * cell_h, ix * pair_w
        sheet[y0 : y0 + CROP_H, x0 : x0 + strip.shape[1]] = strip
        if originals:
            cv2.line(sheet, (x0 + CROP_W, y0), (x0 + CROP_W, y0 + CROP_H), (0, 0, 255), 2)
        caption = f"{r.rel_path}  ink {r.ink_depth:.2f}  brk {r.break_ratio:.2f}  loss {r.bridge_loss:.2f}"
        cv2.putText(
            sheet, caption, (x0 + 4, y0 + CROP_H + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)
    return path


def write_link_dir(root_dir: Path, order: "dict[str, list[PageResult]]", top: int, root: Path) -> int:
    """Папки симлинков на худшие полосы — по одной на таблицу.

    Симлинки нужны потому, что кликать по ссылкам в markdown получается не везде:
    PyCharm и Chrome рендерят превью в Chromium, а он запрещает переход на file://.

    Args:
        root_dir: Корень папки симлинков.
        order: Упорядоченные списки.
        top: Сколько ссылок класть в каждую папку.
        root: Папка полос.

    Returns:
        Сколько симлинков создано.
    """
    made = 0
    for name, items in order.items():
        folder = root_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        for existing in folder.iterdir():
            if not existing.is_symlink():
                raise click.ClickException(f"в {folder} лежит не симлинк: {existing.name}")
            existing.unlink()
        for rank, r in enumerate(items[:top], start=1):
            score = r.ink_depth if name == EXPOSURE else r.thin_residual
            link = folder / f"{rank:02d}_{score:.3f}_{Path(r.rel_path).name}"
            link.symlink_to(root / r.rel_path)
            made += 1
    return made


def write_overlay(path: Path, root: Path, markup: PageMarkup, tile_size: int) -> None:
    """Оверлей: красным — краска, которую щедрая маска видит, а рабочая уже нет.

    Именно это и есть «перемычка ушла под рабочий порог», только показанное на месте.
    Пишется уменьшенным: полосу в 600 dpi листать незачем, вопрос в том, ГДЕ потери —
    по всей полосе или в одном выгоревшем углу.
    """
    gray = read_gray(root / markup.rel_path)
    if gray is None:
        return
    depth = 1.0 - reflectance(gray, WORK_PAPER_DILATE_PX, WORK_PAPER_BLUR_PX)
    d8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
    body = cv2.morphologyEx(d8, cv2.MORPH_OPEN, disk(BODY_RADIUS))
    grid = make_grid(gray.shape[:2], tile_size)
    excluded = excluded_mask(gray.shape[:2], markup)
    roi = cv2.bitwise_not(excluded) if markup.regions else None
    generous, working = page_masks(gray, d8, body, grid, roi)
    lost = (generous > 0) & (working == 0) & (excluded == 0)

    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas[lost] = (0, 0, 255)
    scale = OVERLAY_MAX_SIDE / max(canvas.shape[:2])
    if scale < 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


# --- CLI -------------------------------------------------------------------------------


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli() -> None:
    """Поиск полос с пересвеченными буквами и бледными перемычками."""


@cli.command()
@click.option(
    "--root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Папка очищенных полос.",
)
@click.option(
    "--db", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="База разметки пака."
)
@click.option("--pack-name", required=True, help="Имя пака в базе.")
@click.option(
    "--report-csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="report.csv шага 7: полосы не со статусом ok пропускаются.",
)
@click.option(
    "--out-csv",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Куда писать признаки по всем полосам.",
)
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Процессов счёта.")
@click.option("--tile-size", type=int, default=TILE_SIZE, show_default=True, help="Сторона тайла, пикс.")
@click.option("--only-year", default=None, help="Ограничиться одним годом (проба).")
@click.option(
    "--pages-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Файл со списком rel_path: пересчитать только их (по строке на полосу).",
)
@click.option("--limit", type=int, default=None, help="Взять не больше стольких полос (проба).")
def scan(root, db, pack_name, report_csv, out_csv, jobs, tile_size, only_year, pages_file, limit) -> None:
    """Считает признаки по всем полосам пака и пишет CSV."""
    pages, dropped = select_pages(root, db, pack_name, report_csv, only_year, limit)
    if pages_file:
        wanted = {line.strip() for line in pages_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        pages = [m for m in pages if m.rel_path in wanted]
    click.echo(f"Полос к замеру: {len(pages)}")
    for reason, count in dropped.items():
        if count:
            click.echo(f"  отброшено, {reason}: {count}")
    if not pages:
        raise click.ClickException("мерить нечего")

    tasks = [(str(root), markup, tile_size) for markup in pages]
    workers = max(1, min(jobs, len(tasks)))
    context = get_context("forkserver")
    results: "list[PageResult]" = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
        for result in tqdm(pool.map(_worker, tasks, chunksize=1), total=len(tasks), desc="Анализ", file=sys.stderr):
            results.append(result)

    write_results(out_csv, results)
    failed = [r for r in results if r.error]
    click.echo(f"Записано: {out_csv} ({len(results)} полос, ошибок {len(failed)})")
    for r in failed[:10]:
        click.echo(f"  {r.rel_path}: {r.error}")


@cli.command(name="report")
@click.option(
    "--csv",
    "csv_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV от scan.",
)
@click.option(
    "--root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Папка очищенных полос.",
)
@click.option("--top", type=int, default=30, show_default=True, help="Сколько худших показывать в каждой таблице.")
@click.option(
    "--md", "md_path", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Куда писать markdown-отчёт."
)
@click.option(
    "--sheet", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Куда писать лист врезок 1:1."
)
@click.option("--sheet-columns", type=int, default=2, show_default=True, help="Сколько полос в ряд на листе врезок.")
@click.option(
    "--link-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Папка симлинков на худшие полосы.",
)
@click.option(
    "--overlay-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Папка оверлеев с подсветкой потерянной краски.",
)
@click.option(
    "--originals",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Папка оригиналов: колонка «до чистки» на листе врезок.",
)
@click.option(
    "--db",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="База разметки — нужна только для оверлеев.",
)
@click.option("--pack-name", default=None, help="Имя пака в базе — нужно только для оверлеев.")
@click.option(
    "--tile-size",
    type=int,
    default=TILE_SIZE,
    show_default=True,
    help="Сторона тайла, пикс. Должна совпадать с прогоном scan.",
)
def report_cmd(
    csv_path, root, top, md_path, sheet, sheet_columns, link_dir, overlay_dir, originals, db, pack_name, tile_size
) -> None:
    """Строит отчёт, лист врезок, симлинки и оверлеи по готовому CSV."""
    results = read_results(csv_path)
    order = rankings(results)
    total = len(measurable(results))
    click.echo(f"Измерено полос: {total} из {len(results)}")

    text = markdown_report(order, top, total, root)
    if md_path:
        md_path.write_text(text, encoding="utf-8")
        click.echo(f"Отчёт: {md_path}")
    else:
        click.echo(text)

    chosen = finalists(order, top)
    click.echo(f"Финалистов (объединение трёх топов): {len(chosen)}")
    # Контроль: цветного растра в финалистах быть не должно вовсе — полосы с ним
    # отброшены ещё в scan. Проверяем по колонке CSV, независимо от того фильтра.
    colored = [r.rel_path for r in chosen if any(k in r.raster_kinds.split("+") for k in COLOR_KINDS)]
    if colored:
        raise click.ClickException(f"в финалистах цветной растр: {colored}")
    click.echo("Цветного растра в финалистах нет.")

    if sheet:
        centres = crop_centres(order, chosen)
        click.echo(f"Лист врезок: {contact_sheet(sheet, chosen, root, originals, sheet_columns, centres)}")
    if link_dir:
        click.echo(f"Симлинков: {write_link_dir(link_dir, order, top, root)} в {link_dir}")
    if overlay_dir:
        if not (db and pack_name):
            raise click.ClickException("для оверлеев нужны --db и --pack-name")
        wanted = {r.rel_path for r in chosen}
        markups = {m.rel_path: m for m in load_markup(db, pack_name, only_rel=wanted)}
        for r in tqdm(chosen, desc="Оверлеи", file=sys.stderr):
            markup = markups.get(r.rel_path)
            if markup:
                write_overlay(
                    Path(overlay_dir) / r.rel_path.replace("/", "_").replace(".tif", ".jpg"), root, markup, tile_size
                )
        click.echo(f"Оверлеи: {overlay_dir}")


if __name__ == "__main__":
    cli()
