"""Склейка перекрытых по x связных компонент маски глифов в одну букву (точка «i», кратка «й», оторванная ветвь «к») перед замерами высот."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

# --- Правило склейки (:func:`merge_roots`) ---
# Метка (точка, кратка, оторванный штрих) и её буква перекрываются по x не меньше чем на эту долю ширины
# БОЛЕЕ УЗКОЙ из двух. Слегка налезающие края соседних букв курсива и наклонного набора перекрыты на
# 1–2 px при ширине буквы 5–8 px (доля до 0.3) и не склеиваются; точка над «i», кратка «й», точки «ё»,
# ветвь «к» (1968/05 IMG_0074_1L: ветвь x 807–811 внутри тела x 801–811) лежат над буквой целиком (доля 1).
MERGE_MIN_X_OVERLAP = 0.5
# Метка не выше этой доли высоты буквы-хозяина. Две полноразмерные буквы (обе от икса) не склеиваются
# никогда: ни «р» над «б» соседней строки, ни налезающие буквы курсива; точка «i» — 0.2–0.3 высоты тела,
# кратка «й» — 0.2, ветвь «к» на 150 dpi — 0.3, точка «!» — 0.15 высоты стойки.
MERGE_MARK_SHARE = 0.6
# И не выше этой доли медианной высоты компонент набора (страницы, сегмента, полосы ряда): метка мельче
# обычной буквы. Без этого полноразмерная буква верхнего ряда склеивалась бы с буквой нижнего, у которой
# прилипший обрубок перемычки после ``segment._cut_tall_glyphs`` раздул высоту вдвое (12 px против 22).
MERGE_MARK_MEDIAN_SHARE = 0.6
# Просвет между меткой НАД буквой и её верхом — не больше этой доли высоты буквы. Точка «i», кратка «й»,
# точки «ё» отстоят на 0.1–0.3 икса; дефис строки выше над буквой следующей строки — на шаг строк без
# полутора иксов (≥ 0.6 высоты буквы при шаге 1.8 икса).
MERGE_MAX_GAP_ABOVE = 0.35
# Просвет между меткой ПОД буквой и её низом — не больше этой доли высоты буквы. Под буквой у русского
# набора своих меток нет, кроме точки «!» и «?» (0.1–0.15 высоты стойки) и оторванного Оцу хвоста «д»,
# «у» (просвет 0–1 px); дефис строки ниже под выносным строки выше отстоит на 0.3–0.6 высоты выносного.
MERGE_MAX_GAP_BELOW = 0.2
# Метки обрабатываются пачками по столько штук: попарные матрицы «метка × кандидат в хозяева» в памяти
# не больше пачки на полосу кандидатов по высоте.
MERGE_CHUNK = 256


class GlyphMerge(Enum):
    """Склеивать ли перекрытые по x компоненты маски глифов в одну букву перед замерами высот."""

    # Без склейки: буква — связная компонента, как до v8 разбора (для сравнения «было | стало»).
    NONE = "none"
    # Метка над или под буквой, перекрытая с ней по x, — часть этой буквы (:func:`merge_roots`).
    OVERLAP = "overlap"


# Способ по умолчанию; читается при каждом вызове, стенд может переключить его присваиванием.
DEFAULT_GLYPH_MERGE = GlyphMerge.OVERLAP


def merge_roots(stats: np.ndarray) -> np.ndarray:
    """Номер буквы-хозяина у каждой компоненты: метки над и под буквой получают номер её корня.

    Правило: компонента ``m`` — метка компоненты ``h``, если ``m`` не выше ``MERGE_MARK_SHARE`` высоты ``h`` и
    ``MERGE_MARK_MEDIAN_SHARE`` медианной высоты всех компонент ``stats``,
    их боксы перекрыты по x не меньше чем на ``MERGE_MIN_X_OVERLAP`` ширины более узкой и просвет по
    вертикали не больше ``MERGE_MAX_GAP_ABOVE`` (метка выше середины ``h``) или ``MERGE_MAX_GAP_BELOW``
    (ниже) высоты ``h``. Пересекающиеся по высоте боксы (ветвь «к» внутри тела) имеют отрицательный
    просвет и проходят всегда. Из нескольких хозяев выбирается ближний по просвету, при равенстве —
    с большим перекрытием. Хозяин всегда строго выше метки, поэтому цепочки «метка → хозяин → его
    хозяин» конечны и сходятся к корню — самой высокой букве группы; две полноразмерные буквы через
    общую метку не сливаются: у метки один хозяин.

    Args:
        stats: Статистика компонент как у ``cv2.connectedComponentsWithStats`` (``(count, ≥4)``, строка 0 — фон;
            столбцы ``LEFT, TOP, WIDTH, HEIGHT``).

    Returns:
        Массив ``(count,)``: номер корневой компоненты для каждой (у фона 0, у несклеенной — она сама).
    """
    count = stats.shape[0]
    roots = np.arange(count)
    if count <= 2:
        return roots
    lefts = stats[:, cv2.CC_STAT_LEFT].astype(np.float64)
    tops = stats[:, cv2.CC_STAT_TOP].astype(np.float64)
    widths = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
    heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    rights, bottoms = lefts + widths, tops + heights
    ids = np.arange(1, count)
    max_height = float(heights[1:].max())
    median_height = float(np.median(heights[1:]))
    # Кандидаты в метки: не выше доли самой высокой компоненты (иначе хозяина для них нет) и мельче
    # обычной буквы набора.
    marks = ids[
        (heights[1:] <= MERGE_MARK_SHARE * max_height) & (heights[1:] <= MERGE_MARK_MEDIAN_SHARE * median_height)
    ]
    if marks.size == 0:
        return roots
    # Хозяева — по верху: у хозяина метки верх не выше её верха минус его высота и просвет, и не ниже её
    # низа плюс просвет. Сортировка по верху даёт для пачки меток один отрезок кандидатов.
    by_top = ids[np.argsort(tops[1:], kind="stable")]
    sorted_tops = tops[by_top]
    reach = max(MERGE_MAX_GAP_ABOVE, MERGE_MAX_GAP_BELOW) * max_height
    marks = marks[np.argsort(tops[marks], kind="stable")]
    parent = np.arange(count)
    for start in range(0, marks.size, MERGE_CHUNK):
        chunk = marks[start : start + MERGE_CHUNK]
        lo = int(np.searchsorted(sorted_tops, tops[chunk].min() - max_height - reach, side="left"))
        hi = int(np.searchsorted(sorted_tops, bottoms[chunk].max() + reach, side="right"))
        hosts = by_top[lo:hi]
        if hosts.size == 0:
            continue
        # Попарные меры «метка (строки) × кандидат в хозяева (столбцы)».
        overlap = np.minimum(rights[chunk, None], rights[None, hosts]) - np.maximum(
            lefts[chunk, None], lefts[None, hosts]
        )
        narrower = np.minimum(widths[chunk, None], widths[None, hosts])
        gap = np.maximum(tops[chunk, None], tops[None, hosts]) - np.minimum(bottoms[chunk, None], bottoms[None, hosts])
        # Метка над хозяином или под ним — по серединам боксов; от этого зависит допустимый просвет.
        above = (tops[chunk, None] + bottoms[chunk, None]) < (tops[None, hosts] + bottoms[None, hosts])
        limit = np.where(above, MERGE_MAX_GAP_ABOVE, MERGE_MAX_GAP_BELOW) * heights[None, hosts]
        good = (
            (heights[chunk, None] <= MERGE_MARK_SHARE * heights[None, hosts])
            & (overlap >= MERGE_MIN_X_OVERLAP * narrower)
            & (gap <= limit)
        )
        rows = np.nonzero(good.any(axis=1))[0]
        if rows.size == 0:
            continue
        # Ближний хозяин: наименьший просвет, при равенстве — большее перекрытие (ключ с весом меньше пикселя).
        score = np.where(good, gap - overlap / (2.0 * (overlap.max() + 1.0)), np.inf)
        parent[chunk[rows]] = hosts[np.argmin(score[rows], axis=1)]
    # Корень — по цепочке хозяев (каждый хозяин строго выше своей метки, циклов нет).
    roots = parent.copy()
    while True:
        lifted = parent[roots]
        if np.array_equal(lifted, roots):
            break
        roots = lifted
    return roots


def merged_components(
    binary: np.ndarray, mode: GlyphMerge | None = None
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Связные компоненты маски глифов со склейкой меток с их буквами — замена ``cv2.connectedComponentsWithStats``.

    Пиксели маски не меняются: склейка только перенумеровывает компоненты. Буква-группа получает общую
    метку на карте, общий бокс (объединение боксов частей), сумму площадей и центр масс всей краски.

    Args:
        binary: Маска глифов (ненулевое — краска), любого целого или булева типа.
        mode: Способ склейки; ``None`` — модульное умолчание :data:`DEFAULT_GLYPH_MERGE`.

    Returns:
        ``(число меток, карта меток int32, статистика (count, 5) int32, центры масс (count, 2) float64)`` —
        как у ``cv2.connectedComponentsWithStats(..., 8)``; метка 0 — фон, буквы занумерованы подряд.
    """
    mode = GlyphMerge(DEFAULT_GLYPH_MERGE if mode is None else mode)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    if mode is GlyphMerge.NONE or count <= 2:
        return count, labels, stats, centroids
    roots = merge_roots(stats)
    if np.array_equal(roots, np.arange(count)):
        return count, labels, stats, centroids
    # Новые номера: корни подряд в порядке старых номеров (фон остаётся нулём).
    unique_roots, new_of = np.unique(roots, return_inverse=True)
    new_count = int(unique_roots.size)
    remap = new_of.astype(np.int32)
    lefts = stats[:, cv2.CC_STAT_LEFT].astype(np.int64)
    tops = stats[:, cv2.CC_STAT_TOP].astype(np.int64)
    rights = lefts + stats[:, cv2.CC_STAT_WIDTH]
    bottoms = tops + stats[:, cv2.CC_STAT_HEIGHT]
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    # Бокс группы — объединение боксов частей; площадь — сумма; центр масс — взвешенный площадями.
    new_left = np.full(new_count, np.iinfo(np.int64).max)
    new_top = np.full(new_count, np.iinfo(np.int64).max)
    new_right = np.zeros(new_count, dtype=np.int64)
    new_bottom = np.zeros(new_count, dtype=np.int64)
    np.minimum.at(new_left, remap, lefts)
    np.minimum.at(new_top, remap, tops)
    np.maximum.at(new_right, remap, rights)
    np.maximum.at(new_bottom, remap, bottoms)
    new_area = np.bincount(remap, weights=areas, minlength=new_count)
    weighted_x = np.bincount(remap, weights=centroids[:, 0] * areas, minlength=new_count)
    weighted_y = np.bincount(remap, weights=centroids[:, 1] * areas, minlength=new_count)
    new_stats = np.column_stack(
        [new_left, new_top, new_right - new_left, new_bottom - new_top, new_area.astype(np.int64)]
    ).astype(np.int32)
    new_centroids = np.column_stack([weighted_x, weighted_y]) / np.maximum(new_area, 1.0)[:, None]
    # Фон: статистика и центр — как у cv2 для метки 0 (сама строка фона не сливается ни с кем).
    new_stats[0] = stats[0]
    new_centroids[0] = centroids[0]
    return new_count, remap[labels], new_stats, new_centroids
