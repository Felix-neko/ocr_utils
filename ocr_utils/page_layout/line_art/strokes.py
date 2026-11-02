"""Добор отдельных штрихов рисунка к рамке line art: лучи, линии движения, дым, капли, точки знака вопроса, штрихи тени — мельче ворот пятна и в рамку не попадали.

ЗАЧЕМ. Рамка line art — охват пиксельных кандидатов (связное пятно, прошедшее ворота размера ``features``) и подсказок;
слияние (``merge_boxes``) соединяет только кандидатов. Отдельные штрихи рисунка ворот не проходят никогда, и рамка
оставляла их снаружи: на 37 размеченных рамках ХиП 1991 (стенд ``research/line_art_boxes``, ``sets/hp_1991_gt.csv``)
вне рамки в среднем 1.78 % краски рисунка, у 18 рамок — больше 1 %. В ``scan_cropping.v2`` (выход страниц) такие
штрихи разбор текста принимал за строки — опоры поля по строкам у рисунка.

КАК. К рамке вида рисунка (обе стороны от ``COVER_MIN_SIDE_MM``, вытянутость до ``COVER_MAX_ASPECT``) добирается любая
не-текстовая компонента краски в ``RING_REACH_MM`` от неё (кругами, ``RING_ROUNDS``, каждая сторона — не дальше
``RING_CAP_MM``). Не добираются: буквы строк (:func:`is_text_component`), дефисы и запятые, ряды одинаковых мелких
знаков (звёздочки «* * *»), волоски в пиксель, полосы тени у края картинки, компоненты в растре и таблицах. На стенде:
вне рамки 1.78 % → 0.56 %, рамок с потерей больше 1 % — 18 → 3; лишних букв строк у рисунков — 0; пак-1 с surya —
21 страница из 750 со сменой рамки, 1 захваченная буква.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.line_art.features import LETTER_MAX_HEIGHT_MM, letter_boxes, params_for_dpi
from ocr_utils.page_layout.tables.detector import EDGE_STRIP_MM
from ocr_utils.units import mm2_to_px2, mm_to_px

# --- Какие рамки дорастают ----------------------------------------------------------------------------------------
# Только рамки вида рисунка: обе стороны от COVER_MIN_SIDE_MM и вытянутость не больше COVER_MAX_ASPECT. Ложные рамки на
# кусках таблиц и штампах поверх оглавления плоские (105×10–23 мм) — их рост захватывал до 71 буквы строки (ХиП 1991/02
# IMG_1111_L); орнамент рубрики «КАК ЭТО РЕШИТЬ» — 10×12 мм, проходит.
COVER_MIN_SIDE_MM = 8.0
COVER_MAX_ASPECT = 3.0

# --- Какие компоненты добираются ---------------------------------------------------------------------------------
# Зазор от компоненты до рамки: точки знака вопроса — 1–2 мм от рамки (03/IMG_0144_R), линии движения шара — 2–4 мм
# (02/IMG_1175_R). Кругов RING_ROUNDS (луч за лучом), каждая сторона — не дальше RING_CAP_MM от исходной.
RING_REACH_MM = 4.0
RING_ROUNDS = 3
RING_CAP_MM = 15.0
# Размер по большей стороне: мельче RING_MIN_SIDE_MM — пыль и сор бумаги, крупнее RING_MAX_SIDE_MM — уже не штрих
# рисунка, а соседний объект или линейка колонки. Мельче RING_MIN_AREA_MM2 — крупинки сора (у 01/IMG_1050_R по
# кругам добирались крупинки в 2–5 px при 150 dpi).
RING_MIN_SIDE_MM = 0.8
RING_MAX_SIDE_MM = 25.0
RING_MIN_AREA_MM2 = 0.2
# Волосок тоньше этого поперёк — край тени или сор (обрывки тени края листа у 01/IMG_1050_R — 1 px при 150 dpi).
RING_MIN_MINOR_MM = 0.3
# Дефис, тире переноса, запятая: плоская короткая чёрточка (не выше DASH_MAX_HEIGHT_MM и не длиннее DASH_MAX_LENGTH_MM)
# — знак строки у рисунка, а не луч (по ним рамка спускалась на строку текста: 01/IMG_1013_L, 1004_R, 05/IMG_0325_L);
# 0.9 мм — и плоский обломок засечки буквы заголовка (05/IMG_0274_L: 2.4×0.85 мм тянул рамку на строку ниже).
DASH_MAX_HEIGHT_MM = 0.9
DASH_MAX_LENGTH_MM = 2.5
# Чёрточка плоская: длина не меньше DASH_MIN_ASPECT высоты; точка знака вопроса (~0.9 мм круглая) — не чёрточка.
DASH_MIN_ASPECT = 1.5
# Полоса тени у края картинки может отстоять от края на пиксель-другой (06/IMG_0475_R: 1–2 px при 150 dpi).
EDGE_NEAR_MM = 0.5

# --- Буквы строк ---------------------------------------------------------------------------------------------------
# Компонента — буква строки, если рядом не меньше ROW_MIN_LETTERS букв, накрывающих её центр по высоте, в пределах
# max(высота компоненты, ROW_REACH_MM) по горизонтали; или одна такая, а компонента не ниже ROW_LETTER_SHARE её высоты
# (буква при разгонке строки: «о» в 4 мм от соседки, 05/IMG_0325_L); или одна вплотную — до ROW_TOUCH_MM (палочка
# «Ы» заголовка, 06/IMG_0419_L).
ROW_REACH_MM = 4.0
ROW_MIN_LETTERS = 2
ROW_LETTER_SHARE = 0.6
ROW_TOUCH_MM = 1.0
# Надпись рисунка — буква внутри рамки дальше этого от её кромки (см. :func:`page_ink`).
LABEL_MARGIN_MM = 2.0

# --- Колонка текста рядом с рамкой — граница роста стороны ----------------------------------------------------------
# Ряд — не меньше TEXT_ROW_MIN_LETTERS букв на одной базовой линии (низ в пределах TEXT_ROW_BASELINE_SHARE высоты
# буквы, не меньше 1 px; высоты — не больше чем вдвое разные), соседние — не дальше TEXT_ROW_GAP_MM (разгонка строк
# пака-1 — до 4.5 мм). Сторона рамки не растёт на полосу, где лежат буквы не меньше TEXT_COLUMN_MIN_ROWS строк (рядов
# с разными базовыми линиями):
# это колонка текста (пак-1 1970/02 с.57: рамка «Контейнер ОБ-1» добрала первую букву строки в 4.4 мм от
# следующей и через неё — начала 13 строк соседней колонки). Подписи рисунка — числа оси, легенда графика (1–2 ряда)
# — добираются по-прежнему.
TEXT_ROW_MIN_LETTERS = 3
TEXT_ROW_BASELINE_SHARE = 0.2
TEXT_ROW_GAP_MM = 6.0
TEXT_COLUMN_MIN_ROWS = 3
# Ряды с базовыми линиями ближе этого — одна строка (числа оси под графиком через широкие пробелы — одна строка).
TEXT_LINE_GAP_MM = 1.0

# --- Ряды мелких знаков ----------------------------------------------------------------------------------------------
# Звёздочки «* * *», отточие, точки строки: компактная (вытянутость до MARK_MAX_ASPECT) компонента до MARK_MAX_MM с не
# меньше чем MARK_MIN_ROW соседями той же формы (ширина и высота в пределах MARK_SIMILAR раз) на той же высоте в
# пределах MARK_GAP_MM — знак строки (ряд звёздочек под слоном 01/IMG_1050_R уводил рамку на 10 мм вниз). Травинки и
# штрихи тени вытянуты, лучи солнца — разной длины и наклона: рядом они не считаются.
MARK_MAX_MM = 3.0
MARK_MAX_ASPECT = 2.0
MARK_SIMILAR = 1.4
MARK_GAP_MM = 8.0
MARK_MIN_ROW = 2


@dataclass(frozen=True)
class PageInk:
    """Компоненты краски полосы для добора к рамкам.

    Поля:
        spots: Рамки компонент ``(N, 4)``: ``x0, y0, x1, y1``, px.
        areas: Площадь краски компонент ``(N,)``, px.
        text: Компонента — буква строки (:func:`is_text_component`); считается только рядом с рамками, дальше — ``True``,
            то есть «не добирать».
        edge: Компонента — полоса тени у края картинки.
        rows: Буквы рядов текста рядом с рамками ``(M, 5)``: ``x0, y0, x1, y1, номер ряда`` (:func:`text_rows`) —
            граница роста сторон (:func:`grow_strokes`).
    """

    spots: np.ndarray
    areas: np.ndarray
    text: np.ndarray
    edge: np.ndarray
    rows: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), np.int64))


def is_text_component(box: tuple[int, int, int, int], letters: np.ndarray, dpi: int) -> bool:
    """Компонента — часть строки текста (для добора — с запасом): см. константы ``ROW_*``.

    Мягче правила A′ (``features.letters_in_row``): тире и дефис в конце строки у рисунка, капитель и курсив тоже
    текст. Луч, направленный в строку, тоже может оказаться «текстом» — его просто не доберут.

    Args:
        box: Рамка компоненты ``(x0, y0, x1, y1)``, px.
        letters: Рамки букв полосы (``features.letter_boxes``), без букв внутри рамок рисунков.
        dpi: Разрешение.

    Returns:
        ``True`` — компонента принадлежит строке текста.
    """
    x0, y0, x1, y1 = box
    # Выше буквы — не буква (линии движения у шара 02/IMG_1175_R — 11–14 мм).
    if len(letters) == 0 or y1 - y0 > mm_to_px(LETTER_MAX_HEIGHT_MM, dpi):
        return False
    centre = (y0 + y1) / 2
    reach = max(y1 - y0, mm_to_px(ROW_REACH_MM, dpi))
    # Сама компонента может быть в списке букв — её не считаем.
    same = (letters[:, 0] == x0) & (letters[:, 1] == y0) & (letters[:, 2] == x1) & (letters[:, 3] == y1)
    in_row = (letters[:, 1] <= centre) & (letters[:, 3] >= centre) & ~same
    gap = np.maximum(0, np.maximum(x0 - letters[:, 2], letters[:, 0] - x1))
    near = in_row & (gap <= reach)
    if int(np.count_nonzero(near)) >= ROW_MIN_LETTERS:
        return True
    # Одна буква строки рядом, а компонента сама ростом с букву.
    heights = letters[:, 3] - letters[:, 1]
    if np.any(near & (y1 - y0 >= ROW_LETTER_SHARE * heights)):
        return True
    # Часть буквы: вплотную к букве строки.
    return bool(np.any(in_row & (gap <= mm_to_px(ROW_TOUCH_MM, dpi))))


def text_rows(letters: np.ndarray, dpi: int) -> np.ndarray:
    """Ряды букв на общей базовой линии: буквы, сцепленные по базовой линии и зазору, группами от TEXT_ROW_MIN_LETTERS.

    Args:
        letters: Рамки букв ``(M, 4)`` (``features.letter_boxes``), px.
        dpi: Разрешение.

    Returns:
        Буквы, вошедшие в ряды, ``(K, 5)``: ``x0, y0, x1, y1, номер ряда``.
    """
    count = len(letters)
    if count < TEXT_ROW_MIN_LETTERS:
        return np.zeros((0, 5), np.int64)
    gap_px = mm_to_px(TEXT_ROW_GAP_MM, dpi)
    heights = np.maximum(1, letters[:, 3] - letters[:, 1])
    # Союз-поиск по парам букв одной строки: близкая базовая линия, похожая высота, зазор по горизонтали.
    parent = np.arange(count)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = np.argsort(letters[:, 0])
    for a_pos, a in enumerate(order):
        for b in order[a_pos + 1 :]:
            gap = letters[b, 0] - letters[a, 2]
            if gap > gap_px:
                break  # дальше по x — только дальше
            tolerance = max(1.0, TEXT_ROW_BASELINE_SHARE * min(heights[a], heights[b]))
            if abs(letters[a, 3] - letters[b, 3]) > tolerance:
                continue
            if max(heights[a], heights[b]) > 2 * min(heights[a], heights[b]):
                continue
            parent[root(a)] = root(b)
    roots = np.array([root(i) for i in range(count)])
    ids, sizes = np.unique(roots, return_counts=True)
    keep = np.isin(roots, ids[sizes >= TEXT_ROW_MIN_LETTERS])
    return np.concatenate([letters[keep], roots[keep, None]], axis=1).astype(np.int64)


def page_ink(bitonal: np.ndarray, dpi: int, boxes: list[Box]) -> PageInk:
    """Компоненты краски полосы с признаками «буква строки» и «полоса у края».

    Args:
        bitonal: Битональная копия (краска 0).
        dpi: Её разрешение.
        boxes: Рамки рисунков: буквенность считается только у компонент рядом с ними (проверка строки дорогая), а
            буквы внутри них — надписи рисунка — соседями строк не считаются.

    Returns:
        :class:`PageInk`.
    """
    ink = (bitonal == 0).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    body = stats[1:]
    x, y = body[:, cv2.CC_STAT_LEFT], body[:, cv2.CC_STAT_TOP]
    spots = np.stack([x, y, x + body[:, cv2.CC_STAT_WIDTH], y + body[:, cv2.CC_STAT_HEIGHT]], axis=1).astype(np.int64)
    letters = letter_boxes(body, params_for_dpi(dpi))
    # Ряды текста рядом с рамками — по всем буквам (и внутри рамок): граница роста сторон.
    reach_rows = mm_to_px(RING_REACH_MM + RING_CAP_MM + TEXT_ROW_GAP_MM, dpi)
    around = np.zeros(len(letters), bool)
    for b in boxes:
        around |= ((letters[:, 2] >= b.x0 - reach_rows) & (letters[:, 0] <= b.x1 + reach_rows)
                   & (letters[:, 3] >= b.y0 - reach_rows) & (letters[:, 1] <= b.y1 + reach_rows))  # fmt: skip
    rows = text_rows(letters[around], dpi)
    # Буквы внутри рамок рисунков — надписи рисунка («СТРОЙ-МАТЕРИАЛЫ» на шаре 02/IMG_1175_R): штрих рядом с ними —
    # часть рисунка, а не буква их «строки».
    # Буквы у самой кромки рамки (ближе LABEL_MARGIN_MM) остаются соседями: ложная рамка на слипшихся буквах заголовка
    # стоит в ряду этих букв, и без них соседние буквы заголовка добирались бы как «штрихи» (05/IMG_0274_L).
    margin = mm_to_px(LABEL_MARGIN_MM, dpi)
    for b in boxes:
        inside = (letters[:, 0] >= b.x0 + margin) & (letters[:, 1] >= b.y0 + margin)
        inside &= (letters[:, 2] <= b.x1 - margin) & (letters[:, 3] <= b.y1 - margin)
        letters = letters[~inside]
    # Буквенность — только у компонент в досягаемости добора от какой-нибудь рамки.
    reach = mm_to_px(RING_REACH_MM + RING_CAP_MM, dpi)
    near = np.zeros(len(spots), bool)
    for b in boxes:
        near |= ((spots[:, 2] >= b.x0 - reach) & (spots[:, 0] <= b.x1 + reach) & (spots[:, 3] >= b.y0 - reach)
                 & (spots[:, 1] <= b.y1 + reach))  # fmt: skip
    text = np.ones(len(spots), bool)
    for i in np.flatnonzero(near):
        text[i] = is_text_component(tuple(int(v) for v in spots[i]), letters, dpi)
    # Полосы тени у края картинки (обрез, сгиб у корешка) — не штрих (как ``tables.detector.edge_strips``).
    height, width = ink.shape
    strip = mm_to_px(EDGE_STRIP_MM, dpi)
    gap = mm_to_px(EDGE_NEAR_MM, dpi)
    w, h = spots[:, 2] - spots[:, 0], spots[:, 3] - spots[:, 1]
    edge = (((spots[:, 0] <= gap) | (spots[:, 2] >= width - gap)) & (w <= strip)) | (
        ((spots[:, 1] <= gap) | (spots[:, 3] >= height - gap)) & (h <= strip)
    )
    return PageInk(spots, body[:, cv2.CC_STAT_AREA], text, edge, rows)


def mark_rows(spots: np.ndarray, candidates: np.ndarray, dpi: int) -> np.ndarray:
    """Какие мелкие компактные компоненты стоят в ряду знаков той же формы (звёздочки, отточие, точки строки).

    Args:
        spots: Рамки компонент ``(N, 4)``.
        candidates: Какие компоненты проверять (остальные — ``False``).
        dpi: Разрешение.

    Returns:
        Маска ``(N,)``: у компоненты не меньше ``MARK_MIN_ROW`` соседей той же формы на той же высоте.
    """
    out = np.zeros(len(spots), bool)
    width, height = spots[:, 2] - spots[:, 0], spots[:, 3] - spots[:, 1]
    major = np.maximum(width, height)
    compact = major <= MARK_MAX_ASPECT * np.maximum(1, np.minimum(width, height))
    idx = np.flatnonzero(candidates & compact & (major <= mm_to_px(MARK_MAX_MM, dpi)))
    if len(idx) == 0:
        return out
    centre_y = (spots[:, 1] + spots[:, 3]) / 2
    gap_px = mm_to_px(MARK_GAP_MM, dpi)
    # Соседями могут быть любые компоненты не крупнее двух предельных знаков.
    others = np.flatnonzero(major <= 2 * mm_to_px(MARK_MAX_MM, dpi))
    log_similar = np.log(MARK_SIMILAR)
    for i in idx:
        j = others[others != i]
        same_height = np.abs(centre_y[j] - centre_y[i]) <= max(height[i], 3)
        # Та же форма: ширина и высота в пределах MARK_SIMILAR раз.
        similar = (np.abs(np.log(np.maximum(1, width[j]) / max(1, width[i]))) <= log_similar) & (
            np.abs(np.log(np.maximum(1, height[j]) / max(1, height[i]))) <= log_similar
        )
        gap = np.maximum(0, np.maximum(spots[i, 0] - spots[j, 2], spots[j, 0] - spots[i, 2]))
        out[i] = int(np.count_nonzero(same_height & similar & (gap <= gap_px))) >= MARK_MIN_ROW
    return out


def ring_addable(ink: PageInk, dpi: int) -> np.ndarray:
    """Какие компоненты годятся в добор: не буквы, не полосы у края, не сор, не дефисы, не ряды знаков.

    Args:
        ink: Компоненты полосы.
        dpi: Разрешение.

    Returns:
        Маска ``(N,)``.
    """
    w, h = ink.spots[:, 2] - ink.spots[:, 0], ink.spots[:, 3] - ink.spots[:, 1]
    major, minor = np.maximum(w, h), np.minimum(w, h)
    sized = (major >= mm_to_px(RING_MIN_SIDE_MM, dpi)) & (major <= mm_to_px(RING_MAX_SIDE_MM, dpi))
    sized &= ink.areas >= mm2_to_px2(RING_MIN_AREA_MM2, dpi)
    sized &= minor >= mm_to_px(RING_MIN_MINOR_MM, dpi)
    # Дефис или запятая: плоская короткая чёрточка.
    flat = (h <= mm_to_px(DASH_MAX_HEIGHT_MM, dpi)) & (w <= mm_to_px(DASH_MAX_LENGTH_MM, dpi))
    sized &= ~(flat & (w >= DASH_MIN_ASPECT * h))
    return sized & ~ink.text & ~ink.edge & ~mark_rows(ink.spots, sized, dpi)


def drawing_shaped(box: Box, dpi: int) -> bool:
    """Рамка вида рисунка: обе стороны не меньше ``COVER_MIN_SIDE_MM`` и вытянутость не больше ``COVER_MAX_ASPECT``.

    Args:
        box: Рамка, px.
        dpi: Разрешение.

    Returns:
        ``True`` — к рамке добираются штрихи.
    """
    small, large = sorted((box.width, box.height))
    return small >= mm_to_px(COVER_MIN_SIDE_MM, dpi) and large <= COVER_MAX_ASPECT * max(1, small)


def column_rows(rows: np.ndarray, before: Box, after: Box, line_gap: float = 1.0) -> int:
    """Сколько строк текста (разных базовых линий рядов) задевает прирост рамки ``before`` → ``after``.

    Args:
        rows: Буквы рядов ``(K, 5)`` (:func:`text_rows`).
        before: Рамка до добора компоненты.
        after: Рамка после.
        line_gap: Базовые линии ближе этого, px, — одна строка.

    Returns:
        Наибольшее по сторонам число строк, буквы которых (центром) лежат в полосе прироста стороны (полоса — на всю
        длину стороны новой рамки).
    """
    if not len(rows):
        return 0
    cx, cy = (rows[:, 0] + rows[:, 2]) / 2, (rows[:, 1] + rows[:, 3]) / 2
    strips = []
    if after.x1 > before.x1:
        strips.append((before.x1, after.y0, after.x1, after.y1))
    if after.x0 < before.x0:
        strips.append((after.x0, after.y0, before.x0, after.y1))
    if after.y1 > before.y1:
        strips.append((after.x0, before.y1, after.x1, after.y1))
    if after.y0 < before.y0:
        strips.append((after.x0, after.y0, after.x1, before.y0))
    worst = 0
    for x0, y0, x1, y1 in strips:
        hit = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
        if not hit.any():
            continue
        # Ряды в одну линию (числа оси, слова подписи с широкими пробелами) — одна строка; колонка — строки друг под
        # другом: считаются разные базовые линии (медиана низа букв ряда, ближе TEXT_ROW_GAP_LINE_PX — одна линия).
        baselines = sorted(float(np.median(rows[rows[:, 4] == r, 3])) for r in np.unique(rows[hit, 4]))
        lines = 1 + int(np.count_nonzero(np.diff(baselines) > line_gap)) if baselines else 0
        worst = max(worst, lines)
    return worst


def grow_strokes(
    box: Box, spots: np.ndarray, addable: np.ndarray, dpi: int, barriers: list[Box] = (), rows: np.ndarray | None = None
) -> tuple[Box, int]:
    """Добрать к рамке рисунка годные компоненты в ``RING_REACH_MM`` от неё (кругами, каждая сторона до ``RING_CAP_MM``).

    Компонента не добирается, если прирост рамки задел бы буквы не меньше ``TEXT_COLUMN_MIN_ROWS`` рядов текста — сторона
    упёрлась в колонку (:func:`column_rows`).

    Args:
        box: Рамка рисунка, px.
        spots: Рамки всех компонент полосы ``(N, 4)``.
        addable: Маска ``(N,)``: компонента годится в добор (:func:`ring_addable`).
        dpi: Разрешение.
        barriers: Рамки, в которые добор не заходит (растр, таблицы).
        rows: Буквы рядов текста (:func:`text_rows`) или None — без границы колонки.

    Returns:
        ``(новая рамка, сколько компонент добрано)``.
    """
    reach = mm_to_px(RING_REACH_MM, dpi)
    cap = mm_to_px(RING_CAP_MM, dpi)
    rows = np.zeros((0, 5), np.int64) if rows is None else rows
    line_gap = mm_to_px(TEXT_LINE_GAP_MM, dpi)
    fit = addable.copy()
    # Компоненты, задевающие растр или таблицу, не добираются.
    for barrier in barriers:
        fit &= ~((spots[:, 0] < barrier.x1) & (spots[:, 2] > barrier.x0) & (spots[:, 1] < barrier.y1)
                 & (spots[:, 3] > barrier.y0))  # fmt: skip
    x0, y0, x1, y1 = box.as_tuple()
    limit = (x0 - cap, y0 - cap, x1 + cap, y1 + cap)
    within = (
        (spots[:, 0] >= limit[0]) & (spots[:, 1] >= limit[1]) & (spots[:, 2] <= limit[2]) & (spots[:, 3] <= limit[3])
    )
    taken = np.zeros(len(spots), bool)
    for _ in range(RING_ROUNDS):
        # Зазор от компоненты до текущей рамки по каждой оси (0 — перекрываются по этой оси).
        gap_x = np.maximum(0, np.maximum(x0 - spots[:, 2], spots[:, 0] - x1))
        gap_y = np.maximum(0, np.maximum(y0 - spots[:, 3], spots[:, 1] - y1))
        inside = (spots[:, 0] >= x0) & (spots[:, 1] >= y0) & (spots[:, 2] <= x1) & (spots[:, 3] <= y1)
        new = fit & within & ~inside & ~taken & (np.maximum(gap_x, gap_y) <= reach)
        if not new.any():
            break
        grown = False
        # По одной компоненте, ближние — первыми: каждая проверяется на колонку текста против уже выросшей рамки.
        for i in sorted(np.flatnonzero(new), key=lambda k: max(gap_x[k], gap_y[k])):
            before = Box(x0, y0, x1, y1)
            after = Box(min(x0, int(spots[i, 0])), min(y0, int(spots[i, 1])), max(x1, int(spots[i, 2])),
                        max(y1, int(spots[i, 3])))  # fmt: skip
            if column_rows(rows, before, after, line_gap) >= TEXT_COLUMN_MIN_ROWS:
                fit[i] = False  # сторона упёрлась в колонку текста — эту компоненту больше не пробуем
                continue
            taken[i], grown = True, True
            x0, y0, x1, y1 = after.as_tuple()
        if not grown:
            break
    return Box(x0, y0, x1, y1), int(taken.sum())


def cover_strokes(boxes: list[Box], bitonal: np.ndarray, dpi: int, barriers: list[Box] = ()) -> list[tuple[Box, int]]:
    """Добрать отдельные штрихи к рамкам вида рисунка (:func:`drawing_shaped`); прочие рамки — как есть.

    Args:
        boxes: Рамки line art, px копии ``bitonal``.
        bitonal: Битональная копия полосы (краска 0).
        dpi: Её разрешение.
        barriers: Растр и таблицы (исключения детектора).

    Returns:
        По паре ``(рамка, сколько компонент добрано)`` на каждую входную рамку, в том же порядке.
    """
    drawings = [b for b in boxes if drawing_shaped(b, dpi)]
    if not drawings:
        return [(b, 0) for b in boxes]
    ink = page_ink(bitonal, dpi, drawings)
    addable = ring_addable(ink, dpi)
    return [
        grow_strokes(b, ink.spots, addable, dpi, barriers, ink.rows) if drawing_shaped(b, dpi) else (b, 0)
        for b in boxes
    ]


__all__ = [
    "PageInk",
    "cover_strokes",
    "drawing_shaped",
    "grow_strokes",
    "is_text_component",
    "mark_rows",
    "page_ink",
    "text_rows",
    "column_rows",
    "ring_addable",
]
