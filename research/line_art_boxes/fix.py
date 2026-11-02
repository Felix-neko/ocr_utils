"""Варианты разбора стенда: боевой код (``base``), прежнее поведение до правок ядра 2026-10-25 (``legacy``) и прототипы — подмены функций ``page_layout`` на время разбора, боевой код не меняется."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial

import cv2
import numpy as np

import ocr_utils.page_layout.analysis as layout_analysis
import ocr_utils.page_layout.line_art.detector as art_detector
import ocr_utils.page_layout.line_art.features as art_features
import ocr_utils.page_layout.tables.detector as table_detector
import ocr_utils.page_layout.tables.curve as table_curve
from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.line_art.features import LETTER_MAX_HEIGHT_MM
from ocr_utils.page_layout.line_art.features import letter_boxes as core_letter_boxes
from ocr_utils.page_layout.line_art.features import params_for_dpi
from ocr_utils.page_layout.tables.detector import EDGE_STRIP_MM
from ocr_utils.units import mm_to_px

# В доборе штрихов полоса тени может отстоять от края на пиксель-другой (ХиП 1991/06 IMG_0475_R: x = 1–2 px при 150 dpi).
EDGE_NEAR_MM = 0.5

# --- Добор отдельных штрихов рисунка (рамка мала) ----------------------------------------------------------------
# Лучи солнца и лампы, линии движения, капли пота, дым — отдельные компоненты у рисунка, мельче ворот пятна; рамка их
# не брала (ХиП 1991/06 IMG_0457_L — лучи лампы до 10 мм от рамки, 01/IMG_1045_R — дым над трубой). Компонента краски
# в пределах GROW_REACH_MM от рамки и не буква строки добирается; кругов — GROW_ROUNDS (луч за лучом), а каждая сторона
# уходит не дальше GROW_CAP_MM от исходной.
GROW_REACH_MM = 6.0
GROW_ROUNDS = 3
GROW_CAP_MM = 15.0
# Компонента мельче этого по большей стороне — точка, запятая, дефис переноса у обтекающей рисунок строки или сор;
# не добирается (дефис ~1.5–2 мм, луч и капля — от 2.5 мм).
GROW_MIN_SIDE_MM = 2.5
# Компонента крупнее этого по большей стороне — уже не штрих рисунка, а соседний объект или линейка колонки.
GROW_MAX_SIDE_MM = 25.0
# Добирается только штрих, а не буква: вытянутый (большая сторона / меньшая от GROW_MIN_ASPECT — лучи, линии движения,
# скобки «)))») или разреженный (краска не больше GROW_MAX_FILL рамки — спираль дыма, дуга). Буквы компактны
# (вытянутость 0.5–2, заполнение 0.3–0.6): заглавная «О» в начале строки рядом с логотипом рубрики иначе добиралась.
GROW_MIN_ASPECT = 2.5
GROW_MAX_FILL = 0.2
# Строка текста для добора: не меньше ROW_MIN_LETTERS букв, накрывающих центр компоненты по высоте, в пределах
# max(высота компоненты, ROW_REACH_MM) по горизонтали.
ROW_REACH_MM = 4.0
ROW_MIN_LETTERS = 2
ROW_TOUCH_MM = 1.0
ROW_LETTER_SHARE = 0.6

# --- Охват по краске в окрестности (ring) и по связности на большем масштабе (close) ----------------------------
# ring: как grow, но без условия формы — добираются и точки у знака вопроса (03/IMG_0144_R, «o» ~1.5 мм), капли,
# короткие штрихи тени под рисунком (01/IMG_1050_L). Мельче RING_MIN_SIDE_MM — пыль и сор бумаги; зазор до рамки —
# RING_REACH_MM (точки пиктограмм — 1–2 мм от рамки, линии движения шара — 2–4 мм).
RING_MIN_SIDE_MM = 0.8
RING_REACH_MM = 4.0
# Мельче этой площади — сор и пыль бумаги (у 01/IMG_1050_R по кругам добирались крупинки в 2–5 px), мм².
RING_MIN_AREA_MM2 = 0.2
# Ряд мелких знаков (звёздочки «* * *», отточие, точки строки): компонента до MARK_MAX_MM с не меньше чем MARK_MIN_ROW
# похожими (размер в 0.5–2 раза) соседями на той же высоте в пределах MARK_GAP_MM по горизонтали — знак строки, а не
# штрих рисунка (ряды звёздочек у 01/IMG_1050_R уводили рамку слона на 10 мм вниз).
MARK_MAX_MM = 3.0
MARK_GAP_MM = 8.0
MARK_MIN_ROW = 2
MARK_MAX_ASPECT = 2.0
MARK_SIMILAR = 1.4
DASH_MAX_HEIGHT_MM = 0.6
DASH_MAX_LENGTH_MM = 2.5
# close: не-текстовая краска раздувается кругом радиуса CLOSE_RADIUS_MM (сшиваются куски в 2·радиус друг от друга), и
# рамка берёт исходную краску всех раздутых компонент, задевших её; сторона — не дальше GROW_CAP_MM.
CLOSE_RADIUS_MM = 2.0
# Добор — только к рамкам вида рисунка: обе стороны от COVER_MIN_SIDE_MM и вытянутость не больше COVER_MAX_ASPECT.
# Ложные рамки на кусках таблиц и штампах поверх оглавления плоские (105×10–23 мм) — их рост захватывал до 71 буквы
# строки (02/IMG_1111_L); орнамент рубрики «КАК ЭТО РЕШИТЬ» — 10×12 мм, проходит.
COVER_MIN_SIDE_MM = 8.0
COVER_MAX_ASPECT = 3.0


class FixVariant(StrEnum):
    """Варианты разбора стенда."""

    BASE = "base"  # боевой код как есть: TABLES 7 / LINE_ART 8 (полосы у края, правило A′, добор штрихов strokes)
    LEGACY = "legacy"  # поведение до 2026-10-25 (TABLES 6 / LINE_ART 6): тень у края — штрих, без A′, без добора
    NO_STROKES = "no_strokes"  # TABLES 7 / LINE_ART 7: полосы у края и A′, без добора штрихов
    NO_UNWARP = "no_unwarp"  # base без выпрямления вырезки находки детектора таблиц (TABLES 7 / LINE_ART 9)
    GROW = "grow"  # base + добор отдельных штрихов у рамки по форме (вытянутые, разреженные) — прототип 2026-10-25
    RING = "ring"  # base + добор любой не-текстовой краски в RING_REACH_MM от рамки (точки, капли, штрихи тени)
    CLOSE = "close"  # base + охват по связности не-текстовой краски, раздутой на CLOSE_RADIUS_MM


def letter_boxes(ink: np.ndarray, dpi: int) -> np.ndarray:
    """Рамки букв полосы по маске краски — :func:`line_art.features.letter_boxes` ядра с порогами под ``dpi``.

    Args:
        ink: Маска краски (ненулевое — краска).
        dpi: Разрешение маски.

    Returns:
        Массив ``(N, 4)`` рамок ``x0, y0, x1, y1``.
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    return core_letter_boxes(stats[1:], params_for_dpi(dpi))


def _no_edge_strips(stats: np.ndarray, shape: tuple[int, int], strip_px: int) -> np.ndarray:
    """Подмена ``tables.detector.edge_strips`` для ``legacy``: полос у края нет (прежний ``line_art_mask``)."""
    return np.zeros(len(stats), bool)


def _not_straightened(gray, dpi):
    """Подмена ``tables.curve.straightened`` для ``no_unwarp``: вырезка как есть (TABLES 7)."""
    return gray


def _no_cover(boxes, bitonal, dpi, barriers=()):
    """Подмена ``line_art.detector.cover_strokes``: рамки как есть (без добора штрихов, LINE_ART 7 и раньше)."""
    return [(b, 0) for b in boxes]


def _no_letters(box, letters, params) -> bool:
    """Подмена ``line_art.features.letters_in_row`` для ``legacy``: правила A′ нет."""
    return False


@contextmanager
def patched(variant: FixVariant, page=None):
    """Подменить функции ``page_layout`` на время разбора вариантом ``variant``.

    Args:
        variant: Вариант.
        page: ``PageImage`` разбираемой страницы (для вариантов, которым нужна картинка); пока не используется.

    Yields:
        Ничего; по выходу подмены снимаются.
    """
    original_strips = table_detector.edge_strips
    original_letters = art_features.letters_in_row
    original_detect = layout_analysis.detect_line_art
    original_cover = art_detector.cover_strokes
    original_straightened = table_curve.straightened
    if variant is FixVariant.NO_UNWARP:
        # ``tables.detector._find`` зовёт curve.straightened по имени модуля curve.
        table_curve.straightened = _not_straightened
    if variant is FixVariant.LEGACY:
        # ``line_art_mask`` зовёт edge_strips, а ``analyse_gray`` — letters_in_row по имени своего модуля.
        table_detector.edge_strips = _no_edge_strips
        art_features.letters_in_row = _no_letters
    if variant in (FixVariant.LEGACY, FixVariant.NO_STROKES) + COVER_VARIANTS:
        # ``detect_line_art`` зовёт cover_strokes по имени модуля детектора; варианты добора стенда — поверх v7.
        art_detector.cover_strokes = _no_cover
    if variant in COVER_VARIANTS:
        # ``PageLayout._finish_with`` зовёт detect_line_art по имени модуля разбора.
        layout_analysis.detect_line_art = partial(_detect_and_cover, variant)
    try:
        yield
    finally:
        table_detector.edge_strips = original_strips
        art_features.letters_in_row = original_letters
        layout_analysis.detect_line_art = original_detect
        art_detector.cover_strokes = original_cover
        table_curve.straightened = original_straightened


def is_text_component(box: tuple[int, int, int, int], letters: np.ndarray, dpi: int) -> bool:
    """Компонента — часть строки текста (для добора — с запасом): рядом не меньше ``ROW_MIN_LETTERS`` букв той же строки.

    Буква той же строки — её высота накрывает центр компоненты по высоте, зазор по горизонтали не больше
    ``max(высота компоненты, ROW_REACH_MM)``. Мягче :func:`letters_in_row`: тире «—» и дефис в конце строки у
    рисунка, капитель и курсив тоже текст (иначе тире строки, обтекающей рисунок, добирались к рамке —
    ХиП 1991/01 IMG_1004_R; заголовок «СПРАШИВАЮТ» — 06/IMG_0419_L). Луч, направленный в строку, тоже может
    оказаться «текстом» — его просто не доберут.

    Args:
        box: Рамка компоненты, px.
        letters: Рамки букв полосы (:func:`letter_boxes`).
        dpi: Разрешение.

    Returns:
        True — компонента принадлежит строке текста.
    """
    x0, y0, x1, y1 = box
    # Выше буквы — не буква (линии движения у шара 02/IMG_1175_R — 11–14 мм).
    if len(letters) == 0 or y1 - y0 > mm_to_px(LETTER_MAX_HEIGHT_MM, dpi):
        return False
    centre = (y0 + y1) / 2
    reach = max(y1 - y0, mm_to_px(ROW_REACH_MM, dpi))
    same = (letters[:, 0] == x0) & (letters[:, 1] == y0) & (letters[:, 2] == x1) & (letters[:, 3] == y1)
    in_row = (letters[:, 1] <= centre) & (letters[:, 3] >= centre) & ~same
    gap = np.maximum(0, np.maximum(x0 - letters[:, 2], letters[:, 0] - x1))
    near = in_row & (gap <= reach)
    if int(np.count_nonzero(near)) >= ROW_MIN_LETTERS:
        return True
    # Одна буква строки рядом, а компонента сама ростом с букву (не ниже ROW_LETTER_SHARE её высоты) — буква той же
    # строки при разгонке (05/IMG_0325_L: «о» в конце строки в 4 мм от соседки). Точка у пиктограммы мельче букв.
    heights = letters[:, 3] - letters[:, 1]
    if np.any(near & (y1 - y0 >= ROW_LETTER_SHARE * heights)):
        return True
    # Часть буквы: вплотную (зазор до ROW_TOUCH_MM) к букве строки — палочка «Ы», «Ю» заголовка 06/IMG_0419_L.
    return bool(np.any(in_row & (gap <= mm_to_px(ROW_TOUCH_MM, dpi))))


def stroke_like(spots: np.ndarray, areas: np.ndarray, dpi: int) -> np.ndarray:
    """Какие компоненты годятся в добор по размеру и форме: штрих (вытянутый или разреженный), не точка и не буква.

    Args:
        spots: Рамки компонент ``(N, 4)``: ``x0, y0, x1, y1``.
        areas: Площадь краски компонент ``(N,)``, px.
        dpi: Разрешение.

    Returns:
        Маска ``(N,)``.
    """
    width = np.maximum(1, spots[:, 2] - spots[:, 0])
    height = np.maximum(1, spots[:, 3] - spots[:, 1])
    major, minor = np.maximum(width, height), np.minimum(width, height)
    sized = (major >= mm_to_px(GROW_MIN_SIDE_MM, dpi)) & (major <= mm_to_px(GROW_MAX_SIDE_MM, dpi))
    shaped = (major / minor >= GROW_MIN_ASPECT) | (areas / (width * height) <= GROW_MAX_FILL)
    return sized & shaped


def grow_strokes(
    box: Box,
    spots: np.ndarray,
    addable: np.ndarray,
    dpi: int,
    barriers: list[Box] = (),
    reach_mm: float = GROW_REACH_MM,
) -> tuple[Box, int]:
    """Добрать к рамке рисунка отдельные штрихи рядом с ней (лучи, линии движения, дым).

    Args:
        box: Рамка рисунка, px.
        spots: Рамки всех компонент краски полосы ``(N, 4)``: ``x0, y0, x1, y1``.
        addable: Маска ``(N,)``: компонента годится в добор (штрих по :func:`stroke_like` и не буква строки).
        dpi: Разрешение.
        barriers: Рамки, в которые добор не заходит (растр, таблицы).
        reach_mm: Предельный зазор от компоненты до рамки, мм.

    Returns:
        ``(новая рамка, сколько компонент добрано)``.
    """
    reach = mm_to_px(reach_mm, dpi)
    cap = mm_to_px(GROW_CAP_MM, dpi)
    fit = addable.copy()
    for barrier in barriers:
        fit &= ~((spots[:, 0] < barrier.x1) & (spots[:, 2] > barrier.x0) & (spots[:, 1] < barrier.y1)
                 & (spots[:, 3] > barrier.y0))  # fmt: skip
    x0, y0, x1, y1 = box.as_tuple()
    limit = (x0 - cap, y0 - cap, x1 + cap, y1 + cap)
    taken = np.zeros(len(spots), bool)
    for _ in range(GROW_ROUNDS):
        # Зазор от компоненты до рамки по каждой оси (0 — перекрываются по этой оси).
        gap_x = np.maximum(0, np.maximum(x0 - spots[:, 2], spots[:, 0] - x1))
        gap_y = np.maximum(0, np.maximum(y0 - spots[:, 3], spots[:, 1] - y1))
        inside = (spots[:, 0] >= x0) & (spots[:, 1] >= y0) & (spots[:, 2] <= x1) & (spots[:, 3] <= y1)
        within = (spots[:, 0] >= limit[0]) & (spots[:, 1] >= limit[1]) & (spots[:, 2] <= limit[2])
        within &= spots[:, 3] <= limit[3]
        new = fit & ~inside & ~taken & within & (np.maximum(gap_x, gap_y) <= reach)
        if not new.any():
            break
        taken |= new
        x0 = min(x0, int(spots[new, 0].min()))
        y0 = min(y0, int(spots[new, 1].min()))
        x1 = max(x1, int(spots[new, 2].max()))
        y1 = max(y1, int(spots[new, 3].max()))
    return Box(x0, y0, x1, y1), int(taken.sum())


@dataclass(frozen=True)
class PageInk:
    """Компоненты краски полосы для добора к рамкам.

    Поля:
        spots: Рамки компонент ``(N, 4)``: ``x0, y0, x1, y1``.
        areas: Площадь краски компонент ``(N,)``.
        text: Компонента — часть строки текста (:func:`is_text_component`; считается только рядом с рамками, дальше —
            ``True``, то есть «не добирать»).
        edge: Компонента — полоса тени у края картинки.
        labels: Метки компонент (``connectedComponentsWithStats``, 0 — фон).
    """

    spots: np.ndarray
    areas: np.ndarray
    text: np.ndarray
    edge: np.ndarray
    labels: np.ndarray


def page_ink(bitonal: np.ndarray, dpi: int, boxes: list[Box]) -> PageInk:
    """Компоненты краски полосы с признаками «текст» и «полоса у края».

    Args:
        bitonal: Битональная копия (краска 0).
        dpi: Её разрешение.
        boxes: Рамки line art: буквенность считается только у компонент рядом с ними (проверка строки дорогая).

    Returns:
        :class:`PageInk`.
    """
    ink = (bitonal == 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    body = stats[1:]
    x, y = body[:, cv2.CC_STAT_LEFT], body[:, cv2.CC_STAT_TOP]
    spots = np.stack([x, y, x + body[:, cv2.CC_STAT_WIDTH], y + body[:, cv2.CC_STAT_HEIGHT]], axis=1).astype(np.int64)
    letters = letter_boxes(ink, dpi)
    # Буквы внутри рамок рисунков — надписи рисунка («СТРОЙ-МАТЕРИАЛЫ» на шаре 02/IMG_1175_R): штрих рядом с ними —
    # часть рисунка, а не буква их «строки».
    for b in boxes:
        inside = (letters[:, 0] >= b.x0) & (letters[:, 1] >= b.y0) & (letters[:, 2] <= b.x1) & (letters[:, 3] <= b.y1)
        letters = letters[~inside]
    reach = mm_to_px(GROW_REACH_MM + GROW_CAP_MM, dpi)
    near = np.zeros(len(spots), bool)
    for b in boxes:
        near |= ((spots[:, 2] >= b.x0 - reach) & (spots[:, 0] <= b.x1 + reach) & (spots[:, 3] >= b.y0 - reach)
                 & (spots[:, 1] <= b.y1 + reach))  # fmt: skip
    text = np.ones(len(spots), bool)
    for i in np.flatnonzero(near):
        text[i] = is_text_component(tuple(int(v) for v in spots[i]), letters, dpi)
    # Полосы тени у края вырезки (обрез, сгиб у корешка) — не штрих рисунка (ядро: tables.detector.edge_strips).
    height, width = ink.shape
    strip = mm_to_px(EDGE_STRIP_MM, dpi)
    gap = mm_to_px(EDGE_NEAR_MM, dpi)
    edge = (((spots[:, 0] <= gap) | (spots[:, 2] >= width - gap)) & (spots[:, 2] - spots[:, 0] <= strip)) | (
        ((spots[:, 1] <= gap) | (spots[:, 3] >= height - gap)) & (spots[:, 3] - spots[:, 1] <= strip)
    )
    return PageInk(spots, body[:, cv2.CC_STAT_AREA], text, edge, labels)


def ring_addable(ink: PageInk, dpi: int) -> np.ndarray:
    """Компоненты для варианта ``ring``: не текст, не полоса у края, размером от RING_MIN_SIDE_MM до GROW_MAX_SIDE_MM.

    Args:
        ink: Компоненты полосы.
        dpi: Разрешение.

    Returns:
        Маска ``(N,)``.
    """
    major = np.maximum(ink.spots[:, 2] - ink.spots[:, 0], ink.spots[:, 3] - ink.spots[:, 1])
    sized = (major >= mm_to_px(RING_MIN_SIDE_MM, dpi)) & (major <= mm_to_px(GROW_MAX_SIDE_MM, dpi))
    sized &= ink.areas >= RING_MIN_AREA_MM2 * (dpi / 25.4) ** 2
    # Дефис, тире переноса, запятая: плоская короткая чёрточка (не выше DASH_MAX_HEIGHT_MM и не длиннее
    # DASH_MAX_LENGTH_MM) — знак строки у рисунка, а не луч (у 01/IMG_1013_L, 1004_R, 05/IMG_0325_L по ним рамка
    # спускалась на строку текста). Лучи и линии движения длиннее, капли и точки — круглые.
    w, h = ink.spots[:, 2] - ink.spots[:, 0], ink.spots[:, 3] - ink.spots[:, 1]
    sized &= ~((h <= mm_to_px(DASH_MAX_HEIGHT_MM, dpi)) & (w <= mm_to_px(DASH_MAX_LENGTH_MM, dpi)))
    # Волосок шириной в пиксель — край тени или сор, не штрих (у 01/IMG_1050_R — обрывки тени края листа).
    sized &= np.minimum(ink.spots[:, 2] - ink.spots[:, 0], ink.spots[:, 3] - ink.spots[:, 1]) >= 2
    return sized & ~ink.text & ~ink.edge & ~mark_rows(ink.spots, sized, dpi)


def mark_rows(spots: np.ndarray, candidates: np.ndarray, dpi: int) -> np.ndarray:
    """Какие мелкие компоненты стоят в ряду похожих знаков (звёздочки, отточие, точки строки).

    Args:
        spots: Рамки компонент ``(N, 4)``.
        candidates: Какие компоненты проверять (остальные — ``False``).
        dpi: Разрешение.

    Returns:
        Маска ``(N,)``: у компоненты не меньше MARK_MIN_ROW похожих соседей на той же высоте.
    """
    out = np.zeros(len(spots), bool)
    width, height = spots[:, 2] - spots[:, 0], spots[:, 3] - spots[:, 1]
    major = np.maximum(width, height)
    # Знаки строки компактны (звёздочка, точка: вытянутость до MARK_MAX_ASPECT); травинки и штрихи тени под рисунком
    # тоже стоят рядом, но вытянуты — их ряд не знак строки (03/IMG_0119_L).
    compact = major <= MARK_MAX_ASPECT * np.maximum(1, np.minimum(width, height))
    small = candidates & compact & (major <= mm_to_px(MARK_MAX_MM, dpi))
    idx = np.flatnonzero(small)
    if len(idx) == 0:
        return out
    centre_y = (spots[:, 1] + spots[:, 3]) / 2
    gap_px = mm_to_px(MARK_GAP_MM, dpi)
    others = np.flatnonzero(major <= 2 * mm_to_px(MARK_MAX_MM, dpi))
    for i in idx:
        j = others[others != i]
        same_height = np.abs(centre_y[j] - centre_y[i]) <= max(height[i], 3)
        # Похожие — та же форма: ширина и высота в пределах MARK_SIMILAR раз (звёздочки ряда одинаковы, лучи солнца —
        # разной длины и наклона: 04/IMG_0202_L).
        similar = (np.abs(np.log(np.maximum(1, width[j]) / max(1, width[i]))) <= np.log(MARK_SIMILAR)) & (
            np.abs(np.log(np.maximum(1, height[j]) / max(1, height[i]))) <= np.log(MARK_SIMILAR)
        )
        gap = np.maximum(0, np.maximum(spots[i, 0] - spots[j, 2], spots[j, 0] - spots[i, 2]))
        out[i] = int(np.count_nonzero(same_height & similar & (gap <= gap_px))) >= MARK_MIN_ROW
    return out


def close_cover(box: Box, ink: PageInk, addable: np.ndarray, dpi: int, barriers: list[Box] = ()) -> tuple[Box, int]:
    """Охват рамки по связности не-текстовой краски на большем масштабе (вариант ``close``).

    Краска компонент ``addable`` и всех компонент внутри рамки раздувается кругом радиуса CLOSE_RADIUS_MM; рамка
    берёт исходную краску всех раздутых компонент, задевших её, в пределах GROW_CAP_MM от каждой стороны.

    Args:
        box: Рамка рисунка, px.
        ink: Компоненты полосы.
        addable: Какие компоненты вне рамки можно добирать.
        dpi: Разрешение.
        barriers: Рамки растра и таблиц: их краска не участвует.

    Returns:
        ``(новая рамка, сколько компонент добрано)``.
    """
    cap = mm_to_px(GROW_CAP_MM, dpi)
    radius = mm_to_px(CLOSE_RADIUS_MM, dpi)
    height, width = ink.labels.shape
    # Окно: рамка плюс потолок плюс радиус — дальше рамка не уйдёт.
    wx0, wy0 = max(0, box.x0 - cap - radius), max(0, box.y0 - cap - radius)
    wx1, wy1 = min(width, box.x1 + cap + radius), min(height, box.y1 + cap + radius)
    local = ink.labels[wy0:wy1, wx0:wx1]
    inside = (ink.spots[:, 0] >= box.x0) & (ink.spots[:, 1] >= box.y0) & (ink.spots[:, 2] <= box.x1)
    inside &= ink.spots[:, 3] <= box.y1
    usable = np.concatenate([[False], addable | inside])
    for barrier in barriers:
        hit = ((ink.spots[:, 0] < barrier.x1) & (ink.spots[:, 2] > barrier.x0) & (ink.spots[:, 1] < barrier.y1)
               & (ink.spots[:, 3] > barrier.y0))  # fmt: skip
        usable[1:] &= ~hit | inside
    mask = usable[local].astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    count, blobs = cv2.connectedComponents(cv2.dilate(mask, kernel), connectivity=8)
    # Раздутые компоненты, задевшие рамку.
    bx0, by0 = max(0, box.x0 - wx0), max(0, box.y0 - wy0)
    touched = np.unique(blobs[by0 : box.y1 - wy0, bx0 : box.x1 - wx0])
    touched = touched[touched > 0]
    if not len(touched):
        return box, 0
    keep = np.isin(blobs, touched) & (mask > 0)
    ys, xs = np.nonzero(keep)
    x0 = max(min(box.x0, int(xs.min()) + wx0), box.x0 - cap)
    y0 = max(min(box.y0, int(ys.min()) + wy0), box.y0 - cap)
    x1 = min(max(box.x1, int(xs.max()) + 1 + wx0), box.x1 + cap)
    y1 = min(max(box.y1, int(ys.max()) + 1 + wy0), box.y1 + cap)
    added = len(set(np.unique(local[keep]).tolist()) - set((np.flatnonzero(inside) + 1).tolist()))
    return Box(x0, y0, x1, y1), added


def drawing_shaped(box: Box, dpi: int) -> bool:
    """Рамка вида рисунка: обе стороны не меньше COVER_MIN_SIDE_MM и вытянутость не больше COVER_MAX_ASPECT.

    Args:
        box: Рамка, px.
        dpi: Разрешение.

    Returns:
        True — к рамке можно добирать штрихи.
    """
    small, large = sorted((box.width, box.height))
    return small >= mm_to_px(COVER_MIN_SIDE_MM, dpi) and large <= COVER_MAX_ASPECT * max(1, small)


def _detect_and_cover(variant: FixVariant, inputs):
    """``detect_line_art`` и добор к каждой рамке способом варианта (``grow``, ``ring``, ``close``)."""
    regions = _ORIGINAL_DETECT(inputs)
    if not regions:
        return regions
    ink = page_ink(inputs.bitonal, inputs.dpi, [r.box for r in regions])
    barriers = list(inputs.excluded_boxes)
    if variant is FixVariant.GROW:
        addable = stroke_like(ink.spots, ink.areas, inputs.dpi) & ~ink.text & ~ink.edge
    else:
        addable = ring_addable(ink, inputs.dpi)
    out = []
    for region in regions:
        if not drawing_shaped(region.box, inputs.dpi):
            out.append(region)
            continue
        if variant is FixVariant.CLOSE:
            grown, added = close_cover(region.box, ink, addable, inputs.dpi, barriers)
        else:
            reach = GROW_REACH_MM if variant is FixVariant.GROW else RING_REACH_MM
            grown, added = grow_strokes(region.box, ink.spots, addable, inputs.dpi, barriers, reach)
        out.append(replace(region, box=grown, info={**region.info, "grown_strokes": added}))
    return out


# Варианты с добором к рамкам.
COVER_VARIANTS = (FixVariant.GROW, FixVariant.RING, FixVariant.CLOSE)

# Боевые функции, запомненные до любых подмен.
_ORIGINAL_DETECT = layout_analysis.detect_line_art
