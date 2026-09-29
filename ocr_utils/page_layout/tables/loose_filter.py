"""Отбраковка ложных линеек-сирот по признакам трассы: буквы, короткий пробег, черта дроби, кромка листа.

Сирота (линейка, не вошедшая ни в таблицу, ни в схему) идёт барьером в текстовые блоки, и ложная сирота
режет строки. Фильтр :func:`rules.loose_rule_is_clean` (краска в полосе 1 мм по большей стороне) отсеивает
«линейки», к которым вплотную прилипли буквы строки, но не видит четырёх видов ложных сирот, найденных
стендом ``research/loose_rules`` на паке-1 (``reports/loose_rules_false.md``, 2026-09-29):

* **буквы** — стволы букв двух-трёх строк жирного заголовка, сшитые через межстрочье; низы засечек и
  перекладины букв одной строки. Буквы крупного набора стоят вразрядку, по бокам ствола — чистая бумага,
  поэтому ``side_ink`` их пропускает. Отличает их то, ЧЬЯ краска лежит под трассой: каждая буква — своя
  компактная связная компонента, а линейка — вытянутая, даже с прилипшими хвостами букв;
* **короткий пробег** — тире, сшитые с цифрами («6—9—12»), перекладины «+» и «−» в строке формулы: тире
  вытянуто и за букву не считается, но ни одного длинного куска «не буквы» в цепочке нет;
* **черта дроби** — числитель и знаменатель стоят вплотную с ОБЕИХ сторон, у отбивки и подчёркивания одна
  сторона чистая (скобки формул снимает ``analysis`` по объектам «формула», здесь их нет);
* **кромка листа и тень корешка** — поперёк трассы серая полка, а не чёрное ядро штриха.

Замер на разметке стенда (289 сирот, 71 настоящая): полнота по ложным 0.91, потеряна одна спорная линия
у кромки листа; по паку на выборке по 30 отброшенных на причину — 117 из 120 действительно не отбивки.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import LooseRule
from ocr_utils.page_layout.tables.ruling import mm_to_px

# Компонента «компактна как буква», если вдоль линейки она не длиннее стольких своих поперечников.
# Порог 4, а не 2: узкий заголовочный шрифт выше двух своих ширин, и при 2 стопки его стволов набирали
# долю 0.5 и проходили (стенд, лист v_mid). Настоящая линейка 8 мм × 0,3 мм — 25 поперечников.
GLYPH_ASPECT = 4.0
# Компонента длиннее этого вдоль линейки — не буква, сколь угодно толстая (самые крупные буквы
# заголовков пака — около 12 мм).
GLYPH_MAX_MM = 15.0
# Доля точек трассы на буквах, начиная с которой сирота — буквы. Распределение по паку двугорбое:
# у 8 016 сирот из 9 334 доля ноль, у 1 005 — от 0.9; между ними около двухсот.
MAX_GLYPH_SHARE = 0.5
# Для пробега «не букв» — порог компактности строже (2): тире (2–4 мм при толщине 0,3 мм) при нём не буква,
# и пробег считается по нему, а перекладина «+» — буква и пробег рвёт.
RUN_GLYPH_ASPECT = 2.0
# Самый длинный пробег краски «не букв» короче этого — тире и перекладины, не линейка (у линейки он не
# короче порога её длины, 8 мм).
MIN_RULE_RUN_MM = 6.0
# Краска в полосе ``rules.LOOSE_SIDE_BAND_MM`` по МЕНЬШЕЙ стороне горизонтали выше этой доли — числитель и
# знаменатель дроби. У отбивок и подчёркиваний по меньшей стороне ноль.
MAX_FRACTION_SIDE_INK = 0.15
# Окно поперёк трассы для профиля ядра штриха: по 1 мм в обе стороны.
CORE_WINDOW_MM = 1.0
# Медиана по длине минимума серого поперёк трассы не ниже этого — серая полка кромки или тени, не штрих.
# Профили (серый 0–255): у линеек ядро 0–30, редко до 120; у кромки — 105–145.
MIN_EDGE_CORE_GRAY = 90.0


class LooseDrop(str, Enum):
    """Почему сирота признана ложной (первая сработавшая причина в порядке объявления)."""

    GLYPHS = "glyphs"
    SHORT_RUN = "short_run"
    FRACTION = "fraction"
    EDGE = "edge"


@dataclass(frozen=True)
class PageComponents:
    """Связные компоненты краски полосы — одни на все её сироты.

    Attributes:
        labels: Метки 8-связных компонент бинарной картинки (0 — бумага).
        stats: Габариты компонент (``cv2.CC_STAT_*``), строка на метку.
    """

    labels: np.ndarray
    stats: np.ndarray


def page_components(binary: np.ndarray) -> PageComponents:
    """Компоненты краски полосы для :func:`drop_reason`.

    Args:
        binary: Краска белым на чёрном (``ruling.binarize``) в рабочем разрешении.

    Returns:
        :class:`PageComponents`.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return PageComponents(labels, stats)


def _trace(rule: LooseRule, along_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Точки трассы с шагом в пиксель вдоль оси линейки (в «горизонтальной» системе: вдоль — x).

    Args:
        rule: Линейка в пикселях рабочей копии.
        along_size: Размер картинки вдоль оси линейки (ширина для горизонтали, высота для вертикали).

    Returns:
        Пара массивов: координаты вдоль оси (целые, в кадре) и поперёк (дробные, интерполяция ломаной).
    """
    points = np.asarray(rule.points, dtype=np.float64)
    if not rule.horizontal:
        # Вертикаль сводится к горизонтали: вдоль — y, поперёк — x (как в ``rules.side_ink``).
        points = points[:, ::-1]
    points = points[np.argsort(points[:, 0])]
    along = np.arange(int(round(points[0, 0])), int(round(points[-1, 0])) + 1)
    along = along[(along >= 0) & (along < along_size)]
    return along, np.interp(along, points[:, 0], points[:, 1])


def _longest_run(flags: np.ndarray) -> int:
    """Длина самого длинного пробега ``True`` в булевом ряду."""
    padded = np.concatenate([[False], flags, [False]]).astype(np.int8)
    steps = np.diff(padded)
    starts, ends = np.flatnonzero(steps == 1), np.flatnonzero(steps == -1)
    return int((ends - starts).max()) if starts.size else 0


@dataclass(frozen=True)
class TraceMeasures:
    """Меры трассы сироты, по которым решает :func:`drop_reason`.

    Attributes:
        glyph_share: Доля закрашенных точек трассы, чья компонента компактна как буква (``GLYPH_ASPECT``).
        rule_run_mm: Самый длинный пробег краски «не букв» (при ``RUN_GLYPH_ASPECT``), мм.
        core_gray: Медиана по длине минимума серого поперёк трассы в окне ``CORE_WINDOW_MM``.
    """

    glyph_share: float
    rule_run_mm: float
    core_gray: float


def trace_measures(gray: np.ndarray, components: PageComponents, rule: LooseRule, dpi: int) -> TraceMeasures:
    """Меры трассы: чья краска под ней (буквы или вытянутый штрих) и каково ядро штриха поперёк.

    Для каждой точки трассы берётся самая частая компонента в поперечном окне полосы линейки
    (полтолщины и пиксель запаса), и по её габариту решается, буква это или нет.

    Args:
        gray: Серая копия полосы в рабочем разрешении (та, по которой искались линейки).
        components: Компоненты краски полосы (:func:`page_components`).
        rule: Сирота в пикселях той же копии.
        dpi: Разрешение копии.

    Returns:
        :class:`TraceMeasures`; у пустой трассы (вне кадра) — нули и белый серый.
    """
    labels = components.labels if rule.horizontal else components.labels.T
    gray_axis = gray if rule.horizontal else gray.T
    height, width = labels.shape
    along, across = _trace(rule, width)
    if along.size == 0:
        return TraceMeasures(0.0, 0.0, 255.0)
    centre = np.clip(np.round(across).astype(int), 0, height - 1)
    band = max(1, int(np.ceil(rule.thickness_px / 2.0)) + 1)

    # Самая частая компонента в поперечном окне каждой точки (0 — под точкой бумага).
    dominant = np.zeros(along.size, dtype=np.int64)
    for index, (x, y) in enumerate(zip(along, centre)):
        column = labels[max(0, y - band) : min(height, y + band + 1), x]
        column = column[column > 0]
        if column.size:
            dominant[index] = np.bincount(column).argmax()
    inked = dominant > 0

    # Габарит компоненты вдоль и поперёк оси линейки и её «вытянутость».
    along_extent = components.stats[:, cv2.CC_STAT_WIDTH if rule.horizontal else cv2.CC_STAT_HEIGHT][dominant]
    across_extent = components.stats[:, cv2.CC_STAT_HEIGHT if rule.horizontal else cv2.CC_STAT_WIDTH][dominant]
    ratio = along_extent / np.maximum(1, across_extent)
    short = along_extent <= mm_to_px(GLYPH_MAX_MM, dpi)
    glyph_share = float((inked & short & (ratio <= GLYPH_ASPECT)).sum()) / max(1, int(inked.sum()))
    # Пробег «не букв» — при строгом пороге: тире в нём не буква, перекладина «+» — буква.
    not_glyph = inked & ~(short & (ratio <= RUN_GLYPH_ASPECT))
    rule_run_mm = _longest_run(not_glyph) * 25.4 / dpi

    # Ядро штриха: минимум серого в окне ±1 мм поперёк каждой точки, медиана по длине.
    reach = mm_to_px(CORE_WINDOW_MM, dpi)
    minima = [float(gray_axis[max(0, y - reach) : min(height, y + reach + 1), x].min()) for x, y in zip(along, centre)]
    return TraceMeasures(glyph_share, rule_run_mm, float(np.median(minima)))


def drop_reason(
    gray: np.ndarray, components: PageComponents, side_ink: tuple[float, float], rule: LooseRule, dpi: int
) -> LooseDrop | None:
    """Ложна ли сирота и почему.

    Args:
        gray: Серая копия полосы в рабочем разрешении.
        components: Компоненты краски полосы (:func:`page_components`).
        side_ink: Краска в полосах по обе стороны линейки (``rules.side_ink``).
        rule: Сирота в пикселях той же копии.
        dpi: Разрешение копии.

    Returns:
        Первая сработавшая причина :class:`LooseDrop` или ``None`` — сирота настоящая.
    """
    measures = trace_measures(gray, components, rule, dpi)
    if measures.glyph_share >= MAX_GLYPH_SHARE:
        return LooseDrop.GLYPHS
    if measures.rule_run_mm < MIN_RULE_RUN_MM:
        return LooseDrop.SHORT_RUN
    if rule.horizontal and min(side_ink) >= MAX_FRACTION_SIDE_INK:
        return LooseDrop.FRACTION
    if measures.core_gray >= MIN_EDGE_CORE_GRAY:
        return LooseDrop.EDGE
    return None
