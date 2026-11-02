"""Прототип встраивания стенда (в ядре — ``text_blocks.pseudo_glyphs.filter_page``): первый разбор → кандидаты → правила (и CRAFT) → псевдоглифы закрашиваются белым в сером 300 dpi → повторный разбор; переключаемый, с enum причин."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
import pandas as pd

from ocr_utils.page_layout.text_blocks.page import PageAnalysis
from ocr_utils.page_layout.text_blocks.pseudo_glyphs import erase_mask
from research.pseudo_glyphs.candidates import EdgeSide, components_of, page_candidates
from research.pseudo_glyphs.crops import analyze
from research.pseudo_glyphs.rules import RuleParams, reasons


class FilterMode(StrEnum):
    """Режим фильтра псевдоглифов."""

    OFF = "off"  # как боевой код (без правила 0.3 мм — его задаёт отдельный флаг)
    RULES = "rules"  # только правила по признакам
    RULES_CRAFT = "rules_craft"  # правила + CRAFT для спорных


@dataclass(frozen=True)
class Drop:
    """Закрашенная компонента.

    Поля:
        box: Рамка ``(x0, y0, x1, y1)``, пиксели 300 dpi.
        reason: Причина (:class:`rules.PseudoReason`).
        comp: Номер компоненты в разметке первого прохода.
        row_end: Компонента была крайней в ряду; закрас одних компонент края вне рядов разбор не повторяет.
    """

    box: tuple[int, int, int, int]
    reason: str
    comp: int
    row_end: bool = True  # компонента была краем ряда (иначе — только у края вырезки, ряды её не видели)


@dataclass(frozen=True)
class FilteredPage:
    """Итог фильтра на странице.

    Поля:
        analysis: Разбор после фильтра (если ничего не закрашено — первый разбор).
        first: Первый разбор.
        drops: Закрашенные компоненты.
        gray300: Серый, поданный во второй разбор.
    """

    analysis: PageAnalysis
    first: PageAnalysis
    drops: tuple[Drop, ...]
    gray300: np.ndarray


def craft_scores(frame: pd.DataFrame, craft_map: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Оценка CRAFT каждого кандидата: максимум карты «регион» по пикселям компоненты (как ``edge_guard``).

    Args:
        frame: Кандидаты страницы (``x0, y0, x1, y1, comp``).
        craft_map: Карта CRAFT 0…1 размера серого 300 dpi.
        labels: Метки компонент того же размера.

    Returns:
        Оценки (N,).
    """
    out = np.zeros(len(frame))
    for i, row in enumerate(frame.itertuples()):
        window = labels[row.y0 : row.y1, row.x0 : row.x1] == row.comp
        heat = craft_map[row.y0 : row.y1, row.x0 : row.x1]
        out[i] = float(heat[window].max()) if window.any() else 0.0
    return out


def _page_drops(
    analysis: PageAnalysis,
    gray300: np.ndarray,
    raw300: np.ndarray,
    spine: EdgeSide | None,
    params: RuleParams,
    craft_map: np.ndarray | None,
) -> tuple[np.ndarray, list[Drop]]:
    """Один проход фильтра: кандидаты по разбору, решения правил (и CRAFT), закрас.

    Args:
        analysis: Разбор ``gray300``.
        gray300, raw300, spine: Как у :func:`filter_page`.
        params: Пороги (``craft_doubt`` учитывается, только если дана карта).
        craft_map: Карта CRAFT или ``None``.

    Returns:
        ``(серый с закрашенными псевдоглифами, закрашенные компоненты)``; без находок — исходный серый и пустой список.
    """
    frame = pd.DataFrame(page_candidates(analysis, gray300, raw300, spine))
    if frame.empty:
        return gray300, []
    comps = components_of(gray300)
    if craft_map is not None and params.craft_doubt is not None:
        frame["craft"] = craft_scores(frame, craft_map, comps.labels)
    frame["reason"] = reasons(frame, params)
    hit = frame[frame.reason != ""].drop_duplicates("comp")
    if hit.empty:
        return gray300, []
    # Закрас: пиксели выбранных компонент — в бумагу.
    cleaned = gray300.copy()
    cleaned[erase_mask(comps.labels, hit.comp.to_numpy())] = 255
    # Компонента — край ряда, если хоть у одного ряда она крайняя (одна компонента бывает кандидатом нескольких рядов).
    row_comps = set(frame[(frame.reason != "") & (frame.kind != "edge")].comp)
    drops = [Drop((int(r.x0), int(r.y0), int(r.x1), int(r.y1)), r.reason, int(r.comp), int(r.comp) in row_comps)
             for r in hit.itertuples()]  # fmt: skip
    return cleaned, drops


def filter_page(
    gray300: np.ndarray,
    raw300: np.ndarray,
    spine: EdgeSide | None,
    mode: FilterMode = FilterMode.RULES,
    params: RuleParams = RuleParams(),
    craft_map: np.ndarray | None = None,
    edge_rule: bool = False,
    first: PageAnalysis | None = None,
    passes: int = 2,
) -> FilteredPage:
    """Разбор страницы с фильтром псевдоглифов.

    Проход: кандидаты по текущему разбору → решения → закрас → повторный разбор. Проходов до ``passes``: после
    закраса крайней компоненты краем ряда становится следующая, и крючок из двух кусков снимается за два прохода.

    Args:
        gray300: Серый 300 dpi, как его видит разбор (утолщённый, линейки погашены).
        raw300: Серый 300 dpi без утолщения (тон краски).
        spine: Сторона корешка на картинке или ``None``.
        mode: Режим фильтра.
        params: Пороги правил (у ``RULES_CRAFT`` порог CRAFT — ``params.craft_doubt``).
        craft_map: Карта CRAFT страницы (нужна для ``RULES_CRAFT``).
        edge_rule: Держать ли правило 0.3 мм у края картинки в разборах.
        first: Готовый первый разбор (если уже есть), иначе считается.
        passes: Наибольшее число проходов.

    Returns:
        :class:`FilteredPage`.
    """
    first = first if first is not None else analyze(gray300, edge_rule)
    if mode is FilterMode.OFF:
        return FilteredPage(first, first, (), gray300)
    if mode is FilterMode.RULES or craft_map is None:
        params, craft_map = RuleParams(**{**params.__dict__, "craft_doubt": None}), None
    current, gray, drops = first, gray300, []
    for _ in range(passes):
        # Тон берётся с исходного серого: закрашенные места в нём остаются, но их компонент больше нет.
        gray_next, found = _page_drops(current, gray, raw300, spine, params, craft_map)
        if not found:
            break
        drops.extend(found)
        gray = gray_next
        if not any(d.row_end for d in found):
            # Закрашены только компоненты у края вне рядов: строки их не видели, разбор не меняется — не повторяем.
            break
        current = analyze(gray, edge_rule)
    return FilteredPage(current, first, tuple(drops), gray)


__all__ = ["Drop", "FilterMode", "FilteredPage", "craft_scores", "filter_page"]
