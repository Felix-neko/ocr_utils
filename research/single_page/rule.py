"""Правило «в кадре одна страница» по признакам кадра (``features.FrameFeatures``): разворот, одна страница (лист разрезан по ложному корешку или найден один), страница с полоской соседней, страницы нет (закрытая крышка с ярлыком).

Пороги — по распределению 4 739 кадров пяти кэшей v2 (ПХ 1974/01 и 1975, ХиП 1991, пак-5 1990/01–03, пак-6) и
отсмотру листов по поясам (2026-10-24, ``reports/single_page.md``):

* высота рамки бумаги: ярлыки на закрытой крышке — 53–72 мм, самая низкая страница (ХиП 1991) — 195 мм;
* узкая / широкая трапеция: у одностраничных кадров 0.07–0.39, у разворотов (кроме одностраничных) не ниже 0.714
  (поднятая рукой страница РИСК 1991/03 IMG_0327);
* ширина / высота рамки бумаги: у одностраничных 0.57–0.72, у разворотов не ниже 0.807 (тот же IMG_0327; обычный
  разворот — 0.97–1.41).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class Verdict(StrEnum):
    """Вердикт правила по кадру и что с ним делать на выходе страниц."""

    SPREAD = "spread"  # разворот: две страницы, резать по корешку
    SINGLE = "single"  # одна страница: лист без разреза (ложный корешок по провалу печати или страница одна)
    SLIVER = "sliver"  # одна страница и полоска соседней у настоящего корешка: писать только широкую
    NO_PAGE = "no_page"  # страницы нет: закрытая крышка подшивки, бумага — только ярлык (или не найдена вовсе)


@dataclass(frozen=True)
class RuleParams:
    """Пороги правила.

    Поля:
        min_paper_h_mm: Бумага ниже этого, мм, — не страница (ярлык на крышке 53–72 мм, страница не ниже 195 мм).
        max_width_ratio: Узкая трапеция уже этой доли широкой — кусок, а не страница (одностраничные ≤ 0.39,
            развороты ≥ 0.714).
        max_aspect: Рамка бумаги уже этой доли своей высоты — в кадре лист одной страницы (одностраничные ≤ 0.72,
            развороты ≥ 0.807).
        typical_min_ratio: Разворот «ровный» для типичной ширины страницы набора (отчётная колонка, правило её не
            использует): ширины двух трапеций не дальше этой доли.
    """

    min_paper_h_mm: float = 130.0
    max_width_ratio: float = 0.5
    max_aspect: float = 0.78
    typical_min_ratio: float = 0.85


def typical_widths(rows: list[dict], params: RuleParams) -> dict[str, float]:
    """Типичная ширина страницы набора, мм: медиана ширин страниц ровных разворотов (две трапеции почти равной ширины).

    Args:
        rows: Строки признаков.
        params: Пороги правила.

    Returns:
        Набор → ширина, мм (NaN — ровных разворотов нет).
    """
    out: dict[str, float] = {}
    for name in sorted({r["dataset"] for r in rows}):
        ws = [w for r in rows if r["dataset"] == name and r["n_pages"] == 2 and r["width_ratio"] >= params.typical_min_ratio
              for w in (r["w_left_mm"], r["w_right_mm"])]  # fmt: skip
        out[name] = float(np.median(ws)) if ws else float("nan")
    return out


def verdict_of(row: dict, typical_w_mm: float, params: RuleParams) -> Verdict:
    """Вердикт кадра по его признакам.

    Порядок: нет бумаги или она ниже ``min_paper_h_mm`` — ``NO_PAGE``; одна трапеция — ``SINGLE``; две, и узкая уже
    ``max_width_ratio`` широкой, и рамка бумаги уже ``max_aspect`` высоты — одна страница: при одном экземпляре SAM 3
    (две «страницы» дал запасной разрез ``glue.split_pages`` по провалу печати) — ``SINGLE``, при двух (настоящий
    корешок, у края — полоска соседней страницы) — ``SLIVER``; иначе ``SPREAD``.

    Args:
        row: Строка признаков (``FrameFeatures.row``).
        typical_w_mm: Типичная ширина страницы набора, мм (правилом не используется: оставлено для отчёта).
        params: Пороги правила.

    Returns:
        Вердикт.
    """
    n_quads = int(row["w_left_mm"] > 0) + int(row["w_right_mm"] > 0)
    if row["n_pages"] == 0 or n_quads == 0 or row["paper_h_mm"] < params.min_paper_h_mm:
        return Verdict.NO_PAGE
    if n_quads == 1:
        return Verdict.SINGLE
    if row["width_ratio"] < params.max_width_ratio and row["paper_aspect"] < params.max_aspect:
        return Verdict.SINGLE if row["n_inst"] <= 1 else Verdict.SLIVER
    return Verdict.SPREAD
