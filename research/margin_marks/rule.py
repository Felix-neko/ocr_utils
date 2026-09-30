"""Правило «пометка на полях» по признакам кандидата (``features.MarkFeatures``) и оценка по разметке.

Пометка — рукописный карандашный штрих поверх полосы: подчёркивание строки, скобка у абзаца, дуга через колонку.
Классический детектор даёт её кандидатом «штрих» по одним пикселям; после заливки слов остаётся 1–6 тонких длинных
компонент без углов, ручных (волнистых), а сама рамка — размером с подчёркивание или скобку. Правило разбито на
условия, чтобы в оценке было видно, какое условие отсекло какой случай.

Пороги подобраны по разметке ``sets/labels.csv`` (16 пометок пака-1 в двух вариантах PDF, 2026-09-30) с запасом и
проверяются по всем остальным кандидатам пака (``evaluate``): любое срабатывание не на пометке — лист глазами.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Condition(str, Enum):
    """Условие правила (для разбора, почему кандидат прошёл или нет)."""

    STROKE = "кандидат «штрих» по одним пикселям"
    OUTCOME = "исход — только рисунки или «неясно»"
    FEW = "1–6 длинных компонент"
    THIN = "тонкие"
    NO_CORNER = "без углов"
    SIZE = "размер рамки"
    HAND = "ручной штрих (волнистость) или почти пусто"
    LITTLE_INK = "мало краски"


@dataclass(frozen=True)
class MarkRule:
    """Пороги правила.

    Attributes:
        max_components: Длинных компонент не больше.
        max_thickness_mm: Толщина длинных компонент не больше, мм (карандаш пака 0.22–0.44, заставка — от 0.65).
        min_long_side_mm: Длинная сторона рамки не меньше, мм (мусор у края скана — до 15).
        max_side_mm: Обе стороны рамки не больше, мм (пометки пака — до 44; графики, таблицы, заставки — от 60).
        min_wobble_mm: Волнистость худшей длинной компоненты не меньше, мм (печатные линии — до 0.05).
        empty_ink_share: Или остаток почти пуст (доля краски меньше): штрих лёг на слова и залит вместе с ними.
        max_ink_share: Доля краски остатка не больше.
    """

    max_components: int = 6
    max_thickness_mm: float = 0.5
    min_long_side_mm: float = 18.0
    max_side_mm: float = 50.0
    min_wobble_mm: float = 0.09
    empty_ink_share: float = 0.005
    max_ink_share: float = 0.12

    def failed(self, row: dict) -> list[Condition]:
        """Условия, которые кандидат не прошёл (пусто — пометка).

        Args:
            row: Строка ``features.csv``.

        Returns:
            Список не выполненных условий.
        """
        out = []
        if row["info_kind"] != "штрих" or row["info_sources"] != "ink":
            out.append(Condition.STROKE)
        # Только кандидат, все объекты которого — рисунки, или «неясно»: рисунок вместе с таблицей или формулой не трогается.
        if row["outcome"] != "неясно" and set(row["classes"].split("|")) != {"рисунок"}:
            out.append(Condition.OUTCOME)
        if not 1 <= int(row["long_components"]) <= self.max_components:
            out.append(Condition.FEW)
        if float(row["long_thickness_max_mm"]) > self.max_thickness_mm:
            out.append(Condition.THIN)
        if int(row["corners"]) > 0:
            out.append(Condition.NO_CORNER)
        width, height = float(row["box_w_mm"]), float(row["box_h_mm"])
        if max(width, height) < self.min_long_side_mm or max(width, height) > self.max_side_mm:
            out.append(Condition.SIZE)
        ink = float(row["ink_share"])
        if float(row["wobble_max_mm"]) < self.min_wobble_mm and ink >= self.empty_ink_share:
            out.append(Condition.HAND)
        if ink > self.max_ink_share:
            out.append(Condition.LITTLE_INK)
        return out

    def is_mark(self, row: dict) -> bool:
        """Кандидат — пометка."""
        return not self.failed(row)


__all__ = ["Condition", "MarkRule"]
