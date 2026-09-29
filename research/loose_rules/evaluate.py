"""Правила-кандидаты «сирота ложная» по признакам ``features.csv``: причины отбраковки, пороги, сводка на разметке и по паку."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from research.loose_rules.labels import RuleLabel


class DropReason(str, Enum):
    """Почему сирота признана ложной (первая сработавшая причина в порядке объявления)."""

    # Трасса лежит на буквах: компоненты под ней компактны (стволы через строки, низы засечек).
    GLYPHS = "glyphs"
    # Нет ни одного длинного куска «не буквы»: тире, сшитые с цифрами («6—9—12»), перекладины «+».
    SHORT_RUN = "short_run"
    # Черта дроби или скобка формулы: внутри объекта «формула» или краска вплотную с обеих сторон.
    FORMULA = "formula"
    # Кромка листа или тень корешка: серая полка вместо чёрного ядра штриха.
    EDGE = "edge"


@dataclass(frozen=True)
class Thresholds:
    """Пороги правил; умолчания — выбор по разметке и поясам пака (``reports/loose_rules_false.md``)."""

    # Доля трассы на компактных компонентах (вдоль ≤ 4 поперечников и ≤ 15 мм), выше — буквы.
    glyph_share: float = 0.5
    # Самый длинный пробег «не букв» короче этого (мм) — тире и перекладины, не линейка.
    rule_run_mm: float = 6.0
    # Краска в полосе 1 мм по МЕНЬШЕЙ стороне горизонтали выше этого — числитель и знаменатель дроби.
    fraction_side_ink: float = 0.15
    # Минимум серого поперёк трассы (медиана по длине) выше этого — серая полка кромки, не штрих.
    edge_core_gray: float = 90.0


def reasons(table: pd.DataFrame, limits: Thresholds) -> pd.Series:
    """Первая сработавшая причина отбраковки для каждой строки признаков; пустая строка — линейка остаётся.

    Args:
        table: Признаки (``features.csv``).
        limits: Пороги.

    Returns:
        Серия строк (значения :class:`DropReason` или ``""``) того же индекса.
    """
    # Условия по порядку причин; первая сработавшая и есть причина.
    conditions = [
        (DropReason.GLYPHS, table.glyph_share_40 >= limits.glyph_share),
        (DropReason.SHORT_RUN, table.rule_run_mm < limits.rule_run_mm),
        (
            DropReason.FORMULA,
            (table.in_formula == 1) | ((table.horizontal == 1) & (table.side_ink_min >= limits.fraction_side_ink)),
        ),
        (DropReason.EDGE, table.core_gray >= limits.edge_core_gray),
    ]
    result = pd.Series("", index=table.index, dtype=object)
    for reason, condition in conditions:
        result[(result == "") & condition] = reason.value
    return result


def confusion(labelled: pd.DataFrame, dropped: pd.Series) -> dict:
    """Сводка на разметке: ложные пойманы / пропущены, настоящие потеряны / сохранены.

    Args:
        labelled: Признаки с колонкой ``label`` (значения :class:`RuleLabel`).
        dropped: Причины (:func:`reasons`) того же индекса.

    Returns:
        Словарь чисел: ``tp`` (ложная отброшена), ``fn``, ``fp`` (настоящая отброшена), ``tn``, полнота и
        доля потерь настоящих.
    """
    is_false = labelled.label.map(lambda value: RuleLabel(value).is_false)
    is_dropped = dropped != ""
    tp = int((is_false & is_dropped).sum())
    fn = int((is_false & ~is_dropped).sum())
    fp = int((~is_false & is_dropped).sum())
    tn = int((~is_false & ~is_dropped).sum())
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": round(tp / max(1, tp + fn), 3),
        "rule_loss": round(fp / max(1, fp + tn), 3),
    }
