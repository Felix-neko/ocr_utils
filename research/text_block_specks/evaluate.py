"""Прогон набора стенда с правилами и сверка «до — после» с разметкой глазами.

Прогон (:func:`run_task`) — полоса × правило → JSON концов строк и решений правила. Сверка
(:func:`compare`) сопоставляет концы «после» с концами прогона без правила (по стороне и ординате) и с
разметкой ``labels.csv`` (по ключу «полоса/блок/сторона/ряд» прогона без правила):

* **исправлено** — конец с меткой S (соринка) или M (пометка) ушёл внутрь и лёг на сторону блока
  (отклонение ≥ −0.8 мм);
* **вред** — конец с меткой P (настоящий знак) ушёл внутрь больше чем на ``MOVE_MM``;
* **прочие сдвиги** — все остальные концы, сдвинутые больше чем на ``MOVE_MM`` (на полосах без дефекта это
  ложные срабатывания, если отсмотр не покажет неразмеченную соринку).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ocr_utils.page_layout.text_blocks import WORK_DPI
from research.text_block_specks.dataset import page_ends
from research.text_block_specks.filters import Rule
from research.text_block_specks.pages import PdfPage
from research.text_block_specks.rules import TemplateRule, Thresholds

MM_PER_PX = 25.4 / WORK_DPI
# Сдвиг конца строки больше стольких мм считается изменением.
MOVE_MM = 0.3
# Концы «до» и «после» сопоставляются по ординате с допуском в столько пикселей рабочей копии.
MATCH_Y_PX = 4.0
# Прочий сдвиг считается «к стороне», если модуль отклонения уменьшился больше чем на столько мм.
IMPROVE_MM = 0.2
# Допуск стороны блока (``alignment.ALIGN_TOL_MM``): конец с отклонением не меньше минус столько — на стороне.
ON_SIDE_MM = -0.8

# Варианты правил стенда по имени (``none`` — прогон «до»).
RULES: dict[str, Rule | None] = {
    "none": None,
    "templates": TemplateRule(),
    "templates_nomark": TemplateRule(name="templates_nomark", marks=False),
    "templates_unsure": TemplateRule(name="templates_unsure", unsure_is_noise=True),
    "area_only": TemplateRule(
        name="area_only",
        thresholds=Thresholds(letter_h=99, dot_size=(99, 99), comma_h=(99, 99), hyphen_h=-1, sup_h=(99, 99)),
    ),
}


def run_task(task: tuple[Path, PdfPage, str, Path]) -> tuple[str, str, str]:
    """Задача пула: прогон полосы с правилом и запись ``<out>/<правило>/<ключ>.json``.

    Args:
        task: Каталог PDF, полоса, имя правила (``RULES``), корень выхода.

    Returns:
        ``(ключ, правило, ошибка)``; ошибка пустая, если всё прошло.
    """
    pdf_dir, page, rule_name, out = task
    target = out / rule_name / f"{page.key}.json"
    if target.exists():
        return page.key, rule_name, ""
    try:
        result = page_ends(pdf_dir, page, RULES[rule_name])
    except Exception as error:  # noqa: BLE001 — одна полоса не должна ронять прогон
        return page.key, rule_name, f"{type(error).__name__}: {error}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False))
    return page.key, rule_name, ""


@dataclass(frozen=True)
class EndChange:
    """Конец строки «до» и «после».

    Attributes:
        key: Полоса.
        block, side, row: Конец в прогоне «до».
        x_before, x_after: Абсцисса конца (пиксели рабочей копии); ``x_after`` — NaN, если пары не нашлось.
        resid_before, resid_after: Отклонение от стороны блока, мм.
        moved_mm: Сдвиг конца внутрь строки, мм (плюс — внутрь, минус — наружу).
        label: Метка глазами (S, M, P, L, X) или пусто.
        y: Ордината конца.
    """

    key: str
    block: int
    side: str
    row: int
    x_before: float
    x_after: float
    resid_before: float
    resid_after: float
    moved_mm: float
    label: str
    y: float


def compare(before: dict, after: dict, labels: dict[tuple[str, int, str, int], str]) -> list[EndChange]:
    """Концы строк полосы «до» и «после» с метками.

    Args:
        before: JSON прогона без правила.
        after: JSON прогона с правилом.
        labels: ``(полоса, блок, сторона, ряд) → метка``.

    Returns:
        По концу «до» — :class:`EndChange`.
    """
    out = []
    after_ends = after["ends"]
    for end in before["ends"]:
        # Пара «после» — тот же конец (сторона) на той же ординате и недалеко по x.
        best, best_dy = None, MATCH_Y_PX
        for other in after_ends:
            if other["side"] != end["side"]:
                continue
            dy = abs(other["y"] - end["y"])
            if dy <= best_dy and abs(other["x"] - end["x"]) < 40:
                best, best_dy = other, dy
        inward = 1.0 if end["side"] == "right" else -1.0
        x_after = best["x"] if best else float("nan")
        moved = (end["x"] - x_after) * inward * MM_PER_PX if best else float("nan")
        out.append(
            EndChange(
                key=before["key"],
                block=end["block"],
                side=end["side"],
                row=end["row"],
                x_before=end["x"],
                x_after=x_after,
                resid_before=end["resid_mm"],
                resid_after=best["resid_mm"] if best else float("nan"),
                moved_mm=round(moved, 3) if best else float("nan"),
                label=labels.get((before["key"], end["block"], end["side"], end["row"]), ""),
                y=end["y"],
            )
        )
    return out


def _moved(change: EndChange) -> bool:
    """Конец сдвинут больше чем на ``MOVE_MM`` (в любую сторону)."""
    return not np.isnan(change.moved_mm) and abs(change.moved_mm) > MOVE_MM


def summary(changes: list[EndChange], defect_keys: set[str]) -> dict:
    """Сводные меры варианта по концам строк.

    Args:
        changes: Концы всех полос набора.
        defect_keys: Полосы с дефектом (остальные — без дефекта).

    Returns:
        Словарь мер: сколько S/M исправлено из скольких, сколько P задето, сколько прочих сдвигов на
        полосах с дефектом и без, сколько концов потеряли пару.
    """
    bad = [c for c in changes if c.label in ("S", "M")]
    fixed = [c for c in bad if _moved(c) and c.moved_mm > 0 and c.resid_after >= ON_SIDE_MM]
    helped = [c for c in bad if _moved(c) and c.moved_mm > 0]
    signs = [c for c in changes if c.label == "P"]
    harmed = [c for c in signs if _moved(c) and c.moved_mm > 0]
    other = [c for c in changes if c.label not in ("S", "M", "P") and _moved(c)]
    other_defect = [c for c in other if c.key in defect_keys]
    other_clean = [c for c in other if c.key not in defect_keys]
    # Прочий сдвиг «к стороне»: модуль отклонения от стороны блока уменьшился заметно (соринка без метки).
    better = [c for c in other if abs(c.resid_after) < abs(c.resid_before) - IMPROVE_MM]
    lost = [c for c in changes if np.isnan(c.moved_mm)]
    return {
        "S/M": len(bad),
        "исправлено": len(fixed),
        "сдвинуто внутрь": len(helped),
        "P": len(signs),
        "P задето": len(harmed),
        "прочие сдвиги (дефект)": len(other_defect),
        "прочие сдвиги (чистые)": len(other_clean),
        "из прочих к стороне": len(better),
        "без пары": len(lost),
    }


__all__ = ["EndChange", "MOVE_MM", "RULES", "compare", "run_task", "summary"]
