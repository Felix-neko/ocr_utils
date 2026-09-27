"""Сверка боксов surya Equation с эталоном: сопоставление блоков, полнота, точность, качество рамки, пересчёт на пак."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.surya_equations.labels import PageLabels, Verdict
from research.surya_equations.sources import MM_PER_UNIT, EquationBox, Stratum

# Бокс surya «накрыл» блок, если внутри бокса не меньше этой доли площади блока — или блок
# накрывает не меньше этой доли бокса (бокс — одна строка многострочного блока).
MATCH_SHARE = 0.5
# Слои, из которых выборка случайная (бутстрэп по полосам); остальные взяты целиком.
SAMPLED_STRATA = (Stratum.BOTH, Stratum.INLINE_FRAC_ONLY, Stratum.NO_SIGNAL)

Box = tuple[float, float, float, float]


def area(box: Box) -> float:
    """Площадь рамки ``(x0, y0, x1, y1)``; вывернутая — 0."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: Box, b: Box) -> float:
    """Площадь пересечения двух рамок."""
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def covers(block: Box, box: Box) -> bool:
    """Бокс surya накрывает блок эталона (см. ``MATCH_SHARE``)."""
    common = intersection(block, box)
    return common >= MATCH_SHARE * area(block) or (common > 0 and common >= MATCH_SHARE * area(box))


def union_box(boxes: list[Box]) -> Box:
    """Общая рамка нескольких боксов (блок, порезанный surya по строкам, мерится целиком)."""
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def box_quality(block: Box, found: Box) -> dict[str, float]:
    """Качество рамки surya относительно эталонного блока.

    Args:
        block: эталон (ужат по краске).
        found: рамка surya (или общая рамка нескольких боксов).

    Returns:
        ``iou``; ``block_cover`` — доля блока внутри рамки (1 — ничего не срезано); ``excess`` — доля рамки
        вне блока; ``left_mm``/``top_mm``/``right_mm``/``bottom_mm`` — запас рамки над блоком по сторонам в мм
        (плюс — рамка шире блока, минус — срезала формулу).
    """
    common = intersection(block, found)
    return {
        "iou": common / (area(block) + area(found) - common),
        "block_cover": common / area(block),
        "excess": 1.0 - common / area(found),
        "left_mm": (block[0] - found[0]) * MM_PER_UNIT,
        "top_mm": (block[1] - found[1]) * MM_PER_UNIT,
        "right_mm": (found[2] - block[2]) * MM_PER_UNIT,
        "bottom_mm": (found[3] - block[3]) * MM_PER_UNIT,
    }


def match_page(
    page: str, stratum: Stratum, labels: PageLabels, equations: tuple[EquationBox, ...]
) -> tuple[list[dict], list[dict]]:
    """Сопоставить блоки эталона и боксы surya одной полосы.

    Args:
        page: «год/выпуск/полоса».
        stratum: слой выборки полосы.
        labels: эталон и вердикты полосы.
        equations: боксы surya ``Equation`` из кэша (номера совпадают с ключами вердиктов).

    Returns:
        ``(строки блоков, строки боксов)``. Строка блока: пойман ли, сколько боксов его накрыли и
        качество общей рамки; строка бокса: уверенность, вердикт разметчика, сколько блоков накрыл.
    """
    boxes = [e.box for e in equations]
    block_rows = []
    for index, block in enumerate(labels.blocks):
        hits = [j for j, box in enumerate(boxes) if covers(block.box, box)]
        row = {
            "page": page,
            "stratum": stratum.value,
            "block": index,
            "inline": block.inline,
            "lines": block.lines,
            "note": block.note,
            "height_mm": (block.box[3] - block.box[1]) * MM_PER_UNIT,
            "width_mm": (block.box[2] - block.box[0]) * MM_PER_UNIT,
            "detected": bool(hits),
            "n_boxes": len(hits),
            "boxes": ",".join(f"S{j}" for j in hits),
        }
        if hits:
            row.update(box_quality(block.box, union_box([boxes[j] for j in hits])))
        block_rows.append(row)
    box_rows = []
    for j, equation in enumerate(equations):
        verdict, comment = labels.verdicts[j]
        box_rows.append(
            {
                "page": page,
                "stratum": stratum.value,
                "box": f"S{j}",
                "confidence": equation.confidence,
                "verdict": verdict.value,
                "comment": comment,
                "n_blocks": sum(covers(b.box, equation.box) for b in labels.blocks if not b.inline),
            }
        )
    return block_rows, box_rows


@dataclass(frozen=True)
class PackEstimate:
    """Полнота и точность на весь пак: доли по слоям выборки, взвешенные числом полос слоя."""

    blocks: float  # выносных блоков на паке
    detected: float  # из них поймано
    boxes: float  # боксов surya на паке
    true_boxes: float  # из них на формулах (вердикт formula)

    @property
    def recall(self) -> float:
        """Доля пойманных блоков."""
        return self.detected / self.blocks

    @property
    def precision(self) -> float:
        """Доля боксов на формулах."""
        return self.true_boxes / self.boxes


def per_page(sample: pd.DataFrame, blocks: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    """Счёт по полосам выборки: блоков, пойманных, боксов, верных боксов (нули у полос без них).

    Args:
        sample: полосы выборки (``page``, ``stratum``).
        blocks: строки блоков :func:`match_page`.
        boxes: строки боксов :func:`match_page`.

    Returns:
        Таблица с колонками ``page, stratum, blocks, detected, boxes, true_boxes``.
    """
    display = blocks[~blocks.inline]
    counts = display.groupby("page").agg(blocks=("block", "size"), detected=("detected", "sum"))
    found = boxes.groupby("page").agg(
        boxes=("box", "size"), true_boxes=("verdict", lambda v: int((v == Verdict.FORMULA.value).sum()))
    )
    table = sample[["page", "stratum"]].merge(counts, on="page", how="left").merge(found, on="page", how="left")
    return table.fillna(0)


def estimate(pages: pd.DataFrame, strata_sizes: dict[str, int]) -> PackEstimate:
    """Пересчёт на пак: среднее по полосам слоя × число полос слоя, сумма по слоям.

    Args:
        pages: счёт по полосам (:func:`per_page`).
        strata_sizes: слой → число полос пака в нём. Слои без полос в выборке (``inline_only``)
            считаются пустыми — это допущение отчёта.

    Returns:
        :class:`PackEstimate`.
    """
    totals = {"blocks": 0.0, "detected": 0.0, "boxes": 0.0, "true_boxes": 0.0}
    for stratum, size in strata_sizes.items():
        rows = pages[pages.stratum == stratum]
        if rows.empty:
            continue
        for column in totals:
            totals[column] += rows[column].mean() * size
    return PackEstimate(**totals)


def bootstrap(
    pages: pd.DataFrame, strata_sizes: dict[str, int], rounds: int = 5000, seed: int = 0
) -> tuple[tuple[float, float], tuple[float, float]]:
    """90 % доверительные интервалы полноты и точности на пак: бутстрэп полос внутри случайных слоёв.

    Args:
        pages: счёт по полосам (:func:`per_page`).
        strata_sizes: слой → число полос пака.
        rounds: повторов бутстрэпа.
        seed: семя генератора.

    Returns:
        ``((полнота 5 %, 95 %), (точность 5 %, 95 %))``.
    """
    rng = np.random.default_rng(seed)
    sampled = {s.value for s in SAMPLED_STRATA}
    recalls, precisions = [], []
    for _ in range(rounds):
        parts = []
        for stratum in strata_sizes:
            rows = pages[pages.stratum == stratum]
            if stratum in sampled and len(rows):
                rows = rows.iloc[rng.integers(0, len(rows), len(rows))]
            parts.append(rows)
        result = estimate(pd.concat(parts), strata_sizes)
        recalls.append(result.recall)
        precisions.append(result.precision)
    return tuple(np.quantile(recalls, [0.05, 0.95])), tuple(np.quantile(precisions, [0.05, 0.95]))
