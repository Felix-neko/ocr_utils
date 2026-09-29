"""Мера сращивания строк разного набора: стыки внутри ряда блока — широкий просвет и разный кегль по сторонам."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Сколько глифов с каждой стороны стыка берётся на медиану высоты.
SIDE_GLYPHS = 6
# Стык в учёт — только если просвет шире стольких высот меньшего кегля (межбуквенные зазоры не стыки).
MIN_GAP_HEIGHTS = 0.8
# По обе стороны стыка — не меньше стольких глифов: одиночные знаки на концах строки — шум...
MIN_SIDE_GLYPHS = 3
# ...кроме одиночного глифа (номер выпуска, цифра) за просветом шире стольких высот меньшего кегля.
LONE_GAP_HEIGHTS = 3.0
# Меньшая из высот сторон — не ниже стольких пикселей рабочей копии: обрывки линеек и точки не кегль.
MIN_HEIGHT_PX = 6.0

# Фильтр стыков-кандидатов на сращивание. Подобран глазами по выборке пака (сессия 2026-09-29): без
# условия на крупную сторону в кандидаты шли тире и цифры корпуса. Корпус пака-1 — глиф ≈ 11 px
# рабочей копии, заголовки — от 18 px.
CANDIDATE_MIN_HEIGHT = 18.0
CANDIDATE_RATIO = 1.6
CANDIDATE_GAP = 1.5


class JointKind(str, Enum):
    """Где стык: внутри одной оси (сцепка кусков строки) или между осями одного ряда блока (сборка рядов)."""

    AXIS = "axis"
    ROW = "row"


@dataclass(frozen=True)
class Joint:
    """Стык внутри ряда: просвет между соседними по x глифами и кегль по обе стороны (пиксели рабочей копии)."""

    kind: JointKind
    x_gap0: float  # правый край всего, что левее просвета
    x_gap1: float  # левый край первого глифа правее просвета
    h_left: float  # медианная высота глифов слева
    h_right: float  # и справа
    n_left: int  # глифов ряда слева от просвета
    n_right: int  # и справа

    @property
    def gap(self) -> float:
        """Ширина просвета."""
        return self.x_gap1 - self.x_gap0

    @property
    def ratio(self) -> float:
        """Во сколько раз кегль крупной стороны больше мелкой."""
        return max(self.h_left, self.h_right) / max(1.0, min(self.h_left, self.h_right))

    @property
    def gap_rel(self) -> float:
        """Просвет в высотах меньшего кегля."""
        return self.gap / max(1.0, min(self.h_left, self.h_right))

    @property
    def score(self) -> float:
        """Оценка сращивания: разница кегля (лог) плюс ширина просвета."""
        return float(np.log2(self.ratio) + 0.25 * self.gap_rel)

    @property
    def candidate(self) -> bool:
        """Кандидат ли на сращивание строк разного набора (см. ``CANDIDATE_*``)."""
        return (
            max(self.h_left, self.h_right) >= CANDIDATE_MIN_HEIGHT
            and self.ratio >= CANDIDATE_RATIO
            and self.gap_rel >= CANDIDATE_GAP
        )


def joints_of(glyphs: np.ndarray, owners: np.ndarray | None = None) -> list[Joint]:
    """Стыки ряда: просветы между соседними по x глифами и медианные высоты глифов по обе стороны.

    Args:
        glyphs: Боксы глифов ``(n, 4)`` — ``x0, y0, x1, y1`` (пиксели рабочей копии), любой порядок.
        owners: Номер оси ряда для каждого глифа; ``None`` — все глифы одной оси.

    Returns:
        Стыки слева направо, прошедшие отсев по ширине просвета, высоте и числу глифов.
    """
    order = np.argsort(glyphs[:, 0])
    boxes = glyphs[order]
    owners = np.zeros(len(boxes), dtype=int) if owners is None else np.asarray(owners)[order]
    heights = boxes[:, 3] - boxes[:, 1]
    # Правая кромка уже пройденного — накопленный максимум: перекрытые глифы просвета не дают.
    reach = np.maximum.accumulate(boxes[:, 2])
    out = []
    for i in range(1, len(boxes)):
        gap = boxes[i, 0] - reach[i - 1]
        if gap <= 0:
            continue
        few = i < MIN_SIDE_GLYPHS or len(boxes) - i < MIN_SIDE_GLYPHS
        h_left = float(np.median(heights[max(0, i - SIDE_GLYPHS) : i]))
        h_right = float(np.median(heights[i : i + SIDE_GLYPHS]))
        small = max(1.0, min(h_left, h_right))
        if gap / small < MIN_GAP_HEIGHTS or min(h_left, h_right) < MIN_HEIGHT_PX:
            continue
        if few and gap / small < LONE_GAP_HEIGHTS:
            continue
        # Между осями ряда (сборка рядов блока) или внутри одной оси (сцепка кусков строки).
        kind = JointKind.AXIS if owners[i] == owners[i - 1] else JointKind.ROW
        out.append(Joint(kind, float(reach[i - 1]), float(boxes[i, 0]), h_left, h_right, i, len(boxes) - i))
    return out


def row_glyphs(row) -> tuple[np.ndarray, np.ndarray] | None:
    """Глифы ряда блока и номер оси каждого глифа; ``None`` — у осей ряда нет глифов.

    Args:
        row: Ряд блока (``blocks.Row``).

    Returns:
        Пара ``(glyphs (n, 4), owners (n,))``.
    """
    parts = [(k, np.asarray(a.glyphs, dtype=float)) for k, a in enumerate(row.axes) if a.glyphs is not None]
    parts = [(k, g) for k, g in parts if len(g)]
    if not parts:
        return None
    return np.concatenate([g for _, g in parts]), np.concatenate([np.full(len(g), k) for k, g in parts])


def candidate_joints(analysis) -> list[dict]:
    """Стыки-кандидаты разбора полосы: по рядам всех блоков.

    Args:
        analysis: Разбор текстовых блоков (``page.PageAnalysis``).

    Returns:
        Словари: номер блока и ряда, ордината ряда и поля стыка.
    """
    out = []
    for block_no, block in enumerate(analysis.blocks):
        for row_no, row in enumerate(block.rows):
            found = row_glyphs(row)
            if found is None:
                continue
            for joint in joints_of(*found):
                if joint.candidate:
                    out.append(
                        {"block": block_no, "row": row_no, "y": round(row.y, 1), "kind": joint.kind.value,
                         "x_gap0": joint.x_gap0, "x_gap1": joint.x_gap1, "h_left": joint.h_left,
                         "h_right": joint.h_right, "ratio": round(joint.ratio, 2), "gap_rel": round(joint.gap_rel, 2)}
                    )  # fmt: skip
    return out
