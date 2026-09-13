"""Сквозной прогон по одной вырезанной таблице: сетка → боковые ячейки → текст → замена.

ЗДЕСЬ СОБРАНО ТО, ЧТО В ОСТАЛЬНЫХ МОДУЛЯХ РАЗДЕЛЕНО. Каждый шаг сам по себе сравним с
альтернативами (см. одноимённые подпакеты), а этот модуль фиксирует ПОРЯДОК и то, что
между шагами передаётся. Порядок важен: кегль замены меряется по исходной ячейке, значит,
мерить надо до стирания; сторона поворота уточняется приором таблицы, значит, распознавать
надо после того, как решены все ячейки, а не по ходу.

ЧТО ГДЕ ЖИВЁТ. Всё, кроме GPU-детекторов и surya, считается на CPU и уезжает в пул
процессов по таблице на воркер. GPU-шаги (docTR, surya) в пул не заворачиваются —
видеопамять одна на всех (CLAUDE.md), — и потому в наборе по умолчанию их нет.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import re

from research.legacy.table_processing.geometry import Grid
from research.legacy.table_processing.imaging import load_gray
from research.legacy.table_processing.ocr import resolve as resolve_engines
from research.legacy.table_processing.render.compose import Composed, Replacement, compose, trim
from research.legacy.table_processing.rotation import resolve as resolve_detectors
from research.legacy.table_processing.rotation.base import CellCrop, rotate_cw
from research.legacy.table_processing.rotation.combine import CellDecision, apply_prior, combine_axis
from research.legacy.table_processing.structure.ruling_grid import WORK_DPI, cell_image, extract

logger = logging.getLogger(__name__)

# Разрешение, в котором вырезка разбирается и распознаётся. 300 dpi: на 150 петит шапки
# падает до 7 px и отбор компонент перестаёт отличать букву от точки, на 600 всё вчетверо
# дороже без выигрыша (замер на четырёх таблицах — в README).
DEFAULT_DPI = WORK_DPI

# Арбитр по стороне дорог (0.3 с на ячейку), поэтому зовётся только по тем ячейкам,
# которые дешёвые меры уже назвали боковыми.
ARBITER = "ocr_vote"

# Меньше стольких букв кириллицы в распознанном тексте — подменять нечем. Такая ячейка
# остаётся как была и попадает в отчёт: это либо ошибка детектора (колонка чисел), либо
# ячейка, которую движок не осилил, и в обоих случаях испортить её мы права не имеем.
MIN_LETTERS_TO_REPLACE = 3

CYRILLIC = re.compile(r"[А-Яа-яЁё]")


@dataclass
class CellOutcome:
    """Что случилось с одной ячейкой."""

    row: int
    col: int
    row_span: int
    col_span: int
    rotate_cw: int
    confidence: float
    text: str = ""
    ocr_confidence: float = 0.0
    votes: dict[str, float] = field(default_factory=dict)
    from_prior: bool = False


@dataclass
class TableOutcome:
    """Что случилось с одной таблицей."""

    crop_id: str
    grid: Grid
    cells: list[CellOutcome]
    prior: int
    before: np.ndarray
    after: "np.ndarray | None" = None
    widened: dict[int, int] = field(default_factory=dict)
    failed: list[tuple[int, int]] = field(default_factory=list)
    engine: str = ""
    decisions: list[CellDecision] = field(default_factory=list)

    @property
    def rotated_cells(self) -> list[CellOutcome]:
        return [cell for cell in self.cells if cell.rotate_cw]


def decide_rotation(
    table: np.ndarray, grid: Grid, dpi: int, detector_names: "tuple[str, ...] | None" = None
) -> tuple[list[CellDecision], int]:
    """Решить по каждой ячейке, повёрнута ли она, и назначить таблице общую сторону."""
    detectors = {detector.name: detector for detector in resolve_detectors(detector_names)}
    cheap = [detector for name, detector in detectors.items() if name != ARBITER and detector.stage == "cpu"]
    arbiter = detectors.get(ARBITER)

    decisions: list[CellDecision] = []
    for cell in grid.cells:
        crop = CellCrop("", cell, cell_image(table, cell, dpi), dpi, table=table)
        verdicts = {detector.name: detector.run(crop) for detector in cheap}
        decision = combine_axis(verdicts)
        if decision.rotated and arbiter is not None:
            verdicts[arbiter.name] = arbiter.run(crop)
            decision = combine_axis(verdicts)
        decisions.append(decision)
    return apply_prior(decisions)


def read_cells(
    table: np.ndarray, grid: Grid, decisions: list[CellDecision], dpi: int, engine_name: str
) -> tuple[dict[tuple[int, int], tuple[str, float]], str]:
    """Распознать боковые ячейки выбранным движком."""
    engines = resolve_engines((engine_name,))
    if not engines:
        logger.warning("Движок %s недоступен, ячейки останутся нераспознанными", engine_name)
        return {}, ""
    engine = engines[0]

    targets = [(cell, decision) for cell, decision in zip(grid.cells, decisions) if decision.rotated]
    if not targets:
        return {}, engine.name
    crops = [rotate_cw(cell_image(table, cell, dpi), decision.rotate_cw) for cell, decision in targets]
    results = engine.make()(crops)
    return {cell.key: (result.text, result.confidence) for (cell, _), result in zip(targets, results)}, engine.name


def process(
    crop_path: Path,
    dpi: int = DEFAULT_DPI,
    source_dpi: int = 600,
    detector_names: "tuple[str, ...] | None" = None,
    engine_name: str = "tesseract",
    render: bool = True,
) -> TableOutcome:
    """Вся цепочка по одной вырезанной таблице."""
    table = load_gray(crop_path, dpi, source_dpi)
    grid = extract(table, dpi)
    decisions, prior = decide_rotation(table, grid, dpi, detector_names)
    texts, engine = read_cells(table, grid, decisions, dpi, engine_name)

    outcomes = [
        CellOutcome(
            row=cell.row,
            col=cell.col,
            row_span=cell.row_span,
            col_span=cell.col_span,
            rotate_cw=decision.rotate_cw,
            confidence=round(decision.confidence, 3),
            text=texts.get(cell.key, ("", 0.0))[0],
            ocr_confidence=round(texts.get(cell.key, ("", 0.0))[1], 3),
            votes=decision.axis_votes,
            from_prior=decision.from_prior,
        )
        for cell, decision in zip(grid.cells, decisions)
    ]

    outcome = TableOutcome(crop_id=crop_path.stem, grid=grid, cells=outcomes, prior=prior, before=table, engine=engine)
    outcome.decisions = decisions
    if render:
        replacements = [
            Replacement(cell, texts[cell.key][0], decision.rotate_cw)
            for cell, decision in zip(grid.cells, decisions)
            if decision.rotated and len(CYRILLIC.findall(texts.get(cell.key, ("", 0.0))[0])) >= MIN_LETTERS_TO_REPLACE
        ]
        composed: Composed = compose(table, grid, replacements, dpi)
        # Обе картинки обрезаются по САМОЙ таблице: раздвижка идёт через всю высоту кадра
        # и разрезает то, что лежало под таблицей (подрисуночную подпись), а показывать
        # испорченную окрестность вместо результата незачем. Сетка обрезается вместе с
        # картинкой — иначе оверлей рисуется в чужой системе координат.
        outcome.before, outcome.grid = trim(table, grid, dpi)
        outcome.after, _ = trim(composed.image, composed.grid, dpi)
        outcome.widened = composed.widened
        outcome.failed = composed.failed
    return outcome


def overlay(table: np.ndarray, grid: Grid, decisions: list[CellDecision]) -> np.ndarray:
    """Оверлей для глаз: боковые ячейки красным, прочие зелёным, шапка потолще."""
    canvas = cv2.cvtColor(table, cv2.COLOR_GRAY2BGR)
    for cell, decision in zip(grid.cells, decisions):
        colour = (0, 0, 220) if decision.rotated else (0, 150, 0)
        thickness = 4 if cell.is_header else 2
        cv2.rectangle(canvas, (cell.box.x0, cell.box.y0), (cell.box.x1, cell.box.y1), colour, thickness)
    return canvas
