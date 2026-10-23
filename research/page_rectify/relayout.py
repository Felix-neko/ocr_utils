"""Перекладка прежней раскладки прогона (``<набор>[/<выпуск>]/{output,debug,rectify.csv}``) в общую: ``output/<подпуть>``, ``debug/<подпуть>``, ``summary/<подпуть>.csv``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Имена в прежней раскладке: папки итогов и листов, сводка набора.
OLD_OUTPUT = "output"
OLD_DEBUG = "debug"
OLD_SUMMARY = "rectify.csv"
# Корневые папки новой раскладки: в них не ищем прежние наборы.
NEW_ROOTS = ("output", "debug", "summary")


@dataclass(frozen=True)
class Move:
    """Перенос: откуда и куда."""

    src: Path
    dst: Path


def plan_moves(root: Path) -> list[Move]:
    """Переносы для всех прежних наборов под ``root``.

    Набор прежней раскладки — папка, в которой лежит ``rectify.csv`` (рядом ``output/`` и ``debug/``); подпуть набора —
    её путь относительно ``root`` (``ph_1975``, ``hp_1991/01 оплачено``).

    Args:
        root: Корень прогона.

    Returns:
        Переносы папок итогов и листов и файла сводки.

    Raises:
        FileExistsError: Цель переноса уже есть (повторная перекладка поверх новой раскладки).
    """
    moves = []
    for summary in sorted(root.rglob(OLD_SUMMARY)):
        set_dir = summary.parent
        sub = set_dir.relative_to(root)
        if sub.parts and sub.parts[0] in NEW_ROOTS:
            continue
        candidates = [
            Move(set_dir / OLD_OUTPUT, root / "output" / sub),
            Move(set_dir / OLD_DEBUG, root / "debug" / sub),
            Move(summary, root / "summary" / sub.with_name(sub.name + ".csv")),
        ]
        for move in candidates:
            if not move.src.exists():
                continue
            if move.dst.exists():
                raise FileExistsError(f"цель уже есть: {move.dst}")
            moves.append(move)
    return moves


def apply_moves(root: Path, moves: list[Move]) -> int:
    """Выполняет переносы (``rename`` на том же диске) и удаляет опустевшие папки прежних наборов.

    Args:
        root: Корень прогона.
        moves: Переносы из :func:`plan_moves`.

    Returns:
        Сколько папок удалено.
    """
    for move in moves:
        move.dst.parent.mkdir(parents=True, exist_ok=True)
        move.src.rename(move.dst)
    removed = 0
    # Снизу вверх: сначала папки выпусков, потом наборов.
    for folder in sorted({m.src.parent for m in moves}, key=lambda p: len(p.parts), reverse=True):
        while folder != root and folder.exists() and not any(folder.iterdir()):
            folder.rmdir()
            removed += 1
            folder = folder.parent
    return removed


def count_files(folder: Path) -> int:
    """Число файлов под папкой (для сверки до и после)."""
    return sum(1 for p in folder.rglob("*") if p.is_file())
