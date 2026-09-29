"""Оценка защиты сторон по разметке: что стало с каждым куском сора с выступом (убран CRAFT, помечен недостоверным, пропущен) и с размеченными знаками (выброшены ли, помечены ли зря)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from research.edge_marks.candidates import Candidate

# Рендер 300 dpi → рабочая копия 150 dpi.
K = 2.0


def _overlaps(box: tuple, other: tuple) -> bool:
    """Пересекаются ли рамки ``x0, y0, x1, y1``."""
    return min(box[2], other[2]) > max(box[0], other[0]) and min(box[3], other[3]) > max(box[1], other[1])


def _marked(candidate: Candidate, page: dict) -> bool:
    """Строка кандидата попала в недостоверный участок стороны своего блока на своей стороне.

    Блок — тот, чья сторона ближе всех к стороне кандидата по высоте его строки.
    """
    y = candidate.row_y / K
    x = candidate.side_x / K
    best, spans = None, []
    for block in page["blocks"]:
        curve = block["envelope"][candidate.side]
        near = [p for p in curve if abs(p[1] - y) <= 2]
        if not near:
            continue
        distance = min(abs(p[0] - x) for p in near)
        if best is None or distance < best:
            best, spans = distance, block["envelope"][f"unreliable_{candidate.side}"]
    return any(y0 <= y <= y1 for y0, y1 in spans)


def outcomes(candidates: list[Candidate], mode_dir: Path, ids: set[str]) -> dict[str, str]:
    """Исход по каждому кандидату из ``ids`` в папке режима.

    Исходы: ``removed`` — компонента выброшена фильтром CRAFT; ``marked`` — участок стороны помечен
    недостоверным; ``left`` — ни то, ни другое.

    Args:
        candidates: Кандидаты тестового множества.
        mode_dir: ``guard/<набор>/<режим>`` с ``pages/<ключ>.json``.
        ids: Какие кандидаты оценивать.

    Returns:
        ``id → исход``.
    """
    out = {}
    for c in candidates:
        if c.id not in ids:
            continue
        path = mode_dir / "pages" / f"{c.key}.json"
        if not path.exists():
            out[c.id] = "left"
            continue
        page = json.loads(path.read_text())
        dropped = page["edge_guard"]["dropped"]
        if any(_overlaps(c.box, box) for box in dropped):
            out[c.id] = "removed"
        elif _marked(c, page):
            out[c.id] = "marked"
        else:
            out[c.id] = "left"
    return out


def report(candidates: list[Candidate], base: Path, junk_ids: set[str], modes: tuple[str, ...]) -> dict:
    """Сводка исходов по режимам: сор с выступом и все размеченные знаки на полосах набора.

    Returns:
        ``{режим: {"сор": Counter, "знаки": Counter}}``.
    """
    keys = {p.stem for p in (base / "before" / "pages").glob("*.json")}
    signs = {c.id for c in candidates if c.truth == "sign" and c.key in keys}
    junk = {i for i in junk_ids if any(c.id == i and c.key in keys for c in candidates)}
    return {
        mode: {
            "сор": Counter(outcomes(candidates, base / mode, junk).values()),
            "знаки": Counter(outcomes(candidates, base / mode, signs).values()),
        }
        for mode in modes
    }


__all__ = ["outcomes", "report"]
