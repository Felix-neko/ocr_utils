"""Судья-правило (точка отсчёта): шаблоны знаков прошлого стенда (``text_block_specks.rules.kind_of``) по крайнему компоненту строки в рабочей копии."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from research.edge_marks.candidates import load
from research.edge_marks.judges.common import Scale, Verdict, Writer
from research.text_block_specks.filters import End, end_component, work_binary
from research.text_block_specks.rules import SIGNS, Kind, kind_of

K = RENDER_DPI / WORK_DPI


def line_glyphs(binary: np.ndarray, line_box: tuple, row_y: float, x_h: float, box: tuple, right: bool) -> np.ndarray:
    """Боксы компонентов краски строки в рабочей копии: центр по высоте в полосе строки, в пределах вырезки строки.

    Args:
        binary: Бинарная рабочая копия.
        line_box: Рамка строки, пиксели рендера.
        row_y: Середина строки, пиксели рендера.
        x_h: x-высота, пиксели рендера.
        box: Рамка кандидата, пиксели рендера: компоненты, накрывающие её, берутся всегда — сор бывает выше или
            ниже середины строки, и без этого крайним компонентом строки оказывалась последняя буква.
        right: Кандидат у правого конца строки; компоненты дальше кандидата (соседняя колонка) не берутся.

    Returns:
        Массив ``(n, 4)`` — ``x0, y0, x1, y1`` в пикселях рабочей копии.
    """
    x0, y0, x1, y1 = (int(round(v / K)) for v in line_box)
    part = binary[max(0, y0) : y1, max(0, x0) : x1].astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(part, 8)
    out = []
    band = 0.9 * x_h / K
    bx0, by0, bx1, by1 = (v / K for v in box)
    for index in range(1, count):
        x, y, w, h = stats[index, :4]
        gx0, gy0, gx1, gy1 = x0 + x, y0 + y, x0 + x + w, y0 + y + h
        centre = (gy0 + gy1) / 2.0
        touches = min(gx1, bx1) > max(gx0, bx0) and min(gy1, by1) > max(gy0, by0)
        # За кандидатом — уже соседняя колонка или поле: такие компоненты не этой строки.
        beyond = gx0 >= bx1 + 1 if right else gx1 <= bx0 - 1
        if beyond:
            continue
        if abs(centre - row_y / K) <= band or touches:
            out.append([gx0, gy0, gx1, gy1])
    return np.asarray(out, dtype=np.float64).reshape(-1, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cand-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    writer = Writer(args.out)
    writer.loaded(device="cpu")
    binaries: dict[str, np.ndarray] = {}
    for candidate in load(args.cand_dir):
        started = time.monotonic()
        if candidate.key not in binaries:
            gray = cv2.imread(str(args.cand_dir / "pages" / f"{candidate.key}.png"), cv2.IMREAD_GRAYSCALE)
            binaries = {candidate.key: work_binary(gray)}
        binary = binaries[candidate.key]
        glyphs = line_glyphs(
            binary, candidate.line_box, candidate.row_y, candidate.x_h, candidate.box, candidate.side == "right"
        )
        end = End.RIGHT if candidate.side == "right" else End.LEFT
        component = end_component(glyphs, binary, end)
        if component is None:
            writer.write(candidate.id, Scale.COMPONENT, Verdict.UNSURE, 0.5, time.monotonic() - started, "мало глифов")
            continue
        kind = kind_of(component, end)
        if kind in (Kind.SPECK, Kind.MARK):
            verdict, score = Verdict.JUNK, 1.0
        elif kind in SIGNS:
            verdict, score = Verdict.SIGN, 0.0
        else:
            verdict, score = Verdict.UNSURE, 0.5
        writer.write(candidate.id, Scale.COMPONENT, verdict, score, time.monotonic() - started, kind.value)
    writer.close()


if __name__ == "__main__":
    main()
