"""Разметка стыков множества: блоки под абзацем (подпись, город, эпиграф) — где разрез ожидается и где нет; поиск рядов разбора у стыка и проверка разреза."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ocr_utils.page_layout.text_blocks import WORK_DPI

MM_PER_INCH = 25.4
# Ряд разбора у стыка: верх ряда не дальше стольких мм от верха строки tesseract (рамки tesseract и рядов разбора
# расходятся на долю высоты строки; шаг строк ХиП 1991 ~3.8 мм).
ROW_MATCH_MM = 1.8


class Expect(str, Enum):
    """Ожидание на стыке: разрез (новый блок начинается с нижнего ряда) или один блок."""

    CUT = "cut"
    KEEP = "keep"


@dataclass(frozen=True)
class Joint:
    """Стык двух соседних строк одной колонки.

    Поля:
        page: Ключ страницы ``01/IMG_1008_R``.
        y_mm: Верх нижней строки стыка, мм от верха вырезки.
        x_mm: X середины нижней строки, мм (выбор колонки).
        expect: Ожидание.
        kind: Вид блока снизу (``signature``, ``city``, ``epigraph``, ``right_other``) или вид стыка-«не разреза»
            (``signature_inner`` — внутри подписи, ``paragraph_inner`` — внутри абзаца над подписью).
        note: Пометка разметчика.
    """

    page: str
    y_mm: float
    x_mm: float
    expect: Expect
    kind: str
    note: str = ""


FIELDS = ["page", "y_mm", "x_mm", "expect", "kind", "note"]


def load_joints(path: Path) -> list[Joint]:
    """Стыки из CSV множества (колонки :data:`FIELDS`)."""
    with path.open() as fh:
        return [Joint(r["page"], float(r["y_mm"]), float(r["x_mm"]), Expect(r["expect"]), r["kind"], r.get("note", ""))
                for r in csv.DictReader(fh)]  # fmt: skip


def save_joints(joints: list[Joint], path: Path) -> None:
    """Стыки в CSV множества."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FIELDS)
        for j in joints:
            writer.writerow([j.page, f"{j.y_mm:.1f}", f"{j.x_mm:.1f}", j.expect.value, j.kind, j.note])


def mm_to_work(mm: float) -> float:
    """Мм вырезки → px рабочей копии разбора (``WORK_DPI``)."""
    return mm * WORK_DPI / MM_PER_INCH


def joint_rows(blocks: list[dict], joint: Joint) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Ряды разбора по обе стороны стыка: ``((блок, ряд) сверху, (блок, ряд) снизу)``.

    Нижний — ряд, накрывающий ``x`` стыка, с верхом ближе всего к ``y`` (не дальше :data:`ROW_MATCH_MM`). Верхний —
    ближайший выше ряд, перекрытый с нижним по x.

    Args:
        blocks: Блоки сводки разбора (``bench.summary``: ряды ``[y, x0, x1, h, …]``).
        joint: Стык.

    Returns:
        Индексы рядов или None — ряд у стыка не найден.
    """
    y, x, tol = mm_to_work(joint.y_mm), mm_to_work(joint.x_mm), mm_to_work(ROW_MATCH_MM)
    rows = [(bi, ri, r) for bi, b in enumerate(blocks) for ri, r in enumerate(b["rows"])]
    below = [(abs((r[0] - r[3] / 2) - y), bi, ri, r) for bi, ri, r in rows if r[1] - 2 <= x <= r[2] + 2]
    below = [item for item in below if item[0] <= tol]
    if not below:
        return None
    _, bb, rb, rrow = min(below)
    above = [
        (r[0], bi, ri) for bi, ri, r in rows if r[0] < rrow[0] - rrow[3] / 2 and min(r[2], rrow[2]) > max(r[1], rrow[1])
    ]
    if not above:
        return None
    _, ba, ra = max(above)
    return (ba, ra), (bb, rb)


def joint_cut(blocks: list[dict], joint: Joint) -> bool | None:
    """Разрезан ли стык в разборе: ряды по обе стороны — в разных блоках.

    Args:
        blocks: Блоки сводки разбора.
        joint: Стык.

    Returns:
        True — разрез, False — один блок, None — ряды у стыка не найдены.
    """
    pair = joint_rows(blocks, joint)
    if pair is None:
        return None
    (ba, _), (bb, _) = pair
    return ba != bb


def _joint(page: str, line, expect: Expect, kind: str, note: str) -> Joint:
    """Стык над строкой tesseract ``line`` (верх строки и середина по x)."""
    return Joint(page, line.y0, (line.x0 + line.x1) / 2, expect, kind, note)


def draft_joints(draft: Path, ocr_dir: Path) -> list[Joint]:
    """Стыки множества из черновой разметки по тексту (``draft_*.csv``) и строк tesseract.

    Каждая запись черновика — блок, начинающийся со строки ``first_y_mm`` колонки ``column``:
    - стык над первой строкой — ``cut`` (вид — ``kind`` записи);
    - стыки между строками блока (``n_lines``) — ``keep`` ``signature_inner`` (фамилия → регалии, второй автор);
    - стык между двумя последними строками над блоком — ``keep`` ``paragraph_inner`` (концовка абзаца остаётся в
      абзаце), если над блоком абзац (``after_paragraph = yes``).

    Args:
        draft: CSV черновика (``page, column, kind, first_y_mm, n_lines, first_text, after_paragraph, note``).
        ocr_dir: Папка CSV строк tesseract (``<выпуск>/<кадр>_<L|R>.csv``).

    Returns:
        Стыки; записи, для которых строка tesseract не нашлась, пропущены.
    """
    from research.flush_split.candidates import MIN_LETTERS, columns_of, letters
    from research.flush_split.ocr import load_lines

    issues = {p.name[:2]: p for p in ocr_dir.iterdir() if p.is_dir()}
    out: list[Joint] = []
    with draft.open() as fh:
        for rec in csv.DictReader(fh):
            issue, stem = rec["page"].split("/")
            path = issues[issue] / f"{stem}.csv"
            lines = [ln for ln in load_lines(path) if letters(ln.text) >= MIN_LETTERS]
            cols = columns_of(lines)
            col = cols[min(int(rec["column"] or 0), len(cols) - 1)]
            y = float(rec["first_y_mm"])
            first = min(range(len(col)), key=lambda i: abs(col[i].y0 - y))
            if abs(col[first].y0 - y) > 2.0:
                continue
            note = (rec.get("note") or "").strip()

            out.append(_joint(rec["page"], col[first], Expect.CUT, rec["kind"], note))
            n = max(int(rec["n_lines"] or 1), 1)
            out += [
                _joint(rec["page"], col[i], Expect.KEEP, f"{rec['kind']}_inner", note)
                for i in range(first + 1, min(first + n, len(col)))
            ]
            if (rec.get("after_paragraph") or "").strip() == "yes" and first >= 2:
                out.append(_joint(rec["page"], col[first - 1], Expect.KEEP, "paragraph_inner", note))
    return out


# Стык «один блок» ближе стольких мм к стыку «разрез» той же страницы и колонки — противоречие разметки (на страницах
# афоризмов концовка одной цитаты и автор предыдущей стоят на соседних строках, и стыки черновика совпадают).
CONFLICT_MM = 1.5
# Стыки одной колонки: середины по x ближе стольких мм.
SAME_COLUMN_MM = 40.0


def drop_conflicts(joints: list[Joint]) -> tuple[list[Joint], int]:
    """Убрать стыки ``keep``, совпадающие со стыком ``cut`` той же страницы и колонки (:data:`CONFLICT_MM`).

    Args:
        joints: Стыки.

    Returns:
        (стыки без противоречий, сколько убрано).
    """
    cuts = [j for j in joints if j.expect is Expect.CUT]
    kept = [
        j
        for j in joints
        if j.expect is Expect.CUT
        or not any(
            c.page == j.page and abs(c.y_mm - j.y_mm) <= CONFLICT_MM and abs(c.x_mm - j.x_mm) <= SAME_COLUMN_MM
            for c in cuts
        )
    ]
    return kept, len(joints) - len(kept)
