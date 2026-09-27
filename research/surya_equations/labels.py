"""Эталон формул глазами: блоки с ужатыми по краске рамками, вердикты боксам surya, чтение и запись JSONL."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from scipy import ndimage

LABELS_DIR = Path(__file__).parent / "labels"
BLOCKS_FILE = LABELS_DIR / "blocks.jsonl"
VERDICTS_FILE = LABELS_DIR / "verdicts.jsonl"
SAMPLE_FILE = LABELS_DIR / "sample.csv"

# Краска — темнее этого на заострённом JPEG (фон после размытия — 200–250).
INK_LEVEL = 140
# Пятна меньше этого числа пикселей при 600 dpi — пыль, в рамку формулы не входят.
MIN_SPECK_PX = 30


class Verdict(str, Enum):
    """Вердикт разметчика боксу surya ``Equation``."""

    FORMULA = "formula"  # накрывает выносную формулу или её часть
    FORMULA_TEXT_LINE = "formula_text_line"  # формула внутри строки абзаца, бокс взял строку прозы
    NOT_FORMULA = "not_formula"  # таблица, числа, текст, рисунок


@dataclass(frozen=True)
class Block:
    """Эталонный блок: выносная формула (строки подряд без прозы между ними) в кадре кэша 150 dpi."""

    box: tuple[float, float, float, float]  # ужатая по краске рамка
    rough: tuple[float, float, float, float]  # грубая рамка разметчика с полями по пустому месту
    lines: int | None = None  # формульных строк в блоке
    inline: bool = False  # двухэтажная формула в строке абзаца — в полноту не входит, считается отдельно
    note: str = ""


@dataclass(frozen=True)
class PageLabels:
    """Разметка одной полосы: эталонные блоки и вердикты боксам surya (ключ — номер бокса в кэше)."""

    page: str
    blocks: tuple[Block, ...] = ()
    verdicts: dict[int, tuple[Verdict, str]] = field(default_factory=dict)  # номер → (вердикт, комментарий)


def snap_to_ink(gray: np.ndarray, rough: tuple[float, float, float, float], scale: float) -> tuple[float, ...]:
    """Ужать грубую рамку до краски формулы.

    Пятна краски, касающиеся края грубой рамки, считаются соседним текстом и выбрасываются:
    разметчик рисует рамку с полями по пустому месту, поэтому буквы формулы края не касаются.
    Если после этого не осталось ничего — берутся все пятна крупнее пыли.

    Args:
        gray: серый JPEG полосы.
        rough: грубая рамка в кадре кэша.
        scale: пикселей JPEG на единицу кадра (4.0 для пака-1: 600 / 150 dpi).

    Returns:
        Рамка ``(x0, y0, x1, y1)`` в кадре кэша, округлённая до 0.1; пустая область — сама грубая рамка.
    """
    x0, y0, x1, y1 = (int(v * scale) for v in rough)
    ink = gray[y0:y1, x0:x1] < INK_LEVEL
    labelled, count = ndimage.label(ink)
    if count == 0:
        return tuple(rough)
    sizes = ndimage.sum(ink, labelled, range(1, count + 1))
    big = 1 + np.nonzero(sizes >= MIN_SPECK_PX)[0]
    # Номера пятен на краях грубой рамки — соседние строки прозы.
    border = set(np.unique(np.concatenate([labelled[0], labelled[-1], labelled[:, 0], labelled[:, -1]]))) - {0}
    inner = [c for c in big if c not in border]
    ys, xs = np.nonzero(np.isin(labelled, inner if inner else big))
    if len(xs) == 0:
        return tuple(rough)
    return (
        round((x0 + xs.min()) / scale, 1),
        round((y0 + ys.min()) / scale, 1),
        round((x0 + xs.max() + 1) / scale, 1),
        round((y0 + ys.max() + 1) / scale, 1),
    )


def parse_verdict(text: str) -> tuple[Verdict, str]:
    """Вердикт из строки разметчика вида ``"formula: обрезан знаменатель"``.

    Args:
        text: значение вердикта; комментарий после двоеточия необязателен.

    Returns:
        ``(вердикт, комментарий)``.
    """
    name, _, comment = text.partition(":")
    return Verdict(name.strip()), comment.strip()


def load_labels(blocks_file: Path = BLOCKS_FILE, verdicts_file: Path = VERDICTS_FILE) -> dict[str, PageLabels]:
    """Вся разметка стенда.

    Args:
        blocks_file: JSONL эталонных блоков (строка — полоса).
        verdicts_file: JSONL вердиктов (строка — полоса).

    Returns:
        Полоса → :class:`PageLabels`; полоса попадает сюда, только если у неё есть и блоки, и вердикты
        (разметка не закончена — полосы нет).
    """
    blocks = {}
    for line in blocks_file.read_text().splitlines():
        row = json.loads(line)
        blocks[row["page"]] = tuple(
            Block(tuple(b["box"]), tuple(b["rough"]), b.get("lines"), bool(b.get("inline")), b.get("note", ""))
            for b in row["blocks"]
        )
    labels = {}
    for line in verdicts_file.read_text().splitlines():
        row = json.loads(line)
        if row["page"] not in blocks:
            continue
        verdicts = {int(key.lstrip("S")): parse_verdict(value) for key, value in row["verdicts"].items()}
        labels[row["page"]] = PageLabels(row["page"], blocks[row["page"]], verdicts)
    return labels


def upsert_jsonl(path: Path, row: dict) -> None:
    """Заменить (или дописать) строку полосы ``row["page"]`` в JSONL, сохраняя порядок остальных.

    Args:
        path: файл JSONL; создаётся, если нет.
        row: запись с ключом ``page``.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    rows = [r for r in rows if r["page"] != row["page"]] + [row]
    rows.sort(key=lambda r: r["page"])
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def block_row(page: str, blocks: list[Block]) -> dict:
    """Строка JSONL эталона полосы из блоков (для :func:`upsert_jsonl`)."""
    return {"page": page, "blocks": [asdict(b) for b in blocks]}
