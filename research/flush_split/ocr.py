"""Tesseract по вырезкам холста Кунса: строки текста с рамками в мм (TSV tesseract → таблица строк)."""

from __future__ import annotations

import csv
import io
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Язык и режим сегментации tesseract: автоматическая разметка страницы (колонки, блоки, абзацы).
LANG = "rus"
PSM = 3


@dataclass(frozen=True)
class OcrLine:
    """Строка tesseract.

    Поля:
        block: Номер блока tesseract на странице.
        par: Номер абзаца в блоке.
        line: Номер строки в абзаце.
        x0, y0, x1, y1: Рамка строки, мм от левого верхнего угла вырезки.
        text: Текст строки (слова через пробел).
        conf: Средняя уверенность слов, 0–100.
    """

    block: int
    par: int
    line: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    conf: float


def run_tesseract(png: Path) -> str:
    """TSV tesseract по картинке.

    Args:
        png: Путь к вырезке.

    Returns:
        Текст TSV (уровни page/block/par/line/word).
    """
    # Один поток на процесс: параллелизм — пулом по вырезкам.
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    out = subprocess.run(["tesseract", str(png), "stdout", "-l", LANG, "--psm", str(PSM), "tsv"], env=env,
                         capture_output=True, text=True, check=True)  # fmt: skip
    return out.stdout


def lines_of(tsv: str, dpi: float) -> list[OcrLine]:
    """Строки из TSV tesseract: слова одной строки собираются, рамка — охват слов.

    Args:
        tsv: Вывод tesseract в формате TSV.
        dpi: Разрешение вырезки (перевод px → мм).

    Returns:
        Строки сверху вниз в порядке tesseract; пустые (без слов) пропущены.
    """
    px_mm = dpi / 25.4
    words: dict[tuple[int, int, int], list[dict]] = {}
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row.get("level") != "5" or not (row.get("text") or "").strip():
            continue
        key = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        words.setdefault(key, []).append(row)
    out = []
    for (block, par, line), ws in words.items():
        x0 = min(int(w["left"]) for w in ws)
        y0 = min(int(w["top"]) for w in ws)
        x1 = max(int(w["left"]) + int(w["width"]) for w in ws)
        y1 = max(int(w["top"]) + int(w["height"]) for w in ws)
        conf = sum(float(w["conf"]) for w in ws) / len(ws)
        text = " ".join(w["text"].strip() for w in ws)
        out.append(OcrLine(block, par, line, x0 / px_mm, y0 / px_mm, x1 / px_mm, y1 / px_mm, text, conf))
    return out


def save_lines(lines: list[OcrLine], path: Path) -> None:
    """Строки в CSV (по строке tesseract на строку файла).

    Args:
        lines: Строки.
        path: Куда писать.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["block", "par", "line", "x0", "y0", "x1", "y1", "conf", "text"])
        for ln in lines:
            writer.writerow([ln.block, ln.par, ln.line, f"{ln.x0:.2f}", f"{ln.y0:.2f}", f"{ln.x1:.2f}",
                             f"{ln.y1:.2f}", f"{ln.conf:.0f}", ln.text])  # fmt: skip


def load_lines(path: Path) -> list[OcrLine]:
    """Строки из CSV :func:`save_lines`.

    Args:
        path: CSV строк страницы.

    Returns:
        Строки в порядке файла.
    """
    with path.open() as fh:
        return [OcrLine(int(r["block"]), int(r["par"]), int(r["line"]), float(r["x0"]), float(r["y0"]),
                        float(r["x1"]), float(r["y1"]), r["text"], float(r["conf"])) for r in csv.DictReader(fh)]  # fmt: skip
