"""Общий вызов tesseract для детекторов разметки: PGM во временный файл, запуск, разбор TSV.

Один модуль на всех потребителей (ориентация OSD, арбитр ``ocr_vote``, детектор оглавлений),
чтобы соглашения не расходились: PGM вместо PNG (кодировать нечего, libpng не ругается на
профиль ICC, на 12 тысячах полос это заметная экономия), ``OMP_THREAD_LIMIT=1`` (без него
tesseract разойдётся по всем ядрам ПОВЕРХ пула процессов, и воркеры начнут отбирать ядра
друг у друга), общий таймаут и одинаковый разбор TSV.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Сколько ждать tesseract на одну полосу. Штатно уходит 0.5-3 с; минута — это уже затык.
TIMEOUT_S = 60

# Язык распознавания по умолчанию. Пак — советский отраслевой журнал, латиница на нём
# встречается только в формулах и марках оборудования.
LANGUAGE = "rus"


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def write_pgm(path: Path, gray: np.ndarray) -> None:
    """Голый P5 без сжатия: кодировать нечего, а читается любым tesseract."""
    height, width = gray.shape[:2]
    with open(path, "wb") as handle:
        handle.write(b"P5\n%d %d\n255\n" % (width, height))
        handle.write(np.ascontiguousarray(gray, dtype=np.uint8).tobytes())


def run(gray: np.ndarray, args: list[str], prefix: str = "tess_") -> str:
    """stdout tesseract по серой картинке; пустая строка, если он не справился.

    ``args`` — всё после имени входного файла и ``-`` (например ``["--psm", "6", "-l", "rus", "tsv"]``).
    """
    with tempfile.TemporaryDirectory(prefix=prefix) as work:
        image = Path(work) / "page.pgm"
        write_pgm(image, gray)
        env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
        try:
            done = subprocess.run(
                ["tesseract", str(image), "-", *args], capture_output=True, text=True, env=env, timeout=TIMEOUT_S
            )
        except (OSError, subprocess.SubprocessError):
            return ""
    return done.stdout


def tsv_lines(gray: np.ndarray, psm: int = 6, language: str = LANGUAGE) -> list[str]:
    """Строки TSV-вывода tesseract без заголовка; пустой список, если он не справился."""
    out = run(gray, ["--psm", str(psm), "-l", language, "tsv"], prefix="tesstsv_")
    return out.splitlines()[1:] if out else []


@dataclass(frozen=True)
class Word:
    """Одно слово из TSV tesseract: где стоит (пиксели поданной картинки) и что прочитано."""

    block: int
    paragraph: int
    line: int
    left: int
    top: int
    width: int
    height: int
    confidence: float
    text: str

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def line_key(self) -> tuple[int, int, int]:
        """Строка текста однозначно задаётся тройкой блок/абзац/строка."""
        return self.block, self.paragraph, self.line


def words(gray: np.ndarray, psm: int = 6, language: str = LANGUAGE) -> list[Word]:
    """Слова полосы (уровень 5 в TSV) в порядке чтения tesseract. Пустые слова опускаются."""
    result: list[Word] = []
    for row in tsv_lines(gray, psm, language):
        columns = row.split("\t")
        if len(columns) < 12 or columns[0] != "5":
            continue
        text = columns[11].strip()
        if not text:
            continue
        try:
            result.append(
                Word(
                    int(columns[2]),
                    int(columns[3]),
                    int(columns[4]),
                    int(columns[6]),
                    int(columns[7]),
                    int(columns[8]),
                    int(columns[9]),
                    float(columns[10]),
                    text,
                )
            )
        except ValueError:
            continue
    return result


__all__ = ["LANGUAGE", "TIMEOUT_S", "Word", "run", "tesseract_available", "tsv_lines", "words", "write_pgm"]
