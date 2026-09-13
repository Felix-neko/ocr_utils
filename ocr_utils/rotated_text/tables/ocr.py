"""Чтение ячейки tesseract-ом: подпроцесс, TSV, строки с уверенностью.

ПОЧЕМУ TESSERACT. На 37 боковых ячейках с эталоном (старое исследование, отчёт
``reports/table_processing_ocr.md``) tesseract дал CER 0.034 и 28 точных ячеек за 0.05 с
на ячейку; PaddleOCR — 0.074 за 2.15 с; surya — медианный ноль при среднем 2.36, потому что
на короткой надписи в узкой графе дописывала выдуманный текст. Поэтому tesseract — основной
читатель, surya — второе мнение с фильтром (``second_opinion``).

ВХОД — УЖЕ ВЫПРЯМЛЕННАЯ вырезка внутренности ячейки: поворот делается снаружи, одинаково
для всех движков. ``--psm 6`` («один однородный блок»): ячейка и есть один блок, заголовок
в две-три строки без колонок.

ВЫХОД — СТРОКИ, а не строка: боковой заголовок почти всегда набран в несколько строк с
переносами («матери- / ально-техни- / ческих»), и склеивать их надо с расшивкой переноса,
а не пробелом. Склейка — :func:`join_lines`.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_markup.orientation.detectors.osd import TIMEOUT_S, _write_pgm, tesseract_available

__all__ = ["CellText", "LANGUAGES", "join_lines", "prepare", "read_cell", "tesseract_available", "tesseract_tsv"]

# Язык. Журнал советский отраслевой, латиница встречается только в марках оборудования;
# добавление ``eng`` подменяет похожие кириллические буквы латинскими и замедляет в полтора
# раза, поэтому по умолчанию только русский. Переопределяется опцией ``--lang``.
LANGUAGES = "rus"

# Слово идёт в текст, если tesseract уверен в нём не меньше этого. Ниже — россыпь мусора.
MIN_WORD_CONFIDENCE = 30.0

# Поля вокруг вырезки: на прижатом к краю тексте точность заметно ниже.
PAD_PX = 16

# Ниже этой стороны вырезка увеличивается вдвое: tesseract теряет точность, когда знак ниже
# 20 px, а петит шапки при 300 dpi даёт как раз около 20.
MIN_SIDE_PX = 300
UPSCALE = 2

# Мусор по краям строки: один-два знака без букв и цифр — обрубок линейки, прочитанный как
# «|», «—», «©». Только по краям: внутри такой знак бывает настоящим («1966—1970»).
JUNK_EDGE = re.compile(r"^[^\w]{1,2}(?=\s|$)|(?<=\s)[^\w]{1,2}$", re.UNICODE)

LOWERCASE = re.compile(r"^[а-яёa-z]")
HYPHENS = ("-", "‐", "‑")
DASHES = ("—", "–")


@dataclass
class CellText:
    """Что прочитано в одной ячейке."""

    lines: list[str] = field(default_factory=list)
    confidence: float = 0.0
    engine: str = ""
    seconds: float = 0.0

    @property
    def text(self) -> str:
        return join_lines(self.lines)


def strip_junk(line: str) -> str:
    previous = None
    current = line.strip()
    while current != previous:
        previous = current
        current = JUNK_EDGE.sub("", current).strip()
    return current


def join_lines(lines: list[str]) -> str:
    """Строки в одну с расшивкой переносов.

    «матери-» + «ально» → «материально»: дефис на конце строки перед строчной буквой — это
    перенос. «1966—» + «1970» → «1966—1970»: тире склеивается без пробела в обе стороны, а повторённое
    на новой строке тире («к 1961—» + «—1965») остаётся одно.
    Дефис перед заглавной или цифрой («Северо-» + «Западный») остаётся, но пробел не ставится.
    """
    joined = ""
    for raw in lines:
        piece = strip_junk(raw)
        if not piece:
            continue
        if not joined:
            joined = piece
        elif joined.endswith(HYPHENS):
            joined = (joined[:-1] if LOWERCASE.match(piece) else joined) + piece
        elif joined.endswith(DASHES) and piece.startswith(DASHES):
            # Тире, повторённое в начале следующей строки, — норма советского набора
            # («к 1961— / —1965 гг.»); в тексте оно одно.
            joined += piece.lstrip("".join(DASHES))
        elif joined.endswith(DASHES) or piece.startswith(DASHES):
            joined += piece
        else:
            joined += " " + piece
    return joined


def prepare(gray: np.ndarray) -> np.ndarray:
    """Поля цветом бумаги и увеличение мелкой вырезки — общая подготовка для чтения."""
    image = gray
    if min(image.shape[:2]) < MIN_SIDE_PX:
        image = cv2.resize(image, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    paper = int(np.percentile(image, 90)) if image.size else 255
    return cv2.copyMakeBorder(image, PAD_PX, PAD_PX, PAD_PX, PAD_PX, cv2.BORDER_CONSTANT, value=paper)


def tesseract_tsv(gray: np.ndarray, lang: str = LANGUAGES, psm: int = 6) -> list[str]:
    """Строки TSV-вывода tesseract без заголовка; пусто, если он не справился."""
    with tempfile.TemporaryDirectory(prefix="rottext_") as work:
        image = Path(work) / "cell.pgm"
        _write_pgm(image, gray)
        # OMP_THREAD_LIMIT=1: иначе tesseract разойдётся по всем ядрам поверх пула процессов.
        env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
        try:
            done = subprocess.run(
                ["tesseract", str(image), "-", "--psm", str(psm), "-l", lang, "tsv"],
                capture_output=True,
                text=True,
                env=env,
                timeout=TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError):
            return []
    return done.stdout.splitlines()[1:]


def read_cell(upright: np.ndarray, lang: str = LANGUAGES) -> CellText:
    """Прочитать выпрямленную внутренность ячейки."""
    started = time.time()
    rows: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
    confidences: list[float] = []
    for row in tesseract_tsv(prepare(upright), lang):
        columns = row.split("\t")
        if len(columns) < 12 or not columns[11].strip():
            continue
        try:
            confidence = float(columns[10])
        except ValueError:
            continue
        if confidence < MIN_WORD_CONFIDENCE:
            continue
        key = (int(columns[2]), int(columns[3]), int(columns[4]))  # блок, абзац, строка
        rows.setdefault(key, []).append((int(columns[6]), columns[11]))
        confidences.append(confidence)
    lines = [" ".join(word for _, word in sorted(words)) for _, words in sorted(rows.items())]
    return CellText(
        lines=[line for line in lines if line.strip()],
        confidence=float(np.mean(confidences)) / 100.0 if confidences else 0.0,
        engine="tesseract",
        seconds=time.time() - started,
    )
