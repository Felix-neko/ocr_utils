"""Распознавание ячейки через tesseract (подпроцессом, как везде в проекте).

Он здесь не фаворит, а точка отсчёта: движок без GPU, без весов и без скачиваний, зато
с кириллицей из коробки. Если что-то из тяжёлого не обгоняет его на боковых ячейках, то
тяжёлое не нужно.

``--psm 6`` («один однородный блок текста») выбран потому, что ячейка — это и есть один
блок: заголовок в две-три строки без колонок и без заголовков внутри.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ocr_utils.scan_markup.orientation.detectors.osd import TIMEOUT_S, _write_pgm, tesseract_available

from research.legacy.table_processing.ocr.base import Engine, OcrResult, prepare

LANGUAGE = "rus"

# Слово идёт в текст, если tesseract уверен в нём не меньше этого. Порог тот же, что у
# арбитра ориентации полос, и по той же причине: без него на выходе россыпь мусора.
MIN_WORD_CONFIDENCE = 30.0

CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")


def _tsv(gray: np.ndarray) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="tabocr_") as work:
        image = Path(work) / "cell.pgm"
        _write_pgm(image, gray)
        # OMP_THREAD_LIMIT=1: иначе tesseract разойдётся по всем ядрам поверх пула процессов.
        env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
        try:
            done = subprocess.run(
                ["tesseract", str(image), "-", "--psm", "6", "-l", LANGUAGE, "tsv"],
                capture_output=True,
                text=True,
                env=env,
                timeout=TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError):
            return []
    return done.stdout.splitlines()[1:]


class TesseractRecognizer:
    def __call__(self, images: Sequence[np.ndarray]) -> list[OcrResult]:
        results: list[OcrResult] = []
        for gray in images:
            started = time.time()
            rows: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
            confidences: list[float] = []
            for row in _tsv(prepare(gray)):
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
                rows.setdefault(key, []).append((int(columns[5]), columns[11]))
                confidences.append(confidence)
            lines = [" ".join(word for _, word in sorted(words)) for _, words in sorted(rows.items())]
            results.append(
                OcrResult(
                    lines=[line for line in lines if line.strip()],
                    confidence=float(np.mean(confidences)) / 100.0 if confidences else 0.0,
                    seconds=time.time() - started,
                )
            )
        return results


ALGORITHM = Engine(
    name="tesseract",
    summary="tesseract 5, -l rus --psm 6 (CPU, подпроцессом)",
    stage="cpu",
    make=TesseractRecognizer,
    available=tesseract_available,
)
