"""Слова tesseract на вырезке области line art: рамки, текст, уверенность — в пикселях вырезки."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ocr_utils.rotated_text.tables.ocr import PAD_PX, prepare, tesseract_tsv

# Режимы разбиения страницы tesseract, которыми читается вырезка: 11 — разреженный текст без
# порядка (заголовок среди рисунка), 6 — один сплошной блок (чистый заголовок).
PSM_MODES = (11, 6)

# Язык распознавания: русский без английского (английский подменяет кириллицу латиницей,
# см. ``rotated_text.tables.ocr.LANGUAGES``).
LANGUAGE = "rus"


@dataclass(frozen=True)
class Word:
    """Слово tesseract.

    Attributes:
        x0: Левый край рамки слова в пикселях вырезки.
        y0: Верхний край.
        x1: Правый край (за последним пикселем).
        y1: Нижний край.
        conf: Уверенность tesseract, 0–100.
        text: Прочитанный текст.
        line: Номер строки tesseract внутри блока и абзаца — ``"блок.абзац.строка"``.
    """

    x0: int
    y0: int
    x1: int
    y1: int
    conf: float
    text: str
    line: str


def read_words(gray: np.ndarray, psm: int) -> list[Word]:
    """Прочитать вырезку tesseract-ом и вернуть непустые слова в координатах самой вырезки.

    Args:
        gray: Серая вырезка (краска тёмная).
        psm: Режим разбиения страницы tesseract.

    Returns:
        Слова с непустым текстом и уверенностью ≥ 0 (служебные строки TSV отброшены).
    """
    prepared = prepare(gray)
    # ``prepare`` мог увеличить вырезку вдвое — узнаём во сколько по размеру без полей.
    factor = (prepared.shape[1] - 2 * PAD_PX) / max(1, gray.shape[1])
    words: list[Word] = []
    for row in tesseract_tsv(prepared, LANGUAGE, psm):
        parts = row.split("\t")
        if len(parts) < 12 or parts[0] != "5":
            continue
        text = parts[11].strip()
        conf = float(parts[10])
        if not text or conf < 0:
            continue
        left, top, width, height = (int(v) for v in parts[6:10])
        words.append(
            Word(
                int(round((left - PAD_PX) / factor)),
                int(round((top - PAD_PX) / factor)),
                int(round((left + width - PAD_PX) / factor)),
                int(round((top + height - PAD_PX) / factor)),
                conf,
                text,
                f"{parts[2]}.{parts[3]}.{parts[4]}",
            )
        )
    return words


def read_region(crop_path: str) -> dict[str, list[dict]]:
    """Все режимы ``PSM_MODES`` по одной вырезке.

    Args:
        crop_path: Путь к серому PNG вырезки.

    Returns:
        ``{"psm11": [слово как словарь, …], "psm6": […]}``.
    """
    gray = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
    return {f"psm{psm}": [asdict(word) for word in read_words(gray, psm)] for psm in PSM_MODES}
