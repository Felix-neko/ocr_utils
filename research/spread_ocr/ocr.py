"""Чтение полосы разворота tesseract-ом: строки текста по блокам, номер страницы и колонтитул из верхней строки."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Язык tesseract: журнал русский, латиница попадается в названиях фирм.
LANGUAGES = "rus+eng"
# Предел времени на одну полосу, с.
TIMEOUT_S = 600
# Слова с уверенностью ниже этой в текст не идут (мусор с краёв, пальцев и сгиба).
MIN_WORD_CONFIDENCE = 40.0
# Буква или цифра: слова только из знаков препинания и рамок — мусор с кромки листов и сгиба.
_ALNUM = re.compile(r"[0-9A-Za-zА-Яа-яЁё]")
# Слово из трёх и более букв подряд — признак настоящей строки текста.
_REAL_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")
# Колонтитул ищется в верхней доле высоты полосы (строки выше — кромка соседних листов).
HEADER_ZONE = 0.16
# Номер страницы — 1–3 цифры, иногда с прилипшими знаками.
_NUMBER = re.compile(r"^\D{0,2}(\d{1,3})\D{0,2}$")


@dataclass(frozen=True)
class Word:
    """Слово tesseract.

    Attributes:
        text: Текст слова.
        left, top, width, height: Рамка слова в px полосы.
        confidence: Уверенность 0..100.
        line_key: Ключ строки (блок, абзац, строка) из TSV.
    """

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float
    line_key: tuple[int, int, int]


@dataclass(frozen=True)
class PageText:
    """Прочитанная полоса.

    Attributes:
        lines: Строки текста по порядку чтения tesseract (блок → абзац → строка).
        page_number: Номер страницы из колонтитула или ``None``.
        header: Текст колонтитула без номера (пусто, если не найден).
        header_box: (верх, высота) опоры колонтитула в px полосы или ``None``.
        confidence: Средняя уверенность слов, 0..100.
    """

    lines: list[str]
    page_number: int | None
    header: str
    header_box: tuple[int, int] | None
    confidence: float


def run_tesseract(gray: np.ndarray, psm: int = 3) -> list[Word]:
    """Прогнать tesseract по полутоновой полосе и вернуть слова.

    Args:
        gray: Полутоновая полоса uint8.
        psm: Режим разбиения страницы tesseract (3 — автоматический разбор колонок).

    Returns:
        Слова с уверенностью не ниже ``MIN_WORD_CONFIDENCE``; пустой список, если tesseract упал.
    """
    with tempfile.TemporaryDirectory(prefix="spread_ocr_") as work:
        image = Path(work) / "page.png"
        cv2.imwrite(str(image), gray)
        # OMP_THREAD_LIMIT=1: tesseract не должен расходиться по всем ядрам поверх пула процессов.
        env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
        try:
            done = subprocess.run(
                ["tesseract", str(image), "-", "--psm", str(psm), "-l", LANGUAGES, "tsv"],
                capture_output=True,
                text=True,
                env=env,
                timeout=TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError):
            return []
    words = []
    for row in done.stdout.splitlines()[1:]:
        columns = row.split("\t")
        if len(columns) < 12 or not columns[11].strip():
            continue
        try:
            confidence = float(columns[10])
        except ValueError:
            continue
        if confidence < MIN_WORD_CONFIDENCE or not _ALNUM.search(columns[11]):
            continue
        words.append(
            Word(
                text=columns[11].strip(),
                left=int(columns[6]),
                top=int(columns[7]),
                width=int(columns[8]),
                height=int(columns[9]),
                confidence=confidence,
                line_key=(int(columns[2]), int(columns[3]), int(columns[4])),
            )
        )
    return words


def group_lines(words: list[Word]) -> list[list[Word]]:
    """Сгруппировать слова в строки по ключу TSV, слова внутри строки — слева направо.

    Args:
        words: Слова tesseract.

    Returns:
        Список строк (каждая — список слов) в порядке ключей (блок, абзац, строка).
    """
    rows: dict[tuple[int, int, int], list[Word]] = {}
    for word in words:
        rows.setdefault(word.line_key, []).append(word)
    return [sorted(row, key=lambda w: w.left) for _, row in sorted(rows.items())]


def find_header(words: list[Word], page_height: int) -> tuple[int | None, str, tuple[int, int] | None]:
    """Найти номер страницы и текст колонтитула.

    У журнала номер стоит на правом конце линейки колонтитула у обеих полос (у левой — у сгиба, у правой —
    у наружного края). Опора колонтитула — самое верхнее «настоящее» слово верхней зоны; номер — самый правый
    числовой токен строки опоры.

    Args:
        words: Слова полосы.
        page_height: Высота полосы, px.

    Returns:
        Кортеж (номер или ``None``, текст колонтитула без номера, (верх, высота) опоры в px полосы или ``None``).
    """
    zone = [w for w in words if w.top < HEADER_ZONE * page_height]
    real = [w for w in zone if _REAL_WORD.search(w.text)]
    numbers = [w for w in zone if _NUMBER.match(w.text)]
    if real:
        anchor = min(real, key=lambda w: w.top)
    elif numbers:
        anchor = min(numbers, key=lambda w: w.top)
    else:
        return None, "", None
    # Строка колонтитула — слова зоны, чей центр по высоте близок к центру опоры.
    anchor_center = anchor.top + anchor.height / 2
    tolerance = max(anchor.height * 1.5, 25)
    line = [w for w in zone if abs(w.top + w.height / 2 - anchor_center) <= tolerance]
    line_numbers = [w for w in line if _NUMBER.match(w.text)]
    page_number = None
    if line_numbers:
        # Самый правый числовой токен строки — номер страницы.
        page_number = int(_NUMBER.match(max(line_numbers, key=lambda w: w.left).text).group(1))
    header = " ".join(w.text for w in sorted(line, key=lambda w: w.left) if not _NUMBER.match(w.text))
    return page_number, header, (anchor.top, anchor.height)


def read_number(band: np.ndarray) -> int | None:
    """Дочитать номер страницы в полосе колонтитула отдельным проходом (разреженный текст, psm 11).

    Args:
        band: Полутоновая горизонтальная полоса вокруг строки колонтитула (с запасом за сгиб у левой полосы).

    Returns:
        Самый правый числовой токен из 1–3 цифр с уверенностью ≥ 50 или ``None``.
    """
    tokens = [w for w in run_tesseract(band, psm=11) if _NUMBER.match(w.text) and w.confidence >= 50]
    if not tokens:
        return None
    return int(_NUMBER.match(max(tokens, key=lambda w: w.left).text).group(1))


def read_page(gray: np.ndarray) -> PageText:
    """Прочитать полосу целиком.

    Args:
        gray: Полутоновая выровненная полоса uint8.

    Returns:
        ``PageText`` с текстом, номером, колонтитулом и средней уверенностью.
    """
    words = run_tesseract(gray)
    # Строка остаётся, если в ней есть настоящее слово или число (номер, цифры таблиц).
    lines = [
        " ".join(w.text for w in line)
        for line in group_lines(words)
        if any(_REAL_WORD.search(w.text) or w.text.isdigit() for w in line)
    ]
    page_number, header, header_box = find_header(words, gray.shape[0])
    confidence = float(np.mean([w.confidence for w in words])) if words else 0.0
    return PageText(
        lines=[line for line in lines if line.strip()],
        page_number=page_number,
        header=header,
        header_box=header_box,
        confidence=confidence,
    )
