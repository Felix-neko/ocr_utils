"""Признаки одной полосы: метка surya из кэша разметки и текстовые признаки tesseract.

Текстовые признаки считаются по СЛОВАМ tesseract (TSV, ``--psm 6``), а не по строкам
вывода: нужны координаты. ``--psm 6`` берёт полосу одним блоком, и строки соседних
колонок склеиваются в одну — это на руку: у оглавления в правой колонке (полоса 3 с 1970/04)
последним словом склеенной строки всё равно остаётся номер страницы, а у обычной
двухколонной полосы — обычное слово.

Что считается:

* ``kw_contents`` — на полосе есть слово «СОДЕРЖАНИЕ» (с поправкой на разрядку и на
  одну-две ошибки распознавания);
* ``kw_index`` — «УКАЗАТЕЛЬ» или «ПЕРЕЧЕНЬ» (первая полоса указателя за год);
* ``num_tail_lines`` / ``num_tail_ratio`` — сколько строк кончаются числом из 1-3 цифр, стоящим
  у ОБЩЕГО правого края (номера страниц выровнены в колонку), и их доля среди строк с текстом;
* ``leader_words`` — слова из одних точек: tesseract так читает отточия.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ocr_utils.scan_cropping.image_io import read_dpi
from ocr_utils.scan_markup import tesseract
from ocr_utils.scan_markup.detection import layout_cache
from ocr_utils.scan_markup.table_detection.layout import PageLayout
from ocr_utils.scan_markup.tesseract import Word
from ocr_utils.scan_markup.toc import WORK_DPI

logger = logging.getLogger(__name__)

# Метка surya для блока оглавления и метка таблицы (таблица с числами справа — главный
# источник ложных кандидатов, её площадь пишется в признаки ради разбора).
SURYA_TOC_LABEL = "TableOfContents"
SURYA_TABLE_LABEL = "Table"

# Ниже этой уверенности блок surya не считается (то же значение, что у детектора таблиц).
SURYA_MIN_CONFIDENCE = 0.3

# Ключевые слова. Заголовок «СОДЕРЖАНИЕ» набран КАПИТЕЛЬЮ (в том числе разрядкой) и стоит
# своей короткой строкой; tesseract на 150 dpi путает одну букву — сравнение по расстоянию
# Левенштейна с допуском 1. Регистр обязателен: «содержание запасов» в обычном тексте —
# частое слово, и без проверки регистра оно давало ложное оглавление на каждой третьей полосе.
CONTENTS_WORD = "СОДЕРЖАНИЕ"
CONTENTS_MAX_DISTANCE = 1
# Строка-заголовок: не больше стольких слов с буквами (в 1966 рядом стоит номер выпуска).
HEADING_MAX_WORDS = 3
# Указатель: «Указатель статей, опубликованных в журнале в 1975 г.», «Перечень материалов,
# опубликованных в журнале ... в 1970 г.». Хватает начала слова, но ТОЧНОГО и с большой
# буквы: с допуском в одну букву «переменного» из таблицы норм отгрузки становилось
# «перечнем», а строчная буква — это текст, а не заголовок.
INDEX_PREFIXES = ("УКАЗАТЕЛ", "ПЕРЕЧЕН")
# Рядом с заголовком указателя (в той же или следующей строке) стоит одно из этих слов:
# «Указатель СТАТЕЙ, ОПУБЛИКОВАННЫХ в журнале», «Перечень МАТЕРИАЛОВ, ОПУБЛИКОВАННЫХ...».
# Без них «Указатель» — просто слово с большой буквы в начале предложения.
INDEX_CONTEXT = ("ОПУБЛИКОВАН", "СТАТЕЙ", "МАТЕРИАЛОВ")
# Выходные данные: с 1970/04 оглавление кончается на полосе с редколлегией и адресом
# редакции, и хвост там бывает в одну-две строки, которые tesseract не всегда читает как
# строки с номером. Слово «РЕДКОЛЛЕГИЯ» на этой полосе стоит всегда.
IMPRINT_WORD = "РЕДКОЛЛЕГИЯ"
IMPRINT_MAX_DISTANCE = 1

# Слово tesseract, которое считается номером страницы в конце строки: 1-3 цифры, иногда с
# прилипшей точкой отточия («...12»). Диапазон «12—14» тоже бывает в указателях.
PAGE_NUMBER = re.compile(r"^[.·]*(\d{1,3})([—–-]\d{1,3})?[.·]*$")
# Слово из одних точек — отточия.
LEADER = re.compile(r"^[.·…]{3,}$")
# Слово со «значимым» текстом: хотя бы две буквы или цифры.
TEXTUAL = re.compile(r"[\wа-яёА-ЯЁ]{2,}")

# Номера страниц стоят в колонку: правый край числа не дальше этой доли ширины полосы от
# медианы правых краёв. 3 % на 150 dpi у полосы 170 мм — около 25 px, полторы цифры.
RIGHT_EDGE_TOLERANCE = 0.03

# Слова с уверенностью ниже не участвуют в ключевых словах (мусор с фотографий), но в
# геометрии строк участвуют все: у отточий уверенность низкая всегда.
MIN_KEYWORD_CONFIDENCE = 40.0


@dataclass(frozen=True)
class PageFeatures:
    """Признаки полосы. Поля surya — нули, если кэша разметки нет; текстовые — нули при
    ошибке tesseract (``error`` объясняет)."""

    rel_path: str
    order_index: int
    idx_from_end: int
    surya_toc_conf: float = 0.0
    surya_toc_area: float = 0.0
    surya_table_area: float = 0.0
    kw_contents: bool = False
    kw_index: bool = False
    kw_imprint: bool = False
    text_lines: int = 0
    num_tail_lines: int = 0
    num_tail_ratio: float = 0.0
    leader_words: int = 0
    error: str = ""
    # Уменьшенная копия для контактного листа (JPEG), только по запросу.
    thumbnail: bytes | None = field(default=None, repr=False, compare=False)

    def metrics(self) -> dict[str, float]:
        """Числовые признаки для CSV и таблиц разделения."""
        return {
            "surya_toc_conf": self.surya_toc_conf,
            "surya_toc_area": self.surya_toc_area,
            "surya_table_area": self.surya_table_area,
            "kw_contents": float(self.kw_contents),
            "kw_index": float(self.kw_index),
            "kw_imprint": float(self.kw_imprint),
            "text_lines": float(self.text_lines),
            "num_tail_lines": float(self.num_tail_lines),
            "num_tail_ratio": self.num_tail_ratio,
            "leader_words": float(self.leader_words),
        }


METRIC_NAMES = tuple(PageFeatures("", 0, 0).metrics())


# --- surya --------------------------------------------------------------------------


def surya_features(layout: PageLayout | None) -> dict[str, float]:
    """Уверенность и площадь блоков оглавления, площадь таблиц — в долях полосы."""
    if layout is None or layout.width <= 0 or layout.height <= 0:
        return {"surya_toc_conf": 0.0, "surya_toc_area": 0.0, "surya_table_area": 0.0}
    page_area = float(layout.width * layout.height)
    toc_conf = 0.0
    toc_area = 0.0
    table_area = 0.0
    for block in layout.blocks:
        if block.confidence < SURYA_MIN_CONFIDENCE:
            continue
        area = block.box.area / page_area
        if block.label == SURYA_TOC_LABEL:
            toc_conf = max(toc_conf, block.confidence)
            toc_area += area
        elif block.label == SURYA_TABLE_LABEL:
            table_area += area
    return {
        "surya_toc_conf": round(toc_conf, 3),
        "surya_toc_area": round(min(1.0, toc_area), 3),
        "surya_table_area": round(min(1.0, table_area), 3),
    }


# --- текст ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str, limit: int) -> int:
    """Расстояние Левенштейна; всё, что больше ``limit``, возвращается как ``limit + 1``."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        best = i
        for j, cb in enumerate(b, 1):
            cost = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            current.append(cost)
            best = min(best, cost)
        if best > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _normalise(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalpha()).replace("Ё", "Е")


@dataclass(frozen=True)
class Token:
    """Слово строки без пунктуации и цифр: в верхнем регистре, плюс как оно было набрано."""

    text: str
    capitalised: bool  # с большой буквы
    upper: bool  # целиком капителью (или разрядкой)


def keyword_tokens(line_words: Sequence[Word]) -> list[Token]:
    """Слова строки по буквам, с разрядкой, склеенной обратно.

    «С О Д Е Р Ж А Н И Е» tesseract отдаёт десятью словами по одной букве; подряд идущие
    однобуквенные слова склеиваются в одно (и считаются капителью). Пунктуация и цифры
    выбрасываются.
    """
    tokens: list[Token] = []
    spaced = ""
    for word in line_words:
        raw = "".join(ch for ch in word.text if ch.isalpha())
        letters = _normalise(raw)
        if not letters:
            continue
        if len(letters) == 1:
            spaced += letters
            continue
        if spaced:
            tokens.append(Token(spaced, True, True))
            spaced = ""
        tokens.append(Token(letters, raw[0].isupper(), raw.isupper()))
    if spaced:
        tokens.append(Token(spaced, True, True))
    return tokens


def is_contents_word(token: str, upper: bool = True) -> bool:
    return upper and _levenshtein(token, CONTENTS_WORD, CONTENTS_MAX_DISTANCE) <= CONTENTS_MAX_DISTANCE


def is_index_word(token: str, capitalised: bool = True) -> bool:
    return capitalised and any(token.startswith(prefix) for prefix in INDEX_PREFIXES)


def is_imprint_word(token: str, capitalised: bool = True) -> bool:
    return capitalised and _levenshtein(token, IMPRINT_WORD, IMPRINT_MAX_DISTANCE) <= IMPRINT_MAX_DISTANCE


def has_index_context(tokens: Sequence[Token]) -> bool:
    return any(t.text.startswith(INDEX_CONTEXT[0]) or t.text in INDEX_CONTEXT[1:] for t in tokens)


def _lines(all_words: Sequence[Word]) -> list[list[Word]]:
    grouped: dict[tuple[int, int, int], list[Word]] = {}
    for word in all_words:
        grouped.setdefault(word.line_key, []).append(word)
    return [sorted(line, key=lambda w: w.left) for _, line in sorted(grouped.items())]


def text_features(all_words: Sequence[Word], width: int) -> dict[str, float | int | bool]:
    """Текстовые признаки по словам tesseract; ``width`` — ширина поданной картинки."""
    lines = _lines(all_words)
    kw_contents = False
    kw_index = False
    kw_imprint = False
    leader_words = 0
    text_lines = 0
    tails: list[tuple[int, int]] = []  # (правый край числа, индекс строки)

    line_tokens = [keyword_tokens([w for w in line if w.confidence >= MIN_KEYWORD_CONFIDENCE]) for line in lines]
    for index, line in enumerate(lines):
        last = max(line, key=lambda w: w.right)
        has_tail = bool(PAGE_NUMBER.match(last.text)) and any(
            TEXTUAL.search(w.text) and not w.text.isdigit() for w in line
        )
        tokens = line_tokens[index]
        # «СОДЕРЖАНИЕ» — короткая строка капителью; в 1966 году рядом номер выпуска.
        if len(tokens) <= HEADING_MAX_WORDS and any(is_contents_word(t.text, t.upper) for t in tokens):
            kw_contents = True
        # Заголовок указателя стоит своей строкой без номера страницы, и рядом (в той же или
        # следующей строке) — «статей», «материалов», «опубликованных». Строка оглавления
        # «Указатель статей, опубликованных в журнале ... 90» — это ссылка НА указатель.
        if not has_tail and any(is_index_word(t.text, t.capitalised) for t in tokens):
            nearby = tokens + (line_tokens[index + 1] if index + 1 < len(lines) else [])
            if has_index_context(nearby):
                kw_index = True
        if any(is_imprint_word(t.text, t.capitalised) for t in tokens):
            kw_imprint = True
        leader_words += sum(1 for w in line if LEADER.match(w.text))
        textual = [w for w in line if TEXTUAL.search(w.text)]
        if len(textual) < 2:
            continue
        text_lines += 1
        if has_tail:
            tails.append((last.right, index))

    num_tail_lines = 0
    if tails and width > 0:
        edges = np.array([edge for edge, _ in tails], dtype=np.float64)
        median = float(np.median(edges))
        num_tail_lines = int((np.abs(edges - median) <= RIGHT_EDGE_TOLERANCE * width).sum())
    ratio = num_tail_lines / text_lines if text_lines else 0.0
    return {
        "kw_contents": kw_contents,
        "kw_index": kw_index,
        "kw_imprint": kw_imprint,
        "text_lines": text_lines,
        "num_tail_lines": num_tail_lines,
        "num_tail_ratio": round(ratio, 3),
        "leader_words": leader_words,
    }


# --- полоса целиком -----------------------------------------------------------------


def read_gray(path: Path, default_dpi: int = 600, work_dpi: int = WORK_DPI) -> np.ndarray:
    """Серая копия полосы в ``work_dpi``. Оригинал на 600 dpi уменьшается вчетверо."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("не читается как изображение")
    dpi = read_dpi(path, default=None)
    dpi = int(dpi) if dpi and dpi >= 72 else default_dpi
    scale = work_dpi / dpi
    if scale < 0.999:
        height, width = gray.shape[:2]
        gray = cv2.resize(
            gray, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
        )
    return gray


def thumbnail_jpeg(gray: np.ndarray, width: int = 420) -> bytes:
    """JPEG-миниатюра для контактного листа."""
    scale = width / max(1, gray.shape[1])
    small = cv2.resize(gray, (width, max(1, round(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return encoded.tobytes() if ok else b""


def page_features(
    path: Path,
    rel_path: str,
    order_index: int,
    idx_from_end: int,
    layout_cache_dir: Path | None,
    want_thumbnail: bool = False,
    default_dpi: int = 600,
) -> PageFeatures:
    """Все признаки полосы. Исключения не выпускает — кладёт их в ``error``."""
    features = PageFeatures(rel_path, order_index, idx_from_end)
    try:
        cached = layout_cache.load(layout_cache_dir, rel_path) if layout_cache_dir is not None else None
        features = replace(features, **surya_features(cached.layout if cached is not None else None))
        gray = read_gray(path, default_dpi)
        if want_thumbnail:
            features = replace(features, thumbnail=thumbnail_jpeg(gray))
        words = tesseract.words(gray, psm=6)
        features = replace(features, **text_features(words, gray.shape[1]))
        return features
    except Exception as exc:  # noqa: BLE001 — одна битая полоса не должна валить прогон
        return replace(features, error=str(exc))


__all__ = [
    "METRIC_NAMES",
    "PageFeatures",
    "Token",
    "is_contents_word",
    "is_index_word",
    "keyword_tokens",
    "page_features",
    "read_gray",
    "surya_features",
    "text_features",
]
