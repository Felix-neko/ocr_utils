"""Отбор страниц с концами статей по строкам tesseract: пустое место под текстом колонки и строки-подписи (инициалы с фамилией прописными, «г. Город», регалии); текст низа перед концом — на чтение глазами."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from research.flush_split.ocr import OcrLine

# Строка — текст, если в ней не меньше стольких букв (номер страницы, сор и обрывки не в счёт).
MIN_LETTERS = 4
# Колонки: строка «узкая», если уже такой доли ширины текста страницы; при доле узких строк от NARROW_SHARE страница
# в две колонки, граница — середина охвата текста.
NARROW_WIDTH_SHARE = 0.6
NARROW_SHARE = 0.5
# Конец статьи: просвет между соседними строками колонки не меньше стольких мм (шаг строк ХиП 1991 ~3.8 мм:
# отбивка абзацев ~5 мм, конец статьи и заголовок следующей — от 8 мм)...
END_GAP_MM = 8.0
# ...или текст колонки кончается выше такой доли высоты текста страницы.
END_BOTTOM_SHARE = 0.85
# Сколько строк перед концом печатать на чтение.
TAIL_LINES = 10

# Строки-подписи: инициалы и фамилия прописными, «г. Город», регалии (корни слов, без учёта регистра).
SURNAME_RE = re.compile(r"^[А-ЯЁ]\.\s?([А-ЯЁ]\.\s?)?[А-ЯЁ]{3,}")
CITY_RE = re.compile(r"^г\.\s?[А-ЯЁ][а-яё]+")
TITLE_RE = re.compile(
    r"юрисконсульт|доцент|профессор|кандидат\s\w+\sнаук|доктор\s\w+\sнаук|председател|заместител|корреспондент"
    r"|заведующ|начальник|директор|экономист|юрист\b|сотрудник|аспирант|научн\w* сотрудн|академи|инженер",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PageEnd:
    """Место, где колонка страницы кончается (конец статьи) или стоит строка-подпись.

    Поля:
        column: Колонка (0 — левая или единственная, 1 — правая).
        y_mm: Низ последней строки перед концом (или верх строки-подписи), мм.
        reason: Почему отобрано: ``gap`` — просвет, ``bottom`` — колонка кончилась высоко, ``sign`` — строка-подпись.
        tail: Строки перед концом сверху вниз (до :data:`TAIL_LINES`).
        x_span: Охват колонки по x, мм (для отступов строк при печати).
    """

    column: int
    y_mm: float
    reason: str
    tail: tuple[OcrLine, ...]
    x_span: tuple[float, float]


def letters(text: str) -> int:
    """Число букв в строке."""
    return sum(ch.isalpha() for ch in text)


def columns_of(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """Строки по колонкам сверху вниз: одна колонка или две (по доле узких строк), граница — середина охвата.

    Args:
        lines: Строки-тексты страницы.

    Returns:
        Колонки слева направо, строки в каждой отсортированы по верху.
    """
    if not lines:
        return []
    x0, x1 = min(ln.x0 for ln in lines), max(ln.x1 for ln in lines)
    narrow = np.mean([ln.x1 - ln.x0 < NARROW_WIDTH_SHARE * (x1 - x0) for ln in lines])
    if narrow < NARROW_SHARE:
        return [sorted(lines, key=lambda ln: ln.y0)]
    mid = (x0 + x1) / 2
    left = [ln for ln in lines if (ln.x0 + ln.x1) / 2 < mid]
    right = [ln for ln in lines if (ln.x0 + ln.x1) / 2 >= mid]
    return [sorted(col, key=lambda ln: ln.y0) for col in (left, right) if col]


def page_ends(lines: list[OcrLine]) -> list[PageEnd]:
    """Концы статей и строки-подписи страницы.

    Args:
        lines: Строки tesseract страницы.

    Returns:
        Места в порядке колонок и сверху вниз; у одного места — одна запись (подпись внутри хвоста конца не
        дублируется).
    """
    text = [ln for ln in lines if letters(ln.text) >= MIN_LETTERS]
    if not text or is_toc(text):
        return []
    top, bottom = min(ln.y0 for ln in text), max(ln.y1 for ln in text)
    out: list[PageEnd] = []
    for ci, col in enumerate(columns_of(text)):
        span = (min(ln.x0 for ln in col), max(ln.x1 for ln in col))
        marks: dict[int, str] = {}
        for i in range(1, len(col)):
            if col[i].y0 - col[i - 1].y1 >= END_GAP_MM:
                marks[i - 1] = "gap"
        if col[-1].y1 < top + END_BOTTOM_SHARE * (bottom - top):
            marks.setdefault(len(col) - 1, "bottom")
        for i, ln in enumerate(col):
            if SURNAME_RE.match(ln.text) or CITY_RE.match(ln.text) or TITLE_RE.search(ln.text):
                # Подпись — конец хвоста на три строки ниже (регалии под фамилией видны целиком).
                marks.setdefault(min(i + 3, len(col) - 1), "sign")
        # Места ближе TAIL_LINES строк друг к другу — одно место с нижним концом (хвосты не повторяются).
        kept: list[int] = []
        for i in sorted(marks):
            if kept and i - kept[-1] < TAIL_LINES // 2:
                reason = marks[kept[-1]] if marks[kept[-1]] != "sign" else marks[i]
                marks[i] = reason
                kept[-1] = i
            else:
                kept.append(i)
        for i in kept:
            tail = tuple(col[max(0, i - TAIL_LINES + 1) : i + 1])
            out.append(PageEnd(ci, col[i].y1, marks[i], tail, span))
    return out


# Оглавление: строк, кончающихся номером страницы, не меньше такой доли — страница не идёт в отбор (фамилии авторов
# статей в оглавлении прописными, а подписей под статьями там нет).
TOC_SHARE = 0.25
TOC_RE = re.compile(r"\d{1,3}\W*$")


def is_toc(lines: list[OcrLine]) -> bool:
    """Страница оглавления: доля строк с номером страницы в конце не меньше :data:`TOC_SHARE`."""
    return float(np.mean([bool(TOC_RE.search(ln.text)) for ln in lines])) >= TOC_SHARE


def render_end(key: str, end: PageEnd) -> str:
    """Хвост конца статьи текстом: по строке на строку tesseract с отступами слева и справа в мм.

    Args:
        key: Ключ страницы.
        end: Место.

    Returns:
        Markdown-фрагмент.
    """
    x0, x1 = end.x_span
    head = f"### {key} кол {end.column} y≈{end.y_mm:.0f} мм ({end.reason})"
    rows = [f"`{ln.y0:5.1f} |{ln.x0 - x0:5.1f} {x1 - ln.x1:5.1f}|` {ln.text}" for ln in end.tail]
    return "\n".join([head, *rows, ""])
