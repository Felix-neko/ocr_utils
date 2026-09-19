"""Текстовый слой страницы FineReader: разбор потока содержимого до слов и глифов.

КАК УСТРОЕН СЛОЙ (замер по паку-1). Каждое слово — свой маркированный спан
``/Span <</MCID n>> BDC … EMC``, внутри него один текстовый объект ``BT … ET``: ``Tf``,
``3 Tr`` (невидимый текст), ``a 0 0 d x y Tm`` с неравномерным растяжением под ширину
слова (a от 0.5 до 3.8), затем на каждый глиф свой ``(код)Tj`` и ``dx 0 Td``. Три формы
спанов: обычная; «только Td» без ``Tm`` (одноглифные слова и пробелы); повёрнутая ``Tm``
(``0 b c 0 x y Tm``) — FineReader ИНОГДА пишет вертикальный текст правильно, и такие слова
трогать нельзя. Страничный сдвиг ``1 0 0 1 0 dy cm`` лежит ВНУТРИ первого спана перед
``BT``, поэтому удалять спан целиком нельзя — только ``BT … ET`` (см. ``rewrite``).

ПРИВЯЗКА К СИМВОЛАМ. Порядок символов ``rawdict`` PyMuPDF не совпадает с порядком в потоке
(пересортировка по блокам, синтетические пробелы), но начало каждого глифа, посчитанное
по ``Tm``/``Td``/``cm``, совпадает с ``origin`` символа до 1e-4 pt, и сопоставление по
ближайшему началу даёт 100 % на пробных страницах. Отсюда у каждого глифа появляются
символ и bbox, а у слова — текст и рамка. ``get_texttrace()`` в PyMuPDF 1.27 на этих PDF
падает, поэтому не используется.

Разбор — чистая функция над байтами потока; шрифты (ширины глифов, байт на код) берутся
через pikepdf из ресурсов страницы: ширина нужна только там, где в одной строке несколько
глифов (текст, вставленный не FineReader-ом, и синтетика в тестах).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum

import fitz
import pikepdf

logger = logging.getLogger(__name__)

# Допуск при сопоставлении начала глифа с ``origin`` символа rawdict, pt. Замер: расхождение
# 1e-4 pt; допуск в тысячи раз больше, чтобы пережить округления при пересборке потока.
MATCH_TOLERANCE_PT = 0.35

# Ширина глифа, когда шрифт её не сообщает (доля кегля): половина em — типичная ширина буквы.
DEFAULT_GLYPH_WIDTH = 0.5

# Вертикальная рамка глифа без символа rawdict, в долях кегля: выносные вверх и вниз.
ASCENT = 0.8
DESCENT = 0.2

# Матрица текста считается повёрнутой, если её недиагональные элементы заметны на фоне диагональных.
ROTATION_EPS = 1e-3


class SpanShape(StrEnum):
    """Форма текстового объекта в спане (см. докстринг модуля)."""

    PLAIN = "plain"  # ``Tm`` с растяжением, по ``Td`` на глиф — обычное слово FineReader
    TD_ONLY = "td_only"  # без ``Tm``: одноглифные слова и пробелы
    ROTATED = "rotated"  # ``Tm`` с поворотом: FineReader сам написал вертикальный текст
    EMPTY = "empty"  # спан без текстового объекта или без глифов


Matrix = tuple[float, float, float, float, float, float]

IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def multiply(first: Matrix, second: Matrix) -> Matrix:
    """Произведение матриц PDF ``first × second`` (сначала применяется ``first``).

    Args:
        first: Матрица ``(a, b, c, d, e, f)``, применяемая первой.
        second: Матрица, применяемая второй.

    Returns:
        Матрица композиции в том же формате.
    """
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = second
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def apply(matrix: Matrix, x: float, y: float) -> tuple[float, float]:
    """Точка ``(x, y)`` после применения матрицы.

    Args:
        matrix: Матрица ``(a, b, c, d, e, f)``.
        x: Абсцисса.
        y: Ордината.

    Returns:
        Преобразованная точка.
    """
    a, b, c, d, e, f = matrix
    return (a * x + c * y + e, b * x + d * y + f)


@dataclass(frozen=True)
class FontMetrics:
    """Что нужно знать о шрифте, чтобы пройти по строке глифов: байт на код и ширины."""

    name: str
    bytes_per_code: int
    widths: dict[int, float] = field(default_factory=dict)
    default_width: float = DEFAULT_GLYPH_WIDTH
    # Стандартный шрифт без /Widths (Helvetica и прочие base-14): ширины берутся у PyMuPDF по
    # коду как по символу Latin-1. У шрифтов FineReader /Widths есть всегда.
    fallback: "fitz.Font | None" = None

    def width(self, code: int) -> float:
        """Ширина глифа в долях кегля (единицы /1000 уже поделены).

        Args:
            code: Код глифа в кодировке шрифта.

        Returns:
            Ширина в em (1.0 — кегль целиком).
        """
        if code in self.widths:
            return self.widths[code]
        if self.fallback is not None and code < 256:
            try:
                return float(self.fallback.glyph_advance(code))
            except Exception:  # noqa: BLE001 — нет глифа: ширина по умолчанию
                pass
        return self.default_width

    def codes(self, data: bytes) -> list[bytes]:
        """Строка ``Tj`` → список кодов глифов по ``bytes_per_code`` байт.

        Args:
            data: Байты строкового операнда после снятия экранирования.

        Returns:
            Список байтовых кодов; неполный последний код (нечётная длина) отбрасывается.
        """
        step = self.bytes_per_code
        return [data[i : i + step] for i in range(0, len(data) - step + 1, step)]


@dataclass
class Glyph:
    """Один показанный глиф: где он начинается и какой символ ему нашёлся в rawdict."""

    index: int
    code: bytes
    # Начало глифа в ТЕКСТОВОМ пространстве (после Tm, до CTM): для повторной сборки ``Tm``.
    origin_text: tuple[float, float]
    # Начало глифа в координатах fitz (y вниз), после CTM.
    origin: tuple[float, float]
    # Ширина продвижения в pt (по ширине из шрифта и кеглю; для FineReader — оценка).
    advance: float
    char: "str | None" = None
    bbox: "fitz.Rect | None" = None


@dataclass
class Word:
    """Одно слово слоя: спан FineReader (или текстовый объект ``BT … ET`` без спана)."""

    mcid: "int | None"
    shape: SpanShape
    font: str
    size: float
    # Матрица текста ``Tm`` (a, b, c, d) первого позиционирования; для TD_ONLY — единичная.
    matrix: tuple[float, float, float, float]
    # CTM на момент ``BT``: вместе с ``matrix`` даёт направление слова на странице.
    ctm: Matrix
    glyphs: list[Glyph] = field(default_factory=list)
    # Диапазоны байт ``BT … ET`` (от начала ``BT`` до конца ``ET``) внутри спана.
    text_ranges: list[tuple[int, int]] = field(default_factory=list)
    # Диапазон байт спана ``BDC … EMC``; None у текста вне спанов.
    span_range: "tuple[int, int] | None" = None
    # Операторы состояния текста между ``BT`` и первым позиционированием (``Tf``, ``Tr`` …):
    # при пересборке усечённого слова повторяются как есть.
    state_ops: bytes = b""
    render_mode: int = 0
    struct_line: "str | None" = None

    @property
    def text(self) -> str:
        """Текст слова по привязанным символам (глифы без символа пропускаются)."""
        return "".join(g.char for g in self.glyphs if g.char)

    @property
    def bbox(self) -> fitz.Rect:
        """Рамка слова в координатах fitz — объединение рамок глифов."""
        rect = fitz.Rect()
        for glyph in self.glyphs:
            if glyph.bbox is not None:
                rect |= glyph.bbox
        return rect

    @property
    def unmatched(self) -> int:
        """Сколько глифов остались без символа rawdict."""
        return sum(1 for g in self.glyphs if g.char is None)

    @property
    def direction(self) -> tuple[float, float]:
        """Единичный вектор направления строки на странице (в координатах fitz, y вниз)."""
        if len(self.glyphs) >= 2:
            (x0, y0), (x1, y1) = self.glyphs[0].origin, self.glyphs[-1].origin
            norm = math.hypot(x1 - x0, y1 - y0)
            if norm > 1e-6:
                return ((x1 - x0) / norm, (y1 - y0) / norm)
        a, b, c, d = self.matrix
        composed = multiply((a, b, c, d, 0.0, 0.0), self.ctm)
        dx, dy = composed[0], composed[1]
        norm = math.hypot(dx, dy) or 1.0
        return (dx / norm, -dy / norm)

    @property
    def stretch(self) -> float:
        """Горизонтальное растяжение ``a`` матрицы текста (у FineReader — подгонка под ширину слова)."""
        return abs(self.matrix[0]) if abs(self.matrix[0]) > ROTATION_EPS else abs(self.matrix[1])


@dataclass
class TextLayer:
    """Текстовый слой одной страницы."""

    page_index: int
    page_rect: fitz.Rect
    words: list[Word]
    # Число глифов без символа rawdict по всей странице.
    unmatched_glyphs: int = 0
    # Число символов rawdict, к которым не пришёл ни один глиф (синтетические пробелы и т.п.).
    orphan_chars: int = 0
    # Глифы, чьё начало лежит вне страницы: FineReader кладёт текст обрезанной части образа за
    # CropBox (замер: 1966/04 с. 28 — 1225 глифов над верхним краем); rawdict их не отдаёт.
    offpage_glyphs: int = 0

    def word_by_mcid(self, mcid: int) -> "Word | None":
        """Слово по номеру спана.

        Args:
            mcid: Номер маркированного содержимого.

        Returns:
            Слово или None, если такого спана нет.
        """
        for word in self.words:
            if word.mcid == mcid:
                return word
        return None


# --- Токенизатор потока содержимого -----------------------------------------------------

TOKEN_RE = re.compile(
    rb"""
    (?P<ws>[\s\x00]+) |
    (?P<comment>%[^\r\n]*) |
    (?P<number>[+-]?(?:\d+\.?\d*|\.\d+)) |
    (?P<name>/[^\s/\[\]<>(){}%]*) |
    (?P<dict_open><<) | (?P<dict_close>>>) |
    (?P<hex><[0-9A-Fa-f\s]*>) |
    (?P<string_open>\() |
    (?P<array_open>\[) | (?P<array_close>\]) |
    (?P<brace>[{}]) |
    (?P<op>[A-Za-z'"*][A-Za-z0-9'"*]*)
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    """Лексема потока: вид, значение и диапазон байт в исходном потоке."""

    kind: str
    value: object
    start: int
    end: int


STRING_ESCAPES = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f", b"(": b"(", b")": b")", b"\\": b"\\"}


def _read_string(raw: bytes, start: int) -> tuple[bytes, int]:
    """Литеральная строка ``(...)`` с экранированием и вложенными скобками.

    Args:
        raw: Весь поток.
        start: Позиция открывающей скобки.

    Returns:
        Байты строки после снятия экранирования и позиция за закрывающей скобкой.
    """
    depth = 0
    out = bytearray()
    i = start
    while i < len(raw):
        ch = raw[i : i + 1]
        if ch == b"\\":
            nxt = raw[i + 1 : i + 2]
            if nxt in STRING_ESCAPES:
                out += STRING_ESCAPES[nxt]
                i += 2
                continue
            if nxt.isdigit():
                digits = raw[i + 1 : i + 4]
                length = 0
                while length < 3 and length < len(digits) and digits[length : length + 1].isdigit():
                    length += 1
                out.append(int(digits[:length], 8) & 0xFF)
                i += 1 + length
                continue
            if nxt in (b"\n", b"\r"):
                i += 2
                if nxt == b"\r" and raw[i : i + 1] == b"\n":
                    i += 1
                continue
            out += nxt
            i += 2
            continue
        if ch == b"(":
            depth += 1
            if depth > 1:
                out += ch
            i += 1
            continue
        if ch == b")":
            depth -= 1
            if depth == 0:
                return bytes(out), i + 1
            out += ch
            i += 1
            continue
        out += ch
        i += 1
    return bytes(out), i


def tokenize(raw: bytes) -> list[Token]:
    """Лексемы потока содержимого в порядке следования.

    Args:
        raw: Распакованный поток содержимого страницы.

    Returns:
        Список лексем; пробелы и комментарии пропущены. Строки отдаются со снятым
        экранированием (``kind == "string"``), шестнадцатеричные — тоже как ``string``.
    """
    tokens: list[Token] = []
    pos = 0
    length = len(raw)
    while pos < length:
        match = TOKEN_RE.match(raw, pos)
        if match is None:
            pos += 1  # незнакомый байт: пропускаем, чтобы не зациклиться
            continue
        kind = match.lastgroup
        if kind in ("ws", "comment"):
            pos = match.end()
            continue
        if kind == "string_open":
            value, end = _read_string(raw, pos)
            tokens.append(Token("string", value, pos, end))
            pos = end
            continue
        text = match.group()
        if kind == "number":
            tokens.append(Token("number", float(text), pos, match.end()))
        elif kind == "hex":
            digits = re.sub(rb"\s", b"", text[1:-1])
            if len(digits) % 2:
                digits += b"0"
            tokens.append(Token("string", bytes.fromhex(digits.decode("ascii")), pos, match.end()))
        elif kind == "name":
            tokens.append(Token("name", text[1:].decode("latin-1"), pos, match.end()))
        elif kind == "op":
            tokens.append(Token("op", text.decode("latin-1"), pos, match.end()))
        else:
            tokens.append(Token(kind, text.decode("latin-1"), pos, match.end()))
        pos = match.end()
    return tokens


def _collect_operand(tokens: list[Token], i: int) -> tuple[object, int]:
    """Один операнд начиная с лексемы ``i``: словарь и массив сворачиваются в структуру.

    Args:
        tokens: Все лексемы.
        i: Индекс первой лексемы операнда.

    Returns:
        Значение операнда и индекс следующей лексемы.
    """
    token = tokens[i]
    if token.kind == "dict_open":
        result: dict = {}
        i += 1
        while i < len(tokens) and tokens[i].kind != "dict_close":
            key = tokens[i].value if tokens[i].kind == "name" else None
            value, i = _collect_operand(tokens, i + 1)
            if key is not None:
                result[key] = value
        return result, i + 1
    if token.kind == "array_open":
        items: list = []
        i += 1
        while i < len(tokens) and tokens[i].kind != "array_close":
            value, i = _collect_operand(tokens, i)
            items.append(value)
        return items, i + 1
    return token.value, i + 1


# --- Интерпретатор текстовых операторов ------------------------------------------------


@dataclass
class _TextState:
    """Состояние текстового объекта между ``BT`` и ``ET``."""

    tm: Matrix = IDENTITY
    tlm: Matrix = IDENTITY
    font: str = ""
    size: float = 0.0
    char_spacing: float = 0.0
    word_spacing: float = 0.0
    horizontal_scale: float = 1.0
    leading: float = 0.0
    rise: float = 0.0
    render_mode: int = 0
    had_tm: bool = False
    first_matrix: "tuple[float, float, float, float] | None" = None


def page_matrix(page: fitz.Page) -> Matrix:
    """Матрица «координаты PDF → координаты fitz» страницы (учитывает MediaBox/CropBox и поворот).

    Args:
        page: Страница PyMuPDF.

    Returns:
        Матрица ``(a, b, c, d, e, f)``. У страниц FineReader с MediaBox не от нуля (замер:
        1966/04 с. 28, сдвиг по y на 300 pt) простой переворот по высоте страницы врёт.
    """
    m = page.transformation_matrix
    return (m.a, m.b, m.c, m.d, m.e, m.f)


def parse_content(
    raw: bytes, page_height: "float | Matrix", fonts: "dict[str, FontMetrics] | None" = None
) -> tuple[list[Word], int]:
    """Разобрать поток содержимого страницы до слов и глифов.

    Args:
        raw: Распакованный поток содержимого (все потоки страницы, склеенные через перевод строки).
        page_height: Матрица «PDF → fitz» (:func:`page_matrix`) либо, для простых страниц с
            MediaBox от нуля, высота страницы в pt (тогда матрица — переворот по высоте).
        fonts: Метрики шрифтов по имени ресурса (``F0`` → ...); без них ширина каждого глифа
            берётся по умолчанию, что для FineReader безразлично (по ``Td`` на глиф).

    Returns:
        Список слов в порядке потока и число текстовых объектов ``BT`` без спана (для статистики).
    """
    fonts = fonts or {}
    to_fitz: Matrix = page_height if isinstance(page_height, tuple) else (1.0, 0.0, 0.0, -1.0, 0.0, float(page_height))
    tokens = tokenize(raw)
    words: list[Word] = []
    operands: list[object] = []
    ctm: Matrix = IDENTITY
    ctm_stack: list[Matrix] = []
    # Стек открытых маркированных спанов: (mcid или None, позиция начала, слово или None).
    marks: list[tuple["int | None", int, "Word | None"]] = []
    state = _TextState()
    current: "Word | None" = None
    bt_start = 0
    state_ops_start = 0
    positioned = False
    orphan_objects = 0
    # Начало первого операнда текущего оператора: операторы состояния копируются до него.
    operands_start = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.kind != "op":
            if not operands:
                operands_start = token.start
            value, i = _collect_operand(tokens, i)
            operands.append(value)
            continue
        op = token.value
        i += 1
        op_start = operands_start if operands else token.start
        try:
            if op == "q":
                ctm_stack.append(ctm)
            elif op == "Q":
                if ctm_stack:
                    ctm = ctm_stack.pop()
            elif op == "cm" and len(operands) >= 6:
                ctm = multiply(tuple(float(v) for v in operands[-6:]), ctm)  # type: ignore[arg-type]
            elif op in ("BDC", "BMC"):
                mcid: "int | None" = None
                if op == "BDC" and len(operands) >= 2 and isinstance(operands[-1], dict):
                    raw_mcid = operands[-1].get("MCID")
                    if isinstance(raw_mcid, float):
                        mcid = int(raw_mcid)
                span_word = None
                if mcid is not None:
                    span_word = Word(mcid, SpanShape.EMPTY, "", 0.0, (1.0, 0.0, 0.0, 1.0), ctm)
                    words.append(span_word)
                marks.append((mcid, op_start, span_word))
            elif op == "EMC":
                if marks:
                    mcid, start, span_word = marks.pop()
                    if span_word is not None:
                        span_word.span_range = (start, token.end)
            elif op == "BT":
                # Параметры текста (шрифт, режим, интервалы) живут в графическом состоянии и
                # переживают ET; BT сбрасывает только матрицы. FineReader ставит ``3 Tr`` один
                # раз на страницу, в первом спане.
                state = _TextState(
                    font=state.font,
                    size=state.size,
                    char_spacing=state.char_spacing,
                    word_spacing=state.word_spacing,
                    horizontal_scale=state.horizontal_scale,
                    leading=state.leading,
                    rise=state.rise,
                    render_mode=state.render_mode,
                )
                bt_start = token.start
                state_ops_start = token.end
                positioned = False
                span_word = marks[-1][2] if marks else None
                if span_word is not None and not span_word.text_ranges and not span_word.glyphs:
                    current = span_word
                else:
                    # Текст вне спана (или второй BT в спане): своё слово без MCID.
                    current = Word(None, SpanShape.EMPTY, "", 0.0, (1.0, 0.0, 0.0, 1.0), ctm)
                    words.append(current)
                    orphan_objects += 1
                current.ctm = ctm
            elif op == "ET":
                if current is not None:
                    current.text_ranges.append((bt_start, token.end))
                    current.render_mode = state.render_mode
                    if not current.glyphs:
                        current.shape = SpanShape.EMPTY
                    current = None
            elif op == "Tf" and len(operands) >= 2:
                state.font = str(operands[-2])
                state.size = float(operands[-1])  # type: ignore[arg-type]
            elif op == "Tr" and operands:
                state.render_mode = int(operands[-1])  # type: ignore[arg-type]
            elif op == "Tz" and operands:
                state.horizontal_scale = float(operands[-1]) / 100.0  # type: ignore[arg-type]
            elif op == "Tc" and operands:
                state.char_spacing = float(operands[-1])  # type: ignore[arg-type]
            elif op == "Tw" and operands:
                state.word_spacing = float(operands[-1])  # type: ignore[arg-type]
            elif op == "TL" and operands:
                state.leading = float(operands[-1])  # type: ignore[arg-type]
            elif op == "Ts" and operands:
                state.rise = float(operands[-1])  # type: ignore[arg-type]
            elif op == "Tm" and len(operands) >= 6:
                matrix = tuple(float(v) for v in operands[-6:])
                state.tm = state.tlm = matrix  # type: ignore[assignment]
                state.had_tm = True
                if current is not None and not positioned:
                    current.state_ops = raw[state_ops_start:op_start]
                    positioned = True
            elif op in ("Td", "TD") and len(operands) >= 2:
                tx, ty = float(operands[-2]), float(operands[-1])  # type: ignore[arg-type]
                if op == "TD":
                    state.leading = -ty
                state.tlm = multiply((1.0, 0.0, 0.0, 1.0, tx, ty), state.tlm)
                state.tm = state.tlm
                if current is not None and not positioned:
                    current.state_ops = raw[state_ops_start:op_start]
                    positioned = True
            elif op == "T*":
                state.tlm = multiply((1.0, 0.0, 0.0, 1.0, 0.0, -state.leading), state.tlm)
                state.tm = state.tlm
            elif op in ("Tj", "'", '"') and operands and isinstance(operands[-1], bytes):
                if op != "Tj":
                    if op == '"' and len(operands) >= 3:
                        state.word_spacing = float(operands[-3])  # type: ignore[arg-type]
                        state.char_spacing = float(operands[-2])  # type: ignore[arg-type]
                    state.tlm = multiply((1.0, 0.0, 0.0, 1.0, 0.0, -state.leading), state.tlm)
                    state.tm = state.tlm
                if current is not None:
                    _show(current, state, ctm, operands[-1], fonts, to_fitz)
            elif op == "TJ" and operands and isinstance(operands[-1], list):
                if current is not None:
                    for item in operands[-1]:
                        if isinstance(item, bytes):
                            _show(current, state, ctm, item, fonts, to_fitz)
                        elif isinstance(item, float):
                            shift = -item / 1000.0 * state.size * state.horizontal_scale
                            state.tm = multiply((1.0, 0.0, 0.0, 1.0, shift, 0.0), state.tm)
        finally:
            operands = []
    return words, orphan_objects


def _show(
    current: Word, state: _TextState, ctm: Matrix, data: bytes, fonts: dict[str, FontMetrics], to_fitz: Matrix
) -> None:
    """Показать строку глифов: добавить глифы в слово и продвинуть матрицу текста."""
    metrics = fonts.get(state.font) or FontMetrics(state.font, 1)
    if not current.glyphs:
        current.font, current.size = state.font, state.size
        current.matrix = state.tm[:4]
        current.shape = _shape(state)
    for code in metrics.codes(data):
        code_int = int.from_bytes(code, "big")
        origin_text = apply(state.tm, 0.0, state.rise)
        x_pdf, y_pdf = apply(ctm, *origin_text)
        origin_fitz = apply(to_fitz, x_pdf, y_pdf)
        width = metrics.width(code_int) * state.size
        advance = (width + state.char_spacing + (state.word_spacing if code == b" " else 0.0)) * state.horizontal_scale
        current.glyphs.append(
            Glyph(
                index=len(current.glyphs),
                code=code,
                origin_text=origin_text,
                origin=origin_fitz,
                advance=advance * math.hypot(state.tm[0], state.tm[1]) * math.hypot(ctm[0], ctm[1]),
            )
        )
        state.tm = multiply((1.0, 0.0, 0.0, 1.0, advance, 0.0), state.tm)


def _shape(state: _TextState) -> SpanShape:
    """Форма спана по состоянию на момент первого глифа."""
    if not state.had_tm:
        return SpanShape.TD_ONLY
    a, b, c, d = state.tm[:4]
    scale = max(abs(a), abs(b), abs(c), abs(d), 1e-9)
    if abs(b) > ROTATION_EPS * scale or abs(c) > ROTATION_EPS * scale:
        return SpanShape.ROTATED
    return SpanShape.PLAIN


# --- Шрифты ------------------------------------------------------------------------------


def _widths_simple(font: pikepdf.Object) -> tuple[dict[int, float], float]:
    """Ширины простого шрифта: ``/FirstChar`` + ``/Widths``; по умолчанию ``/MissingWidth``."""
    widths: dict[int, float] = {}
    first = int(font.get("/FirstChar", 0))
    for offset, value in enumerate(font.get("/Widths", [])):
        widths[first + offset] = float(value) / 1000.0
    descriptor = font.get("/FontDescriptor")
    missing = float(descriptor.get("/MissingWidth", 0)) / 1000.0 if descriptor is not None else 0.0
    return widths, missing or DEFAULT_GLYPH_WIDTH


def _widths_type0(font: pikepdf.Object) -> tuple[dict[int, float], float]:
    """Ширины составного шрифта: массив ``/W`` потомка в двух формах записи, по умолчанию ``/DW``."""
    widths: dict[int, float] = {}
    descendants = font.get("/DescendantFonts")
    if not descendants:
        return widths, DEFAULT_GLYPH_WIDTH
    child = descendants[0]
    default = float(child.get("/DW", 1000)) / 1000.0
    array = list(child.get("/W", []))
    i = 0
    while i < len(array):
        first = int(array[i])
        if i + 1 < len(array) and isinstance(array[i + 1], pikepdf.Array):
            for offset, value in enumerate(array[i + 1]):
                widths[first + offset] = float(value) / 1000.0
            i += 2
        elif i + 2 < len(array):
            last, value = int(array[i + 1]), float(array[i + 2]) / 1000.0
            for code in range(first, min(last, first + 65535) + 1):
                widths[code] = value
            i += 3
        else:
            break
    return widths, default


def _base14_font(base_font: str) -> "fitz.Font | None":
    """Шрифт PyMuPDF для стандартного имени (``Helvetica-Bold`` → ``hebo``), иначе None."""
    lowered = base_font.lower()
    family = (
        "helv"
        if "helvetica" in lowered or "arial" in lowered
        else "tiro" if "times" in lowered else "cour" if "courier" in lowered else ""
    )
    if not family:
        return None
    bold, italic = "bold" in lowered, ("italic" in lowered or "oblique" in lowered)
    name = {
        ("helv", False, False): "helv",
        ("helv", True, False): "hebo",
        ("helv", False, True): "heit",
        ("helv", True, True): "hebi",
        ("tiro", False, False): "tiro",
        ("tiro", True, False): "tibo",
        ("tiro", False, True): "tiit",
        ("tiro", True, True): "tibi",
        ("cour", False, False): "cour",
        ("cour", True, False): "cobo",
        ("cour", False, True): "coit",
        ("cour", True, True): "cobi",
    }[(family, bold, italic)]
    try:
        return fitz.Font(name)
    except Exception:  # noqa: BLE001
        return None


def page_fonts(page: pikepdf.Page) -> dict[str, FontMetrics]:
    """Метрики всех шрифтов из ресурсов страницы по имени ресурса.

    Args:
        page: Страница pikepdf.

    Returns:
        Словарь ``имя ресурса → FontMetrics``; шрифты без ширин получают ширину по умолчанию.
    """
    result: dict[str, FontMetrics] = {}
    resources = page.obj.get("/Resources")
    fonts = resources.get("/Font") if resources is not None else None
    if fonts is None:
        return result
    for key, font in fonts.items():
        name = str(key).lstrip("/")
        try:
            subtype = str(font.get("/Subtype", ""))
            if subtype == "/Type0":
                widths, default = _widths_type0(font)
                result[name] = FontMetrics(name, 2, widths, default)
            else:
                widths, default = _widths_simple(font)
                fallback = None
                if not widths:
                    fallback = _base14_font(str(font.get("/BaseFont", "")).lstrip("/"))
                result[name] = FontMetrics(name, 1, widths, default, fallback)
        except Exception as error:  # noqa: BLE001 — кривой шрифт не должен валить разбор страницы
            logger.warning("шрифт %s не разобран: %s", name, error)
            result[name] = FontMetrics(name, 1)
    return result


# --- Привязка к rawdict -------------------------------------------------------------------

# Шаг сетки индекса символов, pt: ячейка в разы крупнее допуска, поиск — по 3×3 соседям.
GRID_PT = 1.0


def _char_index(rawdict: dict) -> dict[tuple[int, int], list[tuple[float, float, str, fitz.Rect, int]]]:
    """Индекс символов rawdict по сетке начал: ``(x, y, символ, рамка, порядковый номер)``."""
    index: dict[tuple[int, int], list] = {}
    number = 0
    for block in rawdict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    x, y = char["origin"]
                    key = (int(math.floor(x / GRID_PT)), int(math.floor(y / GRID_PT)))
                    index.setdefault(key, []).append((x, y, char["c"], fitz.Rect(char["bbox"]), number))
                    number += 1
    return index


def attach_chars(words: list[Word], rawdict: dict, tolerance: float = MATCH_TOLERANCE_PT) -> tuple[int, int]:
    """Привязать к глифам символы rawdict по ближайшему началу.

    Args:
        words: Слова из :func:`parse_content`; глифы дополняются символом и рамкой на месте.
        rawdict: Результат ``page.get_text("rawdict")``.
        tolerance: Наибольшее расстояние между началом глифа и ``origin`` символа, pt.

    Returns:
        Пара: число глифов без символа и число символов, к которым не пришёл ни один глиф.
    """
    index = _char_index(rawdict)
    total_chars = sum(len(items) for items in index.values())
    used: set[int] = set()
    unmatched = 0
    for word in words:
        for glyph in word.glyphs:
            gx, gy = glyph.origin
            cx, cy = int(math.floor(gx / GRID_PT)), int(math.floor(gy / GRID_PT))
            best = None
            best_distance = tolerance
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for x, y, char, bbox, number in index.get((cx + dx, cy + dy), ()):
                        if number in used:
                            continue
                        distance = math.hypot(x - gx, y - gy)
                        if distance <= best_distance:
                            best, best_distance = (char, bbox, number), distance
            if best is None:
                unmatched += 1
                glyph.bbox = _fallback_bbox(word, glyph)
                continue
            glyph.char, glyph.bbox = best[0], best[1]
            used.add(best[2])
    return unmatched, total_chars - len(used)


def _fallback_bbox(word: Word, glyph: Glyph) -> fitz.Rect:
    """Рамка глифа без символа: от начала на ширину продвижения и на кегль по вертикали."""
    x, y = glyph.origin
    dx, dy = word.direction
    height = word.size * max(abs(word.matrix[3]), abs(word.matrix[2]), 1e-3)
    length = max(glyph.advance, 0.1)
    # Прямоугольник вдоль направления строки: базовая линия — начало, выносные — поперёк.
    px, py = -dy, dx
    corners = [
        (x - px * height * DESCENT, y - py * height * DESCENT),
        (x + px * height * ASCENT, y + py * height * ASCENT),
        (x + dx * length - px * height * DESCENT, y + dy * length - py * height * DESCENT),
        (x + dx * length + px * height * ASCENT, y + dy * length + py * height * ASCENT),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return fitz.Rect(min(xs), min(ys), max(xs), max(ys))


# --- Дерево структуры ---------------------------------------------------------------------


def struct_lines(pdf: pikepdf.Pdf, page_index: int) -> dict[int, str]:
    """Какому элементу дерева структуры (строке/абзацу FineReader) принадлежит каждый спан.

    Args:
        pdf: Открытый pikepdf-документ.
        page_index: Номер страницы с нуля.

    Returns:
        Словарь ``MCID → идентификатор элемента`` (``"obj gen"``); пусто, если дерева нет.
    """
    try:
        page = pdf.pages[page_index]
        parents_key = page.obj.get("/StructParents")
        tree_root = pdf.Root.get("/StructTreeRoot")
        if parents_key is None or tree_root is None or tree_root.get("/ParentTree") is None:
            return {}
        tree = pikepdf.NumberTree(tree_root.ParentTree)
        entries = tree.get(int(parents_key))
        if entries is None:
            return {}
        result: dict[int, str] = {}
        for mcid, element in enumerate(entries):
            if element is None or not isinstance(element, pikepdf.Dictionary):
                continue
            objgen = element.objgen
            result[mcid] = f"{objgen[0]} {objgen[1]}" if objgen != (0, 0) else f"inline{mcid}"
        return result
    except Exception as error:  # noqa: BLE001 — без дерева структуры слой всё равно разбирается
        logger.warning("дерево структуры страницы %d не прочитано: %s", page_index, error)
        return {}


# --- Сборка --------------------------------------------------------------------------------


def page_content(page: fitz.Page) -> bytes:
    """Склеенный распакованный поток содержимого страницы (у FineReader он один).

    Args:
        page: Страница PyMuPDF.

    Returns:
        Байты потока; несколько потоков склеиваются через перевод строки, как делает просмотрщик.
    """
    parts = [page.parent.xref_stream(xref) or b"" for xref in page.get_contents()]
    return b"\n".join(parts)


def load_layer(page: fitz.Page, pdf: "pikepdf.Pdf | None" = None) -> TextLayer:
    """Разобрать текстовый слой страницы: слова, глифы, символы, строки дерева.

    Args:
        page: Страница PyMuPDF (документ должен быть открыт на чтение).
        pdf: Тот же документ в pikepdf — для метрик шрифтов и дерева структуры; без него
            ширины глифов берутся по умолчанию, а ``struct_line`` не заполняется.

    Returns:
        Слой страницы со статистикой привязки.
    """
    fonts: dict[str, FontMetrics] = {}
    lines: dict[int, str] = {}
    if pdf is not None:
        fonts = page_fonts(pdf.pages[page.number])
        lines = struct_lines(pdf, page.number)
    words, _ = parse_content(page_content(page), page_matrix(page), fonts)
    unmatched, orphans = attach_chars(words, page.get_text("rawdict", flags=0))
    offpage = count_offpage(words, page.rect)
    for word in words:
        if word.mcid is not None:
            word.struct_line = lines.get(word.mcid)
    return TextLayer(
        page.number,
        page.rect,
        words,
        unmatched_glyphs=unmatched - offpage,
        orphan_chars=orphans,
        offpage_glyphs=offpage,
    )


def count_offpage(words: list[Word], page_rect: fitz.Rect) -> int:
    """Сколько глифов без символа начинаются вне страницы (их и не могло быть в rawdict).

    Args:
        words: Слова после :func:`attach_chars`.
        page_rect: Рамка страницы в координатах fitz.

    Returns:
        Число таких глифов.
    """
    return sum(
        1 for word in words for g in word.glyphs if g.char is None and not page_rect.contains(fitz.Point(*g.origin))
    )
