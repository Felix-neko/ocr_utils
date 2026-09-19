"""Правка текстового слоя: удаление и усечение слов FineReader, вставка своего невидимого текста.

ПРАВИЛО УДАЛЕНИЯ. Вырезается только текстовый объект ``BT … ET`` внутри спана; сам спан
``/Span <</MCID n>> BDC … EMC`` остаётся пустым. Так страничный сдвиг ``cm``, который FineReader
кладёт внутрь первого спана, остаётся на месте (иначе уехала бы картинка), а ссылки дерева
структуры на MCID остаются действительными — дерево чистить не нужно.

УСЕЧЕНИЕ. Слово, задетое зоной частично, пересобирается из оставленных глифов: те же
операторы состояния (``Tf`` …), ``Tm`` с прежними ``a b c d`` и началом первого оставленного
глифа, между глифами ``Td`` в текстовом пространстве, коды глифов — как в исходнике,
шестнадцатеричными строками. Шрифт и кодировка не меняются.

ВСТАВКА. Своё чтение пишется в тот же поток: ``BT /<ресурс> <кегль> Tf 3 Tr a b c d e f Tm
<GID…> Tj ET`` со шрифтом Noto Sans, встроенным через ``page.insert_font`` (Type0,
Identity-H, ToUnicode есть — поиск и ``pdftotext`` его читают). Матрица задаётся напрямую:
угол по ``rotate_cw`` из ``rotated_text`` (90 — текст читается снизу вверх), растяжение —
чтобы строка легла в длинную сторону зоны. Никаких ``TextWriter``/``insert_text``: они
плодят по потоку на вызов и не дают управлять матрицей.

Картинка страницы и все прочие объекты не трогаются: меняется один поток содержимого.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from fontTools.ttLib import TTFont

from ocr_utils.text_layer_fix.text_layer import Glyph, Matrix, Word, apply, multiply, page_matrix

logger = logging.getLogger(__name__)

# Шрифт для вставок: полная кириллица, есть в системе.
DEFAULT_FONT_PATH = Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf")
FONT_RESOURCE = "TLFnoto"

# Кегль вставки, pt: не крупнее обычного текста и не мельче, чем выделяется мышью.
MAX_FONT_PT = 10.0
MIN_FONT_PT = 2.5

# Доля кегля над и под базовой линией при укладке строк в зону.
LINE_ASCENT = 0.78
LINE_HEIGHT = 1.15

# Невидимый текст: режим 3 — без заливки и обводки.
INVISIBLE_RENDER_MODE = 3

# Шаг перебора кегля при укладке слов в зону, pt.
FONT_STEP_PT = 0.25

# Слово слоя, вылезшее за обрезанную страницу, ужимается в границу, но не меньше стольких pt по
# каждой стороне: иначе оно выпадет из извлечения (rawdict не видит глифы за CropBox).
MIN_REFIT_PT = 2.0


@dataclass(frozen=True)
class Insert:
    """Что вписать: текст (строки через перевод строки), рамка зоны в pt и поворот текста."""

    text: str
    rect: fitz.Rect
    # На сколько градусов по часовой повернуть страницу, чтобы этот текст стал прямым
    # (валюта ``rotated_text.tables.orientation``): 90 — читается снизу вверх, 270 — сверху вниз.
    rotate_cw: int = 0
    fontsize: "float | None" = None


@dataclass
class EditStats:
    """Что сделано с одной страницей."""

    blanked: int = 0
    trimmed: int = 0
    inserted: int = 0
    inserted_lines: int = 0
    refitted: int = 0  # слов ужато в границу обрезанной страницы
    skipped_inserts: list[str] = field(default_factory=list)


# --- Удаление и усечение ------------------------------------------------------------------


def _splice(raw: bytes, edits: list[tuple[int, int, bytes]]) -> bytes:
    """Заменить непересекающиеся диапазоны байт, идя с конца, чтобы смещения не поплыли."""
    out = raw
    for start, end, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        out = out[:start] + replacement + out[end:]
    return out


def blank_edits(words: list[Word]) -> list[tuple[int, int, bytes]]:
    """Правки «вырезать ``BT … ET``» для слов (диапазоны — в исходном потоке)."""
    return [(start, end, b" ") for word in words for start, end in word.text_ranges]


def blank_words(raw: bytes, words: list[Word]) -> bytes:
    """Убрать текстовые объекты указанных слов, оставив их спаны пустыми.

    Args:
        raw: Поток содержимого страницы.
        words: Слова, чьи ``BT … ET`` вырезаются.

    Returns:
        Новый поток.
    """
    return _splice(raw, blank_edits(words))


def _hex_string(codes: list[bytes]) -> bytes:
    return b"<" + b"".join(code.hex().encode("ascii") for code in codes) + b">"


def _fmt(value: float) -> bytes:
    return f"{value:.4f}".rstrip("0").rstrip(".").encode("ascii") or b"0"


def rebuild_text_object(word: Word, keep: tuple[int, ...]) -> bytes:
    """Собрать ``BT … ET`` слова заново только из глифов с индексами ``keep``.

    Args:
        word: Разобранное слово.
        keep: Индексы оставляемых глифов (порядок не важен, сортируется).

    Returns:
        Байты нового текстового объекта; пустая строка, если оставлять нечего.
    """
    kept = [word.glyphs[i] for i in sorted(set(keep)) if 0 <= i < len(word.glyphs)]
    if not kept:
        return b""
    a, b, c, d = word.matrix
    det = a * d - b * c
    parts = [b"BT ", word.state_ops.strip(), b" "]
    first = kept[0]
    parts += [
        _fmt(a),
        b" ",
        _fmt(b),
        b" ",
        _fmt(c),
        b" ",
        _fmt(d),
        b" ",
        _fmt(first.origin_text[0]),
        b" ",
        _fmt(first.origin_text[1]),
        b" Tm ",
    ]
    previous = first
    for glyph in kept:
        if glyph is not previous:
            dx = glyph.origin_text[0] - previous.origin_text[0]
            dy = glyph.origin_text[1] - previous.origin_text[1]
            # Td задаётся в текстовом пространстве до Tm: (tx, ty) × [[a, b], [c, d]] = (dx, dy).
            if abs(det) > 1e-9:
                tx = (dx * d - dy * c) / det
                ty = (dy * a - dx * b) / det
            else:
                tx, ty = dx, dy
            parts += [_fmt(tx), b" ", _fmt(ty), b" Td "]
        parts += [_hex_string([glyph.code]), b"Tj "]
        previous = glyph
    parts.append(b"ET")
    return b"".join(parts)


def trim_edits(trims: dict[int, tuple[Word, tuple[int, ...]]]) -> list[tuple[int, int, bytes]]:
    """Правки «пересобрать ``BT … ET`` из оставленных глифов» (диапазоны — в исходном потоке)."""
    edits: list[tuple[int, int, bytes]] = []
    for word, keep in trims.values():
        if not word.text_ranges:
            continue
        start, _ = word.text_ranges[0]
        _, end = word.text_ranges[-1]
        edits.append((start, end, rebuild_text_object(word, keep)))
    return edits


def trim_words(raw: bytes, trims: dict[int, tuple[Word, tuple[int, ...]]]) -> bytes:
    """Усечь слова до оставленных глифов.

    Args:
        raw: Поток содержимого страницы.
        trims: ``id(слова) → (слово, индексы оставляемых глифов)``.

    Returns:
        Новый поток.
    """
    return _splice(raw, trim_edits(trims))


# --- Вставка -------------------------------------------------------------------------------


class InsertFont:
    """Шрифт вставок: глифы по fontTools, длины по PyMuPDF, ресурс на странице по ``insert_font``."""

    def __init__(self, path: Path = DEFAULT_FONT_PATH) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"нет шрифта для вставок текстового слоя: {self.path} (нужен TTF с кириллицей, "
                "например пакет fonts-noto-core; другой файл — опция --insert-font)"
            )
        self.ttf = TTFont(str(self.path))
        self.cmap = self.ttf.getBestCmap()
        self.fitz_font = fitz.Font(fontfile=str(self.path))
        self._space = self.gid(" ")

    def gid(self, char: str) -> "int | None":
        """Номер глифа символа или None, если его нет в шрифте."""
        name = self.cmap.get(ord(char))
        return self.ttf.getGlyphID(name) if name else None

    def encode(self, text: str) -> list[bytes]:
        """Коды глифов (2 байта, Identity-H) для строки; символы без глифа — пробел."""
        codes: list[bytes] = []
        for char in text:
            gid = self.gid(char)
            if gid is None:
                gid = self._space or 0
            codes.append(gid.to_bytes(2, "big"))
        return codes

    def length(self, text: str, fontsize: float) -> float:
        """Длина строки в pt при кегле ``fontsize``."""
        return self.fitz_font.text_length(text, fontsize=fontsize)

    def ensure_resource(self, page: fitz.Page) -> str:
        """Добавить шрифт в ресурсы страницы (повторно — без дубля) и вернуть имя ресурса."""
        names = {entry[4] for entry in page.get_fonts(full=True)}
        if FONT_RESOURCE not in names:
            page.insert_font(fontname=FONT_RESOURCE, fontfile=str(self.path))
        return FONT_RESOURCE


def fit_fontsize(font: InsertFont, lines: list[str], length_pt: float, thickness_pt: float) -> float:
    """Кегль, при котором строки помещаются в зону по длине и по толщине.

    Args:
        font: Шрифт вставки.
        lines: Строки текста (уже разбитые).
        length_pt: Длина зоны вдоль строки, pt.
        thickness_pt: Толщина зоны поперёк строк, pt (на все строки).

    Returns:
        Кегль в pt, ограниченный ``MIN_FONT_PT``..``MAX_FONT_PT``.
    """
    longest = max((font.length(line, 1.0) for line in lines if line), default=1.0)
    by_length = length_pt / longest if longest > 0 else MAX_FONT_PT
    by_height = thickness_pt / (LINE_HEIGHT * max(1, len(lines)))
    return max(MIN_FONT_PT, min(MAX_FONT_PT, by_length, by_height))


def wrap_lines(font: InsertFont, text: str, length_pt: float, thickness_pt: float) -> tuple[list[str], float]:
    """Уложить слова текста в зону с переносом по словам, выбрав наибольший кегль, при котором всё
    помещается и по длине строк, и по толщине зоны.

    Строки, как их отдал OCR, не сохраняются: текст режется на слова заново. Слово длиннее зоны
    идёт своей строкой (её ужмёт растяжение в :func:`text_object`).

    Args:
        font: Шрифт вставки.
        text: Текст (переводы строк — как пробелы).
        length_pt: Длина зоны вдоль строки, pt.
        thickness_pt: Толщина зоны поперёк строк, pt.

    Returns:
        Строки и кегль; пустой текст — ``([], 0.0)``.
    """
    words = text.split()
    if not words:
        return [], 0.0
    fontsize = MAX_FONT_PT
    while True:
        lines = _wrap_at(font, words, length_pt, fontsize)
        if len(lines) * LINE_HEIGHT * fontsize <= thickness_pt or fontsize <= MIN_FONT_PT:
            return lines, max(MIN_FONT_PT, fontsize)
        fontsize = max(MIN_FONT_PT, fontsize - FONT_STEP_PT)


def _wrap_at(font: InsertFont, words: list[str], length_pt: float, fontsize: float) -> list[str]:
    """Жадная укладка слов в строки не длиннее ``length_pt`` при данном кегле."""
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if current and font.length(candidate, fontsize) > length_pt:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _frame(
    rect: fitz.Rect, rotate_cw: int
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], float, float]:
    """Система «прямого» текста для зоны: начало, орт вдоль строки, орт вниз по строкам (в
    координатах fitz, y вниз), длина зоны вдоль строки и толщина поперёк."""
    angle = rotate_cw % 360
    if angle == 90:  # читается снизу вверх: строки идут слева направо, начало — левый нижний угол
        return (rect.x0, rect.y1), (0.0, -1.0), (1.0, 0.0), rect.height, rect.width
    if angle == 270:  # сверху вниз: начало — правый верхний угол, строки справа налево
        return (rect.x1, rect.y0), (0.0, 1.0), (-1.0, 0.0), rect.height, rect.width
    if angle == 180:  # вверх ногами: начало — правый нижний угол
        return (rect.x1, rect.y1), (-1.0, 0.0), (0.0, -1.0), rect.width, rect.height
    return (rect.x0, rect.y0), (1.0, 0.0), (0.0, 1.0), rect.width, rect.height


def _invert(m: Matrix) -> Matrix:
    """Обратная матрица PDF."""
    a, b, c, d, e, f = m
    det = a * d - b * c
    ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
    return (ia, ib, ic, id_, -(e * ia + f * ic), -(e * ib + f * id_))


def text_object(insert: Insert, font: InsertFont, resource: str, to_fitz: Matrix) -> tuple[bytes, int, float]:
    """Текстовый объект вставки для потока содержимого.

    Args:
        insert: Что и куда вписать.
        font: Шрифт вставки.
        resource: Имя ресурса шрифта на странице.
        to_fitz: Матрица «PDF → fitz» страницы (:func:`text_layer.page_matrix`); вставка
            переводится обратной матрицей, поэтому MediaBox не от нуля не мешает.

    Returns:
        Байты ``BT … ET``, число строк и выбранный кегль.
    """
    origin, along, down, length, thickness = _frame(insert.rect, insert.rotate_cw)
    if insert.fontsize:
        lines, fontsize = _wrap_at(font, insert.text.split(), length, insert.fontsize), insert.fontsize
    else:
        lines, fontsize = wrap_lines(font, insert.text, length, thickness)
    if not lines:
        return b"", 0, 0.0
    line_step = fontsize * LINE_HEIGHT
    # Если строк по толщине помещается меньше, чем есть, сжимаем шаг, но не кегль: пусть лучше
    # строки перекроются в невидимом слое, чем текст выйдет за зону и наедет на соседей.
    if line_step * len(lines) > thickness and len(lines) > 1:
        line_step = thickness / len(lines)
    # Растяжение вдоль строки, чтобы самая длинная строка легла ровно в зону: FineReader делает так же.
    longest = max(font.length(line, fontsize) for line in lines)
    stretch = min(1.0, length / longest) if longest > 0 else 1.0
    # Матрица текста в координатах PDF: орт «вдоль» — направление строки, орт «вверх» — против
    # ``down``. Векторы переводятся из fitz в PDF линейной частью обратной матрицы, точка — целиком.
    to_pdf = _invert(to_fitz)
    ia, ib, ic, id_ = to_pdf[:4]
    ax, ay = (along[0] * ia + along[1] * ic) * stretch, (along[0] * ib + along[1] * id_) * stretch
    ux, uy = -(down[0] * ia + down[1] * ic), -(down[0] * ib + down[1] * id_)
    parts = [
        b"BT /",
        resource.encode("ascii"),
        b" ",
        _fmt(fontsize),
        b" Tf ",
        str(INVISIBLE_RENDER_MODE).encode(),
        b" Tr ",
    ]
    for number, line in enumerate(lines):
        offset = fontsize * LINE_ASCENT + number * line_step
        x = origin[0] + down[0] * offset
        y = origin[1] + down[1] * offset
        px, py = apply(to_pdf, x, y)
        parts += [_fmt(ax), b" ", _fmt(ay), b" ", _fmt(ux), b" ", _fmt(uy), b" ", _fmt(px), b" ", _fmt(py), b" Tm "]
        parts += [_hex_string(font.encode(line)), b"Tj "]
    parts.append(b"ET")
    return b"".join(parts), len(lines), fontsize


# --- Подгонка слов под обрезанную страницу -------------------------------------------------


def refit_target(bbox: fitz.Rect, crop: fitz.Rect, min_pt: float = MIN_REFIT_PT) -> "fitz.Rect | None":
    """Куда ужать рамку слова, чтобы она легла внутрь обрезанной страницы.

    Args:
        bbox: Рамка слова (fitz, координаты исходной страницы).
        crop: Рамка обрезанной страницы в тех же координатах.
        min_pt: Наименьшая сторона результата, pt.

    Returns:
        ``None`` — слово целиком внутри, трогать не надо; иначе пересечение рамок, расширенное
        внутрь страницы до ``min_pt`` по каждой стороне; слово целиком снаружи — полоска у
        ближайшего края.
    """
    if bbox.is_empty or crop.contains(bbox):
        return None
    x0, x1 = max(bbox.x0, crop.x0), min(bbox.x1, crop.x1)
    y0, y1 = max(bbox.y0, crop.y0), min(bbox.y1, crop.y1)
    x0, x1 = _widen(x0, x1, crop.x0, crop.x1, min_pt)
    y0, y1 = _widen(y0, y1, crop.y0, crop.y1, min_pt)
    return fitz.Rect(x0, y0, x1, y1)


def _widen(low: float, high: float, bound_low: float, bound_high: float, min_size: float) -> tuple[float, float]:
    """Отрезок ``[low, high]`` (возможно вырожденный или вывернутый) → не короче ``min_size`` внутри границ."""
    min_size = min(min_size, bound_high - bound_low)
    if high < low:  # пересечения нет: к ближайшему краю (отрезок целиком до нижней или за верхней границей)
        edge = bound_low if high < bound_low else bound_high
        low = high = edge
    if high - low < min_size:
        center = (low + high) / 2
        low, high = center - min_size / 2, center + min_size / 2
        if low < bound_low:
            low, high = bound_low, bound_low + min_size
        if high > bound_high:
            low, high = bound_high - min_size, bound_high
    return low, high


def _mat2(m: Matrix) -> tuple[float, float, float, float]:
    return m[0], m[1], m[2], m[3]


def _mul2(
    p: tuple[float, float, float, float], q: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Произведение линейных частей в соглашении PDF (строка-вектор: ``v · p · q``)."""
    a, b, c, d = p
    e, f, g, h = q
    return a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h


def refit_text_object(word: Word, target: fitz.Rect, to_fitz: Matrix, keep: "tuple[int, ...] | None" = None) -> bytes:
    """Пересобрать ``BT … ET`` слова так, чтобы его рамка легла в ``target``.

    Отображение «рамка слова → target» — осевое (масштаб по x и y плюс перенос) в координатах
    fitz; оно переводится в пространство текстового объекта через CTM слова и матрицу страницы,
    так что кегль, растяжение и порядок глифов не меняются — меняются только ``Tm`` и ``Td``.

    Args:
        word: Разобранное слово (нужны ``bbox``, ``ctm``, ``glyphs``).
        target: Куда положить рамку слова (fitz).
        to_fitz: Матрица «PDF → fitz» страницы.
        keep: Индексы оставляемых глифов (усечение); ``None`` — все.

    Returns:
        Байты нового текстового объекта; пустая строка, если глифов нет.
    """
    glyphs = word.glyphs if keep is None else [word.glyphs[i] for i in sorted(set(keep)) if 0 <= i < len(word.glyphs)]
    bbox = word.bbox
    if not glyphs or bbox.is_empty:
        return b""
    sx = target.width / bbox.width if bbox.width > 1e-9 else 1.0
    sy = target.height / bbox.height if bbox.height > 1e-9 else 1.0
    # Точка текстового пространства → fitz: сначала CTM слова, потом матрица страницы.
    user_to_fitz = multiply(word.ctm, to_fitz)
    fitz_to_user = _invert(user_to_fitz)

    def to_user(fx: float, fy: float) -> tuple[float, float]:
        return apply(fitz_to_user, fx, fy)

    # Линейная часть отображения в пространстве текста: L⁻¹ · S · L (строка-вектор).
    lin = _mat2(user_to_fitz)
    scale_user = _mul2(_mul2(lin, (sx, 0.0, 0.0, sy)), _mat2(fitz_to_user))
    a, b, c, d = word.matrix
    a2, b2 = a * scale_user[0] + b * scale_user[2], a * scale_user[1] + b * scale_user[3]
    c2, d2 = c * scale_user[0] + d * scale_user[2], c * scale_user[1] + d * scale_user[3]
    det = a2 * d2 - b2 * c2

    def mapped(glyph: Glyph) -> tuple[float, float]:
        """Новое начало глифа в текстовом пространстве: через fitz, где отображение осевое."""
        fx, fy = apply(user_to_fitz, *glyph.origin_text)
        return to_user(target.x0 + (fx - bbox.x0) * sx, target.y0 + (fy - bbox.y0) * sy)

    parts = [b"BT ", word.state_ops.strip(), b" "]
    first = mapped(glyphs[0])
    parts += [
        _fmt(a2),
        b" ",
        _fmt(b2),
        b" ",
        _fmt(c2),
        b" ",
        _fmt(d2),
        b" ",
        _fmt(first[0]),
        b" ",
        _fmt(first[1]),
        b" Tm ",
    ]
    previous = first
    for number, glyph in enumerate(glyphs):
        if number:
            current = mapped(glyph)
            dx, dy = current[0] - previous[0], current[1] - previous[1]
            if abs(det) > 1e-9:
                tx, ty = (dx * d2 - dy * c2) / det, (dy * a2 - dx * b2) / det
            else:
                tx, ty = dx, dy
            parts += [_fmt(tx), b" ", _fmt(ty), b" Td "]
            previous = current
        parts += [_hex_string([glyph.code]), b"Tj "]
    parts.append(b"ET")
    return b"".join(parts)


def refit_edits(
    refits: dict[int, tuple[Word, fitz.Rect]],
    to_fitz: Matrix,
    trims: "dict[int, tuple[Word, tuple[int, ...]]] | None" = None,
) -> list[tuple[int, int, bytes]]:
    """Правки «пересобрать слово в новую рамку» (диапазоны — в исходном потоке); усечённые слова
    пересобираются из оставленных глифов."""
    edits: list[tuple[int, int, bytes]] = []
    for key, (word, target) in refits.items():
        if not word.text_ranges:
            continue
        keep = trims[key][1] if trims and key in trims else None
        start, _ = word.text_ranges[0]
        _, end = word.text_ranges[-1]
        edits.append((start, end, refit_text_object(word, target, to_fitz, keep)))
    return edits


# --- Применение и проверка ----------------------------------------------------------------


def apply_edits(
    doc: fitz.Document,
    page_index: int,
    raw: bytes,
    blanks: list[Word],
    trims: dict[int, tuple[Word, tuple[int, ...]]],
    inserts: list[Insert],
    font: "InsertFont | None" = None,
    refits: "dict[int, tuple[Word, fitz.Rect]] | None" = None,
) -> EditStats:
    """Записать правки в поток содержимого страницы документа (документ — открытая копия).

    Args:
        doc: Документ PyMuPDF, открытый из КОПИИ исходного PDF.
        page_index: Номер страницы с нуля.
        raw: Исходный поток, по которому разбирались ``blanks``/``trims`` (смещения должны совпадать).
        blanks: Слова на удаление.
        trims: Слова на усечение (см. :func:`trim_words`).
        inserts: Вставки.
        font: Шрифт вставок; None — Noto Sans по умолчанию.
        refits: Слова, которые надо ужать в новую рамку (``id(слова) → (слово, рамка fitz)``), —
            для страниц, обрезаемых после правки; слово и в ``trims`` — усекается и ужимается.

    Returns:
        Статистика правок.
    """
    stats = EditStats()
    page = doc[page_index]
    xrefs = page.get_contents()
    if len(xrefs) != 1:
        raise ValueError(f"страница {page_index}: потоков содержимого {len(xrefs)}, ожидается один")
    refits = refits or {}
    # Все правки — одним проходом по ИСХОДНОМУ потоку: диапазоны слов посчитаны по нему, и
    # после первой же вырезки они бы поплыли (замер: усечение после удалений теряло соседние слова).
    plain_trims = {key: value for key, value in trims.items() if key not in refits}
    edited = _splice(raw, blank_edits(blanks) + trim_edits(plain_trims) + refit_edits(refits, page_matrix(page), trims))
    stats.blanked, stats.trimmed, stats.refitted = len(blanks), len(trims), len(refits)
    if inserts:
        font = font or InsertFont()
        resource = font.ensure_resource(page)
        objects: list[bytes] = []
        for insert in inserts:
            data, lines, _ = text_object(insert, font, resource, page_matrix(page))
            if not data:
                stats.skipped_inserts.append("пустой текст")
                continue
            objects.append(data)
            stats.inserted += 1
            stats.inserted_lines += lines
        if objects:
            # Свой блок в конце потока, в собственной паре q/Q: состояние страницы не наследуется.
            edited = edited.rstrip() + b"\nq " + b" ".join(objects) + b" Q\n"
    doc.update_stream(xrefs[0], edited)
    return stats


@dataclass
class VerifyReport:
    """Сверка страницы после правки с тем, что ожидалось."""

    kept_missing: int = 0  # слов KEEP не нашлось на прежнем месте
    deleted_remaining: int = 0  # удалённых слов всё ещё видно в извлечении
    inserts_missing: int = 0  # вставок не находит поиск
    refits_missing: int = 0  # ужатых слов не находит поиск в их новой рамке
    image_changed: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.kept_missing
            or self.deleted_remaining
            or self.inserts_missing
            or self.refits_missing
            or self.image_changed
        )


def image_digest(doc: fitz.Document, xref: int) -> str:
    """md5 сырых байт картинки — чтобы убедиться, что растр не пересжат."""
    return hashlib.md5(doc.xref_stream_raw(xref)).hexdigest()


def pixels_digest(doc: fitz.Document, xref: int) -> str:
    """md5 декодированных пикселей картинки — когда сырые байты пересжаты (Flate при ``deflate=True``), а сам растр цел."""
    pixmap = fitz.Pixmap(doc, xref)
    return hashlib.md5(bytes([pixmap.width, pixmap.height & 255, pixmap.n]) + pixmap.samples).hexdigest()


def same_image(before: fitz.Document, before_xref: int, after: fitz.Document, after_xref: int) -> bool:
    """Та же ли картинка: сначала по сырым байтам (JBIG2 не пересжимается), иначе по пикселям."""
    if image_digest(before, before_xref) == image_digest(after, after_xref):
        return True
    return pixels_digest(before, before_xref) == pixels_digest(after, after_xref)


def _char_index(page: fitz.Page) -> dict[tuple[int, int], list[tuple[float, float, str, str]]]:
    """Символы страницы по сетке начал (как в ``text_layer``), для посимвольной сверки: (x, y, символ, шрифт)."""
    index: dict[tuple[int, int], list[tuple[float, float, str, str]]] = {}
    for block in page.get_text("rawdict", flags=0)["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    x, y = char["origin"]
                    index.setdefault((int(math.floor(x)), int(math.floor(y))), []).append(
                        (x, y, char["c"], span["font"])
                    )
    return index


def _font_key(name: str) -> str:
    """Имя шрифта без пробелов и дефисов в нижнем регистре: fitz зовёт шрифт «Noto Sans Regular», rawdict — «NotoSans-Regular»."""
    return name.replace(" ", "").replace("-", "").lower()


def _glyph_present(
    index: dict, glyph_origin: tuple[float, float], char: str, tolerance: float = 0.35, ignore_font: "str | None" = None
) -> bool:
    """Есть ли на странице такой символ с началом в ``tolerance`` pt от ожидаемого.

    ``ignore_font`` — символы этого шрифта не считаются (свои вставки: они ложатся туда же, где
    стояла удалённая россыпь, и та же буква на том же месте — не «неудалённое» слово).
    """
    gx, gy = glyph_origin
    cx, cy = int(math.floor(gx)), int(math.floor(gy))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for x, y, c, font_name in index.get((cx + dx, cy + dy), ()):
                if ignore_font and _font_key(font_name) == ignore_font:
                    continue
                if c == char and math.hypot(x - gx, y - gy) <= tolerance:
                    return True
    return False


def verify_page(
    before: fitz.Page,
    after: fitz.Page,
    kept: list[Word],
    deleted: list[Word],
    inserts: list[Insert],
    image_xref: "int | None" = None,
    after_image_xref: "int | None" = None,
    font: "InsertFont | None" = None,
    shift: tuple[float, float] = (0.0, 0.0),
    refits: "list[tuple[Word, fitz.Rect]] | None" = None,
) -> VerifyReport:
    """Сверить страницу-копию с ожиданиями (посимвольно, по началам глифов).

    Args:
        before: Страница исходного PDF.
        after: Та же страница исправленной копии.
        kept: Слова, которые должны были остаться: каждый их глиф ищется на прежнем месте.
        deleted: Слова, которых быть не должно: ни один их глиф не должен найтись.
        inserts: Вставки, которые должен находить ``search_for`` в своей рамке.
        image_xref: Картинка страницы в исходном документе для сверки md5; ``None`` — не сверять.
        after_image_xref: Та же картинка в копии, если её xref отличается (страница скопирована в
            новый документ); по умолчанию тот же xref, что и ``image_xref``.
        font: Шрифт вставок — чтобы искать первую строку в той же укладке (:func:`wrap_lines`);
            без него ищется первое слово.
        shift: На сколько сдвинуты координаты копии относительно исходной страницы (fitz):
            у обрезанной страницы — минус начало CropBox.
        refits: Ужатые слова с новыми рамками: каждое должно находиться поиском в своей рамке.

    Returns:
        Отчёт сверки.
    """
    report = VerifyReport()
    index = _char_index(after)
    dx, dy = shift
    for word in kept:
        glyphs = [g for g in word.glyphs if g.char and not g.char.isspace()]
        if glyphs and not all(_glyph_present(index, (g.origin[0] + dx, g.origin[1] + dy), g.char) for g in glyphs):
            report.kept_missing += 1
    own_font = _font_key(font.fitz_font.name) if font is not None else None
    for word in deleted:
        glyphs = [g for g in word.glyphs if g.char and not g.char.isspace()]
        if glyphs and any(
            _glyph_present(index, (g.origin[0] + dx, g.origin[1] + dy), g.char, ignore_font=own_font) for g in glyphs
        ):
            report.deleted_remaining += 1
    for insert in inserts:
        if font is not None:
            _, _, _, length, thickness = _frame(insert.rect, insert.rotate_cw)
            lines, _ = wrap_lines(font, insert.text, length, thickness)
            first = lines[0] if lines else ""
        else:
            first = next(iter(insert.text.split()), "")
        if not first:
            continue
        rect = insert.rect + (dx, dy, dx, dy)
        if not any(hit.intersects(rect) for hit in after.search_for(first)):
            report.inserts_missing += 1
            report.notes.append(f"не найдена вставка «{first[:30]}»")
    for word, target in refits or ():
        text = word.text.strip()
        if not text:
            continue
        rect = target + (dx, dy, dx, dy)
        if not any(hit.intersects(rect) for hit in after.search_for(text)):
            report.refits_missing += 1
            report.notes.append(f"не найдено ужатое слово «{text[:30]}»")
    if image_xref is not None:
        try:
            after_xref = image_xref if after_image_xref is None else after_image_xref
            report.image_changed = not same_image(before.parent, image_xref, after.parent, after_xref)
        except Exception as error:  # noqa: BLE001
            report.notes.append(f"md5 картинки не сверен: {error}")
    return report
