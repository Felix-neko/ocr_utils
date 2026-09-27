"""Второй проход DeepSeek-OCR-2: слова первого прохода залиты белым, остаток снова смотрят DeepSeek и классический детектор line art.

Зачем. Там, где DeepSeek на первом проходе видел только текст, рядом с надписью может прятаться
рисунок (ветвь у «100 лет», 1970/04 с.14–22) или буквица рубрики-вензеля. Если стереть слова,
которые модель обвела, в картинке остаётся только то, что текстом не было. Нашёл ли там что-то
DeepSeek (``image``/``table``/``equation``) — это объект; не нашёл — область и правда надпись.

Шаги на область:

1. Вырезка бинаризуется Оцу по рамке области (краска 0, бумага 255, вне рамки — поле как есть).
2. Каждая рамка слова DeepSeek (промпт ``ocr``) достраивается по буквоподобным пятнам, которые
   задевает (крупнейшее пятно — буквица с линейкой — не достраивает, иначе рамка слова съест её),
   и заливается белым.
3. По залитой вырезке — DeepSeek ``markdown`` (отдельный прогон воркера) и классический
   ``line_art.features.analyse_gray`` при 300 dpi.
4. Вердикт: нетекстовые блоки DeepSeek на рамке области → объекты с рамкой DeepSeek, достроенной
   по краске залитой вырезки; если с ней согласна рамка классики — берётся рамка классики.
   Иначе — надпись.
"""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box, intersection, union
from ocr_utils.page_layout.line_art.classes import ObjectClass
from ocr_utils.page_layout.line_art.expand import FIGURE_GROW_MM, FORMULA_GROW_MM, grow_to_components
from ocr_utils.page_layout.line_art.features import analyse_gray, params_for_dpi
from ocr_utils.page_layout.line_art.deepseek.rules import CROP_DPI, block_class, ink_blobs, on_region

# Согласие рамок DeepSeek и классики: пересечение не меньше этой доли меньшей рамки (как у детектора).
AGREE_SHARE = 0.5


# Горизонтальная линейка длиннее этого, мм, в рамку объекта не входит: подчёркивающая линейка
# рубрики-вензеля тянула рамку буквицы на всю ширину рубрики. Толщина — не больше RULE_MAX_MM.
RULE_MIN_MM = 25.0
RULE_MAX_MM = 1.2
# Кусок линейки при поиске, мм, и высота полосы сшитой наклонной линейки, мм.
RULE_PIECE_MM = 8.0
RULE_BAND_MM = 3.0


def strip_rules(binary: np.ndarray) -> np.ndarray:
    """Та же картинка без длинных тонких горизонтальных линеек (для рамок; DeepSeek видит всё).

    Линейка бывает слегка наклонной: прямой отрезок в 25 мм в неё целиком не укладывается, и от
    неё оставался кусок в 18 мм (1972/02 с.83). Поэтому кусок линейки ищется короче
    (``RULE_PIECE_MM``), соседние куски сшиваются, и линейкой считается сшитая полоса длиннее
    ``RULE_MIN_MM`` и не выше ``RULE_BAND_MM``.

    Args:
        binary: Вырезка, краска 0.

    Returns:
        Копия, где пиксели таких линеек стали бумагой.
    """
    mm = CROP_DPI / 25.4
    ink = (binary == 0).astype(np.uint8)
    pieces = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(round(RULE_PIECE_MM * mm)), 1))
    )
    # Толстые горизонтали (плашка, жирная черта) не трогаем — только тонкие линии.
    thick = cv2.morphologyEx(
        pieces, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(round(RULE_MAX_MM * mm)) + 1))
    )
    pieces = pieces & (1 - thick)
    # Сшивка кусков наклонной линейки: по горизонтали через разрыв до 1 мм, по вертикали — до ступеньки 3 px.
    joined = cv2.morphologyEx(pieces, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(round(mm)), 5)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    long_band = np.zeros(count, bool)
    long_band[1:] = (stats[1:, cv2.CC_STAT_WIDTH] >= RULE_MIN_MM * mm) & (
        stats[1:, cv2.CC_STAT_HEIGHT] <= RULE_BAND_MM * mm
    )
    rules = (pieces > 0) & long_band[labels]
    # Линейка слегка раздута, чтобы стереть и её размытые края.
    rules = cv2.dilate(rules.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    stripped = binary.copy()
    stripped[rules > 0] = 255
    return stripped


# После снятия ВСЕХ длинных тонких линий (и горизонталей, и вертикалей) в рамке объекта должно
# остаться не меньше этой краски, мм²: иначе объект — одна рамка или угол из линеек (1974/05 с.54,
# край страницы оглавления после стирания отточий DeepSeek звал ``image``). Буквица вензеля — от
# 20 мм², ветвь — сотни.
MIN_OBJECT_INK_MM2 = 6.0


def strip_all_rules(binary: np.ndarray) -> np.ndarray:
    """Без длинных тонких линий обоих направлений: горизонтали — :func:`strip_rules`, вертикали — поворотом."""
    once = strip_rules(binary)
    return np.ascontiguousarray(strip_rules(np.ascontiguousarray(once.T)).T)


def has_body(binary: np.ndarray, box: Box) -> bool:
    """Есть ли в рамке краска кроме длинных линеек (не меньше ``MIN_OBJECT_INK_MM2``)."""
    mm2 = (25.4 / CROP_DPI) ** 2
    return float(np.count_nonzero(strip_all_rules(binary)[box.slice] == 0)) * mm2 >= MIN_OBJECT_INK_MM2


def tight_box(binary: np.ndarray, box: Box) -> Box | None:
    """Плотная рамка краски ``binary`` внутри ``box`` (``None``, если краски нет)."""
    ys, xs = np.nonzero(binary[box.slice] == 0)
    if xs.size == 0:
        return None
    return Box(box.x0 + int(xs.min()), box.y0 + int(ys.min()), box.x0 + int(xs.max()) + 1, box.y0 + int(ys.max()) + 1)


def word_boxes_grown(words: list[dict], blobs) -> list[tuple[int, int, int, int]]:
    """Рамки слов, достроенные до краёв буквоподобных пятен, которые они задевают.

    Args:
        words: Слова DeepSeek (``x0 … y1`` в пикселях вырезки).
        blobs: Краска вырезки (``rules.Blobs``).

    Returns:
        Рамки ``(x0, y0, x1, y1)`` в пикселях вырезки.
    """
    grown = []
    for word in words:
        x0, y0, x1, y1 = word["x0"], word["y0"], word["x1"], word["y1"]
        for index, (sx0, sy0, sx1, sy1) in enumerate(blobs.boxes):
            # Крупнейшее и не буквоподобное пятно слову не принадлежит (буквица с линейкой вензеля).
            if index == blobs.largest or not blobs.letter_like[index]:
                continue
            if sx0 < x1 and sx1 > x0 and sy0 < y1 and sy1 > y0:
                x0, y0, x1, y1 = min(x0, sx0), min(y0, sy0), max(x1, sx1), max(y1, sy1)
        grown.append((x0, y0, x1, y1))
    return grown


def fill_words(
    gray: np.ndarray, inner: list[int], words: list[dict]
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Бинаризованная вырезка с залитыми белым словами.

    Args:
        gray: Серая вырезка (с полем).
        inner: Рамка области в вырезке ``[x0, y0, x1, y1]``.
        words: Слова DeepSeek (промпт ``ocr``).

    Returns:
        Пара (картинка: краска 0, бумага 255, поле вне рамки области — бумага; залитые рамки слов).
    """
    blobs = ink_blobs(gray, inner, CROP_DPI)
    x0, y0, x1, y1 = inner
    # Порог Оцу по самой области: поле вокруг (другой фон) порог не сдвигает.
    _, local = cv2.threshold(gray[y0:y1, x0:x1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, whole = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = whole.copy()
    binary[y0:y1, x0:x1] = local
    # Поле вокруг области — бумага: иначе обрывки соседних слов в поле (не обведённые DeepSeek,
    # срезанные краем рамки) модель снова читает как заголовок (1970/04 с.14, «лет» над ветвью).
    outside = np.ones(binary.shape, bool)
    outside[y0:y1, x0:x1] = False
    binary[outside] = 255
    boxes = word_boxes_grown(words, blobs)
    for bx0, by0, bx1, by1 in boxes:
        binary[max(0, by0) : by1, max(0, bx0) : bx1] = 255
    return despeckle(binary), boxes


# Пятно краски, обе стороны которого меньше этого, мм, после заливки слов — крапина: точка
# отточия, недозалитый обрывок буквы. Россыпь отточий DeepSeek на втором проходе звал ``image``
# (1974/05 с.54: оглавление с номерами страниц). Точка отточия пака ~0.7 мм, штрих буквицы и
# стрелка схемы — от 3 мм.
SPECK_MAX_MM = 2.0
# Тонкий короткий обрывок: толщина не больше SLIVER_MAX_MM и длина меньше SLIVER_MAX_LEN_MM.
SLIVER_MAX_MM = 0.3
SLIVER_MAX_LEN_MM = 5.0


def despeckle(binary: np.ndarray) -> np.ndarray:
    """Стереть мелкие отдельные пятна краски (обе стороны меньше ``SPECK_MAX_MM``).

    Args:
        binary: Вырезка, краска 0.

    Returns:
        Копия без крапин.
    """
    limit = SPECK_MAX_MM * CROP_DPI / 25.4
    ink = (binary == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    small = np.zeros(count, bool)
    widths, heights = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
    thin = np.minimum(widths, heights) <= SLIVER_MAX_MM * CROP_DPI / 25.4
    short = np.maximum(widths, heights) < SLIVER_MAX_LEN_MM * CROP_DPI / 25.4
    # Крапина (обе стороны малы) или тонкий короткий обрывок (край снятой линейки, 1972/02 с.83).
    small[1:] = ((widths < limit) & (heights < limit)) | (thin & short)
    cleaned = binary.copy()
    cleaned[small[labels]] = 255
    return cleaned


def classic_boxes(binary: np.ndarray, inner: list[int]) -> list[tuple[int, int, int, int]]:
    """Классический детектор line art по залитой вырезке (только рамки внутри области).

    Args:
        binary: Залитая вырезка (краска 0).
        inner: Рамка области в вырезке.

    Returns:
        Рамки находок ``analyse_gray`` при 300 dpi, лежащие на области хотя бы наполовину.
    """
    findings = analyse_gray(strip_rules(binary), params_for_dpi(CROP_DPI))
    keep = []
    for box in findings.boxes:
        as_dict = {"x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]}
        if on_region(as_dict, inner):
            keep.append(tuple(int(v) for v in box))
    return keep


def verdict_pass2(
    binary: np.ndarray, inner, blocks: list[dict], classic: list[tuple[int, int, int, int]]
) -> list[dict]:
    """Итог второго прохода: объекты (класс, рамка, чья) или надпись.

    Args:
        binary: Залитая вырезка.
        inner: Рамка области в вырезке.
        blocks: Блоки DeepSeek ``markdown`` по залитой вырезке.
        classic: Рамки классики.

    Returns:
        Объекты ``[{"class", "box", "box_source"}]`` (рамки — в пикселях вырезки); пусто — объекта нет.
    """
    # Без линеек и без крапин: снятая линейка оставляет по краям обрывки, которые иначе растягивали
    # плотную рамку буквицы на всю ширину подчёркивающей линейки (1972/02 с.83).
    stripped = despeckle(strip_rules(binary))
    ink = stripped == 0
    objects = []
    for block in blocks:
        cls = block_class(block)
        # Формулы на втором проходе не принимаются: на залитой картинке DeepSeek выдумывает LaTeX
        # (буквица «С» → ``\( x^{2} \)``, пустая рамка → ``\boxed{}``: 90 ложных на паке-1).
        if cls in (None, ObjectClass.FORMULA) or not on_region(block, inner):
            continue
        box = Box(block["x0"], block["y0"], block["x1"], block["y1"]).clipped(binary.shape[1], binary.shape[0])
        limit = FORMULA_GROW_MM if cls is ObjectClass.FORMULA else FIGURE_GROW_MM
        grown = grow_to_components(box, ink, CROP_DPI, limit).box
        # Все согласные с блоком рамки классики входят в объект вместе с ним: одна коробка схемы,
        # целиком лежащая в большом блоке DeepSeek, тоже «согласна», и раньше объектом становилась
        # только она (1967/10 с.41 — вся блок-схема свелась к верхней коробке).
        agreeing = [
            Box(*c)
            for c in classic
            if (common := intersection(grown, Box(*c))) is not None
            and common.area >= AGREE_SHARE * max(1, min(grown.area, Box(*c).area))
        ]
        source = "DeepSeek + классика" if agreeing else "DeepSeek"
        extent = union([grown] + agreeing)
        # Рамка — плотно по краске без длинных линеек (у вензеля — только буквица с чертами).
        tight = tight_box(stripped, extent)
        if tight is None or not has_body(binary, tight):
            continue
        objects.append({"class": cls.value, "box": list(tight.as_tuple()), "box_source": source})
    return objects


__all__ = [
    "classic_boxes",
    "despeckle",
    "fill_words",
    "has_body",
    "strip_rules",
    "tight_box",
    "verdict_pass2",
    "word_boxes_grown",
]
