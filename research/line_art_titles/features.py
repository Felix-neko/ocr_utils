"""Признаки области line art по вырезке и словам tesseract: сколько краски — буквы, сколько — рисунок.

Главный признак — ``text_ink_share``: доля краски области внутри рамок уверенных слов. У
заголовка, набранного акцидентным шрифтом, она высокая, у схемы с подписями — низкая (подписи
мелкие, основная краска — линии), у орнамента без текста — ноль. Остальные признаки
объясняют промахи главного: связность краски (у рисунка есть одно огромное пятно, у надписи —
россыпь букв), доля длинных прямых штрихов (линейки, рамки, оси графиков) и мнение surya о
том, текст ли это.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.page_layout.surya.blocks import TEXT_LABELS

# Слово засчитывается «уверенным», если tesseract уверен в нём не меньше этого (0–100).
GOOD_CONF = 60.0

# Буква — кириллица или латиница; у настоящего слова букв не меньше двух и они не меньше
# ``MIN_LETTER_SHARE`` знаков (иначе это «|—|», прочитанный из линейки, или число оси графика).
LETTER = re.compile(r"[А-Яа-яЁёA-Za-z]")
MIN_LETTERS = 2
MIN_LETTER_SHARE = 0.6

# Блоки surya-заголовков.
HEADER_LABELS = ("SectionHeader", "Title", "PageHeader")

# Длинный прямой штрих — от стольких мм; на такой длине буква любого кегля пака ещё не бывает.
LONG_STROKE_MM = 12.0

# Буквоподобное пятно: высота в этих пределах, мм, и отношение сторон не дальше этого от 1.
LETTER_HEIGHT_MM = (1.5, 30.0)
LETTER_MAX_ASPECT = 4.0

# Поле вокруг рамки слова при подсчёте краски под словами, px вырезки: хвосты букв за рамку.
WORD_PAD_PX = 2


def is_letter_word(word: dict) -> bool:
    """Слово из букв при любой уверенности (буквы — как у :func:`is_good_word`).

    Args:
        word: Слово tesseract словарём.

    Returns:
        ``True``, если в слове не меньше ``MIN_LETTERS`` букв и они не меньше ``MIN_LETTER_SHARE`` знаков.
    """
    text = word["text"]
    letters = len(LETTER.findall(text))
    return letters >= MIN_LETTERS and letters >= MIN_LETTER_SHARE * len(text)


def _share_under(ink: np.ndarray, words: list[dict], total: int) -> float:
    """Доля краски ``ink`` под рамками ``words`` (с полем ``WORD_PAD_PX``) от ``total``."""
    covered = np.zeros_like(ink)
    for w in words:
        covered[
            max(0, w["y0"] - WORD_PAD_PX) : w["y1"] + WORD_PAD_PX, max(0, w["x0"] - WORD_PAD_PX) : w["x1"] + WORD_PAD_PX
        ] = 1
    return float((ink * covered).sum()) / total


def is_good_word(word: dict) -> bool:
    """Уверенное слово из букв (см. ``GOOD_CONF``, ``MIN_LETTERS``, ``MIN_LETTER_SHARE``).

    Args:
        word: Слово tesseract словарём (поля :class:`research.line_art_titles.ocr.Word`).

    Returns:
        ``True``, если слово засчитывается как настоящий текст.
    """
    text = word["text"]
    letters = len(LETTER.findall(text))
    return word["conf"] >= GOOD_CONF and letters >= MIN_LETTERS and letters >= MIN_LETTER_SHARE * len(text)


def ink_of(gray: np.ndarray) -> np.ndarray:
    """Маска краски вырезки по Оцу: 1 — краска, 0 — бумага.

    Args:
        gray: Серая вырезка.

    Returns:
        Маска ``uint8`` того же размера.
    """
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _lines_of(words: list[dict]) -> int:
    """Число строк среди слов: центры по высоте ближе половины медианной высоты слова — одна строка."""
    if not words:
        return 0
    centers = sorted((w["y0"] + w["y1"]) / 2 for w in words)
    step = 0.5 * float(np.median([w["y1"] - w["y0"] for w in words]))
    lines = 1
    for previous, current in zip(centers, centers[1:]):
        if current - previous > step:
            lines += 1
    return lines


@dataclass(frozen=True)
class Blobs:
    """Краска области и её связные пятна — то, по чему считаются признаки формы.

    Attributes:
        ink: Маска краски вырезки (1 — краска), обнулённая вне рамки области.
        labels: Метки связных пятен ``cv2.connectedComponentsWithStats`` (0 — бумага).
        areas: Площади пятен, px; индекс ``i`` — пятно с меткой ``i + 1``.
        letter_like: Буквоподобно ли пятно (высота ``LETTER_HEIGHT_MM``, стороны не дальше ``LETTER_MAX_ASPECT``).
        largest: Индекс самого крупного пятна или ``-1``, если пятен нет.
    """

    ink: np.ndarray
    labels: np.ndarray
    areas: np.ndarray
    letter_like: np.ndarray
    largest: int

    @property
    def total(self) -> int:
        """Площадь краски области, px (не меньше 1)."""
        return max(1, int(self.ink.sum()))


def ink_blobs(gray: np.ndarray, crop_inner: list[int], dpi: int) -> Blobs:
    """Краска внутри рамки области и её связные пятна с признаком буквоподобия.

    Args:
        gray: Серая вырезка (с полем).
        crop_inner: Рамка области в пикселях вырезки ``[x0, y0, x1, y1]``.
        dpi: Разрешение вырезки.

    Returns:
        :class:`Blobs` вырезки.
    """
    mm = dpi / 25.4
    x0, y0, x1, y1 = crop_inner
    ink = ink_of(gray)
    inner = np.zeros_like(ink)
    inner[y0:y1, x0:x1] = 1
    ink = ink * inner
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.zeros(0, np.int64)
    heights = stats[1:, cv2.CC_STAT_HEIGHT] if count > 1 else np.zeros(0, np.int64)
    widths = stats[1:, cv2.CC_STAT_WIDTH] if count > 1 else np.zeros(0, np.int64)
    letter_like = (
        (heights >= LETTER_HEIGHT_MM[0] * mm)
        & (heights <= LETTER_HEIGHT_MM[1] * mm)
        & (np.maximum(widths, heights) <= LETTER_MAX_ASPECT * np.maximum(1, np.minimum(widths, heights)))
    )
    largest = int(np.argmax(areas)) if areas.size else -1
    return Blobs(ink, labels, areas, letter_like, largest)


def weak_words(blobs: Blobs, words_by_psm: dict[str, list[dict]]) -> tuple[float, list[dict]]:
    """Слова из букв при любой уверенности из того режима tesseract, где под ними больше краски.

    Args:
        blobs: Краска области.
        words_by_psm: Слова tesseract по режимам.

    Returns:
        Пара (доля краски под этими словами — признак ``weak_text_ink_share``, сами слова).
    """
    best: tuple[float, list[dict]] = (0.0, [])
    for words in words_by_psm.values():
        letters = [w for w in words if is_letter_word(w)]
        share = _share_under(blobs.ink, letters, blobs.total)
        if share > best[0]:
            best = (share, letters)
    return best


def region_features(row: dict, words_by_psm: dict[str, list[dict]], gray: np.ndarray, dpi: int) -> dict:
    """Признаки одной области.

    Args:
        row: Строка ``regions.jsonl`` (см. ``detect.detect_page``).
        words_by_psm: Слова tesseract по режимам (``ocr.read_region``).
        gray: Серая вырезка области (с полем).
        dpi: Разрешение вырезки.

    Returns:
        Плоский словарь признаков; ``best_psm`` — режим, у которого доля краски под уверенными
        словами больше, и все текстовые признаки — по нему.
    """
    mm = dpi / 25.4
    x0, y0, x1, y1 = row["crop_inner"]
    blobs = ink_blobs(gray, row["crop_inner"], dpi)
    ink, total = blobs.ink, blobs.total

    # Доля краски под уверенными словами — для каждого режима, берётся лучший.
    best: tuple[float, str, list[dict]] = (-1.0, "", [])
    for psm, words in words_by_psm.items():
        good = [w for w in words if is_good_word(w)]
        share = _share_under(ink, good, total)
        if share > best[0]:
            best = (share, psm, good)
    text_share, best_psm, good = best
    all_words = words_by_psm.get(best_psm, [])
    # То же без порога уверенности: срезанное краем полосы или вычурное слово tesseract читает
    # с уверенностью 20–40, но буквы в нём всё-таки видит.
    weak_share, _ = weak_words(blobs, words_by_psm)

    # Связность: доля краски в самом большом пятне и в буквоподобных пятнах.
    areas, letter_like = blobs.areas, blobs.letter_like

    # Длинные прямые штрихи: открытие горизонтальным и вертикальным отрезком ``LONG_STROKE_MM``.
    length = max(3, int(round(LONG_STROKE_MM * mm)))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
    long_share = float(np.count_nonzero(horizontal | vertical)) / total

    surya = row.get("surya", [])
    width_mm = (x1 - x0) / mm
    height_mm = (y1 - y0) / mm
    word_heights = [(w["y1"] - w["y0"]) / mm for w in good]
    return {
        "id": row["id"],
        "page": row["page"],
        "kind": row["info"].get("kind", ""),
        "sources": ",".join(row["info"].get("sources", [])),
        "width_mm": round(width_mm, 1),
        "height_mm": round(height_mm, 1),
        "aspect": round(width_mm / max(0.1, height_mm), 2),
        "ink_frac": round(total / max(1, (x1 - x0) * (y1 - y0)), 4),
        "best_psm": best_psm,
        "text_ink_share": round(text_share, 4),
        "weak_text_ink_share": round(weak_share, 4),
        "good_words": len(good),
        "all_words": len(all_words),
        "good_conf_mean": round(float(np.mean([w["conf"] for w in good])), 1) if good else 0.0,
        "good_lines": _lines_of(good),
        "word_height_mm": round(float(np.median(word_heights)), 2) if word_heights else 0.0,
        "largest_cc_share": round(float(areas.max()) / total, 4) if areas.size else 0.0,
        "letter_cc_share": round(float(areas[letter_like].sum()) / total, 4) if areas.size else 0.0,
        # То же без самого крупного пятна: у рубрики-вензеля буквица сливается с подчёркивающей
        # линейкой в одно большое пятно, а всё остальное — буквы.
        "letter_cc_share_rest": (
            round(float(areas[letter_like & (areas < areas.max())].sum()) / max(1, total - int(areas.max())), 4)
            if areas.size > 1
            else 0.0
        ),
        "cc_count": int(areas.size),
        "long_stroke_share": round(long_share, 4),
        "surya_text_share": round(min(1.0, sum(b["share"] for b in surya if b["label"] in TEXT_LABELS)), 3),
        "surya_header_share": round(min(1.0, sum(b["share"] for b in surya if b["label"] in HEADER_LABELS)), 3),
        "surya_figure_share": round(
            min(1.0, sum(b["share"] for b in surya if b["label"] in ("Figure", "Picture", "Form"))), 3
        ),
        "text": " ".join(w["text"] for w in good)[:200],
    }


# Правило «область — надпись, а не рисунок» (калибровка по 300 размеченным глазами областям
# пака-1, ``reports/line_art_titles.md``): буквоподобная краска без самого крупного пятна,
# высота области и краска под словами из букв при любой уверенности tesseract.
TITLE_MIN_LETTER_SHARE = 0.9
TITLE_MAX_HEIGHT_MM = 40.0
TITLE_MIN_WEAK_TEXT_SHARE = 0.3

# У области из трёх пятен и меньше «крупнейшее пятно» — это и есть буква: берётся доля по всем пятнам.
FEW_BLOBS = 3


def is_title_like(features: dict) -> bool:
    """Надпись ли область (заголовок, рубрика-вензель, обрывок жирного набора) по признакам :func:`region_features`.

    Args:
        features: Строка ``features.csv`` (числа могут прийти строками).

    Returns:
        ``True``, если по правилу это надпись и на ней можно запускать детектор строк
        (у рубрики-вензеля — после снятия буквицы и линеек, см. отчёт).
    """
    blobs = int(features["cc_count"])
    letters = float(features["letter_cc_share"] if blobs <= FEW_BLOBS else features["letter_cc_share_rest"])
    return (
        letters >= TITLE_MIN_LETTER_SHARE
        and float(features["height_mm"]) <= TITLE_MAX_HEIGHT_MM
        and float(features["weak_text_ink_share"]) >= TITLE_MIN_WEAK_TEXT_SHARE
    )


# --- DeepSeek-OCR-2 -----------------------------------------------------------------------------

# Метки блоков DeepSeek-OCR-2 (промпт ``markdown``), которые не текст: рисунок, таблица, формула.
DEEPSEEK_IMAGE_LABELS = ("image",)
DEEPSEEK_TABLE_LABELS = ("table",)
DEEPSEEK_FORMULA_LABELS = ("equation",)
DEEPSEEK_NON_TEXT_LABELS = DEEPSEEK_IMAGE_LABELS + DEEPSEEK_TABLE_LABELS + DEEPSEEK_FORMULA_LABELS


# Блок DeepSeek относится к области, если хотя бы такая доля его площади лежит на рамке области:
# блоки в поле вокруг неё (соседняя колонка текста, чужая таблица) в вердикт не идут.
DEEPSEEK_BLOCK_INSIDE_SHARE = 0.5


def inside_share(block: dict, inner: list[int]) -> float:
    """Доля площади рамки ``block`` (``x0, y0, x1, y1``), лежащая внутри ``inner`` ``[x0, y0, x1, y1]``."""
    width = min(block["x1"], inner[2]) - max(block["x0"], inner[0])
    height = min(block["y1"], inner[3]) - max(block["y0"], inner[1])
    area = max(1, (block["x1"] - block["x0"]) * (block["y1"] - block["y0"]))
    return max(0, width) * max(0, height) / area


from ocr_utils.page_layout.line_art.deepseek.rules import on_region  # noqa: E402 — перенесено в пакет


def deepseek_features(blobs: Blobs, markdown: list[dict], words: list[dict], inner: list[int] | None = None) -> dict:
    """Признаки области по выводу DeepSeek-OCR-2 в режиме grounding.

    Args:
        blobs: Краска области (:func:`ink_blobs`).
        markdown: Элементы промпта ``markdown`` — блоки вёрстки: ``label`` — тип блока
            (``sub_title``, ``text``, ``image``, ``table`` …), ``text`` — их текст по-русски.
        words: Элементы промпта ``ocr`` — рамки слов и строк; ``label`` — прочитанный текст
            (кириллицу модель в этом режиме пишет латинскими двойниками, поэтому текст годится
            только на «буквы ли это», а не на чтение).
        inner: Рамка области в пикселях вырезки ``[x0, y0, x1, y1]``; нетекстовый блок засчитывается
            в ``ds_has_non_text_block``, только если лежит на ней хотя бы на ``DEEPSEEK_BLOCK_INSIDE_SHARE``
            своей площади. ``None`` — все блоки вырезки.

    Returns:
        Словарь признаков ``ds_*``: доли краски под всеми словами, под словами из букв, под
        текстовыми блоками, блоками-рисунками и таблицами; число слов; типы блоков и их текст.
    """
    word_boxes = [{**w, "text": w["label"]} for w in words]
    letter_words = [w for w in word_boxes if is_letter_word(w)]
    image_blocks = [b for b in markdown if b["label"] in DEEPSEEK_IMAGE_LABELS]
    table_blocks = [b for b in markdown if b["label"] in DEEPSEEK_TABLE_LABELS]
    formula_blocks = [b for b in markdown if b["label"] in DEEPSEEK_FORMULA_LABELS]
    text_blocks = [b for b in markdown if b["label"] not in DEEPSEEK_NON_TEXT_LABELS]
    total = blobs.total
    return {
        "ds_word_ink_share": round(_share_under(blobs.ink, word_boxes, total), 4),
        "ds_letter_word_ink_share": round(_share_under(blobs.ink, letter_words, total), 4),
        "ds_words": len(word_boxes),
        "ds_letter_words": len(letter_words),
        "ds_text_block_ink_share": round(_share_under(blobs.ink, text_blocks, total), 4),
        "ds_image_block_ink_share": round(_share_under(blobs.ink, image_blocks, total), 4),
        "ds_table_block_ink_share": round(_share_under(blobs.ink, table_blocks, total), 4),
        "ds_formula_block_ink_share": round(_share_under(blobs.ink, formula_blocks, total), 4),
        # Нашла ли модель в области хоть один нетекстовый блок (рисунок, таблицу, формулу).
        "ds_has_non_text_block": int(
            any(b["label"] in DEEPSEEK_NON_TEXT_LABELS and (inner is None or on_region(b, inner)) for b in markdown)
        ),
        "ds_block_labels": ",".join(sorted({b["label"] for b in markdown})),
        "ds_text": " ".join(" ".join(b["text"].split()) for b in text_blocks)[:200],
    }


# Вариант «слова DeepSeek»: третье условие правила tesseract на краске под всеми словами, которые
# модель нашла в режиме ``ocr`` (без порога уверенности — её у модели нет). Порог 0.2 — лучший по
# распределению на размеченных (reports/line_art_titles.md); правило проигрывает блочному.
TITLE_MIN_DS_WORD_SHARE = 0.2


def title_letters(features: dict) -> float:
    """Доля буквоподобной краски, по которой судит правило: по всем пятнам, если их ≤ 3, иначе без крупнейшего."""
    few = int(features["cc_count"]) <= FEW_BLOBS
    return float(features["letter_cc_share"] if few else features["letter_cc_share_rest"])


def is_title_like_deepseek_words(features: dict) -> bool:
    """Правило «надпись» со словами DeepSeek-OCR-2 вместо tesseract (буквы и высота — как у :func:`is_title_like`).

    Args:
        features: Строка ``features_deepseek.csv``.

    Returns:
        ``True``, если по правилу это надпись.
    """
    return (
        title_letters(features) >= TITLE_MIN_LETTER_SHARE
        and float(features["height_mm"]) <= TITLE_MAX_HEIGHT_MM
        and float(features["ds_word_ink_share"]) >= TITLE_MIN_DS_WORD_SHARE
    )


def is_title_like_deepseek(features: dict) -> bool:
    """Правило «надпись» по блокам DeepSeek-OCR-2 (промпт ``markdown``): только текстовые блоки и невысокая область.

    Условия: модель не нашла в области ни рисунка, ни таблицы, ни формулы
    (``DEEPSEEK_NON_TEXT_LABELS``); нашла хотя бы один текстовый блок над краской (пустую рамку
    она оставляет без блоков вовсе); высота области ≤ ``TITLE_MAX_HEIGHT_MM``. Пиксельные буквы не
    нужны: схему и рисунок модель сама называет ``image``.

    Args:
        features: Строка ``features_deepseek.csv``.

    Returns:
        ``True``, если по правилу это надпись.
    """
    return (
        int(features["ds_has_non_text_block"]) == 0
        and float(features["ds_text_block_ink_share"]) > 0
        and float(features["height_mm"]) <= TITLE_MAX_HEIGHT_MM
    )
