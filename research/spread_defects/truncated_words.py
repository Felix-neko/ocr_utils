"""Слова-обрубки у сгиба: буквы, ушедшие под переплёт, по OCR (tesseract) и словарю (pymorphy3), локально.

Идея. Если начало строки правой полосы (или конец строки левой) ушло под корешок, крайнее у сгиба слово читается
обрубком — «величить» вместо «увеличить», — которого нет в словаре. Законные обрубки — части слова при переносе:
строка левой полосы кончается на «-», и её хвост — первое слово следующей строки. Поэтому крайнее слово
проверяется вместе с переносом: «дополни-» + «тельно» → «дополнительно». Ошибки OCR дают незнакомые слова и
без корешка; их уровень меряется на противоположном (внешнем) краю тех же строк той же полосы — там корешка
нет, и разница «незнакомых у сгиба» и «незнакомых у внешнего края» — признак букв под переплётом.

Литература: готового детектора не нашлось (поиск 2026-10-03); на контроле оцифровки «text lost in gutter» ловят
глазами; ближе всего — рассуждение об обрезанных строках (Cutter 2015) и словарная проверка OCR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from ocr_utils.rotated_text.tables.ocr import tesseract_tsv

# Заступ строки за сгиб (буквы, частично лежащие за линией сгиба) и поле по высоте строки, в высотах буквы.
OVER_FOLD, LINE_PAD = 1.0, 0.35
# Полоса за дальней буквой строки (знак переноса, точка), в высотах буквы.
FAR_TAIL = 1.2
# Соседние строки — одна за другой, если их центры ближе стольких медианных шагов строк (иначе между ними
# пропущенная строка и перенос не склеивается).
NEXT_LINE = 1.5
# Промежуток между строками в склеенной картинке для tesseract, px.
GAP = 24
# Слово годится для проверки: не короче стольких букв (кириллица), без цифр.
MIN_LETTERS = 2

_CYR = re.compile(r"[а-яё]+(?:-[а-яё]+)*", re.IGNORECASE)


@lru_cache(maxsize=1)
def _morph():
    """Морфоанализатор pymorphy3 (грузится один раз на процесс)."""
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


def is_known(word: str) -> bool:
    """Есть ли слово в словаре (с учётом ё/е и регистра).

    Args:
        word: Слово кириллицей, возможно через дефис.

    Returns:
        True, если словарь знает слово целиком или все части через дефис.
    """
    morph = _morph()
    parts = word.lower().replace("ё", "е").split("-")
    return all(part and morph.word_is_known(part) for part in parts)


def line_strip(gray: np.ndarray, g, line, fold: np.ndarray, side: str) -> np.ndarray:
    """Картинка одной строки у сгиба: только буквы своей строки, всё прочее закрашено цветом бумаги.

    У сгиба строка круто изгибается, и прямоугольная рамка строки захватывает буквы соседних строк — tesseract
    тогда склеивает строки. Поэтому в вырезке оставлены лишь рамки букв строки (с полем) и полоса от концевой
    буквы до сгиба и за него на высоте концевой буквы — там ищутся буквы, уходящие под переплёт.

    Args:
        gray: Полутоновый кадр uint8.
        g: Буквы кадра (``lines.Glyphs``).
        line: Строка (``lines.Line``).
        fold: Прямая сгиба.
        side: ``L``/``R``.

    Returns:
        Вырезка строки (uint8).
    """
    idx = line.glyphs
    hgt = float(np.median(g.h[idx]))
    pad = int(LINE_PAD * hgt)
    fold_x = float(np.polyval(fold, line.cy))
    end = idx[0]
    ex0, ey0, ex1, ey1 = g.x0[end], g.y0[end], g.x0[end] + g.w[end], g.y0[end] + g.h[end]
    if side == "R":
        tail = (int(fold_x - OVER_FOLD * hgt), ey0 - pad, ex0, ey1 + pad)
    else:
        tail = (ex1, ey0 - pad, int(fold_x + OVER_FOLD * hgt), ey1 + pad)
    # Дальний конец строки: знак переноса и точки ниже порога высоты буквы, поэтому их рамок в строке нет —
    # добавляем полосу на ``FAR_TAIL`` высот за последней буквой на её высоте.
    last = idx[-1]
    lx0, ly0, lx1, ly1 = g.x0[last], g.y0[last], g.x0[last] + g.w[last], g.y0[last] + g.h[last]
    far = (
        (lx1, ly0 - pad, int(lx1 + FAR_TAIL * hgt), ly1 + pad)
        if side == "R"
        else (int(lx0 - FAR_TAIL * hgt), ly0 - pad, lx0, ly1 + pad)
    )
    boxes = [(g.x0[i] - pad, g.y0[i] - pad, g.x0[i] + g.w[i] + pad, g.y0[i] + g.h[i] + pad) for i in idx]
    boxes += [tail, far]
    bx = np.array(boxes, float)
    x0, y0 = int(max(0, bx[:, 0].min())), int(max(0, bx[:, 1].min()))
    x1, y1 = int(min(gray.shape[1], bx[:, 2].max())), int(min(gray.shape[0], bx[:, 3].max()))
    crop = gray[y0:y1, x0:x1]
    keep = np.zeros(crop.shape, bool)
    for a, b, c, d in bx.astype(int):
        keep[max(0, b - y0) : max(0, d - y0), max(0, a - x0) : max(0, c - x0)] = True
    paper = np.uint8(np.percentile(crop, 90)) if crop.size else 255
    return np.where(keep, crop, paper).astype(np.uint8)


def read_lines(strips: list[np.ndarray]) -> list[list[tuple[str, float]]]:
    """Чтение строк одним вызовом tesseract: строки склеены столбиком через промежутки.

    Args:
        strips: Вырезки строк по порядку.

    Returns:
        Для каждой вырезки — слова (текст, уверенность) слева направо; пусто, если строку не прочитали.
    """
    if not strips:
        return []
    width = max(s.shape[1] for s in strips)
    paper = int(np.median([np.percentile(s, 90) for s in strips]))
    rows, spans, y = [], [], 0
    for strip in strips:
        canvas = np.full((strip.shape[0] + GAP, width), paper, np.uint8)
        canvas[: strip.shape[0], : strip.shape[1]] = strip
        rows.append(canvas)
        spans.append((y, y + strip.shape[0]))
        y += canvas.shape[0]
    image = np.vstack(rows)
    # Буква ~25 px при 450 dpi — с запасом для tesseract; уменьшение вдвое по площади ускоряет чтение.
    scale = 0.7
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out: list[list[tuple[float, str, float]]] = [[] for _ in strips]
    for row in tesseract_tsv(small, lang="rus", psm=6):
        cols = row.split("\t")
        if len(cols) < 12 or not cols[11].strip():
            continue
        left, top, _, h = (float(c) / scale for c in cols[6:10])
        # ``tesseract_tsv`` не добавляет полей — координаты прямо в картинке-столбике.
        center = top + h / 2
        for n, (a, b) in enumerate(spans):
            if a - GAP / 2 <= center < b + GAP / 2:
                out[n].append((left, cols[11].strip(), float(cols[10])))
                break
    return [[(text, conf) for _, text, conf in sorted(words)] for words in out]


def clean(token: str) -> str | None:
    """Кириллическое слово из токена OCR без пунктуации; None — если это не слово (цифры, латиница, короткое).

    Args:
        token: Токен tesseract.

    Returns:
        Слово (с сохранением дефиса на конце, если он был) или None.
    """
    if re.search(r"\d", token):
        return None
    hyphen = token.rstrip("»).,;:!?\"'").endswith(("-", "—", "¬"))
    found = _CYR.findall(token)
    if not found:
        return None
    word = max(found, key=len)
    if len(word.replace("-", "")) < MIN_LETTERS:
        return None
    return word + ("-" if hyphen and not word.endswith("-") else "")


@dataclass(frozen=True)
class EdgeCheck:
    """Проверка крайних слов строк одной полосы.

    Attributes:
        fold_checked, fold_unknown: Сколько крайних у сгиба слов проверено и сколько из них незнакомы словарю.
        outer_checked, outer_unknown: То же для дальнего края тех же строк (контроль ошибок OCR).
        examples: До 8 незнакомых слов у сгиба (для листов отсмотра).
    """

    fold_checked: int
    fold_unknown: int
    outer_checked: int
    outer_unknown: int
    examples: tuple[str, ...]

    @property
    def excess(self) -> float:
        """Доля незнакомых у сгиба минус доля у дальнего края (NaN, если проверено мало)."""
        if self.fold_checked < 8 or self.outer_checked < 8:
            return float("nan")
        return self.fold_unknown / self.fold_checked - self.outer_unknown / self.outer_checked


def check_lines(texts: list[list[str]], side: str, adjacent: list[bool] | None = None) -> EdgeCheck:
    """Крайние слова строк у сгиба и у дальнего края против словаря, с учётом переносов.

    Строки идут сверху вниз в пределах одной колонки (собраны ``lines.side_lines``). Строка левой полосы,
    кончающаяся переносом, проверяется склейкой с первым словом следующей строки; начало строки правой полосы —
    склейкой с концом предыдущей, если та кончается переносом.

    Args:
        texts: Слова каждой строки слева направо (сверху вниз).
        side: ``L``/``R``.
        adjacent: ``adjacent[n]`` — строка ``n + 1`` идёт сразу за строкой ``n`` (переносы склеиваются только
            между такими); None — все соседние строки смежны.

    Returns:
        ``EdgeCheck``.
    """
    adjacent = adjacent if adjacent is not None else [True] * len(texts)
    words = [[w for w in (clean(t) for t in row) if w] for row in texts]
    fold_checked = fold_unknown = outer_checked = outer_unknown = 0
    examples = []
    for n, row in enumerate(words):
        if len(row) < 2:
            continue
        prev_end = words[n - 1][-1] if n > 0 and words[n - 1] and adjacent[n - 1] else None
        next_start = words[n + 1][0] if n + 1 < len(words) and words[n + 1] and adjacent[n] else None
        start = (prev_end[:-1] + row[0]) if prev_end and prev_end.endswith("-") else row[0]
        if row[-1].endswith("-"):
            end = row[-1][:-1] + next_start if next_start else None
        else:
            end = row[-1]
        fold_word, outer_word = (start, end) if side == "R" else (end, start)
        if fold_word:
            fold_checked += 1
            if not is_known(fold_word.strip("-")):
                fold_unknown += 1
                if len(examples) < 8:
                    examples.append(fold_word)
        if outer_word:
            outer_checked += 1
            outer_unknown += not is_known(outer_word.strip("-"))
    return EdgeCheck(fold_checked, fold_unknown, outer_checked, outer_unknown, tuple(examples))


def side_check(gray: np.ndarray, g, fold: np.ndarray, side: str) -> EdgeCheck:
    """Проверка одной полосы: строки у сгиба (``lines.side_lines``), чтение и словарь.

    Args:
        gray: Полутоновый кадр uint8.
        g: Буквы кадра.
        fold: Прямая сгиба.
        side: ``L``/``R``.

    Returns:
        ``EdgeCheck``; строки, не доходящие до края колонки, в проверку у сгиба не идут, но остаются
        соседями для переносов.
    """
    from research.spread_defects.lines import side_lines

    lines = sorted(side_lines(g, fold, side), key=lambda line: line.cy)
    texts = [[t for t, _ in row] for row in read_lines([line_strip(gray, g, l, fold, side) for l in lines])]
    # Строки не у края колонки (концевые строки абзаца) — у сгиба не проверяем: обнуляем их слово у сгиба.
    marked = [row if line.at_edge else ([] if side == "L" else row) for row, line in zip(texts, lines)]
    return check_lines(marked, side, adjacency([line.cy for line in lines]))


def adjacency(cys: list[float]) -> list[bool]:
    """Смежность соседних строк по шагу: ``out[n]`` — строка ``n + 1`` идёт сразу за ``n``.

    Args:
        cys: Высоты строк сверху вниз.

    Returns:
        Флаги длины ``len(cys) - 1`` (пусто для одной строки).
    """
    steps = np.diff(cys)
    if len(steps) == 0:
        return []
    pitch = float(np.median(steps))
    return [bool(step < NEXT_LINE * pitch) for step in steps]


# --- Вариант на surya (детектор строк + распознаватель) по полосе целиком ----------------------------------
# Склейка строк из компонент на изогнутых строках у сгиба ненадёжна (строки сращиваются), а tesseract на
# вырезках читает плохо. Surya сам находит строки полосы и читает их заметно лучше; результат кэшируется на диск.

# Полоса для surya: от сгиба наружу, px полного кадра; заступ за сгиб.
SURYA_WIDTH, SURYA_OVER = 2600, 60
# Тело полосы по высоте (без колонтитулов и обреза), доли кадра.
TOP, BOTTOM = 0.06, 0.94
# Строка «у сгиба»: её край у сгиба ближе стольких px к линии сгиба (полный кадр).
SURYA_NEAR = 220


def side_crop(gray: np.ndarray, fold_x: float, side: str) -> tuple[np.ndarray, int, int]:
    """Вырезка полосы для surya.

    Args:
        gray: Полутоновый кадр uint8.
        fold_x: x сгиба посередине высоты.
        side: ``L``/``R``.

    Returns:
        (вырезка, x0, y0) — смещение вырезки в кадре.
    """
    height, width = gray.shape
    y0, y1 = int(TOP * height), int(BOTTOM * height)
    if side == "R":
        x0, x1 = max(0, int(fold_x - SURYA_OVER)), min(width, int(fold_x + SURYA_WIDTH))
    else:
        x0, x1 = max(0, int(fold_x - SURYA_WIDTH)), min(width, int(fold_x + SURYA_OVER))
    return gray[y0:y1, x0:x1], x0, y0


def surya_lines(result, x0: int, y0: int) -> list[dict]:
    """Строки ответа surya в координатах кадра.

    Args:
        result: ``OCRResult`` surya по вырезке.
        x0, y0: Смещение вырезки.

    Returns:
        Строки: ``{"bbox": [x0, y0, x1, y1], "text": str, "conf": float}``.
    """
    out = []
    for line in result.text_lines:
        a, b, c, d = map(float, line.bbox)
        out.append({"bbox": [a + x0, b + y0, c + x0, d + y0], "text": line.text, "conf": float(line.confidence or 0)})
    return out


def check_surya_side(lines: list[dict], fold_x: float, side: str) -> EdgeCheck:
    """Проверка крайних у сгиба слов по строкам surya одной полосы.

    Строки делятся на колонки по положению (строки одной колонки перекрываются по x), колонка у сгиба —
    та, чей край ближе к сгибу; внутри колонки строки идут сверху вниз, переносы склеиваются между смежными.
    Строка проверяется у сгиба, если её край у сгиба ближе ``SURYA_NEAR`` к сгибу.

    Args:
        lines: Строки surya (``surya_lines``).
        fold_x: x сгиба.
        side: ``L``/``R``.

    Returns:
        ``EdgeCheck`` (внешний край — дальний край строк той же колонки).
    """
    if not lines:
        return EdgeCheck(0, 0, 0, 0, ())
    near_edge = [abs((l["bbox"][0] if side == "R" else l["bbox"][2]) - fold_x) for l in lines]
    column = [l for l, d in zip(lines, near_edge) if d < SURYA_WIDTH / 2]
    if not column:
        return EdgeCheck(0, 0, 0, 0, ())
    edge_x = np.median([l["bbox"][0] if side == "R" else l["bbox"][2] for l in column])
    # Колонка у сгиба: строки, чей край у сгиба не дальше 400 px от медианного края колонки.
    column = [l for l in column if abs((l["bbox"][0] if side == "R" else l["bbox"][2]) - edge_x) < 400]
    column.sort(key=lambda l: (l["bbox"][1] + l["bbox"][3]) / 2)
    cys = [(l["bbox"][1] + l["bbox"][3]) / 2 for l in column]
    texts = []
    for l in column:
        fold_side = l["bbox"][0] if side == "R" else l["bbox"][2]
        words = l["text"].split()
        # Строку, не доходящую до сгиба (абзацная), у сгиба не проверяем: для левой полосы — пусто, для правой
        # (отступ абзаца) — тоже пусто, переносы соседей при этом не теряются только у правой.
        texts.append(words if abs(fold_side - fold_x) < SURYA_NEAR else (words if side == "R" else []))
    near_flags = [abs((l["bbox"][0] if side == "R" else l["bbox"][2]) - fold_x) < SURYA_NEAR for l in column]
    check = check_lines(texts, side, adjacency(cys))
    # Отступ абзаца у правой полосы не обрубок — но его слово попало в проверку; пересчёт только по строкам у сгиба.
    if side == "R":
        kept = [t if f else [] for t, f in zip(texts, near_flags)]
        partial = check_lines(kept, side, adjacency(cys))
        return EdgeCheck(partial.fold_checked, partial.fold_unknown, check.outer_checked, check.outer_unknown,
                         partial.examples)  # fmt: skip
    return check
