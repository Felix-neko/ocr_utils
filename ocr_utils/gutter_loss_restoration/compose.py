"""Вклейка восстановленных букв: место, бумага, литеры.

ПОРЯДОК. (1) У корешка кадр РАЗДВИГАЕТСЯ: между полосами вставляется пустая полоска, и
обе страницы получают внутреннее поле, которого их лишил переплёт. (2) Последнее слово
строки стирается целиком — от начала слова, найденного распознавателем, до нового
сгиба. (3) На освободившемся месте синтезируется бумага. (4) Слово набирается заново
литерами-донорами с той же полосы и подгоняется по ширине.

ПОЧЕМУ КАДР РАЗДВИГАЕТСЯ, А НЕ ПОДЖИМАЕТСЯ СОСЕД. В «Плановом хозяйстве» 1926/08 текст
правой полосы начинается в 6–16 px от сгиба: свободного поля, куда можно было бы
подвинуть край страницы, там просто нет (в 1926/11 оно было, и приём работал). Значит,
место можно только добавить — выходной кадр шире исходного на ширину вставки.

ПОЧЕМУ СЛОВО ПЕРЕВЁРСТЫВАЕТСЯ ЦЕЛИКОМ, А НЕ ДОПИСЫВАЕТСЯ. Распознаватель на обрезанном
слове ошибается на символ: «развернут» читается как «разверну». Дописывать от такого
места — значит копить его ошибку. Начало слова же он даёт точно, и переверстка от
начала снимает вопрос, где именно проходит срез.
"""

from dataclasses import dataclass

import numpy as np

import cv2

from ocr_utils.gutter_loss_restoration.glyphs import CUT_UP, Glyph, local_baseline

# Ширина вставки с каждой стороны сгиба, в шагах строк.
PAD_PITCHES = 1.6

# Сколько пикселей вокруг сгиба выбрасывается вместе с его тёмным следом.
FOLD_TRIM = 5

# Полоса стирания относительно базовой линии, в долях шага строк. Фиксированные
# пиксели не годятся: кегль в паке разный, а недобрать сверху нельзя — от обрезанного
# слова остаются верхушки выносных, и они торчат над дорисованным.
BAND_UP_K, BAND_DOWN_K = 0.86, 0.36

# Межбуквенный пробел: обычный и минимальный, в долях ширины очка «о».
BEARING, BEARING_MIN = 0.16, 0.07

# Насколько узкой позволено стать литере при подгонке. Ниже — не вклеиваем вовсе:
# нечитаемая буква хуже отсутствующей.
MIN_SQUEEZE = 0.55


@dataclass
class Placement:
    """Одна вклейка.

    Attributes:
        side: Полоса, "L" или "R".
        word: Что набрать.
        x_start: Левый край набора в координатах ВЫХОДНОГО кадра.
        x_stop: Правая граница, дальше которой набор не идёт.
        baseline: Базовая линия строки в выходном кадре.
        squeeze: Во сколько раз сжаты литеры.
    """

    side: str
    word: str
    x_start: int
    x_stop: int
    baseline: int
    squeeze: float = 1.0


def widen(image: np.ndarray, fold_at: np.ndarray, pad: int) -> tuple[np.ndarray, np.ndarray]:
    """Раздвигает кадр у корешка, освобождая место обеим полосам.

    ПОСТРОЧНО, а не по одному столбцу: сгиб наклонён — на съёмке с рук он уезжает на
    два десятка пикселей по высоте кадра. Резать по среднему столбцу значит на дальних
    от середины строках срезать живой текст; замерено — у неисправленных строк пропадали
    последние буквы, и кадр становился хуже исходного.

    Args:
        image: Исходный кадр BGR.
        fold_at: Столбец сгиба для каждой строки кадра.
        pad: Полная ширина вставки в пикселях.

    Returns:
        Пара (новый кадр, столбец нового сгиба для каждой строки).
    """
    height, width = image.shape[:2]
    out = np.zeros((height, width + pad, 3), np.float32)
    left_paper = _row_paper(image[:, max(0, int(fold_at.min()) - 300) : max(1, int(fold_at.min()) - 40)])
    right_paper = _row_paper(
        image[:, min(width - 2, int(fold_at.max()) + 40) : min(width - 1, int(fold_at.max()) + 300)]
    )
    new_fold = fold_at + pad // 2
    for y in range(height):
        left_end = max(1, min(width - 1, int(fold_at[y]) - FOLD_TRIM))
        right_start = max(left_end + 1, min(width - 1, int(fold_at[y]) + FOLD_TRIM))
        out[y, :left_end] = image[y, :left_end]
        out[y, right_start + pad :] = image[y, right_start:]
        middle = int(new_fold[y])
        out[y, left_end:middle] = left_paper[y]
        out[y, middle : right_start + pad] = right_paper[y]
    return out, new_fold


def _row_paper(strip: np.ndarray) -> np.ndarray:
    """Уровень бумаги в каждой строке кадра: верхний квартиль яркости полосы."""
    level = np.percentile(strip, 75, axis=1)
    return cv2.GaussianBlur(level.astype(np.float32), (0, 0), 6)


def fit_word(word: str, library: dict[str, Glyph], available: float) -> tuple[float, float] | None:
    """Подбирает сжатие и межбуквенный пробел под доступную ширину.

    Args:
        word: Слово целиком.
        library: Библиотека литер.
        available: Сколько пикселей есть под набор.

    Returns:
        Пара (сжатие, пробел в пикселях) либо None, если слово не влезает читаемым.
    """
    if any(char not in library for char in word):
        return None
    widths = [library[char].ink_w for char in word]
    unit = float(np.median(widths))
    total = sum(widths)
    count = len(word)
    for bearing in (BEARING, BEARING_MIN):
        need = total + (count - 1) * bearing * unit
        if need <= available:
            slack = available - total
            gap = slack / (count - 1) if count > 1 else 0.0
            return 1.0, min(gap, BEARING * unit)
    squeeze = (available - (count - 1) * BEARING_MIN * unit) / max(total, 1.0)
    if squeeze < MIN_SQUEEZE:
        return None
    return squeeze, BEARING_MIN * unit * squeeze


def paper_texture(shape: tuple[int, int], source: np.ndarray) -> np.ndarray:
    """Высокочастотная фактура бумаги, размноженная с чистого участка.

    Args:
        shape: Размер нужного куска (h, w).
        source: Чистый участок бумаги BGR.

    Returns:
        Массив приращений той же формы.
    """
    texture = np.clip(source - cv2.GaussianBlur(source, (0, 0), 26), -5.0, 5.0)
    height, width = shape
    tiles = (height // texture.shape[0] + 1, width // texture.shape[1] + 1, 1)
    return np.tile(texture, tiles)[:height, :width]


def diffuse(roi: np.ndarray, unknown: np.ndarray, iterations: int = 700) -> np.ndarray:
    """Синтезирует бумагу в маске релаксацией Лапласа на уменьшенной сетке.

    Args:
        roi: Кусок кадра BGR.
        unknown: Маска неизвестных пикселей.
        iterations: Сколько итераций релаксации.

    Returns:
        Кусок той же формы с заполненной маской.
    """
    height, width = unknown.shape
    small_h, small_w = max(3, height // 5), max(3, width // 5)
    small = cv2.resize(unknown.astype(np.float32), (small_w, small_h), interpolation=cv2.INTER_AREA)
    known = (small < 0.25).astype(np.float32)
    out = np.empty_like(roi)
    for channel in range(3):
        source = cv2.resize(roi[..., channel], (small_w, small_h), interpolation=cv2.INTER_AREA)
        if known.sum() < 6:
            out[..., channel] = float(np.median(roi[..., channel]))
            continue
        current = np.where(known > 0, source, float((source * known).sum() / known.sum()))
        for _ in range(iterations):
            current = cv2.GaussianBlur(current, (0, 0), 1.4)
            current = np.where(known > 0, source, current)
        big = cv2.resize(current, (width, height), interpolation=cv2.INTER_CUBIC)
        out[..., channel] = cv2.GaussianBlur(big, (0, 0), 3.5)
    return out


def align_to_line(
    mask: np.ndarray, word: str, library: dict, baseline: int, squeeze: float, x_from: int, x_to: int, search: int = 26
) -> int:
    """Подгоняет базовую линию вставки по уцелевшему тексту той же строки.

    ЗАЧЕМ, ЕСЛИ БАЗОВАЯ ЛИНИЯ УЖЕ ПОСЧИТАНА. Любая её оценка немного врёт, и вклеенное
    слово садится на полтора десятка пикселей мимо строки — на глаз это четверть
    междустрочья и видно сразу. Здесь ответ берётся не из оценки, а из сравнения:
    вертикальный профиль набираемого слова прикладывается к профилю соседнего текста
    той же строки и сдвигается туда, где они совпадают. Способ не зависит от того,
    чем именно врёт оценка.

    Args:
        mask: Маска краски кадра (в координатах ИСХОДНОГО кадра).
        word: Набираемое слово.
        library: Библиотека литер.
        baseline: Исходная оценка базовой линии.
        squeeze: Сжатие литер.
        x_from: Левая граница окна с уцелевшим текстом.
        x_to: Правая граница окна.
        search: Насколько далеко искать сдвиг.

    Returns:
        Уточнённая базовая линия.
    """
    letters = [library[c] for c in word if c in library]
    if not letters or x_to - x_from < 40:
        return baseline
    height = letters[0].ratio.shape[0]
    want = np.zeros(height, np.float32)
    for glyph in letters:
        want += (glyph.ratio.mean(axis=2) < 0.6).sum(axis=1) * squeeze
    if want.sum() <= 0:
        return baseline
    want -= want.mean()
    best, best_score = baseline, -1e18
    for shift in range(-search, search + 1):
        top = baseline - CUT_UP + shift
        if top < 0 or top + height > mask.shape[0]:
            continue
        have = mask[top : top + height, x_from:x_to].sum(axis=1).astype(np.float32)
        if have.sum() <= 0:
            continue
        score = float(np.dot(want, have - have.mean()) / (np.linalg.norm(have - have.mean()) + 1e-6))
        if score > best_score:
            best, best_score = baseline + shift, score
    return best


def paste_word(canvas: np.ndarray, placement: Placement, library: dict[str, Glyph], gap: float) -> None:
    """Набирает слово литерами-донорами на месте.

    Args:
        canvas: Выходной кадр BGR float32; меняется на месте.
        placement: Куда и что набрать.
        library: Библиотека литер.
        gap: Межбуквенный пробел в пикселях.
    """
    cursor = float(placement.x_start)
    for char in placement.word:
        glyph = library[char]
        ratio = glyph.ratio
        new_width = max(1, int(round(ratio.shape[1] * placement.squeeze)))
        scaled = cv2.resize(ratio, (new_width, ratio.shape[0]), interpolation=cv2.INTER_CUBIC)
        x = int(round(cursor - glyph.left_pad * placement.squeeze))
        y = placement.baseline - CUT_UP
        height, width = scaled.shape[:2]
        if y < 0 or x < 0 or y + height > canvas.shape[0] or x + width > canvas.shape[1]:
            cursor += glyph.ink_w * placement.squeeze + gap
            continue
        fade_x = np.ones(width, np.float32)
        fade_y = np.ones(height, np.float32)
        fade_x[:2], fade_x[-2:] = (0.0, 0.5), (0.5, 0.0)
        fade_y[:2], fade_y[-2:] = (0.0, 0.5), (0.5, 0.0)
        alpha = (fade_y[:, None] * fade_x[None, :])[..., None]
        canvas[y : y + height, x : x + width] *= scaled * alpha + (1.0 - alpha)
        cursor += glyph.ink_w * placement.squeeze + gap
