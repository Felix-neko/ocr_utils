"""Библиотека литер, собранная из самой полосы.

ЗАЧЕМ ДОНОРЫ, А НЕ ШРИФТ. Дорисованная буква должна совпасть с соседями по гарнитуре,
насыщенности краски, размытию объектива, зерну бумаги и её тону. Подобрать это шрифтом
нельзя; зато на той же полосе стоят тысячи нужных букв — их и берём.

КАК ЛИТЕРА ОПОЗНАЁТСЯ БЕЗ РУЧНОГО ТРУДА. У слова известен текст (его прочитал surya) и
известен бокс. Если число вертикальных промежутков краски внутри бокса совпало с числом
букв слова, соответствие однозначно, и каждая буква ложится в библиотеку. Слова, где не
совпало (буквы слиплись, распознаватель ошибся), просто пропускаются: на полосе их
хватает и без того.

ОТКУДА БРАТЬ. Только из середины полосы: у корешка набор перспективно сжат, и такая
литера принесла бы искажение с собой.
"""

from dataclasses import dataclass

import numpy as np

import cv2

# Окно вырезки относительно базовой линии: выносные элементы вверх и вниз.
CUT_UP, CUT_DOWN = 46, 18

# Из какой части ширины полосы брать доноров (доли от наружного края к корешку).
DONOR_ZONE = (0.10, 0.72)

# Сколько образцов на букву хранить, прежде чем выбрать типичный.
KEEP_SAMPLES = 12


@dataclass
class Glyph:
    """Литера-донор.

    Attributes:
        ratio: Отношение пикселей литеры к уровню бумаги (BGR).
        ink_w: Ширина очка в пикселях.
        left_pad: Сколько пустого поля слева включено в вырезку.
    """

    ratio: np.ndarray
    ink_w: float
    left_pad: float


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Маска краски штрихового масштаба.

    Args:
        gray: Полутоновый кадр полного разрешения.

    Returns:
        Двоичная маска.
    """
    paper = cv2.GaussianBlur(gray, (0, 0), 12)
    response = cv2.GaussianBlur(gray, (0, 0), 1.0) - cv2.GaussianBlur(gray, (0, 0), 4.0)
    return (response < -0.055 * np.maximum(paper, 1.0)).astype(np.uint8)


def local_baseline(mask: np.ndarray, x0: int, x1: int, guess: float, half: int = 30) -> int:
    """Базовая линия по нижней границе полосы строчных в окне.

    Args:
        mask: Маска краски.
        x0: Левая граница окна.
        x1: Правая граница окна.
        guess: Ожидаемое положение базовой линии.
        half: Полувысота окна поиска.

    Returns:
        Строка базовой линии.
    """
    top = int(guess - half)
    band = mask[top : int(guess + half), x0:x1].sum(axis=1).astype(float)
    if band.max() <= 0:
        return int(guess)
    level = 0.4 * band.max()
    runs, inside, start = [], False, 0
    for i, value in enumerate(band):
        if not inside and value > level:
            inside, start = True, i
        elif inside and value <= level:
            inside = False
            runs.append((start, i - 1))
    if inside:
        runs.append((start, len(band) - 1))
    if not runs:
        return int(guess)
    best = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 + top - guess))
    return best[1] + top


def _paper_level(image: np.ndarray, mask: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Уровень бумаги вокруг литеры — по пикселям, далёким от краски."""
    context = image[y0:y1, x0:x1].astype(np.float32)
    distance = cv2.distanceTransform(1 - mask[y0:y1, x0:x1], cv2.DIST_L2, 3)
    far = distance > (6 if (distance > 6).sum() > 50 else 3)
    if far.sum() < 20:
        return np.array([np.median(context[..., c]) for c in range(3)], np.float32)
    return np.array([np.median(context[..., c][far]) for c in range(3)], np.float32)


def build_library(image: np.ndarray, mask: np.ndarray, rows, pitch: float, x_from: int, x_to: int) -> dict[str, Glyph]:
    """Собирает библиотеку литер по сшитым строкам полосы.

    Args:
        image: Кадр BGR полного разрешения.
        mask: Маска краски.
        rows: Список (верх, низ, текст) строк полосы.
        pitch: Шаг строк.
        x_from: Левая граница зоны доноров.
        x_to: Правая граница зоны доноров.

    Returns:
        Словарь «буква -> литера».
    """
    from ocr_utils.gutter_loss_restoration.layout import align, groups, word_groups

    samples: dict[str, list[tuple[int, int, int]]] = {}
    for top, bottom, text in rows:
        if not text:
            continue
        words = word_groups(mask, top, bottom, x_from, x_to, pitch)
        paired = align(words, text)
        if not paired:
            continue
        baseline_guess = bottom - (bottom - top) * 0.22
        for group, token in paired:
            letters = token.strip()
            if not letters:
                continue
            # Только строго разделённые буквы. Резать слипшиеся по минимуму профиля
            # пробовали — ширины очка выходят от 3 до 168 px вместо тридцати, и такой
            # «донор» превращает вклейку в чёрную плашку. Слов на выпуске хватает и без
            # них: библиотека собирается разом по всей папке, а не по одной полосе.
            runs = groups(mask, top, bottom, group.x0 - 1, group.x1 + 2)
            if len(runs) != len(letters):
                continue
            baseline = local_baseline(mask, group.x0, group.x1, baseline_guess)
            for char, run in zip(letters, runs):
                if run.width >= 3:
                    samples.setdefault(char, []).append((run.x0, run.x1, baseline))

    return _pick(samples, image, mask)


def _pick(samples, image, mask) -> dict[str, Glyph]:
    """Выбирает по одному типичному образцу на букву и режет из кадра."""
    library: dict[str, Glyph] = {}
    for char, found in samples.items():
        found = _typical(found)
        if not found:
            continue
        widths = sorted(found, key=lambda s: s[1] - s[0])
        a, b, baseline = widths[len(widths) // 2]
        y0, y1 = baseline - CUT_UP, baseline + CUT_DOWN
        if y0 < 0 or y1 > image.shape[0]:
            continue
        left, right = a, b
        while left > a - 6 and left > 1 and mask[y0:y1, left - 1].sum() == 0:
            left -= 1
        while right < b + 6 and right < image.shape[1] - 2 and mask[y0:y1, right + 1].sum() == 0:
            right += 1
        patch = image[y0:y1, left : right + 1].astype(np.float32)
        level = _paper_level(image, mask, y0, y1, max(0, left - 40), min(image.shape[1], right + 41))
        library[char] = Glyph(
            ratio=patch / np.maximum(level[None, None, :], 1.0), ink_w=float(b - a + 1), left_pad=float(a - left)
        )
    return library


def _typical(found: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Оставляет образцы с типичной шириной очка."""
    widths = np.array([b - a for a, b, _ in found], float)
    median = float(np.median(widths))
    keep = [s for s, w in zip(found, widths) if abs(w - median) <= 2]
    return keep or found
