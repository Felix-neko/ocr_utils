"""Вёрстка строки по краске: где кончается строка, где слова, где буквы.

ЗАЧЕМ, ЕСЛИ ЕСТЬ РАСПОЗНАВАТЕЛЬ. Surya отдаёт текст строки уверенно, но геометрию —
нет: боксы слов у него здесь не заполняются (все слова строки получают один и тот же
прямоугольник), а бокс строки прихватывает соседние. Восстановление же опирается именно
на геометрию: где начинается последнее слово, куда ставить буквы, какой ширины очко.
Поэтому текст берётся у распознавателя, а разметка считается по маске краски.

СОПОСТАВЛЕНИЕ. Слова текста и группы краски сшиваются по числу: если их поровну,
соответствие однозначно. Не сошлось — строка пропускается. На полосе строк много,
и терять часть из них дешевле, чем угадывать.
"""

from dataclasses import dataclass

import numpy as np

import cv2


@dataclass(frozen=True)
class Group:
    """Связная группа краски (слово или буква).

    Attributes:
        x0: Левый край.
        x1: Правый край (включительно).
    """

    x0: int
    x1: int

    @property
    def width(self) -> int:
        """Ширина группы."""
        return self.x1 - self.x0 + 1


def band(mask: np.ndarray, top: int, bottom: int, x0: int, x1: int) -> tuple[int, int]:
    """Уточняет полосу строки внутри широкого бокса распознавателя.

    Args:
        mask: Маска краски.
        top: Верх бокса.
        bottom: Низ бокса.
        x0: Левая граница строки.
        x1: Правая граница строки.

    Returns:
        Пара (верх, низ) собственно строки.
    """
    rows = mask[top:bottom, x0:x1].sum(axis=1).astype(np.float32)
    if rows.max() <= 0:
        return top, bottom
    rows = cv2.blur(rows.reshape(-1, 1), (1, 3)).ravel()
    level = 0.25 * float(rows.max())
    centre = len(rows) // 2
    runs, inside, start = [], False, 0
    for i, value in enumerate(rows):
        if not inside and value > level:
            inside, start = True, i
        elif inside and value <= level:
            inside = False
            runs.append((start, i))
    if inside:
        runs.append((start, len(rows)))
    if not runs:
        return top, bottom
    best = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - centre))
    return top + best[0], top + best[1]


def groups(mask: np.ndarray, top: int, bottom: int, x0: int, x1: int, min_gap: int = 1) -> list[Group]:
    """Группы краски по x, разделённые промежутками не меньше ``min_gap``.

    Args:
        mask: Маска краски.
        top: Верх полосы строки.
        bottom: Низ полосы строки.
        x0: Левая граница.
        x1: Правая граница.
        min_gap: Минимальный промежуток, разделяющий группы.

    Returns:
        Список групп слева направо.
    """
    columns = mask[top:bottom, x0:x1].sum(axis=0) > 0
    out, inside, start, gap = [], False, 0, 0
    for i, value in enumerate(columns):
        if value:
            if not inside:
                inside, start = True, i
            gap = 0
        elif inside:
            gap += 1
            if gap >= min_gap:
                out.append(Group(x0 + start, x0 + i - gap))
                inside = False
    if inside:
        out.append(Group(x0 + start, x1 - 1))
    return [g for g in out if g.width >= 2]


def word_gap(letters: list[Group]) -> int:
    """Порог межсловного пробела, взятый по самой строке.

    Фиксированная доля кегля не годится: набор выключён по формату, и пробелы гуляют
    от строки к строке в полтора раза. Зато внутри строки промежутки распадаются на две
    кучки — межбуквенные и межсловные, — и граница между ними видна как самый большой
    скачок в упорядоченном списке.

    Args:
        letters: Группы краски строки, полученные с минимальным промежутком.

    Returns:
        Порог в пикселях.
    """
    gaps = sorted(b.x0 - a.x1 - 1 for a, b in zip(letters, letters[1:]))
    if len(gaps) < 4:
        return 12
    jumps = [(gaps[i + 1] - gaps[i], i) for i in range(len(gaps) - 1)]
    size, index = max(jumps)
    if size < 3:
        return gaps[-1] + 1
    return (gaps[index] + gaps[index + 1]) // 2 + 1


def word_groups(mask: np.ndarray, top: int, bottom: int, x0: int, x1: int, pitch: float) -> list[Group]:
    """Слова строки: группы краски, разделённые межсловным пробелом.

    Args:
        mask: Маска краски.
        top: Верх полосы строки.
        bottom: Низ полосы строки.
        x0: Левая граница.
        x1: Правая граница.
        pitch: Шаг строк — мера кегля.

    Returns:
        Список слов слева направо.
    """
    return groups(mask, top, bottom, x0, x1, min_gap=max(5, int(round(0.17 * pitch))))


def edge_word(
    mask: np.ndarray, top: int, bottom: int, x0: int, x1: int, token: str, widths: dict, side: str
) -> Group | None:
    """Границы краевого слова строки — по ОЖИДАЕМОЙ ширине его букв.

    ПОЧЕМУ НЕ ПО ПРОБЕЛАМ. Набор выключён по формату, межсловные пробелы гуляют, и любой
    порог «промежуток шире стольких-то пикселей» то склеивает два слова в одно (замерено:
    «специально остано» ушло одной группой в полтысячи пикселей, и стиралось всё разом),
    то рвёт слово пополам. Зато ширина очка каждой буквы известна из библиотеки, а текст
    слова — от распознавателя. Идём от края строки внутрь и набираем буквы, пока не
    наберётся ожидаемая ширина: способ проверяет сам себя.

    Args:
        mask: Маска краски.
        top: Верх полосы строки.
        bottom: Низ полосы строки.
        x0: Левая граница строки.
        x1: Правая граница строки.
        token: Видимая часть краевого слова.
        widths: Ширины очка букв, «буква -> px».
        side: "L" — слово в конце строки, "R" — в начале.

    Returns:
        Границы слова либо None, если строка пуста.
    """
    letters = groups(mask, top, bottom, x0, x1, min_gap=1)
    if not letters:
        return None
    typical = float(np.median(list(widths.values()))) if widths else 28.0
    want = sum(widths.get(c, typical) for c in token) if token else typical
    order = letters[::-1] if side == "L" else letters
    edge = order[0]
    best, best_error = edge, abs(edge.width - want)
    for run in order[1:]:
        span = (edge.x1 - run.x0 + 1) if side == "L" else (run.x1 - edge.x0 + 1)
        if span > want * 1.6:
            break
        error = abs(span - want)
        if error < best_error:
            best, best_error = run, error
    return Group(best.x0, edge.x1) if side == "L" else Group(edge.x0, best.x1)


def baseline_in_band(mask: np.ndarray, top: int, bottom: int, x0: int, x1: int) -> int:
    """Базовая линия строки — низ полосы строчных внутри её собственной полосы.

    Искать её окном вокруг «примерно там» нельзя: при шаге строк в 70 px окно в ±30 px
    цепляет выносные элементы соседней строки, и базовая линия уезжает на строку вниз —
    замерено, слово вклеивалось в межстрочье.

    Args:
        mask: Маска краски.
        top: Верх полосы строки.
        bottom: Низ полосы строки.
        x0: Левая граница окна.
        x1: Правая граница окна.

    Returns:
        Строка базовой линии.
    """
    rows = mask[top:bottom, x0:x1].sum(axis=1).astype(np.float32)
    if rows.max() <= 0:
        return bottom
    # Базовая линия — низ полосы строчных. Ищется снизу вверх: первая строка, где
    # краски снова столько же, сколько в теле строки. Ниже неё идут только выносные
    # элементы, а их считаные проценты, и порог в половину ПИКА профиля их не замечает.
    # Порог от медианы полосы не годится: полоса строки прихватывает выносные соседней,
    # медиана из-за них поднимается, и базовая линия уезжает на два десятка пикселей
    # вниз — слово садится в межстрочье. Долю накопленной краски тоже пробовали: она
    # зависит от того, где обрезалась полоса, и разброс выходил в 15 px.
    level = 0.55 * float(rows.max())
    if level <= 0:
        return bottom
    # Берём конец САМОГО ДЛИННОГО куска выше порога, а не первый сверху или снизу:
    # выносные элементы соседней строки дают одиночные всплески, и «первый снизу»
    # цепляется за них — замерено, базовая линия уезжала на восемь пикселей вниз.
    best, best_len, start = None, 0, None
    for i, value in enumerate(rows):
        if value >= level and start is None:
            start = i
        elif value < level and start is not None:
            if i - start > best_len:
                best, best_len = i - 1, i - start
            start = None
    if start is not None and len(rows) - start > best_len:
        best = len(rows) - 1
    return top + best if best is not None else bottom


def text_lines(mask: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> list[tuple[int, int]]:
    """Находит полосы печатных строк по профилю краски.

    Боксы строк у распознавателя перекрываются на две строки сразу, и опираться на них
    нельзя: внутри такого бокса каждая колонка занята краской, и слова не отделяются.
    Поэтому строки ищутся своим проходом по краске, а текст к ним пришивается по
    вертикальному перекрытию.

    Args:
        mask: Маска краски.
        x0: Левая граница наборной полосы.
        x1: Правая граница.
        y0: Верх зоны поиска.
        y1: Низ зоны поиска.

    Returns:
        Список пар (верх, низ) строк сверху вниз.
    """
    rows = mask[y0:y1, x0:x1].sum(axis=1).astype(np.float32)
    rows = cv2.blur(rows.reshape(-1, 1), (1, 3)).ravel()
    level = 0.16 * float(np.percentile(rows, 95))
    out, inside, start = [], False, 0
    for i, value in enumerate(rows):
        if not inside and value > level:
            inside, start = True, i
        elif inside and value <= level:
            inside = False
            if i - start >= 6:
                out.append((y0 + start, y0 + i))
    if inside and len(rows) - start >= 6:
        out.append((y0 + start, y1))
    return out


def match_text(bands: list[tuple[int, int]], ocr_lines) -> list[str]:
    """Пришивает текст распознанных строк к найденным полосам краски.

    Args:
        bands: Полосы строк сверху вниз.
        ocr_lines: Строки распознавателя.

    Returns:
        Список текстов той же длины, что и ``bands``; пустая строка — пары не нашлось.
    """
    out = []
    for top, bottom in bands:
        best, score = "", 0
        for line in ocr_lines:
            overlap = min(bottom, line.bottom) - max(top, line.top)
            if overlap > score:
                best, score = line.text, overlap
        out.append(best if score > 0.4 * (bottom - top) else "")
    return out


def split_to(mask: np.ndarray, top: int, bottom: int, group: Group, count: int) -> list[Group] | None:
    """Режет слипшуюся группу краски на заданное число букв.

    ЗАЧЕМ. В наборе соседние буквы то и дело смыкаются, и требовать «промежутков ровно
    столько же, сколько букв» — значит выбросить почти все слова: на полосе «Планового
    хозяйства» такому условию отвечает одно слово из сорока. Между сомкнутыми буквами
    профиль краски всё равно даёт провал, по нему и режем.

    Результат проверяется по ширинам: если какой-то кусок вышел вдвое шире или вдвое уже
    типичного, разрез не удался и слово пропускается — испорченная литера хуже,
    чем ненайденная.

    Args:
        mask: Маска краски.
        top: Верх строки.
        bottom: Низ строки.
        group: Группа краски (слово или его часть).
        count: Сколько букв в ней должно быть.

    Returns:
        Список групп-букв либо None, если разрезать не удалось.
    """
    if count <= 1:
        return [group]
    columns = mask[top:bottom, group.x0 : group.x1 + 1].sum(axis=0).astype(np.float32)
    if columns.size < count * 2:
        return None
    smooth = cv2.blur(columns.reshape(1, -1), (3, 1)).ravel()
    width = group.width / count
    cuts = []
    for k in range(1, count):
        centre = int(round(k * width))
        lo, hi = max(1, centre - int(width * 0.42)), min(len(smooth) - 1, centre + int(width * 0.42))
        if hi <= lo:
            return None
        cuts.append(lo + int(np.argmin(smooth[lo:hi])))
    edges = [0] + sorted(cuts) + [group.width]
    pieces = [Group(group.x0 + edges[i], group.x0 + edges[i + 1] - 1) for i in range(count)]
    widths = np.array([p.width for p in pieces], float)
    if widths.min() < 0.40 * width or widths.max() > 2.0 * width:
        return None
    return pieces


def align(words: list[Group], text: str) -> list[tuple[Group, str]] | None:
    """Сшивает группы краски со словами текста, если их поровну.

    Args:
        words: Группы краски слева направо.
        text: Текст строки.

    Returns:
        Список пар (группа, слово) либо None, если число не сошлось.
    """
    tokens = [t for t in text.split() if t]
    if len(tokens) != len(words) or not tokens:
        return None
    return list(zip(words, tokens))
