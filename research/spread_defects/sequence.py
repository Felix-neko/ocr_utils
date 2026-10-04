"""Пропуски, дубли и пересъёмки разворотов в выпуске: номера страниц из колонтитула и смещение «номер − 2·порядок кадра».

Номер страницы у журнала стоит в наружном верхнем углу полосы (левая — чётная, правая — нечётная, правая = левая + 1).
Для каждого кадра читается верхняя полоса tesseract-ом (psm 11, разреженный текст), на каждой стороне сгиба берётся
самый верхний числовой токен. Затем по выпуску строится «номер левой полосы» ``s`` и смещение ``s − 2·k``, где ``k`` —
порядковый номер кадра в выпуске. Внутри выпуска смещение постоянно; скачок вверх — пропущенные развороты, вниз —
повтор (дубль или пересъёмка). Кадры без номера (обложка, вкладки, полосы без колонтитула) пропускаются, но
не ломают ряд: смещение сравнивается с ближайшими пронумерованными соседями.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from ocr_utils.rotated_text.tables.ocr import tesseract_tsv

# Верхняя полоса кадра, где ищется колонтитул, доли высоты.
HEADER_TOP, HEADER_BOTTOM = 0.04, 0.30
# Минимальная уверенность tesseract для числового токена.
MIN_CONFIDENCE = 30.0
# Номер страницы — 1–3 цифры.
_NUMBER = re.compile(r"\d{1,3}")
# Токены не выше самого верхнего на столько px (масштаб чтения) считаются той же строкой колонтитула.
SAME_LINE = 40


class Side(str, Enum):
    """Сторона разворота относительно сгиба."""

    LEFT = "L"
    RIGHT = "R"


@dataclass(frozen=True)
class NumberToken:
    """Числовой токен верхней полосы кадра.

    Attributes:
        value: Число.
        x, y: Левый верхний угол токена, px кадра чтения.
        confidence: Уверенность tesseract, 0..100.
    """

    value: int
    x: int
    y: int
    confidence: float


def header_tokens(gray: np.ndarray) -> list[NumberToken]:
    """Числовые токены верхней полосы кадра.

    Args:
        gray: Полутоновый кадр uint8 (масштаб чтения, обычно половина полного).

    Returns:
        Токены из 1–3 цифр с уверенностью не ниже ``MIN_CONFIDENCE``; координаты — в px переданного кадра.
    """
    height = gray.shape[0]
    top = int(HEADER_TOP * height)
    band = gray[top : int(HEADER_BOTTOM * height)]
    tokens = []
    for line in tesseract_tsv(band, lang="rus", psm=11):
        columns = line.split("\t")
        if len(columns) < 12:
            continue
        text = columns[11].strip()
        if not _NUMBER.fullmatch(text) or float(columns[10]) < MIN_CONFIDENCE:
            continue
        tokens.append(NumberToken(int(text), int(columns[6]), int(columns[7]) + top, float(columns[10])))
    return tokens


def page_number(tokens: list[NumberToken], fold_x: float, side: Side) -> NumberToken | None:
    """Номер страницы одной стороны разворота: самый верхний числовой токен, а среди токенов той же строки — самый наружный.

    Args:
        tokens: Числовые токены верхней полосы.
        fold_x: x сгиба в px того же масштаба.
        side: Сторона.

    Returns:
        Токен номера или None, если на стороне нет числовых токенов.
    """
    own = [t for t in tokens if (t.x < fold_x) == (side == Side.LEFT)]
    if not own:
        return None
    top = min(t.y for t in own)
    line = [t for t in own if t.y <= top + SAME_LINE]
    # Номер — в наружном углу: у левой полосы левее всех, у правой — правее всех.
    return min(line, key=lambda t: t.x) if side == Side.LEFT else max(line, key=lambda t: t.x)


def offset_candidates(k: int, left: int | None, right: int | None) -> set[int]:
    """Кандидаты смещения ``s − 2·k`` кадра по прочитанным номерам.

    Каждая сторона голосует отдельно: левая — если чётная, правая — если нечётная (``s = right − 1``).
    Несогласованная пара (одна сторона прочитана неверно) даёт двух кандидатов, и выбор между ними делают соседи.

    Args:
        k: Порядковый номер кадра в выпуске.
        left: Номер левой полосы или None.
        right: Номер правой полосы или None.

    Returns:
        Множество кандидатов (пустое, если номеров нет или чётность не та).
    """
    found = set()
    if left is not None and left % 2 == 0:
        found.add(left - 2 * k)
    if right is not None and right % 2 == 1:
        found.add(right - 1 - 2 * k)
    return found


class Event(str, Enum):
    """Событие в ряду кадров выпуска."""

    GAP = "gap"  # номер вырос сильнее, чем кадров между, — пропущены развороты
    REPEAT = "repeat"  # номер повторился — дубль или пересъёмка разворота
    EXTRA = "extra"  # кадров между больше, чем разворотов, — вкладка без номера или пересъёмка без номера
    BACKWARD = "backward"  # номер уменьшился — беспорядок кадров или ошибка чтения
    FOREIGN = (
        "foreign"  # участок вне главной цепочки: обложка, реклама, вкладка со своей нумерацией, серия ошибок чтения
    )


@dataclass(frozen=True)
class SequenceEvent:
    """Разрыв ряда между двумя пронумерованными кадрами выпуска.

    Attributes:
        kind: Вид события.
        before, after: Порядковые номера кадров (в выпуске) по обе стороны разрыва.
        number_before, number_after: Номера левых полос этих кадров.
        missing_spreads: Пропущено разворотов (``GAP``, плюс) или лишних кадров (``REPEAT``/``EXTRA``, минус):
            ``(number_after − number_before)/2 − (after − before)``.
    """

    kind: Event
    before: int
    after: int
    number_before: int
    number_after: int
    missing_spreads: int


# Окно голосования соседей за смещение кадра, кадров в каждую сторону.
VOTE_WINDOW = 4
# Минимальная длина участка с одним смещением, чтобы он шёл в расчёт (одиночки — ошибки чтения, обложки).
MIN_RUN = 2


def chosen_offsets(candidates: list[set[int]]) -> list[int | None]:
    """Смещение каждого кадра: кандидат с наибольшей поддержкой среди соседей в окне ``±VOTE_WINDOW``.

    Args:
        candidates: Кандидаты смещения по кадрам выпуска.

    Returns:
        Выбранное смещение кадра или None (кандидатов нет).
    """
    chosen = []
    for k, own in enumerate(candidates):
        if not own:
            chosen.append(None)
            continue
        window = candidates[max(0, k - VOTE_WINDOW) : k + VOTE_WINDOW + 1]
        # Поддержка — сколько кадров окна имеют того же кандидата; при равенстве — меньший (стабильный порядок).
        chosen.append(max(sorted(own), key=lambda o: sum(o in c for c in window)))
    return chosen


@dataclass(frozen=True)
class Run:
    """Участок выпуска с одним смещением номеров.

    Attributes:
        offset: Смещение ``s − 2·k``.
        frames: Порядковые номера кадров участка (только пронумерованные).
    """

    offset: int
    frames: tuple[int, ...]


def offset_runs(chosen: list[int | None]) -> list[Run]:
    """Участки подряд идущих пронумерованных кадров с одинаковым смещением, не короче ``MIN_RUN``.

    Args:
        chosen: Смещение каждого кадра или None.

    Returns:
        Участки по порядку; кадры без номера участок не рвут.
    """
    runs: list[list] = []
    for k, offset in enumerate(chosen):
        if offset is None:
            continue
        if runs and runs[-1][0] == offset:
            runs[-1][1].append(k)
        else:
            runs.append([offset, [k]])
    return [Run(offset, tuple(frames)) for offset, frames in runs if len(frames) >= MIN_RUN]


def main_chain(runs: list[Run]) -> list[int]:
    """Главная цепочка участков: неубывающая по номерам последовательность с наибольшим числом кадров.

    Номера в конце выпуска (задняя обложка, реклама со своей нумерацией) и отрезки, где цифра систематически
    прочитана неверно («7» как «1»), дают участки, выпадающие из ряда. Без цепочки каждый такой участок
    рождает пару ложных событий «назад» и «вперёд».

    Args:
        runs: Участки по порядку съёмки.

    Returns:
        Индексы участков цепочки по порядку.
    """
    if not runs:
        return []
    first = [r.offset + 2 * r.frames[0] for r in runs]
    last = [r.offset + 2 * r.frames[-1] for r in runs]
    best = [len(r.frames) for r in runs]
    parent = [-1] * len(runs)
    # Взвешенная наибольшая неубывающая подпоследовательность, O(n²): участков в выпуске — десятки.
    for n in range(len(runs)):
        for m in range(n):
            if last[m] <= first[n] and best[m] + len(runs[n].frames) > best[n]:
                best[n], parent[n] = best[m] + len(runs[n].frames), m
    chain, at = [], int(np.argmax(best))
    while at >= 0:
        chain.append(at)
        at = parent[at]
    return chain[::-1]


def sequence_events(candidates: list[set[int]]) -> list[SequenceEvent]:
    """Разрывы ряда номеров по выпуску.

    Args:
        candidates: Кандидаты смещения по кадрам выпуска (``offset_candidates``), по порядку съёмки.

    Returns:
        События между соседними участками главной цепочки (``GAP``, ``REPEAT``, ``EXTRA``) и по одному ``FOREIGN``
        на каждый участок вне цепочки, по порядку кадров.
    """
    runs = offset_runs(chosen_offsets(candidates))
    chain = main_chain(runs)
    events = []
    for a, b in zip(chain, chain[1:]):
        prev, nxt = runs[a], runs[b]
        before, after = prev.frames[-1], nxt.frames[0]
        number_before, number_after = prev.offset + 2 * before, nxt.offset + 2 * after
        # Сколько разворотов не хватает (+) или сколько кадров лишних (−) между двумя участками.
        balance = (number_after - number_before) // 2 - (after - before)
        if number_after == number_before:
            kind = Event.REPEAT
        elif balance > 0:
            kind = Event.GAP
        elif balance < 0:
            kind = Event.EXTRA
        else:
            continue
        events.append(SequenceEvent(kind, before, after, number_before, number_after, balance))
    for n, run in enumerate(runs):
        if n not in chain:
            first, last = run.offset + 2 * run.frames[0], run.offset + 2 * run.frames[-1]
            events.append(SequenceEvent(Event.FOREIGN, run.frames[0], run.frames[-1], first, last, 0))
    return sorted(events, key=lambda e: e.before)


def fold_x_of(gray: np.ndarray) -> float:
    """Грубый x сгиба для деления номеров по сторонам (масштаб переданного кадра).

    Args:
        gray: Полутоновый кадр.

    Returns:
        x сгиба посередине высоты.
    """
    from research.spread_defects.gutter import find_fold

    fold, _ = find_fold(gray.astype(np.float32))
    return float(np.polyval(fold, gray.shape[0] / 2))


def read_numbers(
    rgb_half: np.ndarray, fold_x: float | None = None
) -> tuple[NumberToken | None, NumberToken | None, list[NumberToken], float]:
    """Номера обеих полос кадра.

    Args:
        rgb_half: Кадр RGB uint8 в масштабе чтения (половина полного).
        fold_x: x сгиба в масштабе чтения, если уже известен; None — найти здесь.

    Returns:
        (номер левой, номер правой, все числовые токены полосы, x сгиба) — в px масштаба чтения.
    """
    gray = cv2.cvtColor(rgb_half, cv2.COLOR_RGB2GRAY)
    fold_x = fold_x_of(gray) if fold_x is None else fold_x
    tokens = header_tokens(gray)
    return page_number(tokens, fold_x, Side.LEFT), page_number(tokens, fold_x, Side.RIGHT), tokens, fold_x


def frame_numbers(candidates: list[set[int]]) -> list[int | None]:
    """Номер левой полосы каждого кадра выпуска: из главной цепочки участков, между участками — по соседям.

    Кадр внутри участка цепочки получает ``offset + 2·k``. Кадр вне участков (номер не прочитан или выпал)
    получает номер от ближайших пронумерованных соседей, только если предыдущий и следующий участки цепочки
    дают одно смещение (между ними нет ни пропуска, ни лишнего кадра — номер однозначен).

    Args:
        candidates: Кандидаты смещения по кадрам выпуска (``offset_candidates``).

    Returns:
        Номер левой полосы или None, если он не определяется однозначно.
    """
    runs = offset_runs(chosen_offsets(candidates))
    chain = [runs[n] for n in main_chain(runs)]
    numbers: list[int | None] = [None] * len(candidates)
    for run in chain:
        for k in run.frames:
            numbers[k] = run.offset + 2 * k
    for k in range(len(candidates)):
        if numbers[k] is not None:
            continue
        before = [r for r in chain if r.frames[-1] < k]
        after = [r for r in chain if r.frames[0] > k]
        if before and after and before[-1].offset == after[0].offset:
            numbers[k] = before[-1].offset + 2 * k
    # Одиночный кадр без номера между двумя пронумерованными соседями, которые отстоят ровно на два разворота.
    for k in range(1, len(candidates) - 1):
        prev, nxt = numbers[k - 1], numbers[k + 1]
        if numbers[k] is None and prev is not None and nxt is not None and nxt - prev == 4:
            numbers[k] = prev + 2
    return numbers
