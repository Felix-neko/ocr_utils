"""Восстановление одного разворота: что съел корешок — то и дописываем.

Сводит вместе распознавание (`pageocr`), словарь выпуска (`lexicon`), вёрстку по краске
(`layout`), разрешение хвостов (`tails`), библиотеку литер (`glyphs`) и вклейку
(`compose`).

РАЗДЕЛЕНИЕ ТРУДА. Текст строки берётся у распознавателя, вся геометрия — своя, по маске
краски: боксы surya здесь перекрываются на две строки и не годятся ни на что, кроме
чтения. Для места вставки полная сшивка слов не нужна — достаточно крайней группы краски
(это и есть обрезанное слово) и крайнего слова текста. Полная сшивка нужна только
библиотеке литер, и там она обязательна.

ЧТО ЧИНИТСЯ. Только строки, для которых слово вычислено однозначно по переносу. Всё
остальное остаётся нетронутым и попадает в отчёт с причиной: дописать в архивный скан
неверную букву — не опечатка, а подделка источника.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import cv2

from ocr_utils.gutter_loss_restoration.compose import (
    BAND_DOWN_K,
    BAND_UP_K,
    align_to_line,
    PAD_PITCHES,
    Placement,
    diffuse,
    fit_word,
    paper_texture,
    paste_word,
    widen,
)
from ocr_utils.gutter_loss_restoration.glyphs import ink_mask
from ocr_utils.gutter_loss_restoration.layout import (
    Group,
    baseline_in_band,
    edge_word,
    match_text,
    text_lines,
    word_groups,
)
from ocr_utils.gutter_loss_restoration.tails import Tail, _candidates, letters_only
from ocr_utils.gutter_loss_restoration.lexicon import normalize

# Отступ набора от нового сгиба.
GUARD = 6

# Запас вокруг области стирания, где диффузия берёт известную бумагу.
ROI_PAD = 30

# Границы наборной полосы в долях ширины кадра.
OUTER, INNER = 0.07, 0.006

# Зона, из которой берутся литеры-доноры (доли ширины полосы от наружного края).
DONOR_FROM, DONOR_TO = 0.06, 0.80


@dataclass
class Row:
    """Строка полосы: полоса краски, текст и группы слов.

    Attributes:
        top: Верх строки.
        bottom: Низ строки.
        text: Текст, прочитанный распознавателем.
        words: Группы краски, слева направо.
    """

    top: int
    bottom: int
    text: str
    words: list[Group]


@dataclass
class LineReport:
    """Что сделано (или не сделано) со строкой.

    Attributes:
        side: Полоса.
        index: Номер строки.
        visible: Видимая часть слова.
        word: Слово целиком, как набрано.
        added: Что дописано.
        done: Починена ли строка.
        reason: Пояснение.
    """

    side: str
    index: int
    visible: str
    word: str
    added: str
    done: bool
    reason: str


def _rows(mask: np.ndarray, x0: int, x1: int, ocr_lines, pitch_hint: float) -> tuple[list[Row], float]:
    """Собирает строки полосы: полосы краски, текст и слова."""
    height = mask.shape[0]
    bands = text_lines(mask, x0, x1, int(height * 0.10), int(height * 0.93))
    if len(bands) < 5:
        return [], pitch_hint
    pitch = float(np.median(np.diff([(a + b) / 2 for a, b in bands])))
    if not np.isfinite(pitch) or pitch <= 4:
        pitch = pitch_hint
    texts = match_text(bands, ocr_lines)
    rows = []
    for (top, bottom), text in zip(bands, texts):
        rows.append(Row(top=top, bottom=bottom, text=text, words=word_groups(mask, top, bottom, x0, x1, pitch)))
    return rows, pitch


def _tail_of(row: Row) -> tuple[Group, str] | None:
    """Обрезанное слово в конце строки: крайняя группа краски и крайнее слово текста."""
    tokens = row.text.split()
    if not row.words or not tokens:
        return None
    return row.words[-1], tokens[-1]


def _head_of(row: Row) -> tuple[Group, str] | None:
    """Обрезанное слово в начале строки."""
    tokens = row.text.split()
    if not row.words or not tokens:
        return None
    return row.words[0], tokens[0]


def _resolve(rows: list[Row], side: str, lexicon: set[str], fold: int, pitch: float) -> list[Tail]:
    """Разрешает обрезанные слова полосы.

    На левой полосе срезан КОНЕЦ строки, на правой — НАЧАЛО: корешок у них с разных
    сторон, и пара «слово и его продолжение» берётся в разном порядке.
    """
    out = []
    for i, row in enumerate(rows):
        pair = _tail_of(row) if side == "L" else _head_of(row)
        if pair is None:
            out.append(Tail(i, reason="строка без слов"))
            continue
        group, token = pair
        visible_raw = letters_only(token)
        visible = normalize(visible_raw)
        if not visible:
            out.append(Tail(i, reason="край строки не слово"))
            continue
        neighbour = (
            rows[i + 1] if (side == "L" and i + 1 < len(rows)) else (rows[i - 1] if (side == "R" and i) else None)
        )
        other = neighbour.text.split() if neighbour else []
        head = normalize(letters_only(other[0] if side == "L" else other[-1])) if other else ""

        # РЕШАЕТ НЕ ХВОСТ, А СОСЕД. Если край соседней строки сам по себе не слово —
        # значит, там вторая половина перенесённого слова, и разрыв точно был. Это
        # свидетельство сразу двух строк, и оно сильнее, чем «похоже на слово».
        # Если же сосед — нормальное слово, переноса не было, и остаётся слабая гипотеза
        # «у целого слова срезаны последние буквы».
        prefix, suffix = (visible, head) if side == "L" else (head, visible)
        if head and head not in lexicon:
            found = _candidates(prefix, suffix, lexicon)
            if not found:
                out.append(Tail(i, visible=visible_raw, x_start=group.x0, reason="перенос не собрался по словарю"))
                continue
            if len(found) > 1:
                out.append(
                    Tail(
                        i,
                        visible=visible_raw,
                        x_start=group.x0,
                        reason=f"перенос неоднозначен ({len(found)} вариантов)",
                    )
                )
                continue
            insert = found[0]
            word = (visible_raw + insert + "-") if side == "L" else (insert + visible_raw)
        else:
            if visible in lexicon:
                out.append(Tail(i, visible=visible_raw, x_start=group.x0, reason="край сам по себе слово — не трогаем"))
                continue
            if side == "R" or len(visible) < 4:
                out.append(Tail(i, visible=visible_raw, x_start=group.x0, reason="нет опоры для догадки"))
                continue
            found = [x for x in _candidates(visible, "", lexicon) if x and len(x) <= 2]
            if len(found) != 1:
                out.append(
                    Tail(
                        i,
                        visible=visible_raw,
                        x_start=group.x0,
                        reason=f"конец слова неоднозначен ({len(found)} вариантов)",
                    )
                )
                continue
            insert, word = found[0], visible_raw + found[0]
        if word == visible_raw:
            out.append(Tail(i, visible=visible_raw, x_start=group.x0, reason="дописывать нечего"))
            continue
        out.append(
            Tail(
                index=i,
                word=word,
                visible=visible_raw,
                x_start=group.x0,
                added=insert,
                reason="собрано по словарю",
                ok=True,
            )
        )
    return out


def _fold_per_row(gray: np.ndarray, fold: int) -> np.ndarray:
    """Столбец сгиба для каждой строки кадра.

    Сгиб на съёмке с рук наклонён; прямая по тёмному следу описывает его точнее, чем
    одно число, и без неё раздвижка кадра срезает текст на дальних строках.
    """
    height = gray.shape[0]
    blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
    step = max(8, height // 200)
    ys, xs = [], []
    for y in range(int(height * 0.08), int(height * 0.94), step):
        a, b = max(0, fold - 90), min(gray.shape[1], fold + 90)
        ys.append(y + step / 2)
        xs.append(a + int(np.argmin(blurred[y : y + step, a:b].mean(axis=0))))
    ys, xs = np.array(ys, float), np.array(xs, float)
    keep = np.abs(xs - np.median(xs)) < 45
    if keep.sum() < 8:
        return np.full(height, float(fold))
    line = np.polyfit(ys[keep], xs[keep], 1)
    keep &= np.abs(xs - np.polyval(line, ys)) < 18
    if keep.sum() >= 8:
        line = np.polyfit(ys[keep], xs[keep], 1)
    return np.polyval(line, np.arange(height, dtype=float))


def restore_spread(path: Path, fold: int, halves: dict, lexicon: set[str], shared: dict | None = None):
    """Восстанавливает обе полосы разворота.

    Args:
        path: Путь к кадру.
        fold: Столбец сгиба в исходном кадре.
        halves: Распознанные полосы (текст строк).
        lexicon: Словарь выпуска.
        shared: Библиотека литер выпуска — ею закрываются буквы, которых не нашлось на
            самом кадре. Собственные литеры кадра всегда в приоритете: они точнее
            совпадут по свету и размытию.

    Returns:
        Тройка (выходной кадр BGR uint8 либо None, отчёт по строкам, причина отказа).
    """
    image = cv2.imread(str(path))
    if image is None:
        return None, [], "кадр не читается"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = ink_mask(gray)
    height, width = gray.shape

    zones = {
        "L": (int(width * OUTER), int(fold - width * INNER)),
        "R": (int(fold + width * INNER), int(width * (1 - OUTER))),
    }
    rows_by_side, pitches = {}, []
    for side, (x0, x1) in zones.items():
        rows, pitch = _rows(mask, x0, x1, halves.get(side, []), 60.0)
        rows_by_side[side] = rows
        if rows:
            pitches.append(pitch)
    if not pitches:
        return None, [], "строки не найдены"
    pitch = float(np.median(pitches))

    # Литеры берутся ТОЛЬКО из библиотеки выпуска. Собирать их заново по одной полосе
    # пробовали: образцов там полтора десятка, согласия между ними нет, и в букву «о»
    # приезжает «п» — слово набирается как «предлпжения». Выпуск же набран одной
    # гарнитурой и снят за один сеанс, так что общая библиотека подходит всем кадрам.
    library = dict(shared or {})
    if len(library) < 22:
        return None, [], f"мало литер в библиотеке ({len(library)})"

    pad = int(round(2 * PAD_PITCHES * pitch))
    fold_at = _fold_per_row(gray, fold)
    canvas, new_fold_at = widen(image.astype(np.float32), fold_at, pad)
    new_fold = int(round(float(new_fold_at[height // 2])))
    shift = {"L": 0, "R": pad}

    reports: list[LineReport] = []
    placements: list[tuple[Placement, float]] = []
    for side in ("L", "R"):
        rows = rows_by_side[side]
        for tail in _resolve(rows, side, lexicon, fold, pitch):
            row = rows[tail.index]
            if not tail.ok:
                reports.append(LineReport(side, tail.index, tail.visible, "", "", False, tail.reason))
                continue
            widths = {c: g.ink_w for c, g in library.items()}
            group = edge_word(
                mask, row.top, row.bottom, zones[side][0], zones[side][1], tail.visible, widths, side
            ) or (row.words[-1] if side == "L" else row.words[0])
            baseline = baseline_in_band(mask, row.top, row.bottom, zones[side][0], zones[side][1])
            if side == "L":
                start = group.x0 + shift[side]
                available = (new_fold - GUARD) - start
            else:
                stop = group.x1 + shift[side]
                available = stop - (new_fold + GUARD)
            fitted = fit_word(tail.word, library, available)
            if fitted is None:
                reports.append(
                    LineReport(side, tail.index, tail.visible, tail.word, tail.added, False, "нет места или нет литеры")
                )
                continue
            squeeze, gap = fitted
            # Оценку базовой линии уточняем по уцелевшему тексту той же строки.
            if side == "L":
                probe = (max(0, group.x0 - 320), max(1, group.x0 - 4))
            else:
                probe = (min(width - 2, group.x1 + 4), min(width - 1, group.x1 + 320))
            baseline = align_to_line(mask, tail.word, library, int(baseline), squeeze, probe[0], probe[1])
            need = sum(library[c].ink_w for c in tail.word) * squeeze + gap * (len(tail.word) - 1)
            x_start = start if side == "L" else int(round(stop - need))
            placements.append(
                (
                    Placement(
                        side=side,
                        word=tail.word,
                        x_start=int(x_start),
                        x_stop=int(x_start + need),
                        baseline=int(baseline),
                        squeeze=squeeze,
                    ),
                    gap,
                )
            )
            reports.append(LineReport(side, tail.index, tail.visible, tail.word, tail.added, True, tail.reason))

    if not placements:
        return None, reports, "нечего восстанавливать"

    _repaint(canvas, placements, new_fold, mask, shift, pitch)
    for placement, gap in placements:
        paste_word(canvas, placement, library, gap)
    return np.clip(canvas, 0, 255).astype(np.uint8), reports, ""


def _repaint(canvas, placements, new_fold, mask, shift, pitch) -> None:
    """Стирает обрезанные хвосты и доращивает бумагу до нового сгиба."""
    height = canvas.shape[0]
    up, down = int(round(BAND_UP_K * pitch)), int(round(BAND_DOWN_K * pitch))
    clean = canvas[int(height * 0.93) :, :400]
    for side in ("L", "R"):
        mine = [p for p, _ in placements if p.side == side]
        if not mine:
            continue
        if side == "L":
            x0, x1 = min(p.x_start for p in mine) - ROI_PAD, new_fold + 2
        else:
            x0, x1 = new_fold - 1, max(p.x_stop for p in mine) + ROI_PAD
        x0, x1 = max(0, int(x0)), min(canvas.shape[1], int(x1))
        y0 = max(0, min(p.baseline for p in mine) - up - ROI_PAD)
        y1 = min(height, max(p.baseline for p in mine) + down + ROI_PAD)
        band = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for placement in mine:
            top = max(0, placement.baseline - up - y0)
            bottom = min(y1 - y0, placement.baseline + down - y0)
            if side == "L":
                left, right = max(0, placement.x_start - 3 - x0), x1 - x0
            else:
                left, right = 0, min(x1 - x0, placement.x_stop + 3 - x0)
            band[top:bottom, left:right] = 1
        roi = canvas[y0:y1, x0:x1].copy()
        shifted = _shifted_mask(mask, shift[side], y0, y1, x0, x1)
        unknown = np.clip(band + cv2.dilate(shifted, np.ones((7, 7), np.uint8)), 0, 1)
        paper = diffuse(roi, unknown) + paper_texture((y1 - y0, x1 - x0), clean)
        soft = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 2.6)[..., None]
        canvas[y0:y1, x0:x1] = roi * (1 - soft) + paper * soft


def _shifted_mask(mask, shift, y0, y1, x0, x1) -> np.ndarray:
    """Маска краски, перенесённая в координаты раздвинутого кадра."""
    out = np.zeros((y1 - y0, x1 - x0), np.uint8)
    src_x0, src_x1 = x0 - shift, x1 - shift
    a, b = max(0, src_x0), min(mask.shape[1], src_x1)
    if b > a:
        out[:, a - src_x0 : b - src_x0] = mask[y0:y1, a:b]
    return out
