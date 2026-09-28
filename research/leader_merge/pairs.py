"""Мера «две оси одного ряда сошлись на одном глифе и не срослись»: пары осей, перекрытые по x на одной высоте.

Типичный случай — ряд таблицы с отточием (1966/03 IMG_0131_2R): левая ось «В металлообработке.»
продлена по отточию до последней точки (``segment.extend_with_leaders``), а последние точки
при смыкании RLSA прилипли к числу «4,2», и правая ось начинается с них. Обе оси проходят над
одними и теми же точками, но остаются разными строками.

Пара ``(A, B)``, A левее B, считается находкой, если одновременно:

* оси перекрыты по x (``B.x0 < A.x1``), но не больше чем на ``MAX_OVERLAP_SHARE`` более короткой
  (иначе это две строки друг над другом или слово над чертой, а не соседи по ряду);
* на перекрытии оси идут на одной высоте: расхождение меньше ``MAX_DY_MM`` (треть шага строк пака-1;
  долей высоты оси не мерится — у куска «. . 2,2» высоту задают точки, 3 px);
* в перекрытии лежит общий глиф — по сохранённым осям это низкая метка (точка, запятая) одной
  из осей (``mark_spans``), а при наличии боксов глифов (``glyphs``) — любой глиф одной оси,
  над которым идёт другая.

Все координаты — пиксели рабочей копии (150 dpi), как в JSON разбора (:func:`report.page_json`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ocr_utils.page_layout import mm_to_px
from ocr_utils.page_layout.text_blocks import WORK_DPI

# Перекрытие больше этой доли более короткой оси — не соседи по ряду, а строки друг над другом.
MAX_OVERLAP_SHARE = 0.5
# На перекрытии оси расходятся не больше чем на столько миллиметров бумаги (шаг строк пака-1 около 3 мм).
MAX_DY_MM = 1.0
# Сколько точек перекрытия сверяется по высоте.
DY_SAMPLES = 5
# Хвост левой оси без букв перед перекрытием от стольких высот строки — продление по отточию, а не
# межсловный пробел или запятая.
LEADER_TAIL_HEIGHTS = 3.0
# Глиф считается «под осью», если середина его бокса по y не дальше стольких высот строки от оси.
GLYPH_BAND_HEIGHTS = 0.7


@dataclass(frozen=True)
class SharedPair:
    """Находка: две оси одного ряда, перекрытые по x на общем глифе.

    Args:
        left, right: Номера осей в списке ``axes`` полосы (левая и правая).
        x0, x1: Перекрытие по x.
        y: Высота ряда на перекрытии (средняя двух осей).
        dy: Наибольшее расхождение осей на перекрытии.
        marks: Общий глиф — низкая метка (точка отточия, запятая), а не буква.
        tail: Длина хвоста левой оси без букв перед перекрытием (от конца последней буквы, не метки,
            до начала перекрытия), в высотах строки; ``None`` — боксов глифов нет. Хвост от
            ``LEADER_TAIL_HEIGHTS`` — ось продлена по отточию (:func:`is_leader`).
    """

    left: int
    right: int
    x0: float
    x1: float
    y: float
    dy: float
    marks: bool
    tail: float | None = None

    def is_leader(self) -> bool:
        """Сошлись ли оси на отточии: левая дошла до перекрытия пустым хвостом (продлением по точкам)."""
        return self.tail is not None and self.tail >= LEADER_TAIL_HEIGHTS

    def to_json(self) -> dict:
        """Словарь для JSON и CSV (числа округлены)."""
        out = asdict(self)
        for field in ("x0", "x1", "y", "dy"):
            out[field] = round(float(out[field]), 1)
        if out["tail"] is not None:
            out["tail"] = round(float(out["tail"]), 2)
        return out


def _shared_glyph(axis: dict, other: dict, x0: float, x1: float) -> tuple[bool, bool]:
    """Лежит ли в перекрытии ``[x0, x1]`` глиф оси ``axis``, над которым проходит ось ``other``.

    Args:
        axis: Ось, чьи глифы проверяются (словарь JSON: ``points``, ``height``, ``mark_spans``, ``glyphs``).
        other: Вторая ось пары.
        x0, x1: Перекрытие осей по x.

    Returns:
        Пара ``(есть общий глиф, этот глиф — низкая метка)``.
    """
    points = np.asarray(other["points"], dtype=np.float64)
    band = GLYPH_BAND_HEIGHTS * float(other["height"])
    # Низкие метки (точки отточия, запятые) — по ``mark_spans``: их середина внутри перекрытия.
    for start, stop in axis.get("mark_spans", []):
        if x0 <= (start + stop) / 2.0 <= x1:
            return True, True
    # Боксы глифов есть только в JSON этого стенда: середина бокса под второй осью.
    for gx0, gy0, gx1, gy1 in axis.get("glyphs", []) or []:
        cx, cy = (gx0 + gx1) / 2.0, (gy0 + gy1) / 2.0
        if x0 <= cx <= x1 and abs(cy - float(np.interp(cx, points[:, 0], points[:, 1]))) <= band:
            return True, False
    return False, False


def _tail(axis: dict, x0: float) -> float | None:
    """Хвост оси без букв перед ``x0`` в высотах строки: от конца последней буквы (не низкой метки) левее ``x0``.

    Args:
        axis: Ось JSON с ``glyphs`` и ``height``.
        x0: Начало перекрытия.

    Returns:
        Длина хвоста в высотах строки; ``None`` — боксов глифов нет.
    """
    glyphs = axis.get("glyphs")
    if not glyphs:
        return None
    boxes = np.asarray(glyphs, dtype=np.float64)
    heights = boxes[:, 3] - boxes[:, 1]
    # Буква — выше половины медианной высоты глифа оси: точки и запятые ниже.
    letters = boxes[(heights > 0.5 * np.median(heights)) & (boxes[:, 2] <= x0)]
    if letters.shape[0] == 0:
        return None
    height = max(float(axis["height"]), 1.0)
    return float(x0 - letters[:, 2].max()) / height


def shared_pairs(axes: list[dict], dpi: float = WORK_DPI) -> list[SharedPair]:
    """Пары осей полосы, сошедшиеся на общем глифе и не сросшиеся.

    Args:
        axes: Оси полосы — словари JSON разбора: ``points`` (N, 2), ``height``, ``mark_spans`` и,
            если есть, ``glyphs`` (боксы ``x0, y0, x1, y1``).
        dpi: Разрешение рабочей копии, в которой даны оси.

    Returns:
        Находки :class:`SharedPair` в порядке номеров осей.
    """
    curves = [np.asarray(axis["points"], dtype=np.float64) for axis in axes]
    out: list[SharedPair] = []
    for first in range(len(axes)):
        for second in range(first + 1, len(axes)):
            a, b = curves[first], curves[second]
            if a.shape[0] < 2 or b.shape[0] < 2:
                continue
            # Левая ось пары — та, что начинается левее.
            left, right = (first, second) if a[0, 0] <= b[0, 0] else (second, first)
            pl, pr = curves[left], curves[right]
            x0, x1 = float(pr[0, 0]), float(min(pl[-1, 0], pr[-1, 0]))
            if x1 - x0 <= 1.0:
                continue
            shorter = min(pl[-1, 0] - pl[0, 0], pr[-1, 0] - pr[0, 0])
            if x1 - x0 > MAX_OVERLAP_SHARE * shorter:
                continue
            # Высота на перекрытии: обе оси в одних и тех же точках.
            xs = np.linspace(x0, x1, DY_SAMPLES)
            ya = np.interp(xs, pl[:, 0], pl[:, 1])
            yb = np.interp(xs, pr[:, 0], pr[:, 1])
            dy = float(np.abs(ya - yb).max())
            if dy >= mm_to_px(MAX_DY_MM, dpi):
                continue
            # Общий глиф — у любой из двух осей.
            found, marks = _shared_glyph(axes[right], axes[left], x0, x1)
            if not found:
                found, marks = _shared_glyph(axes[left], axes[right], x0, x1)
            if not found:
                continue
            out.append(
                SharedPair(
                    left=left,
                    right=right,
                    x0=x0,
                    x1=x1,
                    y=float((ya.mean() + yb.mean()) / 2.0),
                    dy=dy,
                    marks=marks,
                    tail=_tail(axes[left], x0),
                )
            )
    return out


__all__ = ["SharedPair", "shared_pairs"]
