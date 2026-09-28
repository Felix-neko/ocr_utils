"""Классические правила «знак или мусор» для крайнего компонента строки: шаблоны знаков и признаки соринки.

Компонент меряется относительно своей строки (:class:`filters.EndComponent`: размеры в высотах строчной ``x_h``,
низ — от базовой линии, верх — от верха строчных). Сначала он примеряется к шаблонам настоящих знаков,
которые бывают у края строки: буква (или цифра, скобка, кавычка), точка, запятая, дефис, надстрочный знак
сноски. Подошёл — знак. Не подошёл — мелкий компонент считается соринкой, а длинный тонкий штрих — пометкой
(карандаш, отчёркивание). Всё прочее — «спорно»: такие компоненты по желанию отдаются судье
(:mod:`judge`) или остаются.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from research.text_block_specks.filters import End, EndComponent


class Kind(str, Enum):
    """Вердикт правила по компоненту."""

    LETTER = "letter"  # буква, цифра, скобка, кавычка — всё, что с x-height и выше
    PERIOD = "period"
    COMMA = "comma"
    HYPHEN = "hyphen"
    SUPERSCRIPT = "superscript"  # знак сноски над строкой
    SPECK = "speck"  # мелкий мусор
    MARK = "mark"  # тонкий длинный штрих: карандаш, отчёркивание
    UNSURE = "unsure"


# Настоящие знаки — это то, что правило оставляет.
SIGNS = {Kind.LETTER, Kind.PERIOD, Kind.COMMA, Kind.HYPHEN, Kind.SUPERSCRIPT}


@dataclass(frozen=True)
class Thresholds:
    """Пороги правил (линейные — в ``x_h``, площадь — в ``x_h²``).

    Attributes:
        letter_h: Буква — не ниже стольких ``x_h``…
        letter_area: …с площадью краски не меньше…
        letter_fill: …и заполнением бокса не меньше.
        dot_size: Точка — ширина и высота в этих пределах…
        dot_fill: …заполнение не меньше…
        dot_base: …низ не дальше стольких ``x_h`` от базовой линии.
        comma_h: Запятая — высота в пределах…
        comma_w: …ширина не больше…
        comma_bottom: …низ ниже базовой линии в этих пределах…
        comma_fill: …заполнение не меньше.
        hyphen_h: Дефис — высота не больше…
        hyphen_w: …ширина в пределах…
        hyphen_mid: …середина по высоте в этом поясе (доля ``x_h`` вверх от базовой линии)…
        hyphen_fill: …заполнение не меньше.
        sup_h: Надстрочный знак — высота в пределах…
        sup_top: …верх выше верха строчных не меньше чем на столько…
        sup_fill: …заполнение не меньше…
        sup_w: …ширина не меньше (цифра, а не штрих)…
        sup_stroke: …и штрих не тоньше этой доли от штриха букв строки.
        sign_gap: Знак стоит у слова: зазор не больше стольких ``x_h`` (у надстрочного — тоже).
        speck_area: Соринка — не подошла к шаблонам и площадь меньше…
        mark_len: Пометка — длина (большая сторона бокса) не меньше стольких ``x_h``…
        mark_stroke: …и штрих тоньше этой доли от штриха букв строки, или заполнение ниже ``mark_fill``.
        mark_fill: См. ``mark_stroke``.
        lone_gap: Крупный глиф за пробелом шире стольких ``x_h`` — «спорно» (висячий предлог или клякса).
        stack_overlap: Верхняя точка — бокс заходит на предыдущий знак по x не меньше чем на столько ``x_h``…
        stack_size: …ширина и высота в этих пределах…
        stack_lift: …низ выше базовой линии не меньше чем на столько.
        dash_gap: Тире — зазор до слова не больше стольких ``x_h``…
        dash_w: …ширина не меньше…
        dash_fill: …заполнение не меньше.
        stick_h: Палочка — буква, если не ниже стольких ``x_h``…
        stick_fill: …и заполнение не меньше.
        bar_gap: Брусок-дефис и пятно — зазор до слова не больше…
        bar_w: …ширина бруска в этих пределах (до длины тире; у строки из прописных ``x_h`` — высота прописной,
            и дефис относительно мелкий)…
        bar_aspect: …брусок шире своей высоты во столько раз…
        bar_mid: …середина в этом поясе (``x_h`` вверх от базовой линии)…
        bar_fill: …заполнение не меньше.
        blob_size: Пятно — ширина и высота в этих пределах…
        blob_fill: …заполнение не меньше.
    """

    letter_h: float = 0.62
    letter_area: float = 0.18
    letter_fill: float = 0.28
    dot_size: tuple[float, float] = (0.15, 0.6)
    dot_fill: float = 0.5
    dot_base: float = 0.25
    comma_h: tuple[float, float] = (0.3, 0.85)
    comma_w: float = 0.8
    comma_bottom: tuple[float, float] = (0.1, 0.65)
    comma_fill: float = 0.3
    hyphen_h: float = 0.55
    hyphen_w: tuple[float, float] = (0.3, 1.4)
    hyphen_mid: tuple[float, float] = (0.2, 1.1)
    hyphen_fill: float = 0.55
    sup_h: tuple[float, float] = (0.4, 0.9)
    sup_w: float = 0.3
    sup_stroke: float = 0.6
    sup_top: float = 0.2
    sup_fill: float = 0.3
    sign_gap: float = 0.7
    speck_area: float = 0.4
    mark_len: float = 1.6
    mark_stroke: float = 0.7
    mark_fill: float = 0.2
    lone_gap: float = 1.0
    dash_gap: float = 2.0
    stack_overlap: float = 0.15
    stack_size: tuple[float, float] = (0.2, 0.6)
    stack_lift: float = 0.4
    dash_w: float = 1.0
    dash_fill: float = 0.8
    stick_h: float = 0.8
    stick_fill: float = 0.6
    bar_gap: float = 0.3
    bar_w: tuple[float, float] = (0.2, 3.0)
    bar_aspect: float = 1.2
    bar_mid: tuple[float, float] = (-0.4, 1.3)
    bar_fill: float = 0.4
    blob_size: tuple[float, float] = (0.3, 0.75)
    blob_fill: float = 0.75


def kind_of(c: EndComponent, end: End, t: Thresholds = Thresholds()) -> Kind:
    """Вердикт по крайнему компоненту строки.

    Args:
        c: Признаки компонента.
        end: У какого конца строки он стоит (у левого нет запятой и дефиса-переноса).
        t: Пороги.

    Returns:
        :class:`Kind`.
    """
    # Середина компонента по высоте, в x_h вверх от базовой линии.
    mid = -(c.bottom - c.h / 2.0)
    length = max(c.w, c.h)
    # Пометка: длинный тонкий или разреженный штрих — раньше буквы, у неё тоже большая высота.
    if length >= t.mark_len and (c.stroke < t.mark_stroke or c.fill < t.mark_fill):
        return Kind.MARK
    if c.h >= t.letter_h and c.area >= t.letter_area and c.fill >= t.letter_fill:
        return Kind.LETTER
    # Палочка во всю высоту строчной, сплошная: «ы», «і», «1», «!», распавшиеся на куски жирного шрифта.
    if c.h >= t.stick_h and c.fill >= t.stick_fill:
        return Kind.LETTER
    near = c.gap <= t.sign_gap
    # Точка над точкой: верхняя точка двоеточия, точки с запятой, «ё», «!» — её бокс по x накрывает предыдущий
    # знак (зазор отрицательный), и сама она плотная и мелкая.
    if (
        c.gap <= -t.stack_overlap
        and t.stack_size[0] <= c.w <= t.stack_size[1]
        and t.stack_size[0] <= c.h <= t.stack_size[1]
        and c.bottom <= -t.stack_lift
        and c.fill >= t.dot_fill
    ):
        return Kind.PERIOD
    # Тире за пробелом: длинный плотный брусок в поясе строчных.
    if c.gap <= t.dash_gap and c.w >= t.dash_w and c.h <= t.hyphen_h and c.fill >= t.dash_fill and -0.4 <= mid <= 1.3:
        return Kind.HYPHEN
    # Одиночный крупный глиф за широким пробелом: висячий предлог «в», «и», «с» — или клякса размером
    # с букву. По форме их не различить — это спорный случай для судьи.
    if c.gap >= t.lone_gap and c.h >= t.letter_h:
        return Kind.UNSURE
    lo, hi = t.dot_size
    if near and lo <= c.w <= hi and lo <= c.h <= hi and c.fill >= t.dot_fill and abs(c.bottom) <= t.dot_base:
        return Kind.PERIOD
    if (
        near
        and end is End.RIGHT
        and t.comma_h[0] <= c.h <= t.comma_h[1]
        and c.w <= t.comma_w
        and t.comma_bottom[0] <= c.bottom <= t.comma_bottom[1]
        and c.fill >= t.comma_fill
    ):
        return Kind.COMMA
    if (
        near
        and c.h <= t.hyphen_h
        and t.hyphen_w[0] <= c.w <= t.hyphen_w[1]
        and t.hyphen_mid[0] <= mid <= t.hyphen_mid[1]
        and c.fill >= t.hyphen_fill
    ):
        return Kind.HYPHEN
    # Горизонтальный брусок вплотную к слову где угодно по высоте строки — дефис (у изогнутой строки и
    # у шрифтов с высоким или низким дефисом середина уходит из пояса ``hyphen_mid``).
    if (
        c.gap <= t.bar_gap
        and c.h <= t.hyphen_h
        and c.w >= t.bar_aspect * c.h
        and t.bar_w[0] <= c.w <= t.bar_w[1]
        and t.bar_mid[0] <= mid <= t.bar_mid[1]
        and c.fill >= t.bar_fill
    ):
        return Kind.HYPHEN
    if (
        near
        and t.sup_h[0] <= c.h <= t.sup_h[1]
        and c.w >= t.sup_w
        and c.top <= -t.sup_top
        and c.fill >= t.sup_fill
        and c.stroke >= t.sup_stroke
    ):
        return Kind.SUPERSCRIPT
    # Компактное сплошное пятно вплотную к слову не на базовой линии: квадратный дефис жирного заголовка
    # или клякса-точка — по форме не различить, спорно.
    if (
        c.gap <= t.bar_gap
        and t.blob_size[0] <= c.w <= t.blob_size[1]
        and t.blob_size[0] <= c.h <= t.blob_size[1]
        and c.fill >= t.blob_fill
    ):
        return Kind.UNSURE
    if c.area < t.speck_area:
        return Kind.SPECK
    return Kind.UNSURE


@dataclass(frozen=True)
class TemplateRule:
    """Правило для :class:`filters.SpeckPatch`: мусор — соринка и пометка (и «спорно», если ``unsure_is_noise``).

    Attributes:
        name: Имя варианта (подпись прогона).
        thresholds: Пороги.
        marks: Срезать ли пометки (``Kind.MARK``).
        unsure_is_noise: Считать ли «спорно» мусором.
    """

    name: str = "templates"
    thresholds: Thresholds = Thresholds()
    marks: bool = True
    unsure_is_noise: bool = False

    def __call__(self, component: EndComponent, end: End, inner: EndComponent | None = None) -> bool:
        kind = kind_of(component, end, self.thresholds)
        # Второй знак подряд за дефисом или точкой (дефис-дефис, дефис-точка, точка-дефис), стоящий правее, —
        # соринка, похожая на знак: в наборе таких пар у края строки не бывает.
        # Пары, стоящие друг над другом (точка с запятой, двоеточие), и «.,» после сокращения — настоящие.
        if inner is not None and kind in (Kind.HYPHEN, Kind.PERIOD) and component.gap > 0:
            if kind_of(inner, end, self.thresholds) in (Kind.HYPHEN, Kind.PERIOD):
                return True
        if kind is Kind.SPECK:
            return True
        if kind is Kind.MARK:
            return self.marks
        if kind is Kind.UNSURE:
            return self.unsure_is_noise
        return False


__all__ = ["Kind", "SIGNS", "TemplateRule", "Thresholds", "kind_of"]
