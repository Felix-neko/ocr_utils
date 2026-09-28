"""Экспериментальные фильтры соринок у края строки: признаки крайнего компонента, правила и подмена функций детектора.

Код ``ocr_utils`` не меняется: :class:`SpeckPatch` на время разбора подменяет ``segments_of`` в движке
``ink`` (срезает с концов сегмента компоненты, которые правило признало мусором) и ``text_ink`` в
``page`` (гасит краску срезанных компонентов, чтобы край ряда и продление второй оси не нашли их снова).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

import cv2
import numpy as np

import ocr_utils.page_layout.text_blocks.engines.ink as ink_module
import ocr_utils.page_layout.text_blocks.page as page_module
from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks.baseline_axis import line_x_height
from ocr_utils.page_layout.text_blocks.segment import Segment

# Высота строчной считается по строке, если у неё не меньше стольких глифов (иначе признаки не считаются).
MIN_LINE_GLYPHS = 4
# Глиф строчного класса — высота в пределах этой доли от высоты строчной (как ``baseline_axis.X_CLASS_TOLERANCE``).
X_CLASS_TOLERANCE = 0.22
# Базовая линия у конца строки — по стольким ближайшим к концу глифам.
LOCAL_GLYPHS = 10
# Сколько компонентов подряд можно срезать с одного конца сегмента.
MAX_TRIM = 2


class End(str, Enum):
    """Конец строки."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class EndComponent:
    """Крайний компонент строки и его признаки (линейные — в высотах строчной ``x_h``, площадь — в ``x_h²``).

    Attributes:
        box: Бокс ``x0, y0, x1, y1`` (пиксели рабочей копии).
        x_h: Высота строчной строки, пиксели.
        w, h: Ширина и высота бокса.
        area: Площадь краски в боксе.
        fill: Заполнение бокса краской.
        gap: Зазор до соседнего глифа со стороны текста.
        bottom: Низ компонента относительно базовой линии (ниже — плюс).
        top: Верх компонента относительно верха строчных (выше — минус).
        stroke: Толщина штриха (удвоенный максимум distance transform) относительно медианы по строке.
    """

    box: tuple[float, float, float, float]
    x_h: float
    w: float
    h: float
    area: float
    fill: float
    gap: float
    bottom: float
    top: float
    stroke: float


def stroke_width(binary: np.ndarray, box: np.ndarray) -> float:
    """Толщина штриха компонента в боксе: удвоенный максимум расстояния до фона, пиксели.

    Args:
        binary: Бинарная рабочая копия (краска — ``True``).
        box: Бокс ``x0, y0, x1, y1``.

    Returns:
        Толщина; 0.0 для пустого бокса.
    """
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    crop = binary[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)].astype(np.uint8)
    if crop.size == 0 or not crop.any():
        return 0.0
    # Поле в пиксель, чтобы краска на краю бокса мерилась до фона, а не до края массива.
    padded = cv2.copyMakeBorder(crop, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    return 2.0 * float(cv2.distanceTransform(padded, cv2.DIST_L2, 3).max())


def end_component(glyphs: np.ndarray, binary: np.ndarray, end: End) -> EndComponent | None:
    """Признаки крайнего глифа строки у конца ``end``.

    Args:
        glyphs: Боксы глифов строки ``(n, 4)``.
        binary: Бинарная рабочая копия.
        end: Какой конец.

    Returns:
        :class:`EndComponent` или ``None``, если глифов меньше ``MIN_LINE_GLYPHS``.
    """
    if glyphs.shape[0] < MIN_LINE_GLYPHS:
        return None
    x_h = line_x_height(glyphs)
    if x_h <= 0:
        return None
    right = end is End.RIGHT
    order = np.argsort(-glyphs[:, 2] if right else glyphs[:, 0])
    last, rest = glyphs[order[0]], glyphs[order[1:]]
    # Зазор до ближайшего глифа со стороны текста (отрицательный — боксы перекрываются по x).
    gap = last[0] - rest[:, 2].max() if right else rest[:, 0].min() - last[2]
    # Базовая линия и верх строчных — медианы низов и верхов глифов строчного класса У ЭТОГО КОНЦА строки:
    # на изогнутой полосе (nogeo) медиана по всей строке расходится с базовой линией у края на полвысоты
    # строчной, и дефис выглядел «ниже строки».
    local = glyphs[order[1 : 1 + LOCAL_GLYPHS]]
    heights = local[:, 3] - local[:, 1]
    x_class = np.abs(heights - x_h) <= X_CLASS_TOLERANCE * x_h
    base = float(np.median(local[x_class, 3])) if x_class.any() else float(np.median(local[:, 3]))
    top = float(np.median(local[x_class, 1])) if x_class.any() else base - x_h
    x0, y0, x1, y1 = (int(round(v)) for v in last)
    area = float(binary[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)].sum())
    w, h = float(last[2] - last[0]), float(last[3] - last[1])
    # Толщина штриха строки — медиана по глифам строчного класса у конца (или по всем глифам у конца).
    sample = local[x_class] if x_class.sum() >= 3 else local
    line_stroke = float(np.median([stroke_width(binary, box) for box in sample])) or 1.0
    return EndComponent(
        box=tuple(float(v) for v in last),  # type: ignore[arg-type]
        x_h=x_h,
        w=w / x_h,
        h=h / x_h,
        area=area / x_h**2,
        fill=area / max(1.0, w * h),
        gap=float(gap) / x_h,
        bottom=(last[3] - base) / x_h,
        top=(last[1] - top) / x_h,
        stroke=stroke_width(binary, last) / line_stroke,
    )


class Rule(Protocol):
    """Правило: признать крайний компонент мусором (``True``) или оставить (``False``).

    ``inner`` — признаки соседнего с ним компонента со стороны текста (следующий крайний, если убрать этот).
    """

    name: str

    def __call__(self, component: EndComponent, end: End, inner: EndComponent | None = None) -> bool: ...


def inner_component(glyphs: np.ndarray, box: np.ndarray, binary: np.ndarray, end: End) -> EndComponent | None:
    """Признаки соседа крайнего компонента со стороны текста: крайний компонент строки без ``box``."""
    rest = glyphs[~np.all(glyphs == np.asarray(box), axis=1)]
    return end_component(rest, binary, end)


@dataclass(frozen=True)
class Decision:
    """Решение правила по одному компоненту (для журнала и отсмотра).

    Attributes:
        box: Бокс компонента (пиксели рабочей копии).
        end: У какого конца строки.
        component: Признаки.
        noise: Признан мусором и убран.
        source: ``segment`` — крайний глиф сегмента, ``ink`` — бесхозная краска у конца строки.
    """

    box: tuple[float, float, float, float]
    end: End
    component: EndComponent
    noise: bool
    source: str


def trim_segment(
    segment: Segment, binary: np.ndarray, rule: Rule
) -> tuple[Segment, list[tuple[float, float, float, float]], list[Decision]]:
    """Сегмент без крайних компонентов, которые правило признало мусором.

    С каждого конца срезается до ``MAX_TRIM`` компонентов подряд. У сегмента обрезаются бокс по x, точки
    центр-линии, отрезки низких меток и глифы.

    Args:
        segment: Сегмент строки.
        binary: Бинарная рабочая копия.
        rule: Правило мусора.

    Returns:
        Тройка: сегмент (тот же объект, если ничего не срезано), боксы срезанных компонентов и решения
        правила по всем проверенным крайним компонентам.
    """
    if segment.glyphs is None:
        return segment, [], []
    glyphs = np.asarray(segment.glyphs, dtype=np.float64)
    removed: list[tuple[float, float, float, float]] = []
    decisions: list[Decision] = []
    for end in (End.RIGHT, End.LEFT):
        for _ in range(MAX_TRIM):
            component = end_component(glyphs, binary, end)
            if component is None:
                break
            noise = rule(component, end, inner_component(glyphs, np.array(component.box), binary, end))
            decisions.append(Decision(component.box, end, component, noise, "segment"))
            if not noise:
                break
            removed.append(component.box)
            glyphs = glyphs[~np.all(glyphs == np.array(component.box), axis=1)]
    if not removed:
        return segment, [], decisions
    # Новый бокс по x — по оставшимся глифам; центр-линия и метки — в его пределах.
    x0, x1 = float(glyphs[:, 0].min()), float(glyphs[:, 2].max())
    keep = (segment.xs >= x0 - 0.5) & (segment.xs <= x1 + 0.5)
    if keep.sum() < 2:
        return segment, [], decisions
    marks = tuple((a, b) for a, b in segment.mark_spans if b > x0 and a < x1)
    trimmed = replace(
        segment,
        x0=int(np.floor(x0)),
        x1=int(np.ceil(x1)),
        xs=segment.xs[keep],
        ys=segment.ys[keep],
        weights=segment.weights[keep],
        mark_spans=marks,
        glyphs=glyphs,
    )
    return trimmed, removed, decisions


# Бесхозная краска у конца строки ищется не дальше стольких мм за крайним глифом (как ``blocks.AXIS_REACH_MM``).
ORPHAN_REACH_MM = 6.0
# …и по вертикали — в полосе сегмента, раздутой на эту долю его высоты.
ORPHAN_BAND = 0.3
# Хвост за крайним глифом короче стольких пикселей рабочей копии не проверяется (антиалиас, засечка).
TAIL_MIN_PX = 2.0
# Компоненты больше стольких высот строки (линейки, рамки) бесхозной краской не считаются.
ORPHAN_MAX_HEIGHTS = 4.0


def orphan_decisions(
    ink: np.ndarray, segments: list[Segment], binary: np.ndarray, rule: Rule, k: float, dpi: float
) -> tuple[np.ndarray, list[Decision]]:
    """Краска текста без бесхозных компонентов-мусора у концов строк.

    Бесхозный компонент — связная краска ``text_ink``, лежащая вне x-диапазона всех строк, но в полосе
    строки и не дальше ``ORPHAN_REACH_MM`` от её крайнего глифа: ровно такую краску подбирает край ряда
    (``blocks.ink_edge``), и соринка, не попавшая в глифы, всё равно тянула край. Компоненты у конца
    строки проверяются по порядку удаления; признанный знаком становится новым краем строки.

    Args:
        ink: Краска текста ``RENDER_DPI`` (``bool``).
        segments: Сегменты страницы (уже со срезанными концами).
        binary: Бинарная рабочая копия.
        rule: Правило мусора.
        k: Во сколько раз рендер крупнее рабочей копии.
        dpi: Разрешение рабочей копии.

    Returns:
        Пара: краска без мусора и решения по всем проверенным компонентам.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    if count <= 1:
        return ink, []
    reach = ORPHAN_REACH_MM / 25.4 * dpi
    boxes = np.column_stack(
        [
            stats[:, cv2.CC_STAT_LEFT] / k,
            stats[:, cv2.CC_STAT_TOP] / k,
            (stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH]) / k,
            (stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT]) / k,
        ]
    )
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0
    heights = boxes[:, 3] - boxes[:, 1]
    # Каждый компонент приписывается ОДНОЙ строке — ближайшей среди тех, чья полоса его накрывает и до
    # чьего x-диапазона (с запасом ``reach``) он дотягивается. Расстояние — пробел по x до глифов строки
    # плюс расстояние по y до её середины. Полосы соседних строк перекрываются, и без этого соринка за
    # концом короткой строки считалась «внутри» соседней, более длинной строки; а только по y дефис
    # конца строки левой колонки доставался строке правой колонки как «краска у её левого конца».
    owner = np.full(count, -1)
    best = np.full(count, np.inf)
    for index, segment in enumerate(segments):
        pad = ORPHAN_BAND * segment.height
        covered = (cy >= segment.y0 - pad) & (cy <= segment.y1 + pad)
        covered &= (cx >= segment.x0 - reach) & (cx <= segment.x1 + reach)
        gap_x = np.maximum(0.0, np.maximum(segment.x0 - boxes[:, 2], boxes[:, 0] - segment.x1))
        distance = gap_x + np.abs(cy - segment.cy)
        closer = covered & (distance < best)
        owner[closer], best[closer] = index, distance[closer]
    owner[0] = -1
    drop = np.zeros(count, dtype=bool)
    # Хвосты: краска компонентов, выходящая за крайний глиф строки и признанная мусором (маска ``RENDER_DPI``).
    cut = np.zeros(ink.shape, dtype=bool)
    decisions: list[Decision] = []
    for index, segment in enumerate(segments):
        if segment.glyphs is None or len(segment.glyphs) < MIN_LINE_GLYPHS:
            continue
        mine = (owner == index) & (heights <= ORPHAN_MAX_HEIGHTS * segment.height)
        glyph_x0, glyph_x1 = float(segment.glyphs[:, 0].min()), float(segment.glyphs[:, 2].max())
        for end in (End.RIGHT, End.LEFT):
            glyphs = np.asarray(segment.glyphs, dtype=np.float64)
            # Хвост: компонент начинается внутри строки, а кончается за её крайним глифом больше чем на
            # ``TAIL_MIN_PX`` — соринка или пометка, слипшаяся в краске с последним знаком.
            if end is End.RIGHT:
                tails = mine & (boxes[:, 0] < glyph_x1 - 1) & (boxes[:, 2] > glyph_x1 + TAIL_MIN_PX)
            else:
                tails = mine & (boxes[:, 2] > glyph_x0 + 1) & (boxes[:, 0] < glyph_x0 - TAIL_MIN_PX)
            for component_index in np.flatnonzero(tails):
                tail = _tail_mask(labels, stats, component_index, glyph_x1 if end is End.RIGHT else glyph_x0, end, k)
                if tail is None:
                    continue
                local, (top, left), box = tail
                trial = np.vstack([glyphs, box])
                component = end_component(trial, binary, end)
                if component is None:
                    continue
                noise = rule(component, end, inner_component(trial, box, binary, end))
                decisions.append(Decision(tuple(float(v) for v in box), end, component, noise, "tail"))  # type: ignore[arg-type]
                if noise:
                    cut[top : top + local.shape[0], left : left + local.shape[1]] |= local
            # Бесхозный — за крайним глифом строки (внутри x-диапазона строки лежат её буквы и отточия).
            if end is End.RIGHT:
                near = mine & (boxes[:, 0] >= glyph_x1 - 1) & (cx > segment.x1 - 1)
                order = np.argsort(boxes[:, 0])
            else:
                near = mine & (boxes[:, 2] <= glyph_x0 + 1) & (cx < segment.x0 + 1)
                order = np.argsort(-boxes[:, 2])
            for component_index in order:
                if not near[component_index] or drop[component_index]:
                    continue
                trial = np.vstack([glyphs, boxes[component_index]])
                component = end_component(trial, binary, end)
                if component is None:
                    break
                noise = rule(component, end, inner_component(trial, boxes[component_index], binary, end))
                box = tuple(float(v) for v in boxes[component_index])
                decisions.append(Decision(box, end, component, noise, "ink"))  # type: ignore[arg-type]
                if noise:
                    drop[component_index] = True
                else:
                    # Знак у конца строки: дальше край строки — по нему.
                    glyphs = trial
    ink = ink & ~cut
    if not drop.any():
        return ink, decisions
    return ink & ~drop[labels], decisions


def _tail_mask(
    labels: np.ndarray, stats: np.ndarray, index: int, edge: float, end: End, k: float
) -> tuple[np.ndarray, tuple[int, int], np.ndarray] | None:
    """Часть компонента ``index`` за краем строки ``edge``: локальная маска ``RENDER_DPI`` и бокс в пикселях рабочей копии.

    Args:
        labels: Метки компонент краски ``RENDER_DPI``.
        stats: Статистика компонент (``connectedComponentsWithStats``).
        index: Номер компонента.
        edge: Край крайнего глифа строки (пиксели рабочей копии).
        end: Какой конец строки.
        k: Во сколько раз рендер крупнее рабочей копии.

    Returns:
        Тройка «маска хвоста в боксе компонента, (верх, лево) этого бокса на рендере, бокс хвоста» или
        ``None``, если хвост пуст.
    """
    x, y, w, h = (int(v) for v in stats[index, :4])
    local = labels[y : y + h, x : x + w] == index
    # Столбцы за краем глифа с запасом в пиксель рабочей копии.
    cols = np.arange(x, x + w)
    beyond = cols >= (edge + 1) * k if end is End.RIGHT else cols <= (edge - 1) * k
    local = local & beyond[None, :]
    if not local.any():
        return None
    ys, xs = np.nonzero(local)
    box = np.array([(x + xs.min()) / k, (y + ys.min()) / k, (x + xs.max() + 1) / k, (y + ys.max() + 1) / k])
    return local, (y, x), box


def work_binary(gray300: np.ndarray, dpi: float = WORK_DPI) -> np.ndarray:
    """Бинарная рабочая копия, как у сегментации (INTER_AREA, порог Оцу): краска — ``True``."""
    size = (max(1, round(gray300.shape[1] * dpi / RENDER_DPI)), max(1, round(gray300.shape[0] * dpi / RENDER_DPI)))
    work = cv2.resize(gray300, size, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary > 0


class SpeckPatch:
    """Подмена ``segments_of`` движка ``ink`` и ``text_ink`` страницы на время разбора (контекстный менеджер).

    Разбор страниц в одном процессе идёт по очереди, поэтому сегменты и срезанные боксы последней
    сегментации хранятся в самом объекте и используются следующим вызовом ``text_ink``.

    Attributes:
        rule: Правило мусора; ``None`` — ничего не меняется (прогон «до»).
        orphans: Чистить ли бесхозную краску у концов строк (:func:`orphan_decisions`).
        removed: Срезанные с сегментов боксы последней сегментации (пиксели рабочей копии).
        decisions: Решения правила по всем проверенным компонентам последнего разбора.
    """

    def __init__(self, rule: Rule | None, orphans: bool = True) -> None:
        self.rule = rule
        self.orphans = orphans
        self.removed: list[tuple[float, float, float, float]] = []
        self.decisions: list[Decision] = []
        self._segments: list[Segment] = []
        self._segments_of = ink_module.segments_of
        self._text_ink = page_module.text_ink

    def segments_of(self, gray300: np.ndarray, separators, dpi: float = WORK_DPI, *args, **kwargs):
        """Сегменты исходной функции со срезанными концами (сигнатура — как у ``segment.segments_of``)."""
        segments, rules = self._segments_of(gray300, separators, dpi, *args, **kwargs)
        self.removed, self.decisions = [], []
        if self.rule is None:
            self._segments = list(segments)
            return segments, rules
        binary = work_binary(gray300, dpi)
        out = []
        for segment in segments:
            trimmed, removed, decisions = trim_segment(segment, binary, self.rule)
            out.append(trimmed)
            self.removed.extend(removed)
            self.decisions.extend(decisions)
        self._segments = out
        return out, rules

    def text_ink(self, gray300: np.ndarray, dpi: float = WORK_DPI, work=None, leaders=None) -> np.ndarray:
        """Краска текста исходной функции без срезанных компонентов и без бесхозного мусора у концов строк."""
        ink = self._text_ink(gray300, dpi, work=work, leaders=leaders)
        if self.rule is None:
            return ink
        k = RENDER_DPI / dpi
        for x0, y0, x1, y1 in self.removed:
            ink[int(y0 * k) : int(np.ceil(y1 * k)), int(x0 * k) : int(np.ceil(x1 * k))] = False
        if self.orphans:
            ink, decisions = orphan_decisions(ink, self._segments, work_binary(gray300, dpi), self.rule, k, dpi)
            self.decisions.extend(decisions)
        return ink

    def __enter__(self) -> "SpeckPatch":
        ink_module.segments_of = self.segments_of
        page_module.text_ink = self.text_ink
        return self

    def __exit__(self, *exc) -> None:
        ink_module.segments_of = self._segments_of
        page_module.text_ink = self._text_ink


__all__ = [
    "Decision",
    "End",
    "EndComponent",
    "Rule",
    "SpeckPatch",
    "end_component",
    "orphan_decisions",
    "stroke_width",
    "trim_segment",
    "work_binary",
]
