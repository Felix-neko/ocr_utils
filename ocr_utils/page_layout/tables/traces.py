"""Трассы линеек таблицы: осевая кривая каждой линейки сглаживающим сплайном, для наклонных и изогнутых таблиц.

ЗАЧЕМ. ``Segment`` хранит линейку габаритом и одним средним наклоном. На изогнутой полосе этого мало:
у верхней линейки таблицы 1 полосы 1968/03 с.4 (108 мм) наклон по пятым долям длины −1,8 −2,3 −4,8
−4,4 −0,5°, и отрезок «начало–конец» отходит от неё до 8,4 px при 300 dpi (0,7 мм — вдвое толще самой
линейки). Трасса — функция «поперёк от вдоль» (``y(x)`` у горизонтали, ``x(y)`` у вертикали), по
которой известны положение и наклон линейки в любой её точке.

ОТКУДА ПИКСЕЛИ. Из фрагментов 3 мм детектора (``rules.fragment_layers``) и их цепочек
(``rules.chain_groups``): открытие коротким ядром держит линейку одной компонентой во всю длину даже
при локальном наклоне 4,8° (штрих толщиной t px переживает открытие ядром L px, пока t > L·tg θ: при
300 dpi и t = 4 px — до 6,5°). Берутся пиксели ИМЕННО фрагментов цепочки, а не всё в её габарите:
габарит наклонной линейки высотой в сантиметр и накрывает чужую краску.

КАК СТРОИТСЯ КРИВАЯ.

1. По каждому столбцу (строке у вертикали) — пробег краски линейки, ближайший к предсказанию; его
   середина — отсчёт. Пробег толще 1,8 медианной толщины отбрасывается: это пересечение с
   перпендикулярной линейкой или прилипшая буква.
2. Сглаживающий сплайн (``scipy.interpolate.make_smoothing_spline``) с весами Тьюки — две итерации,
   чтобы засечка буквы, коснувшейся линейки в паре столбцов, не гнула кривую.
3. Полоса сглаживания h — НЕ автоматический выбор GCV: он идёт за лесенкой пикселей (наклон скачет
   ±25° на ровной линейке, остаток 0,03 px), шум середины пробега коррелирован и квантован. Берётся
   наибольшая полоса из ``SMOOTH_LADDER_MM``, при которой остаток, усреднённый в окнах 3 мм, не выше
   шума: настоящий изгиб оставляет в остатке системный горб, и лестница спускается к меньшей полосе;
   шум сам по себе спуска не вызывает. На 1968/03 выбирается 3–4 мм: остаток ≤ 1,25 px, профиль
   наклона верхней линейки −1,8 −2,4 −4,6 −4,3 −0,8° против измеренных −1,8 −2,3 −4,8 −4,4 −0,5°.
4. Концы — естественный кубический сплайн (вторая производная ноль), раскачки на концах нет. За
   концами кривая не определена: продолжение по касательной — только явно (``extend_px``).

СИСТЕМА КООРДИНАТ. Кривая строится в пикселях рабочей копии (вырезка таблицы в 300 dpi), наружу
отдаётся в пикселях той картинки, куда её перенесли ``mapped(scale, dx, dy)``: сдвиг и масштаб
оставляют функцию функцией, перестраивать сплайн не нужно. Центр пикселя ``i`` — координата ``i``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

import cv2
import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline, make_smoothing_spline

from ocr_utils.page_layout.tables.rules import (
    CHAIN_GAP_MM,
    CHAIN_OFFSET_MM,
    MAX_ISOLATION,
    ChainedRule,
    FragmentLayer,
    chain_groups,
    fragment_layers,
    isolation,
)
from ocr_utils.page_layout.tables.ruling import MIN_RULE_MM, binarize, mm_to_px

# Лестница полос сглаживания, от грубой к тонкой: берётся наибольшая, при которой остаток не
# превышает шума. 6 мм и выше теряют S-изгиб верхней линейки 1968/03 (остаток до 3,8 px при 8 мм),
# 1,5 мм — предел, ниже которого сплайн начинает повторять лесенку пикселей.
SMOOTH_LADDER_MM = (6.0, 4.0, 3.0, 2.0, 1.5)

# Полоса, на которой меряется шум отсчётов (MAD остатка): 1 мм — тоньше любого изгиба линейки.
NOISE_BANDWIDTH_MM = 1.0

# Окно, в котором усредняется остаток при выборе полосы: 3 мм. Шум в нём гасится (≈35 отсчётов при
# 300 dpi), а горб недосглаженного изгиба длиной от сантиметра — нет.
RESIDUAL_WINDOW_MM = 3.0

# Допуск усреднённого остатка: не ниже этого в пикселях, иначе на идеально ровной линейке шум
# в сотые доли пикселя спускал бы лестницу до самого низа.
RESIDUAL_FLOOR_PX = 0.5

# Нижняя граница оценки шума: квантование середины пробега даёт не меньше 0,15 px.
NOISE_FLOOR_PX = 0.15

# Пробег толще этой доли медианной толщины линейки — пересечение или прилипшая буква, не отсчёт.
THICK_RUN_FACTOR = 1.8

# Итерации перевзвешивания по Тьюки и его порог (в пикселях, но не меньше толщины линейки).
TUKEY_ITERATIONS = 2
TUKEY_MIN_PX = 1.5

# Сколько толщин штриха у каждого конца линейки не идут в подгонку (угол, косой срез штриха).
END_TRIM_FACTOR = 1.5

# Пол веса отсчёта: отброшенный Тьюки отсчёт остаётся в сплайне с этим весом (нулевой scipy не принимает).
MIN_WEIGHT = 1e-3

# Столбцы без краски подряд длиннее этого — разрыв линейки (пунктир, продавленная бумага): 0,5 мм.
GAP_MIN_MM = 0.5

# ПРОДОЛЖЕНИЕ ЗА КОНЦЫ. Фрагмент 3 мм не переживает наклона больше ~6,5° (штрих 4 px) — а у края
# полосы линейка загибается круче: боковая таблица 1976/08 IMG_0067_1L за последней вертикалью уходит
# вниз под 13° на 7 мм, и эти куски не находились вовсе. Трасса продолжается от каждого конца шагом в
# столбец: предсказание по касательной (прямая по последним EXTEND_FIT_MM), в окне ±EXTEND_WINDOW_MM
# берётся тонкий пробег краски. Пересечение с перпендикулярной линейкой (толстый пробег в окне)
# проходится до EXTEND_CROSSING_MM, просвет бумаги — не больше EXTEND_PAPER_GAP_MM: у строки текста
# просветы между буквами шире, и по ней трасса не утечёт. Продолжение короче EXTEND_MIN_MM
# отбрасывается (обрывок буквы у конца линейки).
EXTEND_FIT_MM = 3.0
EXTEND_WINDOW_MM = 0.4
EXTEND_PAPER_GAP_MM = 0.5
# Просвет с пересечением внутри — узел таблицы: линейка обрывается за 1–1,5 мм до вертикали и
# продолжается за ней (1976/08 IMG_0067_1L). Такой просвет проходится целиком до этой длины.
EXTEND_JUNCTION_MM = 3.0
EXTEND_MIN_MM = 1.0
# Кусок продолжения ПОСЛЕ просвета (узла) принимается, только если он сплошной не короче этого: у
# загиба 1976/08 за вертикалью — 7 мм, а штрих буквы над узлом — до 2,5 мм (вертикаль 1967/10 с.33
# уходила через линейку под «Годы» вверх по букве, 1975/07 с.24 — по «кг/мм» шапки).
EXTEND_AFTER_GAP_MM = 3.0
EXTEND_MAX_SLOPE = 1.0  # тангенс 45°: круче — это уже не линейка этой оси

# Шаг ломаной в JSON и расстояние между узлами компактного сплайна в JSON.
SAMPLE_STEP_MM = 1.0
JSON_KNOT_STEP_MM = 5.0


class Axis(StrEnum):
    """Ось линейки-ребра таблицы."""

    HORIZONTAL = "горизонталь"
    VERTICAL = "вертикаль"


@dataclass(frozen=True)
class Frame:
    """Перенос из рабочих пикселей в пиксели картинки: ``наружу = рабочие · scale + (dx, dy)``."""

    scale: float = 1.0
    dx: float = 0.0
    dy: float = 0.0

    def then(self, scale: float, dx: float, dy: float) -> "Frame":
        """Этот перенос, за которым следует ещё один (масштаб ``scale``, сдвиг ``dx, dy``).

        Args:
            scale: Масштаб второго переноса.
            dx: Сдвиг второго переноса по x.
            dy: Сдвиг второго переноса по y.

        Returns:
            Составной перенос.
        """
        return Frame(self.scale * scale, self.dx * scale + dx, self.dy * scale + dy)


@dataclass(frozen=True, eq=False)
class RuleTrace:
    """Линейка таблицы осевой кривой: сглаживающий сплайн «поперёк от вдоль» с отсчётами, толщиной и разрывами.

    «Вдоль» — x у горизонтали и y у вертикали, «поперёк» — наоборот. Отсчёты ``along/across/weights``
    и сплайн — в РАБОЧИХ пикселях, все методы принимают и отдают координаты в пикселях картинки
    ``frame`` (по умолчанию это те же рабочие пиксели).
    """

    axis: Axis
    along: np.ndarray  # вдоль, рабочие пиксели: столбцы (строки), где нашлась краска, — все, и концевые
    across: np.ndarray  # поперёк: середина пробега краски линейки в этом столбце
    weights: np.ndarray  # веса Тьюки последней итерации (0 — отсчёт выброшен: чужая краска или конец линейки)
    spline: BSpline  # сглаживающий сплайн across(along), рабочие пиксели
    start: float  # первый отсчёт вдоль, рабочие пиксели
    end: float  # последний отсчёт вдоль, рабочие пиксели
    thickness_px: float  # медианная толщина штриха, рабочие пиксели
    coverage: float  # доля столбцов между концами, где нашлась краска
    gaps: tuple[tuple[float, float], ...]  # разрывы длиннее GAP_MIN_MM: (начало, конец) вдоль, рабочие
    bandwidth_mm: float  # выбранная полоса сглаживания
    residual_rms_px: float  # остаток отсчётов от сплайна (по отсчётам с весом), рабочие пиксели
    residual_max_px: float
    dpi: int  # разрешение рабочей копии
    frame: Frame = field(default_factory=Frame)

    # --- перенос координат ---------------------------------------------------------

    @property
    def horizontal(self) -> bool:
        return self.axis == Axis.HORIZONTAL

    def _along_shift(self) -> float:
        return self.frame.dx if self.horizontal else self.frame.dy

    def _across_shift(self) -> float:
        return self.frame.dy if self.horizontal else self.frame.dx

    def _to_work(self, along: np.ndarray) -> np.ndarray:
        return (along - self._along_shift()) / self.frame.scale

    def _from_work(self, along: np.ndarray) -> np.ndarray:
        return along * self.frame.scale + self._along_shift()

    @property
    def output_dpi(self) -> float:
        """Разрешение картинки ``frame``: рабочее, умноженное на масштаб."""
        return self.dpi * self.frame.scale

    @property
    def start_out(self) -> float:
        """Начало линейки вдоль, в пикселях картинки."""
        return float(self._from_work(np.asarray(self.start)))

    @property
    def end_out(self) -> float:
        """Конец линейки вдоль, в пикселях картинки."""
        return float(self._from_work(np.asarray(self.end)))

    @property
    def length_mm(self) -> float:
        return (self.end - self.start) * 25.4 / self.dpi

    def mapped(self, scale: float, dx: float, dy: float) -> "RuleTrace":
        """Та же трасса, перенесённая в другую картинку: координаты умножаются на ``scale`` и сдвигаются.

        Args:
            scale: Во сколько раз новая картинка крупнее нынешней ``frame``.
            dx: Сдвиг по x в пикселях новой картинки.
            dy: Сдвиг по y в пикселях новой картинки.

        Returns:
            Трасса с новым переносом; сплайн не перестраивается.
        """
        return replace(self, frame=self.frame.then(scale, dx, dy))

    # --- кривая ----------------------------------------------------------------------

    def across_at(self, along: "float | np.ndarray", extend_px: float = 0.0) -> np.ndarray:
        """Поперечная координата линейки в точках ``along`` (пиксели картинки).

        Args:
            along: Координаты вдоль линейки: число или массив.
            extend_px: На сколько пикселей картинки за концами продолжать линейку по касательной;
                дальше — NaN.

        Returns:
            Массив поперечных координат той же формы (NaN вне линейки и её продолжения).
        """
        values = np.atleast_1d(np.asarray(along, dtype=float))
        work = self._to_work(values)
        reach = extend_px / self.frame.scale
        inside = np.clip(work, self.start, self.end)
        across = self.spline(inside)
        # За концами — по касательной на конце, но не дальше reach.
        slope = self.spline.derivative()(inside)
        across = across + slope * (work - inside)
        across = np.where((work < self.start - reach) | (work > self.end + reach), np.nan, across)
        result = across * self.frame.scale + self._across_shift()
        return result if np.ndim(along) else result[:1]

    def slope_at(self, along: "float | np.ndarray") -> np.ndarray:
        """Наклон линейки «d поперёк / d вдоль» в точках ``along`` (пиксели картинки); масштаб его не меняет.

        Args:
            along: Координаты вдоль линейки.

        Returns:
            Массив наклонов; за концами — наклон на ближнем конце.
        """
        work = np.clip(self._to_work(np.atleast_1d(np.asarray(along, dtype=float))), self.start, self.end)
        return self.spline.derivative()(work)

    def angle_deg_at(self, along: "float | np.ndarray") -> np.ndarray:
        """Наклон в градусах в валюте ``Segment.angle_deg``: на сколько повернуть ПО ЧАСОВОЙ, чтобы выпрямить.

        У горизонтали положительный угол — правый конец ниже левого, у вертикали — нижний конец левее.

        Args:
            along: Координаты вдоль линейки.

        Returns:
            Массив углов в градусах.
        """
        angle = np.degrees(np.arctan(self.slope_at(along)))
        return angle if self.horizontal else -angle

    def point_at(self, along: "float | np.ndarray", extend_px: float = 0.0) -> np.ndarray:
        """Точки линейки ``(x, y)`` в пикселях картинки, массив ``(N, 2)``.

        Args:
            along: Координаты вдоль линейки.
            extend_px: Продолжение за концами по касательной, пиксели картинки.

        Returns:
            Точки ``(x, y)``.
        """
        along = np.atleast_1d(np.asarray(along, dtype=float))
        across = self.across_at(along, extend_px)
        return np.column_stack([along, across] if self.horizontal else [across, along])

    @property
    def chord(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Концы линейки ``((x, y), (x, y))`` — отрезок «начало–конец» в пикселях картинки."""
        ends = self.point_at(np.array([self.start_out, self.end_out]))
        return (float(ends[0, 0]), float(ends[0, 1])), (float(ends[1, 0]), float(ends[1, 1]))

    @property
    def chord_angle_deg(self) -> float:
        """Наклон отрезка «начало–конец» в той же валюте, что ``angle_deg_at``."""
        (x0, y0), (x1, y1) = self.chord
        angle = (
            np.degrees(np.arctan2(y1 - y0, x1 - x0)) if self.horizontal else np.degrees(np.arctan2(x1 - x0, y1 - y0))
        )
        return float(angle if self.horizontal else -angle)

    def along_grid(self, step_px: float = 1.0) -> np.ndarray:
        """Координаты вдоль от начала до конца с шагом ``step_px`` пикселей картинки, оба конца включены."""
        start, end = self.start_out, self.end_out
        count = max(2, int(np.ceil((end - start) / step_px)) + 1)
        return np.linspace(start, end, count)

    def max_chord_deviation_px(self) -> tuple[float, float]:
        """Наибольшее расстояние от кривой до отрезка «начало–конец» и где оно достигается.

        Returns:
            ``(расстояние в пикселях картинки, координата вдоль)``; расстояние — по перпендикуляру к хорде.
        """
        along = self.along_grid(1.0)
        across = self.across_at(along)
        (x0, y0), (x1, y1) = self.chord
        a0, c0, a1, c1 = (x0, y0, x1, y1) if self.horizontal else (y0, x0, y1, x1)
        chord = c0 + (c1 - c0) * (along - a0) / max(a1 - a0, 1e-9)
        cosine = abs(a1 - a0) / max(np.hypot(a1 - a0, c1 - c0), 1e-9)
        distance = np.abs(across - chord) * cosine
        index = int(np.argmax(distance))
        return float(distance[index]), float(along[index])

    def sample(self, step_mm: float = SAMPLE_STEP_MM) -> np.ndarray:
        """Ломаная по кривой с шагом ``step_mm``: точки ``(x, y)`` в пикселях картинки, концы включены."""
        return self.point_at(self.along_grid(step_mm * self.output_dpi / 25.4))

    # --- JSON ----------------------------------------------------------------------

    def to_json(self) -> dict:
        """Трасса в JSON: концы, наклоны, отклонение хорды, ломаная через 1 мм и компактный сплайн.

        Всё — в пикселях картинки ``frame``, длины — в миллиметрах. Компактный сплайн — кубический
        с узлами через ``JSON_KNOT_STEP_MM``, подогнанный к трассе МНК; ``from_json`` восстанавливает
        трассу по нему (без отсчётов).

        Returns:
            Словарь, готовый к ``json.dumps``.
        """
        to_mm = 25.4 / self.output_dpi
        along = self.along_grid(1.0)
        across = self.across_at(along)
        spline = _compact_spline(along, across, JSON_KNOT_STEP_MM / to_mm)
        deviation, where = self.max_chord_deviation_px()
        angles = self.angle_deg_at(along)
        start, end = self.chord
        return {
            "axis": self.axis.value,
            "start": [round(v, 2) for v in start],
            "end": [round(v, 2) for v in end],
            "length_mm": round(self.length_mm, 2),
            "chord_angle_deg": round(self.chord_angle_deg, 3),
            "angle_range_deg": [round(float(angles.min()), 3), round(float(angles.max()), 3)],
            "max_chord_deviation_mm": round(deviation * to_mm, 3),
            "max_chord_deviation_at": round(where, 1),
            "thickness_mm": round(self.thickness_px * 25.4 / self.dpi, 3),
            "coverage": round(self.coverage, 3),
            "gaps_mm": [[round(a * 25.4 / self.dpi, 2), round(b * 25.4 / self.dpi, 2)] for a, b in self.gaps],
            "bandwidth_mm": self.bandwidth_mm,
            "residual_rms_px": round(self.residual_rms_px, 3),
            "residual_max_px": round(self.residual_max_px, 3),
            "dpi": round(self.output_dpi, 3),
            "polyline": [[round(float(x), 2), round(float(y), 2)] for x, y in self.sample()],
            "spline": {"t": spline.t.round(4).tolist(), "c": spline.c.round(4).tolist(), "k": int(spline.k)},
        }

    @staticmethod
    def from_json(payload: dict) -> "RuleTrace":
        """Трасса из ``to_json``: по компактному сплайну, без отсчётов; перенос — тождественный.

        Args:
            payload: Словарь ``to_json``.

        Returns:
            Трасса в пикселях той картинки, в которой её сохраняли.
        """
        axis = Axis(payload["axis"])
        spline = BSpline(np.array(payload["spline"]["t"]), np.array(payload["spline"]["c"]), payload["spline"]["k"])
        index = 0 if axis == Axis.HORIZONTAL else 1
        start, end = float(payload["start"][index]), float(payload["end"][index])
        dpi = float(payload["dpi"])
        to_px = dpi / 25.4
        empty = np.zeros(0)
        return RuleTrace(
            axis=axis,
            along=empty,
            across=empty,
            weights=empty,
            spline=spline,
            start=start,
            end=end,
            thickness_px=payload["thickness_mm"] * to_px,
            coverage=payload["coverage"],
            gaps=tuple((a * to_px, b * to_px) for a, b in payload["gaps_mm"]),
            bandwidth_mm=payload["bandwidth_mm"],
            residual_rms_px=payload["residual_rms_px"],
            residual_max_px=payload["residual_max_px"],
            dpi=int(round(dpi)) if abs(dpi - round(dpi)) < 1e-6 else dpi,
        )


def _compact_spline(along: np.ndarray, across: np.ndarray, knot_step: float) -> BSpline:
    """Кубический МНК-сплайн с узлами через ``knot_step`` — компактная запись кривой для JSON.

    Args:
        along: Координаты вдоль (возрастают, шаг около пикселя).
        across: Значения кривой в них.
        knot_step: Расстояние между внутренними узлами.

    Returns:
        Сплайн на отрезке ``[along[0], along[-1]]``.
    """
    inner = max(0, int((along[-1] - along[0]) // knot_step) - 1)
    interior = np.linspace(along[0], along[-1], inner + 2)[1:-1]
    knots = np.concatenate([[along[0]] * 4, interior, [along[-1]] * 4])
    return make_lsq_spline(along, across, knots, k=3)


# --- отсчёты -----------------------------------------------------------------------


@dataclass(frozen=True)
class _Samples:
    """Отсчёты одной линейки: центры тяжести тёмности пробегов по столбцам, толщина штриха, разрывы."""

    along: np.ndarray
    across: np.ndarray
    runs: np.ndarray  # длина выбранного пробега, пиксели
    gaps: tuple[tuple[float, float], ...]
    coverage: float


def _runs(column: np.ndarray) -> list[tuple[int, int]]:
    """Пробеги ненулевых значений в одномерном массиве: пары ``(начало, конец включительно)``."""
    index = np.flatnonzero(column)
    if index.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(index) > 1)
    starts = np.concatenate([[index[0]], index[breaks + 1]])
    ends = np.concatenate([index[breaks], [index[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def _sample_rule(
    mask: np.ndarray, darkness: np.ndarray, horizontal: bool, offset_along: int, offset_across: int, predicted, dpi: int
) -> _Samples:
    """Отсчёты линейки: по каждому столбцу (строке) — центр тяжести тёмности пробега, ближайшего к предсказанию.

    Центр берётся не как середина пробега бинарной маски, а по тёмности серого (с запасом в пиксель по
    краям пробега): середина бинарного пробега у почти горизонтальной линейки ходит лесенкой ±0,5 px с
    периодом 1/наклон — до сантиметра, — и сплайн рисовал по ней ложную волну наклона ±2°.

    Args:
        mask: Пиксели линейки в её габарите (``bool``), строки — y.
        darkness: Тёмность серого в том же габарите: бумага − яркость, не меньше нуля (float).
        horizontal: Ось линейки.
        offset_along: Координата габарита вдоль (x0 у горизонтали, y0 у вертикали).
        offset_across: Координата габарита поперёк.
        predicted: Функция «вдоль → ожидаемое поперёк» (рабочие пиксели).
        dpi: Разрешение — для длины разрыва.

    Returns:
        Отсчёты; пробеги толще ``THICK_RUN_FACTOR`` медианы уже отброшены.
    """
    lines = mask.T if horizontal else mask  # по строкам этого массива идёт «вдоль»
    dark_lines = darkness.T if horizontal else darkness
    along: list[float] = []
    across: list[float] = []
    runs: list[int] = []
    for index, line in enumerate(lines):
        found = _runs(line)
        if not found:
            continue
        position = offset_along + index
        guess = predicted(position) - offset_across
        low, high = min(found, key=lambda run: abs((run[0] + run[1]) / 2.0 - guess))
        # Центр тяжести тёмности пробега с запасом в пиксель: край штриха серый, и в нём — доли пикселя.
        start, stop = max(0, low - 1), min(line.size - 1, high + 1)
        weights = dark_lines[index, start : stop + 1]
        total = float(weights.sum())
        centre = start + float(np.dot(np.arange(weights.size), weights)) / total if total > 0 else (low + high) / 2.0
        along.append(float(position))
        across.append(offset_across + centre)
        runs.append(high - low + 1)
    along_arr, across_arr, runs_arr = np.array(along), np.array(across), np.array(runs)
    if runs_arr.size:
        # Толстый пробег — пересечение с перпендикулярной линейкой или прилипшая буква.
        keep = runs_arr <= THICK_RUN_FACTOR * max(1.0, float(np.median(runs_arr)))
        along_arr, across_arr, runs_arr = along_arr[keep], across_arr[keep], runs_arr[keep]
    gaps = _gaps(along_arr, mm_to_px(GAP_MIN_MM, dpi))
    span = along_arr[-1] - along_arr[0] + 1 if along_arr.size else 1.0
    return _Samples(along_arr, across_arr, runs_arr, gaps, float(along_arr.size / span))


def _gaps(along: np.ndarray, minimal: int) -> tuple[tuple[float, float], ...]:
    """Разрывы между соседними отсчётами длиннее ``minimal`` пикселей."""
    if along.size < 2:
        return ()
    steps = np.diff(along)
    where = np.flatnonzero(steps > minimal)
    return tuple((float(along[i]), float(along[i + 1])) for i in where)


def _extend_end(
    samples: _Samples, lines: np.ndarray, dark_lines: np.ndarray, direction: int, thickness: float, dpi: int
) -> list[tuple[float, float, int]]:
    """Продолжение линейки за один конец следованием по краске (см. ``EXTEND_*``).

    Args:
        samples: Отсчёты линейки (вдоль возрастает).
        lines: Бинарная краска, индексированная [вдоль, поперёк] (у горизонтали — транспонированная).
        dark_lines: Тёмность серого в той же индексации.
        direction: +1 — за правый (нижний) конец, −1 — за левый (верхний).
        thickness: Толщина штриха линейки, пиксели.
        dpi: Разрешение.

    Returns:
        Новые отсчёты ``(вдоль, поперёк, длина пробега)`` по порядку ухода от конца; пусто, если
        продолжение короче ``EXTEND_MIN_MM``.
    """
    fit = mm_to_px(EXTEND_FIT_MM, dpi)
    window = max(3, mm_to_px(EXTEND_WINDOW_MM, dpi))
    paper_limit = mm_to_px(EXTEND_PAPER_GAP_MM, dpi)
    junction_limit = mm_to_px(EXTEND_JUNCTION_MM, dpi)
    # Опорные точки для касательной: последние EXTEND_FIT_MM отсчётов у этого конца.
    # Порядок — от самого конца внутрь: tail_along[0] — крайний отсчёт.
    if direction > 0:
        tail_along, tail_across = list(samples.along[-fit:][::-1]), list(samples.across[-fit:][::-1])
    else:
        tail_along, tail_across = list(samples.along[:fit]), list(samples.across[:fit])
    if len(tail_along) < 4:
        return []
    slope = float(np.polyfit(tail_along, tail_across, 1)[0])
    position, current = int(tail_along[0]), float(tail_across[0])
    gap, crossed = 0, False  # столбцов с последнего отсчёта и было ли в этом просвете пересечение
    found: list[tuple[float, float, int]] = []
    pieces = [0]  # индексы в found, с которых начинается кусок после просвета
    size_along, size_across = lines.shape
    while abs(slope) <= EXTEND_MAX_SLOPE:
        position += direction
        if position < 0 or position >= size_along:
            break
        predicted = current + slope * direction
        # Пробеги ищутся в окне пошире: пересекающая линейка тянется далеко за окно.
        reach = window + int(4 * thickness) + 2
        low = int(max(0, np.floor(predicted - reach)))
        high = int(min(size_across - 1, np.ceil(predicted + reach)))
        runs = [(a + low, b + low) for a, b in _runs(lines[position, low : high + 1])]
        near = [(a, b) for a, b in runs if abs((a + b) / 2.0 - predicted) <= window or a <= predicted <= b]
        allowed = THICK_RUN_FACTOR * thickness * np.hypot(1.0, slope) + abs(slope) + 1.0
        thin = [(a, b) for a, b in near if b - a + 1 <= allowed and abs((a + b) / 2.0 - predicted) <= window]
        if thin:
            # Просвет длиннее бумажного без пересечения — обрыв линейки; то, что за ним, — не она.
            if gap > paper_limit and not crossed:
                break
            if gap > 0 and found:
                pieces.append(len(found))
            a, b = min(thin, key=lambda run: abs((run[0] + run[1]) / 2.0 - predicted))
            start, stop = max(0, a - 1), min(size_across - 1, b + 1)
            weights = dark_lines[position, start : stop + 1]
            total = float(weights.sum())
            centre = start + float(np.dot(np.arange(weights.size), weights)) / total if total > 0 else (a + b) / 2.0
            found.append((float(position), centre, b - a + 1))
            current, gap, crossed = centre, 0, False
            tail_along = ([float(position)] + tail_along)[:fit]
            tail_across = ([centre] + tail_across)[:fit]
            if len(tail_along) >= 4 and abs(tail_along[0] - tail_along[-1]) >= 4:
                slope = float(np.polyfit(tail_along, tail_across, 1)[0])
            continue
        # Толстый пробег у предсказания — пересечение с перпендикулярной линейкой; иначе — бумага.
        gap += 1
        crossed = crossed or bool(near)
        current = predicted
        if gap > junction_limit or (gap > paper_limit and not crossed and gap > junction_limit // 2):
            break
    # Куски после просветов: короткий обрывает продолжение — это штрих буквы за узлом, а не линейка.
    after_gap = mm_to_px(EXTEND_AFTER_GAP_MM, dpi)
    bounds = pieces + [len(found)]
    kept = bounds[1] if len(bounds) > 1 else len(found)
    for start, stop in zip(bounds[1:-1], bounds[2:]):
        if stop - start < after_gap:
            break
        kept = stop
    found = found[:kept]
    return found if len(found) >= mm_to_px(EXTEND_MIN_MM, dpi) else []


def _extended(samples: _Samples, binary: np.ndarray, darkness: np.ndarray, horizontal: bool, dpi: int) -> _Samples:
    """Отсчёты, продолженные за оба конца линейки следованием по краске.

    Args:
        samples: Отсчёты линейки по её фрагментам.
        binary: Бинарная краска всей картинки (0/255).
        darkness: Тёмность серого всей картинки.
        horizontal: Ось линейки.
        dpi: Разрешение.

    Returns:
        Отсчёты с продолжениями; без изменений, если продолжать нечего.
    """
    if samples.along.size < 4:
        return samples
    lines = (binary.T if horizontal else binary) > 0
    dark_lines = darkness.T if horizontal else darkness
    thickness = float(np.median(samples.runs))
    tail = _extend_end(samples, lines, dark_lines, +1, thickness, dpi)
    head = _extend_end(samples, lines, dark_lines, -1, thickness, dpi)
    if not tail and not head:
        return samples
    extra = sorted(head + tail)
    along = np.concatenate([samples.along, [p[0] for p in extra]])
    across = np.concatenate([samples.across, [p[1] for p in extra]])
    runs = np.concatenate([samples.runs, [p[2] for p in extra]])
    order = np.argsort(along, kind="stable")
    along, across, runs = along[order], across[order], runs[order]
    span = along[-1] - along[0] + 1
    return _Samples(along, across, runs, _gaps(along, mm_to_px(GAP_MIN_MM, dpi)), float(along.size / span))


# --- сглаживание ---------------------------------------------------------------------


def _smooth(along: np.ndarray, across: np.ndarray, weights: np.ndarray, bandwidth_px: float) -> BSpline:
    """Сглаживающий сплайн с полосой ``bandwidth_px``: ``lam = плотность · h⁴``.

    Штраф ``lam ∫ f''²`` против суммы весов квадратов остатка даёт сглаживание с эквивалентной
    полосой ``(lam / плотность)^(1/4)``, где плотность — сумма весов на пиксель длины.

    Args:
        along: Координаты вдоль, строго возрастают.
        across: Отсчёты поперёк.
        weights: Веса отсчётов (0 допускается).
        bandwidth_px: Полоса сглаживания в пикселях.

    Returns:
        Сплайн ``across(along)``.
    """
    span = max(along[-1] - along[0], 1.0)
    density = float(weights.sum()) / span
    # make_smoothing_spline требует строго положительных весов, а Тьюки обнуляет выбросы: пол
    # MIN_WEIGHT оставляет такой отсчёт в задаче, но его влияние на кривую — тысячные доли.
    return make_smoothing_spline(along, across, w=np.maximum(weights, MIN_WEIGHT), lam=density * bandwidth_px**4)


def _window_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Скользящее среднее по ``window`` соседним отсчётам (окно у краёв укорачивается)."""
    if values.size == 0:
        return values
    window = max(1, min(window, values.size))
    kernel = np.ones(window)
    total = np.convolve(values, kernel, mode="same")
    count = np.convolve(np.ones_like(values), kernel, mode="same")
    return total / count


def _tukey(residual: np.ndarray, limit: float) -> np.ndarray:
    """Веса Тьюки: 1 у малых остатков, плавно к 0 у ``limit`` и дальше."""
    ratio = np.clip(np.abs(residual) / limit, 0.0, 1.0)
    return (1.0 - ratio**2) ** 2


@dataclass(frozen=True)
class _Fit:
    """Итог подгонки: сплайн, веса Тьюки, выбранная полоса сглаживания и остатки отсчётов."""

    spline: BSpline
    weights: np.ndarray
    bandwidth_mm: float
    rms: float
    worst: float


def _fit(along: np.ndarray, across: np.ndarray, thickness: float, dpi: int) -> _Fit:
    """Робастный сглаживающий сплайн с выбором полосы по шуму (см. докстринг модуля).

    Args:
        along: Координаты вдоль, строго возрастают.
        across: Отсчёты поперёк.
        thickness: Толщина штриха — порог Тьюки не меньше её.
        dpi: Разрешение — для перевода полос из мм в пиксели.

    Returns:
        Сплайн, веса, выбранная полоса и остатки.
    """
    to_px = dpi / 25.4
    weights = np.ones_like(across)
    # Робастные веса — на средней полосе лестницы: достаточно гладко, чтобы засечка буквы была
    # выбросом, и достаточно гибко, чтобы им не стал настоящий изгиб.
    limit = max(TUKEY_MIN_PX, thickness)
    for _ in range(TUKEY_ITERATIONS):
        spline = _smooth(along, across, weights, SMOOTH_LADDER_MM[1] * to_px)
        weights = _tukey(across - spline(along), limit)
        if weights.sum() < 4:
            weights = np.ones_like(across)
            break
    used = weights > MIN_WEIGHT
    # Шум отсчётов — MAD остатка от почти несглаженной кривой.
    fine = _smooth(along, across, weights, NOISE_BANDWIDTH_MM * to_px)
    residual = (across - fine(along))[used]
    noise = max(
        NOISE_FLOOR_PX, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) if residual.size else 0.0
    )
    tolerance = max(2.0 * noise, RESIDUAL_FLOOR_PX)
    window = int(round(RESIDUAL_WINDOW_MM * to_px))
    # Лестница от грубой полосы к тонкой; не нашлось подходящей — остаётся последняя, самая тонкая.
    for chosen in SMOOTH_LADDER_MM:
        spline = _smooth(along, across, weights, chosen * to_px)
        trend = _window_mean(np.where(used, across - spline(along), 0.0), window)
        if float(np.abs(trend[used]).max(initial=0.0)) <= tolerance:
            break
    residual = (across - spline(along))[used]
    rms = float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0
    worst = float(np.abs(residual).max(initial=0.0))
    return _Fit(spline, weights, chosen, rms, worst)


def _build(axis: Axis, samples: _Samples, dpi: int) -> "RuleTrace | None":
    """Трасса по отсчётам; ``None``, если отсчётов меньше, чем нужно кубическому сплайну."""
    if samples.along.size < 8:
        return None
    thickness = float(np.median(samples.runs))
    # Концы линейки в подгонку не идут: там угол с перпендикулярной линейкой слит в одно пятно, а
    # срез штриха косой, и отсчёты смещены до пикселя. Протяжённость — по всем отсчётам.
    trim = int(round(END_TRIM_FACTOR * thickness))
    keep = (samples.along >= samples.along[0] + trim) & (samples.along <= samples.along[-1] - trim)
    if keep.sum() < 8:
        keep = np.ones_like(samples.along, dtype=bool)
    fit = _fit(samples.along[keep], samples.across[keep], thickness, dpi)
    # В трассе — ВСЕ отсчёты (по ним сшивка строит кривую заново), у концевых вес ноль.
    weights = np.zeros(samples.along.size)
    weights[keep] = fit.weights
    return RuleTrace(
        axis=axis,
        along=samples.along,
        across=samples.across,
        weights=weights,
        spline=fit.spline,
        start=float(samples.along[0]),
        end=float(samples.along[-1]),
        thickness_px=thickness,
        coverage=samples.coverage,
        gaps=samples.gaps,
        bandwidth_mm=fit.bandwidth_mm,
        residual_rms_px=fit.rms,
        residual_max_px=fit.worst,
        dpi=dpi,
    )


# --- линейки картинки -----------------------------------------------------------------


@dataclass(frozen=True)
class RuleTraces:
    """Трассы линеек картинки по осям и маски их пикселей (рабочие пиксели)."""

    horizontal: list[RuleTrace]
    vertical: list[RuleTrace]
    horizontal_mask: np.ndarray  # uint8 0/255: пиксели фрагментов, вошедших в горизонтальные трассы
    vertical_mask: np.ndarray
    dpi: int

    @property
    def all(self) -> list[RuleTrace]:
        return self.horizontal + self.vertical

    @property
    def mask(self) -> np.ndarray:
        return cv2.bitwise_or(self.horizontal_mask, self.vertical_mask)


def _rule_mask(layer: FragmentLayer, rule: ChainedRule) -> tuple[np.ndarray, int, int]:
    """Пиксели фрагментов одной цепочки в её габарите: маска, x0, y0."""
    box = rule.segment.box
    ids = [layer.label_ids[index] for index in rule.members]
    return np.isin(layer.labels[box.slice], ids), box.x0, box.y0


def _trace_rule(
    gray: np.ndarray, binary: np.ndarray, page_darkness: np.ndarray, layer: FragmentLayer, rule: ChainedRule, dpi: int
) -> "RuleTrace | None":
    """Трасса одной цепочки: предсказание по среднему наклону, отсчёты, второй проход по первой кривой, продолжение за концы.

    Args:
        gray: Серая картинка (для субпиксельных отсчётов).
        binary: Бинарная краска картинки — по ней трасса продолжается за концы.
        page_darkness: Тёмность серого всей картинки (бумага − яркость, не меньше нуля).
        layer: Слой фрагментов оси цепочки.
        rule: Цепочка.
        dpi: Разрешение.

    Returns:
        Трасса или ``None``, если отсчётов не набралось.
    """
    segment = rule.segment
    horizontal = segment.horizontal
    mask, x0, y0 = _rule_mask(layer, rule)
    box = segment.box
    # Тёмность относительно бумаги габарита: бумага — медиана яркости вне линейки.
    crop = gray[box.slice].astype(np.float32)
    paper = float(np.median(crop[~mask])) if (~mask).any() else float(crop.max())
    darkness = np.clip(paper - crop, 0.0, None)
    # Предсказание первого прохода — прямая через центр габарита со средним наклоном цепочки.
    centre_along = (box.x0 + box.x1 - 1) / 2.0 if horizontal else (box.y0 + box.y1 - 1) / 2.0
    centre_across = (box.y0 + box.y1 - 1) / 2.0 if horizontal else (box.x0 + box.x1 - 1) / 2.0
    slope = float(np.tan(np.radians(segment.angle_deg if horizontal else -segment.angle_deg)))
    offsets = (x0, y0) if horizontal else (y0, x0)
    axis = Axis.HORIZONTAL if horizontal else Axis.VERTICAL
    first = _sample_rule(
        mask, darkness, horizontal, *offsets, lambda a: centre_across + slope * (a - centre_along), dpi
    )
    if first.along.size < 8:
        return None
    # Второй проход: предсказание — уже кривая (одна грубая подгонка, без лестницы и весов), пробег
    # в каждом столбце выбирается по ней.
    rough = _smooth(first.along, first.across, np.ones_like(first.across), SMOOTH_LADDER_MM[1] * dpi / 25.4)
    start, end = first.along[0], first.along[-1]
    second = _sample_rule(mask, darkness, horizontal, *offsets, lambda a: float(rough(np.clip(a, start, end))), dpi)
    samples = second if second.along.size >= 8 else first
    return _build(axis, _extended(samples, binary, page_darkness, horizontal, dpi), dpi)


def _combined(first: RuleTrace, second: RuleTrace, dpi: int) -> "RuleTrace | None":
    """Трасса по объединённым отсчётам двух трасс одной оси (в общих столбцах — отсчёт первой)."""
    along = np.concatenate([first.along, second.along])
    across = np.concatenate([first.across, second.across])
    order = np.argsort(along, kind="stable")
    along, across = along[order], across[order]
    unique = np.concatenate([[True], np.diff(along) > 0])
    runs = np.full(int(unique.sum()), (first.thickness_px + second.thickness_px) / 2.0)
    samples = _Samples(along[unique], across[unique], runs, _gaps(along[unique], mm_to_px(GAP_MIN_MM, dpi)), 1.0)
    joined = _build(first.axis, samples, dpi)
    if joined is None:
        return None
    span = max(joined.end - joined.start + 1, 1.0)
    return replace(joined, coverage=min(1.0, samples.along.size / span))


def _same(first: RuleTrace, second: RuleTrace, dpi: int) -> "RuleTrace | None":
    """Слить две трассы одной линейки, идущие вместе (куски, каждый продолженный за концы); иначе ``None``.

    Условие: общий участок — не меньше половины короткой, и кривые на нём расходятся (по медиане)
    не больше толщины штриха.
    """
    start, end = max(first.start, second.start), min(first.end, second.end)
    shorter = min(first.end - first.start, second.end - second.start)
    if end - start < 0.5 * shorter or end <= start:
        return None
    along = np.arange(np.ceil(start), np.floor(end) + 1)
    difference = np.abs(first.spline(along) - second.spline(along))
    if float(np.median(difference)) > max(first.thickness_px, second.thickness_px, 2.0):
        return None
    return _combined(first, second, dpi)


def _join(first: RuleTrace, second: RuleTrace, dpi: int) -> "RuleTrace | None":
    """Сшить две трассы одной оси, если вторая продолжает первую; иначе ``None``.

    Сшивка лечит промах ``rules.continues``: он продолжает фрагмент прямой через центр габарита со
    средним наклоном, и у длинной изогнутой линейки конец уходит от этой прямой дальше допуска.
    Здесь продолжаются КОНЦЫ кривых по их касательным.
    """
    if first.start > second.start:
        first, second = second, first
    gap = second.start - first.end
    tolerance = mm_to_px(CHAIN_OFFSET_MM, dpi)
    if gap > mm_to_px(CHAIN_GAP_MM, dpi) or gap < -tolerance:
        return None
    joint = (first.end + second.start) / 2.0
    reach = abs(gap) + 1.0
    left = first.across_at(joint, extend_px=reach)[0]
    right = second.across_at(joint, extend_px=reach)[0]
    if not np.isfinite(left) or not np.isfinite(right) or abs(left - right) > tolerance:
        return None
    return _combined(first, second, dpi)


def join_traces(traces: list[RuleTrace], dpi: int) -> list[RuleTrace]:
    """Сшивать трассы одной оси, пока находятся пары-продолжения или совпадающие куски одной линейки.

    Args:
        traces: Трассы одной оси, рабочие пиксели.
        dpi: Разрешение.

    Returns:
        Трассы после сшивки.
    """
    current = sorted(traces, key=lambda trace: trace.start)
    merged = True
    while merged:
        merged = False
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                joined = _same(current[i], current[j], dpi) or _join(current[i], current[j], dpi)
                if joined is not None:
                    current = [t for k, t in enumerate(current) if k not in (i, j)] + [joined]
                    current.sort(key=lambda trace: trace.start)
                    merged = True
                    break
            if merged:
                break
    return current


def trace_rules(gray: np.ndarray, dpi: int, min_mm: float = MIN_RULE_MM) -> RuleTraces:
    """Трассы всех линеек картинки: фрагменты 3 мм → цепочки → фильтр клякс → кривые → сшивка.

    Args:
        gray: Серая картинка — обычно вырезка таблицы в 300 dpi.
        dpi: Её разрешение.
        min_mm: Минимальная длина линейки. Сетка по кривым берёт 4 мм, чтобы не потерять короткие
            вертикали подшапки, и потом оставляет из коротких только «мостики» между линейками.

    Returns:
        Трассы по осям и маски их пикселей, всё в пикселях ``gray``.
    """
    binary = binarize(gray)
    layers = fragment_layers(binary, dpi)
    height, width = binary.shape[:2]
    chained = [chain_groups(layer.fragments, dpi, min_mm) for layer in layers]

    # Маски — только пиксели фрагментов, вошедших в цепочки: по ним считается краска без линеек.
    masks = [np.zeros((height, width), np.uint8), np.zeros((height, width), np.uint8)]
    for axis_index, layer in enumerate(layers):
        for rule in chained[axis_index]:
            piece, x0, y0 = _rule_mask(layer, rule)
            box = rule.segment.box
            masks[axis_index][box.slice][piece] = 255
    fat = cv2.dilate(cv2.bitwise_or(*masks), np.ones((3, 3), np.uint8))
    ink = cv2.bitwise_and(binary, cv2.bitwise_not(fat))

    # Тёмность всей картинки для продолжений: бумага — медиана яркости (на вырезке таблицы её большинство).
    page_darkness = np.clip(float(np.median(gray)) - gray.astype(np.float32), 0.0, None)
    traces: list[list[RuleTrace]] = []
    for axis_index, layer in enumerate(layers):
        found: list[RuleTrace] = []
        for rule in chained[axis_index]:
            # Кляксу у корешка отличает соседство: краска по обе стороны (см. rules.isolation).
            if isolation(ink, rule.segment, dpi) > MAX_ISOLATION:
                continue
            trace = _trace_rule(gray, binary, page_darkness, layer, rule, dpi)
            if trace is not None:
                found.append(trace)
        traces.append(join_traces(found, dpi))
    return with_masks(traces[0], traces[1], binary, dpi)


def rule_mask(traces: list[RuleTrace], binary: np.ndarray) -> np.ndarray:
    """Краска линеек: пиксели бинарной краски в полосе вдоль трасс (толщина штриха + запас), без разрывов.

    Маска строится по кривым, а не по фрагментам: в неё входят и продолжения за концы, которых во
    фрагментах нет, а буквы и штампы, не ставшие трассами, — не входят.

    Args:
        traces: Трассы одной оси в пикселях ``binary``.
        binary: Бинарная краска (0/255).

    Returns:
        Маска uint8 0/255 размером с ``binary``.
    """
    band = np.zeros(binary.shape[:2], np.uint8)
    for trace in traces:
        along = trace.along_grid(1.0)
        inside = np.ones(along.size, bool)
        # Разрывы линейки (пунктир, продавленная бумага) в маску не идут: там краски нет.
        for start, stop in trace.gaps:
            low, high = trace._from_work(np.array([start, stop]))
            inside &= ~((along > low) & (along < high))
        points = trace.point_at(along)
        width = max(1, int(round(trace.thickness_px * trace.frame.scale)) + 2)
        breaks = np.flatnonzero(np.diff(inside.astype(int)) != 0) + 1
        for piece, keep in zip(np.split(points, breaks), np.split(inside, breaks)):
            if keep.size and keep[0] and len(piece) >= 2:
                cv2.polylines(band, [np.round(piece).astype(np.int32)], False, 255, width)
    return cv2.bitwise_and(binary, band)


def with_masks(horizontal: list[RuleTrace], vertical: list[RuleTrace], binary: np.ndarray, dpi: int) -> RuleTraces:
    """Набор трасс с масками краски, построенными по самим трассам (``rule_mask``).

    Args:
        horizontal: Горизонтальные трассы.
        vertical: Вертикальные трассы.
        binary: Бинарная краска картинки.
        dpi: Разрешение.

    Returns:
        Трассы и маски их краски.
    """
    return RuleTraces(horizontal, vertical, rule_mask(horizontal, binary), rule_mask(vertical, binary), dpi)


__all__ = ["Axis", "Frame", "RuleTrace", "RuleTraces", "join_traces", "rule_mask", "trace_rules", "with_masks"]
