"""Профиль сжатия страницы: сводка стороны для отбора, медианы по поясам, поправка мер на место в строке (начала строк — прописные, концы — знаки препинания) и робастная подгонка модели сжатия ``c(d, y) = 1 − B(y)·exp(−(d − e)/L)`` от края текста с линейной по высоте амплитудой."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

from research.gutter_compression.measure import HORIZONTAL, Method, Side

# Пояса расстояния от края текста у корешка для сводки, мм (центр окна 5 мм; первое окно — 2.5 мм от края).
NEAR_MM = (0.0, 5.6)
MID_MM = (8.0, 14.0)
# Сторона в счёт, если у неё не меньше стольких строк с окнами у края.
MIN_LINES = 8
# Узлов P-сплайна амплитуды по высоте (кубический, равномерные узлы на [0, 1]).
SPLINE_KNOTS = 7


class LinePosition(StrEnum):
    """Место края текста в строке: у корешка левой страницы — концы строк, правой — начала."""

    START = "start"  # начала строк: больше прописных и отступы абзацев — средняя ширина знака больше
    END = "end"  # концы строк: больше знаков препинания и дефисов — средняя ширина знака меньше


def spine_position(side: str) -> LinePosition:
    """Что стоит у корешка на стороне: у левой страницы — концы строк, у правой — начала."""
    return LinePosition.END if side == Side.LEFT.value else LinePosition.START


def correct_position(table: pd.DataFrame, bias: pd.DataFrame | None) -> pd.DataFrame:
    """Поправка отношений мер окон на место в строке.

    У начала строки чаще прописные (шире), у конца — знаки препинания и дефисы (уже); на мерах по форме штрихов это
    даёт сдвиг отношения у края текста без всякого сжатия. Поправка — медиана той же меры у ВНЕШНЕГО края текста
    (там нет корешка) в том же поясе расстояния от края (``outer``, ``position_bias.csv``): для левой страницы — по
    концам строк, для правой — по началам. Отношение делится на неё.

    Args:
        table: Окна стороны или разворота (``side``, ``from_edge_mm``, ``<мера>_ratio``).
        bias: Таблица ``position_bias.csv`` (position, bin, ``<мера>_ratio``) или None — без поправки.

    Returns:
        Копия таблицы с поправленными отношениями.
    """
    if bias is None or table.empty:
        return table
    out = table.copy()
    step = float(np.diff(np.unique(bias.bin))[0]) if bias.bin.nunique() > 1 else 1.0
    bins = (out.from_edge_mm // step) * step
    for method in Method:
        column = ratio_column(method)
        source = f"{method.value}_edge"
        if source not in bias:
            continue
        lookup = bias.set_index(["position", "bin"])[source]
        keys = list(zip([spine_position(s).value for s in out.side], bins))
        factor = np.array([lookup.get(k, 1.0) for k in keys], np.float64)
        out[column] = out[column] / np.where(np.isfinite(factor) & (factor > 0), factor, 1.0)
    return out


def ratio_column(method: Method) -> str:
    """Имя столбца отношения меры к опоре строки."""
    return f"{method.value}_ratio"


def side_summary(table: pd.DataFrame) -> dict:
    """Сводка стороны: медианы отношений у края текста и чуть дальше, отступ текста от корешка.

    Args:
        table: Окна одной стороны (``measure.page_windows``).

    Returns:
        Поля: ``lines``, ``edge_mm`` (медианный отступ края текста от корешка), ``near_<мера>``, ``mid_<мера>``,
        ``near_aniso`` (шаг по автокорреляции к высоте строчных у края — сжатие без поправки на общий масштаб).
    """
    near = table[(table.from_edge_mm >= NEAR_MM[0]) & (table.from_edge_mm <= NEAR_MM[1])]
    mid = table[(table.from_edge_mm >= MID_MM[0]) & (table.from_edge_mm <= MID_MM[1])]
    out = {"lines": int(near.line.nunique()), "edge_mm": round(float(table.edge_mm.median()), 2)}
    for method in Method:
        column = ratio_column(method)
        out[f"near_{method.value}"] = round(float(near[column].median()), 3) if len(near) else np.nan
        out[f"mid_{method.value}"] = round(float(mid[column].median()), 3) if len(mid) else np.nan
    aniso = near[ratio_column(Method.ACF)] / near[ratio_column(Method.HEIGHT)]
    out["near_aniso"] = round(float(aniso.median()), 3) if len(near) else np.nan
    return out


def binned(table: pd.DataFrame, method: Method, step_mm: float = 1.0, by: str = "from_edge_mm") -> pd.DataFrame:
    """Медиана отношения меры по поясам расстояния с бутстрепным разбросом по строкам.

    Args:
        table: Окна одной стороны.
        method: Мера.
        step_mm: Ширина пояса, мм.
        by: Расстояние: ``from_edge_mm`` (от края текста) или ``d_mm`` (от корешка).

    Returns:
        Таблица: ``bin`` (начало пояса, мм), ``median``, ``n`` (окон), ``lo``/``hi`` (5-й и 95-й перцентили медианы
        по 200 бутстрепам строк).
    """
    column = ratio_column(method)
    data = table[[by, "line", column]].dropna()
    data = data.assign(bin=(data[by] // step_mm) * step_mm)
    rng = np.random.default_rng(0)
    rows = []
    for start, part in data.groupby("bin"):
        values = part[column].to_numpy()
        boots = [np.median(rng.choice(values, len(values))) for _ in range(200)] if len(values) >= 3 else [np.nan]
        rows.append({"bin": start, "median": float(np.median(values)), "n": len(values),
                     "lo": float(np.percentile(boots, 5)), "hi": float(np.percentile(boots, 95))})  # fmt: skip
    return pd.DataFrame(rows)


def spline_basis(y_frac: np.ndarray) -> np.ndarray:
    """Базис кубического B-сплайна по доле высоты: равномерные узлы на [0, 1].

    Args:
        y_frac: Доли высоты (N,).

    Returns:
        float (N, SPLINE_KNOTS + 2) — значения базисных функций.
    """
    inner = np.linspace(0.0, 1.0, SPLINE_KNOTS)
    knots = np.r_[[0.0] * 3, inner, [1.0] * 3]
    count = len(knots) - 4
    y = np.clip(np.asarray(y_frac, np.float64), 0.0, 1.0)
    return BSpline.design_matrix(y, knots, 3).toarray() if count > 0 else np.ones((len(y), 1))


@dataclass(frozen=True)
class CompressionModel:
    """Модель горизонтального сжатия страницы у корешка: ``c(d, y) = 1 − B(y)·exp(−(d − e)/L)`` при ``d ≥ e``.

    ``d`` — расстояние от линии корешка по холсту (мм), ``e`` — отступ края текста от корешка (медиана строк стороны),
    ``y`` — доля высоты страницы от 0 (верх) до 1 (низ), ``B(y) = amplitude + tilt·(y − 0.5)`` — сжатие вдоль линии
    корешка постоянно или меняется линейно (книга раскрыта неровно: вверху растянута, внизу нет). В пустом поле у
    корешка (``d < e``) сжатия нет, на первых :data:`EDGE_RAMP_MM` текста оно нарастает гладко. ``c`` — во сколько раз
    отрезок строки на холсте короче, чем вдали от корешка; растяжение ``1/c`` плавно уходит в 1 с удалением от корешка.

    Почему от края текста, а не от корешка: при отсчёте от корешка экспонента продлевается через пустое поле 2–8 мм,
    амплитуда на корешке упирается в предел (0.7 у сильного и заметного поясов) и меняется с отступом, а не со сжатием.

    Поля:
        amplitude: Сжатие у края текста в середине высоты (доля).
        length_mm: Длина спада сжатия от края текста, мм.
        tilt: Изменение амплитуды от верха до низа страницы (доля).
        windows: Окон в подгонке.
        lines: Строк в подгонке.
        residual: Робастный разброс остатков (1.4826·MAD), доля.
        edge_mm: Отступ края текста от корешка, мм.
        spline_coef: Коэффициенты P-сплайна амплитуды по высоте (:func:`spline_basis`) или пусто — амплитуда
            ``amplitude + tilt·(y − 0.5)``. Шаг ``shape``: сплайн со штрафом 30 хуже линии на 1 % сжатых сторон и
            лучше на 8 % (сторона со ступенью сжатия у низа — 01/IMG_1052 R).
    """

    amplitude: float = 0.0
    length_mm: float = 5.0
    tilt: float = 0.0
    windows: int = 0
    lines: int = 0
    residual: float = float("nan")
    edge_mm: float = 0.0
    spline_coef: tuple[float, ...] = ()

    def amplitude_at(self, y_frac: np.ndarray | float) -> np.ndarray:
        """Амплитуда на доле высоты страницы (не меньше нуля): по сплайну, если он задан, иначе линия."""
        y = np.asarray(y_frac, np.float64)
        if self.spline_coef:
            return np.maximum((spline_basis(y.ravel()) @ np.asarray(self.spline_coef)).reshape(y.shape), 0.0)
        return np.maximum(self.amplitude + self.tilt * (y - 0.5), 0.0)

    def scale(self, d_mm: np.ndarray, y_frac: np.ndarray | float = 0.5) -> np.ndarray:
        """Сжатие ``c`` в точке на расстоянии ``d_mm`` от корешка: 1 — как вдали от корешка.

        В поле у корешка (``d < e``) сжатия нет, у края текста оно нарастает гладко (кубический шаг) на
        :data:`EDGE_RAMP_MM`: поле сдвигается целиком, на одно и то же расстояние на любой высоте страницы. Иначе
        висячие дефисы и запятые за краем букв (сторона блока по краске проходит по ним) уезжали бы по-разному на
        разных высотах при сжатии, меняющемся по высоте, и сторона блока у корешка гнулась (до 0.46 мм).
        """
        x = np.asarray(d_mm, np.float64) - self.edge_mm
        ramp = np.clip(x / EDGE_RAMP_MM, 0.0, 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        return 1.0 - self.amplitude_at(y_frac) * np.exp(-np.maximum(x, 0.0) / self.length_mm) * ramp

    @property
    def null(self) -> bool:
        """Модель без сжатия."""
        return self.amplitude <= 0.0 and self.tilt == 0.0 and not self.spline_coef


# Нарастание сжатия от края текста внутрь, мм (кубический шаг; поле у корешка не растягивается).
EDGE_RAMP_MM = 0.5
# Пределы подгонки: амплитуда у края текста, длина спада (мм), наклон по высоте.
AMPLITUDE_MAX = 0.6
LENGTH_MM = (1.5, 30.0)
TILT_MAX = 0.3
# Робастность подгонки: масштаб мягкой L1 (доля отношения) — шум одного окна по контролю ~0.1 (остатки 0.09–0.11).
FIT_SCALE = 0.08


def model_residuals(params: np.ndarray, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Остатки модели сжатия по окнам.

    Args:
        params: ``(amplitude, length_mm, tilt)``.
        x: Расстояние окон от края текста стороны (не меньше 0), мм.
        y: Доля высоты страницы окон.
        r: Отношение меры окна к опоре строки.

    Returns:
        ``r − c``.
    """
    amp = np.maximum(params[0] + params[2] * (y - 0.5), 0.0)
    return r - (1.0 - amp * np.exp(-x / params[1]))


def fit_model(table: pd.DataFrame, method: Method, page_height: float, page_top: float,
              max_from_edge_mm: float = 30.0, use_tilt: bool = True,
              tilt_max: float | None = None) -> CompressionModel:  # fmt: skip
    """Робастная подгонка модели сжатия к отношениям меры в окнах строк стороны.

    Опора строки — её окна дальше 25 мм от края текста, поэтому отношение там ≈ 1 и модель на них прижимается к нулю
    сжатия. Подгонка — мягкая L1 по окнам ближе ``max_from_edge_mm`` к краю текста своей строки; расстояние в модели —
    от корешка минус отступ края текста стороны (медиана строк): короткие строки (конец абзаца на левой странице,
    отступ на правой) лежат дальше от корешка и сжаты меньше — модель это учитывает.

    Args:
        table: Окна одной стороны.
        method: Мера (горизонтальная).
        page_height: Высота страницы на холсте, px.
        page_top: Верх страницы на холсте, px.
        max_from_edge_mm: Окна не дальше этого от края текста.
        use_tilt: Подгонять наклон амплитуды по высоте.
        tilt_max: Предел модуля наклона или None — :data:`TILT_MAX`. Шаг ``shape`` (2026-10-25): у 56 % сжатых
            сторон перепад амплитуды сверху вниз больше 0.3 (медиана 0.33, p10 0.67), и предел 0.3 упирался.

    Returns:
        Модель (нулевая — окон мало).
    """
    column = ratio_column(method)
    edge = float(table.edge_mm.median()) if len(table) else 0.0
    data = table[(table.from_edge_mm <= max_from_edge_mm)][["d_mm", "y", "line", column]].dropna()
    data = data[(data[column] > 0.3) & (data[column] < 2.0)]
    if data.line.nunique() < MIN_LINES:
        return CompressionModel(windows=len(data), lines=int(data.line.nunique()), edge_mm=edge)
    x = np.maximum(data.d_mm.to_numpy() - edge, 0.0)
    y = np.clip((data.y.to_numpy() - page_top) / max(page_height, 1.0), 0.0, 1.0)
    r = data[column].to_numpy()
    tilt_limit = (TILT_MAX if tilt_max is None else tilt_max) if use_tilt else 1e-9
    lower = [0.0, LENGTH_MM[0], -tilt_limit]
    upper = [AMPLITUDE_MAX, LENGTH_MM[1], tilt_limit]
    fit = least_squares(model_residuals, x0=[0.1, 6.0, 0.0], bounds=(lower, upper), loss="soft_l1",
                        f_scale=FIT_SCALE, args=(x, y, r))  # fmt: skip
    res = model_residuals(fit.x, x, y, r)
    mad = 1.4826 * float(np.median(np.abs(res - np.median(res))))
    return CompressionModel(float(fit.x[0]), float(fit.x[1]), float(fit.x[2]) if use_tilt else 0.0, len(data),
                            int(data.line.nunique()), mad, edge)  # fmt: skip


__all__ = ["HORIZONTAL", "CompressionModel", "binned", "fit_model", "ratio_column", "side_summary"]
