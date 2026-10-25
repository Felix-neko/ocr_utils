"""Подгонка модели сжатия у корешка по ширине отдельных букв, а не по окнам 5 мм: предсказание для буквы — сжатие модели, усреднённое по её собственному отрезку строки, так что короткий спад (1–2 мм) различим и не сводится к произведению «амплитуда × длина» одного окна."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from research.gutter_compression.profile import LENGTH_MM
from research.gutter_compression.shape import (
    HeightShape,
    MAX_FROM_EDGE_MM,
    ShapeFit,
    amplitude_of,
    initial_and_bounds,
    second_diff,
)


class FitSource(StrEnum):
    """По чему подгоняется модель сжатия стороны."""

    WINDOWS = "windows"  # окна 5 мм строки (мера ``acf`` и др., прежняя подгонка)
    LETTERS = "letters"  # ширина каждой буквы к опоре строки (``letter_profile.spread_letters``)


# Робастность подгонки по буквам: масштаб мягкой L1 по логарифму отношения. Разброс ширины одной буквы к опоре строки
# вдали от корешка — 1.4826·MAD 0.12 (22 кадра ХиП 1991, буквы дальше 25 мм), хвост вправо — слипшиеся буквы (p95 1.43).
LETTER_FIT_SCALE = 0.12
# Отношения вне этих пределов — сор, слипшиеся или разорванные буквы; в подгонку не идут.
LETTER_RATIO_RANGE = (0.3, 2.0)
# Букв стороны в подгонке не меньше (у стороны ~500 букв в 30 мм от края), иначе модель нулевая.
MIN_LETTERS = 120
# Пределы длины спада при подгонке по буквам, мм: снизу — полбуквы (окно 5 мм не различало спады короче 1.5 мм,
# буквы различают: 07/IMG_0078 L — последняя буква 0.82, вторая 0.9, третья — норма), сверху — как ``profile.LENGTH_MM``.
LETTER_LENGTH_MM = (0.5, LENGTH_MM[1])
# Точек вдоль буквы для усреднения сжатия по её отрезку (середины равных частей).
LETTER_SAMPLES = 5


@dataclass(frozen=True)
class LetterData:
    """Буквы стороны, готовые к подгонке.

    Поля:
        centre: Центр буквы от линии корешка по холсту, мм.
        half: Половина ширины буквы на холсте, мм.
        y: Доля высоты страницы центра пояса строки.
        r: Ширина буквы к опоре своей строки.
        line: Номер строки.
    """

    centre: np.ndarray
    half: np.ndarray
    y: np.ndarray
    r: np.ndarray
    line: np.ndarray


def letter_data(letters: pd.DataFrame, page_top: float, page_height: float, px_per_mm: float) -> LetterData:
    """Буквы одной стороны для подгонки: не дальше :data:`MAX_FROM_EDGE_MM` от края своей строки, без выбросов.

    Args:
        letters: Буквы стороны (``letter_profile.spread_letters``: ``edge_mm`` — край строки от корешка,
            ``from_edge_mm`` — центр буквы от края строки, ``width_px``, ``ratio``, ``y``, ``line``).
        page_top: Верх страницы на холсте, px.
        page_height: Высота страницы, px.
        px_per_mm: Плотность, px/мм.

    Returns:
        Буквы стороны.
    """
    lo, hi = LETTER_RATIO_RANGE
    part = letters[(letters.from_edge_mm <= MAX_FROM_EDGE_MM) & (letters.ratio > lo) & (letters.ratio < hi)]
    return LetterData(
        (part.edge_mm + part.from_edge_mm).to_numpy(np.float64),
        (part.width_px / (2.0 * px_per_mm)).to_numpy(np.float64),
        np.clip((part.y.to_numpy(np.float64) - page_top) / max(page_height, 1.0), 0.0, 1.0),
        part.ratio.to_numpy(np.float64),
        part.line.to_numpy(),
    )


def letter_prediction(length_mm: float, amplitude: np.ndarray, data: LetterData, origin_mm: float) -> np.ndarray:
    """Ожидаемое отношение ширины буквы к норме по модели ``c(u) = 1 − B·exp(−max(u − e, 0)/L)``.

    Буква занимает на холсте отрезок ``[центр − половина, центр + половина]``; её ширина на бумаге — ``∫ 1/c`` по
    отрезку, так что ожидаемое отношение — гармоническое среднее ``c`` по отрезку (так же модель разжимает строку при
    растяжении: ``∫ (1/c − 1)``).

    Args:
        length_mm: Длина спада ``L``, мм.
        amplitude: Амплитуда ``B`` на высоте каждой буквы (N,).
        data: Буквы.
        origin_mm: Начало спада ``e`` от корешка, мм.

    Returns:
        Ожидаемые отношения (N,).
    """
    # Середины LETTER_SAMPLES равных частей отрезка буквы: смещения от центра в долях половины ширины.
    steps = (np.arange(LETTER_SAMPLES) + 0.5) / LETTER_SAMPLES * 2.0 - 1.0
    u = data.centre[:, None] + data.half[:, None] * steps[None, :]
    c = 1.0 - amplitude[:, None] * np.exp(-np.maximum(u - origin_mm, 0.0) / length_mm)
    return 1.0 / np.mean(1.0 / np.maximum(c, 1e-3), axis=1)


def letter_residuals(params: np.ndarray, shape: HeightShape, data: LetterData, origin_mm: float,
                     penalty: float) -> np.ndarray:  # fmt: skip
    """Остатки подгонки по буквам: ``log r − log r̂`` и хвост штрафа P-сплайна.

    Логарифм делает разброс букв симметричным (слипшиеся буквы дают отношение до 2, узкие — до 0.5).

    Args:
        params: ``(L, параметры амплитуды…)``.
        shape: Семейство амплитуды по высоте.
        data: Буквы.
        origin_mm: Начало спада от корешка, мм.
        penalty: Штраф за вторую разность коэффициентов (только SPLINE).

    Returns:
        Остатки букв, у P-сплайна — с хвостом штрафа.
    """
    length, coef = params[0], params[1:]
    amp = np.maximum(amplitude_of(shape, coef, data.y), 0.0)
    out = np.log(data.r) - np.log(letter_prediction(length, amp, data, origin_mm))
    if shape is HeightShape.SPLINE and penalty > 0:
        # Штраф масштабирован к числу букв, как у ``shape.residuals`` к числу окон.
        out = np.r_[out, np.sqrt(penalty * len(data.r) / 1000.0) * (second_diff(len(coef)) @ coef)]
    return out


def fit_letters(data: LetterData, shape: HeightShape, origin_mm: float, penalty: float = 0.0) -> ShapeFit:
    """Робастная подгонка длины спада и амплитуды по высоте к ширине букв стороны.

    Args:
        data: Буквы стороны.
        shape: Семейство амплитуды по высоте.
        origin_mm: Начало спада от корешка (начало зоны текста), мм.
        penalty: Штраф P-сплайна.

    Returns:
        Подгонка (``windows`` — число букв, ``residual`` — 1.4826·MAD остатков логарифма).
    """
    start, lo, hi = initial_and_bounds(shape)
    start, lo, hi = np.r_[3.0, start], np.r_[LETTER_LENGTH_MM[0], lo], np.r_[LETTER_LENGTH_MM[1], hi]
    fit = least_squares(letter_residuals, x0=start, bounds=(lo, hi), loss="soft_l1", f_scale=LETTER_FIT_SCALE,
                        args=(shape, data, origin_mm, penalty))  # fmt: skip
    res = letter_residuals(fit.x, shape, data, origin_mm, 0.0)
    mad = 1.4826 * float(np.median(np.abs(res - np.median(res))))
    return ShapeFit(shape, tuple(float(v) for v in fit.x[1:]), float(fit.x[0]), penalty, mad, len(data.r))


# Буквы ближе этого к краю своей строки — «у края» в сводке до и после растяжения, мм (последние 2–3 буквы).
NEAR_LETTER_MM = 3.0


def letter_summary(letters: pd.DataFrame) -> dict:
    """Ширина букв у края текста стороны к опоре строки: последние буквы и полоса :data:`NEAR_LETTER_MM`.

    Args:
        letters: Буквы стороны.

    Returns:
        ``last_letter`` — медиана отношения ближайшей к корешку буквы строки, ``near_letter`` — медиана отношения букв
        ближе :data:`NEAR_LETTER_MM` к краю строки, ``letters`` — число букв (NaN — букв нет).
    """
    if len(letters) == 0:
        return {"last_letter": np.nan, "near_letter": np.nan, "letters": 0}
    last = letters.loc[letters.groupby("line").from_edge_mm.idxmin(), "ratio"]
    near = letters.loc[letters.from_edge_mm < NEAR_LETTER_MM, "ratio"]
    return {"last_letter": round(float(last.median()), 3),
            "near_letter": round(float(near.median()), 3) if len(near) else np.nan, "letters": len(letters)}  # fmt: skip


__all__ = ["FitSource", "LetterData", "fit_letters", "letter_data", "letter_prediction", "letter_summary"]
