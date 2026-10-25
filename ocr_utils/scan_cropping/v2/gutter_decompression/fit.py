"""Подгонка модели сжатия у корешка по ширине отдельных букв: предсказание для буквы — сжатие модели, усреднённое по её собственному отрезку строки (гармоническое среднее), так что короткий сильный спад в одну-две буквы различим и не сводится к произведению «амплитуда × длина» окна."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ocr_utils.scan_cropping.v2.gutter_decompression.model import (
    AMPLITUDE_RANGE,
    HEIGHT_SHAPE,
    SPLINE_PENALTY,
    CompressionModel,
    HeightShape,
    amplitude_of,
    n_params,
)

# Буквы дальше этого от края своей строки в подгонку не идут, мм (дальше — опора строки, отношение ≈ 1).
MAX_FROM_EDGE_MM = 30.0
# Робастность подгонки: масштаб мягкой L1 по логарифму отношения. Разброс ширины одной буквы к опоре строки вдали от
# корешка — 1.4826·MAD 0.12 (22 кадра ХиП 1991, буквы дальше 25 мм), хвост вправо — слипшиеся буквы (p95 1.43).
FIT_SCALE = 0.12
# Отношения вне этих пределов — сор, слипшиеся или разорванные буквы.
RATIO_RANGE = (0.3, 2.0)
# Букв стороны в подгонке не меньше (у страницы ХиП ~500 букв в 30 мм от края) и строк не меньше, иначе модель нулевая.
MIN_LETTERS = 120
MIN_LINES = 8
# Пределы длины спада, мм: снизу — полбуквы (окна 5 мм стенда не различали спады короче 1.5 мм, буквы различают:
# ХиП 1991/07 IMG_0078 L — последняя буква 0.82, вторая 0.9, третья — норма), сверху — дальше спада как такового.
LENGTH_MM = (0.5, 30.0)
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
    """

    centre: np.ndarray
    half: np.ndarray
    y: np.ndarray
    r: np.ndarray


def letter_data(letters: pd.DataFrame, page_top: float, page_height: float, px_per_mm: float) -> LetterData:
    """Буквы одной стороны для подгонки: не дальше :data:`MAX_FROM_EDGE_MM` от края своей строки, без выбросов.

    Args:
        letters: Буквы стороны (``letters.page_letters``).
        page_top: Верх страницы на холсте, px.
        page_height: Высота страницы, px.
        px_per_mm: Плотность, px/мм.

    Returns:
        Буквы стороны.
    """
    lo, hi = RATIO_RANGE
    part = letters[(letters.from_edge_mm <= MAX_FROM_EDGE_MM) & (letters.ratio > lo) & (letters.ratio < hi)]
    return LetterData(
        (part.edge_mm + part.from_edge_mm).to_numpy(np.float64),
        (part.width_px / (2.0 * px_per_mm)).to_numpy(np.float64),
        np.clip((part.y.to_numpy(np.float64) - page_top) / max(page_height, 1.0), 0.0, 1.0),
        part.ratio.to_numpy(np.float64),
    )


def letter_prediction(length_mm: float, amplitude: np.ndarray, data: LetterData, origin_mm: float) -> np.ndarray:
    """Ожидаемое отношение ширины буквы к норме по модели ``c(u) = 1 − B·exp(−max(u − e, 0)/L)``.

    Буква занимает на холсте отрезок ``центр ± половина``; её ширина на бумаге — ``∫ 1/c`` по отрезку, так что
    ожидаемое отношение — гармоническое среднее ``c`` по отрезку (так же модель разжимает строку: ``∫ (1/c − 1)``).

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


def second_diff(k: int) -> np.ndarray:
    """Матрица вторых разностей (k − 2, k) — штраф P-сплайна."""
    return np.diff(np.eye(k), 2, axis=0)


def residuals(params: np.ndarray, shape: HeightShape, data: LetterData, origin_mm: float,
              penalty: float) -> np.ndarray:  # fmt: skip
    """Остатки подгонки: ``log r − log r̂`` по буквам и хвост штрафа P-сплайна.

    Логарифм делает разброс букв симметричным (слипшиеся буквы дают отношение до 2, узкие — до 0.5).

    Args:
        params: ``(L, параметры амплитуды…)``.
        shape: Форма амплитуды по высоте.
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
        # Штраф масштабирован к числу букв: λ одинаково сглаживает сторону из 300 и из 3000 букв.
        out = np.r_[out, np.sqrt(penalty * len(data.r) / 1000.0) * (second_diff(len(coef)) @ coef)]
    return out


def initial_and_bounds(shape: HeightShape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Начальная точка и пределы параметров ``(L, амплитуда…)``.

    Args:
        shape: Форма амплитуды по высоте.

    Returns:
        (начальные параметры, нижние пределы, верхние пределы).
    """
    lo_a, hi_a = AMPLITUDE_RANGE
    if shape is HeightShape.SPLINE:
        k = n_params(shape)
        start, lo, hi = np.full(k, 0.1), np.full(k, lo_a), np.full(k, hi_a)
    else:
        # Линия: амплитуда в середине и половина перепада сверху вниз.
        start, lo, hi = np.array([0.1, 0.0]), np.array([lo_a, -0.6]), np.array([hi_a, 0.6])
    return np.r_[3.0, start], np.r_[LENGTH_MM[0], lo], np.r_[LENGTH_MM[1], hi]


def fit_model(letters: pd.DataFrame, page_top: float, page_height: float, px_per_mm: float, origin_mm: float,
              shape: HeightShape = HEIGHT_SHAPE, penalty: float = SPLINE_PENALTY) -> CompressionModel:  # fmt: skip
    """Робастная подгонка длины спада и амплитуды по высоте к ширине букв стороны.

    Args:
        letters: Буквы стороны (``letters.page_letters``).
        page_top: Верх страницы на холсте, px.
        page_height: Высота страницы, px.
        px_per_mm: Плотность, px/мм.
        origin_mm: Начало спада от корешка (начало зоны текста стороны), мм.
        shape: Форма амплитуды по высоте.
        penalty: Штраф P-сплайна.

    Returns:
        Модель (нулевая — букв или строк мало).
    """
    lines = int(letters.line.nunique()) if len(letters) else 0
    data = letter_data(letters, page_top, page_height, px_per_mm) if len(letters) else None
    if data is None or lines < MIN_LINES or len(data.r) < MIN_LETTERS:
        return CompressionModel(shape, (), origin_mm=origin_mm, letters=0 if data is None else len(data.r),
                                lines=lines)  # fmt: skip
    start, lo, hi = initial_and_bounds(shape)
    fit = least_squares(residuals, x0=start, bounds=(lo, hi), loss="soft_l1", f_scale=FIT_SCALE,
                        args=(shape, data, origin_mm, penalty))  # fmt: skip
    res = residuals(fit.x, shape, data, origin_mm, 0.0)
    mad = 1.4826 * float(np.median(np.abs(res - np.median(res))))
    return CompressionModel(shape, tuple(float(v) for v in fit.x[1:]), float(fit.x[0]), origin_mm, len(data.r), lines,
                            mad)  # fmt: skip


__all__ = ["LetterData", "fit_model", "letter_data", "letter_prediction"]
