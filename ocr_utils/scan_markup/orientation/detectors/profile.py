"""Классический проекционный профиль: ось текста по периодичности строк.

Метод Постля и всё семейство «projection profile / Radon»: строки текста дают правильное
чередование краски и бумаги поперёк себя, и вдоль верной оси профиль проекции периодичен.
Здесь периодичность меряется автокорреляцией профиля: высота первого пика на лаге, равном
межстрочному расстоянию.

ЗАЧЕМ ОН, если ``ink_axis`` уже даёт ось. Затем, что он меряет ДРУГОЕ. RLSA ищет вытянутые
сгустки краски и ошибётся там, где сгустки есть, а строк нет (столбец цифр, пунктир, ряд
однотипных значков). Автокорреляция ищет РЕГУЛЯРНОСТЬ и ошибётся ровно наоборот — на
странице с одной-двумя строками ей нечего складывать. Расхождение этих двух — сигнал
посмотреть на полосу глазами, поэтому они и стоят рядом в отчёте.

Предобработка (отбор компонент размера глифа) взята та же, что в ``ink_axis``, и намеренно:
на сырой бинаризации длинные штрихи чертежа перебивают профиль целиком, и сравнивать было
бы нечего. Разница между детекторами — в мере, а не во входе.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from ocr_utils.scan_markup.orientation.detectors.base import Detector, Frame, Verdict, unknown
from ocr_utils.scan_markup.orientation.detectors.ink_axis import MIN_LINE_INK_FRAC, glyph_mask

# Диапазон межстрочного расстояния, в пикселях копии 150 dpi. 12 px это 2 мм (плотный
# петит таблицы), 47 px это 8 мм (разрежённый заголовочный набор). Пик автокорреляции
# ищется только здесь: на меньших лагах сидит толщина самого штриха, на больших — случайные
# совпадения абзацев.
PITCH_MIN_PX = 12
PITCH_MAX_PX = 47

# Окно скользящего среднего, которое вычитается из профиля перед автокорреляцией, в долях
# максимального межстрочного расстояния. Без этого шага мера не работает вовсе: профиль
# текстовой полосы это не только чередование строк, но и ОГИБАЮЩАЯ — плато на текстовом
# блоке и ноль на полях. Огибающая коррелирует сама с собой на любом малом лаге, и на
# обычной полосе пик выходил 0.918 вдоль строк против 0.798 поперёк, то есть перевес 0.07
# при пороге 0.15 — детектор молчал на всех полосах подряд. Вычитание скользящего среднего
# убирает плато и оставляет ровно то, ради чего мера затевалась.
DETREND_FACTOR = 2.0

# Ниже этого перевеса одной оси над другой ось считается неразличимой.
AXIS_MARGIN_THR = 0.15


def periodicity(mask: np.ndarray, horizontal: bool) -> float:
    """Высота пика автокорреляции профиля проекции — мера регулярности строк."""
    profile = mask.sum(axis=1 if horizontal else 0).astype(np.float64)
    if profile.size <= PITCH_MAX_PX * 2:
        return 0.0
    window = int(PITCH_MAX_PX * DETREND_FACTOR) | 1
    profile -= uniform_filter1d(profile, size=window, mode="nearest")
    energy = float((profile * profile).sum())
    if energy <= 0.0:
        return 0.0
    correlation = np.correlate(profile, profile, mode="full")[profile.size - 1 :] / energy
    window = correlation[PITCH_MIN_PX : PITCH_MAX_PX + 1]
    return float(window.max()) if window.size else 0.0


def detect(frame: Frame) -> Verdict:
    mask = glyph_mask(frame.gray150)
    if mask.sum() < MIN_LINE_INK_FRAC * mask.size * 255:
        return unknown("мало текста")

    horizontal = max(0.0, periodicity(mask, horizontal=True))
    vertical = max(0.0, periodicity(mask, horizontal=False))
    total = horizontal + vertical
    if total <= 0.0:
        return unknown("профиль без периодичности")

    axis_score = (horizontal - vertical) / total
    metrics = {"axis_score": axis_score, "peak_h": horizontal, "peak_v": vertical}
    if abs(axis_score) < AXIS_MARGIN_THR:
        return Verdict(0, 0.0, metrics=metrics, note="ось не различается")

    # Периодичность профиля симметрична относительно переворота на 180: строки остаются
    # строками. Поэтому детектор принципиально о стороне не судит.
    rotation = 0 if axis_score > 0 else 90
    return Verdict(rotation, min(1.0, abs(axis_score)), axis_only=True, metrics=metrics)


ALGORITHM = Detector(
    name="profile",
    summary="классика: ось по периодичности проекционного профиля (автокорреляция)",
    stage="cpu",
    run=detect,
)
