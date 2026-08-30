"""Балл «текст ушёл под корешок» и признак таблицы у сгиба.

ЧТО МЕРИМ. Внутреннее поле полосы — расстояние от сгиба до конца строки — выраженное
в ШАГАХ СТРОК. Единица нужна своя: пак — сорок лет выпусков, кегль и расстояние до
камеры в них разные, а шаг строк меняется вместе с ними и потому годится за масштаб.

ПОЧЕМУ БАЛЛ СЧИТАЕТСЯ ПО СУММЕ ПОЛЕЙ ОБЕИХ ПОЛОС. Тугой переплёт — свойство разворота,
а не страницы: съедает он обе полосы сразу, просто несимметрично. На размеченной папке
«1926/08» (148 кадров, 32 помечены вручную) сумма полей ранжирует заметно лучше, чем
любая из полос по отдельности и чем доля строк, дотянувшихся до сгиба:

    поле левой полосы            AUC 0.77
    поле правой полосы           AUC 0.84
    доля строк у сгиба           AUC 0.76
    СУММА полей (эта метрика)    AUC 0.88

БЕРЁТСЯ 10-Й ПРОЦЕНТИЛЬ, а не медиана: строки в наборе разной длины, и вопрос «теряем ли
текст» решают самые тесные из них, а не типичные.
"""

from dataclasses import dataclass

import numpy as np

from ocr_utils.gutter_loss_detection.geometry import SideGeometry, SpreadGeometry

# Внутреннее поле, при котором полоса считается здоровой, в шагах строк. Ниже балл
# растёт линейно до единицы (текст упирается в сгиб).
HEALTHY_SIDE = 0.60
HEALTHY_GAP = 1.20

# Порог по баллу. Откалиброван по размеченной папке как лучший F1: соответствует
# суммарному коридору около 0.8 шага строки.
THRESHOLD = 0.35

# Сколько длинных линеек у корешка считать таблицей.
TABLE_RULES_V = 2
TABLE_RULES_H = 2


def side_bite(side: SideGeometry) -> float:
    """Балл полосы: 0 — поле не меньше нормы, 1 — текст упирается в сгиб.

    Args:
        side: Разбор полосы.

    Returns:
        Балл в диапазоне [0, 1]; NaN, если поле не измерено.
    """
    tight = side.tight
    if not np.isfinite(tight):
        return float("nan")
    return float(np.clip(1.0 - tight / HEALTHY_SIDE, 0.0, 1.0))


def spread_bite(geometry: SpreadGeometry) -> float:
    """Балл кадра по суммарному коридору у корешка.

    Args:
        geometry: Разбор разворота.

    Returns:
        Балл в диапазоне [0, 1]; NaN, если кадр не измерен.
    """
    if geometry.problem or len(geometry.sides) != 2:
        return float("nan")
    gap = sum(s.tight for s in geometry.sides)
    if not np.isfinite(gap):
        return float("nan")
    return float(np.clip(1.0 - gap / HEALTHY_GAP, 0.0, 1.0))


def is_tabular(side: SideGeometry) -> bool:
    """Похожа ли приосевая зона полосы на таблицу.

    Args:
        side: Разбор полосы.

    Returns:
        True, если у корешка нашлись линейки таблицы.
    """
    return side.rules_v >= TABLE_RULES_V or side.rules_h >= TABLE_RULES_H


@dataclass(frozen=True)
class Verdict:
    """Что делать с кадром.

    Attributes:
        code: "ок", "текст" либо "таблица".
        why: Пояснение одной строкой.
    """

    code: str
    why: str


def verdict(geometry: SpreadGeometry, threshold: float = THRESHOLD) -> Verdict:
    """Решает судьбу кадра: чист, восстановим по контексту или только пересканировать.

    РАЗДЕЛЕНИЕ ВАЖНО. Съеденный корешком СВЯЗНЫЙ ТЕКСТ восстанавливается по контексту:
    слово разорвано переносом, и продолжение стоит на следующей строке. Съеденная
    ТАБЛИЦА не восстанавливается никак — утраченную цифру неоткуда взять, догадка о
    ней была бы подделкой данных. Такие кадры надо снимать заново, и в отчёте они
    отделены от текстовых.

    Args:
        geometry: Разбор разворота.
        threshold: Порог по баллу.

    Returns:
        ``Verdict`` с кодом и пояснением.
    """
    if geometry.problem:
        return Verdict("ок", geometry.problem)
    score = spread_bite(geometry)
    if not np.isfinite(score) or score < threshold:
        return Verdict("ок", "поле у корешка в норме")
    bitten = [s for s in geometry.sides if np.isfinite(side_bite(s)) and side_bite(s) >= threshold]
    if any(is_tabular(s) for s in bitten or geometry.sides):
        return Verdict("таблица", "у корешка таблица — восстановить нельзя, только пересканировать")
    return Verdict("текст", "связный текст — восстановим по контексту")
