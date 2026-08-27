"""Деление кадра-разворота на две полосы по корешку.

Просвет — дефект ОДНОЙ полосы, а не разворота: у кадра ``02-03_0146`` из размеченной
папки просвечивает только правая страница, левая чистая. Поэтому мерить надо каждую
половину отдельно, и делить их надо по настоящему корешку, а не пополам: корешок гуляет
по кадру на сотни пикселей от съёмки к съёмке.

СПОСОБ. Корешок ищется как самая широкая ВПАДИНА профиля краски по столбцам в
центральной части кадра. Именно впадина, а не тёмная линия: сама линия корешка узкая
и после сглаживания окном в проценты ширины кадра почти не видна, а вот два внутренних
поля по её сторонам дают уверенный провал плотности набора шириной в сотни пикселей.
"""

from dataclasses import dataclass

import numpy as np

import cv2

# Ширина полосы поиска вокруг центра кадра, в долях ширины. Корешок на съёмке с рук
# уезжает от центра, но не на четверть кадра.
SEARCH_BAND = 0.20

# Окно сглаживания профиля краски, в долях ширины. Порядка внутреннего поля: узкую
# тень корешка съедает, широкий провал между наборными полосами оставляет.
SMOOTH_FRACTION = 0.03

# До какой ширины уменьшать кадр перед поиском. Признак крупный, гонять его по
# 11 мегапикселям незачем.
WORK_WIDTH = 1200

# Уровень краски относительно локальной бумаги (та же граница, что в zones.INK_LEVEL).
INK_LEVEL = 0.65

# Насколько провал должен быть глубже медианы профиля, чтобы ему поверить.
MIN_RELATIVE_DEPTH = 0.5

# Доля глубины провала, по которой очерчивается его ШИРИНА. Корешок берётся серединой
# этой ширины, а не точкой минимума: между наборными полосами лежит плато из двух
# внутренних полей, и минимум внутри плато выбирается шумом — он гуляет на сотню
# пикселей от кадра к кадру.
VALLEY_LEVEL = 0.25

# Ширина окна, в котором очерчивается провал, в долях ширины кадра. Шире полосы поиска
# (сам провал шире неё, и упёршийся в границу край сдвинул бы середину), но не весь кадр:
# иначе в провал засчитались бы наружные поля разворота.
VALLEY_WINDOW = 0.50

# Ниже этого отношения ширины к высоте кадр считается ОДИНОЧНОЙ страницей (обложка,
# вклейка, титул) и не делится вовсе. Разворот пака заметно шире высоты (1.35–1.42),
# одиночная страница заметно уже (около 0.71) — граница проходит по пустому месту.
SPREAD_MIN_ASPECT = 1.15

LEFT, RIGHT, WHOLE = "L", "R", "-"
SIDES = (LEFT, RIGHT)


@dataclass(frozen=True)
class Spread:
    """Найденное деление кадра.

    Attributes:
        gutter: Столбец корешка в пикселях исходного кадра.
        confident: Нашёлся ли уверенный провал; False — деление взято пополам.
    """

    gutter: int
    confident: bool


def find_gutter(gray: np.ndarray) -> Spread:
    """Ищет столбец корешка разворота.

    Args:
        gray: Полутоновый кадр-разворот.

    Returns:
        ``Spread`` со столбцом корешка в координатах исходного кадра.
    """
    height, width = gray.shape[:2]
    middle = width // 2
    if width < 64:
        return Spread(gutter=middle, confident=False)

    scale = min(1.0, WORK_WIDTH / width)
    small = cv2.resize(gray, (max(16, int(width * scale)), max(16, int(height * scale))), interpolation=cv2.INTER_AREA)
    work = small.astype(np.float32)

    # Локальная бумага — грубо, лишь бы снять неравномерность света: точность здесь
    # не нужна, речь о том, где текста МНОГО, а где его нет.
    side = max(3, (small.shape[1] // 20) * 2 + 1)
    paper = cv2.blur(cv2.dilate(work, np.ones((side, side), np.uint8)), (side, side))
    ink = (work < INK_LEVEL * np.maximum(paper, 1.0)).astype(np.float32)

    profile = ink.mean(axis=0)
    window = max(3, int(round(small.shape[1] * SMOOTH_FRACTION)) | 1)
    profile = cv2.blur(profile.reshape(1, -1), (window, 1)).ravel()

    centre = small.shape[1] // 2
    half_band = max(2, int(round(small.shape[1] * SEARCH_BAND / 2)))
    lo, hi = max(0, centre - half_band), min(small.shape[1], centre + half_band + 1)
    band = profile[lo:hi]
    if band.size == 0:
        return Spread(gutter=middle, confident=False)

    bottom = float(band.min())
    reference = float(np.median(profile))
    depth = (reference - bottom) / max(reference, 1e-6)
    if depth < MIN_RELATIVE_DEPTH:
        # Кадр без выраженного корешка: одиночная страница, разворот в обрез, пустой
        # лист. Делим пополам — хуже, чем по корешку, но лучше, чем не делить вовсе.
        return Spread(gutter=middle, confident=False)

    # Края провала, а не его дно: тень самого корешка иногда читается как краска и
    # разрывает плато надвое, поэтому берём КРАЙНИЕ точки ниже уровня, а не связный кусок.
    # Края ищутся по ПОЛНОМУ профилю, а не внутри полосы поиска: провал шире полосы,
    # и упёршийся в её границу край сдвинул бы середину на сотню пикселей.
    # Края берутся КРАЙНИМИ точками ниже уровня, а не связным куском: тень самого
    # корешка местами читается как краска и разрывает плато надвое.
    level = bottom + VALLEY_LEVEL * (reference - bottom)
    half_window = max(4, int(round(small.shape[1] * VALLEY_WINDOW / 2)))
    wlo, whi = max(0, centre - half_window), min(small.shape[1], centre + half_window + 1)
    inside = np.flatnonzero(profile[wlo:whi] <= level)
    if inside.size == 0:
        return Spread(gutter=middle, confident=False)
    best = (int(inside[0]) + int(inside[-1])) / 2.0 + wlo
    return Spread(gutter=int(round(best / scale)), confident=True)


def split_spread(gray: np.ndarray, spread: Spread | None = None) -> list[tuple[str, np.ndarray]]:
    """Режет кадр на две полосы по корешку.

    Args:
        gray: Полутоновый кадр-разворот.
        spread: Готовый результат ``find_gutter``; None — посчитать заново.

    Returns:
        Список пар (сторона, полоса) в порядке «левая, правая». Для одиночной страницы —
        один элемент со стороной ``WHOLE``. Пустые куски (корешок уехал к самому краю
        кадра) отбрасываются.
    """
    height, width = gray.shape[:2]
    if width < SPREAD_MIN_ASPECT * height:
        return [(WHOLE, gray)]
    spread = spread or find_gutter(gray)
    gutter = spread.gutter
    halves = [(LEFT, gray[:, :gutter]), (RIGHT, gray[:, gutter:])]
    return [(side, half) for side, half in halves if half.shape[1] >= 64]
