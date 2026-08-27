"""Синтетические полосы с управляемым просветом с оборота.

Все страницы рисуются из одного набора штрихов, чтобы отличаться ровно одним свойством:
силой просвета, цветом бумаги, экспозицией, кеглем или плотностью набора. Именно на
таких парах и проверяется главное требование к метрике — реагировать на просвет,
а не на вёрстку и не на съёмку.

КАК ИМИТИРУЕТСЯ ПРОСВЕТ. Берётся вторая, независимо сгенерированная страница, ЗЕРКАЛИТСЯ
по горизонтали (лист смотрят на просвет — оборот виден отражённым), размывается (бумага
рассеивает) и подмешивается с малым весом. Порядок именно такой: сначала зеркало, потом
размытие, иначе тест прошёл бы и на незеркальном подмесе, а зеркальность — единственное,
что физически отличает просвет от грязи на бумаге.

СНИМОК, А НЕ ВЁРСТКА. Готовая страница обязательно прогоняется через ``scan()`` —
лёгкое размытие, имитирующее разрешающую способность съёмки. Без него синтетика
получается неправдоподобно контрастной: у штрихов нет полутоновой каймы, гистограмма
вырождается в два пика, и порог Оцу садится вплотную к краске (0.19 вместо 0.64 на
настоящем скане). Метрика ``ghost_ink``, которая этот порог и моделирует, на такой
странице не работала бы вовсе — и тест проверял бы не то, что нужно.

СТРАНИЦА С ПОЛЯМИ. У всех страниц есть чистые поля: основная метрика меряет межстрочья
ОТНОСИТЕЛЬНО полей, и на странице без полей её просто нельзя посчитать.
"""

import numpy as np
import pytest

import cv2

PAPER = 238
INK = 40

# Поля в долях стороны: у настоящей журнальной полосы примерно столько.
MARGIN_X = 0.12
MARGIN_Y = 0.09


def draw_page(
    height: int = 1400,
    width: int = 1000,
    *,
    fill: float = 1.0,
    stroke: int = 3,
    line_height: int = 26,
    columns: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Рисует полосу со строками текста заданного кегля и плотности.

    Args:
        height: Высота кадра.
        width: Ширина кадра.
        fill: Какую долю высоты наборной полосы занять текстом.
        stroke: Толщина штриха в пикселях (кегль).
        line_height: Шаг строк.
        columns: Число колонок набора.
        seed: Зерно генератора для воспроизводимости.

    Returns:
        Полутоновый кадр uint8: бумага светлая, краска тёмная.
    """
    rng = np.random.default_rng(seed)
    page = np.full((height, width), float(PAPER))
    x0, x1 = int(width * MARGIN_X), int(width * (1 - MARGIN_X))
    y0, y1 = int(height * MARGIN_Y), int(height * (1 - MARGIN_Y))
    column_width = (x1 - x0) // columns
    bottom = y0 + int((y1 - y0) * fill)

    for column in range(columns):
        left = x0 + column * column_width
        right = left + column_width - stroke * 4
        for y in range(y0, bottom - line_height, line_height):
            x = left
            while x < right - stroke * 5:
                glyph = int(rng.integers(stroke * 2, stroke * 5))
                page[y : y + stroke * 4, x : x + stroke] = INK  # вертикальный штрих
                page[y : y + stroke, x : x + glyph] = INK  # горизонтальный элемент
                x += glyph + int(rng.integers(stroke, stroke * 3))
    return np.clip(page, 0, 255).astype(np.uint8)


def add_show_through(page: np.ndarray, strength: float, *, blur: float = 1.6, seed: int = 100) -> np.ndarray:
    """Подмешивает в полосу зеркальный, размытый и ослабленный текст с оборота.

    Args:
        page: Лицевая полоса.
        strength: Сила просвета: 0.0 — ничего, 0.25 — тяжёлый случай.
        blur: Сигма рассеяния в бумаге.
        seed: Зерно генератора оборотной страницы.

    Returns:
        Полутоновый кадр uint8 того же размера.
    """
    if strength <= 0.0:
        return page.copy()
    verso = draw_page(page.shape[0], page.shape[1], seed=seed)
    ghost = cv2.GaussianBlur(verso[:, ::-1].astype(np.float32), (0, 0), blur)
    # Подмешивается именно ЗАТЕМНЕНИЕ относительно бумаги, а не сама картинка: иначе
    # просвет осветлял бы настоящие буквы, чего в природе не бывает.
    darkening = np.clip(PAPER - ghost, 0.0, None)
    return np.clip(page.astype(np.float32) - strength * darkening, 0, 255).astype(np.uint8)


def scan(page: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Имитация съёмки: лёгкое размытие, дающее штрихам полутоновую кайму.

    Сигма подобрана по факту: при 0.8 порог Оцу синтетической полосы (0.64) совпадает
    с замеренным на настоящих сканах пака (0.655).

    Args:
        page: Страница.
        sigma: Сигма размытия в пикселях.

    Returns:
        Полутоновый кадр uint8.
    """
    return np.clip(cv2.GaussianBlur(page.astype(np.float32), (0, 0), sigma), 0, 255).astype(np.uint8)


def add_stains(page: np.ndarray, count: int = 60, depth: float = 30.0, seed: int = 7) -> np.ndarray:
    """Ставит на бумагу гладкие пятна — имитация лисьих пятен и грязи.

    Пятна попадают в тот же диапазон уровней, что и просвет, но не имеют штриховой
    структуры. Метрика обязана их игнорировать, иначе грязная бумага пойдёт на
    пересканирование наравне с просвечивающей.

    Args:
        page: Полоса.
        count: Сколько пятен поставить.
        depth: Насколько пятно темнее бумаги.
        seed: Зерно генератора.

    Returns:
        Полутоновый кадр uint8.
    """
    rng = np.random.default_rng(seed)
    height, width = page.shape
    canvas = np.zeros((height, width), np.float32)
    for _ in range(count):
        cy, cx = int(rng.integers(0, height)), int(rng.integers(0, width))
        radius = int(rng.integers(12, 30))
        cv2.circle(canvas, (cx, cy), radius, float(depth), -1)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 9.0)
    return np.clip(page.astype(np.float32) - canvas, 0, 255).astype(np.uint8)


def expose(page: np.ndarray, gain: float = 1.0, offset: float = 0.0, noise: float = 0.0, seed: int = 1) -> np.ndarray:
    """Меняет экспозицию и цвет бумаги — имитация другого года съёмки.

    Args:
        page: Полоса.
        gain: Множитель яркости.
        offset: Сдвиг яркости.
        noise: Сигма гауссова шума.
        seed: Зерно генератора шума.

    Returns:
        Полутоновый кадр uint8.
    """
    out = page.astype(np.float32) * gain + offset
    if noise > 0:
        out = out + np.random.default_rng(seed).normal(0.0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def spread(left: np.ndarray, right: np.ndarray, gutter: int = 90) -> np.ndarray:
    """Складывает две полосы в кадр-разворот с тёмной полосой корешка между ними.

    Args:
        left: Левая полоса.
        right: Правая полоса.
        gutter: Ширина корешкового провала в пикселях.

    Returns:
        Полутоновый кадр-разворот uint8.
    """
    height = min(left.shape[0], right.shape[0])
    middle = np.full((height, gutter), float(PAPER))
    middle[:, gutter // 2 - 2 : gutter // 2 + 2] = INK
    return np.hstack([left[:height], middle.astype(np.uint8), right[:height]])


@pytest.fixture
def clean_page() -> np.ndarray:
    """Полоса без просвета."""
    return scan(draw_page())


@pytest.fixture
def bleeding_page() -> np.ndarray:
    """Полоса с сильным просветом с оборота."""
    return scan(add_show_through(draw_page(), 0.36))
