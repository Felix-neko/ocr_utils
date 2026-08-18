"""Синтетические «газетные полосы» для проверки метрик резкости.

Все страницы рисуются из одного набора штрихов, чтобы отличаться ровно одним свойством:
плотностью текста, кеглем, экспозицией или степенью размытия. Именно на таких парах и
проверяется главное требование к метрике — реагировать на фокус, а не на вёрстку.
"""

import cv2
import numpy as np
import pytest

PAPER = 235
INK = 35


def draw_page(
    height: int = 1024, width: int = 1024, *, fill: float = 1.0, stroke: int = 2, line_height: int = 12, seed: int = 0
) -> np.ndarray:
    """Рисует страницу со строками текста заданного кегля и плотности.

    Args:
        height: Высота кадра.
        width: Ширина кадра.
        fill: Доля площади, занятая текстом (1.0 — вся страница, 0.25 — четверть).
        stroke: Толщина штриха в пикселях (кегль).
        line_height: Расстояние между строками.
        seed: Зерно генератора для воспроизводимости.

    Returns:
        Полутоновый кадр uint8.
    """
    rng = np.random.default_rng(seed)
    page = np.full((height, width), float(PAPER))
    text_bottom = int(height * fill)
    for y in range(line_height, text_bottom - line_height, line_height):
        x = 40
        while x < width - 60:
            glyph = int(rng.integers(stroke * 2, stroke * 5))
            page[y : y + stroke * 4, x : x + stroke] = INK  # вертикальный штрих
            page[y : y + stroke, x : x + glyph] = INK  # горизонтальный элемент
            x += glyph + int(rng.integers(stroke, stroke * 3))
    return np.clip(page, 0, 255).astype(np.uint8)


def draw_text_lines(
    height: int = 1024,
    width: int = 1024,
    *,
    stroke: int = 2,
    line_height: int = 20,
    columns: int = 1,
    slant: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Рисует полосу построчно и возвращает полигоны строк — «выход детектора».

    Нужна для проверки режима оценки фокуса по областям строк без запуска surya.

    НАКЛОН РИСУЕТСЯ СДВИГОМ ЦЕЛЫХ ПИКСЕЛЕЙ, а не поворотом кадра. Это принципиально:
    поворот с интерполяцией сам размыл бы штрихи, и тест «наклонная строка меряется так
    же, как прямая» проверял бы не то, что нужно. При сдвиге на целое число пикселей
    рисунок штриха дословно тот же, что у прямой строки, поэтому любая разница в
    измеренной σ означает ошибку нарезки, а не рендеринга.

    ``columns`` задаёт число колонок вёрстки, и по умолчанию оно единица только ради
    простых тестов на одну строку. Для всего, что касается ЗОН, колонок нужно несколько:
    строка во всю ширину кадра своим центром тяжести всегда попадает в средний столбец
    тайлов, и боковые тайлы остаются пустыми — как на настоящей газетной полосе не бывает.

    Args:
        height: Высота кадра.
        width: Ширина кадра.
        stroke: Толщина штриха в пикселях (кегль).
        line_height: Высота строки и шаг между строками.
        columns: Число колонок вёрстки.
        slant: Наклон строк как тангенс (0.05 — примерно 3°).
        seed: Зерно генератора.

    Returns:
        Кортеж (кадр uint8, список полигонов строк (4, 2) в координатах кадра).
    """
    rng = np.random.default_rng(seed)
    page = np.full((height, width), float(PAPER))
    polygons: list[np.ndarray] = []

    left, right = 40, width - 40
    gutter = max(stroke * 4, 20)
    column_width = (right - left - gutter * (columns - 1)) / columns
    glyph_height = stroke * 4
    # Запас сверху и снизу, чтобы наклонная строка не выехала за кадр.
    margin = int(abs(slant) * column_width) + 2 * line_height

    for baseline in range(margin, height - margin, line_height * 2):
        for column in range(columns):
            x_start = int(left + column * (column_width + gutter))
            x_end = int(x_start + column_width)
            x = x_start
            while x < x_end - stroke * 5:
                dy = int(round(slant * (x - x_start)))
                glyph = int(rng.integers(stroke * 2, stroke * 5))
                page[baseline + dy : baseline + dy + glyph_height, x : x + stroke] = INK
                page[baseline + dy : baseline + dy + stroke, x : x + glyph] = INK
                x += glyph + int(rng.integers(stroke, stroke * 3))

            dy_end = slant * (x_end - x_start)
            polygons.append(
                np.array(
                    [
                        [x_start, baseline],
                        [x_end, baseline + dy_end],
                        [x_end, baseline + dy_end + glyph_height],
                        [x_start, baseline + glyph_height],
                    ],
                    dtype=np.float64,
                )
            )
    return np.clip(page, 0, 255).astype(np.uint8), polygons


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Размывает кадр гауссианой (имитация расфокуса).

    Args:
        image: Полутоновый кадр.
        sigma: Сигма размытия в пикселях; 0 — вернуть кадр как есть.

    Returns:
        Размытый кадр uint8.
    """
    if sigma <= 0:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigma)


def expose(image: np.ndarray, gain: float, offset: float = 0.0, noise: float = 0.0, seed: int = 1) -> np.ndarray:
    """Меняет экспозицию и добавляет шум (имитация другой ISO и освещения).

    Args:
        image: Полутоновый кадр.
        gain: Множитель контраста относительно средней яркости.
        offset: Сдвиг яркости в уровнях.
        noise: СКО аддитивного гауссова шума.
        seed: Зерно генератора шума.

    Returns:
        Кадр uint8 с изменённой экспозицией.
    """
    rng = np.random.default_rng(seed)
    mean = float(image.mean())
    out = (image.astype(np.float64) - mean) * gain + mean + offset
    if noise > 0:
        out = out + rng.normal(0.0, noise, image.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


@pytest.fixture
def sharp_page() -> np.ndarray:
    """Плотно набранная резкая полоса мелким шрифтом."""
    return draw_page()
