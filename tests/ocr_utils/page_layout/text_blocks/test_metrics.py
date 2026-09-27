"""Мера перескока «ступенька»: изогнутые и наклонённые строки её не дают, перескок на соседний ряд — даёт."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.text_blocks.metrics import STEP_PITCH_SHARE, jumping_of, step_of, steps_of

PITCH = 23.0
STEP_PX = 6.0  # узлы оси — через 1 мм рабочей копии 150 dpi


def _axis(ys_of, length: float = 700.0, gaps: tuple[tuple[float, float], ...] = ()) -> np.ndarray:
    """Ось длиной ``length`` с ординатами ``ys_of(x)``; ``gaps`` — участки без точек (межсловные пропуски)."""
    xs = np.arange(0.0, length + 0.1, STEP_PX)
    keep = np.ones(xs.size, dtype=bool)
    for low, high in gaps:
        keep &= (xs < low) | (xs > high)
    xs = xs[keep]
    return np.column_stack([xs, ys_of(xs)])


def _bow(sag: float, length: float = 700.0):
    """Дуга: середина ниже концов на ``sag``."""
    return lambda xs: 100.0 + sag * (1.0 - ((xs - length / 2) / (length / 2)) ** 2)


def test_bowed_and_tilted_lines_have_no_steps():
    """Дуга в один и два шага, наклон 5°, S-образная строка — ни одной ступеньки."""
    axes = [
        _axis(_bow(PITCH)),
        _axis(_bow(2 * PITCH)),
        _axis(lambda xs: 100.0 + np.tan(np.radians(5.0)) * xs),
        _axis(lambda xs: 100.0 + 0.8 * PITCH * np.sin(xs / 700.0 * 2 * np.pi)),
    ]
    assert steps_of(axes, PITCH) == 0
    # Прежняя мера «от хорды» дугу в шаг считает перескоком — ради этого новая и заведена.
    assert jumping_of(axes[:1], PITCH) == 1


def test_curled_end_is_not_a_step():
    """Завиток конца строки у корешка (полшага на последних 60 px, как «…как при» на 1966/01 IMG_0049_1L) — не ступенька."""
    axis = _axis(lambda xs: 100.0 + np.where(xs > 640.0, (xs - 640.0) / 60.0 * 0.5 * PITCH, 0.0))
    assert step_of(axis, PITCH) < STEP_PITCH_SHARE * PITCH


def test_jump_to_the_next_row_is_a_step():
    """Отвесная ступенька в шаг посередине строки и косой переход через ряд — перескок."""
    sheer = _axis(lambda xs: 100.0 + np.where(xs > 350.0, PITCH, 0.0))
    slanted = _axis(lambda xs: 100.0 + np.clip((xs - 300.0) / 60.0, 0.0, 1.0) * PITCH)
    assert steps_of([sheer, slanted], PITCH) == 2


def test_sparse_jump_across_word_gaps_is_a_step():
    """Перескок по редким точкам через пропуски (строка крупного масштаба по высоким буквам)."""
    axis = _axis(lambda xs: 100.0 + np.where(xs > 250.0, PITCH, 0.0), gaps=((130.0, 215.0), (230.0, 270.0)))
    assert step_of(axis, PITCH) > STEP_PITCH_SHARE * PITCH


def test_small_bumps_are_not_steps():
    """Подъём оси на прописных в пятую часть шага — не перескок; пропуски в точках ложной ступеньки не дают."""
    bumps = _axis(lambda xs: 100.0 + np.where((xs > 200) & (xs < 260), 0.2 * PITCH, 0.0))
    gapped = _axis(_bow(PITCH), gaps=((100.0, 150.0), (400.0, 440.0)))
    assert steps_of([bumps, gapped], PITCH) == 0
