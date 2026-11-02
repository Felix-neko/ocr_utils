"""Тесты гладкой стороны блока (``smooth_sides.side_curve``): строки с неизвестным краем (флаги переданы явно) не ломают сторону; серии выключки без отступов при коротких абзацах."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.text_blocks.alignment import ALIGN_TOL_MM
from ocr_utils.page_layout.text_blocks.smooth_sides import TREND_MODE, EdgeKind, TrendMode, side_curve

DPI = 150.0
# Левые края строк, мм, как у блока правой страницы ХиП 1991/07 IMG_0099 после Кунса: выключка по формату (4.4 мм),
# абзацные отступы и пункты списка (7.0–8.5 мм) и две строки, к которым прилипла тень сгиба у края вырезки (−0.34 мм).
LEFTS_MM = [8.3, 8.38, 7.45, 4.32, 7.03, 4.4, 6.94, 4.4, 7.28, 4.4, 4.4, 4.4,
            4.4, 4.4, 8.3, 8.3, 4.4, 4.4, 8.3, 8.47, 4.4, -0.34, -0.34, 4.57]  # fmt: skip
CUT = [21, 22]


def mm(value):
    """Мм → пиксели рабочей копии ``DPI``."""
    return np.asarray(value, dtype=np.float64) / 25.4 * DPI


def left_side(known: np.ndarray | None, mode: TrendMode | None = None):
    """Левая сторона блока по ``LEFTS_MM``: шаг строк 3.7 мм, высота полосы 2.4 мм.

    Args:
        known: Флаги строк с известным краем или None.
        mode: Способ серий и строк тренда; ``None`` — по умолчанию.

    Returns:
        ``(SideCurve, ординаты строк px)``.
    """
    ys = mm(95.0 + 3.7 * np.arange(len(LEFTS_MM)))
    half = float(mm(1.2))
    # Левая сторона — в «наружной» координате: минус x.
    return side_curve(ys, -mm(LEFTS_MM), ys - half, ys + half, float(mm(3.7)), DPI, known, mode), ys


def test_cut_rows_break_side_without_mask() -> None:
    """Прежний способ серий без флагов: две строки с краем у края картинки рассыпают серии, тренда стороны нет."""
    curve, _ = left_side(None, TrendMode.RUNS)
    assert curve.core is None


def test_cut_rows_without_mask_skip_indent() -> None:
    """Серии без отступов держат сторону и без флагов: тренд на 4.4 мм, строки у края — ступенька наружу."""
    curve, ys = left_side(None, TrendMode.SKIP_INDENT)
    assert curve.core is not None
    core_mm = -np.interp(ys, curve.ys, curve.core) * 25.4 / DPI
    assert np.all(np.abs(core_mm - 4.4) < 0.3)
    assert all(curve.kinds[i] is not EdgeKind.SPLINE for i in CUT)


def test_unknown_edges_do_not_break_side() -> None:
    """С флагами, переданными явно, сторона строится по краям выключки: тренд на 4.4 мм, строки у края — ступенька.

    Флаги здесь задаёт тест (``CUT``): в разборе правило «край ближе 0.3 мм к краю картинки — неизвестен» не
    подключено, по умолчанию известны все края (``smooth_envelope.smooth_envelope(known_left=None, …)``).
    """
    known = np.ones(len(LEFTS_MM), dtype=bool)
    known[CUT] = False
    curve, ys = left_side(known)
    assert curve.core is not None
    core_mm = -np.interp(ys, curve.ys, curve.core) * 25.4 / DPI
    assert np.all(np.abs(core_mm - 4.4) < 0.3)
    # Строки с неизвестным краем торчат наружу на 4.7 мм — ступенька (участок недостоверный), а не тренд.
    assert all(curve.kinds[i] is EdgeKind.STEP for i in CUT)
    assert len(curve.unreliable) >= len(CUT)


def test_all_known_is_default() -> None:
    """Флаги «все известны» дают ту же сторону, что и без флагов."""
    ys = mm(np.arange(10) * 4.0 + 20.0)
    lefts = mm(np.where(np.arange(10) % 3 == 0, 9.0, 5.0))
    half, pitch = float(mm(1.2)), float(mm(4.0))
    plain = side_curve(ys, -lefts, ys - half, ys + half, pitch, DPI)
    flagged = side_curve(ys, -lefts, ys - half, ys + half, pitch, DPI, np.ones(10, dtype=bool))
    assert np.allclose(plain.us, flagged.us) and plain.kinds == flagged.kinds


# Левые края строк блока статей 93–95 левой страницы ХиП 1991/07 IMG_0099 после Кунса (мм холста) и их ординаты:
# выключка по формату с остаточным прогибом 8.6 → 7.9 → 10.0 мм, абзацные отступы и пункты списка (12.9–14.6 мм)
# через 2–3 строки и два заголовка по центру (30 мм).
L99_Y_MM = [28.1, 31.8, 35.8, 39.6, 43.6, 47.6, 51.3, 55.3, 60.4, 64.3, 69.9, 73.8, 77.7, 81.6, 85.5, 89.2, 93.1,
            97.0, 100.8, 104.7, 108.6, 111.9, 115.7, 119.9, 123.8, 127.2, 130.9, 134.8, 140.2, 143.9, 149.3, 153.1,
            156.8, 160.8, 164.6]  # fmt: skip
L99_X_MM = [13.46, 8.64, 8.72, 8.64, 8.72, 8.64, 13.55, 8.64, 13.04, 30.14, 13.29, 8.47, 13.38, 8.30, 8.21, 13.12,
            8.13, 8.13, 13.04, 8.21, 8.21, 12.87, 7.87, 7.96, 8.04, 13.04, 8.04, 8.21, 13.38, 30.99, 13.89, 9.06,
            14.56, 9.82, 9.99]  # fmt: skip


def side_0099(mode: TrendMode):
    """Левая сторона блока IMG_0099 L способом ``mode``.

    Args:
        mode: Способ серий и строк тренда.

    Returns:
        ``(SideCurve, края px, ординаты px)``.
    """
    ys, xs = mm(L99_Y_MM), mm(L99_X_MM)
    half = float(mm(1.2))
    return side_curve(ys, -xs, ys - half, ys + half, float(mm(3.9)), DPI, mode=mode), xs, ys


def flush_resid_mm(curve, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Невязки строк выключки (край левее 11 мм) от тренда стороны, мм, «внутрь — плюс»."""
    core = -np.interp(ys, curve.ys, curve.core)
    flush = xs < mm(11.0)
    return (xs[flush] - core[flush]) * 25.4 / DPI


def test_short_paragraphs_runs_lose_middle() -> None:
    """Прежний способ: серии рассыпаются на коротких абзацах, тренд в середине уходит от выключки дальше допуска."""
    curve, xs, ys = side_0099(TrendMode.RUNS)
    assert curve.core is not None
    assert flush_resid_mm(curve, xs, ys).min() < -ALIGN_TOL_MM


def test_short_paragraphs_skip_indent() -> None:
    """Серии без отступов: строки выключки в середине блока идут в тренд и лежат на нём; доля в допуске — от 0.8."""
    curve, xs, ys = side_0099(TrendMode.SKIP_INDENT)
    assert curve.core is not None
    resid = flush_resid_mm(curve, xs, ys)
    assert resid.size == 21
    # Середина блока (y 74–135 мм, прежде вне серий): тренд не дальше 0.6 мм от краёв (было до 1.28 мм).
    middle = (np.asarray(L99_Y_MM)[np.asarray(L99_X_MM) < 11.0] > 70.0) & (
        np.asarray(L99_Y_MM)[np.asarray(L99_X_MM) < 11.0] < 140.0
    )
    assert np.all(np.abs(resid[middle]) <= 0.6)
    # Жёсткий сплайн (8 шагов строк) не догоняет прогиб у самого низа (последняя строка — 0.83 мм), но сторона
    # выровнена: в допуске не меньше 0.8 строк (``alignment.CORE_SHARE``).
    assert np.mean(np.abs(resid) <= ALIGN_TOL_MM) >= 0.8
    # Отступы и заголовки — внутри стороны, по сплайну, а не ступеньками.
    assert all(curve.kinds[i] is EdgeKind.SPLINE for i, x in enumerate(L99_X_MM) if x > 11.0)


def test_default_mode_is_skip_indent() -> None:
    """Способ по умолчанию — серии без отступов."""
    assert TREND_MODE is TrendMode.SKIP_INDENT
    default, _, _ = side_0099(None)
    chosen, _, _ = side_0099(TrendMode.SKIP_INDENT)
    assert np.allclose(default.core, chosen.core)


def test_ragged_right_column_stays_ragged() -> None:
    """Колонка с рваным правым краем (строки короче меры на 0–7.6 мм): правая сторона не выровнена, левая — да."""
    from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
    from ocr_utils.page_layout.text_blocks.page import analyse_gray
    from tests.ocr_utils.page_layout.text_blocks.synthetic import column_page

    for seed in range(3):
        analysis = analyse_gray(column_page(columns=1, justify="left", seed=seed), InkEngine())
        index = max(range(len(analysis.blocks)), key=lambda item: len(analysis.blocks[item].rows))
        alignment = analysis.alignments[index]
        assert alignment.left.aligned and not alignment.right.aligned
