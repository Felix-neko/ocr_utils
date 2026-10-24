"""Гладкие боковые стороны блока: одна кривая x(y) на сторону по краям строк — сплайн по выровненным сериям, горбы на висячей пунктуации, ступеньки только на настоящей разнице длин строк и на выносах за колонку (недостоверные участки)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ocr_utils.page_layout import mm_to_px
from ocr_utils.page_layout.text_blocks.alignment import ALIGN_TOL_MM
from ocr_utils.page_layout.text_blocks.sides import PARABOLA_GAIN_ROWS, _fit_robust, _statuses, aligned_runs

# Вынос края строки наружу за сторону не больше этого — висячая пунктуация, дефис, кавычка, соринка у
# конца строки (1966/01 IMG_0004_2R, «поставленных»: 1.7 мм — ось дотянута до пятнышка за словом):
# сторона обходит его гладким горбом. Больше — строка шире колонки или пометка на полях: честная
# ступенька, участок стороны недостоверный.
PUNCT_MM = 2.5
# Разброс краёв выровненных строк, который сторона поглощает сдвигом наружу (квантиль остатков).
JITTER_MM = 0.6
JITTER_QUANTILE = 0.9
# На невыровненной стороне соседние края, отличающиеся меньше этого, — шум одной ступени: они
# сводятся к общей прямой, а ступенька остаётся только при разнице больше.
STEP_MM = 2.0
# Жёсткость сплайна стороны: длина сглаживания в шагах строк — трапецию и изгиб у корешка кривая
# повторяет, дрожание краёв в пару пикселей — нет.
SMOOTH_PITCHES = 8.0
# Слабая привязка сплайна к устойчивой подгонке (без неё сетка вне данных не определена).
ANCHOR_WEIGHT = 1e-4
# Сторона выровнена и без серий, если на кривой не меньше этой доли строк (висячий отступ в оглавлении).
HALF_ON_SHARE = 0.5
# Устойчивая кривая стороны строится по краям не дальше этого внутрь от наружного края соседей (±OUTER_WINDOW строк),
# мм: концы абзацев и строки без висячего знака её не тянут внутрь (:func:`_outer_rows`).
OUTER_KEEP_MM = 3.0
OUTER_WINDOW = 3
# ...или если на кривой не меньше этой доли строк, не ушедших внутрь (набор по формату с короткими абзацами).
FLUSH_ON_SHARE = 0.75
# Шаг сетки стороны по высоте, пиксели рабочей копии.
GRID_STEP_PX = 1.0


class EdgeKind(str, Enum):
    """Как край строки входит в сторону."""

    SPLINE = "spline"  # на выровненной серии: сторона идёт по сплайну (короткие строки добиты до него)
    BUMP = "bump"  # на серии, но край чуть снаружи (пунктуация): гладкий горб
    STEP = "step"  # вынос за колонку: ступенька по краю строки, участок недостоверный
    OWN = "own"  # вне серии (рваный край, заголовок): по своему краю, сглаженному внутри ступени


@dataclass(frozen=True)
class SideCurve:
    """Сторона блока: кривая ``u(y)`` в координате «наружу — больше» и недостоверные участки.

    Attributes:
        ys: Ординаты сетки сверху вниз, пиксели рабочей копии.
        us: Координата стороны в «наружной» оси: у правой стороны это x, у левой — минус x.
        unreliable: Отрезки ``(y0, y1)``, где сторона — ступенька по выносу за колонку.
        kinds: Как вошёл край каждой строки.
        core: Тренд стороны по телу блока на той же сетке или ``None``.
    """

    ys: np.ndarray
    us: np.ndarray
    unreliable: tuple[tuple[float, float], ...]
    kinds: tuple[EdgeKind, ...]
    # Тренд стороны по телу блока: сплайн по краям строк «на кривой» без сдвига наружу, без горбов и
    # ступенек (для мер выключки, как ``core_*`` прежней огибающей); ``None`` — сторона не выровнена.
    core: np.ndarray | None = None


def _spline(
    ys: np.ndarray, us: np.ndarray, weights: np.ndarray, grid: np.ndarray, pitch: float, anchor: np.ndarray
) -> np.ndarray:
    """Штрафной сплайн ``u(y)`` на сетке: данные плюс вторая разность плюс слабая привязка к ``anchor``.

    Args:
        ys, us, weights: Края строк и их веса (0 — не участвует).
        grid: Ординаты сетки.
        pitch: Шаг строк, пиксели (задаёт длину сглаживания).
        anchor: Опорная кривая на сетке.

    Returns:
        Значения на сетке.
    """
    count = grid.size
    step = grid[1] - grid[0] if count > 1 else 1.0
    position = np.clip((ys - grid[0]) / step, 0.0, count - 1.000001)
    left = np.floor(position).astype(int)
    frac = position - left
    right = np.minimum(left + 1, count - 1)
    rows = np.repeat(np.arange(ys.size), 2)
    cols = np.column_stack([left, right]).ravel()
    values = np.column_stack([1.0 - frac, frac]).ravel()
    interp = sparse.csr_matrix((values, (rows, cols)), shape=(ys.size, count))
    second = sparse.diags(
        [np.ones(count - 2), -2.0 * np.ones(count - 2), np.ones(count - 2)], [0, 1, 2], shape=(count - 2, count)
    )
    # Длина сглаживания L: вес второй разности (L/шаг)^4 на плотность данных — тогда кривая гнётся не
    # круче, чем на длине L, при любом числе опорных строк (редкие строки на кривой не дают ей гулять).
    length = SMOOTH_PITCHES * max(pitch, 1.0)
    density = max(float(weights.sum()), 1.0) / count
    stiffness = (length / step) ** 4 * density
    system = (
        interp.T @ sparse.diags(weights) @ interp
        + stiffness * (second.T @ second)
        + ANCHOR_WEIGHT * sparse.identity(count)
    )
    rhs = interp.T @ (weights * us) + ANCHOR_WEIGHT * anchor
    return np.asarray(spsolve(system.tocsc(), rhs), dtype=np.float64)


def _clusters(us: np.ndarray, limit: float) -> list[tuple[int, int]]:
    """Подряд идущие строки, соседние края которых отличаются меньше ``limit``: ``[(начало, конец+1)]``."""
    out, start = [], 0
    for index in range(1, us.size + 1):
        if index == us.size or abs(us[index] - us[index - 1]) >= limit:
            out.append((start, index))
            start = index
    return out


def _on_count(coef: np.ndarray, ys: np.ndarray, us: np.ndarray, tol: float) -> int:
    """Сколько краёв строк лежит на кривой ``coef`` с допуском ``tol`` (пиксели)."""
    return int((np.abs(us - np.polyval(coef, ys)) <= tol).sum())


def _outer_rows(ys: np.ndarray, us: np.ndarray, dpi: float) -> tuple[np.ndarray, np.ndarray]:
    """Края строк для устойчивой кривой стороны — без ушедших внутрь дальше ``OUTER_KEEP_MM`` от наружного края соседей.

    Устойчивая кривая (медиана сдвигов у Тейла—Сена) уходила внутрь, когда коротких строк в блоке почти половина:
    в библиографии по формату с записями в 2–3 строки (ХиП 1991/01 IMG_1030, итог) из девяти правых краёв четыре —
    концы записей (22–93 мм при колонке 110 мм), ещё два без висячего тире (105.6–105.9 мм); кривая встала на 105.6 мм,
    и доля строк на ней — 0.40, сторона «рваная». Наружный край соседей — максимум ``us`` по ±``OUTER_WINDOW`` строкам.
    Если отобранных меньше трёх — берутся все края. Кривая по ним — основной кандидат; кривая по всем краям берётся,
    если вписывает заметно больше строк: на стороне с разбросом краёв около миллиметра кривая по наружным уходила
    к верхней кучке краёв (01/IMG_1021, итог, левая страница: 7 строк на кривой против 11).

    Args:
        ys: Середины строк сверху вниз.
        us: Края строк в «наружной» координате.
        dpi: Разрешение рабочей копии.

    Returns:
        ``(ys, us)`` отобранных строк.
    """
    keep_px = mm_to_px(OUTER_KEEP_MM, dpi)
    outer = np.array([us[max(0, i - OUTER_WINDOW) : i + OUTER_WINDOW + 1].max() for i in range(us.size)])
    keep = us >= outer - keep_px
    return (ys[keep], us[keep]) if keep.sum() >= 3 else (ys, us)


def side_curve(
    ys: np.ndarray, us: np.ndarray, tops: np.ndarray, bottoms: np.ndarray, pitch: float, dpi: float
) -> SideCurve:
    """Гладкая сторона блока по краям его строк.

    1. Устойчивая кривая по краям (Тейл–Сен или RANSAC-парабола, ``sides._fit_robust``) и статусы
       концов «на кривой / отступ / мимо» (``sides._statuses``); выровненные серии —
       ``sides.aligned_runs``.
    2. На сериях — сплайн по краям «на кривой», сдвинутый наружу на разброс краёв (не больше
       ``JITTER_MM``). Короткие строки добиваются до него, край чуть снаружи (до ``PUNCT_MM``) — гладкий
       горб на высоте строки, край дальше — ступенька и недостоверный участок.
    3. Вне серий — свои края; соседние края ближе ``STEP_MM`` сводятся к общей прямой, сдвинутой
       наружу до крайнего из них.

    Args:
        ys: Середины строк сверху вниз.
        us: Края строк в «наружной» координате.
        tops, bottoms: Верх и низ полосы каждой строки (пиксели).
        pitch: Шаг строк.
        dpi: Разрешение рабочей копии.

    Returns:
        :class:`SideCurve` на сетке от верха первой строки до низа последней.
    """
    n = ys.size
    punct, jitter, step_px = mm_to_px(PUNCT_MM, dpi), mm_to_px(JITTER_MM, dpi), mm_to_px(STEP_MM, dpi)
    grid = np.arange(float(np.floor(tops.min())), float(np.ceil(bottoms.max())) + GRID_STEP_PX, GRID_STEP_PX)
    # Границы зоны каждой строки по высоте: середины промежутков к соседям.
    edges = np.concatenate([[grid[0]], (bottoms[:-1] + tops[1:]) / 2.0, [grid[-1] + 1.0]])
    owner = np.clip(np.searchsorted(edges, grid, side="right") - 1, 0, n - 1)
    kinds = [EdgeKind.OWN] * n
    target = us.astype(np.float64).copy()
    spline = None
    core = None
    member = np.zeros(n, dtype=bool)
    if n >= 3:
        # Кривая по наружным краям (:func:`_outer_rows`); по всем краям — только если на неё ложится заметно больше строк
        # (на ``PARABOLA_GAIN_ROWS``): у библиографии кривая по всем краям шла косо через концы записей и вписывала на
        # одну строку больше (01/IMG_1030, итог: 109.9 → 93.6 мм, 4 строки против 3).
        tol_px = mm_to_px(ALIGN_TOL_MM, dpi)
        outer_coef, all_coef = _fit_robust(*_outer_rows(ys, us, dpi), tol_px)[0], _fit_robust(ys, us, tol_px)[0]
        gain = _on_count(all_coef, ys, us, tol_px) - _on_count(outer_coef, ys, us, tol_px)
        coef = all_coef if gain >= PARABOLA_GAIN_ROWS else outer_coef
        fitted = np.polyval(coef, ys)
        # Остаток «внутрь — плюс» в мм: у наружной координаты внутрь — меньше.
        resid_mm = (fitted - us) * 25.4 / dpi
        statuses = _statuses(resid_mm)
        member = aligned_runs(statuses)
        on_all = np.array([status.value == "on" for status in statuses])
        # Висячий отступ (оглавление, список: вторая строка записи отодвинута внутрь через одну) даёт
        # долю строк «на кривой» ниже порога серий (0.75), и сторона шла лесенкой по каждой строке
        # (1968/12 IMG_0149_1L). Сторона выровнена и тогда, когда на кривой не меньше половины строк
        # и никто не торчит наружу дальше пунктуации: строки внутрь между ними добиваются ниже.
        outward = (us - fitted) > punct
        # Набор по формату с короткими абзацами (библиография: запись в 2–3 строки, ХиП 1991/01 IMG_1030) — строки
        # на кривой разделены концами записей, серий нет, и строк на кривой меньше половины блока; но среди строк,
        # не ушедших внутрь (не «отступ»), на кривой почти все. Тогда сторона тоже выровнена.
        inward = np.array([status.value == "indent" for status in statuses])
        flush = on_all.sum() >= 3 and on_all.sum() >= FLUSH_ON_SHARE * max(int((~inward).sum()), 1)
        if not member.any() and (on_all.mean() >= HALF_ON_SHARE or flush) and on_all.sum() >= 3 and not outward.any():
            member = on_all.copy()
        on = on_all & member
        if on.sum() >= 3:
            anchor = np.polyval(coef, grid)
            spline = _spline(ys, us, on.astype(np.float64), grid, pitch, anchor)
            at_rows = np.interp(ys, grid, spline)
            outward = np.maximum(us[on] - at_rows[on], 0.0)
            core = spline.copy()
            spline = spline + min(float(np.quantile(outward, JITTER_QUANTILE)) if outward.size else 0.0, jitter)
    at_rows = np.interp(ys, grid, spline) if spline is not None else None
    if n < 3:
        on_all = np.zeros(n, dtype=bool)
    # Сторона держится не только на самих сериях: строка, ушедшая ВНУТРЬ, между выровненными строками
    # (или сразу за крайней из них) стороны не рвёт, сколько бы таких строк ни шло подряд — конец
    # абзаца и абзацный отступ следующего стоят двумя строками внутрь (1966/01 IMG_0041_2R), и сторона
    # проваливалась к ним ступенькой на 6 мм.
    if spline is not None and member.any():
        # Короткие строки у верха и низа блока (конец последнего абзаца, подпись в подбор справа —
        # 1970/03 IMG_0148_2R, «Э. САВИНА») тоже добиваются: иначе у конца блока оставалась ступенька.
        member = member | (us <= at_rows)
    for index in range(n):
        if spline is None or not member[index]:
            continue
        excess = us[index] - at_rows[index]
        if excess <= 0:
            kinds[index] = EdgeKind.SPLINE
        elif excess <= punct:
            kinds[index] = EdgeKind.BUMP
        else:
            kinds[index] = EdgeKind.STEP
    # Невыровненные строки: кластеры близких краёв сводятся к общей прямой, сдвинутой наружу.
    own = [index for index in range(n) if kinds[index] is EdgeKind.OWN]
    if own:
        runs, start = [], 0
        for position in range(1, len(own) + 1):
            if position == len(own) or own[position] != own[position - 1] + 1:
                runs.append(own[start:position])
                start = position
        for run in runs:
            values = us[run]
            for a, b in _clusters(values, step_px):
                part = run[a:b]
                if len(part) >= 2:
                    line = np.polyfit(ys[part], us[part], 1)
                    shift = float(np.max(us[part] - np.polyval(line, ys[part])))
                    target[part] = np.polyval(line, ys[part]) + shift
    # Кривая на сетке.
    out = np.empty(grid.size)
    for row in range(n):
        zone = owner == row
        if kinds[row] in (EdgeKind.SPLINE, EdgeKind.BUMP):
            out[zone] = spline[zone]
        elif kinds[row] is EdgeKind.STEP:
            out[zone] = np.maximum(spline[zone], us[row])
        else:
            out[zone] = target[row]
    # Горбы пунктуации: косинусный подъём шириной — высота строки плюс по два шага строки вверх и вниз (пологий склон — не ступенька).
    # Горбы соседних строк не складываются: берётся наибольший.
    bump = np.zeros(grid.size)
    for row in range(n):
        if kinds[row] is not EdgeKind.BUMP:
            continue
        excess = us[row] - np.interp(ys[row], grid, out)
        if excess <= 0:
            continue
        flat = (bottoms[row] - tops[row]) / 2.0
        half = flat + 2.0 * pitch
        distance = np.abs(grid - ys[row])
        ramp = np.clip((distance - flat) / max(half - flat, 1.0), 0.0, 1.0)
        shape = np.where(distance <= half, 0.5 * (1.0 + np.cos(np.pi * ramp)), 0.0)
        bump = np.maximum(bump, excess * shape)
    out = out + bump
    unreliable = []
    for row in range(n):
        if kinds[row] is EdgeKind.STEP:
            zone = grid[owner == row]
            # Ряд-ступенька может не владеть ни одним узлом сетки (узкий ряд между соседями) — тогда и недостоверного
            # участка нет (ПХ 1975/03 IMG_0043_L после выпрямления: разбор падал на min() пустого массива).
            if zone.size:
                unreliable.append((float(zone.min()), float(zone.max())))
    return SideCurve(ys=grid, us=out, unreliable=tuple(unreliable), kinds=tuple(kinds), core=core)


__all__ = ["EdgeKind", "SideCurve", "side_curve"]
