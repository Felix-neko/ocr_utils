"""Гладкие боковые стороны блока: одна кривая x(y) на сторону по краям строк — сплайн по выровненным сериям, горбы на висячей пунктуации, ступеньки только на настоящей разнице длин строк и на выносах за колонку (недостоверные участки)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ocr_utils.page_layout import mm_to_px
from ocr_utils.page_layout.text_blocks.alignment import ALIGN_TOL_MM
from ocr_utils.page_layout.text_blocks.sides import PARABOLA_GAIN_ROWS, RowStatus, _fit_robust, _statuses, aligned_runs

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
# Длина сглаживания тренда в мм высоты блока у способа ``TrendMode.FLEX`` (вместо ``SMOOTH_PITCHES`` шагов строк:
# при шаге 3.9 мм это 31 мм, здесь — втрое гибче).
FLEX_SMOOTH_MM = 10.0
# Разрыв распределения отклонений краёв от устойчивой кривой, отделяющий выключку от отступов и концовок, у способа
# ``TrendMode.FLUSH_GAP``, мм (выключка ХиП 1991/07 IMG_0099 L лежит в 0.8 мм, абзацные отступы — на 4–5 мм внутрь).
FLUSH_GAP_MM = 1.5


class TrendMode(str, Enum):
    """По каким строкам строится тренд (сплайн) выровненной стороны и как считаются её серии."""

    # Прежний: серии — подряд идущие строки «на кривой» (``sides.aligned_runs``): одиночный отступ между ними серию не
    # рвёт, но отступов в серии не больше четверти. При коротких абзацах (отступ через 2–3 строки) серии
    # рассыпались, тренд шёл только по уцелевшим краям у верха и низа блока и в середине уходил от выключки на
    # 1.3 мм (ХиП 1991/07 IMG_0099 L).
    RUNS = "runs"
    # A: как ``RUNS``, но длина сглаживания сплайна — ``FLEX_SMOOTH_MM`` высоты, а не ``SMOOTH_PITCHES`` шагов строк.
    # Отвергнут (регрессия 33 кадров ХиП 1991, 2026-10-25): IMG_0099 L чинит, но гибкий тренд тащит дрожание краёв в
    # поле выпрямления — разброс сторон итога медиана 0.049 → 0.073 мм, p95 0.166 → 0.198 мм.
    FLEX = "flex"
    # B: серии — как ``RUNS``; тренд — по всем краям внешнего сгустка отклонений (до первого разрыва
    # ``FLUSH_GAP_MM`` внутрь), если хотя бы одна серия есть. Отвергнут (та же регрессия): меняет 15 кадров из 33,
    # теряет две стороны блоков (05/IMG_0307), p95 сторон у 07/IMG_0109 0.41 → 0.49 мм.
    FLUSH_GAP = "flush_gap"
    # D: серии ищутся по строкам без отступов — строки, ушедшие внутрь от ``sides.INDENT_MIN_MM`` (абзацный отступ,
    # пункт списка, конец абзаца, заголовок по центру), серию не рвут и в её долю не входят; в сплайн идут все
    # строки «на кривой» в сериях. Выбран (та же регрессия): меняет 9 кадров из 33, стороны блоков 213 → 215,
    # медиана и p95 разброса сторон итога 0.049 и 0.166 → 0.165 мм, IMG_0099 L — левая сторона выровнена.
    SKIP_INDENT = "skip_indent"


# Способ тренда по умолчанию (``side_curve`` без ``mode``).
TREND_MODE = TrendMode.SKIP_INDENT


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
    ys: np.ndarray,
    us: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
    pitch: float,
    anchor: np.ndarray,
    length: float | None = None,
) -> np.ndarray:
    """Штрафной сплайн ``u(y)`` на сетке: данные плюс вторая разность плюс слабая привязка к ``anchor``.

    Args:
        ys, us, weights: Края строк и их веса (0 — не участвует).
        grid: Ординаты сетки.
        pitch: Шаг строк, пиксели (задаёт длину сглаживания ``SMOOTH_PITCHES`` шагов, если ``length`` не задана).
        anchor: Опорная кривая на сетке.
        length: Длина сглаживания, пиксели; ``None`` — ``SMOOTH_PITCHES`` шагов строк.

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
    length = SMOOTH_PITCHES * max(pitch, 1.0) if length is None else max(length, 1.0)
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


def _runs_skip_indent(statuses: list[RowStatus], outward: np.ndarray) -> np.ndarray:
    """Выровненные серии по строкам без отступов (:attr:`TrendMode.SKIP_INDENT`).

    Строки «отступ» (ушли внутрь от ``INDENT_MIN_MM``: абзацный отступ, пункт списка, конец абзаца, заголовок по центру)
    из последовательности вынимаются, серии ищутся ``sides.aligned_runs`` по оставшимся строкам, флаги возвращаются
    на места. Отступ серию не рвёт и в её долю не входит: в статье с короткими абзацами (ХиП 1991/07 IMG_0099 L,
    статьи 93–95) отступ стоит через 2–3 строки, два отступа подряд (абзац после заголовка) рвали серию, а в
    уцелевшей серии отступов было 5 из 18 (0.28 > 0.25), и 13 строк выключки в тренд не шли.

    Серия, в которой есть строка, торчащая наружу дальше пунктуации (``outward``), так не принимается: на её участке
    остаются серии прежнего хода (``aligned_runs`` по всем строкам). Без отступов в последовательности случайные
    совпадения краёв рваного набора смыкаются в серию: у края 800, 740, 790, 700, 760, 720, 800, 690 px косая кривая
    через 800, 790, 760, 690 вписывала 4 строки из 5 оставшихся, хотя строка 800 у низа торчала наружу на 14 мм
    (``tests/research/block_envelopes``, ``test_ragged_edge_keeps_steps``).

    Args:
        statuses: Положения концов строк сверху вниз.
        outward: Флаги строк, торчащих наружу от кривой дальше ``PUNCT_MM`` ``(n,)``.

    Returns:
        Флаги ``(n,)`` строк в сериях; строки-отступы внутри принятой серии — ``False`` (их добивает до стороны
        :func:`side_curve`).
    """
    keep = np.nonzero(np.array([status is not RowStatus.INDENT for status in statuses], dtype=bool))[0]
    member = np.zeros(len(statuses), dtype=bool)
    if keep.size == 0:
        return member
    reduced = aligned_runs([statuses[index] for index in keep])
    fallback = aligned_runs(statuses)
    # Серии в укороченной последовательности: подряд идущие флаги; участок серии в исходных строках — от её первой
    # до последней строки.
    position = 0
    while position < keep.size:
        if not reduced[position]:
            position += 1
            continue
        end = position
        while end < keep.size and reduced[end]:
            end += 1
        first, last = int(keep[position]), int(keep[end - 1])
        if outward[first : last + 1].any():
            member[first : last + 1] = fallback[first : last + 1]
        else:
            member[keep[position:end]] = True
        position = end
    return member


def _flush_rows(resid_mm: np.ndarray) -> np.ndarray:
    """Строки внешнего сгустка отклонений от устойчивой кривой (:attr:`TrendMode.FLUSH_GAP`).

    Берутся строки не дальше ``ALIGN_TOL_MM`` наружу; по возрастанию отклонения «внутрь — плюс» сгусток идёт до первого
    разрыва больше ``FLUSH_GAP_MM`` — дальше отступы и концовки.

    Args:
        resid_mm: Отклонения краёв от кривой, мм, «внутрь — плюс».

    Returns:
        Флаги ``(n,)``.
    """
    candidates = np.nonzero(resid_mm >= -ALIGN_TOL_MM)[0]
    out = np.zeros(resid_mm.size, dtype=bool)
    if candidates.size == 0:
        return out
    order = candidates[np.argsort(resid_mm[candidates])]
    values = resid_mm[order]
    gaps = np.nonzero(np.diff(values) > FLUSH_GAP_MM)[0]
    stop = int(gaps[0]) + 1 if gaps.size else values.size
    out[order[:stop]] = True
    return out


def _fit_side(
    ys: np.ndarray, us: np.ndarray, dpi: float, mode: TrendMode = TrendMode.RUNS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Устойчивая кривая стороны по краям строк, строки, лежащие на ней сериями, и строки для тренда.

    Шаг 1 :func:`side_curve`: кривая по наружным краям (:func:`_outer_rows`), по всем краям — только если на неё
    ложится заметно больше строк; статусы «на кривой / отступ / мимо», выровненные серии (по способу ``mode``,
    :class:`TrendMode`), а без серий — висячий отступ и набор по формату с короткими абзацами.

    Args:
        ys: Середины строк сверху вниз (не меньше трёх).
        us: Края строк в «наружной» координате.
        dpi: Разрешение рабочей копии.
        mode: Способ серий и строк тренда.

    Returns:
        ``(coef, member, on_all, trend)``: коэффициенты кривой (``np.polyval``), флаги строк в выровненных сериях,
        флаги строк «на кривой» и флаги строк, по которым строится сплайн тренда (все ``(n,)``).
    """
    # Кривая по наружным краям (:func:`_outer_rows`); по всем краям — только если на неё ложится заметно больше строк
    # (на ``PARABOLA_GAIN_ROWS``): у библиографии кривая по всем краям шла косо через концы записей и вписывала на
    # одну строку больше (01/IMG_1030, итог: 109.9 → 93.6 мм, 4 строки против 3).
    tol_px = mm_to_px(ALIGN_TOL_MM, dpi)
    punct = mm_to_px(PUNCT_MM, dpi)
    outer_coef, all_coef = _fit_robust(*_outer_rows(ys, us, dpi), tol_px)[0], _fit_robust(ys, us, tol_px)[0]
    gain = _on_count(all_coef, ys, us, tol_px) - _on_count(outer_coef, ys, us, tol_px)
    coef = all_coef if gain >= PARABOLA_GAIN_ROWS else outer_coef
    fitted = np.polyval(coef, ys)
    # Остаток «внутрь — плюс» в мм: у наружной координаты внутрь — меньше.
    resid_mm = (fitted - us) * 25.4 / dpi
    statuses = _statuses(resid_mm)
    # Строки, торчащие наружу дальше пунктуации.
    outward = (us - fitted) > punct
    # Серии: подряд идущие строки «на кривой» или — у ``SKIP_INDENT`` — то же по строкам без отступов.
    member = _runs_skip_indent(statuses, outward) if mode is TrendMode.SKIP_INDENT else aligned_runs(statuses)
    on_all = np.array([status is RowStatus.ON for status in statuses])
    # Висячий отступ (оглавление, список: вторая строка записи отодвинута внутрь через одну) даёт
    # долю строк «на кривой» ниже порога серий (0.75), и сторона шла лесенкой по каждой строке
    # (1968/12 IMG_0149_1L). Сторона выровнена и тогда, когда на кривой не меньше половины строк
    # и никто не торчит наружу дальше пунктуации: строки внутрь между ними добиваются ниже.
    # Набор по формату с короткими абзацами (библиография: запись в 2–3 строки, ХиП 1991/01 IMG_1030) — строки
    # на кривой разделены концами записей, серий нет, и строк на кривой меньше половины блока; но среди строк,
    # не ушедших внутрь (не «отступ»), на кривой почти все. Тогда сторона тоже выровнена.
    inward = np.array([status is RowStatus.INDENT for status in statuses])
    flush = on_all.sum() >= 3 and on_all.sum() >= FLUSH_ON_SHARE * max(int((~inward).sum()), 1)
    if not member.any() and (on_all.mean() >= HALF_ON_SHARE or flush) and on_all.sum() >= 3 and not outward.any():
        member = on_all.copy()
    # Строки тренда: «на кривой» в сериях; у ``FLUSH_GAP`` при найденной серии — весь внешний сгусток краёв.
    trend = on_all & member
    if mode is TrendMode.FLUSH_GAP and member.any():
        trend = trend | _flush_rows(resid_mm)
    return coef, member, on_all, trend


def side_curve(
    ys: np.ndarray,
    us: np.ndarray,
    tops: np.ndarray,
    bottoms: np.ndarray,
    pitch: float,
    dpi: float,
    known: np.ndarray | None = None,
    mode: TrendMode | None = None,
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

    Строки с неизвестным краем (``known`` — False: строка обрезана краем разбираемой картинки или к ней прилипла
    краска у края — тень сгиба, кромка соседнего листа) в подгонку кривой, серии и сплайн не идут: их край —
    край картинки, а не конец строки. Иначе такая строка, торчащая наружу на 4–5 мм, вытесняла соседей из
    наружных краёв (:func:`_outer_rows`), кривая рассыпалась, и выровненная сторона блока пропадала целиком
    (ХиП 1991/07 IMG_0099, правая страница после Кунса: две строки у корешка с краем на 0 мм против 4.4 мм у
    остальных — левая сторона блока из 24 строк «рваная», доля рядов на тренде 0.53; без них — 1.00). При
    найденной стороне такие строки — её часть: внутри неё — по сплайну, снаружи — ступенькой по своему краю
    (участок недостоверный).

    Args:
        ys: Середины строк сверху вниз.
        us: Края строк в «наружной» координате.
        tops, bottoms: Верх и низ полосы каждой строки (пиксели).
        pitch: Шаг строк.
        dpi: Разрешение рабочей копии.
        known: Флаги строк с известным краем ``(n,)``; ``None`` — у всех известен.
        mode: Способ серий и строк тренда (:class:`TrendMode`); ``None`` — :data:`TREND_MODE`.

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
    on_all = np.zeros(n, dtype=bool)
    known = np.ones(n, dtype=bool) if known is None else np.asarray(known, dtype=bool)
    mode = TREND_MODE if mode is None else mode
    # Длина сглаживания тренда: у ``FLEX`` — в мм высоты, иначе ``SMOOTH_PITCHES`` шагов строк (``None``).
    length = mm_to_px(FLEX_SMOOTH_MM, dpi) if mode is TrendMode.FLEX else None
    trend = np.zeros(n, dtype=bool)
    if known.sum() >= 3:
        # Кривая и серии — только по строкам с известным краем; строки с неизвестным краем стоят между ними
        # и в сериях не участвуют (их флаги — False до решения ниже).
        coef, member_known, on_known, trend_known = _fit_side(ys[known], us[known], dpi, mode)
        member[known], on_all[known], trend[known] = member_known, on_known, trend_known
        on = trend
        if on.sum() >= 3:
            anchor = np.polyval(coef, grid)
            spline = _spline(ys, us, on.astype(np.float64), grid, pitch, anchor, length)
            at_rows = np.interp(ys, grid, spline)
            outward = np.maximum(us[on] - at_rows[on], 0.0)
            core = spline.copy()
            spline = spline + min(float(np.quantile(outward, JITTER_QUANTILE)) if outward.size else 0.0, jitter)
    at_rows = np.interp(ys, grid, spline) if spline is not None else None
    # Сторона держится не только на самих сериях: строка, ушедшая ВНУТРЬ, между выровненными строками
    # (или сразу за крайней из них) стороны не рвёт, сколько бы таких строк ни шло подряд — конец
    # абзаца и абзацный отступ следующего стоят двумя строками внутрь (1966/01 IMG_0041_2R), и сторона
    # проваливалась к ним ступенькой на 6 мм.
    if spline is not None and member.any():
        # Короткие строки у верха и низа блока (конец последнего абзаца, подпись в подбор справа —
        # 1970/03 IMG_0148_2R, «Э. САВИНА») тоже добиваются: иначе у конца блока оставалась ступенька.
        member = member | (us <= at_rows)
        # Строка с неизвестным краем при найденной стороне — её часть: внутри стороны — по сплайну, снаружи —
        # ступенька по своему краю (полоса строки в контуре остаётся, участок недостоверный).
        member = member | ~known
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


__all__ = ["EdgeKind", "SideCurve", "TREND_MODE", "TrendMode", "side_curve"]
