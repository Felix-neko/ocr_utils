"""Мера «текст впритык к линии разреза у корешка» по холсту Кунса: выровненные стороны блоков у корешка, строки впритык, свободный зазор на другой странице и место для уточнённой линии разреза."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ocr_utils.page_layout.text_blocks.blocks import TextBlock
from ocr_utils.scan_cropping.v2.rectify.coons import Layout
from ocr_utils.scan_cropping.v2.rectify.text_field import INK_LEVEL, PageText
from ocr_utils.units import mm_to_px_float, px_to_mm

# Сторона блока «у корешка»: медиана её тренда по телу блока ближе этого к линии разреза, мм.
FLUSH_MM = 1.5
# Строка «впритык»: край краски ряда со стороны корешка ближе этого к линии разреза, мм (справочно: разбор отбрасывает
# обрубки букв у края вырезки, и край ряда срезанной строки отходит от разреза на 0.5–1.3 мм — пак-6 РИСК 1991/05 fix).
TIGHT_MM = 0.5
# Касание по краске: полоса шириной TOUCH_MM у разреза со стороны страницы; ряд касается, если доля краски в полосе
# его ядра (y ± TOUCH_CORE·высоты) выше фона полосы между рядами на TOUCH_CONTRAST (тень сгиба — и в рядах, и между).
TOUCH_MM = 1.0
# Пересечение: краска ряда в полосе CROSS_MM по обе стороны разреза — линия режет буквы (а не сгиб их съел).
CROSS_MM = 0.3
TOUCH_CORE = 0.3
TOUCH_CONTRAST = 0.1
# Выровненная сторона у корешка не короче этого, мм: короче — подпись, колонтитул или сор, а не граница блока.
MIN_FLUSH_MM = 25.0
# Рядов, касающихся разреза краской, у стороны не меньше, и их доля среди рядов блока — не меньше.
MIN_TIGHT_ROWS = 6
MIN_TIGHT_SHARE = 0.4
# Зазор свободной страницы считается по рядам её блоков на высоте стороны впритык (с запасом по высоте), мм.
GAP_Y_MARGIN_MM = 3.0
# Зазор, когда на той же высоте у другой страницы нет рядов, мм (вся полоса до края вырезки свободна).
NO_ROWS_GAP_MM = 99.0
# Сдвиг разреза: запас до настоящего текста свободной страницы и потолок сдвига, мм; меньше MIN_GAP_MM — не сдвигаем.
SAFETY_MM = 0.7
MAX_SHIFT_MM = 4.0
MIN_GAP_MM = 2.0
# Чистая колонка за корешком: в полосах рядов впритык краски не больше фона между рядами плюс эта доля, подряд не
# меньше CLEAN_RUN_MM.
CLEAN_SHARE = 0.03
CLEAN_RUN_MM = 0.4


class Verdict(StrEnum):
    """Предварительный вердикт правила по развороту."""

    SHIFT = "shift"  # впритык с одной стороны, у другой есть место — разрез сдвигается
    BOTH = "both"  # впритык с обеих сторон — разрез на месте
    NO_GAP = "no_gap"  # впритык с одной стороны, но места у другой нет
    IN_FOLD = "in_fold"  # впритык с одной стороны, но за разрезом сразу чисто: буквы ушли в сгиб, сдвигать нечего
    NONE = "none"  # ни одна сторона не впритык
    SINGLE = "single"  # одна страница или страница без разбора


@dataclass(frozen=True)
class SideMeasure:
    """Сторона страницы у корешка.

    Поля:
        flush_mm: Длина самой длинной выровненной стороны блока у корешка (0 — нет такой), мм.
        flush_dist_mm: Расстояние медианы этой стороны до линии разреза, мм (плюс — внутри страницы).
        rows: Рядов в блоке этой стороны.
        tight_rows: Рядов, касающихся разреза краской (:func:`touching_rows`).
        crossing_rows: Рядов, чья краска есть по обе стороны разреза (:data:`CROSS_MM`) — разрез режет буквы.
        y0: Верх стороны на холсте, px (NaN — нет стороны).
        y1: Низ стороны на холсте, px.
        tight: Сторона впритык (длина, расстояние, число и доля касающихся рядов — по порогам).
    """

    flush_mm: float = 0.0
    flush_dist_mm: float = float("nan")
    rows: int = 0
    tight_rows: int = 0
    crossing_rows: int = 0
    y0: float = float("nan")
    y1: float = float("nan")
    tight: bool = False


@dataclass(frozen=True)
class GutterMeasure:
    """Мера разворота у корешка.

    Поля:
        left: Сторона левой страницы (её правый край у корешка).
        right: Сторона правой страницы (левый край).
        gap_mm: Зазор свободной страницы на высоте стороны впритык, мм (NaN — сдвиг не рассматривался).
        overshoot_mm: На сколько краска рядов впритык уходит за линию разреза до чистой колонки, мм (0 — за разрезом
            сразу чисто: буквы ушли в сгиб, а не отрезаны линией; NaN — чистой колонки в пределах нет или сдвиг не
            рассматривался).
        shift_mm: Предлагаемый сдвиг разреза в сторону свободной страницы, мм (0 — на месте).
        cut_x: X уточнённой линии разреза на холсте Кунса, px.
        verdict: Вердикт правила.
    """

    left: SideMeasure
    right: SideMeasure
    gap_mm: float
    overshoot_mm: float
    shift_mm: float
    cut_x: float
    verdict: Verdict


def block_rows_canvas(page: PageText, block: TextBlock) -> np.ndarray:
    """Ряды блока на холсте: ``(N, 4)`` — ``y, height, x0, x1`` в px холста.

    Args:
        page: Разбор страницы (привязка рабочей копии к холсту).
        block: Блок.

    Returns:
        Массив рядов.
    """
    if not block.rows:
        return np.zeros((0, 4))
    rows = np.array([(r.y, r.height, r.x0, r.x1) for r in block.rows], np.float64)
    out = rows.copy()
    out[:, 0] = rows[:, 0] * page.scale + page.offset[1]
    out[:, 1] = rows[:, 1] * page.scale
    out[:, 2] = rows[:, 2] * page.scale + page.offset[0]
    out[:, 3] = rows[:, 3] * page.scale + page.offset[0]
    return out


def row_masks(rows: np.ndarray, height: int, core: float) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Полосы ядер рядов и маска «между рядами» по высоте холста.

    Args:
        rows: Ряды ``(N, 4)`` на холсте (:func:`block_rows_canvas`).
        height: Высота холста, px.
        core: Полувысота ядра ряда в долях высоты ряда.

    Returns:
        Полосы ``(y0, y1)`` ядер рядов и маска bool строк холста между рядами (в пределах блока, вне полных рядов).
    """
    bands = [(max(int(y - core * h), 0), min(int(np.ceil(y + core * h)), height)) for y, h, _, _ in rows]
    between = np.zeros(height, bool)
    if len(rows):
        top = max(int(rows[:, 0].min() - rows[:, 1].max()), 0)
        bottom = min(int(rows[:, 0].max() + rows[:, 1].max()), height)
        between[top:bottom] = True
        for y, h, _, _ in rows:
            between[max(int(y - 0.6 * h), 0) : min(int(np.ceil(y + 0.6 * h)), height)] = False
    return bands, between


def strip_excess(gray: np.ndarray, rows: np.ndarray, x0: int, x1: int) -> np.ndarray:
    """Превышение доли краски в ядре каждого ряда над фоном между рядами в колонках ``[x0, x1)`` холста, ``(N,)``."""
    ink = gray[:, max(x0, 0) : min(x1, gray.shape[1])] < INK_LEVEL
    bands, between = row_masks(rows, gray.shape[0], TOUCH_CORE)
    if ink.shape[1] == 0:
        return np.zeros(len(bands))
    background = float(ink[between].mean()) if between.any() else 0.0
    return np.array([ink[b0:b1].mean() - background if b1 > b0 else 0.0 for b0, b1 in bands])


def touching_rows(gray: np.ndarray, rows: np.ndarray, cut_x: float, left_page: bool, dpi: float) -> tuple[int, int]:
    """Сколько рядов касаются линии разреза краской и сколько её пересекают.

    Касание — в полосе :data:`TOUCH_MM` у разреза со стороны страницы краски в ядре ряда больше, чем между рядами
    (тень сгиба тёмная и там, и там), на :data:`TOUCH_CONTRAST`. Пересечение — то же в полосах :data:`CROSS_MM` по обе
    стороны разреза: буквы режет линия, а не сгиб.

    Args:
        gray: Холст Кунса, серый.
        rows: Ряды блока на холсте.
        cut_x: X линии разреза.
        left_page: Страница левая (полоса касания — левее разреза).
        dpi: Разрешение холста.

    Returns:
        ``(касающихся, пересекающих)``.
    """
    c = int(round(cut_x))
    touch = max(1, int(round(mm_to_px_float(TOUCH_MM, dpi))))
    cross = max(1, int(round(mm_to_px_float(CROSS_MM, dpi))))
    own = strip_excess(gray, rows, c - touch, c) if left_page else strip_excess(gray, rows, c, c + touch)
    inner = strip_excess(gray, rows, c - cross, c)
    outer = strip_excess(gray, rows, c, c + cross)
    crossing = (inner >= TOUCH_CONTRAST) & (outer >= TOUCH_CONTRAST)
    return int((own >= TOUCH_CONTRAST).sum()), int(crossing.sum())


def side_measure(page: PageText | None, gray: np.ndarray, cut_x: float, left_page: bool, dpi: float) -> SideMeasure:
    """Самая длинная выровненная сторона блока у линии разреза и её ряды, касающиеся разреза.

    Args:
        page: Разбор страницы (нулевой проход, холст Кунса) или None.
        gray: Холст Кунса, серый.
        cut_x: X линии разреза на холсте.
        left_page: Левая страница (у корешка — правая сторона блоков).
        dpi: Разрешение холста.

    Returns:
        Мера стороны (пустая — нет выровненной стороны у корешка).
    """
    if page is None:
        return SideMeasure()
    best = SideMeasure()
    for block, alignment in zip(page.analysis.blocks, page.analysis.alignments):
        stats = alignment.right if left_page else alignment.left
        curve = block.envelope.core_right if left_page else block.envelope.core_left
        if curve is None:
            curve = block.envelope.right if left_page else block.envelope.left
        if not stats.aligned or curve is None or len(curve) < 2:
            continue
        pts = page.to_canvas(np.asarray(curve, np.float64))
        # Расстояние стороны до разреза внутрь страницы: у левой — разрез правее стороны.
        dist = (cut_x - np.median(pts[:, 0])) if left_page else (np.median(pts[:, 0]) - cut_x)
        dist_mm = px_to_mm(float(dist), dpi)
        length_mm = px_to_mm(float(np.ptp(pts[:, 1])), dpi)
        if dist_mm > FLUSH_MM or length_mm <= best.flush_mm:
            continue
        rows = block_rows_canvas(page, block)
        tight_rows, crossing_rows = touching_rows(gray, rows, cut_x, left_page, dpi)
        tight = (length_mm >= MIN_FLUSH_MM and tight_rows >= MIN_TIGHT_ROWS
                 and tight_rows >= MIN_TIGHT_SHARE * max(len(rows), 1))  # fmt: skip
        best = SideMeasure(length_mm, dist_mm, len(rows), tight_rows, crossing_rows, float(pts[:, 1].min()),
                           float(pts[:, 1].max()), tight)  # fmt: skip
    return best


def free_gap(page: PageText | None, cut_x: float, left_page: bool, y0: float, y1: float, dpi: float) -> float:
    """Зазор от линии разреза до краски рядов свободной страницы на высоте ``[y0, y1]``, мм.

    Берутся все ряды всех блоков (не только выровненные стороны): настоящий текст свободной страницы задевать нельзя.
    Псевдоглифы, отсеянные фильтром разбора, в ряды не входят и зазор не съедают. Зазор — 10-й процентиль по рядам
    (одиночный выброс ряда не решает).

    Args:
        page: Разбор свободной страницы или None.
        cut_x: X линии разреза.
        left_page: Свободная страница — левая.
        y0: Верх полосы, px холста.
        y1: Низ полосы.
        dpi: Разрешение холста.

    Returns:
        Зазор, мм (:data:`NO_ROWS_GAP_MM` — рядов на этой высоте нет).
    """
    if page is None:
        return NO_ROWS_GAP_MM
    margin = mm_to_px_float(GAP_Y_MARGIN_MM, dpi)
    gaps = []
    for block in page.analysis.blocks:
        rows = block_rows_canvas(page, block)
        if not len(rows):
            continue
        inside = (rows[:, 0] >= y0 - margin) & (rows[:, 0] <= y1 + margin)
        edge = rows[inside, 3] if left_page else rows[inside, 2]
        gaps.extend((cut_x - edge) if left_page else (edge - cut_x))
    if not gaps:
        return NO_ROWS_GAP_MM
    return px_to_mm(float(np.percentile(gaps, 10)), dpi)


def tight_rows_of(page: PageText, side: SideMeasure) -> np.ndarray:
    """Ряды всех блоков страницы на высоте стороны впритык (с запасом в высоту ряда), ``(N, 4)`` на холсте."""
    parts = [block_rows_canvas(page, block) for block in page.analysis.blocks]
    rows = np.vstack([p for p in parts if len(p)]) if any(len(p) for p in parts) else np.zeros((0, 4))
    keep = (rows[:, 0] >= side.y0 - rows[:, 1]) & (rows[:, 0] <= side.y1 + rows[:, 1])
    return rows[keep]


def overshoot(gray: np.ndarray, page: PageText, cut_x: float, left_page: bool, side: SideMeasure, limit_mm: float,
              dpi: float) -> float:  # fmt: skip
    """Насколько краска рядов впритык уходит за линию разреза: первая чистая колонка за ней, мм.

    По ядрам рядов страницы впритык на высоте её стороны считается доля краски в каждой колонке за разрезом в сторону
    свободной страницы против той же колонки между рядами (тень сгиба — в обоих); чистая — превышение не больше
    :data:`CLEAN_SHARE` на :data:`CLEAN_RUN_MM` подряд.

    Args:
        gray: Холст Кунса, серый (после выравнивания освещения).
        page: Разбор страницы впритык.
        cut_x: X линии разреза.
        left_page: Страница впритык — левая (краска уходит вправо).
        side: Мера её стороны (высота стороны).
        limit_mm: Сколько смотреть за разрез, мм.
        dpi: Разрешение холста.

    Returns:
        Расстояние от разреза до начала чистой колонки, мм (0 — сразу чисто), или NaN — чистой колонки в пределах нет.
    """
    rows = tight_rows_of(page, side)
    if not len(rows):
        return float("nan")
    bands, between = row_masks(rows, gray.shape[0], TOUCH_CORE)
    in_rows = np.zeros(gray.shape[0], bool)
    for b0, b1 in bands:
        in_rows[b0:b1] = True
    limit = int(np.ceil(mm_to_px_float(limit_mm, dpi)))
    c = int(round(cut_x))
    x0, x1 = (c, min(c + limit, gray.shape[1])) if left_page else (max(c - limit, 0), c)
    ink = gray[:, x0:x1] < INK_LEVEL
    excess = ink[in_rows].mean(axis=0) - (ink[between].mean(axis=0) if between.any() else 0.0)
    if not left_page:
        excess = excess[::-1]  # от разреза наружу
    run = max(1, int(round(mm_to_px_float(CLEAN_RUN_MM, dpi))))
    count = 0
    for i, value in enumerate(excess):
        count = count + 1 if value <= CLEAN_SHARE else 0
        if count >= run:
            return px_to_mm(float(i - run + 1), dpi)
    return float("nan")


def gutter_measure(gray: np.ndarray, layout: Layout, pages: list[PageText | None], dpi: float) -> GutterMeasure:
    """Мера разворота у корешка и предлагаемая уточнённая линия разреза.

    Правило (этап 0 плана, пороги — константы модуля): сторона впритык — выровненная сторона блока у разреза длиной от
    :data:`MIN_FLUSH_MM`, чьи ряды касаются разреза краской. Обе впритык — разрез на месте. Одна — смотрим зазор
    свободной страницы на той же высоте; от :data:`MIN_GAP_MM` разрез сдвигается к первой чистой колонке за краской
    рядов впритык (не дальше зазора без :data:`SAFETY_MM` и :data:`MAX_SHIFT_MM`), а без чистой колонки — на весь
    допустимый сдвиг. Чисто сразу за разрезом — буквы ушли в сгиб, сдвигать нечего (вердикт ``in_fold``).

    Args:
        gray: Холст Кунса, серый.
        layout: Расклад холста (линия корешка — ``spine_x``).
        pages: Разборы левой и правой страницы (нулевой проход) или None.
        dpi: Разрешение холста.

    Returns:
        Мера.
    """
    cut_x = layout.spine_x
    nan = float("nan")
    if layout.single or len(pages) != 2 or any(p is None for p in pages):
        empty = SideMeasure()
        return GutterMeasure(empty, empty, nan, nan, 0.0, cut_x, Verdict.SINGLE)
    left = side_measure(pages[0], gray, cut_x, True, dpi)
    right = side_measure(pages[1], gray, cut_x, False, dpi)
    if left.tight and right.tight:
        return GutterMeasure(left, right, nan, nan, 0.0, cut_x, Verdict.BOTH)
    if not left.tight and not right.tight:
        return GutterMeasure(left, right, nan, nan, 0.0, cut_x, Verdict.NONE)
    tight_left = left.tight
    side = left if tight_left else right
    gap = free_gap(pages[1] if tight_left else pages[0], cut_x, not tight_left, side.y0, side.y1, dpi)
    allowed = min(gap - SAFETY_MM, MAX_SHIFT_MM)
    if gap < MIN_GAP_MM or allowed <= 0:
        return GutterMeasure(left, right, gap, nan, 0.0, cut_x, Verdict.NO_GAP)
    over = overshoot(gray, pages[0] if tight_left else pages[1], cut_x, tight_left, side, allowed, dpi)
    if over == 0.0:
        return GutterMeasure(left, right, gap, over, 0.0, cut_x, Verdict.IN_FOLD)
    shift = allowed if np.isnan(over) else min(over + SAFETY_MM / 2, allowed)
    direction = 1.0 if tight_left else -1.0
    return GutterMeasure(left, right, gap, over, shift, cut_x + direction * mm_to_px_float(shift, dpi), Verdict.SHIFT)
