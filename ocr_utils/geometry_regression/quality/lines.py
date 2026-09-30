"""Качество строк: одна мера на строку — размах оси по вертикали на чистых участках (без перескоков и знаков препинания), порча и выигрыш FineReader по парам строк B ↔ A.

МЕРА. Ось строки ``text_blocks`` — сглаженная ломаная через центр краски (или базовую линию глифов).
Отклонение прямой строки длиной L под наклоном θ даёт размах L·tg θ; дуга с прогибом s — размах s; волна —
её двойную амплитуду. Поэтому размах ``Q = max y − min y`` (мм) на общем для B и A участке меряет наклон и
кривизну сразу, в тех же миллиметрах ухода, что и прежние метрики наклона линеек и строк.

ЧИСТЫЕ УЧАСТКИ. Точки оси в участках перескока на соседнюю строку (``jump_spans``) и над точками и
запятыми (``mark_spans``) выбрасываются: там ось уходит не за строкой.

ПАРЫ. Чистые точки оси B переносятся полем смещений в кадр A; парой считается ось A, которая перекрывает
перенесённую ось по x (от короткой) и лежит на той же высоте. Q обеих сравнивается только на общем
отрезке по x — разная нарезка строк в B и A не выдаёт себя за наклон.

ВЕС. Разность ``Q_A − Q_B`` умножается на вес кегля ``√(h / h_корпуса)`` в пределах [1, 2]: заголовок,
наклонённый на тот же миллиметр, заметнее строки корпуса (решение пользователя 2026-09-29: вклад
строки растёт с длиной и кеглем; длина уже сидит в мере в мм).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.geometry_regression.field import Field
from ocr_utils.page_layout import px_to_mm
from ocr_utils.page_layout.text_blocks.store import AxisRecord, PageGeometry

# Строка короче (по общему отрезку) — не меряется: на коротком отрезке размах — шум оси.
LINE_MIN_MM = 25.0
# Выигрыш по одной строке — только от этой длины: выпрямленный заголовок.
LINE_GAIN_MIN_MM = 25.0
# Точек оси меньше — участок не берётся.
MIN_POINTS = 6
# Пара B ↔ A: перекрытие по x от короткой из двух и допуск по высоте в долях высоты строки.
MATCH_OVERLAP = 0.8
MATCH_DY_FRAC = 0.5
# Вес кегля: √(h / h_корпуса) в этих пределах.
WEIGHT_MIN, WEIGHT_MAX = 1.0, 2.0
# Выигрыш колонки (блока) — лучшее окно из GAIN_WINDOW соседних строк блока (по высоте), в окне — медиана
# выигрыша строк; блок с меньшим числом пар не меряется. Окно, а не медиана всего блока: FineReader выпрямляет
# прежде всего гнутый низ колонки у корешка, и на 1967/09 с.81 медиана 60 строк давала 0.1–0.16 мм при
# выпрямлении нижних двенадцати с 1–1.6 мм до 0.1–0.2 (эталон good).
BLOCK_MIN_PAIRS = 5
GAIN_WINDOW = 8
# Строка без пары в A (``transfer_lines``) переносится полем, только если она похожа на строку текста: в B уже ровная
# (размах не больше TRANSFER_MAX_QB_MM — у подписи 1972/10 с.79 0.98 мм, у «строк» в шуме фото обложки 1973/10
# с.1 и в снимке 1968/11 с.49 — 2.1–5 мм), полоса B ↔ A' коррелирует не ниже TRANSFER_MIN_NCC (строго, чем
# рамки рисунков: полоса узкая) и строка лежит вне растра.
TRANSFER_MAX_QB_MM = 1.2
TRANSFER_MIN_NCC = 0.6
# Строка текста, а не полоса фото, которое разбор не отметил растром: у строки текста хотя бы с одной стороны —
# белый просвет. Доля краски в полосе TRANSFER_GAP высоты строки над ней и под ней; меньшая из двух — не больше
# TRANSFER_MAX_GAP_INK (на отсмотре смен v17 → v18: строки текста 0.00–0.04, «строки» в полутоне фото 0.10–0.67).
TRANSFER_GAP = 0.3
TRANSFER_MAX_GAP_INK = 0.06
# Уверенность точки перенесённой оси (``lineart_flow.transfer_axis``) в долях медианы по оси: ниже — точка
# выбрасывается (конец оси у вертикальной линейки, 1971/06 с.3 и 1974/08 с.3: сдвиг конца на 10–20 px на неизменной
# строке); строка берётся, если уверенных точек не меньше TRANSFER_MIN_SURE.
TRANSFER_MIN_CONFIDENCE = 0.3
TRANSFER_MIN_SURE = 0.7


@dataclass(frozen=True)
class LinePair:
    """Пара строк B ↔ A с мерами на общем отрезке.

    Attributes:
        b: Номер оси в разборе B.
        a: Номер оси в разборе A.
        block: Блок оси B (−1 — вне блоков).
        q_b: Размах оси B, мм.
        q_a: Размах оси A, мм.
        weight: Вес кегля.
        length_mm: Длина общего отрезка, мм.
        box_b: Рамка общего участка в B ``(x0, y0, x1, y1)``, пиксели.
        box_a: Рамка общего участка в A.
        inside_objects: Строка B лежит в таблице, рисунке или формуле — только выигрыш, не порча.
        transferred: Оси A нет в разборе A, она построена переносом оси B полем (``a`` = −1).
    """

    b: int
    a: int
    block: int
    q_b: float
    q_a: float
    weight: float
    length_mm: float
    box_b: tuple[float, float, float, float]
    box_a: tuple[float, float, float, float]
    inside_objects: bool
    transferred: bool = False  # оси A в разборе нет: она перенесена из B плотным полем (:func:`transfer_lines`)

    @property
    def delta(self) -> float:
        """Порча строки со знаком, мм с весом: плюс — в A строка стала кривее или косее."""
        return self.weight * (self.q_a - self.q_b)


def clean_points(axis: AxisRecord) -> np.ndarray:
    """Точки оси вне участков перескока и знаков препинания, по возрастанию x.

    Args:
        axis: Ось из сайдкара.

    Returns:
        ``(N, 2)``; пусто, если чистых точек меньше ``MIN_POINTS``.
    """
    points = np.asarray(axis.points, dtype=np.float64)
    keep = np.ones(len(points), dtype=bool)
    for start, stop in (*axis.jump_spans, *axis.mark_spans):
        keep &= ~((points[:, 0] >= min(start, stop)) & (points[:, 0] <= max(start, stop)))
    points = points[keep]
    if len(points) < MIN_POINTS:
        return np.zeros((0, 2))
    return points[np.argsort(points[:, 0], kind="stable")]


def spread_mm(points: np.ndarray, dpi: float, column: int = 1) -> float:
    """Размах координаты точек (мм): ``column=1`` — по вертикали (строки), ``0`` — по горизонтали (стороны)."""
    if len(points) < 2:
        return 0.0
    values = points[:, column]
    return px_to_mm(float(values.max() - values.min()), dpi)


def body_height(geometry: PageGeometry) -> float:
    """Высота строки корпуса: медиана высот осей в блоках от трёх рядов (иначе — всех осей), пиксели."""
    rows = {block.index: block.rows for block in geometry.blocks}
    heights = [axis.height for axis in geometry.axes if rows.get(axis.block, 0) >= 3]
    if not heights:
        heights = [axis.height for axis in geometry.axes]
    return float(np.median(heights)) if heights else 1.0


def _inside(point: np.ndarray, boxes: list) -> bool:
    """Точка внутри одной из рамок."""
    return any(x0 <= point[0] <= x1 and y0 <= point[1] <= y1 for x0, y0, x1, y1 in boxes)


def _box(points: np.ndarray) -> tuple[float, float, float, float]:
    return (float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max()))


def match_lines(b: PageGeometry, a: PageGeometry, field: Field | None, objects_b: list) -> list[LinePair]:
    """Пары строк B ↔ A с мерами качества на общем отрезке.

    Args:
        b: Разбор без коррекции.
        a: Разбор с коррекцией.
        field: Поле смещений B → A; ``None`` — переноса нет (точки B берутся как есть).
        objects_b: Рамки таблиц, рисунков и формул на B (150 dpi): строки в них — только выигрыш.

    Returns:
        Пары; каждая ось A — не больше чем в одной паре (берётся ближайшая по высоте).
    """
    dpi = b.dpi
    h_body = body_height(b)
    clean_a = [clean_points(axis) for axis in a.axes]
    used: set[int] = set()
    candidates = []
    for index_b, axis_b in enumerate(b.axes):
        points_b = clean_points(axis_b)
        if not len(points_b):
            continue
        mapped = field.transform(points_b) if field is not None else points_b
        lo_b, hi_b = float(mapped[:, 0].min()), float(mapped[:, 0].max())
        best = None
        for index_a, points_a in enumerate(clean_a):
            if not len(points_a):
                continue
            lo, hi = max(lo_b, points_a[0, 0]), min(hi_b, points_a[-1, 0])
            shorter = min(hi_b - lo_b, points_a[-1, 0] - points_a[0, 0])
            if shorter <= 0 or hi - lo < MATCH_OVERLAP * shorter:
                continue
            # Высота: перенесённая ось B против оси A на общем отрезке.
            xs = np.linspace(lo, hi, 16)
            dy = np.median(
                np.abs(np.interp(xs, mapped[:, 0], mapped[:, 1]) - np.interp(xs, points_a[:, 0], points_a[:, 1]))
            )
            limit = MATCH_DY_FRAC * max(axis_b.height, a.axes[index_a].height)
            if dy > limit:
                continue
            if best is None or dy < best[0]:
                best = (float(dy), index_a, lo, hi)
        if best is None:
            continue
        candidates.append((best[0], index_b, best[1], best[2], best[3], points_b, mapped))
    out = []
    # Ближайшие по высоте пары — первыми: спорную ось A забирает лучшая пара.
    for dy, index_b, index_a, lo, hi, points_b, mapped in sorted(candidates, key=lambda item: item[0]):
        if index_a in used:
            continue
        common_b = points_b[(mapped[:, 0] >= lo) & (mapped[:, 0] <= hi)]
        points_a = clean_a[index_a]
        common_a = points_a[(points_a[:, 0] >= lo) & (points_a[:, 0] <= hi)]
        if len(common_b) < MIN_POINTS or len(common_a) < MIN_POINTS:
            continue
        length = px_to_mm(hi - lo, dpi)
        if length < min(LINE_MIN_MM, LINE_GAIN_MIN_MM):
            continue
        used.add(index_a)
        axis_b = b.axes[index_b]
        weight = float(np.clip(np.sqrt(max(axis_b.height, 1e-6) / max(h_body, 1e-6)), WEIGHT_MIN, WEIGHT_MAX))
        middle = common_b[len(common_b) // 2]
        out.append(
            LinePair(
                b=index_b,
                a=index_a,
                block=axis_b.block,
                q_b=spread_mm(common_b, dpi),
                q_a=spread_mm(common_a, dpi),
                weight=weight,
                length_mm=length,
                box_b=_box(common_b),
                box_a=_box(common_a),
                inside_objects=_inside(middle, objects_b),
            )
        )
    return out


def _gap_ink(gray: np.ndarray, points: np.ndarray, height: float) -> float:
    """Доля краски в просвете над строкой и под ней (меньшая из двух): у строки текста просвет белый хотя бы с одной стороны.

    Args:
        gray: Рендер B (серый, пиксели разбора).
        points: Точки оси строки ``(n, 2)``.
        height: Высота строки, пиксели.

    Returns:
        Меньшая из долей краски (яркость ниже ``lineart_flow.INK_LEVEL``) в полосах высотой ``TRANSFER_GAP``·h над
        корпусом строки и под ним, по ширине оси; 1.0, если полоса вне страницы.
    """
    from ocr_utils.geometry_regression.quality.lineart_flow import INK_LEVEL

    centre = float(np.median(points[:, 1]))
    x0, x1 = int(max(0, points[:, 0].min())), int(min(gray.shape[1], points[:, 0].max() + 1))
    gap = TRANSFER_GAP * height
    shares = []
    for top, bottom in (
        (centre - height / 2 - gap, centre - height / 2),
        (centre + height / 2, centre + height / 2 + gap),
    ):
        y0, y1 = int(max(0, top)), int(min(gray.shape[0], bottom))
        shares.append(float((gray[y0:y1, x0:x1] < INK_LEVEL).mean()) if y1 > y0 and x1 > x0 else 1.0)
    return min(shares)


def transfer_lines(
    b: PageGeometry,
    pairs: list[LinePair],
    gray_b: np.ndarray,
    aligned_a: np.ndarray,
    page_map: np.ndarray,
    objects_b: list,
    raster_b: list,
) -> list[LinePair]:
    """Пары для строк B, оставшихся без пары: ось A — перенос оси B плотным полем (``lineart_flow.transfer_axis``).

    Разбор A теряет строки, и такая строка раньше не мерилась вовсе (1972/10 с.79: подпись под рисунком наклонена
    FineReader на 2.5°, а оси в разборе A у неё нет). Решение пользователя 2026-09-30: такая строка — обычная
    строка, её порча идёт в ``line_quality_mm``.

    Args:
        b: Разбор без коррекции.
        pairs: Уже найденные пары (:func:`match_lines`) — их оси B не переносятся.
        gray_b: Рендер B, 150 dpi (``b.dpi`` должно быть тем же).
        aligned_a: Рендер A в кадре B (``lineart_flow.page_alignment``).
        page_map: Матрица страницы B → A.
        objects_b: Рамки таблиц, рисунков и формул на B: строки в них — только выигрыш.
        raster_b: Рамки растра на B (фото): строки в них не переносятся.

    Returns:
        Пары с ``a = −1`` и ``transferred=True``; строка берётся, если чистый отрезок не короче ``LINE_MIN_MM``,
        размах в B не больше ``TRANSFER_MAX_QB_MM``, середина вне растра, просвет над или под строкой белый
        (:func:`_gap_ink`) и корреляция полосы B ↔ A' не ниже ``TRANSFER_MIN_NCC``.
    """
    from ocr_utils.geometry_regression.quality.lineart_flow import transfer_axis

    dpi = b.dpi
    h_body = body_height(b)
    taken = {pair.b for pair in pairs}
    out = []
    for index_b, axis_b in enumerate(b.axes):
        if index_b in taken:
            continue
        points_b = clean_points(axis_b)
        if len(points_b) < MIN_POINTS:
            continue
        length = px_to_mm(float(points_b[-1, 0] - points_b[0, 0]), dpi)
        if length < LINE_MIN_MM or spread_mm(points_b, dpi) > TRANSFER_MAX_QB_MM:
            continue
        if _inside(points_b[len(points_b) // 2], raster_b):
            continue
        if _gap_ink(gray_b, points_b, float(axis_b.height)) > TRANSFER_MAX_GAP_INK:
            continue
        moved = transfer_axis(gray_b, aligned_a, page_map, points_b, float(axis_b.height))
        if moved is None or moved[1] < TRANSFER_MIN_NCC:
            continue
        # Точки с малой уверенностью переноса выбрасываются из обеих осей: размах B и A — по одним и тем же точкам.
        sure = moved[2] >= TRANSFER_MIN_CONFIDENCE
        if sure.mean() < TRANSFER_MIN_SURE or sure.sum() < MIN_POINTS:
            continue
        points_b, points_a = points_b[sure], moved[0][sure]
        length = px_to_mm(float(points_b[-1, 0] - points_b[0, 0]), dpi)
        if length < LINE_MIN_MM:
            continue
        weight = float(np.clip(np.sqrt(max(axis_b.height, 1e-6) / max(h_body, 1e-6)), WEIGHT_MIN, WEIGHT_MAX))
        out.append(
            LinePair(
                b=index_b,
                a=-1,
                block=axis_b.block,
                q_b=spread_mm(points_b, dpi),
                q_a=spread_mm(points_a, dpi),
                weight=weight,
                length_mm=length,
                box_b=_box(points_b),
                box_a=_box(points_a),
                inside_objects=_inside(points_b[len(points_b) // 2], objects_b),
                transferred=True,
            )
        )
    return out


def line_metrics(pairs: list[LinePair]) -> tuple[dict[str, float], dict[str, dict]]:
    """Метрики страницы по парам строк.

    * ``line_quality_mm`` — худшая порча строки вне таблиц и рисунков: ``max w·(Q_A − Q_B)``, строки от
      ``LINE_MIN_MM``;
    * ``line_quality_gain_mm`` — лучший выигрыш одной строки (выпрямленный заголовок), от ``LINE_GAIN_MIN_MM``,
      включая строки внутри таблиц и рисунков;
    * ``lines_quality_gain_mm`` — выигрыш колонки: лучшее окно из ``GAIN_WINDOW`` соседних строк блока, в окне —
      медиана ``Q_B − Q_A`` (блоки от ``BLOCK_MIN_PAIRS`` пар; меньше строк, чем окно, — весь блок);
    * ``line_quality_transferred_mm`` — то же только по строкам, перенесённым полем (:func:`transfer_lines`), для
      разбора; ``line_transferred`` — их число;
    * счётчики и сводки для отчёта.

    Args:
        pairs: Пары строк (:func:`match_lines`).

    Returns:
        ``(метрики, виновники)``: виновник — рамки худшей и лучшей строки в B и A.
    """
    metrics = {
        "line_pairs": float(len(pairs)),
        "line_quality_mm": 0.0,
        "line_quality_gain_mm": 0.0,
        "lines_quality_gain_mm": 0.0,
        "line_q_b_p90": 0.0,
        "line_q_a_p90": 0.0,
        "line_transferred": float(sum(p.transferred for p in pairs)),
        "line_quality_transferred_mm": 0.0,
    }
    culprits: dict[str, dict] = {}
    if not pairs:
        return metrics, culprits
    # Для разбора: худшая порча среди строк, перенесённых полем (она же входит в ``line_quality_mm``).
    moved = [p.delta for p in pairs if p.transferred and not p.inside_objects and p.length_mm >= LINE_MIN_MM]
    metrics["line_quality_transferred_mm"] = max(0.0, max(moved, default=0.0))
    body = [p for p in pairs if not p.inside_objects and p.length_mm >= LINE_MIN_MM]
    if body:
        worst = max(body, key=lambda p: p.delta)
        if worst.delta > 0:
            metrics["line_quality_mm"] = worst.delta
            culprits["line_quality_mm"] = {"b": list(worst.box_b), "a": list(worst.box_a)}
    long = [p for p in pairs if p.length_mm >= LINE_GAIN_MIN_MM]
    if long:
        best = min(long, key=lambda p: p.delta)
        if best.delta < 0:
            metrics["line_quality_gain_mm"] = -best.delta
            culprits["line_quality_gain_mm"] = {"b": list(best.box_b), "a": list(best.box_a)}
    metrics["lines_quality_gain_mm"] = window_gain(pairs)
    metrics["line_q_b_p90"] = float(np.percentile([p.q_b for p in pairs], 90))
    metrics["line_q_a_p90"] = float(np.percentile([p.q_a for p in pairs], 90))
    return metrics, culprits


def window_gain(pairs: list[LinePair]) -> float:
    """Выигрыш колонки: лучшая медиана ``Q_B − Q_A`` по окну из ``GAIN_WINDOW`` соседних строк одного блока.

    Args:
        pairs: Пары строк страницы.

    Returns:
        Выигрыш, мм (не меньше нуля).
    """
    blocks: dict[int, list[LinePair]] = {}
    for pair in pairs:
        if pair.block >= 0:
            blocks.setdefault(pair.block, []).append(pair)
    best = 0.0
    for members in blocks.values():
        if len(members) < BLOCK_MIN_PAIRS:
            continue
        # Строки блока сверху вниз: окно — соседние строки, а не случайные.
        gains = [p.q_b - p.q_a for p in sorted(members, key=lambda p: p.box_b[1])]
        width = min(GAIN_WINDOW, len(gains))
        for start in range(len(gains) - width + 1):
            best = max(best, float(np.median(gains[start : start + width])))
    return best


__all__ = [
    "LinePair",
    "body_height",
    "clean_points",
    "line_metrics",
    "match_lines",
    "spread_mm",
    "transfer_lines",
    "window_gain",
]
