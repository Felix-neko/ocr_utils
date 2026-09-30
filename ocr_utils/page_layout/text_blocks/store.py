"""Геометрия разбора полосы на диске: оси строк, стороны блоков и трассы линеек — в сайдкар ``.npz`` рядом с JSON, и чтение обратно.

ЗАЧЕМ. Итоговый JSON разбора пака (``pack_analysis.final``) несёт объекты полосы, границы блоков и сводки,
но не осевые кривые строк (их тысячи точек на полосу) и не трассы линеек таблиц. Детектору порчи геометрии
FineReader (``research/geometry_quality``) они нужны: качество строк мерится по осям без участков перескока,
качество краёв — по выровненным сторонам блоков с заплатками поверх аномалий. Объёмное — в сжатый
``.npz`` (плоские массивы со смещениями), мелкое — в JSON; :func:`load_page` собирает всё обратно в
облегчённые dataclass без пересчёта разбора.

КООРДИНАТЫ. Всё в пикселях рабочей копии текстовых блоков (``WORK_DPI`` = 150), как ``text_blocks`` в JSON.
Трассы линеек строятся в 300 dpi по вырезке таблицы и переносятся в те же 150 dpi.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

# Версия раскладки массивов сайдкара: меняется при смене имён или смысла массивов.
STORE_FORMAT_VERSION = 1


class PointKind(IntEnum):
    """Точка дополнительной линии стороны блока: своя или заплатка, и над чем заплатка."""

    ALIGNED = 0  # точка самой стороны на выровненном участке
    GAP = 1  # заплатка PCHIP над невыровненным участком (отступ, короткая строка)
    ANOMALY = 2  # заплатка над недостоверным участком (выступ сора, ступенька выноса — ``edge_guard``)


class SideCode(IntEnum):
    """Код стороны блока в массивах сайдкара."""

    LEFT = 0
    RIGHT = 1


class RuleAxisCode(IntEnum):
    """Код оси трассы линейки в массивах сайдкара."""

    HORIZONTAL = 0
    VERTICAL = 1


@dataclass(frozen=True)
class AxisRecord:
    """Ось строки из сайдкара.

    Attributes:
        points: Ломаная оси ``(N, 2)`` слева направо (основная ось разбора).
        body_points: Вторая ось по базовой линии глифов ``(M, 2)`` или пустой массив.
        height: Высота строки, пиксели.
        column: Номер колонки.
        cross: Кусок строки, разрезанной межколонником.
        block: Номер блока, в ряд которого входит ось; −1 — вне блоков.
        row: Номер ряда в блоке; −1 — вне блоков.
        jump_spans: Участки по x, где ось перескочила на соседнюю строку.
        mark_spans: Участки по x, занятые точками и запятыми.
    """

    points: np.ndarray
    body_points: np.ndarray
    height: float
    column: int
    cross: bool
    block: int
    row: int
    jump_spans: tuple[tuple[float, float], ...] = ()
    mark_spans: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class SideRecord:
    """Вертикальная сторона блока из сайдкара.

    Attributes:
        block: Номер блока.
        side: ``SideCode``.
        raw: Сторона огибающей блока как есть ``(N, 2)`` сверху вниз — без заплаток, с аномалиями.
        points: Дополнительная линия стороны ``(M, 2)`` (``sides.filled_side``): невыровненные концы
            отброшены, невыровненная середина и недостоверные участки заменены PCHIP; пустой — не построилась.
        kinds: ``PointKind`` каждой точки ``points``.
    """

    block: int
    side: SideCode
    raw: np.ndarray
    points: np.ndarray
    kinds: np.ndarray


@dataclass(frozen=True)
class TraceRecord:
    """Трасса линейки таблицы из сайдкара: ломаная через 1 мм по сплайну ``tables.traces.RuleTrace``.

    Attributes:
        points: Ломаная ``(N, 2)``.
        axis: ``RuleAxisCode``.
        table: Номер объекта-таблицы в ``objects`` JSON полосы.
        thickness: Толщина линейки, пиксели.
    """

    points: np.ndarray
    axis: RuleAxisCode
    table: int
    thickness: float


@dataclass(frozen=True)
class BlockRecord:
    """Текстовый блок: мелкие поля из JSON и его стороны из сайдкара.

    Attributes:
        index: Номер блока.
        polygon: Контур блока ``(N, 2)``.
        rows: Число рядов.
        alignment: Вид выключки (``alignment.AlignKind``, строкой).
        aligned_left: Текст выровнен по левой стороне.
        aligned_right: Текст выровнен по правой стороне.
        unreliable_left: Недостоверные участки левой стороны, отрезки по y.
        unreliable_right: То же для правой.
        sides: Вертикальные стороны блока по ``SideCode``.
    """

    index: int
    polygon: np.ndarray
    rows: int
    alignment: str
    aligned_left: bool
    aligned_right: bool
    unreliable_left: tuple[tuple[float, float], ...]
    unreliable_right: tuple[tuple[float, float], ...]
    sides: dict[SideCode, SideRecord] = field(default_factory=dict)


@dataclass(frozen=True)
class PageGeometry:
    """Геометрия разбора полосы: всё, что нужно мерам формы, без пересчёта разбора.

    Attributes:
        page: Имя полосы.
        dpi: Разрешение координат (рабочая копия текстовых блоков).
        size: Размер полосы в родных пикселях ``[ширина, высота]`` (``dpi_native``).
        dpi_native: Родное разрешение полосы (координаты ``objects`` и ``loose_rules``).
        axes: Оси строк.
        blocks: Блоки.
        traces: Трассы линеек таблиц.
        objects: Объекты полосы из JSON (класс, рамка в родных пикселях, источник).
        loose_rules: Линейки-сироты из JSON.
    """

    page: str
    dpi: float
    size: tuple[int, int]
    dpi_native: float
    axes: tuple[AxisRecord, ...]
    blocks: tuple[BlockRecord, ...]
    traces: tuple[TraceRecord, ...]
    objects: tuple[dict, ...]
    loose_rules: tuple[dict, ...]


def _flat(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Список кривых ``(N_i, 2)`` → плоский массив точек ``float32`` и смещения ``(K + 1,)``."""
    offsets = np.zeros(len(curves) + 1, dtype=np.int64)
    for index, curve in enumerate(curves):
        offsets[index + 1] = offsets[index] + len(curve)
    points = (
        np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1, 2) for c in curves])
        if curves and offsets[-1]
        else np.zeros((0, 2), np.float32)
    )
    return points, offsets


def _split(points: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """Обратное к :func:`_flat`: плоский массив и смещения → список кривых."""
    return [points[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def axis_membership(analysis) -> tuple[np.ndarray, np.ndarray]:
    """Принадлежность осей страницы рядам блоков.

    Сначала — по ``id`` объекта оси (``page.marked_jumps`` подменяет ось в обоих местах сразу). Но
    поздние шаги разбора (вторая ось, защита сторон) пересобирают оси рядов новыми объектами, поэтому
    запасной ключ — геометрия: концы оси и медиана её ординаты с точностью до 0.1 px.

    Args:
        analysis: Разбор текстовых блоков (``page.PageAnalysis``).

    Returns:
        ``(блок, ряд)`` — массивы ``(A,)`` по осям страницы; −1 — ось не вошла ни в один ряд.
    """
    by_id, by_shape = {}, {}
    for block_index, block in enumerate(analysis.blocks):
        for row_index, row in enumerate(block.rows):
            for axis in row.axes:
                by_id[id(axis)] = (block_index, row_index)
                by_shape.setdefault(_shape_key(axis.points), (block_index, row_index))
    block = np.full(len(analysis.axes), -1, dtype=np.int32)
    row = np.full(len(analysis.axes), -1, dtype=np.int32)
    for index, axis in enumerate(analysis.axes):
        found = by_id.get(id(axis)) or by_shape.get(_shape_key(axis.points))
        if found is not None:
            block[index], row[index] = found
    return block, row


def _shape_key(points: np.ndarray) -> tuple[float, float, float]:
    """Геометрический ключ оси: абсциссы концов и медиана ординаты, округлённые до 0.1 px."""
    points = np.asarray(points, dtype=np.float64)
    return (round(float(points[0, 0]), 1), round(float(points[-1, 0]), 1), round(float(np.median(points[:, 1])), 1))


def point_kinds(line, unreliable: tuple[tuple[float, float], ...]) -> np.ndarray:
    """``PointKind`` точек дополнительной линии стороны.

    Args:
        line: ``sides.FilledSide``.
        unreliable: Недостоверные участки этой стороны (отрезки по y кадра).

    Returns:
        ``int8 (N,)``: своя точка, заплатка над невыровненным или над недостоверным участком.
    """
    kinds = np.where(np.asarray(line.filled, dtype=bool), PointKind.GAP, PointKind.ALIGNED).astype(np.int8)
    ys = np.asarray(line.points)[:, 1]
    for y0, y1 in unreliable:
        kinds[np.asarray(line.filled, dtype=bool) & (ys >= y0) & (ys <= y1)] = PointKind.ANOMALY
    return kinds


def page_arrays(analysis, lines: list[dict], traces: list[tuple[int, object]] | None = None) -> dict[str, np.ndarray]:
    """Массивы сайдкара полосы.

    Args:
        analysis: Разбор текстовых блоков (``page.PageAnalysis``).
        lines: Дополнительные линии сторон по блокам (``pack_analysis.final.side_lines``): словарь
            ``sides.SideKind → FilledSide | None``.
        traces: Трассы линеек таблиц ``(номер объекта-таблицы, RuleTrace)`` в пикселях рабочей копии; ``None`` — нет.

    Returns:
        Словарь имя → массив для ``np.savez_compressed``.
    """
    from ocr_utils.page_layout.text_blocks.sides import SideKind

    axes = list(analysis.axes)
    block_of, row_of = axis_membership(analysis)
    axis_points, axis_offsets = _flat([a.points for a in axes])
    body_points, body_offsets = _flat([a.body_points if a.body_points is not None else np.zeros((0, 2)) for a in axes])
    jumps = [(i, s, e) for i, a in enumerate(axes) for s, e in a.jump_spans]
    marks = [(i, s, e) for i, a in enumerate(axes) for s, e in a.mark_spans]
    # Стороны: сырая кривая огибающей и дополнительная линия — по одной записи на (блок, сторона).
    raw_curves, fill_curves, fill_kinds, side_block, side_code = [], [], [], [], []
    for block_index, (block, own) in enumerate(zip(analysis.blocks, lines)):
        for kind, code in ((SideKind.LEFT, SideCode.LEFT), (SideKind.RIGHT, SideCode.RIGHT)):
            envelope = block.envelope
            raw = envelope.left if code is SideCode.LEFT else envelope.right
            unreliable = tuple(getattr(envelope, f"unreliable_{kind.value}", ()))
            line = own.get(kind)
            raw_curves.append(np.asarray(raw, dtype=np.float64).reshape(-1, 2))
            fill_curves.append(np.asarray(line.points).reshape(-1, 2) if line is not None else np.zeros((0, 2)))
            fill_kinds.append(point_kinds(line, unreliable) if line is not None else np.zeros(0, np.int8))
            side_block.append(block_index)
            side_code.append(int(code))
    raw_points, raw_offsets = _flat(raw_curves)
    fill_points, fill_offsets = _flat(fill_curves)
    traced = traces or []
    trace_points, trace_offsets = _flat([np.asarray(t.sample(1.0)) for _, t in traced])
    return {
        "format": np.array([STORE_FORMAT_VERSION], np.int32),
        "dpi": np.array([analysis.dpi], np.float32),
        "axis_points": axis_points,
        "axis_offsets": axis_offsets,
        "axis_body_points": body_points,
        "axis_body_offsets": body_offsets,
        "axis_height": np.array([a.height for a in axes], np.float32),
        "axis_column": np.array([a.column for a in axes], np.int32),
        "axis_cross": np.array([a.cross for a in axes], np.bool_),
        "axis_block": block_of,
        "axis_row": row_of,
        "axis_jumps": np.array(jumps, np.float32).reshape(-1, 3),
        "axis_marks": np.array(marks, np.float32).reshape(-1, 3),
        "side_block": np.array(side_block, np.int32),
        "side_code": np.array(side_code, np.int8),
        "side_raw_points": raw_points,
        "side_raw_offsets": raw_offsets,
        "side_fill_points": fill_points,
        "side_fill_offsets": fill_offsets,
        "side_fill_kinds": (np.concatenate(fill_kinds) if fill_kinds else np.zeros(0)).astype(np.int8),
        "trace_points": trace_points,
        "trace_offsets": trace_offsets,
        "trace_axis": np.array(
            [RuleAxisCode.HORIZONTAL if t.horizontal else RuleAxisCode.VERTICAL for _, t in traced], np.int8
        ),
        "trace_table": np.array([index for index, _ in traced], np.int32),
        "trace_thickness": np.array([t.thickness_px for _, t in traced], np.float32),
    }


def write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Записать сайдкар атомарно: во временный файл и переименованием (прерванный прогон не оставит обрывок)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _spans(table: np.ndarray, index: int) -> tuple[tuple[float, float], ...]:
    """Участки оси ``index`` из массива ``(K, 3)`` «ось, от, до»."""
    return tuple((float(s), float(e)) for i, s, e in table if int(i) == index)


def load_page(json_path: Path) -> PageGeometry:
    """Геометрия разбора полосы по итоговому JSON и его сайдкару.

    Args:
        json_path: ``<разбор>/pages/<полоса>.json``; сайдкар — имя из ``text_blocks.arrays`` рядом с ним.

    Returns:
        :class:`PageGeometry`. Без сайдкара (разбор старше формата) — оси, стороны и трассы пустые.
    """
    payload = json.loads(Path(json_path).read_text())
    blocks_json = payload["text_blocks"]
    name = blocks_json.get("arrays")
    arrays = None
    if name:
        with np.load(Path(json_path).parent / name) as data:
            arrays = {key: data[key] for key in data.files}
    axes: list[AxisRecord] = []
    sides: dict[int, dict[SideCode, SideRecord]] = {}
    traces: list[TraceRecord] = []
    if arrays is not None:
        points = _split(arrays["axis_points"], arrays["axis_offsets"])
        bodies = _split(arrays["axis_body_points"], arrays["axis_body_offsets"])
        for i, curve in enumerate(points):
            axes.append(
                AxisRecord(
                    points=curve.astype(np.float64),
                    body_points=bodies[i].astype(np.float64),
                    height=float(arrays["axis_height"][i]),
                    column=int(arrays["axis_column"][i]),
                    cross=bool(arrays["axis_cross"][i]),
                    block=int(arrays["axis_block"][i]),
                    row=int(arrays["axis_row"][i]),
                    jump_spans=_spans(arrays["axis_jumps"], i),
                    mark_spans=_spans(arrays["axis_marks"], i),
                )
            )
        raws = _split(arrays["side_raw_points"], arrays["side_raw_offsets"])
        fills = _split(arrays["side_fill_points"], arrays["side_fill_offsets"])
        kinds = _split(arrays["side_fill_kinds"].reshape(-1, 1), arrays["side_fill_offsets"])
        for i, (raw, fill, kind) in enumerate(zip(raws, fills, kinds)):
            block, code = int(arrays["side_block"][i]), SideCode(int(arrays["side_code"][i]))
            sides.setdefault(block, {})[code] = SideRecord(
                block, code, raw.astype(np.float64), fill.astype(np.float64), kind.ravel().astype(np.int8)
            )
        for i, curve in enumerate(_split(arrays["trace_points"], arrays["trace_offsets"])):
            traces.append(
                TraceRecord(
                    curve.astype(np.float64),
                    RuleAxisCode(int(arrays["trace_axis"][i])),
                    int(arrays["trace_table"][i]),
                    float(arrays["trace_thickness"][i]),
                )
            )
    blocks = []
    for index, block in enumerate(blocks_json["blocks"]):
        alignment = block.get("alignment") or {}
        blocks.append(
            BlockRecord(
                index=index,
                polygon=np.asarray(block["polygon"], dtype=np.float64),
                rows=int(block.get("rows", 0)),
                alignment=str(alignment.get("kind", "")),
                aligned_left=bool(alignment.get("left", False)),
                aligned_right=bool(alignment.get("right", False)),
                unreliable_left=tuple(tuple(span) for span in block.get("unreliable_left", [])),
                unreliable_right=tuple(tuple(span) for span in block.get("unreliable_right", [])),
                sides=sides.get(index, {}),
            )
        )
    return PageGeometry(
        page=payload["page"],
        dpi=float(blocks_json["dpi"]),
        size=tuple(payload["size"]),
        dpi_native=float(payload["dpi"]),
        axes=tuple(axes),
        blocks=tuple(blocks),
        traces=tuple(traces),
        objects=tuple(payload.get("objects", [])),
        loose_rules=tuple(payload.get("loose_rules", [])),
    )


__all__ = [
    "AxisRecord",
    "BlockRecord",
    "PageGeometry",
    "PointKind",
    "RuleAxisCode",
    "STORE_FORMAT_VERSION",
    "SideCode",
    "SideRecord",
    "TraceRecord",
    "axis_membership",
    "load_page",
    "page_arrays",
    "point_kinds",
    "write_arrays",
]
