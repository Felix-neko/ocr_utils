"""Валидационные метрики разбора: чем мерить, стало лучше или хуже.

Считается по выкладке прогона (``--out-dir``, там лежат ``pages/*.json``), а не по живому
разбору: так сравниваются два прогона с разными настройками, и числа не зависят от того, что
сейчас в коде.

Главная метрика — **скрещивания осей**. Две оси, перекрывающиеся по x, при верном разборе не
меняют порядок по вертикали внутри перекрытия: строки не пересекаются. Перескок оси на соседнюю
строку именно меняет порядок, поэтому счётчик скрещиваний ловит ровно тот дефект, ради которого
переделывалась сцепка, и не требует ни поля хода строк, ни межстрочного шага.

Прежняя мера перескока — «ось ушла от своей хорды дальше полушага» — оставлена для
сопоставимости с записанными ранее числами, но на изогнутой бумаге она срабатывает и на честно
изогнутых строках.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Оси короче этого (пиксели рабочей копии) в мерах не участвуют: у обрывка в две буквы ни хорды,
# ни порядка по вертикали толком нет.
MIN_AXIS_PX = 100.0
# Перекрытие пары осей по x, от которого имеет смысл говорить об их взаимном порядке.
MIN_OVERLAP_PX = 20.0
# На сколько пикселей оси должны разойтись, чтобы смену порядка считать настоящей: полпикселя —
# это шум сглаживания, а не скрещивание.
CROSS_MARGIN_PX = 2.0
# Сколько точек берётся вдоль перекрытия при проверке порядка.
CROSS_SAMPLES = 24
# Ряд-половинка: шаг до соседнего ряда меньше этой доли медианного шага блока.
HALF_ROW_SHARE = 0.7
# Две оси СБЛИЗИЛИСЬ, если где-то внутри общего охвата по x расстояние между ними падает ниже
# этой доли межстрочного шага. Соседние строки разбора стоят на шаге друг от друга; перескок
# подводит ось вплотную к соседней, даже когда формально её не пересекает.
CONVERGE_SHARE = 0.45


@dataclass(frozen=True)
class PageMetrics:
    """Числа одной разобранной полосы."""

    key: str
    axes: int
    crossings: int
    converging: int
    jumping: int
    outside: int
    half_rows: int
    rows: int
    blocks: int
    overlap_px2: float
    ink_share: float
    seconds: float

    def row(self) -> str:
        """Строка markdown-таблицы."""
        return (
            f"| {self.key} | {self.axes} | {self.crossings} | {self.converging} | {self.jumping} | {self.outside} | "
            f"{self.half_rows}/{self.rows} | {self.blocks} | {self.overlap_px2:.0f} | "
            f"{self.ink_share:.1%} | {self.seconds:.1f} |"
        )


def _axis_points(axis: dict) -> np.ndarray:
    """Точки оси из JSON в массив ``(n, 2)``, без столбцов точек и запятых.

    Над точкой и запятой ось провисает к базовой линии: знак стоит на ней, а не на оси строки.
    Такие участки в мерах перескока не участвуют — иначе провисание в конце строки считалось бы
    сближением с соседней строкой.
    """
    points = np.asarray(axis.get("points") or [], dtype=np.float64).reshape(-1, 2)
    spans = axis.get("mark_spans") or []
    if not spans or points.shape[0] == 0:
        return points
    keep = np.ones(points.shape[0], dtype=bool)
    for x0, x1 in spans:
        keep &= ~((points[:, 0] >= x0) & (points[:, 0] <= x1))
    return points[keep] if int(keep.sum()) >= 2 else points


def crossings_of(axes: list[np.ndarray]) -> int:
    """Сколько пар осей скрещивается: внутри общего охвата по x они меняются местами.

    Args:
        axes: Оси страницы, каждая — ``(n, 2)`` слева направо.

    Returns:
        Число пар. Пары без достаточного перекрытия по x пропускаются.
    """
    count = 0
    long = [axis for axis in axes if axis.shape[0] >= 2 and axis[-1, 0] - axis[0, 0] >= MIN_AXIS_PX]
    for first in range(len(long)):
        a = long[first]
        for second in range(first + 1, len(long)):
            b = long[second]
            left = max(a[0, 0], b[0, 0])
            right = min(a[-1, 0], b[-1, 0])
            if right - left < MIN_OVERLAP_PX:
                continue
            grid = np.linspace(left, right, CROSS_SAMPLES)
            difference = np.interp(grid, a[:, 0], a[:, 1]) - np.interp(grid, b[:, 0], b[:, 1])
            above = difference <= -CROSS_MARGIN_PX
            below = difference >= CROSS_MARGIN_PX
            if above.any() and below.any():
                count += 1
    return count


def converging_of(axes: list[np.ndarray], pitch: float) -> int:
    """Сколько пар осей сближается теснее ``CONVERGE_SHARE`` шага внутри общего охвата по x.

    Это и есть прямая мера перескока: ось, ушедшая на соседнюю строку, подходит к её оси
    вплотную. Скрещивание (смена порядка) ловит только сквозной проход, а «дотянулся и слился»
    ловится именно сближением.

    Args:
        axes: Оси страницы.
        pitch: Межстрочный шаг страницы (пиксели рабочей копии).

    Returns:
        Число пар.
    """
    if pitch <= 0:
        return 0
    count = 0
    long = [axis for axis in axes if axis.shape[0] >= 2 and axis[-1, 0] - axis[0, 0] >= MIN_AXIS_PX]
    for first in range(len(long)):
        a = long[first]
        for second in range(first + 1, len(long)):
            b = long[second]
            left = max(a[0, 0], b[0, 0])
            right = min(a[-1, 0], b[-1, 0])
            if right - left < MIN_OVERLAP_PX:
                continue
            grid = np.linspace(left, right, CROSS_SAMPLES)
            gap = np.abs(np.interp(grid, a[:, 0], a[:, 1]) - np.interp(grid, b[:, 0], b[:, 1]))
            if float(gap.min()) < CONVERGE_SHARE * pitch:
                count += 1
    return count


def jumping_of(axes: list[np.ndarray], pitch: float) -> int:
    """Сколько осей уходит от собственной хорды дальше полушага строк (прежняя мера перескока)."""
    if pitch <= 0:
        return 0
    count = 0
    for axis in axes:
        if axis.shape[0] < 2 or axis[-1, 0] - axis[0, 0] < MIN_AXIS_PX:
            continue
        xs, ys = axis[:, 0], axis[:, 1]
        chord = ys[0] + (ys[-1] - ys[0]) * (xs - xs[0]) / max(xs[-1] - xs[0], 1e-6)
        if float(np.abs(ys - chord).max()) > 0.5 * pitch:
            count += 1
    return count


def outside_of(axes: list[np.ndarray], polygons: list[np.ndarray]) -> int:
    """Сколько осей не попало серединой ни в один контур блока."""
    if not polygons:
        return len(axes)
    hulls = [np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2) for polygon in polygons if len(polygon) >= 3]
    count = 0
    for axis in axes:
        if axis.shape[0] == 0:
            continue
        middle = axis[axis.shape[0] // 2]
        point = (float(middle[0]), float(middle[1]))
        if not any(cv2.pointPolygonTest(hull, point, False) >= 0 for hull in hulls):
            count += 1
    return count


def half_rows_of(blocks: list[dict]) -> tuple[int, int]:
    """Ряды-половинки и общее число рядов: шаг до соседа меньше доли медианного шага блока."""
    bad = total = 0
    for block in blocks:
        ys = np.sort(np.array([row["y"] for row in block.get("rows", [])], dtype=np.float64))
        total += ys.size
        if ys.size < 3:
            continue
        steps = np.diff(ys)
        median = float(np.median(steps))
        if median > 0:
            bad += int((steps < HALF_ROW_SHARE * median).sum())
    return bad, total


def overlap_of(polygons: list[np.ndarray]) -> float:
    """Суммарная площадь пересечения контуров блоков (px²) — мера «контуры лезут друг на друга»."""
    hulls = [
        cv2.convexHull(np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2))
        for polygon in polygons
        if len(polygon) >= 3
    ]
    total = 0.0
    for first in range(len(hulls)):
        for second in range(first + 1, len(hulls)):
            area, _ = cv2.intersectConvexConvex(hulls[first], hulls[second])
            total += float(area)
    return total


def pitch_of(blocks: list[dict]) -> float:
    """Шаг строк полосы: медиана шагов между соседними РЯДАМИ всех блоков.

    Медиана по шагам БЛОКОВ для этого не годится: блок заголовка из двух строк с большим
    интервалом задирает её вдвое. На 1973/11 с.79 медиана по блокам даёт 41.9 px при настоящем
    шаге 23 px, и мера сближения начинает срабатывать на здоровых соседних строках.
    """
    steps: list[float] = []
    for block in blocks:
        ys = np.sort(np.array([row["y"] for row in block.get("rows", [])], dtype=np.float64))
        if ys.size >= 2:
            steps.extend(np.diff(ys).tolist())
    return float(np.median(steps)) if steps else 0.0


def metrics_of(page: dict) -> PageMetrics:
    """Метрики одной полосы по её JSON-разбору."""
    axes = [_axis_points(axis) for axis in page.get("axes", [])]
    blocks = page.get("blocks", [])
    polygons = [np.asarray(block["envelope"]["polygon"], dtype=np.float64) for block in blocks if block.get("envelope")]
    pitch = pitch_of(blocks)
    bad_rows, rows = half_rows_of(blocks)
    return PageMetrics(
        key=str(page.get("key") or f"{page.get('name')}_{page.get('page')}_{page.get('variant')}"),
        axes=len(axes),
        crossings=crossings_of(axes),
        converging=converging_of(axes, pitch),
        jumping=jumping_of(axes, pitch),
        outside=outside_of(axes, polygons),
        half_rows=bad_rows,
        rows=rows,
        blocks=len(blocks),
        overlap_px2=overlap_of(polygons),
        ink_share=float(page.get("ink_share") or 0.0),
        seconds=float(page.get("seconds") or 0.0),
    )


def read_pages(directory: Path) -> list[PageMetrics]:
    """Метрики по всем ``pages/*.json`` выкладки прогона."""
    folder = directory / "pages" if (directory / "pages").is_dir() else directory
    return [metrics_of(json.loads(path.read_text(encoding="utf-8"))) for path in sorted(folder.glob("*.json"))]


def totals(pages: list[PageMetrics]) -> PageMetrics:
    """Итог по набору: суммы счётчиков, среднее покрытие краски, общее время."""
    return PageMetrics(
        key="ИТОГО",
        axes=sum(page.axes for page in pages),
        crossings=sum(page.crossings for page in pages),
        converging=sum(page.converging for page in pages),
        jumping=sum(page.jumping for page in pages),
        outside=sum(page.outside for page in pages),
        half_rows=sum(page.half_rows for page in pages),
        rows=sum(page.rows for page in pages),
        blocks=sum(page.blocks for page in pages),
        overlap_px2=sum(page.overlap_px2 for page in pages),
        ink_share=float(np.mean([page.ink_share for page in pages])) if pages else 0.0,
        seconds=sum(page.seconds for page in pages),
    )


def table(pages: list[PageMetrics], against: list[PageMetrics] | None = None, brief: bool = False) -> str:
    """Таблица markdown по страницам и итог.

    Args:
        pages: Метрики полос прогона.
        against: Метрики прогона, с которым сравниваем; тогда печатаются только два итога.
        brief: Печатать один итог без построчной части.
    """
    header = (
        "| страница | осей | скрещиваний | сближений | от хорды | вне блоков | половинок | блоков | "
        "пересечения px² | краска | с |\n|---|---|---|---|---|---|---|---|---|---|"
    )
    if against is None:
        rows = [] if brief else [page.row() for page in pages]
        return "\n".join([header, *rows, totals(pages).row()])
    left, right = totals(against), totals(pages)
    return "\n".join([header, left.row().replace("ИТОГО", "было"), right.row().replace("ИТОГО", "стало")])


__all__ = ["PageMetrics", "converging_of", "crossings_of", "pitch_of", "metrics_of", "read_pages", "table", "totals"]
