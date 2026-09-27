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
# На столько пикселей оси позволено выйти за контур своего блока: пиксель — это округление при
# растеризации контура, а не настоящий выход.
ESCAPE_MARGIN_PX = 1.0
# Ступенька оси (:func:`steps_of`) — мера перескока, не зависящая от изгиба строки. В каждой точке оси
# по обе стороны от неё берутся участки ``[x ± STEP_GAP_PITCHES·шаг, x ± STEP_REACH_PITCHES·шаг]``,
# через каждый проводится прямая, и обе прямые продолжаются до ``x``. У изогнутой или наклонённой
# строки они встречаются (изгиб на отрезке в пару шагов — доли пикселя), у перескока расходятся на
# высоту ступеньки. Ступенька больше ``STEP_PITCH_SHARE`` шага — перескок. Центральный зазор нужен,
# чтобы косой переход через ряд (перескок бывает не отвесным: 1971/07 IMG_0020_1L — 40–80 px)
# не попал ни в одну из прямых.
STEP_GAP_PITCHES = 1.5
STEP_REACH_PITCHES = 4.0
STEP_PITCH_SHARE = 0.75
# Наклоны сторон расходятся больше чем на столько градусов — изгиб, а не ступенька.
STEP_BEND_DEG = 10.0


@dataclass(frozen=True)
class PageMetrics:
    """Числа одной разобранной полосы."""

    key: str
    axes: int
    crossings: int
    converging: int
    jumping: int
    steps: int
    outside: int
    escaping: int
    escaping_px: float
    half_rows: int
    rows: int
    blocks: int
    overlap_px2: float
    ink_share: float
    seconds: float

    def row(self) -> str:
        """Строка markdown-таблицы."""
        return (
            f"| {self.key} | {self.axes} | {self.crossings} | {self.converging} | {self.steps} | {self.jumping} | "
            f"{self.outside} | "
            f"{self.escaping} ({self.escaping_px:.0f}) | {self.half_rows}/{self.rows} | {self.blocks} | "
            f"{self.overlap_px2:.0f} | {self.ink_share:.1%} | {self.seconds:.1f} |"
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


def _side_slope(xs: np.ndarray, ys: np.ndarray, gap: float) -> float | None:
    """Наклон одной стороны, если по ней его можно мерить: от трёх точек на охвате от ``gap / 3``."""
    if xs.size < 3 or np.ptp(xs) < gap / 3.0:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


def _joint_step(xs: np.ndarray, ys: np.ndarray, at: float, gap: float, reach: float) -> float | None:
    """Ступенька в точке ``at``: МНК ``y = a + b·(x − at) + c·[x > at]`` по участкам слева и справа от неё.

    Наклон ``b`` общий для обеих сторон: у перескоковой оси точек мало и они с пропусками (крупный
    масштаб собирает строку по отдельным высоким буквам), и прямая по двум узлам одной стороны,
    продолженная на полтора шага, давала бы ложную ступеньку. Изгиб строки на участке в несколько
    шагов — доли пикселя, ступенька перескока — шаг строки.

    Args:
        xs, ys: Точки оси.
        at: Где мерить.
        gap: Полуширина зазора вокруг ``at``, точки в котором не берутся (косой переход через ряд).
        reach: До какого расстояния от ``at`` брать точки.

    Returns:
        Ступенька ``c`` в пикселях; ``None`` — с какой-то стороны меньше двух точек.
    """
    offset = xs - at
    left = (offset <= -gap) & (offset >= -reach)
    right = (offset >= gap) & (offset <= reach)
    if left.sum() < 2 or right.sum() < 2:
        return None
    # Если обе стороны сами по себе прямые и их наклоны расходятся — это изгиб (завиток конца строки у
    # корешка: 1966/01 IMG_0049_1L, «…как при»), а не ступенька: общий наклон его не описывает.
    slopes = [_side_slope(offset[side], ys[side], gap) for side in (left, right)]
    if None not in slopes and abs(slopes[0] - slopes[1]) > np.tan(np.radians(STEP_BEND_DEG)):
        return None
    chosen = left | right
    design = np.column_stack([np.ones(chosen.sum()), offset[chosen], right[chosen].astype(np.float64)])
    coefficients, *_ = np.linalg.lstsq(design, ys[chosen], rcond=None)
    # Ступенька не больше размаха самих точек: при коротком крутом участке на конце строки общий
    # наклон и ступенька делят изгиб между собой, и МНК мог насчитать ступеньку втрое больше размаха.
    spread = float(np.ptp(ys[chosen]))
    return float(np.clip(coefficients[2], -spread, spread))


def step_of(axis: np.ndarray, pitch: float) -> float:
    """Наибольшая ступенька оси (пиксели): см. :func:`_joint_step`, точки проверки — через пятую долю шага.

    Args:
        axis: Ось ``(n, 2)`` слева направо.
        pitch: Межстрочный шаг.

    Returns:
        Ступенька в пикселях; 0.0, если ось короче, чем нужно для двух участков.
    """
    if pitch <= 0 or axis.shape[0] < 4:
        return 0.0
    xs, ys = axis[:, 0], axis[:, 1]
    gap, reach = STEP_GAP_PITCHES * pitch, STEP_REACH_PITCHES * pitch
    best = 0.0
    for at in np.arange(xs[0] + gap, xs[-1] - gap, max(1.0, pitch / 5.0)):
        step = _joint_step(xs, ys, float(at), gap, reach)
        if step is not None:
            best = max(best, abs(step))
    return best


def steps_of(axes: list[np.ndarray], pitch: float) -> int:
    """Сколько осей со ступенькой больше ``STEP_PITCH_SHARE`` шага — прямая мера перескока (см. ``step_of``).

    В отличие от «отклонения от хорды» (:func:`jumping_of`) не срабатывает на изогнутых и наклонённых
    строках: прогиб в целый шаг на длине строки — плавный, а перескок — ступенька.

    Args:
        axes: Оси страницы.
        pitch: Межстрочный шаг страницы.

    Returns:
        Число осей.
    """
    return sum(1 for axis in axes if step_of(axis, pitch) > STEP_PITCH_SHARE * pitch)


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


def escaping_of(axes: list[np.ndarray], polygons: list[np.ndarray]) -> tuple[int, float]:
    """Сколько осей ВЫХОДИТ ЗА КОНТУР своего блока и насколько далеко худшая (пиксели копии).

    ``outside_of`` спрашивает только «попала ли середина оси хоть в какой-нибудь блок»; здесь
    вопрос строже и отвечает ровно тому, что видно на оверлее: зелёная линия не должна вылезать
    за синюю рамку своего блока. Свой блок определяется по середине оси — так же, как там.

    Args:
        axes: Точки осей полосы.
        polygons: Контуры блоков.

    Returns:
        Пара ``(сколько осей вылезло, на сколько пикселей худшая)``.
    """
    hulls = [np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2) for polygon in polygons if len(polygon) >= 3]
    count, worst = 0, 0.0
    for axis in axes:
        if axis.shape[0] == 0:
            continue
        middle = axis[axis.shape[0] // 2]
        own = next(
            (hull for hull in hulls if cv2.pointPolygonTest(hull, (float(middle[0]), float(middle[1])), False) >= 0),
            None,
        )
        if own is None:
            continue
        away = -min(cv2.pointPolygonTest(own, (float(x), float(y)), True) for x, y in axis)
        if away > ESCAPE_MARGIN_PX:
            count += 1
            worst = max(worst, away)
    return count, worst


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
    escaping, escaping_px = escaping_of(axes, polygons)
    return PageMetrics(
        key=str(page.get("key") or f"{page.get('name')}_{page.get('page')}_{page.get('variant')}"),
        axes=len(axes),
        crossings=crossings_of(axes),
        converging=converging_of(axes, pitch),
        jumping=jumping_of(axes, pitch),
        steps=steps_of(axes, pitch),
        outside=outside_of(axes, polygons),
        escaping=escaping,
        escaping_px=escaping_px,
        half_rows=bad_rows,
        rows=rows,
        blocks=len(blocks),
        overlap_px2=overlap_of(polygons),
        ink_share=float(page.get("ink_share") or 0.0),
        seconds=float(page.get("seconds") or 0.0),
    )


def blocks_over_cells(blocks: list[dict], cells: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    """Сколько блоков накрывают больше ОДНОЙ ячейки таблицы и сколько блоков вообще в таблицах.

    Прямая мера требования «блок не попадает на несколько ячеек»: контур блока не должен
    пересекать рёбра таблицы. Блоки вне таблиц в счёт не идут — им ячеек не назначено.

    Args:
        blocks: Блоки страницы из JSON разбора.
        cells: Боксы ячеек ``(x0, y0, x1, y1)`` в пикселях рабочей копии.

    Returns:
        Пара ``(блоков на нескольких ячейках, блоков в таблицах)``.
    """
    if not cells:
        return 0, 0
    bad = total = 0
    for block in blocks:
        polygon = np.asarray((block.get("envelope") or {}).get("polygon") or [], dtype=np.float64).reshape(-1, 2)
        if polygon.shape[0] < 3:
            continue
        box = (polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max())
        # Ячейка считается задетой, если блок накрывает больше половины её площади: касание
        # краем — это округление координат, а не захват соседней графы.
        touched = 0
        for x0, y0, x1, y1 in cells:
            overlap = max(0.0, min(box[2], x1) - max(box[0], x0)) * max(0.0, min(box[3], y1) - max(box[1], y0))
            if overlap > 0.5 * max((x1 - x0) * (y1 - y0), 1.0):
                touched += 1
        if touched:
            total += 1
            bad += touched > 1
    return bad, total


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
        steps=sum(page.steps for page in pages),
        outside=sum(page.outside for page in pages),
        escaping=sum(page.escaping for page in pages),
        escaping_px=max([page.escaping_px for page in pages], default=0.0),
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
        "| страница | осей | скрещиваний | сближений | ступенек | от хорды | вне блоков | за контуром (px) | "
        "половинок | блоков | пересечения px² | краска | с |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    if against is None:
        rows = [] if brief else [page.row() for page in pages]
        return "\n".join([header, *rows, totals(pages).row()])
    left, right = totals(against), totals(pages)
    return "\n".join([header, left.row().replace("ИТОГО", "было"), right.row().replace("ИТОГО", "стало")])


__all__ = [
    "PageMetrics",
    "blocks_over_cells",
    "converging_of",
    "step_of",
    "steps_of",
    "crossings_of",
    "escaping_of",
    "pitch_of",
    "metrics_of",
    "read_pages",
    "table",
    "totals",
]
