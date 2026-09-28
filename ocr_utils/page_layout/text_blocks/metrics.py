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

from ocr_utils.page_layout import mm_to_px, px_to_mm

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
# Ось через межколонник (:func:`gutter_crossings_of`). Пустота под осью уже этого (мм) — пробел
# между словами, а не межколонник: межколонник в наборе журнала 3–6 мм, пробел корпуса — 1–2 мм.
CROSSING_GAP_MM = 2.5
# ...и уже стольких высот строки: у заголовка пробел между словами шире 2.5 мм, но уже его
# собственной высоты.
CROSSING_GAP_HEIGHTS = 1.2
# Полуокно пробы краски под осью по вертикали — в высотах строки (не меньше MIN_PROBE_PX).
CROSSING_PROBE_HEIGHTS = 0.6
CROSSING_MIN_PROBE_PX = 3.0
# По обе стороны пустоты вдоль оси должно быть не меньше стольких мм краски строки.
CROSSING_SIDE_MM = 5.0
# Пустой столбец продлевается вверх и вниз не дальше этого (мм) и считается перегороженным, если
# краска занимает больше этой доли его ширины (как ``columns.BLOCK_INK_SHARE``).
CROSSING_REACH_MM = 40.0
CROSSING_BLOCK_SHARE = 0.3
# Столбец сужается с краёв на эту долю: буква соседней колонки, заходящая в межколонник на
# пиксель, не должна его перегораживать.
CROSSING_SHRINK_SHARE = 0.2
# Пустота под осью должна быть шире медианы ОСТАЛЬНЫХ пробелов той же строки во столько раз: в строке,
# сшитой через межколонник, он 4–5 мм против пробелов 1–2 мм, а в строке с «рекой» пробелов по формату
# все пробелы растянуты одинаково (выборка по паку 2026-09-28: пояса «1–3 оси на полосу» — сплошь реки).
CROSSING_SPACE_RATIO = 1.8
# Пробелом между словами считается пустота под осью от этой ширины (мм): уже — просвет между буквами.
CROSSING_SPACE_MIN_MM = 0.5
# Ширина пустоты, общей для ВСЕХ строк пикселей столбца (мм): у межколонника края колонок стоят
# на месте, а «река» пробелов по формату гуляет от строки к строке, и общей пустоты почти не остаётся.
CROSSING_COMMON_MM = 2.0
# Строк (осей), подходящих к пустому столбцу с каждой стороны на его высоте, включая саму ось. У «реки»
# пробелов по формату — 3–4, у настоящего межколонника — от 5 (выборка по паку 2026-09-28: 24 полосы
# поясов «1–9 осей», 8 настоящих). Цена: сращённый фрагмент короче 5 строк мера не видит.
CROSSING_NEIGHBOURS = 5
# На каком расстоянии от края пустоты (мм) проверяется, что ось соседней строки подходит к ней.
CROSSING_NEIGHBOUR_REACH_MM = 2.0


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


@dataclass(frozen=True)
class GutterCrossing:
    """Ось строки, идущая через межколонник: пустая вертикальная полоса под ней и по соседям."""

    axis: int  # номер оси в ``axes`` страницы
    x0: float  # пустота под осью по x, пиксели рабочей копии
    x1: float
    y: float  # ордината оси посередине пустоты
    top: float  # насколько пустой столбец тянется вверх и вниз
    bottom: float
    left: int  # сколько осей подходит к столбцу слева и справа на его высоте (с самой осью)
    right: int
    gap_mm: float  # ширина пустоты под осью

    def to_json(self) -> dict:
        """Словарь для JSON и CSV стенда."""
        return {key: (round(value, 1) if isinstance(value, float) else value) for key, value in self.__dict__.items()}


def _y_at(points: np.ndarray, x: float) -> float | None:
    """Ордината оси на абсциссе ``x``; ``None`` — ``x`` за концами оси."""
    if points.shape[0] < 2 or not points[0, 0] <= x <= points[-1, 0]:
        return None
    return float(np.interp(x, points[:, 0], points[:, 1]))


def _column_clear(glyphs: np.ndarray, x0: float, x1: float, y: int) -> bool:
    """Пуста ли полоса ``[x0, x1)`` маски глифов в строке пикселей ``y`` (краски не больше доли)."""
    a, b = int(max(0, np.floor(x0))), int(min(glyphs.shape[1], np.ceil(x1)))
    if b <= a or not 0 <= y < glyphs.shape[0]:
        return False
    return bool(glyphs[y, a:b].mean() <= CROSSING_BLOCK_SHARE)


def _empty_extent(glyphs: np.ndarray, x0: float, x1: float, y: float, slope: float, reach: int) -> tuple[int, int]:
    """Докуда вверх и вниз от ``y`` тянется пустой столбец ``[x0, x1]``.

    Межколонник перпендикулярен строкам, поэтому на наклонной полосе столбец идёт с наклоном
    ``-slope`` (сдвиг по x на пиксель высоты).

    Args:
        glyphs: Маска глифов рабочей копии.
        x0, x1: Границы столбца на высоте ``y``.
        y: Откуда идти.
        slope: Наклон строки dy/dx.
        reach: Дальше этого (пиксели) не ходить.

    Returns:
        Пара ``(top, bottom)`` — крайние пустые строки пикселей.
    """
    start = int(round(y))
    top = start
    while start - top < reach and _column_clear(
        glyphs, x0 + (top - 1 - y) * -slope, x1 + (top - 1 - y) * -slope, top - 1
    ):
        top -= 1
    bottom = start
    while bottom - start < reach and _column_clear(
        glyphs, x0 + (bottom + 1 - y) * -slope, x1 + (bottom + 1 - y) * -slope, bottom + 1
    ):
        bottom += 1
    return top, bottom


def _common_width(
    glyphs: np.ndarray, x0: float, x1: float, y: float, slope: float, top: int, bottom: int, reach: float
) -> float:
    """Ширина пустоты вокруг середины ``[x0, x1]``, общей для всех строк пикселей от ``top`` до ``bottom``.

    Args:
        glyphs: Маска глифов рабочей копии.
        x0, x1: Пустота под осью на высоте ``y``.
        y: Ордината оси.
        slope: Наклон строки dy/dx (столбец идёт с наклоном ``-slope``).
        top, bottom: Высота пустого столбца.
        reach: Насколько шире пустоты под осью смотреть в стороны (пиксели).

    Returns:
        Ширина в пикселях рабочей копии; 0 — середина перегорожена хоть в одной строке.
    """
    left = int(np.floor(x0 - reach))
    width = int(np.ceil(x1 + reach)) - left
    free = np.ones(width, dtype=bool)
    for row in range(max(0, top), min(glyphs.shape[0], bottom + 1)):
        shift = int(round((row - y) * -slope))
        a = left + shift
        # Окно строки, обрезанное краями маски; вне маски — пусто.
        lo, hi = max(0, a), min(glyphs.shape[1], a + width)
        if hi <= lo:
            continue
        line = np.zeros(width, dtype=bool)
        line[lo - a : hi - a] = glyphs[row, lo:hi]
        free &= ~line
    centre = int(round((x0 + x1) / 2.0)) - left
    if not 0 <= centre < width or not free[centre]:
        return 0.0
    start = centre
    while start > 0 and free[start - 1]:
        start -= 1
    end = centre
    while end < width - 1 and free[end + 1]:
        end += 1
    return float(end - start + 1)


def _neighbours(curves: list[np.ndarray], x: float, top: float, bottom: float) -> int:
    """Сколько осей проходит через абсциссу ``x`` на высоте между ``top`` и ``bottom``."""
    count = 0
    for curve in curves:
        y = _y_at(curve, x)
        if y is not None and top <= y <= bottom:
            count += 1
    return count


def gutter_crossings_of(axes: list[dict], glyphs: np.ndarray, dpi: float) -> list[GutterCrossing]:
    """Оси, идущие через межколонник: под осью пустота шире пробела, и по соседним строкам она продолжается.

    Признак сращивания колонок (1966/02 IMG_0076_1L): ось левой колонки уходит в правую через
    пустую полосу. Ось помечается, если одновременно:

    1. под ней (в окне ± ``CROSSING_PROBE_HEIGHTS`` высоты) есть пустота по x шире
       ``CROSSING_GAP_MM`` и ``CROSSING_GAP_HEIGHTS`` высот строки, а по обе стороны пустоты вдоль
       оси не меньше ``CROSSING_SIDE_MM`` краски;
    2. пустота продолжается пустым столбцом вверх и вниз, и на высоте этого столбца к нему с каждой
       стороны подходит не меньше ``CROSSING_NEIGHBOURS`` осей (с самой осью): пробелы заголовков и
       случайные совпадения пробелов в паре строк так отсекаются;
    3. по всей высоте столбца остаётся ОБЩАЯ пустота не уже ``CROSSING_COMMON_MM``: «река» пробелов
       в наборе по формату гуляет от строки к строке, а края колонок у межколонника стоят;
    4. пустота шире остальных пробелов той же строки в ``CROSSING_SPACE_RATIO`` раз: у реки пробелы
       строки растянуты одинаково.

    Отточия через межколонник пустоты под осью не дают (точки — краска), заголовок, набранный
    через межколонник, — тоже.

    Args:
        axes: Оси страницы из JSON разбора (``points``, ``height``), пиксели рабочей копии.
        glyphs: Маска глифов рабочей копии (ненулевое — краска глифа), того же размера, что разбор.
        dpi: Разрешение рабочей копии.

    Returns:
        Находки по одной на ось (самая широкая пустота), в порядке осей.
    """
    glyphs = np.asarray(glyphs) > 0
    curves = [np.asarray(axis.get("points") or [], dtype=np.float64).reshape(-1, 2) for axis in axes]
    min_gap_mm = mm_to_px(CROSSING_GAP_MM, dpi)
    side = mm_to_px(CROSSING_SIDE_MM, dpi)
    reach = mm_to_px(CROSSING_REACH_MM, dpi)
    near = mm_to_px(CROSSING_NEIGHBOUR_REACH_MM, dpi)
    common = mm_to_px(CROSSING_COMMON_MM, dpi)
    space_min = mm_to_px(CROSSING_SPACE_MIN_MM, dpi)
    out: list[GutterCrossing] = []
    for index, (axis, curve) in enumerate(zip(axes, curves)):
        if curve.shape[0] < 2 or curve[-1, 0] - curve[0, 0] < 2 * side + min_gap_mm:
            continue
        height = float(axis.get("height") or 0.0)
        half = max(CROSSING_MIN_PROBE_PX, CROSSING_PROBE_HEIGHTS * height)
        # Краска под осью по столбцам: окно ± half вокруг ординаты оси.
        xs = np.arange(int(np.ceil(curve[0, 0])), int(np.floor(curve[-1, 0])) + 1)
        xs = xs[(xs >= 0) & (xs < glyphs.shape[1])]
        if xs.size == 0:
            continue
        ys = np.interp(xs, curve[:, 0], curve[:, 1])
        lo = np.clip(np.round(ys - half).astype(int), 0, glyphs.shape[0])
        hi = np.clip(np.round(ys + half).astype(int) + 1, 0, glyphs.shape[0])
        inked = np.array([glyphs[a:b, x].any() for x, a, b in zip(xs, lo, hi)])
        # Пустоты внутри оси: от первой краски до последней.
        filled = np.nonzero(inked)[0]
        if filled.size == 0:
            continue
        empty = ~inked
        empty[: filled[0]] = False
        empty[filled[-1] + 1 :] = False
        edges = np.diff(np.r_[0, empty.astype(np.int8), 0])
        starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
        min_gap = max(min_gap_mm, CROSSING_GAP_HEIGHTS * height)
        widths = ends - starts
        spaces = widths[widths >= space_min]
        best: GutterCrossing | None = None
        for start, end in zip(starts, ends):
            if end - start < min_gap:
                continue
            # Остальные пробелы строки: пустота должна быть заметно шире их медианы.
            others = spaces[spaces != end - start] if spaces.size > 1 else spaces[:0]
            if others.size and end - start < CROSSING_SPACE_RATIO * float(np.median(others)):
                continue
            if inked[:start].sum() < side or inked[end:].sum() < side:
                continue
            gx0, gx1 = float(xs[start]), float(xs[end - 1] + 1)
            shrink = CROSSING_SHRINK_SHARE * (gx1 - gx0)
            y = float(np.interp((gx0 + gx1) / 2.0, curve[:, 0], curve[:, 1]))
            # Наклон строки у пустоты — по участку оси шириной в две пустоты.
            around = (curve[:, 0] >= gx0 - (gx1 - gx0)) & (curve[:, 0] <= gx1 + (gx1 - gx0))
            slope = float(np.polyfit(curve[around, 0], curve[around, 1], 1)[0]) if around.sum() >= 2 else 0.0
            top, bottom = _empty_extent(glyphs, gx0 + shrink, gx1 - shrink, y, slope, reach)
            if _common_width(glyphs, gx0, gx1, y, slope, top, bottom, gx1 - gx0) < common:
                continue
            # Соседей считаем на ПРОДОЛЖЕНИИ столбца: у самого его края слева и справа.
            left = _neighbours(curves, gx0 - near, top, bottom)
            right = _neighbours(curves, gx1 + near, top, bottom)
            if min(left, right) < CROSSING_NEIGHBOURS:
                continue
            found = GutterCrossing(
                axis=index,
                x0=gx0,
                x1=gx1,
                y=y,
                top=float(top),
                bottom=float(bottom),
                left=left,
                right=right,
                gap_mm=round(px_to_mm(gx1 - gx0, dpi), 2),
            )
            if best is None or found.x1 - found.x0 > best.x1 - best.x0:
                best = found
        if best is not None:
            out.append(best)
    return out


__all__ = [
    "GutterCrossing",
    "gutter_crossings_of",
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
