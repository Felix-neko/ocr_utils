"""Меры сравнения движков осей строк без эталона: полнота, дробление, сдвиг, форма, вихляние, завитки концов."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ocr_utils.page_layout.text_blocks.metrics import metrics_of, pitch_of
from ocr_utils.page_layout import px_to_mm

# Движок, по осям которого сопоставляются строки. Это не эталон истины, а общая «сетка строк»:
# по оверлеям он покрывает краску полнее всех (99 %), и строки всех движков меряются на ней.
REFERENCE = "ink"
# Шаг выборки по x при сравнении осей, пиксели рабочей копии (150 dpi — ~0.34 мм).
SAMPLE_PX = 2.0
# Ось движка относится к строке опоры, если после снятия сдвига движка отстоит от неё меньше
# этой доли шага строк: половина шага — граница с соседней строкой, берём с запасом.
MATCH_PITCH_SHARE = 0.4
# Кусок оси движка засчитывается строке опоры, если покрывает хотя бы эту долю её длины: меньше —
# это касание концом, а не кусок строки.
PIECE_SHARE = 0.1
# Доля длины на каждом конце строки, где меряется завиток.
END_SHARE = 0.15
# Доля самых изогнутых строк опоры (по |сагитте|), на которых меряется завиток концов.
CURLED_SHARE = 0.2
# Шаг для меры вихляния по второй разности, мм: примерно ширина буквы.
WOBBLE_STEP_MM = 1.0


@dataclass(frozen=True)
class Axis:
    """Ось строки из JSON разбора: точки слева направо и сагитта."""

    points: np.ndarray
    sagitta_mm: float

    @property
    def x0(self) -> float:
        return float(self.points[0, 0])

    @property
    def x1(self) -> float:
        return float(self.points[-1, 0])

    def y_at(self, xs: np.ndarray) -> np.ndarray:
        """Ордината оси в абсциссах ``xs`` линейной интерполяцией (вне охвата — клампится)."""
        return np.interp(xs, self.points[:, 0], self.points[:, 1])


@dataclass(frozen=True)
class Page:
    """Разбор одной полосы одним движком."""

    key: tuple[str, int, str]
    engine: str
    axes: list[Axis]
    pitch: float
    dpi: float
    raw: dict


@dataclass(frozen=True)
class EngineScore:
    """Итоговые меры одного движка по всему набору.

    Attributes:
        engine: Имя движка.
        pages: Полос в наборе.
        axes: Осей всего.
        blocks: Блоков всего (дробление строк раздувает число блоков).
        ink_share: Средняя доля краски текста под полосами вокруг осей.
        recall: Доля длины строк опоры, покрытая осями движка.
        pieces: Среднее число осей движка на одну покрытую строку опоры (1 — строка целая).
        split_share: Доля покрытых строк опоры, разорванных на 2+ оси.
        bridges: Осей движка, накрывших строки опоры из РАЗНЫХ колонок (слипание через межколонник).
        offset_mm: Медианный знаковый сдвиг оси движка от опоры, мм (+ — ниже).
        shape_mm: Медиана по строкам медианного |dy| после снятия сдвига строки, мм.
        shape_p90_mm: p90 по строкам максимального |dy| после снятия сдвига, мм.
        curl_mm: Медиана |dy| после снятия сдвига на концах самых изогнутых строк опоры, мм.
        wobble_mm: Медиана по осям СКО второй разности ординаты с шагом ~1 мм, мм.
        crossings: Пар скрещённых осей.
        converging: Пар сблизившихся осей.
        half_rows: Рядов-половинок.
        seconds: Среднее время на полосу, с.
    """

    engine: str
    pages: int
    axes: int
    blocks: int
    ink_share: float
    recall: float
    pieces: float
    split_share: float
    bridges: int
    offset_mm: float
    shape_mm: float
    shape_p90_mm: float
    curl_mm: float
    wobble_mm: float
    crossings: int
    converging: int
    half_rows: int
    seconds: float


def read_run(directory: Path) -> list[Page]:
    """Разборы всех полос всеми движками из ``<прогон>/pages/*.json``.

    Args:
        directory: Каталог прогона ``text_blocks analyze``.

    Returns:
        Список :class:`Page`; оси короче двух точек отброшены.
    """
    pages = []
    for path in sorted((directory / "pages").glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        axes = []
        for item in raw.get("axes", []):
            points = np.asarray(item.get("points") or [], dtype=np.float64).reshape(-1, 2)
            if points.shape[0] >= 2:
                axes.append(Axis(points[np.argsort(points[:, 0])], float(item.get("sagitta_mm") or 0.0)))
        pages.append(
            Page(
                key=(raw["name"], int(raw["page"]), raw["variant"]),
                engine=raw["engine"],
                axes=axes,
                pitch=pitch_of(raw.get("blocks", [])),
                dpi=float(raw.get("dpi") or 150.0),
                raw=raw,
            )
        )
    return pages


def _pair_dy(reference: Axis, axis: Axis) -> tuple[np.ndarray, np.ndarray]:
    """Абсциссы общего охвата двух осей и разность ординат ``axis − reference`` в них.

    Returns:
        Пара ``(xs, dy)``; пустые массивы, если охваты не пересекаются.
    """
    left, right = max(reference.x0, axis.x0), min(reference.x1, axis.x1)
    if right - left < SAMPLE_PX * 2:
        return np.zeros(0), np.zeros(0)
    xs = np.arange(left, right, SAMPLE_PX)
    return xs, axis.y_at(xs) - reference.y_at(xs)


def _nearest_offsets(reference: list[Axis], axes: list[Axis], pitch: float) -> list[float]:
    """Знаковые медианные dy ближайших осей движка к осям опоры (для оценки сдвига движка).

    Ближайшая — с наименьшим медианным |dy| на общем охвате, и не дальше шага строк.
    """
    out = []
    for target in reference:
        best = None
        for axis in axes:
            _, dy = _pair_dy(target, axis)
            if dy.size == 0:
                continue
            value = float(np.median(dy))
            if abs(value) < pitch and (best is None or abs(value) < abs(best)):
                best = value
        if best is not None:
            out.append(best)
    return out


def _columns_of(axes: list[Axis], gutters: list[dict]) -> list[int]:
    """Номер колонки каждой оси: сколько межколонников левее её середины на её высоте.

    Args:
        axes: Оси опоры.
        gutters: Межколонники из JSON разбора: ``{"points": [[y, x0, x1], …]}`` — ход межколонника
            по высоте полосы.

    Returns:
        Номер колонки по оси.
    """
    out = []
    for axis in axes:
        middle = axis.points[axis.points.shape[0] // 2]
        count = 0
        for gutter in gutters:
            track = np.asarray(gutter.get("points") or [], dtype=np.float64).reshape(-1, 3)
            # Межколонник, не доходящий до высоты строки, её колонку не определяет.
            if track.shape[0] == 0 or not track[0, 0] <= middle[1] <= track[-1, 0]:
                continue
            centre = np.interp(middle[1], track[:, 0], (track[:, 1] + track[:, 2]) / 2.0)
            count += centre < middle[0]
        out.append(count)
    return out


def _wobble(axis: Axis, dpi: float) -> float:
    """СКО второй разности ординаты оси с шагом ~1 мм, мм: мера пилы и мелкой волны оси."""
    step = WOBBLE_STEP_MM * dpi / 25.4
    xs = np.arange(axis.x0, axis.x1, step)
    if xs.size < 5:
        return float("nan")
    second = np.diff(axis.y_at(xs), n=2)
    return px_to_mm(float(np.sqrt(np.mean(second**2))), dpi)


def _percentile_mm(values: list[float], q: float, dpi: float) -> float:
    """Перцентиль ``q`` списка пикселей в миллиметрах; NaN для пустого списка."""
    return px_to_mm(float(np.percentile(values, q)), dpi) if values else float("nan")


def score_engine(engine: str, pages: list[Page]) -> EngineScore:
    """Меры движка ``engine`` по всем полосам, где есть и он, и опора.

    Args:
        engine: Имя движка.
        pages: Все разборы прогона.

    Returns:
        :class:`EngineScore`.
    """
    by_key = defaultdict(dict)
    for page in pages:
        by_key[page.key][page.engine] = page
    own_pages = [group[engine] for group in by_key.values() if engine in group]
    # Сдвиг движка оценивается по всему набору сразу: у движков по базовой линии он систематический.
    offsets: list[float] = []
    for group in by_key.values():
        if engine in group and REFERENCE in group:
            ref = group[REFERENCE]
            offsets.extend(_nearest_offsets(ref.axes, group[engine].axes, ref.pitch))
    offset = float(np.median(offsets)) if offsets else 0.0

    covered_len = total_len = 0.0
    pieces, shape, shape_max, curl, bridges = [], [], [], [], 0
    dpi = own_pages[0].dpi if own_pages else 150.0
    for group in by_key.values():
        if engine not in group or REFERENCE not in group:
            continue
        ref, run = group[REFERENCE], group[engine]
        tolerance = MATCH_PITCH_SHARE * ref.pitch if ref.pitch > 0 else 8.0
        columns = _columns_of(ref.axes, ref.raw.get("gutters", []))
        # Порог изогнутости опоры: верхние CURLED_SHARE строк по |сагитте| на полосе.
        sagittas = np.abs([axis.sagitta_mm for axis in ref.axes]) if ref.axes else np.zeros(0)
        curled_from = float(np.quantile(sagittas, 1.0 - CURLED_SHARE)) if sagittas.size else np.inf
        # Какие строки опоры (и каких колонок) накрыла каждая ось движка.
        touched: dict[int, set[int]] = defaultdict(set)
        for index, target in enumerate(ref.axes):
            length = target.x1 - target.x0
            total_len += length
            hit = np.zeros(max(1, int(np.ceil(length / SAMPLE_PX))), dtype=bool)
            count = 0
            for number, axis in enumerate(run.axes):
                xs, dy = _pair_dy(target, axis)
                if dy.size == 0:
                    continue
                close = np.abs(dy - offset) < tolerance
                if close.sum() * SAMPLE_PX < PIECE_SHARE * length:
                    continue
                count += 1
                touched[number].add(columns[index])
                slots = np.clip(((xs[close] - target.x0) / SAMPLE_PX).astype(int), 0, hit.size - 1)
                hit[slots] = True
                # Форма: разность после снятия сдвига ИМЕННО этой пары (по медиане на общем охвате).
                residual = dy[close] - np.median(dy[close])
                shape.append(float(np.median(np.abs(residual))))
                shape_max.append(float(np.abs(residual).max()))
                if abs(target.sagitta_mm) >= curled_from and length > 0:
                    ends = (xs[close] < target.x0 + END_SHARE * length) | (xs[close] > target.x1 - END_SHARE * length)
                    if ends.any():
                        curl.append(float(np.median(np.abs(residual[ends]))))
            covered_len += hit.mean() * length
            if count:
                pieces.append(count)
        bridges += sum(1 for cols in touched.values() if len(cols) > 1)

    metrics = [metrics_of(page.raw) for page in own_pages]
    wobbles = [_wobble(axis, page.dpi) for page in own_pages for axis in page.axes]
    return EngineScore(
        engine=engine,
        pages=len(own_pages),
        axes=sum(len(page.axes) for page in own_pages),
        blocks=sum(len(page.raw.get("blocks", [])) for page in own_pages),
        ink_share=float(np.mean([page.raw.get("ink_share") or 0.0 for page in own_pages])) if own_pages else 0.0,
        recall=covered_len / total_len if total_len else float("nan"),
        pieces=float(np.mean(pieces)) if pieces else float("nan"),
        split_share=float(np.mean(np.asarray(pieces) > 1)) if pieces else float("nan"),
        bridges=bridges,
        offset_mm=px_to_mm(offset, dpi),
        shape_mm=_percentile_mm(shape, 50, dpi),
        shape_p90_mm=_percentile_mm(shape_max, 90, dpi),
        curl_mm=_percentile_mm(curl, 50, dpi),
        wobble_mm=float(np.nanmedian(wobbles)) if wobbles else float("nan"),
        crossings=sum(item.crossings for item in metrics),
        converging=sum(item.converging for item in metrics),
        half_rows=sum(item.half_rows for item in metrics),
        seconds=float(np.mean([page.raw.get("seconds") or 0.0 for page in own_pages])) if own_pages else 0.0,
    )


def markdown_table(scores: list[EngineScore]) -> str:
    """Таблица мер всех движков в markdown, строки — в порядке ``scores``."""
    head = (
        "| движок | полос | осей | блоков | краска | полнота | кусков/строку | разорвано | мостов | "
        "сдвиг, мм | форма, мм | форма p90, мм | завиток, мм | вихляние, мм | скрещ. | сближ. | половинок | с/полосу |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = [
        f"| {s.engine} | {s.pages} | {s.axes} | {s.blocks} | {s.ink_share:.1%} | {s.recall:.1%} | {s.pieces:.2f} | "
        f"{s.split_share:.1%} | {s.bridges} | {s.offset_mm:+.2f} | {s.shape_mm:.2f} | {s.shape_p90_mm:.2f} | "
        f"{s.curl_mm:.2f} | {s.wobble_mm:.3f} | {s.crossings} | {s.converging} | {s.half_rows} | {s.seconds:.1f} |"
        for s in scores
    ]
    return head + "\n".join(rows) + "\n"


__all__ = ["EngineScore", "Page", "markdown_table", "read_run", "score_engine"]
