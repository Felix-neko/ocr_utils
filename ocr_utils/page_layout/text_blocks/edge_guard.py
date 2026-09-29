"""Защита выровненных сторон текстовых блоков от сора: выступы («пупырышки», «ступеньки»), фильтр глифов голосованием CRAFT + pero с повторным разбором, пометка оставшихся выступов недостоверными.

Сор и пометки у края колонки затягиваются в строки и выпячивают выровненную сторону блока. На сторонах, по которым
блок выровнен (both / left / right), ищутся местные выступы огибающей (:func:`page_bumps`). Если даны карты
детекторов символов (:mod:`glyph_maps`), краска за аномальной стороной, которую хотя бы один детектор считает сором,
закрашивается, и полоса разбирается заново (:func:`guarded`). Выступы, оставшиеся по итогу, ВСЕГДА помечаются
недостоверными участками стороны (``SmoothEnvelope.unreliable_left/right``): их не берут в меры наклона и кривизны.

Пороги подобраны на стенде ``research/edge_marks`` (бинаризованные PDF FineReader nogeo пака-1, 188 размеченных
кандидатов, 55 полос с сором, 30 чистых, 200 случайных) — ``reports/edge_guard.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from enum import Enum

import cv2
import numpy as np

from ocr_utils.page_layout import px_to_mm
from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.page_layout.text_blocks.alignment import AlignKind, Alignment
from ocr_utils.page_layout.text_blocks.blocks import TextBlock
from ocr_utils.page_layout.text_blocks.smooth_envelope import SmoothEnvelope

# --- Выступы выровненных сторон -------------------------------------------------------------------------------------

# Выступ стороны наружу не меньше этого — аномалия, мм. При 0.6 ловится 71 из 75 кусков сора с выступом, аномальны
# 10 % случайных текстовых полос пака, зря помечено 12 из 82 законных выступов (висячие дефисы, широкие буквы).
BUMP_MM = 0.6
# Блоки короче стольких строк не смотрятся: подписи, выходные данные, заголовки, оглавления дают выступы по 2–70 мм
# (8 ложных аномальных полос из 200), а сора с выступом на них нет.
MIN_ROWS = 6
# Окно строки: ± столько шагов строк вокруг её середины — там ищется наибольший выступ.
WINDOW_PITCHES = 0.5
# Кольцо сравнения: строки на расстоянии от ... до ... шагов — по ним медиана «нормальной» стороны.
RING_PITCHES = (1.5, 4.0)
# Выровненные виды выключки и их выровненные стороны.
ALIGNED_SIDES = {AlignKind.BOTH: ("left", "right"), AlignKind.LEFT: ("left",), AlignKind.RIGHT: ("right",)}


class BumpKind(str, Enum):
    """Вид выступа стороны."""

    BUMP = "bump"  # пупырышек: выступает одна строка
    STEP = "step"  # ступенька: выступ держится две строки и больше подряд


@dataclass(frozen=True)
class SideBump:
    """Выступ выровненной стороны блока наружу.

    Attributes:
        block: Номер блока в ``PageAnalysis.blocks``.
        side: ``left`` или ``right``.
        y0, y1: Участок стороны по высоте, пиксели рабочей копии (полшага строк вокруг крайних строк выступа).
        mm: Наибольший местный выступ на участке, мм.
        rows: Сколько строк подряд выступает.
        kind: Пупырышек или ступенька.
    """

    block: int
    side: str
    y0: float
    y1: float
    mm: float
    rows: int
    kind: BumpKind

    def to_json(self) -> dict:
        """Словарь для JSON разбора."""
        return {"block": self.block, "side": self.side, "y0": round(self.y0, 1), "y1": round(self.y1, 1),
                "mm": round(self.mm, 2), "rows": self.rows, "kind": self.kind.value}  # fmt: skip


def robust_line(points: np.ndarray) -> tuple[float, float]:
    """Прямая ``x = a·y + b`` по точкам стороны с отсевом выбросов (три прохода, > 2.5 MAD).

    Наклон колонки снимается прямой, а выступы в неё почти не влияют: отсев их выбрасывает.

    Args:
        points: Точки стороны ``(N, 2)`` — ``x, y``.

    Returns:
        ``(a, b)``.
    """
    ys, xs = points[:, 1], points[:, 0]
    keep = np.ones(len(ys), dtype=bool)
    a, b = 0.0, float(np.median(xs))
    for _ in range(3):
        if keep.sum() < 2:
            break
        a, b = np.polyfit(ys[keep], xs[keep], 1)
        residual = xs - (a * ys + b)
        mad = float(np.median(np.abs(residual[keep]))) + 0.5
        keep = np.abs(residual) <= 2.5 * mad
    return float(a), float(b)


def row_excess(block: TextBlock, side: str) -> np.ndarray:
    """Местный выступ огибающей стороны наружу у каждой строки блока, мм.

    Отклонение огибающей от устойчивой прямой стороны (наружу — плюс): наибольшее в окне ±``WINDOW_PITCHES``
    шага вокруг строки минус медиана в кольце ``RING_PITCHES`` шагов. Кольцо снимает плавный изгиб стороны и её
    волны, остаётся местный выступ.

    Args:
        block: Блок.
        side: ``left`` или ``right``.

    Returns:
        Массив по строкам блока (порядок ``block.rows``); где сторона пуста — нули.
    """
    curve = np.asarray(getattr(block.envelope, side), dtype=np.float64)
    if len(curve) < 3 or not block.rows:
        return np.zeros(len(block.rows))
    a, b = robust_line(curve)
    # Наружу: у левой стороны — влево (меньше x), у правой — вправо.
    sign = -1.0 if side == "left" else 1.0
    deviation = sign * (curve[:, 0] - (a * curve[:, 1] + b))
    pitch = block.pitch_px if block.pitch_px > 0 else float(np.median([row.height for row in block.rows]))
    out = np.zeros(len(block.rows))
    for index, row in enumerate(block.rows):
        distance = np.abs(curve[:, 1] - row.y)
        window = deviation[distance <= WINDOW_PITCHES * pitch]
        ring = deviation[(distance >= RING_PITCHES[0] * pitch) & (distance <= RING_PITCHES[1] * pitch)]
        if len(window) == 0:
            continue
        base = float(np.median(ring)) if len(ring) else 0.0
        out[index] = px_to_mm(float(window.max()) - base, block.dpi)
    return out


def side_bumps(block: TextBlock, alignment: Alignment, number: int, bump_mm: float = BUMP_MM) -> list[SideBump]:
    """Выступы выровненных сторон блока.

    Смотрятся только стороны, по которым блок выровнен (``both`` — обе, ``left``/``right`` — одна) и которые сами
    признаны выровненными, в блоке не меньше ``MIN_ROWS`` строк. Строки с выступом не меньше ``bump_mm``
    сливаются в участки подряд идущих строк: одна строка — пупырышек, больше — ступенька.

    Args:
        block: Блок.
        alignment: Его выключка.
        number: Номер блока на странице.
        bump_mm: Порог выступа, мм.

    Returns:
        Выступы сверху вниз, по сторонам.
    """
    out: list[SideBump] = []
    if len(block.rows) < MIN_ROWS:
        return out
    pitch = block.pitch_px if block.pitch_px > 0 else 0.0
    for side in ALIGNED_SIDES.get(alignment.kind, ()):
        if not getattr(alignment, side).aligned:
            continue
        flagged = row_excess(block, side)
        excess, flagged = flagged, flagged >= bump_mm
        index = 0
        while index < len(flagged):
            if not flagged[index]:
                index += 1
                continue
            # Участок подряд идущих строк с выступом.
            end = index
            while end + 1 < len(flagged) and flagged[end + 1]:
                end += 1
            rows = end - index + 1
            y0 = block.rows[index].y - 0.5 * max(pitch, block.rows[index].height)
            y1 = block.rows[end].y + 0.5 * max(pitch, block.rows[end].height)
            out.append(SideBump(number, side, float(y0), float(y1), float(excess[index : end + 1].max()), rows,
                                BumpKind.BUMP if rows == 1 else BumpKind.STEP))  # fmt: skip
            index = end + 1
    return out


def page_bumps(
    blocks: tuple[TextBlock, ...], alignments: tuple[Alignment, ...], bump_mm: float = BUMP_MM
) -> list[SideBump]:
    """Все выступы выровненных сторон блоков страницы; страница аномальна, если список не пуст.

    Args:
        blocks: Блоки страницы.
        alignments: Их выключка (в том же порядке).
        bump_mm: Порог выступа, мм.

    Returns:
        Выступы по блокам.
    """
    out: list[SideBump] = []
    for number, (block, alignment) in enumerate(zip(blocks, alignments)):
        out += side_bumps(block, alignment, number, bump_mm)
    return out


def mark_unreliable(blocks: tuple[TextBlock, ...], bumps: list[SideBump]) -> tuple[TextBlock, ...]:
    """Добавить участки выступов к недостоверным участкам сторон огибающих блоков.

    Уже найденные недостоверные участки (ступеньки ``smooth_sides``) сохраняются. У огибающей без гладких сторон
    (ступенчатый запасной ход) полей недостоверности нет — она переводится в :class:`SmoothEnvelope` с теми же
    кривыми.

    Args:
        blocks: Блоки страницы.
        bumps: Выступы (:func:`page_bumps`).

    Returns:
        Блоки с дополненными ``unreliable_left/right``.
    """
    if not bumps:
        return blocks
    out = list(blocks)
    for number in sorted({bump.block for bump in bumps}):
        envelope = out[number].envelope
        if not isinstance(envelope, SmoothEnvelope):
            envelope = SmoothEnvelope(**{item.name: getattr(envelope, item.name) for item in fields(envelope)})
        spans = {"left": list(envelope.unreliable_left), "right": list(envelope.unreliable_right)}
        for bump in bumps:
            if bump.block == number:
                spans[bump.side].append((bump.y0, bump.y1))
        envelope = replace(envelope, unreliable_left=tuple(sorted(spans["left"])),
                           unreliable_right=tuple(sorted(spans["right"])))  # fmt: skip
        out[number] = replace(out[number], envelope=envelope)
    return tuple(out)


# --- Голосование детекторов символов --------------------------------------------------------------------------------


class GlyphEngine(str, Enum):
    """Детектор символов, чья карта полосы голосует «символ или сор» (имя — папка карт ``glyph_maps``)."""

    CRAFT = "craft"  # карта «регион» CRAFT: центры символов
    PERO = "pero"  # вероятность базовой линии pero ParseNet


# Пороги голосования «хотя бы один»: оценка детектора ниже порога — сор. На 188 размеченных кандидатах голосование
# CRAFT + pero отсекает 54 из 75 кусков сора с выступом при 1 знаке из 80 (CRAFT в одиночку — 43 при 0); совместная
# мера (сумма, произведение, регрессия) при том же вреде хуже — 45–50. pero — нижний край плато 0.40–0.49: при 0.45
# выброшена «7» номера в оглавлении (1972/03 IMG_0105_1L, pero 0.42 при CRAFT 0.72).
THRESHOLDS = {GlyphEngine.CRAFT: 0.35, GlyphEngine.PERO: 0.40}
# Окно базовой линии pero: база ниже середины строки на столько x-высот, окно по высоте — ± столько x-высот.
BASELINE_BELOW_XH, BASELINE_WINDOW_XH = 0.5, 0.6
# pero голосует, только если середина компоненты не дальше стольких x-высот от середины её строки: вне полосы строки
# окно базовой линии пусто и pero зовёт сором всё подряд (выбросил «7» номера страницы под последней строкой).
PERO_ROW_BAND_XH = 1.0


def component_scores(labels: np.ndarray, stats: np.ndarray, index: int, maps: dict[GlyphEngine, np.ndarray],
                     row: tuple[float, float]) -> dict[GlyphEngine, float]:  # fmt: skip
    """Оценки «это символ» одной компоненты каждым детектором.

    CRAFT — максимум карты по пикселям компоненты. pero — максимум вероятности базовой линии в столбцах компоненты
    в окне у базовой линии её строки: знак в конце строки продолжает базовую линию, сор — нет. Вне полосы строки
    (``PERO_ROW_BAND_XH``) pero воздерживается — его оценки в ответе нет.

    Args:
        labels: Метки компонент полосы (``connectedComponentsWithStats``).
        stats: Их статистика.
        index: Номер компоненты.
        maps: Карты детекторов, 0…1, в пикселях рендера.
        row: Середина строки и x-высота, пиксели рендера (:func:`row_geometry`).

    Returns:
        ``детектор → оценка``.
    """
    x, y, w, h = (int(v) for v in stats[index, :4])
    pixels = labels[y : y + h, x : x + w] == index
    out = {}
    for engine, heat in maps.items():
        if engine is GlyphEngine.PERO:
            row_y, x_h = row
            if abs(y + h / 2.0 - row_y) > PERO_ROW_BAND_XH * x_h:
                continue  # вне полосы строки pero воздерживается
            base = row_y + BASELINE_BELOW_XH * x_h
            window = heat[max(0, int(base - BASELINE_WINDOW_XH * x_h)) : int(base + BASELINE_WINDOW_XH * x_h) + 1,
                          x : x + w + 1]  # fmt: skip
            out[engine] = float(window.max()) if window.size else 0.0
        else:
            out[engine] = float(heat[y : y + h, x : x + w][pixels].max())
    return out


def is_junk(scores: dict[GlyphEngine, float]) -> bool:
    """Голосование «хотя бы один»: сор, если хотя бы один детектор дал оценку ниже своего порога ``THRESHOLDS``."""
    return any(value < THRESHOLDS[engine] for engine, value in scores.items())


def row_geometry(blocks: tuple[TextBlock, ...], numbers: set[int], dpi: float, y300: float) -> tuple[float, float]:
    """Середина ближайшей по высоте строки аномальных блоков и её x-высота, пиксели рендера.

    Args:
        blocks: Блоки страницы (пиксели рабочей копии).
        numbers: Номера блоков с выступами.
        dpi: Разрешение рабочей копии.
        y300: Ордината компоненты, пиксели рендера.

    Returns:
        ``(середина строки, x-высота)``.
    """
    k = RENDER_DPI / dpi
    rows = [row for number in numbers for row in blocks[number].rows]
    row = min(rows, key=lambda r: abs(r.y * k - y300))
    x_h = row.glyph_h if row.glyph_h > 0 else 0.5 * row.height
    return row.y * k, x_h * k


# --- Зона и фильтр ---------------------------------------------------------------------------------------------------

# Зона фильтра у аномальной стороны: от ``ZONE_FROM_MM`` мм НАРУЖУ от устойчивой прямой стороны ещё на ``ZONE_OUT_MM``
# мм (не дальше края страницы и не внутрь других блоков). Краска внутри колонки выступа не даёт; зона, заходившая
# на 3 мм внутрь, выбросила обломок разорванной бинаризацией «щ» (1975/01 IMG_0026_1L). Зона до края страницы
# цепляла текст вне блоков — тире посреди строк (1975 IMG_0146_1L).
ZONE_FROM_MM = 0.3
ZONE_OUT_MM = 12.0
# Запас по высоте зоны сверху и снизу блока, шагов строк.
ZONE_PAD_PITCHES = 1.0
# Компоненты длиннее этого (мм) не трогаются: линейки, рамки.
MAX_DROP_MM = 20.0
# Горизонтальный штрих (ширина не меньше стольких высот) не трогается: дефис, тире, линейка — детекторы их почти не
# видят (выброшены висячий дефис «Глав-» и тире посреди строки).
BAR_ASPECT = 2.0
# Порог краски рендера.
INK_LEVEL = 128


def side_zones(
    blocks: tuple[TextBlock, ...], bumps: list[SideBump], dpi: float, shape300: tuple[int, int]
) -> np.ndarray:
    """Маска зоны фильтра в пикселях рендера: полосы снаружи аномальных выровненных сторон.

    По каждой стороне блока с выступом: по высоте — весь блок с запасом ``ZONE_PAD_PITCHES`` шага; по ширине — от
    ``ZONE_FROM_MM`` до ``ZONE_FROM_MM + ZONE_OUT_MM`` мм наружу от устойчивой прямой стороны. Контуры других блоков
    страницы вычитаются: соседняя колонка не фильтруется.

    Args:
        blocks: Блоки страницы (пиксели рабочей копии).
        bumps: Выступы (:func:`page_bumps`).
        dpi: Разрешение рабочей копии.
        shape300: Размер рендера ``(высота, ширина)``.

    Returns:
        Булева маска ``shape300``.
    """
    height, width = shape300
    k = RENDER_DPI / dpi
    zone = np.zeros(shape300, dtype=bool)
    sides = {(bump.block, bump.side) for bump in bumps}
    start, outside = (value * RENDER_DPI / 25.4 for value in (ZONE_FROM_MM, ZONE_FROM_MM + ZONE_OUT_MM))
    for number, side in sorted(sides):
        block = blocks[number]
        curve = np.asarray(getattr(block.envelope, side), dtype=np.float64)
        a, b = robust_line(curve)
        pad = ZONE_PAD_PITCHES * block.pitch_px
        y0 = max(0, int((curve[:, 1].min() - pad) * k))
        y1 = min(height, int((curve[:, 1].max() + pad) * k) + 1)
        for y in range(y0, y1):
            # Абсцисса прямой стороны на этой высоте (рабочая копия → рендер).
            x = (a * (y / k) + b) * k
            if side == "right":
                zone[y, max(0, int(x + start)) : min(width, int(x + outside) + 1)] = True
            else:
                zone[y, max(0, int(x - outside)) : max(0, min(width, int(x - start) + 1))] = True
    others = np.zeros(shape300, dtype=np.uint8)
    anomalous = {number for number, _ in sides}
    for number, block in enumerate(blocks):
        if number not in anomalous:
            polygon = np.asarray(block.envelope.polygon, dtype=np.float64) * k
            cv2.fillPoly(others, [polygon.astype(np.int32)], 1)
    return zone & (others == 0)


def keep_gray(
    gray300: np.ndarray,
    zone: np.ndarray,
    maps: dict[GlyphEngine, np.ndarray],
    row_of: Callable[[float], tuple[float, float]],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Закрасить в зоне краску, которую голосование детекторов считает сором.

    Компоненты краски (8-связность), задевающие зону, оцениваются детекторами (:func:`component_scores`) и
    закрашиваются белым, если хотя бы один видит в них сор (:func:`is_junk`). Компоненты длиннее ``MAX_DROP_MM`` и
    горизонтальные штрихи (ширина не меньше ``BAR_ASPECT`` высот: дефис, тире, линейка) не трогаются.

    Args:
        gray300: Серая полоса ``RENDER_DPI``.
        zone: Маска зоны (:func:`side_zones`).
        maps: Карты голосующих детекторов того же размера, 0…1.
        row_of: Середина строки и x-высота по ординате компоненты (пиксели рендера) — для окна базовой линии pero.

    Returns:
        ``(полоса с закрашенным сором, рамки выброшенных компонент x0, y0, x1, y1 в пикселях рендера)``.
    """
    ink = (gray300 < INK_LEVEL).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    max_px = MAX_DROP_MM * RENDER_DPI / 25.4
    # Кандидаты: компоненты, задевшие зону, не штрихи и не длинные линейки.
    touched = np.zeros(count, dtype=bool)
    touched[np.unique(labels[zone & (ink > 0)])] = True
    touched[0] = False
    bar = stats[:, 2] >= BAR_ASPECT * stats[:, 3]
    candidates = np.nonzero(touched & ~bar & (np.maximum(stats[:, 2], stats[:, 3]) <= max_px))[0]
    drop = np.zeros(count, dtype=bool)
    for index in candidates:
        centre_y = stats[index, 1] + stats[index, 3] / 2.0
        drop[index] = is_junk(component_scores(labels, stats, int(index), maps, row_of(centre_y)))
    out = gray300.copy()
    out[drop[labels]] = 255
    boxes = [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, _ in stats[np.nonzero(drop)[0]]]
    return out, boxes


# --- Проход целиком -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeGuardReport:
    """Что сделала защита сторон на полосе.

    Attributes:
        bumps_first: Выступы первого разбора (полоса аномальна, если не пусто).
        bumps_final: Выступы по итогу — они помечены недостоверными.
        second_pass: Был ли повторный разбор с фильтром глифов.
        engines: Голосовавшие детекторы (пусто — карт не было).
        dropped: Рамки выброшенной фильтром краски, пиксели рендера.
    """

    bumps_first: tuple[SideBump, ...] = ()
    bumps_final: tuple[SideBump, ...] = ()
    second_pass: bool = False
    engines: tuple[GlyphEngine, ...] = ()
    dropped: tuple[tuple[int, int, int, int], ...] = ()

    @property
    def anomalous(self) -> bool:
        """Есть ли выступы на выровненных сторонах в первом разборе."""
        return bool(self.bumps_first)

    def to_json(self) -> dict:
        """Словарь для JSON разбора."""
        return {
            "anomalous": self.anomalous,
            "second_pass": self.second_pass,
            "engines": [engine.value for engine in self.engines],
            "bumps_first": [bump.to_json() for bump in self.bumps_first],
            "bumps_final": [bump.to_json() for bump in self.bumps_final],
            "dropped": [list(box) for box in self.dropped],
        }


def second_pass_gray(
    gray300: np.ndarray,
    blocks: tuple[TextBlock, ...],
    dpi: float,
    bumps: list[SideBump],
    maps: dict[GlyphEngine, np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Полоса для повторного разбора: сор за аномальными сторонами закрашен голосованием детекторов.

    Args:
        gray300: Исходная полоса ``RENDER_DPI``.
        blocks: Блоки первого разбора.
        dpi: Разрешение рабочей копии.
        bumps: Выступы первого разбора.
        maps: Карты детекторов.

    Returns:
        ``(полоса, рамки выброшенной краски)``.
    """
    zone = side_zones(blocks, bumps, dpi, gray300.shape)
    numbers = {bump.block for bump in bumps}
    return keep_gray(gray300, zone, maps, lambda y: row_geometry(blocks, numbers, dpi, y))


__all__ = [
    "BAR_ASPECT",
    "BUMP_MM",
    "BumpKind",
    "EdgeGuardReport",
    "GlyphEngine",
    "MAX_DROP_MM",
    "MIN_ROWS",
    "SideBump",
    "THRESHOLDS",
    "ZONE_FROM_MM",
    "ZONE_OUT_MM",
    "component_scores",
    "is_junk",
    "keep_gray",
    "mark_unreliable",
    "page_bumps",
    "robust_line",
    "row_excess",
    "row_geometry",
    "second_pass_gray",
    "side_bumps",
    "side_zones",
]
