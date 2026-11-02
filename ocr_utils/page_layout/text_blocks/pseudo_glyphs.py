"""Фильтр псевдоглифов перед разбором текстовых блоков: компоненты краски, которые разбор принял бы за буквы (кромка листа и тень сгиба у корешка, нитки прошивки, полоса обложки, карандаш, сор), — кандидаты у концов рядов и края картинки, правила по признакам в мм, голос CRAFT по местным вырезкам, закрас и повторный разбор (стенд и замеры — ``research/pseudo_glyphs``, ``reports/pseudo_glyphs.md``)."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

from ocr_utils.page_layout.text_blocks.orient import sideways_block
from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.page_layout.text_blocks.alignment import _core_curve
from ocr_utils.page_layout.text_blocks.alignment import Side as AlignSide
from ocr_utils.units import mm_to_px_float, px_to_mm

if TYPE_CHECKING:
    from ocr_utils.page_layout.text_blocks.page import PageAnalysis

log = logging.getLogger(__name__)


class PseudoMode(StrEnum):
    """Режим фильтра псевдоглифов в ``analyse_gray``."""

    OFF = "off"  # без фильтра (пак-1 и все прежние потребители)
    RULES = "rules"  # только правила по признакам
    RULES_CRAFT = "rules_craft"  # правила + CRAFT для спорных серых кандидатов (местные вырезки на CPU)


# --- Кандидаты и признаки ---------------------------------------------------------------------------------------------

# Ряд: полоса по высоте — середина ряда ± столько его высот (компонента должна задеть её хотя бы на долю
# ``BAND_MIN_SHARE`` меньшей из двух высот).
BAND_HALF_HEIGHTS = 0.5
BAND_MIN_SHARE = 0.5
# Поиск крайней компоненты за концом ряда наружу, мм: край ряда тянется по краске (``extend_to_ink``) и за ось.
OUTWARD_MM = 3.0
# Компонента мельче этого по обеим сторонам (мм) — точка, пыль: крайней не считается (``component_mask`` у корпуса
# режет всё ниже 3 px при 150 dpi ≈ 0.5 мм).
MIN_SIDE_MM = 0.5
# Компонента «у края вырезки»: ближе стольких мм к краю картинки.
EDGE_MM = 1.0
# Компонента у края вырезки идёт в кандидаты, если хотя бы одна её сторона не меньше стольких мм.
EDGE_MIN_SIDE_MM = 1.0
# Пробел внутри ряда считается межсловным, если он шире стольких высот строчной (уже — межбуквенный).
WORD_SPACE_XH = 0.3
# Квантиль межсловных пробелов страницы, с которым сравнивается отрыв кандидата.
SPACE_QUANTILE = 99
# Пробел шире стольких медианных межсловных пробелов в робастный максимум не идёт (разгон строк по формату — до ~2.5).
SPACE_CAP_MEDIANS = 3.0
# Край ряда задан компонентой, если её край ближе стольких px рабочей копии к краю ряда.
DEFINES_EDGE_PX = 1.5


class CandidateKind(StrEnum):
    """Откуда кандидат: левый или правый конец ряда блока или край вырезки (вне рядов)."""

    ROW_LEFT = "row_left"
    ROW_RIGHT = "row_right"
    EDGE = "edge"


class EdgeSide(StrEnum):
    """Ближайший край картинки."""

    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass(frozen=True)
class Components:
    """Компоненты краски серого 300 dpi (8-связность).

    Поля:
        labels: Карта меток.
        stats: Статистика ``connectedComponentsWithStats``.
        threshold: Порог краски (Оцу рабочей копии, как в ``segment.component_mask``).
    """

    labels: np.ndarray
    stats: np.ndarray
    threshold: float


def components_of(gray300: np.ndarray) -> Components:
    """Компоненты краски с порогом Оцу (по копии 150 dpi, как маска глифов разбора).

    Args:
        gray300: Серый 300 dpi, как его видит разбор.

    Returns:
        :class:`Components`.
    """
    work = cv2.resize(gray300, (gray300.shape[1] // 2, gray300.shape[0] // 2), interpolation=cv2.INTER_AREA)
    threshold, _ = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary = (gray300 < threshold).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    return Components(labels, stats, float(threshold))


def stroke_mm(mask: np.ndarray) -> float:
    """Средняя толщина штриха компоненты: площадь / длина скелета, мм (300 dpi).

    Args:
        mask: Булева маска компоненты в её рамке.

    Returns:
        Толщина, мм.
    """
    skeleton = int(skeletonize(mask).sum())
    return px_to_mm(float(mask.sum()) / max(skeleton, 1), RENDER_DPI)


def _upright_blocks(analysis: "PageAnalysis") -> list:
    """Блоки страницы без боковых (:func:`orient.sideways_block`): ряды фильтра — горизонтальные полосы страницы."""
    return [block for block in analysis.blocks if not sideways_block(block)]


def word_spaces_mm(analysis: "PageAnalysis", comps: Components) -> tuple[float, float]:
    """Медиана и робастный максимум межсловных пробелов страницы в рядах блоков, мм.

    Пробелы — промежутки между соседними по x компонентами полосы ряда, шире ``WORD_SPACE_XH`` строчной. Робастный
    максимум — квантиль ``SPACE_QUANTILE`` пробелов не шире ``SPACE_CAP_MEDIANS`` медиан: без отсечки в ряды попадают
    промежутки «псевдоглиф — текст» и пустоты заголовков, и p99 страницы доходил до 28 мм.

    Args:
        analysis: Разбор страницы.
        comps: Компоненты краски.

    Returns:
        ``(медиана, робастный максимум)``, мм; ``nan`` — рядов нет.
    """
    k = RENDER_DPI / analysis.dpi
    gaps = []
    for block in _upright_blocks(analysis):
        for row in block.rows:
            members = row_members(row, comps, k)
            if len(members) < 2:
                continue
            x_h = (row.glyph_h if row.glyph_h > 0 else 0.5 * row.height) * k
            spans = np.array([(comps.stats[i, 0], comps.stats[i, 0] + comps.stats[i, 2]) for i in members], float)
            spans = spans[np.argsort(spans[:, 0])]
            # Промежуток — от правого края уже пройденной краски до левого края следующей компоненты.
            reach = np.maximum.accumulate(spans[:, 1])
            between = spans[1:, 0] - reach[:-1]
            gaps.extend(between[between > WORD_SPACE_XH * x_h].tolist())
    if not gaps:
        return float("nan"), float("nan")
    values = np.asarray(gaps, np.float64)
    median = float(np.median(values))
    top = float(np.percentile(values[values <= SPACE_CAP_MEDIANS * median], SPACE_QUANTILE))
    return px_to_mm(median, RENDER_DPI), px_to_mm(top, RENDER_DPI)


def row_members(row, comps: Components, k: float, outward_px: float = 0.0) -> list[int]:
    """Компоненты, лежащие в полосе ряда по высоте и в его пределах по x (с запасом наружу).

    Args:
        row: Ряд блока (``blocks.Row``, пиксели рабочей копии).
        comps: Компоненты 300 dpi.
        k: Пиксели 300 dpi на пиксель рабочей копии.
        outward_px: Запас за концами ряда, пиксели 300 dpi.

    Returns:
        Номера компонент (без фона и пыли мельче ``MIN_SIDE_MM``).
    """
    stats = comps.stats
    half = BAND_HALF_HEIGHTS * row.height * k
    b0, b1 = row.y * k - half, row.y * k + half
    x0, x1 = row.x0 * k - outward_px, row.x1 * k + outward_px
    min_side = mm_to_px_float(MIN_SIDE_MM, RENDER_DPI)
    xs, ys, ws, hs = stats[1:, 0], stats[1:, 1], stats[1:, 2], stats[1:, 3]
    overlap = np.minimum(ys + hs, b1) - np.maximum(ys, b0)
    ok = (
        (overlap >= BAND_MIN_SHARE * np.minimum(hs, b1 - b0))
        & (xs + ws > x0)
        & (xs < x1)
        & ((ws >= min_side) | (hs >= min_side))
    )
    return (np.nonzero(ok)[0] + 1).tolist()


def _block_glyph_p95(block, k: float) -> float:
    """95-й процентиль высот глифов блока (самая высокая нормальная буква), пиксели 300 dpi; ``nan`` — глифов нет."""
    heights = [g[3] - g[1] for row in block.rows for axis in row.axes if axis.glyphs is not None for g in axis.glyphs]
    return float(np.percentile(heights, 95)) * k if heights else float("nan")


def _side_excess_mm(block, alignment, side: AlignSide, row_index: int) -> tuple[float, bool]:
    """Насколько край ряда выходит наружу за тренд стороны блока (мм, наружу — плюс) и выровнена ли сторона."""
    stats = alignment.left if side is AlignSide.LEFT else alignment.right
    curve = _core_curve(block.envelope, side)
    if curve is None or len(curve) < 2:
        return float("nan"), bool(stats.aligned)
    row = block.rows[row_index]
    fitted = float(np.interp(row.y, curve[:, 1], curve[:, 0]))
    excess = (fitted - row.x0) if side is AlignSide.LEFT else (row.x1 - fitted)
    return px_to_mm(excess, block.dpi), bool(stats.aligned)


def _component_features(
    index: int, comps: Components, raw300: np.ndarray, ink_gray: float, paper_gray: float
) -> dict[str, float]:
    """Признаки одной компоненты, не зависящие от ряда: размеры, заполнение, штрих, край картинки, тон."""
    x, y, w, h, area = (int(v) for v in comps.stats[index])
    height, width = comps.labels.shape
    mask = comps.labels[y : y + h, x : x + w] == index
    distances = {EdgeSide.LEFT: x, EdgeSide.RIGHT: width - (x + w), EdgeSide.TOP: y, EdgeSide.BOTTOM: height - (y + h)}
    edge_side = min(distances, key=distances.get)
    tone = float(np.median(raw300[y : y + h, x : x + w][mask]))
    return {
        "x0": x,
        "y0": y,
        "x1": x + w,
        "y1": y + h,
        "h_mm": px_to_mm(h, RENDER_DPI),
        "w_mm": px_to_mm(w, RENDER_DPI),
        "area_mm2": area * (25.4 / RENDER_DPI) ** 2,
        "fill": area / max(1, w * h),
        "aspect": h / max(w, 1),
        "stroke_mm": stroke_mm(mask),
        "edge_mm": px_to_mm(distances[edge_side], RENDER_DPI),
        "edge_side": edge_side.value,
        "tone": tone,
        # Тон относительно краски текста страницы: 0 — как медианная краска, 1 — как бумага.
        "tone_rel": (tone - ink_gray) / max(paper_gray - ink_gray, 1.0),
    }


def _rows_spanned(index: int, comps: Components, analysis: "PageAnalysis", k: float) -> int:
    """Сколько рядов блоков страницы компонента задевает по высоте (в своих пределах по x)."""
    x, y, w, h = (int(v) for v in comps.stats[index, :4])
    count = 0
    for block in _upright_blocks(analysis):
        for row in block.rows:
            half = BAND_HALF_HEIGHTS * row.height * k
            if y < row.y * k + half and y + h > row.y * k - half and x < row.x1 * k + 1 and x + w > row.x0 * k - 1:
                count += 1
    return count


def page_candidates(
    analysis: "PageAnalysis", gray300: np.ndarray, raw300: np.ndarray, spine_side: EdgeSide | None
) -> list[dict]:
    """Кандидаты страницы с признаками.

    Кандидат ряда — крайняя (слева или справа) компонента полосы ряда, включая краску до ``OUTWARD_MM`` за концом
    ряда. Кандидат края — любая компонента ближе ``EDGE_MM`` к краю картинки со стороной от ``EDGE_MIN_SIDE_MM``,
    не попавшая в кандидаты рядов.

    Args:
        analysis: Разбор страницы (рабочая копия ``analysis.dpi``).
        gray300: Серый 300 dpi, как его видит разбор.
        raw300: Серый 300 dpi без утолщения — для тона.
        spine_side: Сторона корешка на картинке (у правой страницы — слева) или ``None``.

    Returns:
        Словари признаков; координаты рамок — пиксели 300 dpi.
    """
    comps = components_of(gray300)
    stats = comps.stats
    k = RENDER_DPI / analysis.dpi
    outward = mm_to_px_float(OUTWARD_MM, RENDER_DPI)
    space_median, space = word_spaces_mm(analysis, comps)
    # Тон краски текста и бумаги страницы: медиана серого по пикселям краски в рядах и медиана вне краски.
    binary = comps.labels > 0
    row_mask = np.zeros(gray300.shape, bool)
    for block in _upright_blocks(analysis):
        for row in block.rows:
            half = BAND_HALF_HEIGHTS * row.height * k
            row_mask[int(max(0, row.y * k - half)) : int(row.y * k + half) + 1,
                     int(max(0, row.x0 * k)) : int(row.x1 * k) + 1] = True  # fmt: skip
    ink_pixels = raw300[binary & row_mask]
    ink_gray = float(np.median(ink_pixels)) if ink_pixels.size else 60.0
    paper_gray = float(np.median(raw300[~binary]))
    seen: set[int] = set()
    out: list[dict] = []
    for block_index, (block, alignment) in enumerate(zip(analysis.blocks, analysis.alignments)):
        # Блоки боковых областей (вертикальные строки): концы их рядов — не левый и правый край, кандидатов нет.
        if sideways_block(block):
            continue
        p95 = _block_glyph_p95(block, k)
        for row_index, row in enumerate(block.rows):
            members = row_members(row, comps, k, outward)
            if not members:
                continue
            x_h = (row.glyph_h if row.glyph_h > 0 else 0.5 * row.height) * k
            # Верх и низ полосы глифов ряда (без выносных): середина ± пол-строчной… берём середину ряда ± высота/2.
            glyph_top, glyph_bottom = row.y * k - 0.5 * row.height * k, row.y * k + 0.5 * row.height * k
            for side, kind in ((AlignSide.LEFT, CandidateKind.ROW_LEFT), (AlignSide.RIGHT, CandidateKind.ROW_RIGHT)):
                if side is AlignSide.LEFT:
                    order = sorted(members, key=lambda i: stats[i, 0])
                else:
                    order = sorted(members, key=lambda i: -(stats[i, 0] + stats[i, 2]))
                index = order[0]
                # Отрыв от соседней краски ряда внутрь: от края кандидата до ближайшей другой компоненты.
                if side is AlignSide.LEFT:
                    inner = stats[index, 0] + stats[index, 2]
                    rest = [stats[i, 0] for i in order[1:] if stats[i, 0] + stats[i, 2] > stats[index, 0]]
                    gap = (min(rest) - inner) if rest else np.nan
                    end = row.x0 * k
                    edge_hit = abs(stats[index, 0] - end) <= DEFINES_EDGE_PX * k
                    beyond = end - stats[index, 0]
                else:
                    inner = stats[index, 0]
                    rest = [
                        stats[i, 0] + stats[i, 2] for i in order[1:] if stats[i, 0] < stats[index, 0] + stats[index, 2]
                    ]
                    gap = (inner - max(rest)) if rest else np.nan
                    end = row.x1 * k
                    edge_hit = abs(stats[index, 0] + stats[index, 2] - end) <= DEFINES_EDGE_PX * k
                    beyond = stats[index, 0] + stats[index, 2] - end
                excess, aligned = _side_excess_mm(block, alignment, side, row_index)
                h = stats[index, 3]
                feats = _component_features(index, comps, raw300, ink_gray, paper_gray)
                gap_mm = px_to_mm(max(float(gap), 0.0), RENDER_DPI) if np.isfinite(gap) else float("nan")
                feats.update(
                    kind=kind.value,
                    comp=int(index),
                    block=block_index,
                    row=row_index,
                    block_rows=len(block.rows),
                    aligned=aligned,
                    excess_mm=excess,
                    defines_edge=bool(edge_hit),
                    beyond_mm=px_to_mm(float(beyond), RENDER_DPI),
                    x_h_mm=px_to_mm(x_h, RENDER_DPI),
                    row_h_mm=px_to_mm(row.height * k, RENDER_DPI),
                    h_rel_row=h / max(row.height * k, 1.0),
                    h_rel_block=h / p95 if np.isfinite(p95) and p95 > 0 else float("nan"),
                    above_xh=(glyph_top - stats[index, 1]) / max(x_h, 1.0),
                    below_xh=(stats[index, 1] + h - glyph_bottom) / max(x_h, 1.0),
                    gap_mm=gap_mm,
                    gap_rel=gap_mm / space if np.isfinite(space) and space > 0 else float("nan"),
                    n_members=len(members),
                )
                out.append(feats)
                seen.add(int(index))
    # Кандидаты края вырезки, не задевшие ни один ряд как крайние.
    height, width = comps.labels.shape
    edge = mm_to_px_float(EDGE_MM, RENDER_DPI)
    min_side = mm_to_px_float(EDGE_MIN_SIDE_MM, RENDER_DPI)
    xs, ys, ws, hs = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
    near = (xs <= edge) | (ys <= edge) | (xs + ws >= width - edge) | (ys + hs >= height - edge)
    big = (ws >= min_side) | (hs >= min_side)
    for index in np.nonzero(near & big)[0]:
        if index == 0 or int(index) in seen:
            continue
        feats = _component_features(int(index), comps, raw300, ink_gray, paper_gray)
        feats.update(kind=CandidateKind.EDGE.value, comp=int(index), block=-1, row=-1, block_rows=0, aligned=False,
                     excess_mm=float("nan"), defines_edge=False, beyond_mm=float("nan"), x_h_mm=float("nan"),
                     row_h_mm=float("nan"), h_rel_row=float("nan"), h_rel_block=float("nan"), above_xh=float("nan"),
                     below_xh=float("nan"), gap_mm=float("nan"), gap_rel=float("nan"), n_members=0)  # fmt: skip
        out.append(feats)
    for feats in out:
        index = feats["comp"]
        feats["rows_spanned"] = _rows_spanned(index, comps, analysis, k)
        feats["spine_side"] = spine_side is not None and feats["edge_side"] == spine_side.value
        feats["word_space_mm"] = space
        feats["word_space_median_mm"] = space_median
        feats["ink_gray"] = ink_gray
    return out


# --- Правила -----------------------------------------------------------------------------------------------------------


class PseudoReason(StrEnum):
    """Причина, по которой компонента признана псевдоглифом (пустая строка — не псевдоглиф)."""

    EDGE_LINE = "edge_line"  # у края вырезки, тонкая высокая черта (кромка листа, линия сгиба, нитка)
    NEAR_LINE = "near_line"  # в нескольких мм от края, тонкая серая черта (сгиб за белой полоской)
    EDGE_GREY = "edge_grey"  # у края, серее краски (тень сгиба, полоса обложки, крючок нитки)
    EDGE_TALL = "edge_tall"  # у края, выше самой высокой буквы блока
    FAR_GREY = "far_grey"  # далеко от ряда (отрыв больше максимума пробела), серая и не мелочь — карандаш
    FAR_THIN = "far_thin"  # далеко от ряда, тонкий вытянутый штрих — карандаш
    CRAFT = "craft"  # спорная (у края или далеко), серая, CRAFT не видит в ней символа


@dataclass(frozen=True)
class PseudoParams:
    """Пороги правил (замеры — ``reports/pseudo_glyphs.md``, разметка ``sets/labels.csv``).

    Поля:
        edge_mm: Касание края вырезки: компонента ближе стольких мм к краю картинки.
        near_mm: Близость к краю для тонкой серой черты (сгиб за белой полоской у края), мм.
        line_width_mm: Черта не шире стольких мм.
        line_aspect: Черта у края: высота не меньше стольких ширин...
        line_tall: ...и выше стольких высот самой высокой буквы (иначе это ствол буквы, срезанной краем).
        thin_stroke_mm: Тонкий штрих: средняя толщина не больше стольких мм (штрих букв корпуса — от 0.3 мм).
        thin_aspect: Тонкая черта вытянута не меньше чем во столько раз.
        line_tone: Тонкая черта серее этого (тон: 0 — краска страницы, 1 — бумага), либо уже ``near_width_mm``.
        near_width_mm: Тонкая черта вдали от края не шире стольких мм.
        grey_tone: Серая компонента: тон не меньше этого.
        tall: Высокая компонента: выше стольких p95 высот букв блока (у компонент вне рядов — страницы).
        far_gap: Далёкая компонента: отрыв от соседней краски ряда больше стольких робастных максимумов пробела.
        far_gap_min_mm: ...и не меньше стольких мм.
        small: Мелочь: ниже стольких p95 высот букв (кратка «й», точка) — правило «далеко и серая» её не трогает.
        craft_doubt: Порог CRAFT для спорных (оценка ниже — псевдоглиф), ``None`` — без CRAFT.
        craft_tone: CRAFT голосует только у компонент серее этого: у срезанных краем букв и тире CRAFT тоже молчит.
        bar_aspect: Брусок (тире, дефис) — ширина не меньше стольких высот: CRAFT по нему не голосует.
        craft_veto: С CRAFT правила «далеко от ряда» отменяются, если CRAFT видит символ увереннее этого.
        hyphen_aspect, hyphen_rel, hyphen_max_mm: Компонента размером с дефис (ширина от ``hyphen_aspect`` высот, высота
            меньше ``hyphen_rel`` высоты букв, ширина меньше ``hyphen_max_mm`` мм) правилом «у края и серая» не снимается —
            её судит только CRAFT (у дефисов у корешка CRAFT 0.5–0.9, у крючков и полос того же размера 0–0.27).
    """

    edge_mm: float = 1.5
    near_mm: float = 4.0
    line_width_mm: float = 1.0
    line_aspect: float = 6.0
    line_tall: float = 1.2
    thin_stroke_mm: float = 0.25
    thin_aspect: float = 3.0
    line_tone: float = 0.25
    near_width_mm: float = 0.35
    grey_tone: float = 0.3
    tall: float = 1.5
    far_gap: float = 1.0
    far_gap_min_mm: float = 2.0
    small: float = 0.4
    craft_doubt: float | None = None
    craft_tone: float = 0.2
    bar_aspect: float = 1.5
    craft_veto: float = 0.6
    hyphen_aspect: float = 0.9
    hyphen_rel: float = 0.45
    hyphen_max_mm: float = 2.0


def relative_height(frame: pd.DataFrame) -> pd.Series:
    """Высота компоненты в p95 высот букв: у кандидатов рядов — блока, у кандидатов края — самого высокого блока страницы.

    Args:
        frame: Кандидаты (с ``key`` — несколько страниц, без — одна).

    Returns:
        Серия долей (``nan``, если на странице нет ни одного блока с буквами).
    """
    glyph = frame.h_mm / frame.h_rel_block
    page = glyph.groupby(frame.key).transform("max") if "key" in frame else pd.Series(glyph.max(), index=frame.index)
    return frame.h_rel_block.fillna(frame.h_mm / page)


def reasons(frame: pd.DataFrame, params: PseudoParams = PseudoParams()) -> pd.Series:
    """Причина для каждого кандидата таблицы (векторно); пустая строка — оставить.

    Порядок проверки задаёт причину, если сработали несколько.

    Args:
        frame: Кандидаты с признаками (``candidates.page_candidates``); для CRAFT — колонка ``craft``.
        params: Пороги.

    Returns:
        Серия строк (значения :class:`PseudoReason` или ``""``).
    """
    p = params
    # Готовая колонка ``h_rel`` (стенд считает её по всем кандидатам страницы) или расчёт по самой таблице.
    h_rel = frame["h_rel"] if "h_rel" in frame else relative_height(frame)
    edge = frame.edge_mm <= p.edge_mm
    near = frame.edge_mm <= p.near_mm
    thin = (frame.stroke_mm <= p.thin_stroke_mm) & (frame.aspect >= p.thin_aspect)
    narrow = frame.w_mm <= p.line_width_mm
    # Черта у края: высокая (выше букв) и вытянутая — или тонкая и серая (или совсем узкая).
    line_edge = (
        edge
        & narrow
        & (
            ((frame.aspect >= p.line_aspect) & (h_rel >= p.line_tall))
            | (thin & ((frame.tone_rel >= p.line_tone) | (frame.w_mm <= 0.3)))
        )
    )
    line_near = near & (frame.w_mm <= p.near_width_mm) & thin & (frame.tone_rel >= p.line_tone)
    grey = frame.tone_rel >= p.grey_tone
    # Дефис размером: не уже своей высоты, ниже ``hyphen_rel`` высоты букв и короче ``hyphen_max_mm``. У корешка он
    # бывает серым (тень сгиба), и «у края и серая» его съедала — 5 из 51 срабатываний на концах рядов (раунд r4).
    hyphen_like = (frame.w_mm >= p.hyphen_aspect * frame.h_mm) & (h_rel < p.hyphen_rel) & (frame.w_mm < p.hyphen_max_mm)
    tall = h_rel >= p.tall
    far = (frame.gap_rel.fillna(0.0) > p.far_gap) & (frame.gap_mm.fillna(0.0) >= p.far_gap_min_mm)
    big = h_rel >= p.small
    out = pd.Series("", index=frame.index, dtype=object)
    for mask, reason in (
        (line_edge, PseudoReason.EDGE_LINE),
        (line_near, PseudoReason.NEAR_LINE),
        (edge & grey & ~hyphen_like, PseudoReason.EDGE_GREY),
        (edge & tall, PseudoReason.EDGE_TALL),
        (far & grey & big, PseudoReason.FAR_GREY),
        (far & thin & narrow, PseudoReason.FAR_THIN),
    ):
        out[(out == "") & mask] = reason.value
    if p.craft_doubt is not None and "craft" in frame:
        # Вето CRAFT на «далеко от ряда»: сноска «5» перед «См.:» — серая и оторванная, но CRAFT 0.62 (карандаш — 0.36).
        far_reasons = out.isin([PseudoReason.FAR_GREY.value, PseudoReason.FAR_THIN.value])
        out[far_reasons & (frame.craft >= p.craft_veto)] = ""
        # Горизонтальный брусок (тире, дефис: ширина от ``bar_aspect`` высот) CRAFT почти не видит — ему не голосовать.
        bar = frame.w_mm >= p.bar_aspect * frame.h_mm
        doubt = (out == "") & (near | far) & ~bar & (frame.tone_rel >= p.craft_tone) & (frame.craft < p.craft_doubt)
        out[doubt] = PseudoReason.CRAFT.value
    return out


def reason_of(features: dict, params: PseudoParams = PseudoParams()) -> str:
    """Причина для одного кандидата (словарь признаков); пустая строка — оставить."""
    return str(reasons(pd.DataFrame([features]), params).iloc[0])


# --- CRAFT по местным вырезкам ----------------------------------------------------------------------------------------

# Корень CRAFT-pytorch (клон ``src/craft``) и весов ``models/craft/craft_mlt_25k.pth`` — тот же, что у воркеров
# ``engines/workers/craft_worker.py`` (SSD hotstore, вне git).
CRAFT_ROOT = Path("/mnt/hotstore/scan_processing/mts_markup/line_axis_engines")
# Поле вырезки вокруг кандидата, мм (≈ 94 px при 300 dpi): на 150 размеченных кандидатах оценка по вырезке совпала с
# картой по всей странице с корреляцией 0.996, решение «< 0.4» — у всех 150 (стенд, 2026-10-26).
CRAFT_CONTEXT_MM = 8.0
# Кандидат длиннее стольких мм CRAFT не судит: спорными бывают только мелкие серые компоненты, длинные черты снимают
# правила.
CRAFT_MAX_SIDE_MM = 20.0


def craft_crop(
    gray300: np.ndarray, labels: np.ndarray, comp: int, box: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Вырезка вокруг кандидата с полем ``CRAFT_CONTEXT_MM`` и маска его пикселей в ней.

    Args:
        gray300: Серый 300 dpi, как его видит разбор.
        labels: Метки компонент того же размера.
        comp: Номер компоненты.
        box: Рамка компоненты ``(x0, y0, x1, y1)``, px.

    Returns:
        ``(вырезка uint8, маска bool того же размера)``.
    """
    ctx = mm_to_px_float(CRAFT_CONTEXT_MM, RENDER_DPI)
    x0, y0, x1, y1 = box
    wx0, wy0 = max(0, int(x0 - ctx)), max(0, int(y0 - ctx))
    wx1, wy1 = min(gray300.shape[1], int(x1 + ctx)), min(gray300.shape[0], int(y1 + ctx))
    return gray300[wy0:wy1, wx0:wx1].copy(), labels[wy0:wy1, wx0:wx1] == comp


class CraftScorer(ABC):
    """Оценщик «это символ» по вырезкам CRAFT: локальный (модель в этом процессе) или удалённый (сервер на GPU)."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """Можно ли оценивать (есть модель или канал до сервера)."""

    @abstractmethod
    def score_crops(self, crops: list[np.ndarray], masks: list[np.ndarray]) -> list[float] | None:
        """Оценки по вырезкам: максимум карты CRAFT «регион» по пикселям маски каждой.

        Args:
            crops: Вырезки uint8 (:func:`craft_crop`).
            masks: Маски кандидатов в них.

        Returns:
            Оценки 0…1 по порядку; ``None`` — оценить не удалось (тогда фильтр работает одними правилами).
        """


class LocalCraft(CraftScorer):
    """CRAFT в этом процессе: модель грузится при первой оценке, одна на объект (CPU — стенд и тесты; CUDA — сервер v2).

    На CPU — 0.29 с на вырезку в один поток; на GPU в fp16 — сотые доли секунды.
    """

    def __init__(self, root: Path = CRAFT_ROOT, device: str = "cpu"):
        """Оценщик без загрузки модели.

        Args:
            root: Корень клона CRAFT-pytorch и весов.
            device: Устройство torch (``cpu`` или ``cuda``; на CUDA — автокаст fp16).
        """
        self.root = Path(root)
        self.device = device
        self._net = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """Есть ли клон и веса CRAFT."""
        return (self.root / "src/craft/craft.py").exists() and (self.root / "models/craft/craft_mlt_25k.pth").exists()

    def load(self):
        """Модель CRAFT (ленивая загрузка, один раз на объект)."""
        with self._lock:
            if self._net is None:
                import torch

                from ocr_utils.page_layout.text_blocks.engines.workers import craft_worker as cw

                self._net = cw.load_model(str(self.root / "src/craft"), str(self.root / "models/craft/craft_mlt_25k.pth"),
                                          torch.device(self.device))  # fmt: skip
            return self._net

    def score_one(self, crop: np.ndarray, mask: np.ndarray) -> float:
        """Оценка одной вырезки (максимум карты по маске).

        Args:
            crop: Вырезка uint8.
            mask: Маска кандидата.

        Returns:
            Оценка 0…1.
        """
        import torch

        from ocr_utils.page_layout.text_blocks.engines.workers import craft_worker as cw

        if not mask.any():
            return 0.0
        image, ratio = cw.prepare_input(crop, 3840, 1.0)
        device = torch.device(self.device)
        score = cw.run_network(self.load(), image, device, device.type == "cuda")
        # Карта сети — в половинном разрешении входа; обратно к размеру вырезки.
        full = cv2.resize(score[:, :, 0], None, fx=2.0 / ratio, fy=2.0 / ratio, interpolation=cv2.INTER_LINEAR)
        heat = np.zeros(crop.shape, np.float32)
        h, w = min(heat.shape[0], full.shape[0]), min(heat.shape[1], full.shape[1])
        heat[:h, :w] = full[:h, :w]
        return float(np.clip(heat[mask].max(), 0.0, 1.0))

    def score_crops(self, crops: list[np.ndarray], masks: list[np.ndarray]) -> list[float] | None:
        """Оценки по вырезкам по очереди (вырезки разного размера — без пакета)."""
        return [self.score_one(crop, mask) for crop, mask in zip(crops, masks)]


# Оценщик процесса по умолчанию: в воркере v2 — клиент GPU-сервера (:func:`set_default_scorer` в инициализаторе пула),
# иначе — локальный CRAFT на CPU.
_SCORER: CraftScorer | None = None
_SCORER_SET = False


def set_default_scorer(scorer: CraftScorer | None) -> None:
    """Задать оценщик CRAFT процесса (``None`` — без CRAFT: одни правила).

    Args:
        scorer: Оценщик.
    """
    global _SCORER, _SCORER_SET
    _SCORER, _SCORER_SET = scorer, True


def default_scorer() -> CraftScorer | None:
    """Оценщик CRAFT процесса: заданный :func:`set_default_scorer` или локальный CRAFT на CPU; ``None`` — нет CRAFT."""
    global _SCORER, _SCORER_SET
    if not _SCORER_SET:
        _SCORER, _SCORER_SET = LocalCraft(), True
        if not _SCORER.available:
            log.warning("CRAFT для фильтра псевдоглифов не найден в %s — только правила", CRAFT_ROOT)
    return _SCORER if _SCORER is not None and _SCORER.available else None


def doubtful(frame: pd.DataFrame, out: pd.Series, params: PseudoParams = PseudoParams()) -> pd.Series:
    """Кандидаты, по которым голосует CRAFT.

    Это кандидаты, на которых правила промолчали, у края (``near_mm``) или далеко от ряда, серые, не бруски (тире,
    дефис), не длиннее ``CRAFT_MAX_SIDE_MM``; плюс сработавшие ``far_*`` — для вето.

    Args:
        frame: Кандидаты.
        out: Причины правил (:func:`reasons` без CRAFT).
        params: Пороги.

    Returns:
        Булева серия.
    """
    far = (frame.gap_rel.fillna(0.0) > params.far_gap) & (frame.gap_mm.fillna(0.0) >= params.far_gap_min_mm)
    bar = frame.w_mm >= params.bar_aspect * frame.h_mm
    small = np.maximum(frame.w_mm, frame.h_mm) <= CRAFT_MAX_SIDE_MM
    doubt = (out == "") & ((frame.edge_mm <= params.near_mm) | far) & ~bar & (frame.tone_rel >= params.craft_tone)
    veto = out.isin([PseudoReason.FAR_GREY.value, PseudoReason.FAR_THIN.value])
    return (doubt | veto) & small


# --- Проход фильтра и отчёт -------------------------------------------------------------------------------------------

# Закрас захватывает ореол компоненты на столько px (300 dpi ≈ 0.17 мм): пиксели ореола светлее порога Оцу, но темнее
# порога краски разбора (128), и без этого край ряда тянулся к ним по краске (ХиП 1991/07 IMG_0099_R, «Статья 97»).
HALO_PX = 2
# Наибольшее число проходов: после закраса крайней компоненты краем ряда становится следующая (крючок из двух кусков).
MAX_PASSES = 2
# CRAFT в режиме ``RULES_CRAFT``: порог спорных кандидатов (сетка 0.10–0.50 на разметке — вред постоянен).
CRAFT_DOUBT = 0.4


@dataclass(frozen=True)
class PseudoDrop:
    """Закрашенная компонента.

    Поля:
        box: Рамка ``(x0, y0, x1, y1)``, пиксели рендера 300 dpi.
        reason: Причина (:class:`PseudoReason`).
        row_end: Компонента была крайней в ряду (иначе — только у края картинки вне рядов).
        craft: Оценка CRAFT (``nan`` — не считалась).
    """

    box: tuple[int, int, int, int]
    reason: str
    row_end: bool
    craft: float = float("nan")

    def to_json(self) -> dict:
        """Словарь для JSON разбора."""
        return {"box": list(self.box), "reason": self.reason, "row_end": self.row_end,
                "craft": None if np.isnan(self.craft) else round(self.craft, 3)}  # fmt: skip


@dataclass(frozen=True)
class PseudoReport:
    """Что сделал фильтр псевдоглифов на странице.

    Поля:
        mode: Режим.
        drops: Закрашенные компоненты (по всем проходам).
        passes: Сколько раз разбор повторялся после закраса.
        craft_calls: Сколько раз звали CRAFT.
    """

    mode: str
    drops: tuple[PseudoDrop, ...] = ()
    passes: int = 0
    craft_calls: int = 0

    def to_json(self) -> dict:
        """Словарь для JSON разбора."""
        return {"mode": self.mode, "passes": self.passes, "craft_calls": self.craft_calls,
                "drops": [d.to_json() for d in self.drops]}  # fmt: skip


def erase_mask(labels: np.ndarray, drop: np.ndarray, halo_px: int = HALO_PX) -> np.ndarray:
    """Маска закраса: выбранные компоненты с ореолом ``halo_px``, кроме пикселей других компонент.

    Args:
        labels: Метки компонент краски 300 dpi.
        drop: Номера закрашиваемых компонент.
        halo_px: Ширина ореола, px.

    Returns:
        Булева маска размера ``labels``.
    """
    mask = np.isin(labels, drop)
    if halo_px > 0:
        kernel = np.ones((2 * halo_px + 1, 2 * halo_px + 1), np.uint8)
        grown = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        mask = grown & ((labels == 0) | mask)
    return mask


def filter_pass(
    analysis: "PageAnalysis",
    gray300: np.ndarray,
    raw300: np.ndarray,
    params: PseudoParams = PseudoParams(),
    scorer: CraftScorer | None = None,
) -> tuple[np.ndarray, list[PseudoDrop], int]:
    """Один проход: кандидаты по разбору → причины (и голос CRAFT) → закрас.

    Args:
        analysis: Разбор ``gray300``.
        gray300: Серый 300 dpi, как его видит разбор.
        raw300: Серый 300 dpi без утолщения (тон краски).
        params: Пороги (``craft_doubt`` учитывается, только если дан ``scorer``).
        scorer: Оценщик CRAFT или ``None`` — одни правила.

    Returns:
        ``(серый с закрашенными псевдоглифами, закрашенные, вызовов CRAFT)``; без находок — исходный серый.
    """
    frame = pd.DataFrame(page_candidates(analysis, gray300, raw300, None))
    if frame.empty:
        return gray300, [], 0
    comps = components_of(gray300)
    calls = 0
    if scorer is not None and params.craft_doubt is not None:
        base = reasons(frame, PseudoParams(**{**params.__dict__, "craft_doubt": None}))
        ask = doubtful(frame, base, params)
        # Без оценки CRAFT голос не подаётся: по умолчанию «символ» (1.0).
        frame["craft"] = 1.0
        # Одна компонента бывает кандидатом нескольких рядов — CRAFT по ней один раз; вырезки страницы — одним запросом.
        comps_asked = list(dict.fromkeys(int(c) for c in frame.loc[ask, "comp"]))
        if comps_asked:
            first = frame.drop_duplicates("comp").set_index("comp")
            pairs = [craft_crop(gray300, comps.labels, comp, tuple(int(first.at[comp, c]) for c in ("x0", "y0", "x1", "y1")))
                     for comp in comps_asked]  # fmt: skip
            scores = scorer.score_crops([p[0] for p in pairs], [p[1] for p in pairs])
            if scores is None:
                # Сервер CRAFT не ответил: голоса CRAFT нет, вето тоже — проход одними правилами.
                params = PseudoParams(**{**params.__dict__, "craft_doubt": None})
            else:
                calls = len(comps_asked)
                frame["craft"] = frame["comp"].map(dict(zip(comps_asked, scores))).fillna(1.0)
    else:
        params = PseudoParams(**{**params.__dict__, "craft_doubt": None})
        frame["craft"] = np.nan
    frame["reason"] = reasons(frame, params)
    hit = frame[frame.reason != ""].drop_duplicates("comp")
    if hit.empty:
        return gray300, [], calls
    cleaned = gray300.copy()
    cleaned[erase_mask(comps.labels, hit.comp.to_numpy())] = 255
    row_comps = set(frame[(frame.reason != "") & (frame.kind != CandidateKind.EDGE.value)].comp)
    drops = [PseudoDrop((int(r.x0), int(r.y0), int(r.x1), int(r.y1)), r.reason, int(r.comp) in row_comps,
                        float(r.craft) if np.isfinite(r.craft) and r.craft < 1.0 else float("nan"))
             for r in hit.itertuples()]  # fmt: skip
    return cleaned, drops, calls


def filter_page(
    analysis: "PageAnalysis",
    gray300: np.ndarray,
    raw300: np.ndarray | None,
    mode: PseudoMode,
    analyse: Callable[[np.ndarray], "PageAnalysis"],
    params: PseudoParams | None = None,
    scorer: CraftScorer | None = None,
) -> tuple["PageAnalysis", np.ndarray, PseudoReport]:
    """Фильтр псевдоглифов на странице: до ``MAX_PASSES`` проходов «кандидаты → закрас → повторный разбор».

    Разбор повторяется, только если закрашена хотя бы одна крайняя компонента ряда: черты у края вне рядов строки не
    видели, и разбор от их закраса не меняется (на ХиП 1991 — около половины страниц с закрасом).

    Args:
        analysis: Первый разбор ``gray300``.
        gray300: Серый 300 dpi, как его видит разбор.
        raw300: Серый 300 dpi без утолщения штрихов (для тона); ``None`` — тон по ``gray300``.
        mode: Режим (``OFF`` — ничего не делать).
        analyse: Разбор серого (тот же, что дал ``analysis``).
        params: Пороги; ``None`` — по умолчанию (у ``RULES_CRAFT`` порог CRAFT ``CRAFT_DOUBT``).
        scorer: Оценщик CRAFT для ``RULES_CRAFT``; ``None`` — общий оценщик процесса (:func:`default_scorer`).

    Returns:
        ``(итоговый разбор, серый после закраса, отчёт)``.
    """
    mode = PseudoMode(mode)
    if mode is PseudoMode.OFF:
        return analysis, gray300, PseudoReport(mode.value)
    raw = gray300 if raw300 is None else raw300
    if params is None:
        params = PseudoParams(craft_doubt=CRAFT_DOUBT if mode is PseudoMode.RULES_CRAFT else None)
    if mode is PseudoMode.RULES_CRAFT:
        scorer = scorer if scorer is not None else default_scorer()
    else:
        scorer = None
    current, gray, drops, passes, calls = analysis, gray300, [], 0, 0
    for _ in range(MAX_PASSES):
        gray_next, found, asked = filter_pass(current, gray, raw, params, scorer)
        calls += asked
        if not found:
            break
        drops.extend(found)
        gray = gray_next
        if not any(d.row_end for d in found):
            break
        current = analyse(gray)
        passes += 1
    return current, gray, PseudoReport(mode.value, tuple(drops), passes, calls)


__all__ = ["CRAFT_DOUBT", "CandidateKind", "Components", "CraftScorer", "LocalCraft", "craft_crop", "set_default_scorer", "EdgeSide", "PseudoDrop", "PseudoMode",
           "PseudoParams", "PseudoReason", "PseudoReport", "components_of", "default_scorer", "doubtful",
           "erase_mask", "filter_page", "filter_pass", "page_candidates", "reason_of", "reasons",
           "relative_height", "row_members", "stroke_mm", "word_spaces_mm"]  # fmt: skip
