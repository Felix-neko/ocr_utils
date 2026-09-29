"""Разбор одной страницы: рендер → строки движка → оси → колонки → блоки с огибающими и выключкой."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks.alignment import Alignment, alignment_of
from ocr_utils.page_layout.text_blocks.baseline_axis import body_axes, extend_to_ink, use_body
from ocr_utils.page_layout.text_blocks.axes_fix import fixed_axes
from ocr_utils.page_layout.text_blocks.blocks import (
    COARSE_FACTOR,
    DEFAULT_BLOCKS_MODE,
    DILATE_GLYPHS,
    SMOOTH_PITCHES,
    BlocksMode,
    TextBlock,
    blocks_of,
)
from ocr_utils.page_layout.text_blocks.hyphens import hyphens_mask
from ocr_utils.page_layout.text_blocks.engines.base import Engine
from ocr_utils.page_layout.text_blocks.hints import LayoutHints, OrientedZone, masked_ink, zone_mask
from ocr_utils.page_layout.text_blocks.lines import SMOOTH_HEIGHTS, LineAxis, axes_of, with_column, with_jump_spans
from ocr_utils.page_layout.text_blocks.metrics import pitch_of, points_without_marks, row_jumps_of
from ocr_utils.page_layout.orientation.detectors.ink_axis import glyph_mask
from ocr_utils.page_layout.text_blocks.columns import gutters_of, mark_cut_lines, zones_of
from ocr_utils.page_layout.text_blocks.leaders import leaders_mask, leaders_of

# Полоса строки для меры покрытия краски: ось ± столько её высот.
BAND_HEIGHTS = 0.6


class Variant(str, Enum):
    """Какой из двух рендеров FineReader разбираем."""

    GEO = "geo"  # с коррекцией геометрии
    NOGEO = "nogeo"  # без коррекции


@dataclass(frozen=True)
class PageAnalysis:
    """Результат разбора страницы: оси строк, блоки-колонки и вердикты выключки."""

    name: str  # имя PDF без расширения
    page: int  # номер страницы с единицы
    variant: str
    engine: str
    width: int  # размеры рабочей копии
    height: int
    dpi: float
    axes: tuple[LineAxis, ...]
    blocks: tuple[TextBlock, ...]
    alignments: tuple[Alignment, ...]
    gutters: tuple  # локальные межколонники (``columns.Gutter``)
    zones: tuple  # зоны вёрстки (``columns.Zone``)
    seconds: float
    ink_share: float = 0.0  # доля краски текста, накрытая полосами найденных строк
    leaders: tuple = ()  # отточия страницы (``leaders.Leader``)
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.name}_p{self.page:03d}_{self.variant}_{self.engine}"


class AxisKind(Enum):
    """Какая ось строки основная: прежняя по центру масс краски или вторая — по базовой линии глифов."""

    CENTRE = "centre"
    BODY = "body"


def analyse_gray(
    gray300: np.ndarray,
    engine: Engine,
    dpi: float = WORK_DPI,
    smooth_line: float = SMOOTH_HEIGHTS,
    smooth_block: float = SMOOTH_PITCHES,
    coarse_factor: float = COARSE_FACTOR,
    dilate: float = DILATE_GLYPHS,
    name: str = "page",
    page: int = 1,
    variant: str = Variant.NOGEO.value,
    hints: LayoutHints | None = None,
    axis: AxisKind = AxisKind.CENTRE,
    blocks_mode: BlocksMode | None = None,
) -> PageAnalysis:
    """Разбор страницы по серому рендеру ``RENDER_DPI``.

    Args:
        gray300: Серый рендер страницы в ``RENDER_DPI``.
        engine: Поставщик строк (``engines.ink.InkEngine`` или адаптер чужого движка).
        dpi: Разрешение рабочей копии, в котором отдаются все координаты.
        smooth_line: Окно сглаживания оси строки в долях её высоты.
        smooth_block: Окно огибающей блока в межстрочных интервалах.
        coarse_factor: Во сколько раз шире окно крупной огибающей.
        dilate: На какую долю размера символа раздувать границу блока.
        name, page, variant: Чем подписать результат.
        hints: Вспомогательная информация внешних детекторов (:class:`hints.LayoutHints`): маска
            разрешённого текста, области с ориентацией текста, рамки таблиц и блок-схем. ``None``
            — разбор как прежде, одной прямой областью на всю полосу.
        axis: Какая ось строки основная — по ней строятся ряды и блоки: ``CENTRE`` — прежняя, по
            центру масс краски; ``BODY`` — вторая, по базовой линии глифов (:mod:`baseline_axis`).
            Вторая ось считается всегда (``LineAxis.body_points``), выбор влияет только на блоки.
        blocks_mode: Способ группировки строк в блоки и их границы (``blocks.BlocksMode``); ``None`` —
            ``blocks.DEFAULT_BLOCKS_MODE``. У ``SMOOTH`` перед сборкой блоков оси-выбросы крупного
            набора заменяются (:func:`axes_fix.fixed_axes`), и в разборе остаются уже заменённые оси.

    Returns:
        :class:`PageAnalysis` со всеми кривыми в пикселях рабочей копии.
    """
    started = time.monotonic()
    hints = hints or LayoutHints()
    width = int(round(gray300.shape[1] * dpi / RENDER_DPI))
    height = int(round(gray300.shape[0] * dpi / RENDER_DPI))
    areas = hints.zones_or_page(width, height)
    if len(areas) > 1 or areas[0].rotate_cw != 0:
        # Области разбираются ПОРОЗНЬ: строки сращиваются и блоки собираются только внутри одной
        # ориентации, а через границу областей сшивать нечего.
        return _analyse_areas(
            gray300,
            engine,
            areas,
            hints,
            dpi,
            smooth_line,
            smooth_block,
            coarse_factor,
            dilate,
            name,
            page,
            variant,
            started,
            width,
            height,
            axis,
            blocks_mode,
        )
    result = engine.segment(gray300, dpi)
    # Вторая ось (по базовой линии глифов) считается у всех строк сразу: ей нужны соседи.
    axes = body_axes(axes_of(result.lines, dpi, smooth_line))
    # Куски заголовка во всю ширину, набранные через межколонник, помечаются до сборки блоков.
    work = _work_copy(gray300, dpi)
    # Отточия считаются один раз на разбор: они нужны и колонкам (поле точек — не межколонник), и
    # краске текста (точка мельче порога маски глифов), и рядам блока (их не берут в меры набора).
    page_leaders = list(result.leaders) if result.leaders else leaders_of(work, dpi)[0]
    gutters = result.gutters or gutters_of(work, dpi, page_leaders)
    ink = text_ink(gray300, dpi, work=work, leaders=page_leaders)
    # Вторая ось доходит до края краски строки — дефиса переноса, точки (их нет среди глифов).
    axes = extend_to_ink(axes, ink > 0, RENDER_DPI / dpi)
    if axis is AxisKind.BODY:
        axes = use_body(axes)
    cut = mark_cut_lines(axes, gutters, dpi, ink=gray300 < 128, k=RENDER_DPI / dpi)
    axes = [with_column(line, line.column, cross=flag) for line, flag in zip(axes, cut)]
    # Межколонники — свойство страницы, а не движка: чужие сегментаторы их не отдают, и без них
    # три колонки заметки слипались в один блок (1975/05 с.97, pero). Считаем сами по рендеру.
    zones = zones_of(gutters, height, width, dpi)
    mode = DEFAULT_BLOCKS_MODE if blocks_mode is None else BlocksMode(blocks_mode)
    if mode is BlocksMode.SMOOTH:
        # Оси крупного набора, ушедшие наискось от корпуса (крупный курсив — 1971/08 IMG_0072_1L),
        # заменяются до сборки рядов: полосы строк заголовка иначе налезают друг на друга.
        axes, _ = fixed_axes(axes, dpi)
    blocks = blocks_of(
        axes,
        zones,
        gutters,
        width,
        ink,
        result.rules,
        dpi,
        smooth_block,
        coarse_factor,
        dilate,
        page_leaders,
        hints.barrier_lines,
        mode,
    )
    # Участки перескока на соседнюю строку — по готовым рядам (им нужен шаг строк); в осях и рядах
    # блоков меры формы у таких осей пересчитываются без перескока.
    axes, blocks = marked_jumps(axes, blocks)
    alignments = [alignment_of(block) for block in blocks]
    share = ink_share(axes, ink, dpi)
    return PageAnalysis(
        name=name,
        page=page,
        variant=variant,
        engine=result.engine or getattr(engine, "name", "?"),
        width=width,
        height=height,
        dpi=float(dpi),
        axes=tuple(axes),
        blocks=tuple(blocks),
        alignments=tuple(alignments),
        gutters=tuple(gutters),
        leaders=tuple(page_leaders),
        zones=tuple(zones),
        seconds=time.monotonic() - started,
        ink_share=share,
        note=result.note,
    )


def marked_jumps(axes: list[LineAxis], blocks: list[TextBlock]) -> tuple[list[LineAxis], list[TextBlock]]:
    """Проставить осям участки перескока на соседнюю строку (``LineAxis.jump_spans``).

    Перескоки ищет :func:`metrics.row_jumps_of` по точкам осей без столбцов точек и запятых, шаг
    строк — :func:`metrics.pitch_of` по рядам блоков. Оси с перескоком заменяются и в списке осей, и в
    рядах блоков (ряд держит те же объекты осей).

    Args:
        axes: Оси страницы.
        blocks: Блоки, собранные из этих осей.

    Returns:
        ``(оси, блоки)`` — те же, где у осей с перескоком проставлены участки и пересчитаны меры формы.
    """
    pitch = pitch_of([{"rows": [{"y": row.y} for row in block.rows]} for block in blocks])
    points = [points_without_marks(np.asarray(axis.points, dtype=np.float64), axis.mark_spans) for axis in axes]
    jumps = row_jumps_of(points, pitch)
    if not jumps:
        return axes, blocks
    # Одна находка на ось: участок перескока — от начала до конца перехода.
    marked = {jump.axis: with_jump_spans(axes[jump.axis], ((jump.start, jump.stop),)) for jump in jumps}
    swapped = {id(axes[index]): axis for index, axis in marked.items()}
    new_axes = [marked.get(index, axis) for index, axis in enumerate(axes)]
    new_blocks = []
    for block in blocks:
        # Ряды, где есть заменённая ось, пересобираются с новой; остальные — как были.
        rows = tuple(
            (
                replace(row, axes=tuple(swapped.get(id(axis), axis) for axis in row.axes))
                if any(id(axis) in swapped for axis in row.axes)
                else row
            )
            for row in block.rows
        )
        new_blocks.append(replace(block, rows=rows))
    return new_axes, new_blocks


def _analyse_areas(
    gray300: np.ndarray,
    engine: Engine,
    areas: tuple[OrientedZone, ...],
    hints: LayoutHints,
    dpi: float,
    smooth_line: float,
    smooth_block: float,
    coarse_factor: float,
    dilate: float,
    name: str,
    page: int,
    variant: str,
    started: float,
    width: int,
    height: int,
    axis: AxisKind = AxisKind.CENTRE,
    blocks_mode: BlocksMode | None = None,
) -> PageAnalysis:
    """Разбор по ОБЛАСТЯМ с разной ориентацией текста, со сведением результатов в один разбор.

    Каждая область разбирается отдельным проходом по своей вырезке: боковая — повёрнутой до
    прямого текста (:mod:`orient`), прямая — как есть. Так строки сращиваются и блоки собираются
    только внутри одной ориентации, а ни один порог конвейера не трогается.

    Меры выключки считаются только у прямых областей: «левый край блока» у лежащего текста
    зависит от стороны чтения, которую мы намеренно не определяем.
    """
    from ocr_utils.page_layout.text_blocks.orient import back_axis, back_block, back_gutter, back_leader, upright

    scale = RENDER_DPI / dpi
    axes: list[LineAxis] = []
    blocks: list[TextBlock] = []
    alignments: list[Alignment] = []
    gutters: list = []
    leaders: list = []
    zones: list = []
    for index, area in enumerate(areas):
        size = (area.width, area.height)
        # Области перекрываются: полоса целиком идёт первой, а найденные детектором боковые
        # врезки — за ней. Область гасит у себя всё, что забрали ПОСЛЕДУЮЩИЕ: так один и тот же
        # текст не разбирается дважды, и порядок областей и есть их старшинство.
        allowed = _area_allowed(hints.text_allowed, areas[index + 1 :], (height, width))
        crop = upright(masked_ink(gray300, allowed), area, scale)
        if crop.size == 0 or min(crop.shape[:2]) < scale * 4:
            continue
        # Подсказки области — в координатах её выпрямленной вырезки: рамки-запреты и линейки-барьеры.
        # Маска разрешённого текста уже применена к вырезке и дальше не нужна.
        area_hints = LayoutHints(
            barriers=_shifted_barriers(hints.barriers, area),
            rules=hints.barrier_lines.in_area(area.box, area.rotate_cw, _forward_points).as_tuples(),
            dpi=dpi,
        )
        # Движок с подсказками (``InkEngine``) получает подсказки ОБЛАСТИ: со страничными он резал
        # бы вырезку по чужим координатам.
        area_engine = engine.with_hints(area_hints) if hasattr(engine, "with_hints") else engine
        inner = analyse_gray(
            crop,
            area_engine,
            dpi=dpi,
            smooth_line=smooth_line,
            smooth_block=smooth_block,
            coarse_factor=coarse_factor,
            dilate=dilate,
            name=name,
            page=page,
            variant=variant,
            hints=area_hints,
            axis=axis,
            blocks_mode=blocks_mode,
        )
        axes.extend(back_axis(line, size, area) for line in inner.axes)
        for block, alignment in zip(inner.blocks, inner.alignments):
            blocks.append(back_block(block, size, area))
            alignments.append(alignment)
        gutters.extend(item for item in (back_gutter(g, size, area) for g in inner.gutters) if item is not None)
        leaders.extend(item for item in (back_leader(item, size, area) for item in inner.leaders) if item is not None)
        if not area.sideways and area.rotate_cw == 0:
            zones.extend(inner.zones)
    work = _work_copy(gray300, dpi)
    ink = text_ink(gray300, dpi, work=work, leaders=leaders)
    return PageAnalysis(
        name=name,
        page=page,
        variant=variant,
        engine=getattr(engine, "name", "?"),
        width=width,
        height=height,
        dpi=float(dpi),
        axes=tuple(axes),
        blocks=tuple(blocks),
        alignments=tuple(alignments),
        gutters=tuple(gutters),
        leaders=tuple(leaders),
        zones=tuple(zones),
        seconds=time.monotonic() - started,
        ink_share=ink_share(axes, ink, dpi),
        note=f"областей {len(areas)}",
    )


def _area_allowed(
    allowed: np.ndarray | None, later: tuple[OrientedZone, ...], shape: tuple[int, int]
) -> np.ndarray | None:
    """Маска разрешённого текста области: общая маска минус рамки более старших областей."""
    if not later:
        return allowed
    out = np.ones(shape, dtype=bool) if allowed is None else allowed.copy()
    for area in later:
        out &= ~zone_mask(shape, area)
    return out


def _shifted_barriers(
    barriers: tuple[tuple[int, int, int, int], ...], area: OrientedZone
) -> tuple[tuple[int, int, int, int], ...]:
    """Рамки-запреты, пересчитанные в координаты ВЫПРЯМЛЕННОЙ вырезки области.

    Рамка вне области выбрасывается: её рёбра там ничего не разделяют.
    """
    out: list[tuple[int, int, int, int]] = []
    ax0, ay0, ax1, ay1 = area.box
    for x0, y0, x1, y1 in barriers:
        if x1 <= ax0 or x0 >= ax1 or y1 <= ay0 or y0 >= ay1:
            continue
        box = (max(x0, ax0) - ax0, max(y0, ay0) - ay0, min(x1, ax1) - ax0, min(y1, ay1) - ay0)
        if area.rotate_cw == 0:
            out.append(box)
            continue
        # Рамка поворачивается вместе с вырезкой: углы переводятся и берётся их охват.
        from ocr_utils.page_layout.text_blocks.orient import back_points

        corners = np.array([[box[0], box[1]], [box[2], box[1]], [box[0], box[3]], [box[2], box[3]]], dtype=np.float64)
        turned = _forward_points(corners, (area.width, area.height), area.rotate_cw)
        out.append((int(turned[:, 0].min()), int(turned[:, 1].min()), int(turned[:, 0].max()), int(turned[:, 1].max())))
    return tuple(out)


def _forward_points(points: np.ndarray, size: tuple[int, int], rotate_cw: int) -> np.ndarray:
    """Точки исходной вырезки в координаты ВЫПРЯМЛЕННОГО кадра (обратное к ``orient.back_points``)."""
    width, height = size
    xs, ys = points[:, 0], points[:, 1]
    if rotate_cw == 90:
        return np.column_stack([(height - 1) - ys, xs])
    if rotate_cw == 180:
        return np.column_stack([(width - 1) - xs, (height - 1) - ys])
    if rotate_cw == 270:
        return np.column_stack([ys, (width - 1) - xs])
    return points


def _work_copy(gray300: np.ndarray, dpi: float) -> np.ndarray:
    """Рабочая копия рендера в разрешении ``dpi``."""
    size = (max(1, round(gray300.shape[1] * dpi / RENDER_DPI)), max(1, round(gray300.shape[0] * dpi / RENDER_DPI)))
    return cv2.resize(gray300, size, interpolation=cv2.INTER_AREA)


def ink_share(axes: list[LineAxis], ink: np.ndarray, dpi: float) -> float:
    """Доля краски текста, накрытая полосами найденных строк.

    Мера полноты сегментации, одинаковая для всех движков: полоса строки — её ось плюс-минус
    ``BAND_HEIGHTS`` высоты. Чем меньше доля, тем больше текста движок не увидел.

    Args:
        axes: Оси строк страницы (пиксели рабочей копии).
        ink: Краска текста рендера ``RENDER_DPI``.
        dpi: Разрешение рабочей копии.

    Returns:
        Доля от 0 до 1; 0, если краски нет.
    """
    total = int(ink.sum())
    if total == 0:
        return 0.0
    k = RENDER_DPI / dpi
    covered = np.zeros(ink.shape, dtype=np.uint8)
    for axis in axes:
        half = max(1.0, BAND_HEIGHTS * axis.height) * k
        points = np.asarray(axis.points, dtype=np.float64) * k
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            cv2.line(covered, (int(x0), int(y0)), (int(x1), int(y1)), 1, thickness=int(round(2 * half)))
    return float((ink & (covered > 0)).sum() / total)


def text_ink(
    gray300: np.ndarray, dpi: float = WORK_DPI, work: np.ndarray | None = None, leaders: list | None = None
) -> np.ndarray:
    """Краска ТЕКСТА рендера: краска под маской глифов рабочей копии плюс точки отточий.

    Края рядов ищутся по ней, а не по сырой краске: тень корешка и рамка кадра — сплошные
    тёмные полосы у поля страницы, и сырой край ряда цеплялся за них (1975/05 с.97, левая
    колонка). ``glyph_mask`` оставляет только компоненты размером с глиф.
    """
    work = _work_copy(gray300, dpi) if work is None else work
    glyphs = glyph_mask(work)
    mask = glyphs
    # Точки отточий (0.5 мм) ниже нижнего порога маски глифов (5 px), и без них край ряда
    # останавливается на последнем слове, а поле точек остаётся вне блока (1971/10 с.93).
    # Берутся не любые точки, а только собранные в цепочки — пыль краем ряда не станет.
    dots = leaders_mask(work, dpi, leaders) if leaders is not None else leaders_of(work, dpi)[1]
    mask = cv2.max(mask, dots)
    # Дефисы (перенос, составные слова) тоже ниже порога маски глифов (5 × 3 px), и строка с
    # переносом теряла край на 1 мм — правый край колонки по формату выглядел рваным (1976/09 с.92).
    mask = cv2.max(mask, hyphens_mask(work, glyphs, dpi))
    glyphs = cv2.resize(mask, (gray300.shape[1], gray300.shape[0]), interpolation=cv2.INTER_NEAREST)
    return (gray300 < 128) & (glyphs > 0)


def render_page(pdf: Path, page: int) -> np.ndarray:
    """Серый рендер страницы ``page`` (с единицы) из PDF в ``RENDER_DPI``."""
    import fitz

    from ocr_utils.geometry_regression.render import render_gray

    with fitz.open(pdf) as document:
        return render_gray(document, page - 1)


def analyse_pdf_page(
    pdf: Path, page: int, engine: Engine, variant: str = Variant.NOGEO.value, **options
) -> PageAnalysis:
    """Разбор страницы PDF: рендер плюс :func:`analyse_gray`."""
    gray300 = render_page(pdf, page)
    return analyse_gray(gray300, engine, name=pdf.stem, page=page, variant=variant, **options)


__all__ = ["PageAnalysis", "Variant", "analyse_gray", "analyse_pdf_page", "ink_share", "render_page", "text_ink"]
