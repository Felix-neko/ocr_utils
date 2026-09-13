"""Сборка «стало»: таблица, в которой боковой текст заменён прямым.

ТРИ ДЕЙСТВИЯ, и все три обратимы по отдельности, чтобы на листе «было-стало» было видно,
какое из них подвело.

1. РАСШИРЕНИЕ КОЛОНКИ. Прямой текст на месте бокового почти никогда не влезает: колонку
   для того и делали узкой, что заголовок в ней стоял боком. Расширять — законно: эти
   таблицы пойдут отдельными страницами под FineReader, и размер страницы не ограничен, а
   вот мельчить шрифт нельзя, иначе распознавание испортится ровно там, где мы его чиним.
   Колонка раздвигается вставкой перед её правой линейкой, и вставка заполняется
   ПОСТРОЧНОЙ МЕДИАНОЙ самой колонки: в строке горизонтальной линейки вся колонка тёмная,
   и медиана продолжает линейку; в строке текста тёмных пикселей меньшинство, и медиана
   даёт бумагу. Повторять крайний столбец нельзя — он приходится на вертикальную линейку,
   и вставка выходит чёрной полосой (так и вышло на первом прогоне).

   Объединённая шапка, которую вставка разрезает пополам («Шкаф для платья» над тремя
   графами), после раздвижки переносится из исходной картинки целиком и ставится по центру
   новой, более широкой ячейки — иначе в середине слова появляется просвет.

2. СТИРАНИЕ. Внутренность ячейки заливается цветом бумаги. Линейки не трогаются: заливка
   идёт по рамке, ужатой на толщину разделителя. Ничего умнее (LaMa, inpainting) здесь не
   нужно — фон полосы уже выровнен предыдущими шагами конвейера, и заливка одним уровнем
   от него не отличима.

3. НАБОР. Текст кладётся тем же кеглем, каким он был набран в исходнике: кегль меряется по
   медианной высоте компоненты в самой ячейке, а не задаётся числом. Цвет краски — тёмный
   квантиль ТАБЛИЦЫ ЦЕЛИКОМ, чтобы замена не выделялась чернотой на сером скане и при этом
   не бледнела на ячейке, где исходный текст был тонким (по отдельной ячейке квантиль
   уезжает вверх, и замена выходила серой).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from research.legacy.table_processing.detection.ruling import mm_to_px
from research.legacy.table_processing.geometry import Box, Cell, Grid
from research.legacy.table_processing.render.fonts import Fit, fit_text, load_font, required_width
from research.legacy.table_processing.rotation.ink_axis_cell import glyph_mask
from research.legacy.table_processing.structure.ruling_grid import interior

# Во сколько раз кегль больше медианной высоты компоненты. У DejaVu высота заглавной около
# 0.73 кегля, строчной без выносных — около 0.55; медиана по смеси прописных, строчных и
# цифр в этих шапках даёт примерно 0.62 кегля, отсюда множитель.
FONT_FROM_GLYPH = 1.6

# Границы кегля в пикселях рабочего разрешения (300 dpi): 14 px это 1.2 мм, мельче петита;
# 60 px это 5 мм, крупнее любой шапки этих таблиц.
MIN_FONT_PX = 14
MAX_FONT_PX = 60

# Больше стольких строк в ячейке не набираем: дальше начинаются переносы по буквам вида
# «колич-/ество», и FineReader на них спотыкается ровно так же, как на боковом тексте.
MAX_LINES = 3

# Насколько шире делать колонку сверх необходимого: 1 мм запаса на неточность метрик.
WIDEN_SLACK_MM = 1.0


@dataclass(frozen=True)
class Replacement:
    """Что и на что меняем в одной ячейке."""

    cell: Cell
    text: str
    rotate_cw: int = 90


@dataclass
class Composed:
    """Результат сборки со следами того, что пришлось сделать."""

    image: np.ndarray
    grid: Grid
    widened: dict[int, int] = field(default_factory=dict)  # колонка → на сколько раздвинули
    fits: dict[tuple[int, int], Fit] = field(default_factory=dict)
    failed: list[tuple[int, int]] = field(default_factory=list)


def paper_level(gray: np.ndarray) -> int:
    return int(np.percentile(gray, 90)) if gray.size else 255


def ink_level(gray: np.ndarray) -> int:
    return int(np.percentile(gray, 5)) if gray.size else 0


def glyph_height(gray: np.ndarray, dpi: int) -> float:
    """Медианная высота компоненты в ячейке — по ней меряется исходный кегль.

    Ячейка подаётся УЖЕ ВЫПРЯМЛЕННОЙ: у повёрнутого текста «высота» компоненты — это его
    ширина, и мерить надо после поворота.
    """
    mask = glyph_mask(gray, dpi)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return 0.0
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    keep = (widths * heights) >= (0.15 * (dpi / 25.4) ** 2)
    return float(np.median(heights[keep])) if keep.any() else 0.0


def target_font_px(gray: np.ndarray, dpi: int) -> int:
    """Кегль замены: тот же, что был в исходнике."""
    height = glyph_height(gray, dpi)
    if height <= 0:
        return MIN_FONT_PX * 2
    return int(np.clip(round(height * FONT_FROM_GLYPH), MIN_FONT_PX, MAX_FONT_PX))


def widen(image: np.ndarray, grid: Grid, extra: dict[int, int]) -> tuple[np.ndarray, Grid]:
    """Раздвинуть колонки: вставка перед правой линейкой колонки, линейки продолжаются сами."""
    if not extra:
        return image, grid

    height = image.shape[0]
    cuts = grid.cuts
    pieces: list[np.ndarray] = []
    shift = 0
    new_xs: list[int] = []
    new_cuts: list[int] = []
    previous = 0
    for column in range(grid.n_cols):
        # Резать надо по краю штриха линейки, а не по её центру: иначе половина штриха
        # уезжает вместе с левым куском, половина остаётся с правым, и после вставки
        # линейка двоится — на теле таблицы это видно как две черты со щелью между ними.
        left, right = cuts[column], cuts[column + 1]
        new_xs.append(grid.xs[column] + shift)
        new_cuts.append(left + shift)
        pieces.append(image[:, previous:right])
        gap = extra.get(column, 0)
        if gap > 0:
            pieces.append(np.repeat(_gap_column(image[:, left:right])[:, None], gap, axis=1))
            shift += gap
        previous = right
    pieces.append(image[:, previous:])
    new_xs.append(grid.xs[-1] + shift)
    new_cuts.append(cuts[-1] + shift)

    widened = np.concatenate(pieces, axis=1) if pieces else image
    assert widened.shape[0] == height

    # Сдвиг считается ПО ПОЛОЖЕНИЮ, а не поиском старой координаты в словаре: рёбра ячеек
    # меряются по линейке своей полосы и с общими разделителями ``xs`` больше не совпадают
    # (на паке расхождение до 20 px). Словарь на таких координатах промахивался, одна
    # сторона рамки уезжала, другая нет, и рамка выворачивалась наизнанку.
    def shift_x(value: int) -> int:
        return value + sum(gap for column, gap in extra.items() if gap > 0 and cuts[column + 1] <= value)

    def moved_cell(cell: Cell) -> Cell:
        box = Box(shift_x(cell.box.x0), cell.box.y0, shift_x(cell.box.x1), cell.box.y1)
        inner = cell.inner
        if inner is not None:
            # Внутренность двигается ЗА СВОИМИ краями рамки, а не сама по себе. Вставка
            # ложится между правым краем внутренности и линейкой, поэтому прямой сдвиг
            # внутренности её не расширял: рамка ячейки росла, а место под текст — нет,
            # и подбор кегля сваливался к минимуму (замер: 26 px до раздвижки, 14 после).
            inner = Box(inner.x0 + (box.x0 - cell.box.x0), inner.y0, inner.x1 + (box.x1 - cell.box.x1), inner.y1)
        return Cell(
            row=cell.row,
            col=cell.col,
            box=box,
            row_span=cell.row_span,
            col_span=cell.col_span,
            is_header=cell.is_header,
            inner=inner,
        )

    cells = [moved_cell(cell) for cell in grid.cells]
    moved = Grid(
        xs=new_xs,
        ys=list(grid.ys),
        cells=cells,
        header_rows=grid.header_rows,
        double_rule_ys=list(grid.double_rule_ys),
        source=grid.source,
        column_cuts=new_cuts,
    )
    return widened, moved


# Что считать краской при разборе вставки и какая доля строки должна быть краской, чтобы
# строка считалась линейкой. Линейка идёт через ВСЮ колонку, у неё доля близка к единице;
# строка, прошедшая по перекладине цифр, набирает до половины — отсюда порог 0.8, а не 0.5:
# при 0.5 на месте вставки оставались бледные полосы поперёк тела таблицы.
GAP_INK_LEVEL = 0.6
GAP_RULE_SHARE = 0.8


def _gap_column(strip: np.ndarray) -> np.ndarray:
    """Столбец, которым заполняется вставка: линейки сохранены, остальное — бумага."""
    paper = float(np.percentile(strip, 90)) if strip.size else 255.0
    dark_share = (strip < paper * GAP_INK_LEVEL).mean(axis=1)
    median = np.median(strip, axis=1)
    return np.where(dark_share >= GAP_RULE_SHARE, median, paper).astype(strip.dtype)


def erase(image: np.ndarray, box: Box, level: int) -> None:
    """Залить внутренность ячейки цветом бумаги. На месте, линейки не трогая."""
    safe = box.clipped(image.shape[1], image.shape[0])
    if safe.width > 0 and safe.height > 0:
        image[safe.slice] = level


def draw_text(image: np.ndarray, box: Box, fit: Fit, colour: int, font_path: "Path | None" = None) -> None:
    """Набрать текст по центру рамки."""
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    font = load_font(fit.font_px, font_path)
    line_height = int(round(fit.font_px * 1.15))
    top = box.y0 + max(0, (box.height - line_height * len(fit.lines)) // 2)
    for index, line in enumerate(fit.lines):
        width = int(round(font.getlength(line)))
        left = box.x0 + max(0, (box.width - width) // 2)
        draw.text((left, top + index * line_height), line, fill=int(colour), font=font)
    image[:, :] = np.asarray(canvas)


def compose(
    image: np.ndarray, grid: Grid, replacements: list[Replacement], dpi: int, allow_widen: bool = True
) -> Composed:
    """Собрать «стало»: расширить колонки, стереть боковой текст, набрать прямой."""
    fonts: dict[tuple[int, int], int] = {}
    colours: dict[tuple[int, int], int] = {}
    extra: dict[int, int] = {}
    table_ink = ink_level(image)
    slack = mm_to_px(WIDEN_SLACK_MM, dpi)

    for replacement in replacements:
        cell = replacement.cell
        source = image[interior(cell, dpi).slice]
        # Кегль меряется по ВЫПРЯМЛЕННОЙ ячейке: у бокового текста высота буквы — это её
        # ширина в исходной ориентации.
        upright = np.ascontiguousarray(np.rot90(source, k=-(replacement.rotate_cw // 90)))
        font_px = target_font_px(upright, dpi)
        fonts[cell.key] = font_px
        # Цвет краски берём из ИСХОДНОЙ ячейки, а не из раздвинутой: в раздвинутой добавилась
        # бумага, тёмный квантиль по ней уезжает вверх, и замена выходит серой.
        colours[cell.key] = table_ink
        if not allow_widen or cell.col_span > 1:
            continue
        needed = required_width(replacement.text, font_px, cell.box.height, MAX_LINES) + slack
        if needed > cell.box.width:
            column = cell.col
            extra[column] = max(extra.get(column, 0), needed - cell.box.width)

    widened_image, widened_grid = widen(image.copy(), grid, extra)
    by_key = {cell.key: cell for cell in widened_grid.cells}
    if extra:
        _recenter_spanning(image, grid, widened_image, widened_grid, extra, dpi)

    result = Composed(image=widened_image, grid=widened_grid, widened=extra)
    for replacement in replacements:
        cell = by_key.get(replacement.cell.key, replacement.cell)
        inner = interior(cell, dpi)
        level = paper_level(widened_image[inner.slice])
        colour = colours[replacement.cell.key]
        font_px = fonts[replacement.cell.key]
        fit = fit_text(replacement.text, inner.width, inner.height, font_px, MIN_FONT_PX)
        erase(widened_image, inner, level)
        if fit is None:
            result.failed.append(cell.key)
            continue
        draw_text(widened_image, inner, fit, colour)
        result.fits[cell.key] = fit
    return result


def _recenter_spanning(
    source: np.ndarray, source_grid: Grid, target: np.ndarray, target_grid: Grid, extra: dict[int, int], dpi: int
) -> None:
    """Перенести содержимое объединённых ячеек, которые разрезала вставка.

    Вставка идёт по границе КОЛОНКИ, а объединённая шапка накрывает несколько колонок —
    и вставка приходится ей в середину слова. Содержимое такой ячейки берётся из исходной
    картинки целиком и ставится по центру новой; всё, что вставка сделала внутри неё,
    затирается.
    """
    widened_columns = {column for column, gap in extra.items() if gap > 0}
    by_key = {cell.key: cell for cell in target_grid.cells}
    for cell in source_grid.cells:
        if cell.col_span <= 1:
            continue
        inner_columns = set(range(cell.col, cell.col + cell.col_span - 1))
        if not (inner_columns & widened_columns):
            continue
        moved = by_key.get(cell.key)
        if moved is None:
            continue
        old_inner = interior(cell, dpi).clipped(source.shape[1], source.shape[0])
        new_inner = interior(moved, dpi).clipped(target.shape[1], target.shape[0])
        if old_inner.width <= 0 or old_inner.height <= 0 or new_inner.width < old_inner.width:
            continue
        patch = source[old_inner.slice]
        erase(target, new_inner, paper_level(patch))
        left = new_inner.x0 + (new_inner.width - old_inner.width) // 2
        top = new_inner.y0 + (new_inner.height - old_inner.height) // 2
        target[top : top + patch.shape[0], left : left + patch.shape[1]] = patch


# Поле вокруг таблицы на итоговой картинке: 2 мм. Столько же оставляет вырезка, но после
# раздвижки края лучше пересчитать заново — вставка идёт через всю высоту кадра, и то,
# что лежало под таблицей (подрисуночная подпись), она разрезает.
TRIM_MARGIN_MM = 2.0


def table_bounds(grid: Grid, shape: tuple[int, int], dpi: int) -> Box:
    """Рамка самой таблицы с полем — чтобы не показывать искалеченные окрестности."""
    margin = mm_to_px(TRIM_MARGIN_MM, dpi)
    height, width = shape
    return Box(grid.xs[0] - margin, grid.ys[0] - margin, grid.xs[-1] + margin, grid.ys[-1] + margin).clipped(
        width, height
    )


def trim(image: np.ndarray, grid: Grid, dpi: int) -> tuple[np.ndarray, Grid]:
    """Обрезать по таблице картинку И СЕТКУ ВМЕСТЕ.

    Раздельно их обрезать нельзя, и это не предположение: обрезанная картинка уже рисовалась
    с координатами необрезанной, и все рамки оверлея уезжали на левый верхний угол обрезки —
    от 15 до 24 px на разных вырезках. Одна функция на оба действия делает такую ошибку
    невыразимой.
    """
    box = table_bounds(grid, image.shape[:2], dpi)
    return image[box.slice], grid.shifted(-box.x0, -box.y0)
