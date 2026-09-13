"""Синтетические таблицы для тестов: линейки, объединения, боковой текст.

ЗАЧЕМ СИНТЕТИКА, ЕСЛИ ЕСТЬ ПАК. Затем, что на паке нет эталона в машинном виде: где именно
проходит разделитель и что написано в ячейке, знает только глаз. Синтетика знает это по
построению, и потому проверяет то, что на настоящей полосе проверить нечем: точность
разделителей, объединение по отсутствующей линейке, поворот на известный угол.

Числа здесь заданы В ПИКСЕЛЯХ 300 dpi — в том же разрешении, в котором пакет разбирает
вырезки, и с теми же кеглями (петит шапки около 20 px).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from research.legacy.table_processing.render.fonts import load_font

DPI = 300
PAPER = 245
INK = 25
RULE_PX = 3


@dataclass
class SyntheticTable:
    """Собранная таблица и то, что про неё известно по построению."""

    image: np.ndarray
    xs: list[int]
    ys: list[int]
    rotated: set[tuple[int, int]] = field(default_factory=set)
    texts: dict[tuple[int, int], str] = field(default_factory=dict)


def make_table(
    col_widths: "list[int] | None" = None,
    row_heights: "list[int] | None" = None,
    rotated: "dict[tuple[int, int], str] | None" = None,
    upright: "dict[tuple[int, int], str] | None" = None,
    missing_vertical: "set[tuple[int, int]] | None" = None,
    font_px: int = 20,
    margin: int = 30,
    skew_deg: float = 0.0,
    rule_px: int = RULE_PX,
    stray_rule_above: bool = False,
) -> SyntheticTable:
    """Линованная таблица с заданным содержимым.

    ``missing_vertical`` — пары (строка, разделитель), где вертикальная линейка НЕ рисуется:
    так получается объединённая ячейка, накрывающая две графы.

    ``rule_px`` — толщина линеек: в паке она гуляет от 2 до 9 px при 300 dpi, и внутренность
    ячейки обязана отступать от толстой линейки дальше, чем от тонкой.

    ``stray_rule_above`` — линейка над таблицей, какие ставят под заголовком страницы. Из
    неё не должно получиться лишней пустой строки сетки.
    """
    col_widths = col_widths or [220, 120, 120]
    row_heights = row_heights or [160, 90, 90]
    rotated = rotated or {}
    upright = upright or {}
    missing_vertical = missing_vertical or set()

    width = margin * 2 + sum(col_widths)
    height = margin * 2 + sum(row_heights)
    canvas = Image.new("L", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)

    xs = [margin]
    for value in col_widths:
        xs.append(xs[-1] + value)
    ys = [margin]
    for value in row_heights:
        ys.append(ys[-1] + value)

    if stray_rule_above and margin > rule_px * 3:
        draw.rectangle([xs[0], rule_px, xs[-1], rule_px * 2 - 1], fill=INK)
    for y in ys:
        draw.rectangle([xs[0], y, xs[-1], y + rule_px - 1], fill=INK)
    for row in range(len(row_heights)):
        for index, x in enumerate(xs):
            if (row, index) in missing_vertical:
                continue
            draw.rectangle([x, ys[row], x + rule_px - 1, ys[row + 1]], fill=INK)

    font = load_font(font_px)
    for (row, column), text in upright.items():
        _draw_text(canvas, draw, xs, ys, row, column, text, font, rotate=False, rule_px=rule_px)
    for (row, column), text in rotated.items():
        _draw_text(canvas, draw, xs, ys, row, column, text, font, rotate=True, rule_px=rule_px)

    array = np.asarray(canvas)
    if skew_deg:
        import cv2

        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -skew_deg, 1.0)
        array = cv2.warpAffine(array, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=PAPER)
    return SyntheticTable(array, xs, ys, set(rotated), {**upright, **rotated})


def _draw_text(canvas, draw, xs, ys, row, column, text, font, rotate: bool, rule_px: int = RULE_PX) -> None:
    left, right = xs[column] + rule_px + 4, xs[column + 1] - 4
    top, bottom = ys[row] + rule_px + 4, ys[row + 1] - 4
    if rotate:
        # Текст набирается прямо на отдельном холсте и кладётся повёрнутым ПРОТИВ часовой —
        # так набраны боковые шапки в паке, и так его надо будет вернуть поворотом ПО часовой.
        strip = Image.new("L", (max(1, bottom - top), max(1, right - left)), PAPER)
        ImageDraw.Draw(strip).text((4, 2), text, fill=INK, font=font)
        canvas.paste(strip.rotate(90, expand=True), (left, top))
        return
    words = text.split()
    line, y = "", top
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=font) > (right - left) and line:
            draw.text((left, y), line, fill=INK, font=font)
            y += int(font.size * 1.2)
            line = word
        else:
            line = candidate
    if line:
        draw.text((left, y), line, fill=INK, font=font)


def make_frame(text_lines: int = 14, width: int = 520, margin: int = 30, font_px: int = 20) -> np.ndarray:
    """Рамка объявления: прямоугольник вокруг сплошного набора, без внутренних линеек.

    Самый частый ложный друг детектора: 15 из 23 размеченных не-таблиц — именно такие
    рамки (объявление, некролог, книжная обложка).
    """
    height = margin * 2 + int(font_px * 1.6 * text_lines)
    canvas = Image.new("L", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([margin, margin, width - margin, height - margin], outline=INK, width=RULE_PX)
    font = load_font(font_px)
    for index in range(text_lines):
        y = margin + 8 + int(index * font_px * 1.6)
        draw.text((margin + 10, y), "сплошной набор во всю ширину рамки объявления", fill=INK, font=font)
    return np.asarray(canvas)


def make_block_diagram(blocks: int = 6, width: int = 700, height: int = 420) -> np.ndarray:
    """Блок-схема: россыпь прямоугольников со стрелками, решётки не образует."""
    canvas = Image.new("L", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    font = load_font(14)
    step = width // blocks
    for index in range(blocks):
        x0 = 20 + index * step
        y0 = 60 + (index % 3) * 110
        draw.rectangle([x0, y0, x0 + step - 40, y0 + 70], outline=INK, width=RULE_PX)
        draw.text((x0 + 8, y0 + 25), f"блок {index + 1}", fill=INK, font=font)
        if index:
            draw.line([(x0 - 40, y0 + 35), (x0, y0 + 35)], fill=INK, width=2)
    return np.asarray(canvas)


def bend(image: np.ndarray, amplitude_px: float, periods: float = 1.0) -> np.ndarray:
    """Изогнуть картинку синусоидой по вертикали: сагитта известна по построению.

    Ровно то, чего не хватает для проверки выпрямителей: настоящая кривизна на паке
    измеряется, но не задаётся, а здесь известен и ответ.
    """
    import cv2

    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    shift = (amplitude_px * np.sin(2 * np.pi * periods * grid_x / max(1, width - 1))).astype(np.float32)
    return cv2.remap(image, grid_x, grid_y + shift, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def make_page(
    table: "SyntheticTable | None" = None,
    lines_above: int = 0,
    lines_below: int = 0,
    gap_px: int = 24,
    edge_rule: bool = False,
    width: int = 900,
    font_px: int = 20,
    margin: int = 40,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Полоса: абзацы набора, между ними таблица, по желанию — тень корешка у края.

    Отдаёт картинку и рамку таблицы по построению — именно то, чего не хватает, чтобы
    проверить «рамка не захватила чужой абзац» без разметки на паке.

    ``edge_rule`` рисует длинную тонкую вертикаль в 12 px от края кадра, ни с чем не
    пересекающуюся, — это модель тени корешка, из-за которой рамка растягивалась на
    пол-полосы (1968/01 с.31).
    """
    table_image = table.image if table is not None else np.full((1, 1), PAPER, np.uint8)
    step = int(font_px * 1.6)
    top = margin + lines_above * step + (gap_px if lines_above else 0)
    height = top + table_image.shape[0] + (gap_px if lines_below else 0) + lines_below * step + margin
    canvas = Image.new("L", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    font = load_font(font_px)
    for index in range(lines_above):
        draw.text((margin, margin + index * step), "строка сплошного набора над таблицей", fill=INK, font=font)
    for index in range(lines_below):
        y = top + table_image.shape[0] + gap_px + index * step
        draw.text((margin, y), "строка сплошного набора под таблицей", fill=INK, font=font)
    if edge_rule:
        draw.line([(12, margin), (12, height - margin)], fill=INK, width=RULE_PX)
    page = np.asarray(canvas).copy()
    left = margin
    if table is not None:
        page[top : top + table_image.shape[0], left : left + table_image.shape[1]] = table_image
    return page, (left, top, left + table_image.shape[1], top + table_image.shape[0])


def make_linked_blocks(
    blocks: int = 8, width: int = 1700, block_w: int = 300, block_h: int = 130, gap: int = 120, dashed_from: int = 6
) -> np.ndarray:
    """Блок-схема в два ряда: коробки с текстом, связи сплошные, а с ``dashed_from`` — пунктирные.

    Размеры — как у схем пака: коробка 25x11 мм, связь 10 мм (она длиннее порога линейки и
    потому пересекает коробки в связном ядре). Так проверяется, что схема берётся целиком и
    через пунктир: связные ядра линеек пунктир не сшивают (штрих короче линейки), это делает
    заливка по штриховой краске с дилатацией.
    """
    rows = 2
    per_row = (blocks + rows - 1) // rows
    height = 60 + rows * (block_h + gap) + 40
    canvas = Image.new("L", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    font = load_font(22)
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(blocks):
        row, column = divmod(index, per_row)
        x0 = 40 + column * (block_w + gap)
        y0 = 40 + row * (block_h + gap)
        boxes.append((x0, y0, x0 + block_w, y0 + block_h))
        draw.rectangle(boxes[-1], outline=INK, width=RULE_PX)
        # Две строки текста в коробке, как в схемах пака: по краске это схема, а не чертёж.
        draw.text((x0 + 12, y0 + 20), f"блок {index + 1} отдела", fill=INK, font=font)
        draw.text((x0 + 12, y0 + 60), "снабжения и сбыта", fill=INK, font=font)
    for index in range(1, blocks):
        x0, y0, x1, y1 = boxes[index]
        px0, py0, px1, py1 = boxes[index - 1]
        if index % per_row == 0:  # переход на второй ряд: вертикальная связь
            start, end = ((px0 + px1) // 2, py1), ((px0 + px1) // 2, y0)
        else:
            start, end = (px1, (py0 + py1) // 2), (x0, (y0 + y1) // 2)
        if index >= dashed_from:
            _dashed(draw, start, end)
        else:
            draw.line([start, end], fill=INK, width=2)
    return np.asarray(canvas)


def _dashed(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], dash: int = 8) -> None:
    (x0, y0), (x1, y1) = start, end
    length = max(abs(x1 - x0), abs(y1 - y0))
    for offset in range(0, length, dash * 2):
        t0, t1 = offset / max(1, length), min(length, offset + dash) / max(1, length)
        draw.line(
            [(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0), (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)], fill=INK, width=2
        )
