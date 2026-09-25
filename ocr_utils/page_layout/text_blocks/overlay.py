"""Оверлей разбора: огибающие блоков, края рядов и осевые кривые строк поверх страницы."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks.alignment import AlignKind, Alignment
from ocr_utils.page_layout.text_blocks.blocks import dilate_polygon
from ocr_utils.page_layout.text_blocks.leaders import inside_spans
from ocr_utils.page_layout.text_blocks.page import PageAnalysis

# Цвета BGR: огибающая — синяя, крупная огибающая — фиолетовая, оси строк — зелёные,
# найденные края рядов — оранжевые кружки, границы колонок — серые пунктиры.
COLOUR_ENVELOPE = (220, 90, 20)
# Справочная кромка по краске: тот же синий, но приглушённый — главная здесь полоса вокруг оси.
COLOUR_ENVELOPE_INK = (150, 170, 120)
# Границы блоков рисуются полупрозрачно: под ними должны читаться буквы.
ENVELOPE_ALPHA = 0.55
# Подсказки внешних детекторов: рамка-запрет (таблица, схема), область бокового текста и поле,
# где текста быть не должно (растр). Все три — заливкой, значит полупрозрачно.
COLOUR_BARRIER = (60, 90, 210)
COLOUR_SIDEWAYS = (170, 90, 40)
COLOUR_FORBIDDEN = (120, 120, 120)
# Ячейка таблицы как область разбора: их на полосе бывает под полторы сотни, поэтому только
# тонкий КОНТУР, а не заливка — иначе подсказка съест страницу.
COLOUR_CELL = (90, 160, 90)
COLOUR_CELL_SIDEWAYS = (170, 90, 40)
HINT_ALPHA = 0.30
COLOUR_COARSE = (200, 60, 200)
COLOUR_AXIS = (40, 170, 40)
COLOUR_POINT = (140, 140, 0)  # бирюзовый: рядом с оранжевой меткой оси оранжевый же неразличим
COLOUR_COLUMN = (170, 170, 170)
# Участок оси над точкой или запятой: там ось провисает к базовой линии, в меры формы строки он
# не входит и на оверлее рисуется отдельным цветом, чтобы провисание не принимали за дефект.
COLOUR_MARK = (0, 165, 255)
# Хвост короткой последней строки блока — ось, достроенная до линии отсечки по изгибу предыдущей
# строки (``blocks._tail_of``), и сама линия отсечки. Виртуальная, краски под ней нет, поэтому
# своим цветом: розовый на оверлее больше ничем не занят (красный — у раздутой границы).
COLOUR_TAIL = (180, 105, 255)
COLOUR_TEXT = (20, 20, 20)
# Вердикт выравнивания крупной надписью на блоке: по формату — зелёный, по одному краю — синий,
# по центру — фиолетовый, ни по одному — красный. ``ragged`` на картинке пишется как ``none`` (так просил пользователь; в
# JSON и CSV значение прежнее).
VERDICT_TEXT = {
    AlignKind.RAGGED: "none",
    AlignKind.LEFT: "left",
    AlignKind.RIGHT: "right",
    AlignKind.BOTH: "both",
    AlignKind.CENTER: "center",
}
VERDICT_COLOUR = {
    AlignKind.RAGGED: (0, 0, 220),
    AlignKind.LEFT: (200, 90, 20),
    AlignKind.RIGHT: (200, 90, 20),
    AlignKind.BOTH: (0, 140, 0),
    AlignKind.CENTER: (160, 60, 160),
}
# Подложка надписи полупрозрачная: под ней должен читаться текст полосы.
VERDICT_ALPHA = 0.7
# Раздутые границы: полсимвола — жёлтая, символ — красная, прочие доли — серая.
COLOUR_DILATE = {0.5: (0, 200, 255), 1.0: (40, 40, 220)}
COLOUR_DILATE_OTHER = (120, 120, 120)


def _polyline(
    canvas: np.ndarray, points: np.ndarray, colour, thickness: int, scale: float, closed: bool = False
) -> None:
    """Ломаная поверх холста с масштабом ``scale`` (рабочая копия → холст)."""
    if points is None or len(points) < 2:
        return
    pts = np.round(np.asarray(points, dtype=np.float64) * scale).astype(np.int32)
    cv2.polylines(canvas, [pts], closed, colour, thickness, cv2.LINE_AA)


def draw(
    analysis: PageAnalysis, gray: np.ndarray, scale: float = 1.0, dilate_extra: tuple[float, ...] = (), hints=None
) -> np.ndarray:
    """Нарисовать разбор поверх серого изображения страницы.

    Args:
        analysis: Результат разбора (координаты — в пикселях рабочей копии).
        gray: Серое изображение страницы, в котором рисуем.
        scale: Пикселей изображения на пиксель рабочей копии.
        dilate_extra: Доли размера символа, на которые дополнительно раздуть границу блока и
            нарисовать её же: так видно, как меняется контур от дилатации.

    Returns:
        Цветной холст BGR.
    """
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if hints is not None:
        _hints(canvas, hints, scale)
    # Границы блоков — в отдельный слой: его подмешают полупрозрачно, чтобы буквы читались.
    layer = canvas.copy()
    for block in analysis.blocks:
        if block.envelope_ink is not None:
            _polyline(layer, block.envelope_ink.polygon, COLOUR_ENVELOPE_INK, 2, scale, closed=True)
        _polyline(layer, block.envelope.polygon, COLOUR_ENVELOPE, 2, scale, closed=True)
    cv2.addWeighted(layer, ENVELOPE_ALPHA, canvas, 1.0 - ENVELOPE_ALPHA, 0, canvas)
    for gutter in analysis.gutters:
        # Межколонник — ломаная: на трапеции он уезжает вбок вместе с колонками.
        for side in (1, 2):
            points = np.array([[point[side], point[0]] for point in gutter.points], dtype=np.float64)
            _polyline(canvas, points, COLOUR_COLUMN, 1, scale)
    for axis in analysis.axes:
        _axis_line(canvas, axis, scale)
    for block, alignment in zip(analysis.blocks, analysis.alignments):
        _polyline(canvas, block.envelope_coarse.left, COLOUR_COARSE, 1, scale)
        _polyline(canvas, block.envelope_coarse.right, COLOUR_COARSE, 1, scale)
        glyph = block.glyph_size
        for share in dilate_extra:
            outline = dilate_polygon(block.envelope.polygon, share * glyph[0], share * glyph[1], block.dpi)
            _polyline(canvas, outline, COLOUR_DILATE.get(share, COLOUR_DILATE_OTHER), 1, scale, closed=True)
        for row in block.rows:
            if row.tail is not None:
                # Хвост продолжает ось той же толщиной; отсечка — тонкой линией того же цвета.
                _polyline(canvas, row.tail, COLOUR_TAIL, 2, scale)
                _polyline(canvas, row.cut, COLOUR_TAIL, 1, scale)
            for x in (row.x0, row.x1):
                cv2.circle(canvas, (int(x * scale), int(row.y * scale)), 3, COLOUR_POINT, 1, cv2.LINE_AA)
        _caption(canvas, block, alignment, scale)
        draw_verdict(canvas, block.envelope.polygon * scale, alignment.kind)
    return canvas


def _hints(canvas: np.ndarray, hints, scale: float) -> None:
    """Подсказки подложкой: запретное поле, рамки таблиц и схем, области бокового текста, ячейки.

    Области и заливки — полупрозрачно: под подсказкой должны читаться и буквы, и сам разбор.
    Ячейки таблиц рисуются тонким КОНТУРОМ: их на полосе бывает под полторы сотни, и залитые
    прямоугольники съели бы страницу.
    """
    layer = canvas.copy()
    cells = [zone for zone in hints.zones if _is_cell(zone, hints)]
    cell_boxes = {zone.box for zone in cells}
    if hints.text_allowed is not None and not hints.text_allowed.all():
        forbidden = cv2.resize(
            (~hints.text_allowed).astype(np.uint8), (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        layer[forbidden > 0] = COLOUR_FORBIDDEN
    for box in hints.barriers:
        # Рамку ячейки заливкой не даём: её рисует контур ниже, а полторы сотни заливок сольются.
        if _matches(box, cell_boxes):
            continue
        cv2.rectangle(layer, _at(box[0], box[1], scale), _at(box[2], box[3], scale), COLOUR_BARRIER, -1)
    page = (0, 0, canvas.shape[1] / max(scale, 1e-6), canvas.shape[0] / max(scale, 1e-6))
    for zone in hints.zones:
        if zone in cells:
            colour = COLOUR_CELL_SIDEWAYS if zone.sideways else COLOUR_CELL
            cv2.rectangle(layer, _at(zone.box[0], zone.box[1], scale), _at(zone.box[2], zone.box[3], scale), colour, 1)
        elif zone.sideways and not _inside(page, zone.box):
            cv2.rectangle(
                layer, _at(zone.box[0], zone.box[1], scale), _at(zone.box[2], zone.box[3], scale), COLOUR_SIDEWAYS, -1
            )
    cv2.addWeighted(layer, HINT_ALPHA, canvas, 1.0 - HINT_ALPHA, 0, canvas)


def _is_cell(zone, hints) -> bool:
    """Ячейка таблицы среди областей: её внутренность лежит внутри одной из рамок-запретов."""
    return any(_inside(zone.box, box) for box in hints.barriers)


def _inside(box, outer) -> bool:
    """Лежит ли бокс внутри другого (допуск в пиксель на округление координат)."""
    return box[0] >= outer[0] - 1 and box[1] >= outer[1] - 1 and box[2] <= outer[2] + 1 and box[3] <= outer[3] + 1


def _matches(box, boxes) -> bool:
    """Совпадает ли рамка-запрет с боксом какой-нибудь ячейки (с точностью до линеек)."""
    return any(abs(box[0] - own[0]) <= 4 and abs(box[2] - own[2]) <= 4 and abs(box[1] - own[1]) <= 4 for own in boxes)


def _at(x: float, y: float, scale: float) -> tuple[int, int]:
    """Точка рабочей копии в координатах холста."""
    return int(round(x * scale)), int(round(y * scale))


def _axis_line(canvas: np.ndarray, axis, scale: float) -> None:
    """Ось строки: участки над точками и запятыми — своим цветом.

    Такой участок провисает к базовой линии (знак стоит на ней, а не на оси строки), поэтому он
    и выделяется: в меры наклона и формы строки он не входит, и принимать его за дефект не надо.
    """
    points = np.asarray(axis.points, dtype=np.float64)
    spans = list(getattr(axis, "mark_spans", ()) or ())
    if not spans or points.shape[0] < 2:
        _polyline(canvas, points, COLOUR_AXIS, 2, scale)
        return
    # Звено считается «над меткой», если под метку попадает любой из его концов.
    marked = inside_spans(points[:, 0], spans)
    start = 0
    for index in range(1, points.shape[0]):
        own = bool(marked[index - 1] or marked[index])
        last = index == points.shape[0] - 1
        following = bool(marked[index] or marked[index + 1]) if not last else not own
        if own != following or last:
            piece = points[start : index + 1]
            _polyline(canvas, piece, COLOUR_MARK if own else COLOUR_AXIS, 2, scale)
            start = index
    for x0, x1 in spans:
        middle = (x0 + x1) / 2.0
        y = float(np.interp(middle, points[:, 0], points[:, 1]))
        cv2.circle(canvas, (int(middle * scale), int(y * scale)), 3, COLOUR_MARK, -1, cv2.LINE_AA)


def draw_verdict(canvas: np.ndarray, polygon: np.ndarray, kind: AlignKind) -> None:
    """Крупная надпись вердикта выравнивания (none / left / right / both) посередине блока.

    Args:
        canvas: Холст.
        polygon: Контур блока в пикселях ХОЛСТА.
        kind: Вердикт.
    """
    text = VERDICT_TEXT[kind]
    font, size, weight = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (w, h), base = cv2.getTextSize(text, font, size, weight)
    x = int((polygon[:, 0].min() + polygon[:, 0].max()) / 2.0 - w / 2.0)
    y = int((polygon[:, 1].min() + polygon[:, 1].max()) / 2.0 + h / 2.0)
    x0, y0, x1, y1 = (
        max(0, x - 5),
        max(0, y - h - 5),
        min(canvas.shape[1], x + w + 5),
        min(canvas.shape[0], y + base + 3),
    )
    if x1 <= x0 or y1 <= y0:
        return
    # Полупрозрачная белая подложка, по ней — надпись цветом вердикта.
    patch = canvas[y0:y1, x0:x1]
    cv2.addWeighted(np.full_like(patch, 255), VERDICT_ALPHA, patch, 1.0 - VERDICT_ALPHA, 0, patch)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), VERDICT_COLOUR[kind], 1)
    cv2.putText(canvas, text, (x, y), font, size, VERDICT_COLOUR[kind], weight, cv2.LINE_AA)


def _caption(canvas: np.ndarray, block, alignment: Alignment, scale: float) -> None:
    """Подпись блока: колонка, строки, шаг, выключка и меры кромок."""
    lines = [
        f"кол.{block.column}.{block.index}: строк {block.lines}, шаг {block.pitch_mm:.1f} мм, выключка {alignment.kind.value}",
        f"L core {alignment.left.core_share:.2f} dev {alignment.left.envelope_dev_mm:.2f} "
        f"bend {alignment.left.bend_mm:.2f} отступов {alignment.left.indent_rows}",
        f"R core {alignment.right.core_share:.2f} dev {alignment.right.envelope_dev_mm:.2f} "
        f"bend {alignment.right.bend_mm:.2f} отступов {alignment.right.indent_rows}",
    ]
    x = int(block.span[0] * scale) + 4
    y = max(30, int(block.envelope.polygon[:, 1].min() * scale) - 8 - 12 * len(lines))
    for i, text in enumerate(lines):
        # Подложка под подпись: на тексте страницы чёрные буквы иначе не читаются.
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.33, 1)
        cv2.rectangle(canvas, (x - 2, y + i * 12 - h - 2), (x + w + 2, y + i * 12 + 3), (255, 255, 255), -1)
        cv2.putText(canvas, text, (x, y + i * 12), cv2.FONT_HERSHEY_COMPLEX, 0.33, COLOUR_TEXT, 1, cv2.LINE_AA)


def _on_paper(colour: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    """Цвет, каким он ЛЯЖЕТ НА БУМАГУ при подмешивании с прозрачностью ``alpha``.

    Полупрозрачная линия на белой бумаге выглядит светлее своего цвета, и образец в легенде,
    нарисованный непрозрачно, сбивал бы с толку: в легенде насыщенный синий, на странице —
    блёклый. Поэтому образец смешивается с бумагой ровно так же, как сама линия.
    """
    return tuple(int(round(alpha * own + (1.0 - alpha) * 255)) for own in colour)


def _legend(canvas: np.ndarray, dilate_extra: tuple[float, ...], hints=None) -> None:
    """Легенда в правом верхнем углу: что означает каждый цвет.

    Печатается ВСЕГДА: на оверлее семь сущностей, и без легенды их не различить. Образцы
    полупрозрачных линий показываются такими же полупрозрачными (см. :func:`_on_paper`).
    """
    lines = [
        ("граница блока: полоса вокруг оси", COLOUR_ENVELOPE, ENVELOPE_ALPHA),
        ("граница блока по краске, справочно", COLOUR_ENVELOPE_INK, ENVELOPE_ALPHA),
        ("крупная огибающая", COLOUR_COARSE, 1.0),
        ("ось строки", COLOUR_AXIS, 1.0),
        ("ось над точкой, запятой", COLOUR_MARK, 1.0),
        ("хвост последней строки и отсечка", COLOUR_TAIL, 1.0),
        ("края рядов", COLOUR_POINT, 1.0),
        ("межколонник", COLOUR_COLUMN, 1.0),
        ("выравнивание both (по формату)", VERDICT_COLOUR[AlignKind.BOTH], 1.0),
        ("выравнивание left / right", VERDICT_COLOUR[AlignKind.LEFT], 1.0),
        ("выравнивание center (по центру)", VERDICT_COLOUR[AlignKind.CENTER], 1.0),
        ("выравнивание none (рваный набор)", VERDICT_COLOUR[AlignKind.RAGGED], 1.0),
    ]
    lines += [
        (f"граница +{share:g} символа", COLOUR_DILATE.get(share, COLOUR_DILATE_OTHER), 1.0) for share in dilate_extra
    ]
    if hints is not None and not hints.empty:
        lines.append(("подсказка: рамка таблицы, схемы", COLOUR_BARRIER, HINT_ALPHA))
        if any(zone.sideways for zone in hints.zones):
            lines.append(("подсказка: боковой текст", COLOUR_SIDEWAYS, HINT_ALPHA))
        if hints.text_allowed is not None:
            lines.append(("подсказка: текста быть не должно", COLOUR_FORBIDDEN, HINT_ALPHA))
        if any(_is_cell(zone, hints) for zone in hints.zones):
            lines.append(("подсказка: ячейка таблицы", COLOUR_CELL, HINT_ALPHA))
            lines.append(("подсказка: ячейка, текст боком", COLOUR_CELL_SIDEWAYS, HINT_ALPHA))
    width = 250
    x = canvas.shape[1] - width
    cv2.rectangle(canvas, (x - 6, 4), (canvas.shape[1] - 2, 10 + 14 * len(lines)), (255, 255, 255), -1)
    for index, (text, colour, alpha) in enumerate(lines):
        y = 16 + 14 * index
        cv2.line(canvas, (x, y - 3), (x + 18, y - 3), _on_paper(colour, alpha), 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (x + 22, y), cv2.FONT_HERSHEY_COMPLEX, 0.33, COLOUR_TEXT, 1, cv2.LINE_AA)


def write(
    analysis: PageAnalysis,
    gray300: np.ndarray,
    path: Path,
    width: int = 1400,
    dilate_extra: tuple[float, ...] = (),
    hints=None,
) -> Path:
    """Записать оверлей в файл, ужав страницу до ширины ``width``."""
    scale_page = width / gray300.shape[1]
    page = cv2.resize(gray300, (width, int(gray300.shape[0] * scale_page)), interpolation=cv2.INTER_AREA)
    canvas = draw(analysis, page, scale=width / analysis.width, dilate_extra=dilate_extra, hints=hints)
    header = f"{analysis.name} с.{analysis.page} [{analysis.variant}] движок {analysis.engine}"
    cv2.putText(canvas, header, (10, 18), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLOUR_TEXT, 1, cv2.LINE_AA)
    _legend(canvas, dilate_extra, hints)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return path


__all__ = ["draw", "write"]
