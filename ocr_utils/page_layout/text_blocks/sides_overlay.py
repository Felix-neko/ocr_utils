"""Оверлеи сторон границы блока, выравнивания по ним и дополнительной линии вертикальных сторон."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks.alignment import CORE_SHARE, AlignKind, is_centered, verdict
from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from ocr_utils.page_layout.text_blocks.overlay import COLOUR_TAIL, COLOUR_TEXT, VERDICT_COLOUR, draw_verdict
from ocr_utils.page_layout.text_blocks.page import PageAnalysis
from ocr_utils.page_layout.text_blocks.sides import (
    AlignMethod,
    BlockSides,
    RowStatus,
    SideAlignment,
    SideKind,
    SidesMethod,
    aligned_segments,
    filled_side,
    side_alignment,
    side_measures,
    edge_axis,
    sides_of,
)

# Палитра (BGR) по навыку draw-overlay. Вертикальные стороны — синий главной границы блока,
# горизонтальные — оранжевый: на этих картинках точек и запятых нет, и цвет свободен. Угловые
# звенья — пунктиром того же цвета. Оси строк — приглушённо: здесь они фон, а не предмет.
COLOUR_VERTICAL = (220, 90, 20)
COLOUR_HORIZONTAL = (0, 165, 255)
COLOUR_AXIS_MUTED = (150, 205, 150)
COLOUR_RAY = (150, 150, 150)
# Выравнивание: выровненный участок стороны — зелёный, невыровненный — красный (как «принято /
# отвергнуто» у этапных оверлеев), отступ — оранжевый, кривая, от которой мерили, — фиолетовая.
COLOUR_ALIGNED = (0, 150, 0)
COLOUR_RAGGED = (0, 0, 220)
COLOUR_INDENT = (0, 165, 255)
COLOUR_CURVE = (200, 60, 200)
COLOUR_OTHER_SIDE = (170, 170, 170)
SIDE_THICKNESS = 3
# Пунктир угловых звеньев: штрих и промежуток в пикселях холста.
DASH_ON, DASH_OFF = 7, 5
# Дополнительная линия вертикальной стороны (:func:`sides.filled_side`) — синий вертикальной стороны
# (на оверлее «было / стало» он ничем не занят), толсто и полупрозрачно поверх тонкой исходной
# стороны: под линией должны читаться и буквы, и зелёно-красная сторона. Заплатки — пунктиром.
COLOUR_FILLED = COLOUR_VERTICAL
FILLED_THICKNESS = 5
# На оверлее выравнивания та же линия — подложкой под сторону толщиной SIDE_THICKNESS: шире её, чтобы
# синий край был виден по обе стороны от зелёного и красного.
UNDERLAY_THICKNESS = 9
FILLED_ALPHA = 0.55


def _scaled(points: np.ndarray, scale: float) -> np.ndarray:
    """Точки рабочей копии → целые пиксели холста."""
    return np.round(np.asarray(points, dtype=np.float64) * scale).astype(np.int32)


def _segment(canvas: np.ndarray, a: np.ndarray, b: np.ndarray, colour, thickness: int) -> None:
    """Отрезок между двумя точками холста."""
    cv2.line(canvas, tuple(int(v) for v in a), tuple(int(v) for v in b), colour, thickness, cv2.LINE_AA)


def _dashed_run(canvas: np.ndarray, points: np.ndarray, colour, thickness: int) -> None:
    """Ломаная пунктиром: штрихи отмеряются по длине вдоль неё, а не по звеньям.

    Args:
        canvas: Холст.
        points: Точки ломаной на холсте ``(n, 2)``.
        colour: Цвет.
        thickness: Толщина.
    """
    if len(points) < 2:
        return
    step = np.hypot(*np.diff(points.astype(np.float64), axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(step)])
    period = DASH_ON + DASH_OFF
    for start in np.arange(0.0, along[-1], period):
        stop = min(start + DASH_ON, along[-1])
        xs = np.interp([start, stop], along, points[:, 0])
        ys = np.interp([start, stop], along, points[:, 1])
        _segment(canvas, np.array([xs[0], ys[0]]), np.array([xs[1], ys[1]]), colour, thickness)


def _runs(flags: list) -> list[tuple[int, int]]:
    """Серии подряд идущих звеньев с одним и тем же ключом: пары ``(начало, конец)`` без конца."""
    out, start = [], 0
    for index in range(1, len(flags) + 1):
        if index == len(flags) or flags[index] != flags[start]:
            out.append((start, index))
            start = index
    return out


def _draw_sides(canvas: np.ndarray, sides: BlockSides, scale: float) -> None:
    """Контур блока по сторонам: вертикальные синим, горизонтальные оранжевым, углы пунктиром."""
    points = _scaled(sides.polygon, scale)
    n = len(points)
    # Пунктиром — углы и неуверенные концы верха и низа: ни те, ни другие не идут в меры сторон.
    keys = [(label.vertical, bool(flag)) for label, flag in zip(sides.labels, sides.excluded)]
    for start, stop in _runs(keys):
        vertical, corner = keys[start]
        run = points[[(index % n) for index in range(start, stop + 1)]]
        colour = COLOUR_VERTICAL if vertical else COLOUR_HORIZONTAL
        if corner:
            _dashed_run(canvas, run, colour, SIDE_THICKNESS)
        else:
            cv2.polylines(canvas, [run], False, colour, SIDE_THICKNESS, cv2.LINE_AA)
    if sides.rays is not None:
        for origin, hit in sides.rays:
            _segment(canvas, _scaled(origin, scale), _scaled(hit, scale), COLOUR_RAY, 1)
            cv2.circle(canvas, tuple(int(v) for v in _scaled(hit, scale)), 3, COLOUR_RAY, -1, cv2.LINE_AA)


def _draw_edge_extensions(canvas: np.ndarray, block, scale: float) -> None:
    """Продления коротких крайних строк (``sides.edge_axis``) розовым пунктиром: ось виртуальная.

    Args:
        canvas: Холст.
        block: Текстовый блок.
        scale: Масштаб «рабочая копия → холст».
    """
    for top in (True, False):
        edge = edge_axis(block, top)
        if edge is None or not edge.extended:
            continue
        for part in (edge.points[edge.points[:, 0] <= edge.real_x0], edge.points[edge.points[:, 0] >= edge.real_x1]):
            if len(part) >= 2:
                _dashed_run(canvas, _scaled(part, scale), COLOUR_TAIL, 2)


def _draw_axes(canvas: np.ndarray, analysis: PageAnalysis, scale: float) -> None:
    """Оси строк приглушённо — фон для сторон."""
    for axis in analysis.axes:
        cv2.polylines(canvas, [_scaled(axis.points, scale)], False, COLOUR_AXIS_MUTED, 1, cv2.LINE_AA)


def _caption(canvas: np.ndarray, anchor: np.ndarray, lines: list[str]) -> None:
    """Подпись на белой подложке над точкой ``anchor`` холста."""
    x = int(anchor[0]) + 2
    y = max(14, int(anchor[1]) - 6 - 11 * (len(lines) - 1))
    for index, text in enumerate(lines):
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.32, 1)
        top = y + index * 11
        cv2.rectangle(canvas, (x - 2, top - h - 2), (x + w + 2, top + 3), (255, 255, 255), -1)
        cv2.putText(canvas, text, (x, top), cv2.FONT_HERSHEY_COMPLEX, 0.32, COLOUR_TEXT, 1, cv2.LINE_AA)


def _legend_entries(lines: list[tuple]) -> list[LegendEntry]:
    """Строки легенды из ``(подпись, цвет, пунктир[, альфа])``; пунктирные сущности — пунктирным образцом.

    Args:
        lines: Строки ``(подпись, цвет, пунктир)`` или ``(подпись, цвет, пунктир, альфа)``: у
            полупрозрачной линии образец смешивается с бумагой той же альфой.

    Returns:
        Строки для :func:`ocr_utils.page_layout.overlay_frame.legend_strip`.
    """
    return [
        LegendEntry(text, colour, rest[0] if rest else 1.0, SampleStyle.DASHED if dashed else SampleStyle.LINE)
        for text, colour, dashed, *rest in lines
    ]


def _page(gray300: np.ndarray, analysis: PageAnalysis, width: int) -> tuple[np.ndarray, float]:
    """Холст страницы шириной ``width`` и масштаб «рабочая копия → холст»."""
    height = int(gray300.shape[0] * width / gray300.shape[1])
    page = cv2.resize(gray300, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR), width / analysis.width


def _framed(canvas: np.ndarray, analysis: PageAnalysis, what: str, lines: list[tuple]) -> np.ndarray:
    """Холст с шапкой (выпуск, полоса, вариант и что на картинке) и легендой — обе в полях, не на странице.

    Args:
        canvas: Холст страницы с разметкой.
        analysis: Разбор страницы (имя, полоса, вариант).
        what: Что нарисовано.
        lines: Строки легенды (см. :func:`_legend_entries`).

    Returns:
        Картинка BGR.
    """
    header = [f"{analysis.name} с.{analysis.page} [{analysis.variant}] {what}"]
    return framed(canvas, header, _legend_entries(lines))


def draw_sides(analysis: PageAnalysis, gray300: np.ndarray, method: SidesMethod, width: int) -> np.ndarray:
    """Оверлей разметки сторон одним методом.

    Args:
        analysis: Разбор страницы.
        gray300: Серый рендер страницы.
        method: Метод разметки.
        width: Ширина картинки.

    Returns:
        Холст BGR.
    """
    canvas, scale = _page(gray300, analysis, width)
    _draw_axes(canvas, analysis, scale)
    for block in analysis.blocks:
        sides = sides_of(block, method)
        _draw_sides(canvas, sides, scale)
        if method is not SidesMethod.FRAME:
            _draw_edge_extensions(canvas, block, scale)
        measures = {item.side: item for item in side_measures(sides, block.dpi)}
        left, right = measures[SideKind.LEFT], measures[SideKind.RIGHT]
        corners = np.mean([item.corner_share for item in measures.values()])
        _caption(
            canvas,
            _scaled(sides.polygon.min(axis=0), scale),
            [
                f"{block.column}.{block.index}: L {left.tilt_deg:+.2f} гр, изгиб {left.bend_mm:.2f} мм",
                f"R {right.tilt_deg:+.2f} гр, изгиб {right.bend_mm:.2f} мм, углы {corners:.0%}",
            ],
        )
    legend = [
        ("вертикальная сторона", COLOUR_VERTICAL, False),
        ("горизонтальная сторона", COLOUR_HORIZONTAL, False),
        ("угол, вертикальная часть", COLOUR_VERTICAL, True),
        ("угол или неуверенный конец, горизонтальная часть", COLOUR_HORIZONTAL, True),
        ("ось строки (фон)", COLOUR_AXIS_MUTED, False),
    ]
    if method is not SidesMethod.FRAME:
        legend.append(("продление крайней строки по полной", COLOUR_TAIL, True))
    if method is SidesMethod.RAYS:
        legend.append(("луч из конца крайней строки", COLOUR_RAY, False))
    return _framed(canvas, analysis, f"стороны: {method.value}", legend)


def draw_alignment(
    analysis: PageAnalysis, gray300: np.ndarray, method: AlignMethod, sides_method: SidesMethod, width: int
) -> np.ndarray:
    """Оверлей выравнивания по вертикальным сторонам одним методом.

    Под зелёно-красной стороной — подложкой дополнительная линия стороны (:func:`sides.filled_side`):
    широкая полупрозрачная синяя полоса, пунктир на заплатках.

    Args:
        analysis: Разбор страницы.
        gray300: Серый рендер страницы.
        method: Метод проверки выравнивания.
        sides_method: Каким методом размечены стороны (какие звенья вертикальные).
        width: Ширина картинки.

    Returns:
        Холст BGR.
    """
    canvas, scale = _page(gray300, analysis, width)
    _draw_axes(canvas, analysis, scale)
    # Первый проход — разметка и выравнивание каждого блока, а дополнительные линии сторон сразу в слой:
    # слой подмешивается ДО всего остального, и линия ложится подложкой под зелёно-красную сторону.
    layer = canvas.copy()
    prepared = []
    for block in analysis.blocks:
        sides = sides_of(block, sides_method)
        alignments = {}
        for side in (SideKind.LEFT, SideKind.RIGHT):
            alignment = side_alignment(block, side, method)
            if alignment is None:
                continue
            alignments[side] = alignment
            line = filled_side(sides, alignment, block)
            if line is not None:
                _draw_filled_line(layer, line, scale, UNDERLAY_THICKNESS)
        prepared.append((block, sides, alignments))
    cv2.addWeighted(layer, FILLED_ALPHA, canvas, 1.0 - FILLED_ALPHA, 0, canvas)
    for block, sides, alignments in prepared:
        points = _scaled(sides.polygon, scale)
        n = len(points)
        colours = [COLOUR_OTHER_SIDE] * n
        texts = [f"{block.column}.{block.index}:"]
        aligned = {}
        for side, alignment in alignments.items():
            # Сторона выровнена, если в выровненных сериях не меньше CORE_SHARE рядов — тот же
            # порог, что у вердикта ``alignment.py`` для доли рядов на кривой.
            aligned[side] = alignment.aligned_share >= CORE_SHARE
            flags = aligned_segments(sides, alignment, block)
            for index, label in enumerate(sides.labels):
                if label is side:
                    colours[index] = COLOUR_ALIGNED if flags[index] else COLOUR_RAGGED
            _draw_ends(canvas, alignment, scale)
            if alignment.curve is not None:
                cv2.polylines(canvas, [_scaled(alignment.curve, scale)], False, COLOUR_CURVE, 1, cv2.LINE_AA)
            texts.append(
                f"{'L' if side is SideKind.LEFT else 'R'} на кривой {alignment.on_share:.0%}, "
                f"в сериях {alignment.aligned_share:.0%}"
            )
        for start, stop in _runs(colours):
            run = points[[(index % n) for index in range(start, stop + 1)]]
            thick = 1 if colours[start] == COLOUR_OTHER_SIDE else SIDE_THICKNESS
            cv2.polylines(canvas, [run], False, colours[start], thick, cv2.LINE_AA)
        _caption(canvas, _scaled(sides.polygon.min(axis=0), scale), texts)
        if aligned:
            draw_verdict(canvas, sides.polygon * scale, _kind(aligned, block))
    return _framed(
        canvas,
        analysis,
        f"выравнивание: {method.value} (стороны: {sides_method.value})",
        [
            ("сторона: выровнено", COLOUR_ALIGNED, False),
            ("сторона: не выровнено", COLOUR_RAGGED, False),
            ("горизонтальная сторона", COLOUR_OTHER_SIDE, False),
            ("конец строки: на кривой", COLOUR_ALIGNED, False),
            ("конец строки: мимо", COLOUR_RAGGED, False),
            ("конец строки: отступ", COLOUR_INDENT, False),
            ("кривая, от которой мерили", COLOUR_CURVE, False),
            ("доп. линия стороны: по стороне", COLOUR_FILLED, False, FILLED_ALPHA),
            ("доп. линия стороны: заплатка PCHIP", COLOUR_FILLED, True, FILLED_ALPHA),
            ("вердикт both / left, right / center / none", VERDICT_COLOUR[AlignKind.BOTH], False),
        ],
    )


def _draw_filled_line(layer: np.ndarray, line, scale: float, thickness: int) -> None:
    """Дополнительная линия стороны в слой: сплошная по самой стороне, пунктир на заплатках.

    Args:
        layer: Слой, который потом подмешивается к холсту с альфой ``FILLED_ALPHA``.
        line: Линия стороны (``sides.FilledSide``).
        scale: Масштаб «рабочая копия → холст».
        thickness: Толщина линии, пиксели холста.
    """
    points = _scaled(line.points, scale)
    # Серии «настоящая сторона / заплатка»; заплатка берёт крайние точки соседей, чтобы линия не рвалась.
    for start, stop in _runs(list(line.filled)):
        run = points[max(0, start - 1) : stop + 1] if line.filled[start] else points[start:stop]
        if len(run) < 2:
            continue
        if line.filled[start]:
            _dashed_run(layer, run, COLOUR_FILLED, thickness)
        else:
            cv2.polylines(layer, [run], False, COLOUR_FILLED, thickness, cv2.LINE_AA)


def draw_filled(
    analysis: PageAnalysis, gray300: np.ndarray, method: AlignMethod, sides_method: SidesMethod, width: int
) -> np.ndarray:
    """Оверлей «исходная вертикальная сторона против дополнительной линии» одним методом выравнивания.

    Исходная сторона — тонко, зелёным и красным по выровненности (как на ``align_*``). Дополнительная
    линия (:func:`sides.filled_side`) — толсто и полупрозрачно: сплошная на настоящих участках стороны,
    пунктир на заплатках; отброшенные концы не рисуются, под ними видна красная исходная сторона.
    Подпись блока — наклон и изгиб всей стороны → дополнительной линии.

    Args:
        analysis: Разбор страницы.
        gray300: Серый рендер страницы.
        method: Метод выравнивания — по нему размечены выровненные участки.
        sides_method: Каким методом размечены стороны.
        width: Ширина картинки.

    Returns:
        Холст BGR.
    """
    canvas, scale = _page(gray300, analysis, width)
    _draw_axes(canvas, analysis, scale)
    # Дополнительные линии копятся в отдельном слое и подмешиваются одним разом в конце; подписи
    # печатаются уже после смешивания, иначе они бледнели бы вместе с линиями.
    layer = canvas.copy()
    captions = []
    for block in analysis.blocks:
        sides = sides_of(block, sides_method)
        points = _scaled(sides.polygon, scale)
        n = len(points)
        colours = [None] * n
        texts = [f"{block.column}.{block.index}:"]
        for side in (SideKind.LEFT, SideKind.RIGHT):
            alignment = side_alignment(block, side, method)
            if alignment is None:
                continue
            flags = aligned_segments(sides, alignment, block)
            for index, label in enumerate(sides.labels):
                if label is side:
                    colours[index] = COLOUR_ALIGNED if flags[index] else COLOUR_RAGGED
            line = filled_side(sides, alignment, block)
            if line is None:
                continue
            _draw_filled_line(layer, line, scale, FILLED_THICKNESS)
            # Шрифт Hershey не знает «→» и «°» — стрелка и градусы пишутся ASCII и словом.
            texts.append(
                f"{'L' if side is SideKind.LEFT else 'R'} наклон {line.raw_tilt_deg:+.2f} -> {line.tilt_deg:+.2f} гр., "
                f"изгиб {line.raw_bend_mm:.1f} -> {line.bend_mm:.1f} мм, линия {line.length_mm:.0f} мм"
            )
        # Исходная сторона — тонко и непрозрачно: одна-две пиксельные линии текст не закрывают. Она
        # рисуется и на холсте, и в слое, поэтому после смешивания остаётся непрозрачной, а под
        # дополнительной линией просвечивает.
        for start, stop in _runs(colours):
            if colours[start] is None:
                continue
            run = points[[(index % n) for index in range(start, stop + 1)]]
            for target in (canvas, layer):
                cv2.polylines(target, [run], False, colours[start], 1, cv2.LINE_AA)
        captions.append((_scaled(sides.polygon.min(axis=0), scale), texts))
    cv2.addWeighted(layer, FILLED_ALPHA, canvas, 1.0 - FILLED_ALPHA, 0, canvas)
    for anchor, texts in captions:
        _caption(canvas, anchor, texts)
    return _framed(
        canvas,
        analysis,
        f"доп. линия сторон: {method.value} (стороны: {sides_method.value})",
        [
            ("сторона: выровнено", COLOUR_ALIGNED, False),
            ("сторона: не выровнено", COLOUR_RAGGED, False),
            ("доп. линия: по стороне", COLOUR_FILLED, False, FILLED_ALPHA),
            ("доп. линия: заплатка PCHIP", COLOUR_FILLED, True, FILLED_ALPHA),
        ],
    )


def _kind(aligned: dict, block) -> AlignKind:
    """Вердикт стенда: выровненность сторон этим способом, центр — общей мерой ``alignment.is_centered``.

    Однострочный блок выключки не имеет — ``ragged`` (на оверлее ``none``), как и в ``alignment.py``.
    """
    if len(block.rows) < 2:
        return AlignKind.RAGGED
    centered = is_centered(list(block.rows), block.dpi)
    return verdict(aligned.get(SideKind.LEFT, False), aligned.get(SideKind.RIGHT, False), centered)


def _draw_ends(canvas: np.ndarray, alignment: SideAlignment, scale: float) -> None:
    """Точки концов строк у стороны по их положению."""
    colours = {RowStatus.ON: COLOUR_ALIGNED, RowStatus.OFF: COLOUR_RAGGED, RowStatus.INDENT: COLOUR_INDENT}
    for end in alignment.ends:
        cv2.circle(canvas, tuple(int(v) for v in _scaled(end.point, scale)), 3, colours[end.status], -1, cv2.LINE_AA)


def side_by_side(images: list[np.ndarray], titles: list[str]) -> np.ndarray:
    """Несколько картинок одной полосы рядом, с подписью метода над каждой.

    Args:
        images: Холсты (разной высоты — добиваются полем снизу).
        titles: Подписи.

    Returns:
        Склейка BGR.
    """
    tiles = []
    # Легенды у методов разной длины, поэтому картинки добиваются белым полем снизу до общей высоты.
    height = max(image.shape[0] for image in images)
    images = [
        np.vstack([image, np.full((height - image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)])
        for image in images
    ]
    for image, title in zip(images, titles):
        strip = np.full((26, image.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(strip, title, (10, 19), cv2.FONT_HERSHEY_COMPLEX, 0.6, COLOUR_TEXT, 1, cv2.LINE_AA)
        tiles.append(np.vstack([strip, image]))
        tiles.append(np.zeros((tiles[-1].shape[0], 8, 3), dtype=np.uint8))
    return np.hstack(tiles[:-1])


def write_all(
    analysis: PageAnalysis, gray300: np.ndarray, out_dir: Path, width: int, sides_method: SidesMethod
) -> None:
    """Все оверлеи полосы: по методу разметки, по методу выравнивания, дополнительные линии и склейки.

    Args:
        analysis: Разбор страницы.
        gray300: Серый рендер страницы.
        out_dir: Корень выкладки.
        width: Ширина одной картинки.
        sides_method: Разметка, по которой рисуется выравнивание.
    """
    quality = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    sides_images = []
    for method in SidesMethod:
        image = draw_sides(analysis, gray300, method, width)
        sides_images.append(image)
        path = out_dir / f"sides_{method.value}" / f"{analysis.key}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image, quality)
    align_images = []
    for method in AlignMethod:
        image = draw_alignment(analysis, gray300, method, sides_method, width)
        align_images.append(image)
        path = out_dir / f"align_{method.value}" / f"{analysis.key}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image, quality)
    fill_images = {}
    for method in AlignMethod:
        image = draw_filled(analysis, gray300, method, sides_method, width)
        fill_images[method] = image
        path = out_dir / f"fill_{method.value}" / f"{analysis.key}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image, quality)
    compare = out_dir / "compare"
    compare.mkdir(parents=True, exist_ok=True)
    # «Было / стало» по устойчивому методу: исходная раскраска стороны против дополнительной линии.
    robust = list(AlignMethod).index(AlignMethod.ROBUST)
    cv2.imwrite(
        str(compare / f"{analysis.key}_fill.jpg"),
        side_by_side([align_images[robust], fill_images[AlignMethod.ROBUST]], ["вся сторона", "доп. линия"]),
        quality,
    )
    cv2.imwrite(
        str(compare / f"{analysis.key}_sides.jpg"),
        side_by_side(sides_images, [method.value for method in SidesMethod]),
        quality,
    )
    cv2.imwrite(
        str(compare / f"{analysis.key}_align.jpg"),
        side_by_side(align_images, [method.value for method in AlignMethod]),
        quality,
    )


__all__ = ["draw_alignment", "draw_filled", "draw_sides", "side_by_side", "write_all"]
