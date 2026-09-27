"""Отладочные оверлеи по этапам построения осевой линии строки.

Разбор идёт длинной цепочкой преобразований — от бинаризации до сглаженной ломаной, — и когда
ось ведёт себя неправильно, по итоговому оверлею не видно, на каком шаге это случилось. Здесь
каждый шаг рисуется отдельной картинкой С ТОЧКАМИ, по которым он работает: сгустки, центры
кусков, звенья цепочек, отсчёты наклона поля, столбцы центра масс, узлы пересборки.

Модуль ничего не считает сам: он вызывает те же функции, что и рабочий разбор
(:mod:`ocr_utils.page_layout.text_blocks.segment`, :mod:`~.baselines`, :mod:`~.lines`), и показывает их
промежуточные значения. Единственное исключение — отбор компонент маски глифов: рабочая
``segment.component_mask`` отдаёт только принятые, а для картинки нужны и отвергнутые, поэтому
их разбор повторён здесь (те же пороги из ``Scale``).

Запуск: ``python -m ocr_utils.page_layout.text_blocks stages --pdf … --page … --out-dir …``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks import LINKING_DEFAULT, LINKING_ZONES, lines as axis_lines, segment as seg
from ocr_utils.page_layout.text_blocks.legacy_linking import baselines
from ocr_utils.page_layout.text_blocks.legacy_linking import linking as linking_module
from ocr_utils.page_layout.text_blocks.legacy_linking.linking import link_spans, merge_by_field
from ocr_utils.page_layout.text_blocks.columns import gutters_of, rule_separators, separators_for_segmentation
from ocr_utils.page_layout.text_blocks.leaders import flatten_axis, leaders_of, spans_at
from ocr_utils.page_layout import mm_to_px, px_to_mm
from ocr_utils.scan_markup.curved_lines.fitting import centreline, smooth_median

# Цвета (BGR) — одни и те же во всех картинках: принято, отвергнуто, вспомогательное, итог.
COLOUR_OK = (60, 170, 60)
COLOUR_BAD = (60, 60, 220)
COLOUR_HINT = (200, 140, 60)
COLOUR_MARK = (0, 190, 240)
COLOUR_AXIS = (40, 200, 40)
COLOUR_TEXT = (30, 30, 30)
COLOUR_CHAIN = (220, 90, 40)

# Ширина полного оверлея страницы и увеличение вырезки.
PAGE_WIDTH = 1700
CROP_ZOOM = 4
# Цвета зон поиска на отладочном оверлее (BGR) и их прозрачность.
ZONE_LONG = (60, 190, 60)
ZONE_PRIMARY = (220, 120, 40)
ZONE_SECOND = (0, 170, 255)
ZONE_ALPHA = 0.55
# Линия экстраполяции — ось зоны поиска от конца куска до дальнего края зоны.
COLOUR_REACH = (40, 40, 40)
# Соединения на этапе 9 по решению: принято и взаимно, принято, но не взаимно; отказы — группами.
COLOUR_LINK_MUTUAL = COLOUR_OK
COLOUR_LINK_ONE_WAY = (0, 165, 255)
COLOUR_LINK_GEOMETRY = (170, 170, 170)
COLOUR_LINK_SHAPE = COLOUR_BAD
COLOUR_LINK_FORBIDDEN = COLOUR_HINT
# Принятое соединение со ступенькой на стыке больше этой доли шага — вероятный перескок: рисуется
# толще и отдельным цветом.
STEP_PITCH_SHARE = 0.6
COLOUR_LINK_JUMP = (200, 0, 200)
# Легенда в правом верхнем углу вырезки (пиксели увеличенной вырезки).
LEGEND_WIDTH = 330
LEGEND_ROW = 16

# Размер вырезки вокруг разбираемой строки (пиксели рабочей копии): строка целиком плюс
# по строке сверху и снизу — видно, откуда ось может перескочить.
CROP_PAD_X = 20
CROP_PAD_Y = 34
# Поля холста одной строки (пиксели рендера) и его увеличение: подпись не должна ложиться
# на строку, а точки центр-линии должны быть различимы.
LINE_PAD_TOP = 34
LINE_PAD_BOTTOM = 10
LINE_ZOOM = 2


@dataclass(frozen=True)
class StagePicture:
    """Одна отладочная картинка: имя файла, заголовок для отчёта и путь."""

    name: str
    title: str
    path: Path
    note: str = ""


def _canvas(work: np.ndarray) -> np.ndarray:
    """Холст из рабочей копии: краска приглушена, чтобы разметка читалась поверх неё."""
    light = (255 - (255 - work.astype(np.int16)) * 0.45).astype(np.uint8)
    return cv2.cvtColor(light, cv2.COLOR_GRAY2BGR)


# Знаки, которых нет в векторном шрифте OpenCV (рисуются вопросительным знаком).
CAPTION_REPLACEMENTS = {
    "—": "-",
    "–": "-",
    "×": "x",
    "≥": ">=",
    "≤": "<=",
    "±": "+-",
    "∫": "int ",
    "θ": "a",
    "²": "2",
    "°": " град",
    "«": '"',
    "»": '"',
}


def _caption(canvas: np.ndarray, text: str) -> None:
    """Подпись в левом верхнем углу картинки."""
    for source, target in CAPTION_REPLACEMENTS.items():
        text = text.replace(source, target)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (255, 255, 255), -1)
    cv2.putText(canvas, text, (8, 18), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLOUR_TEXT, 1, cv2.LINE_AA)


def _save(canvas: np.ndarray, path: Path, width: int = PAGE_WIDTH) -> Path:
    """Сохранить картинку, ужав до ширины ``width``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if canvas.shape[1] != width:
        scale = width / canvas.shape[1]
        canvas = cv2.resize(canvas, (width, int(canvas.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return path


def _on_paper(colour: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    """Цвет полупрозрачной разметки, как он ложится на белую бумагу (для образца в легенде)."""
    return tuple(int(round(alpha * own + (1.0 - alpha) * 255)) for own in colour)


def _legend(canvas: np.ndarray, entries: list[tuple[str, tuple[int, int, int], float]]) -> None:
    """Легенда в правом верхнем углу: образец цвета и русская подпись на строку.

    Args:
        canvas: Картинка, на которой печатается легенда (меняется на месте, как и вся отрисовка).
        entries: ``(подпись, цвет BGR, прозрачность)``; образец смешивается с бумагой так же,
            как сама разметка.
    """
    if not entries:
        return
    x = max(0, canvas.shape[1] - LEGEND_WIDTH)
    bottom = 34 + LEGEND_ROW * len(entries)
    cv2.rectangle(canvas, (x - 6, 28), (canvas.shape[1] - 2, bottom), (255, 255, 255), -1)
    for index, (text, colour, alpha) in enumerate(entries):
        y = 44 + LEGEND_ROW * index
        cv2.line(canvas, (x, y - 4), (x + 22, y - 4), _on_paper(colour, alpha), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (x + 28, y), cv2.FONT_HERSHEY_COMPLEX, 0.4, COLOUR_TEXT, 1, cv2.LINE_AA)


def _save_crop(
    canvas: np.ndarray,
    crop: tuple[int, int, int, int],
    path: Path,
    zoom: int = CROP_ZOOM,
    legend: list[tuple[str, tuple[int, int, int], float]] | None = None,
) -> Path:
    """Сохранить вырезку ``(x0, y0, x1, y1)``, увеличив её в ``zoom`` раз.

    Args:
        canvas: Картинка страницы (подпись уже напечатана).
        crop: Вырезка в пикселях ``canvas``.
        path: Куда писать JPEG.
        zoom: Во сколько раз увеличить.
        legend: Легенда для вырезки (печатается после увеличения, чтобы читалась); ``None`` — без неё.

    Returns:
        Путь записанного файла.
    """
    x0, y0, x1, y1 = crop
    title = canvas[:26].copy()
    piece = canvas[max(0, y0) : y1, max(0, x0) : x1]
    piece = cv2.resize(piece, (piece.shape[1] * zoom, piece.shape[0] * zoom), interpolation=cv2.INTER_NEAREST)
    # Подпись страницы — в шапку вырезки: без неё на увеличенном куске не понять, что это за этап.
    head = np.full((26, piece.shape[1], 3), 255, dtype=np.uint8)
    width = min(piece.shape[1], title.shape[1])
    head[:, :width] = title[:, :width]
    piece = np.vstack([head, piece])
    _legend(piece, legend or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), piece, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return path


def _boxes_of(stats: np.ndarray, index: int) -> tuple[int, int, int, int]:
    """Бокс компоненты ``(x, y, ширина, высота)``."""
    return (
        int(stats[index, cv2.CC_STAT_LEFT]),
        int(stats[index, cv2.CC_STAT_TOP]),
        int(stats[index, cv2.CC_STAT_WIDTH]),
        int(stats[index, cv2.CC_STAT_HEIGHT]),
    )


def _centre_of(stats: np.ndarray, index: int) -> tuple[float, float]:
    """Центр бокса компоненты."""
    x, y, w, h = _boxes_of(stats, index)
    return x + w / 2.0, y + h / 2.0


def _glyph_verdicts(
    work: np.ndarray, scale: seg.Scale
) -> tuple[np.ndarray, list[tuple[tuple[int, int, int, int], str]]]:
    """Разбор маски глифов с ПРИЧИНАМИ отказа (повторяет ``segment.component_mask``).

    Args:
        work: Серая рабочая копия.
        scale: Масштаб набора.

    Returns:
        Пара ``(маска принятых, список (бокс, причина))``; причина ``""`` — компонента принята.
    """
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    mask = np.zeros(work.shape, dtype=np.uint8)
    out: list[tuple[tuple[int, int, int, int], str]] = []
    # Предел высоты «мостика» считается по тем же компонентам, что прошли основные фильтры.
    passed = [
        float(stats[index, cv2.CC_STAT_HEIGHT])
        for index in range(1, count)
        if scale.min_height <= stats[index, cv2.CC_STAT_HEIGHT] <= scale.max_height
        and (
            stats[index, cv2.CC_STAT_WIDTH] <= 5 * max(int(stats[index, cv2.CC_STAT_HEIGHT]), 1)
            or stats[index, cv2.CC_STAT_AREA]
            / max(1, int(stats[index, cv2.CC_STAT_WIDTH]) * int(stats[index, cv2.CC_STAT_HEIGHT]))
            < seg.LETTER_MAX_FILL
        )
    ]
    bridge_limit = seg.BRIDGE_HEIGHT_RATIO * float(np.median(passed)) if passed else float("inf")
    for index in range(1, count):
        x, y, width, height = _boxes_of(stats, index)
        fill = stats[index, cv2.CC_STAT_AREA] / max(1, width * height)
        if height < scale.min_height:
            reason = "низкая"
        elif height > scale.max_height:
            reason = "высокая"
        elif width > 5 * max(height, 1) and fill >= seg.LETTER_MAX_FILL:
            # Длинная И залитая — это черта; длинное, но рыхлое слипшееся слово заголовка
            # («ОБРАЗОВАНИЕ КАДРОВ», заполнение 0.3–0.5) остаётся буквами.
            reason = "черта"
        elif height > bridge_limit and height > seg.BRIDGE_ASPECT * max(width, 1e-6):
            # Мостик: две буквы соседних строк, соприкоснувшиеся по вертикали.
            reason = "мостик"
        else:
            reason = ""
            mask[labels == index] = 255
        out.append(((x, y, width, height), reason))
    return mask, out


def _worst_axis(analysis) -> tuple[int, int, int, int] | None:
    """Вырезка вокруг самой «непрямой» оси страницы: у неё максимален размах остатка от хорды.

    Именно такие оси и есть перескок на соседнюю строку, поэтому отладку удобно вести на них.
    """
    best = None
    for axis in analysis.axes:
        xs, ys = axis.points[:, 0], axis.points[:, 1]
        if xs[-1] - xs[0] < 100:
            continue
        chord = ys[0] + (ys[-1] - ys[0]) * (xs - xs[0]) / (xs[-1] - xs[0])
        drop = float(np.abs(ys - chord).max())
        if best is None or drop > best[0]:
            best = (drop, axis)
    if best is None:
        return None
    axis = best[1]
    return (
        int(axis.x0 - CROP_PAD_X),
        int(axis.points[:, 1].min() - CROP_PAD_Y),
        int(axis.x1 + CROP_PAD_X),
        int(axis.points[:, 1].max() + CROP_PAD_Y),
    )


def _chain_at(spans: list[list[int]], stats: np.ndarray, crop: tuple[int, int, int, int]) -> list[int] | None:
    """Самая длинная цепочка, попавшая в вырезку: на ней показываются этапы центр-линии."""
    x0, y0, x1, y1 = crop
    best = None
    for span in spans:
        centres = np.array([_centre_of(stats, index) for index in span])
        inside = ((centres[:, 0] >= x0) & (centres[:, 0] <= x1) & (centres[:, 1] >= y0) & (centres[:, 1] <= y1)).sum()
        if inside < max(2, 0.5 * len(span)):
            continue
        width = float(centres[:, 0].max() - centres[:, 0].min())
        if best is None or width > best[0]:
            best = (width, span)
    return None if best is None else best[1]


def _ink_of_chain(
    span: list[int], stats: np.ndarray, labels: np.ndarray, ink300: np.ndarray, k: float
) -> tuple[np.ndarray, int, int, float]:
    """Краска одной цепочки в пикселях рендера (повторяет подготовку в ``segment._segment_of``).

    Returns:
        ``(краска, x0, y0, высота строки)``; ``x0``/``y0`` — в пикселях рабочей копии.
    """
    members = stats[span]
    x0 = int(members[:, cv2.CC_STAT_LEFT].min())
    y0 = int(members[:, cv2.CC_STAT_TOP].min())
    x1 = int((members[:, cv2.CC_STAT_LEFT] + members[:, cv2.CC_STAT_WIDTH]).max())
    y1 = int((members[:, cv2.CC_STAT_TOP] + members[:, cv2.CC_STAT_HEIGHT]).max())
    height = float(np.median(members[:, cv2.CC_STAT_HEIGHT]))
    own = np.isin(labels[y0:y1, x0:x1], span).astype(np.uint8)
    crop = ink300[int(y0 * k) : int(y1 * k), int(x0 * k) : int(x1 * k)]
    own300 = cv2.resize(own, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    return crop & (own300 > 0), x0, y0, height


def _plot_points(
    canvas: np.ndarray, xs: np.ndarray, ys: np.ndarray, colour: tuple[int, int, int], radius: int = 1
) -> None:
    """Набор точек в координатах холста."""
    for x, y in zip(xs, ys):
        cv2.circle(canvas, (int(round(x)), int(round(y))), radius, colour, -1, cv2.LINE_AA)


def _build_segments(
    spans: list[list[int]],
    stats: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    ink300: np.ndarray,
    separators: list[tuple[int, int, int, int]],
    scale: seg.Scale,
    k: float,
    dpi: float,
    leaders: list,
    candidates: list[int],
) -> list[seg.Segment]:
    """Собрать строки из готовых цепочек — тот же хвост, что у ``segment._segments_at_scale.build``.

    Нужен, чтобы показать строки ПЕРВОГО прохода: рабочая функция отдаёт только итог обоих.

    Args:
        spans: Цепочки кусков.
        stats, labels, mask: Сгустки, карта компонент после смыкания и маска глифов.
        ink300: Краска рендера.
        separators: Запреты сцепки (по ним цепочка режется).
        scale: Масштаб набора.
        k: Во сколько раз рендер крупнее рабочей копии.
        dpi: Разрешение рабочей копии.
        leaders: Отточия страницы.
        candidates: Куски-кандидаты (по ним берётся типичная высота строки).

    Returns:
        Строки этого прохода.
    """
    typical_height = float(np.median([int(stats[index, cv2.CC_STAT_HEIGHT]) for index in candidates]))
    rule_height = mm_to_px(seg.RULE_MAX_HEIGHT_MM, dpi)
    min_length = mm_to_px(scale.min_length_mm, dpi)
    max_height = mm_to_px(scale.max_height_mm, dpi)
    out: list[seg.Segment] = []
    for whole in spans:
        for span in seg._split_at_gutters(whole, stats, separators, ink300, k):
            item = seg._segment_of(
                span,
                stats,
                labels,
                mask,
                ink300,
                scale,
                k,
                dpi,
                leaders,
                typical_height,
                rule_height,
                min_length,
                max_height,
            )
            if item is not None:
                out.append(item)
    return out


def _greedy_stages(
    work, stats, labels, mask, candidates, separators, leaders, scale, dpi, ink300, k, page_picture, crop_picture
) -> list[list[int]]:
    """Этап 6 прежнего хода: жадные цепочки кусков.

    Подробные этапы поля хода строк (якоря, отсчёты наклона, выпрямление, ограничитель,
    сшивание) убраны вместе с переездом поля в ``legacy_linking``: разбирать по шагам ход,
    который больше не рабочий, незачем. Сами цепочки показываются — по ним видно, где жадный
    ход уходит на соседнюю строку.
    """
    from ocr_utils.page_layout.text_blocks.legacy_linking.baselines import field_of
    from ocr_utils.page_layout.text_blocks.legacy_linking.linking import link_spans, merge_by_field

    spans = link_spans(stats, separators, scale, dpi, ink300, k)
    canvas = _canvas(work)
    for number, span in enumerate(spans):
        colour = COLOUR_CHAIN if number % 2 == 0 else (150, 60, 160)
        points = [(int(x), int(y)) for x, y in (_centre_of(stats, index) for index in span)]
        for first, second in zip(points[:-1], points[1:]):
            cv2.line(canvas, first, second, colour, 1, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, point, 2, colour, -1, cv2.LINE_AA)
    page_picture(
        canvas,
        6,
        "chains_greedy",
        "Прежний ход: жадные цепочки кусков",
        f"цепочек {len(spans)}, звеньев {sum(len(span) - 1 for span in spans)}",
    )
    crop_picture(canvas, 6, "chains_greedy", "Жадные цепочки")
    first = _build_segments(spans, stats, labels, mask, ink300, separators, scale, k, dpi, leaders, candidates)
    field = field_of(first, work.shape, k, dpi)
    if field is None:
        return spans
    guarded = link_spans(stats, separators, scale, dpi, ink300, k, field=field)
    return merge_by_field(guarded, stats, field, separators, scale, ink300, k, leaders, None)


def _link_colour(verdict, mutual: bool, jump: bool) -> tuple[tuple[int, int, int], int]:
    """Цвет и толщина соединения на этапе 9 по решению сцепки.

    Args:
        verdict: :class:`zones.LinkVerdict`.
        mutual: Принятое соединение взаимно ближайшее (пойдёт в слияние).
        jump: Ступенька на стыке больше ``STEP_PITCH_SHARE`` шага.

    Returns:
        ``(цвет BGR, толщина)``.
    """
    from ocr_utils.page_layout.text_blocks.zones import LinkVerdict

    if verdict is LinkVerdict.ACCEPTED:
        if jump:
            return COLOUR_LINK_JUMP, 2
        return (COLOUR_LINK_MUTUAL, 2) if mutual else (COLOUR_LINK_ONE_WAY, 1)
    if verdict in (LinkVerdict.ANGLE, LinkVerdict.SIDE):
        return COLOUR_LINK_GEOMETRY, 1
    if verdict in (LinkVerdict.HEIGHT, LinkVerdict.OVERLAP, LinkVerdict.SLOPE):
        return COLOUR_LINK_SHAPE, 1
    return COLOUR_LINK_FORBIDDEN, 1


def _junction_step(left, right) -> float:
    """Ступенька на стыке двух кусков: ордината оси правого в его начале минус левого в его конце."""
    return float(right.y_at(right.x0) - left.y_at(left.x1))


def _zone_stages(
    work,
    stats,
    labels,
    mask,
    candidates,
    separators,
    rules,
    leaders,
    scale,
    dpi,
    ink300,
    k,
    page_picture,
    crop_picture,
    min_long_pieces: int | None = None,
) -> list[list[int]]:
    """Этапы 6–12 сцепки по зонам: куски, их оси, зоны, соединения, круги слияния.

    ``min_long_pieces`` — от скольких длинных кусков делается второй круг (``None`` —
    ``zones.MIN_LONG_PIECES``; крупному масштабу текстовой полосы боевой разбор передаёт 0).
    """
    from ocr_utils.page_layout.text_blocks import zones as zn

    min_long_pieces = zn.MIN_LONG_PIECES if min_long_pieces is None else min_long_pieces
    from ocr_utils.page_layout.text_blocks.pieces import page_x_height, pieces_of, pitch_of

    pieces = pieces_of(stats, labels, mask, candidates, dpi, leaders)
    x_h = page_x_height(pieces, dpi)
    pitch = pitch_of(pieces, dpi)

    # --- 6. Куски: длинные и короткие, с буквами внутри ---------------------------------
    canvas = _canvas(work)
    long_count = 0
    marks = 0
    for piece in pieces:
        long_count += 1 if piece.long else 0
        colour = COLOUR_OK if piece.long else COLOUR_MARK
        cv2.rectangle(canvas, (int(piece.x0), int(piece.y0)), (int(piece.x1), int(piece.y1)), colour, 1)
        for (ax, ay), mark in zip(piece.anchors, piece.marks):
            marks += 1 if mark else 0
            cv2.circle(canvas, (int(ax), int(ay)), 1, COLOUR_BAD if mark else COLOUR_CHAIN, -1)
    page_picture(
        canvas,
        6,
        "pieces",
        "Куски строк: длинные (зелёные) и короткие, якоря букв",
        f"кусков {len(pieces)}, длинных {long_count}, низких меток {marks}; икс {x_h:.1f} px, "
        f"шаг строк {pitch:.1f} px; длинный — от {zn.LONG_REACH_XH * 0 + 2} букв и ширины "
        f"{2.2:.1f} икса",
    )
    crop_picture(canvas, 6, "pieces", "Куски и якоря их букв")

    # --- 7. Оси кусков и касательные концов ----------------------------------------------
    canvas = _canvas(work)
    for piece in pieces:
        if not piece.long:
            continue
        grid = np.linspace(piece.anchors[0, 0], piece.anchors[-1, 0], 12)
        coefficients, centre = piece.axis()
        curve = np.column_stack([grid, np.polyval(coefficients, grid - centre)]).astype(np.int32)
        cv2.polylines(canvas, [curve], False, COLOUR_AXIS, 1, cv2.LINE_AA)
        for at_start in (True, False):
            x, y, dx, dy = piece.tangent(at_start)
            length = zn.LONG_REACH_XH * piece.x_h
            cv2.line(canvas, (int(x), int(y)), (int(x + dx * length), int(y + dy * length)), COLOUR_CHAIN, 1)
    page_picture(
        canvas,
        7,
        "piece_axes",
        "Оси кусков (МНК по буквам) и касательные на концах",
        f"касательная берётся по крайним {5} буквам или крайней трети ширины",
    )
    crop_picture(canvas, 7, "piece_axes", "Оси кусков и касательные")

    # --- 8. Зоны поиска -------------------------------------------------------------------
    # Рисуются полупрозрачно и РАЗНЫМИ цветами: зелёным — зоны длинных кусков (от концов оси по
    # касательной), синим — первичные зоны коротких (свой охват, раздутый вверх сильнее, чем
    # вниз), оранжевым — вторичные зоны коротких (второй круг, по наклону соседних длинных).
    # Местный наклон строк — тот же, что берёт боевой круг сцепки (``zones._round``): по нему
    # направлены вторичные зоны и с ним сверяется наклон соединения.
    slopes = zn.neighbour_slopes(pieces)
    first_round = zn.zones_of(pieces, pitch, False, slopes)
    second_round = zn.zones_of(pieces, pitch, True, slopes)
    primary = {(zone.piece, zone.side, zone.capsule) for zone in first_round}
    layer = _canvas(work)
    counts = {"длинных": 0, "первичных": 0, "вторичных": 0}
    for zone in second_round:
        if zone.side == 0:
            colour, key = ZONE_PRIMARY, "первичных"
        elif (zone.piece, zone.side, zone.capsule) in primary:
            colour, key = ZONE_LONG, "длинных"
        else:
            colour, key = ZONE_SECOND, "вторичных"
        counts[key] += 1
        # Зона — конус: радиус растёт от конца куска к дальнему краю (``Capsule.radius_at``).
        # Рисуется кругами вдоль оси, чтобы было видно раскрытие.
        capsule = zone.capsule
        steps = max(2, int(capsule.length // 2) + 1)
        for along in np.linspace(0.0, 1.0, steps):
            x = capsule.ax + (capsule.bx - capsule.ax) * along
            y = capsule.ay + (capsule.by - capsule.ay) * along
            cv2.circle(layer, (int(round(x)), int(round(y))), max(1, int(round(capsule.radius_at(along)))), colour, -1)
    canvas = cv2.addWeighted(_canvas(work), 1.0 - ZONE_ALPHA, layer, ZONE_ALPHA, 0)
    # Линии экстраполяции: ось каждой направленной зоны от конца куска, с точкой на дальнем краю.
    for zone in second_round:
        if zone.side == 0:
            continue
        capsule = zone.capsule
        end = (int(round(capsule.bx)), int(round(capsule.by)))
        cv2.line(canvas, (int(round(capsule.ax)), int(round(capsule.ay))), end, COLOUR_REACH, 1, cv2.LINE_AA)
        cv2.circle(canvas, end, 1, COLOUR_REACH, -1)
    for piece in pieces:
        cv2.rectangle(canvas, (int(piece.x0), int(piece.y0)), (int(piece.x1), int(piece.y1)), (120, 120, 120), 1)
    page_picture(
        canvas,
        8,
        "zones",
        "Зоны поиска: зелёные — длинных кусков, синие — первичные коротких, оранжевые — вторичные",
        ", ".join(f"{key} {value}" for key, value in counts.items())
        + f"; длина зоны длинного {zn.LONG_REACH_XH} икса, радиус {zn.LONG_RADIUS_XH} икса; "
        f"у коротких вверх {zn.SHORT_UP_XH}, вниз {zn.SHORT_DOWN_XH} икса; "
        f"вторичная {zn.SECOND_REACH_PITCHES} шага, радиус {zn.SECOND_RADIUS_XH} икса",
    )
    crop_picture(
        canvas,
        8,
        "zones",
        "Зоны поиска",
        legend=[
            ("зона длинного куска (конус)", ZONE_LONG, ZONE_ALPHA),
            ("первичная зона короткого", ZONE_PRIMARY, ZONE_ALPHA),
            ("вторичная зона короткого", ZONE_SECOND, ZONE_ALPHA),
            ("линия экстраполяции (ось зоны)", COLOUR_REACH, 1.0),
            ("бокс куска", (120, 120, 120), 1.0),
        ],
    )
    zone_list = second_round if long_count >= min_long_pieces else first_round

    # --- 9. Вероятные соединения ----------------------------------------------------------
    # Все встречи зон с решением сцепки (те же проверки, что в боевом круге). Рисуются по осям
    # кусков: от конца левого к началу правого, — так ступенька на стыке видна как наклон отрезка.
    verdicts = zn.link_verdicts(zone_list, pieces, scale, separators, rules, ink300, k, slopes)
    accepted = zn.accept(zn.links_of(zone_list, pieces, scale, separators, rules, ink300, k, slopes), len(pieces))
    accepted_keys = {(link.left, link.right) for link in accepted}
    canvas = _canvas(work)
    counts_verdict: dict[str, int] = {}
    jumps = 0
    drawn: set[tuple[int, int, object]] = set()
    # Сначала отказы, потом принятые — чтобы принятые не прятались под отказами той же пары.
    ordered = sorted(verdicts, key=lambda item: item[1] is zn.LinkVerdict.ACCEPTED)
    for link, verdict in ordered:
        counts_verdict[verdict.value] = counts_verdict.get(verdict.value, 0) + 1
        if (link.left, link.right, verdict) in drawn:
            continue
        drawn.add((link.left, link.right, verdict))
        left, right = pieces[link.left], pieces[link.right]
        step = _junction_step(left, right)
        jump = verdict is zn.LinkVerdict.ACCEPTED and abs(step) > STEP_PITCH_SHARE * pitch
        jumps += 1 if jump and (link.left, link.right) in accepted_keys else 0
        colour, thickness = _link_colour(verdict, (link.left, link.right) in accepted_keys, jump)
        start = (int(left.x1), int(round(left.y_at(left.x1))))
        end = (int(right.x0), int(round(right.y_at(right.x0))))
        cv2.line(canvas, start, end, colour, thickness, cv2.LINE_AA)
    for piece in pieces:
        cv2.rectangle(canvas, (int(piece.x0), int(piece.y0)), (int(piece.x1), int(piece.y1)), (150, 150, 150), 1)
    page_picture(
        canvas,
        9,
        "links",
        "Встречи зон: принятые и отбитые по причине",
        f"встреч {len(verdicts)}, взаимных принятых {len(accepted)}, из них со ступенькой больше "
        f"{STEP_PITCH_SHARE} шага {jumps}; "
        + ", ".join(f"{key} {value}" for key, value in sorted(counts_verdict.items(), key=lambda item: -item[1])),
    )
    crop_picture(
        canvas,
        9,
        "links",
        "Встречи зон и решения",
        legend=[
            ("принято, взаимно ближайшее", COLOUR_LINK_MUTUAL, 1.0),
            ("принято, но не взаимно", COLOUR_LINK_ONE_WAY, 1.0),
            (f"принято, ступенька > {STEP_PITCH_SHARE} шага", COLOUR_LINK_JUMP, 1.0),
            ("отбито: угол подхода, сторона зоны", COLOUR_LINK_GEOMETRY, 1.0),
            ("отбито: кегль, наклон, друг над другом", COLOUR_LINK_SHAPE, 1.0),
            ("отбито: межколонник, черта, барьер", COLOUR_LINK_FORBIDDEN, 1.0),
        ],
    )

    # --- 10–12. Круги слияния -------------------------------------------------------------
    rounds = []
    for secondary in (False, True):
        if secondary and long_count < min_long_pieces:
            break
        for _ in range(zn.MAX_ROUNDS):
            pieces, joined = zn._round(pieces, pitch, secondary, scale, separators, rules, ink300, k)
            if not joined:
                break
            rounds.append((secondary, joined, len(pieces)))
    canvas = _canvas(work)
    for piece in pieces:
        colour = COLOUR_AXIS if len(piece.blobs) > 1 else COLOUR_HINT
        cv2.rectangle(canvas, (int(piece.x0), int(piece.y0)), (int(piece.x1), int(piece.y1)), colour, 1)
    first_stage = sum(joined for secondary, joined, _ in rounds if not secondary)
    second_stage = sum(joined for secondary, joined, _ in rounds if secondary)
    page_picture(
        canvas,
        10,
        "merged",
        "После сращивания: зелёным — куски, собранные из нескольких",
        f"кругов {len(rounds)}, слияний в первом круге {first_stage}, во втором {second_stage}; "
        f"кусков осталось {len(pieces)}",
    )
    crop_picture(canvas, 10, "merged", "Сросшиеся куски")
    return [list(piece.blobs) for piece in pieces]


def render(
    gray300: np.ndarray,
    out_dir: Path,
    name: str,
    page: int,
    variant: str,
    dpi: float = WORK_DPI,
    crop: tuple[int, int, int, int] | None = None,
    linking: str = LINKING_DEFAULT,
    scale_name: str = "корпус",
) -> list[StagePicture]:
    """Нарисовать все этапы построения оси для одной страницы.

    Args:
        gray300: Серый рендер страницы в ``RENDER_DPI``.
        out_dir: Куда класть картинки.
        name: Имя выпуска (для подписей).
        page: Номер полосы.
        variant: Вариант рендера (``geo``/``nogeo``).
        dpi: Разрешение рабочей копии.
        crop: Вырезка ``(x0, y0, x1, y1)`` в пикселях рабочей копии; по умолчанию — вокруг самой
            непрямой оси страницы.
        linking: Способ сцепки: ``zones`` — этапы 6–10 про зоны поиска, ``greedy`` — прежний ход.
        scale_name: Масштаб сегментации (``Scale.name``): ``корпус`` или ``крупный``. У крупного
            маска глифов берётся, как в боевом разборе, без краски корпусных строк
            (``segment._claimed_mask``).

    Returns:
        Список нарисованных картинок по порядку этапов.
    """
    from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
    from ocr_utils.page_layout.text_blocks.page import analyse_gray

    out: list[StagePicture] = []
    k = RENDER_DPI / dpi
    size = (max(1, round(gray300.shape[1] * dpi / RENDER_DPI)), max(1, round(gray300.shape[0] * dpi / RENDER_DPI)))
    work = cv2.resize(gray300, size, interpolation=cv2.INTER_AREA)
    threshold, _ = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink300 = gray300 <= threshold
    scale = next(item for item in seg.SCALES if item.name == scale_name)
    head = f"{name} с.{page} [{variant}] масштаб {scale.name}"

    # Готовый разбор страницы нужен, чтобы выбрать вырезку и показать итог.
    analysis = analyse_gray(gray300, InkEngine(linking=linking), name=name, page=page, variant=variant)
    crop = crop or _worst_axis(analysis) or (0, 0, work.shape[1], work.shape[0])

    def page_picture(canvas: np.ndarray, index: int, slug: str, title: str, note: str = "") -> None:
        _caption(canvas, f"{index:02d}. {title} — {head}")
        path = _save(canvas, out_dir / f"{index:02d}_{slug}.jpg")
        out.append(StagePicture(slug, title, path, note))

    def crop_picture(
        canvas: np.ndarray, index: int, slug: str, title: str, note: str = "", legend: list | None = None
    ) -> None:
        piece = canvas.copy()
        _caption(piece, f"{index:02d}. {title} — {head}")
        path = _save_crop(piece, crop, out_dir / f"{index:02d}_{slug}_zoom.jpg", legend=legend)
        out.append(StagePicture(f"{slug}_zoom", title + " (крупно)", path, note))

    # --- 1. Бинаризация -------------------------------------------------------------------
    binary = np.zeros(work.shape, dtype=np.uint8)
    binary[work <= threshold] = 255
    canvas = cv2.cvtColor(255 - binary, cv2.COLOR_GRAY2BGR)
    page_picture(canvas, 1, "binary", f"Бинаризация Otsu, порог {threshold:.0f}", f"порог {threshold:.0f}")
    crop_picture(canvas, 1, "binary", "Бинаризация Otsu")

    # --- 2. Маска глифов ------------------------------------------------------------------
    mask, verdicts = _glyph_verdicts(work, scale)
    min_long_pieces = None
    if scale is not seg.SCALES[0]:
        # Крупный масштаб в боевом разборе не видит краски корпусных строк основного кегля.
        separators_body = separators_for_segmentation(gutters_of(work, dpi, leaders_of(work, dpi)[0]))
        separators_body += rule_separators(work, dpi)
        body = seg._segments_at_scale(
            gray300, work, ink300, separators_body, seg.SCALES[0], dpi, None, seg.rules_of(work, dpi), linking
        )
        claimed = seg._claimed_mask(seg._text_rows(body), work.shape[:2])
        if claimed is not None:
            min_long_pieces = seg._second_round_gate(mask, scale, dpi, None, None)
            mask = seg._without_claimed(mask, claimed)
    canvas = _canvas(work)
    counts: dict[str, int] = {}
    for (x, y, width, height), reason in verdicts:
        counts[reason or "принято"] = counts.get(reason or "принято", 0) + 1
        colour = COLOUR_OK if not reason else COLOUR_BAD
        cv2.rectangle(canvas, (x, y), (x + width, y + height), colour, 1)
    note = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items(), key=lambda item: -item[1]))
    page_picture(canvas, 2, "glyphs", "Маска глифов: принятые компоненты и отказы", note)
    crop_picture(canvas, 2, "glyphs", "Маска глифов")

    # --- 3. Смыкание RLSA -----------------------------------------------------------------
    smeared = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, scale.gap), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(smeared, 8)
    canvas = _canvas(work)
    canvas[smeared > 0] = (215, 215, 255)
    for index in range(1, count):
        x, y, width, height = _boxes_of(stats, index)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), COLOUR_HINT, 1)
    page_picture(
        canvas,
        3,
        "rlsa",
        f"Смыкание RLSA ядром 1×{scale.gap}: сгустки-куски строк",
        f"сгустков {count - 1}, ядро 1×{scale.gap} px",
    )
    crop_picture(canvas, 3, "rlsa", "Смыкание RLSA")

    # --- 4. Кандидаты ---------------------------------------------------------------------
    candidates = seg.candidates_of(stats, scale, dpi)
    chosen = set(candidates)
    canvas = _canvas(work)
    for index in range(1, count):
        x, y, width, height = _boxes_of(stats, index)
        colour = COLOUR_OK if index in chosen else COLOUR_BAD
        cv2.rectangle(canvas, (x, y), (x + width, y + height), colour, 1)
        if index in chosen:
            cx, cy = _centre_of(stats, index)
            cv2.circle(canvas, (int(cx), int(cy)), 1, COLOUR_CHAIN, -1)
    max_thickness = mm_to_px(scale.max_thickness_mm, dpi)
    page_picture(
        canvas,
        4,
        "candidates",
        "Кандидаты в куски строки и их центры",
        f"принято {len(candidates)} из {count - 1}; высота {scale.min_height}–{max_thickness:.0f} px, "
        f"ширина ≥ 0.5 высоты",
    )
    crop_picture(canvas, 4, "candidates", "Кандидаты и их центры")

    # --- 5. Разделители -------------------------------------------------------------------
    leaders, _ = leaders_of(work, dpi)
    gutters = gutters_of(work, dpi, leaders)
    gutter_bands = separators_for_segmentation(gutters)
    rule_bands = rule_separators(work, dpi)
    separators = gutter_bands + rule_bands
    canvas = _canvas(work)
    for x0, x1, y0, y1 in gutter_bands:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOUR_HINT, 2)
    for x0, x1, y0, y1 in rule_bands:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOUR_MARK, 2)
    for leader in leaders:
        cv2.line(canvas, (int(leader.x0), int(leader.y)), (int(leader.x1), int(leader.y)), COLOUR_BAD, 1)
    page_picture(
        canvas,
        5,
        "separators",
        "Запреты сцепки: межколонники, линейки таблицы, отточия",
        f"межколонников {len(gutter_bands)}, линеек {len(rule_bands)}, отточий {len(leaders)}",
    )

    if linking == LINKING_ZONES:
        spans = _zone_stages(
            work,
            stats,
            labels,
            mask,
            candidates,
            separators,
            seg.rules_of(work, dpi),
            leaders,
            scale,
            dpi,
            ink300,
            k,
            page_picture,
            crop_picture,
            min_long_pieces,
        )
    else:
        spans = _greedy_stages(
            work,
            stats,
            labels,
            mask,
            candidates,
            separators,
            leaders,
            scale,
            dpi,
            ink300,
            k,
            page_picture,
            crop_picture,
        )

    # --- 13–17. Центр-линия одной строки --------------------------------------------------
    chain = _chain_at(spans, stats, crop)
    if chain is not None:
        ink, x0, y0, height = _ink_of_chain(chain, stats, labels, ink300, k)
        typical = float(np.median([int(stats[index, cv2.CC_STAT_HEIGHT]) for index in candidates]))
        xs_raw, ys_raw, weights_raw = centreline(ink)
        xs_ref, ys_ref, weights_ref = seg.refined_centreline(ink, min(height, typical) * k)

        def line_canvas() -> np.ndarray:
            """Холст со строкой в пикселях рендера: сверху и снизу поля под подпись и соседей."""
            piece = (255 - ink.astype(np.uint8) * 110).astype(np.uint8)
            canvas = cv2.cvtColor(piece, cv2.COLOR_GRAY2BGR)
            return cv2.copyMakeBorder(
                canvas, LINE_PAD_TOP, LINE_PAD_BOTTOM, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )

        def line_picture(canvas: np.ndarray, index: int, slug: str, title: str, note: str = "") -> None:
            _caption(canvas, f"{index:02d}. {title} - {head}")
            canvas = cv2.resize(
                canvas, (canvas.shape[1] * LINE_ZOOM, canvas.shape[0] * LINE_ZOOM), interpolation=cv2.INTER_NEAREST
            )
            path = out_dir / f"{index:02d}_{slug}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            out.append(StagePicture(slug, title, path, note))

        canvas = line_canvas()
        _plot_points(canvas, xs_raw, ys_raw + LINE_PAD_TOP, COLOUR_BAD, 1)
        line_picture(
            canvas,
            13,
            "centreline_raw",
            "Центр масс краски по столбцам (первый проход)",
            f"столбцов {xs_raw.size}, высота строки {height:.0f} px рабочей копии " f"({px_to_mm(height, dpi):.1f} мм)",
        )

        canvas = line_canvas()
        anchor = smooth_median(ys_raw, max(3, int(seg.SMOOTH_HEIGHTS * min(height, typical) * k) | 1))
        half = max(2.0, seg.CENTRE_WINDOW_HEIGHTS * min(height, typical) * k)
        # Полоса окна кладётся полупрозрачно, иначе она закрывает сами буквы.
        band = canvas.copy()
        for x, centre in zip(xs_raw, anchor):
            top = int(centre - half) + LINE_PAD_TOP
            cv2.line(band, (int(x), top), (int(x), int(centre + half) + LINE_PAD_TOP), (235, 200, 120), 1)
        canvas = cv2.addWeighted(canvas, 0.6, band, 0.4, 0)
        _plot_points(canvas, xs_raw, anchor + LINE_PAD_TOP, COLOUR_HINT, 1)
        _plot_points(canvas, xs_ref, ys_ref + LINE_PAD_TOP, COLOUR_OK, 1)
        line_picture(
            canvas,
            14,
            "centreline_refined",
            "Второй проход: центр масс в окне вокруг сглаженной линии",
            f"полуокно {seg.CENTRE_WINDOW_HEIGHTS} высоты = {half:.0f} px рендера; сдвиг оси "
            f"медиана {np.median(np.abs(ys_ref - ys_raw)):.2f} px, максимум {np.abs(ys_ref - ys_raw).max():.2f} px",
        )

        own_spans = [((sx0 - x0) * k, (sx1 - x0) * k) for sx0, sx1 in spans_at(leaders, (y0 + height / 2.0), height)]
        xs_flat, ys_flat, weights_flat = flatten_axis(xs_ref, ys_ref, weights_ref, own_spans)
        smoothed = smooth_median(ys_flat, int(seg.SMOOTH_HEIGHTS * height * k))
        canvas = line_canvas()
        _plot_points(canvas, xs_ref, ys_ref + LINE_PAD_TOP, (200, 200, 200), 1)
        _plot_points(canvas, xs_flat, smoothed + LINE_PAD_TOP, COLOUR_AXIS, 1)
        line_picture(
            canvas,
            15,
            "smooth_median",
            f"Медианное сглаживание окном {seg.SMOOTH_HEIGHTS} высоты",
            f"окно {int(seg.SMOOTH_HEIGHTS * height * k)} px рендера; отточий на строке " f"{len(own_spans)}",
        )

        # Пересборка и окончательное сглаживание — уже в координатах рабочей копии.
        points = np.column_stack([xs_flat / k + x0, smoothed / k + y0])
        step = mm_to_px(axis_lines.AXIS_STEP_MM, dpi)
        resampled = axis_lines.resample(points, step)
        canvas = line_canvas()
        _plot_points(canvas, (points[:, 0] - x0) * k, (points[:, 1] - y0) * k, (200, 200, 200), 1)
        for x, y in resampled:
            cv2.drawMarker(
                canvas, (int((x - x0) * k), int((y - y0) * k)), COLOUR_CHAIN, cv2.MARKER_TILTED_CROSS, 6, 1, cv2.LINE_AA
            )
        line_picture(
            canvas,
            16,
            "resample",
            f"Пересборка с шагом {axis_lines.AXIS_STEP_MM:.0f} мм: медиана ординат в окне",
            f"было точек {points.shape[0]}, стало узлов {resampled.shape[0]} (шаг {step:.1f} px рабочей копии)",
        )

        window_px = max(axis_lines.SMOOTH_HEIGHTS * height, step * 3)
        final = axis_lines.smooth_axis(resampled, window_px)
        canvas = line_canvas()
        for x, y in resampled:
            cv2.drawMarker(
                canvas,
                (int((x - x0) * k), int((y - y0) * k) + LINE_PAD_TOP),
                (190, 190, 190),
                cv2.MARKER_TILTED_CROSS,
                6,
                1,
            )
        cv2.polylines(
            canvas,
            [np.column_stack([(final[:, 0] - x0) * k, (final[:, 1] - y0) * k + LINE_PAD_TOP]).astype(np.int32)],
            False,
            COLOUR_AXIS,
            2,
            cv2.LINE_AA,
        )
        chord = final[0, 1] + (final[-1, 1] - final[0, 1]) * (final[:, 0] - final[0, 0]) / max(
            final[-1, 0] - final[0, 0], 1e-6
        )
        cv2.line(
            canvas,
            (int((final[0, 0] - x0) * k), int((final[0, 1] - y0) * k) + LINE_PAD_TOP),
            (int((final[-1, 0] - x0) * k), int((final[-1, 1] - y0) * k) + LINE_PAD_TOP),
            COLOUR_BAD,
            1,
            cv2.LINE_AA,
        )
        bend = float(np.abs(final[:, 1] - chord).max())
        line_picture(
            canvas,
            17,
            "smooth_axis",
            f"Сглаживание оси: медиана + Савицкий–Голей окном {axis_lines.SMOOTH_HEIGHTS} высоты",
            f"окно {window_px:.1f} px рабочей копии; отклонение оси от хорды (красная) "
            f"{bend:.1f} px = {px_to_mm(bend, dpi):.2f} мм",
        )

    # --- 18. Ряды: слияние осей в точке встречи --------------------------------------------
    canvas = _canvas(work)
    pairs = 0
    for block in analysis.blocks:
        for row in block.rows:
            colour = COLOUR_AXIS if len(row.axes) == 1 else COLOUR_MARK
            for axis in row.axes:
                cv2.polylines(canvas, [axis.points.astype(np.int32)], False, colour, 1, cv2.LINE_AA)
            if len(row.axes) > 1:
                pairs += 1
                ordered = sorted(row.axes, key=lambda item: item.x0)
                for left, right in zip(ordered[:-1], ordered[1:]):
                    # Точка встречи: середина между ближними концами (или середина перекрытия).
                    meeting = (max(left.x0, right.x0) + min(left.x1, right.x1)) / 2.0
                    y_left, y_right = left.y_at(meeting), right.y_at(meeting)
                    cv2.line(
                        canvas, (int(meeting), int(y_left)), (int(meeting), int(y_right)), COLOUR_BAD, 2, cv2.LINE_AA
                    )
                    cv2.circle(canvas, (int(meeting), int((y_left + y_right) / 2)), 3, COLOUR_BAD, 1)
    page_picture(
        canvas,
        18,
        "rows",
        "Ряды: оси, слитые в один ряд в точке встречи",
        f"рядов из нескольких осей {pairs}; допуск {'{:.1f}'.format(0.5)} высоты, ординаты сверяются в точке "
        f"встречи с продолжением по наклону конца",
    )
    crop_picture(canvas, 18, "rows", "Слияние осей в ряды")

    # --- 19. Итог -------------------------------------------------------------------------
    canvas = _canvas(work)
    pitch = float(np.median([block.pitch_px for block in analysis.blocks])) if analysis.blocks else 0.0
    suspicious = 0
    for axis in analysis.axes:
        xs, ys = axis.points[:, 0], axis.points[:, 1]
        chord = ys[0] + (ys[-1] - ys[0]) * (xs - xs[0]) / max(xs[-1] - xs[0], 1e-6)
        drop = float(np.abs(ys - chord).max())
        bad = pitch > 0 and drop > 0.5 * pitch
        suspicious += 1 if bad else 0
        cv2.polylines(
            canvas,
            [axis.points.astype(np.int32)],
            False,
            COLOUR_BAD if bad else COLOUR_AXIS,
            2 if bad else 1,
            cv2.LINE_AA,
        )
        cv2.circle(canvas, (int(axis.x0), int(axis.points[0, 1])), 2, COLOUR_CHAIN, -1)
        cv2.circle(canvas, (int(axis.x1), int(axis.points[-1, 1])), 2, COLOUR_HINT, -1)
    x0, y0, x1, y1 = crop
    cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOUR_MARK, 2)
    page_picture(
        canvas,
        19,
        "axes",
        "Итог: оси строк; красным — уходящие от своей хорды дальше полушага",
        f"осей {len(analysis.axes)}, блоков {len(analysis.blocks)}, покрытие краски {analysis.ink_share:.1%}; "
        f"подозрительных осей {suspicious} (шаг строк {pitch:.1f} px); рамкой отмечена разобранная строка",
    )
    crop_picture(canvas, 19, "axes", "Итоговые оси")
    return out


__all__ = ["StagePicture", "render"]
