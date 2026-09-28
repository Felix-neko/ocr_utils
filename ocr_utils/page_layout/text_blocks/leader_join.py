"""Сращивание строк, сошедшихся на отточии: «подпись . . . . число» из двух осей над одними и теми же точками — в одну.

Ряд таблицы с отточием собирается по кускам так (1966/03 IMG_0131_2R, «В металлообработке . . . 4,2»):

1. Сцепка по зонам (:mod:`zones`) собирает «В металлообработке.» и отдельно «4,2»: между ними
   полтора-два сантиметра одиночных точек, мостом они не служат, а цепочку с отточием справа
   сцепка сознательно не наращивает ничем, кроме точек (``LinkVerdict.LEADER``, 1971/10 с.93).
2. :func:`segment.extend_with_leaders` продлевает левую строку по отточию до его конца.
3. Но последние точки отточия при смыкании RLSA прилипли к «4,2», и правая строка начинается с них.

Обе оси проходят над одними и теми же точками и остаются разными строками. Здесь такие пары
сращиваются, если выполнено всё сразу:

* правая строка начинается внутри продлённого хвоста левой, а в перекрытии у неё только точки
  того же отточия, по которому продлена левая (общий глиф), — букв в перекрытии нет;
* угол сращивания проходит ту же проверку, что у сцепки по зонам: зона каждого куска смотрит на
  общую точку отточия под углом не больше ``zones.ANGLE_LIMIT_DEG``; у ДЛИННОГО куска зона идёт по
  касательной к его концу (``zones.long_zones``), у КОРОТКОГО — по местному наклону строк соседних
  длинных кусков (``zones.secondary_zones``), как во втором круге сцепки;
* остальные запреты сцепки (``zones._verdict_of``: разный кегль, наклон соединения против местного,
  межколонник, черта), барьер-линейка и предохранитель по остатку оси (``zones._guard``) — те же,
  по ТЕКСТОВЫМ частям кусков (без точек отточия).

Запрет ``LEADER`` здесь обходится намеренно и только при общем глифе (решение пользователя
2026-09-28): точки уже принадлежат обеим строкам, и ряд таблицы — одна строка.

Координаты — пиксели рабочей копии, как у :class:`segment.Segment`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.page_layout.text_blocks.capsules import angle_at
from ocr_utils.page_layout.text_blocks.pieces import (
    ANCHOR_ABOVE_BASELINE_XH,
    AXIS_MIN_LETTERS,
    Piece,
    anchors_of,
    pitch_of,
)
from ocr_utils.page_layout.text_blocks.segment import LEADER_ABOVE_HEIGHTS, LEADER_BELOW_HEIGHTS, Segment
from ocr_utils.page_layout.text_blocks.zones import (
    ANGLE_LEVER_XH,
    ANGLE_LIMIT_DEG,
    LinkVerdict,
    Zone,
    _barrier_between,
    _guard,
    _verdict_of,
    long_zones,
    neighbour_slopes,
    secondary_zones,
)

# Точка отточия — глиф, центр которого накрыт отточием и лежит от его линии не дальше толщины
# точки (как в ``pieces._leader_dots``), но не меньше стольких пикселей.
DOT_TOLERANCE_PX = 2.0
# Общая точка отточия стоит на высоте хвоста левой строки не дальше стольких иксов полосы: шаг строк
# на корпусе пака-1 — около двух иксов, отточие своей строки — в пределах пары пикселей.
DOT_ROW_XH = 0.5


@dataclass(frozen=True)
class _Parts:
    """Строка, разобранная на текст и точки отточия.

    Args:
        text: Боксы глифов текста ``(n, 4)`` — ``x0, y0, x1, y1``, слева направо.
        dots: Боксы глифов — точек отточия, в том же виде.
        piece: Кусок сцепки по текстовым глифам (``None`` — текста нет).
    """

    text: np.ndarray
    dots: np.ndarray
    piece: Piece | None


def _is_dot(box: np.ndarray, leaders: list) -> bool:
    """Точка ли отточия этот глиф: его центр накрыт каким-нибудь отточием страницы на его высоте."""
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return any(
        leader.covers(cx) and abs(leader.y - cy) <= max(leader.thickness, DOT_TOLERANCE_PX) for leader in leaders
    )


def _piece_of(boxes: np.ndarray, page_x_h: float) -> Piece | None:
    """Кусок сцепки по боксам глифов — так же, как ``pieces.pieces_of`` строит его по буквам сгустка.

    Args:
        boxes: Глифы ``(n, 4)`` — ``x0, y0, x1, y1``.
        page_x_h: Высота строчной полосы: икс для кусков из одной-двух букв и подъём якоря.

    Returns:
        Кусок или ``None``, если глифов нет.
    """
    if boxes.shape[0] == 0:
        return None
    # Буквы в том виде, что у ``pieces.letters_of``: центр, ширина, высота.
    centred = np.column_stack(
        [
            (boxes[:, 0] + boxes[:, 2]) / 2.0,
            (boxes[:, 1] + boxes[:, 3]) / 2.0,
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        ]
    )
    x_h = float(np.median(centred[:, 3])) if boxes.shape[0] >= 3 else page_x_h
    x_h = max(x_h, 1.0)
    anchors, marks = anchors_of(centred, x_h, lift=ANCHOR_ABOVE_BASELINE_XH * page_x_h)
    order = np.argsort(anchors[:, 0])
    return Piece(
        blobs=(),
        anchors=anchors[order],
        sizes=centred[order, 2:4].copy(),
        marks=marks[order],
        x0=float(boxes[:, 0].min()),
        y0=float(boxes[:, 1].min()),
        x1=float(boxes[:, 2].max()),
        y1=float(boxes[:, 3].max()),
        x_h=x_h,
        letter_w=float(np.median(centred[:, 2])),
        leader_dots=0,
    )


def _parts_of(segment, leaders: list, page_x_h: float) -> _Parts:
    """Текст и точки отточия строки; кусок сцепки — по тексту."""
    glyphs = segment.glyphs if segment.glyphs is not None else np.zeros((0, 4))
    dot = np.array([_is_dot(box, leaders) for box in glyphs], dtype=bool)
    text = glyphs[~dot]
    return _Parts(text=text, dots=glyphs[dot], piece=_piece_of(text, page_x_h))


def _zone_toward(piece: Piece, right_side: bool, pitch: float, slope: float) -> Zone:
    """Зона куска в сторону соседа — та же, что у сцепки по зонам.

    У длинного куска (своя ось есть) — конус по касательной к концу (``zones.long_zones``), у
    короткого — вторичная зона по местному наклону строк (``zones.secondary_zones``).

    Args:
        piece: Кусок.
        right_side: ``True`` — зона вправо (кусок слева), ``False`` — влево.
        pitch: Межстрочный шаг страницы.
        slope: Местный наклон строк в месте куска (тангенс).

    Returns:
        Зона; в обоих случаях с проверкой угла (``checked``).
    """
    if piece.long and piece.letters >= AXIS_MIN_LETTERS:
        zones = long_zones(piece, 0, pitch)
    else:
        zones = secondary_zones(piece, 0, pitch, slope)
    return zones[1] if right_side else zones[0]


def angle_ok(piece: Piece, right_side: bool, pitch: float, slope: float, meeting: tuple[float, float]) -> bool:
    """Видит ли зона куска точку встречи под допустимым углом — та же проверка, что у сцепки по зонам.

    Зона — :func:`_zone_toward` (у длинного куска по касательной к концу, у короткого — по местному
    наклону строк), вершина угла отодвинута назад на ``zones.ANGLE_LEVER_XH`` икса куска.

    Args:
        piece: Кусок.
        right_side: ``True`` — кусок слева и смотрит вправо, ``False`` — наоборот.
        pitch: Межстрочный шаг страницы.
        slope: Местный наклон строк в месте куска (тангенс).
        meeting: Точка встречи ``(x, y)``.

    Returns:
        ``True`` — угол не больше ``zones.ANGLE_LIMIT_DEG``.
    """
    zone = _zone_toward(piece, right_side, pitch, slope)
    return angle_at(zone.capsule, meeting[0], meeting[1], ANGLE_LEVER_XH * piece.x_h) <= ANGLE_LIMIT_DEG


def _meeting_point(
    right: _Parts, left_segment, right_segment, leaders: list, page_x_h: float
) -> tuple[float, float] | None:
    """Точка встречи — общая точка отточия, поднятая к уровню оси строки; ``None`` — общей точки нет.

    Общая — точка правой строки в хвосте левой, лежащая на отточии левой строки по тому же правилу,
    что у :func:`segment.extend_with_leaders` (линия отточия — в полосе от чуть выше центра строки до
    её базовой линии), и стоящая на высоте ОБЕИХ строк не дальше ``DOT_ROW_XH`` икса: хвоста левой и
    начала текста правой (ось её куска у левого конца). Высота правой строки мерой полосы не годится:
    у куска «. . 2,2» её задают точки (3 px, 1966/03 IMG_0131_2R). Без высоты хватало совпадения
    по x: на 1971/10 с.93 «Кабели городские телефонные» срастались с «с любым числом» рядом ниже — на
    расстоянии в 17 см угол зоны сдвига на шаг строки почти не видит.

    Якорь точки — её низ, поднятый на ``ANCHOR_ABOVE_BASELINE_XH`` икса полосы, как у якорей кусков.
    Из нескольких общих точек берётся самая левая.
    """
    tail_x1 = float(left_segment.x1)
    for box in right.dots[np.argsort(right.dots[:, 0])]:
        cx = (box[0] + box[2]) / 2.0
        if not float(right_segment.x0) <= cx <= tail_x1:
            continue
        cy = (box[1] + box[3]) / 2.0
        own = [
            leader
            for leader in leaders
            if leader.covers(cx)
            and abs(leader.y - cy) <= max(leader.thickness, DOT_TOLERANCE_PX)
            and _in_band(leader.y, left_segment)
        ]
        if not own:
            continue
        y = float(box[3] - ANCHOR_ABOVE_BASELINE_XH * page_x_h)
        limit = DOT_ROW_XH * page_x_h
        if abs(y - float(np.interp(cx, left_segment.xs, left_segment.ys))) > limit:
            continue
        if abs(y - right.piece.y_at(right.piece.x0)) > limit:
            continue
        return float(cx), y
    return None


def _in_band(y: float, segment) -> bool:
    """Лежит ли линия отточия ``y`` в полосе строки, по которой её продлевает ``segment.extend_with_leaders``."""
    return segment.cy - LEADER_ABOVE_HEIGHTS * segment.height <= y <= segment.cy + LEADER_BELOW_HEIGHTS * segment.height


def _merge(left, right):
    """Слить две строки: ось левой до начала правой, дальше ось правой; глифы и метки — вместе."""
    keep = left.xs < right.x0
    glyphs = [part for part in (left.glyphs, right.glyphs) if part is not None and part.shape[0]]
    boxes = np.vstack(glyphs) if glyphs else None
    if boxes is not None:
        boxes = boxes[np.argsort(boxes[:, 0])]
    # Высота строки — у куска с большим числом глифов: у числа «4,2» она своя, а мерой строки
    # остаётся подпись.
    left_count = 0 if left.glyphs is None else left.glyphs.shape[0]
    right_count = 0 if right.glyphs is None else right.glyphs.shape[0]
    return Segment(
        x0=min(left.x0, right.x0),
        y0=min(left.y0, right.y0),
        x1=max(left.x1, right.x1),
        y1=max(left.y1, right.y1),
        height=left.height if left_count >= right_count else right.height,
        xs=np.concatenate([left.xs[keep], right.xs]),
        ys=np.concatenate([left.ys[keep], right.ys]),
        weights=np.concatenate([left.weights[keep], right.weights]),
        scale=left.scale,
        mark_spans=tuple(sorted(set(left.mark_spans) | set(right.mark_spans))),
        glyphs=boxes,
    )


def _joinable(
    left: _Parts,
    right: _Parts,
    left_segment,
    right_segment,
    leaders: list,
    slopes: tuple[float, float],
    pitch: float,
    page_x_h: float,
    scale,
    separators: list,
    rules: list | None,
    barriers,
    ink300: np.ndarray,
    k: float,
    crosses,
) -> float | None:
    """Можно ли срастить левую строку с правой, сошедшейся с ней на отточии.

    Args:
        left, right: Части строк (:class:`_Parts`).
        left_segment, right_segment: Сами строки (охват и ось).
        leaders: Отточия страницы.
        slopes: Местный наклон строк в месте текста левой и правой.
        pitch: Межстрочный шаг страницы.
        page_x_h: Высота строчной полосы.
        scale: Масштаб набора (``segment.SCALES[0]``).
        separators: Межколонники и вертикальные линейки.
        rules: Сплошные черты.
        barriers: Линейки-барьеры или ``None``.
        ink300, k: Краска рендера и масштаб рендера к рабочей копии (для проверки межколонника).
        crosses: Проверка межколонника (``segment._crosses``).

    Returns:
        Расхождение по высоте в точке встречи (чем меньше, тем лучше), если сращивать можно; ``None`` — нельзя.
    """
    if left.piece is None or right.piece is None or right.dots.shape[0] == 0:
        return None
    # Правая строка начинается в продлённом хвосте левой, правее её текста, и уходит дальше хвоста.
    tail_x0, tail_x1 = left.piece.x1, float(left_segment.x1)
    if not (tail_x0 < right_segment.x0 <= tail_x1 < right_segment.x1):
        return None
    # В перекрытии у правой строки — только точки отточия, букв нет.
    if right.piece.x0 <= tail_x1:
        return None
    meeting = _meeting_point(right, left_segment, right_segment, leaders, page_x_h)
    if meeting is None:
        return None
    # Угол — как у сцепки по зонам: обе зоны видят общую точку по ходу своей оси.
    if not (
        angle_ok(left.piece, True, pitch, slopes[0], meeting)
        and angle_ok(right.piece, False, pitch, slopes[1], meeting)
    ):
        return None
    verdict = _verdict_of(
        left.piece, right.piece, scale, separators, rules, ink300, k, crosses, (slopes[0] + slopes[1]) / 2.0
    )
    # Разный кегль у стыка при большом зазоре (``MIXED_GAP``) — запрет сцепки через ПУСТОТУ: строка
    # подписи и строка заголовка на одной высоте. Здесь зазор занят отточием, общим для обеих строк,
    # а число у стыка законно выше строчной (1966/03 IMG_0131_2R: «…металлообработке.» + «4,2»).
    if verdict not in (LinkVerdict.ACCEPTED, LinkVerdict.MIXED_GAP):
        return None
    if barriers is not None and not barriers.empty and _barrier_between(barriers, left.piece, right.piece):
        return None
    if not _guard(left.piece, right.piece, pitch):
        return None
    # Мера близости — расхождение точки встречи с хвостом левой строки по высоте.
    return abs(meeting[1] - float(np.interp(meeting[0], left_segment.xs, left_segment.ys)))


def join_on_leaders(
    segments: list,
    leaders: list,
    dpi: float,
    scale,
    separators: list,
    rules: list | None,
    barriers,
    ink300: np.ndarray,
    k: float,
    crosses,
) -> list:
    """Срастить строки, сошедшиеся на общем глифе отточия (см. модуль).

    Args:
        segments: Строки масштаба корпуса после :func:`segment.extend_with_leaders`.
        leaders: Отточия страницы (``leaders.Leader``).
        dpi: Разрешение рабочей копии.
        scale: Масштаб набора (``segment.SCALES[0]``).
        separators: Межколонники и вертикальные линейки ``(x0, x1, y0, y1)``.
        rules: Сплошные черты страницы.
        barriers: Линейки-барьеры (``barriers.BarrierLines``) или ``None``.
        ink300: Краска рендера.
        k: Во сколько раз рендер крупнее рабочей копии.
        crosses: Проверка межколонника (``segment._crosses``).

    Returns:
        Строки, где сросшиеся пары заменены одной строкой (цепочкой, если правая сама срастается дальше).
    """
    if not leaders or len(segments) < 2:
        return segments
    boxes = [s.glyphs for s in segments if s.glyphs is not None and s.glyphs.shape[0]]
    if not boxes:
        return segments
    # Икс полосы — медиана высот всех глифов, как у ``pieces.pieces_of``.
    everything = np.vstack(boxes)
    page_x_h = max(float(np.median(everything[:, 3] - everything[:, 1])), 1.0)
    out = list(segments)
    while True:
        parts = [_parts_of(segment, leaders, page_x_h) for segment in out]
        pieces = [part.piece for part in parts if part.piece is not None]
        owners = [index for index, part in enumerate(parts) if part.piece is not None]
        pitch = pitch_of(pieces, dpi)
        # Местный наклон строк — по текстовым кускам полосы, как во втором круге сцепки.
        slopes = np.zeros(len(out))
        slopes[owners] = neighbour_slopes(pieces)
        # Все годные пары прохода; сращиваются взаимно непересекающиеся, от лучших по высоте.
        found: list[tuple[float, int, int]] = []
        for left_index in range(len(out)):
            for right_index in range(len(out)):
                if right_index == left_index:
                    continue
                score = _joinable(
                    parts[left_index],
                    parts[right_index],
                    out[left_index],
                    out[right_index],
                    leaders,
                    (float(slopes[left_index]), float(slopes[right_index])),
                    pitch,
                    page_x_h,
                    scale,
                    separators,
                    rules,
                    barriers,
                    ink300,
                    k,
                    crosses,
                )
                if score is not None:
                    found.append((score, left_index, right_index))
        if not found:
            return out
        taken: set[int] = set()
        joined = []
        for _, left_index, right_index in sorted(found):
            if left_index in taken or right_index in taken:
                continue
            taken |= {left_index, right_index}
            joined.append(_merge(out[left_index], out[right_index]))
        # Сращённые строки встают вместо своих частей и могут срастись дальше на следующем проходе.
        out = [segment for index, segment in enumerate(out) if index not in taken] + joined


__all__ = ["angle_ok", "join_on_leaders"]
