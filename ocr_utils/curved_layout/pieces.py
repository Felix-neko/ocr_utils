"""Куски строк: сгустки после смыкания RLSA, разобранные на буквы, со своей осью и мерами набора.

Главная мысль: у каждого куска есть СОБСТВЕННЫЙ ход, измеренный по буквам внутри него. Прежняя
сцепка вела цепочку по предсказанию от её последних звеньев, и на искажённой бумаге предсказание
дрейфовало — за пять звеньев ось уходила на соседнюю строку. Ось куска ни от чего не зависит:
сколько в нём букв, столько и свидетельств о его направлении.

Единица длины здесь — ВЫСОТА СТРОЧНОЙ БУКВЫ (``x_h``), а не высота бокса сгустка. Бокс включает
выносные элементы и надстрочные знаки и на корпусе пака-1 равен 13–18 px против 10 px у самой
буквы; порог «кусок длиннее двух своих высот», заданный по боксу, отправил бы в короткие
половину слов.

Якорь буквы — её БАЗОВАЯ ЛИНИЯ, поднятая на полвысоты строчной, а не центр бокса. Центр бокса
свой у каждого класса букв: у прописной и цифры он на полразницы высот выше, чем у строчной, а у
точки и запятой — ровно на базовой линии, то есть на полвысоты ниже оси. Кусок из двух-трёх букв
одного класса от этого уезжает целиком, и его соединение с соседом читается как перескок. Низ
буквы такого перекоса не даёт: вниз от базовой линии свисают только выносные элементы, и они
отсеиваются односторонним отсевом (``baselines_of``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.curved_layout import WORK_DPI
from ocr_utils.page_layout import mm_to_px, px_to_mm

# Длинный кусок: не меньше двух букв и шириной хотя бы в столько высот строчной. Два критерия
# вместе: дефис и тире тонкие, но широкие (отношение сторон 3), а буква у них одна — ось по ней
# бессмысленна. Порог 2.2 икса ≈ 22 px ≈ три буквы корпуса.
LONG_MIN_LETTERS = 2
LONG_ASPECT_XH = 2.2
# Низкая метка: точка, запятая, двоеточие. Ниже 0.6 икса и не шире 0.9 икса (дефис шире).
MARK_MAX_HEIGHT_XH = 0.6
MARK_MAX_WIDTH_XH = 0.9
# На сколько поднимается якорь от БАЗОВОЙ ЛИНИИ буквы: ось строки идёт посередине строчной.
ANCHOR_ABOVE_BASELINE_XH = 0.5
# Выносной элемент вниз («р», «у», «ф», «ц», «щ»): низ такой буквы лежит ниже базовой линии.
# Замер по трём трудным полосам (6579 букв в длинных сгустках): низы букв держатся базовой линии
# с точностью −1.4…+0.4 px по квартилям, и только 6.1 % уходят ниже двух пикселей — это и есть
# выносные, их медианная высота 14 px против 10 у остальных. Порог 0.2 икса = 2 px.
DESCENDER_MIN_XH = 0.2
# Поправка на выносные делается от стольких букв: на двух-трёх буквах отличить выносной элемент
# от наклона строки нечем.
DESCENDER_MIN_LETTERS = 4
# Парабола строится от стольких букв; меньше — прямая (по трём точкам парабола ловит форму буквы,
# а не ход строки).
AXIS_PARABOLA_LETTERS = 5
# Своей осью кусок пользуется от стольких букв: прямая по двум точкам — это наклон пары букв
# («ди» против «ры»), он врёт на десятки градусов.
AXIS_MIN_LETTERS = 3
# Направление на конце куска: по стольким крайним буквам или по крайней доле ширины — что длиннее.
TANGENT_LETTERS = 5
TANGENT_SHARE = 0.33
# Насколько наклон КОНЦА куска может отличаться от наклона всего куска. Конец меряется по
# тридцати-сорока пикселям, и разброс якорей в пару пикселей даёт там 8–10° — замер на
# 1973/08 с.85: у двух соседних слов одной строки концы разошлись на 8.4° и 10.5°, зоны
# разминулись, и строка распалась. Наклон всего куска устойчив, поэтому конец зажимается вокруг
# него.
TANGENT_CLAMP_DEG = 5.0
# Разумные пределы межстрочного шага (мм бумаги) — те же, что были у поля хода строк.
PITCH_MIN_MM = 2.0
PITCH_MAX_MM = 12.0
# Запасная оценка шага, если по соседям он не намерился: столько высот строчной буквы.
PITCH_FALLBACK_XH = 2.3
# Доля перекрытия по x, при которой кусок считается «стоящим над» другим (для оценки шага).
PITCH_OVERLAP_SHARE = 0.5


@dataclass(frozen=True)
class Piece:
    """Кусок строки: сгустки RLSA, якоря их букв и ось по этим якорям.

    Args:
        blobs: Индексы сгустков — ровно их сцепка и отдаёт наружу.
        anchors: Якоря букв ``(n, 2)`` слева направо: базовая линия буквы, поднятая к оси строки.
        sizes: Размеры тех же букв ``(n, 2)`` — ширина и высота.
        marks: Булев вектор «это низкая метка» по тем же буквам.
        x0, y0, x1, y1: Охват куска на рабочей копии.
        x_h: Медианная высота строчной буквы куска.
        letter_w: Медианная ширина буквы.
        leader_dots: Сколько букв куска — точки отточий страницы.
    """

    blobs: tuple[int, ...]
    anchors: np.ndarray
    sizes: np.ndarray
    marks: np.ndarray
    x0: float
    y0: float
    x1: float
    y1: float
    x_h: float
    letter_w: float
    leader_dots: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def letters(self) -> int:
        return int(self.anchors.shape[0])

    @property
    def long(self) -> bool:
        """Длинный ли кусок: своей оси хватает, чтобы выпустить от неё зону поиска."""
        return self.letters >= LONG_MIN_LETTERS and self.width >= LONG_ASPECT_XH * max(self.x_h, 1e-6)

    @property
    def cy(self) -> float:
        """Уровень куска: медиана якорей, то есть ось строки, а не середина бокса."""
        return float(np.median(self.anchors[:, 1]))

    def axis(self) -> tuple[np.ndarray, float]:
        """Коэффициенты МНК-оси по якорям букв и центр абсцисс, от которого они отсчитаны."""
        return fit_axis(self.anchors)

    def y_at(self, x: float) -> float:
        """Ордината оси куска в точке ``x`` (внутри охвата; за концами — продолжение параболы)."""
        coefficients, centre = self.axis()
        return float(np.polyval(coefficients, x - centre))

    def tangent(self, at_start: bool) -> tuple[float, float, float, float]:
        """Конец оси и единичное направление там: ``(x, y, dx, dy)``.

        Направление берётся по КРАЙНИМ буквам, а не по всей параболе: одна парабола на длинный
        сросшийся кусок плохо описывает его целиком, а зоне нужен местный ход на конце. Низкие
        метки и куски-черты из подгонки исключаются — запятая в конце слова сидит вне оси.
        """
        return end_direction(self.anchors, self.marks, self.sizes, at_start)


def letters_of(mask: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Буквы страницы и номер сгустка каждой.

    Буква — компонента маски глифов ДО смыкания RLSA; её сгусток читается по карте компонент
    ПОСЛЕ смыкания. Тот же приём, что у ``segment._glyph_count``, но разом на всю страницу и с
    размерами.

    Сгусток буквы берётся по ЛЮБОМУ её собственному пикселю, а не по метке под центром бокса.
    Смыкание — надмножество маски и связности не рвёт, поэтому все пиксели одной буквы лежат в
    одном сгустке и любой из них даёт верный ответ. А центр бокса у «2», «5», «з», «э» попадает в
    собственный просвет буквы, которого смыкание ядром 1×N не залило: метка там нулевая, и буква
    терялась вместе со своим якорем. Замер на 1973/08 с.85: так отваливалась 151 буква из 3624
    (4.2 %), а у сгустка «25,» не осталось ни одной цифры — только запятая, и кусок из числа
    превращался в «низкую метку».

    Args:
        mask: Маска глифов рабочей копии.
        labels: Карта компонент после смыкания.

    Returns:
        Пара ``(боксы, номера сгустков)``: боксы ``(n, 4)`` — ``cx, cy, ширина, высота``.
    """
    count, components, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if count <= 1:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.int32)
    boxes = np.column_stack(
        [
            centroids[1:, 0],
            centroids[1:, 1],
            stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64),
            stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64),
        ]
    )
    # Рассылка «пиксель буквы → её сгусток». Пишутся только НЕнулевые метки: смыкание ядром чётной
    # ширины в OpenCV не вполне расширяюще (dilate берёт максимум по окну без отражения ядра, а
    # erode — минимум по нему же, и при несимметричном якоре края фигуры срезаются), поэтому у
    # буквы 3–5 краевых пикселей могут оказаться вне сгустка. Любой её пиксель ВНУТРИ сгустка даёт
    # верный ответ, и ноль остаётся только у буквы, целиком выпавшей из смыкания.
    owners = np.zeros(count, dtype=np.int32)
    ink = components > 0
    letter, blob = components[ink], labels[ink]
    found = blob > 0
    owners[letter[found]] = blob[found]
    return boxes, owners[1:]


def is_low_mark(width: float, height: float, x_h: float) -> bool:
    """Точка, запятая или двоеточие: низкая и неширокая. Дефис не проходит по ширине."""
    return height <= MARK_MAX_HEIGHT_XH * x_h and width <= MARK_MAX_WIDTH_XH * x_h


def _line_at(xs: np.ndarray, ys: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Прямая МНК по ``(xs, ys)``, посчитанная в точках ``at``; при вырождении — медиана ``ys``."""
    if xs.size < 2 or float(xs.max() - xs.min()) < 1e-6:
        return np.full(at.shape, float(np.median(ys)))
    return np.polyval(np.polyfit(xs, ys, 1), at)


def baselines_of(boxes: np.ndarray, x_h: float) -> np.ndarray:
    """Базовая линия каждой буквы — низ её бокса с поправкой на выносные элементы вниз.

    Низ буквы и есть базовая линия у строчных без выноса, у прописных, у цифр и у точки. Ниже
    базовой линии свисают только «р», «у», «ф», «ц», «щ» и запятая; вверх не уходит никто.
    Поэтому отсев односторонний: по всем низам проводится прямая, буквы, ушедшие ниже неё больше
    чем на ``DESCENDER_MIN_XH`` икса, объявляются выносными, прямая пересчитывается без них, и их
    базовая линия берётся с прямой.

    Args:
        boxes: Буквы куска ``(n, 4)`` — ``cx, cy, ширина, высота``.
        x_h: Высота строчной буквы куска (единица порога).

    Returns:
        Ординаты базовой линии по каждой букве ``(n,)``.
    """
    xs = boxes[:, 0]
    bottoms = boxes[:, 1] + boxes[:, 3] / 2.0
    if bottoms.size < DESCENDER_MIN_LETTERS:
        return bottoms
    limit = DESCENDER_MIN_XH * x_h
    hangs = bottoms - _line_at(xs, bottoms, xs) > limit
    if not hangs.any() or int((~hangs).sum()) < 2:
        return bottoms
    fitted = _line_at(xs[~hangs], bottoms[~hangs], xs)
    return np.where(bottoms - fitted > limit, fitted, bottoms)


def anchors_of(boxes: np.ndarray, x_h: float, lift: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Якоря букв и признак низкой метки.

    Якорь — БАЗОВАЯ ЛИНИЯ буквы, поднятая на ``lift`` к оси строки, а не центр её бокса. Центр
    бокса зависит от того, какая это буква: у прописной и цифры он сидит на полразницы высот выше,
    чем у строчной (на корпусе пака-1 это 2 px при иксе 10). Внутри длинного слова буквы
    перемешаны и разница усредняется, а кусок из двух-трёх прописных или цифр целиком уезжает
    вверх, и соединение с соседним куском читается как перескок: на 1973/08 с.85 в сноске
    «ч. II, с. 421» шаг «ч.» → «II,» по центрам боксов дал −13.8° при настоящих −7.6°, и сцепка
    его отбила.

    Args:
        boxes: Буквы ``(n, 4)`` — ``cx, cy, ширина, высота``.
        x_h: Высота строчной буквы куска (единица порога выносных и низких меток).
        lift: На сколько поднять якорь от базовой линии. ``None`` — ``ANCHOR_ABOVE_BASELINE_XH``
            икса КУСКА; ``pieces_of`` передаёт сюда одну и ту же величину на всю полосу, чтобы
            уровни разных кусков были сравнимы напрямую.

    Returns:
        Пара ``(якоря (n, 2), булев вектор «низкая метка»)``.
    """
    marks = np.array([is_low_mark(width, height, x_h) for _, _, width, height in boxes], dtype=bool)
    above = ANCHOR_ABOVE_BASELINE_XH * x_h if lift is None else lift
    anchors = boxes[:, :2].astype(np.float64).copy()
    anchors[:, 1] = baselines_of(boxes, x_h) - above
    return anchors, marks


def fit_axis(anchors: np.ndarray) -> tuple[np.ndarray, float]:
    """МНК-ось куска по якорям букв: парабола от ``AXIS_PARABOLA_LETTERS`` букв, иначе прямая.

    Args:
        anchors: Якоря ``(n, 2)``.

    Returns:
        Пара ``(коэффициенты полинома, центр абсцисс)``; абсциссы центрируются ради
        обусловленности.
    """
    xs, ys = anchors[:, 0], anchors[:, 1]
    centre = float(xs.mean())
    if xs.size == 1 or float(xs.max() - xs.min()) < 1e-6:
        return np.array([float(ys.mean())]), centre
    degree = 2 if xs.size >= AXIS_PARABOLA_LETTERS else 1
    return np.polyfit(xs - centre, ys, degree), centre


def end_direction(
    anchors: np.ndarray, marks: np.ndarray, sizes: np.ndarray, at_start: bool
) -> tuple[float, float, float, float]:
    """Конец куска и направление его хода там: ``(x, y, dx, dy)`` с единичным ``(dx, dy)``.

    Из подгонки исключаются низкие метки и вытянутые компоненты (дефис в конце строки с
    переносом): они сидят вне оси и разворачивают касательную. Если после отсева точек меньше
    двух, берётся наклон по всем якорям куска, а при одной букве — горизонталь.

    Args:
        anchors: Якоря букв ``(n, 2)``.
        marks: Признак низкой метки по тем же буквам.
        sizes: Размеры тех же букв ``(n, 2)``.
        at_start: Нужен левый конец (иначе правый).

    Returns:
        Точка конца оси и единичный вектор: наружу от куска (влево для начала, вправо для конца).
    """
    order = np.argsort(anchors[:, 0])
    xs, ys = anchors[order, 0], anchors[order, 1]
    keep = ~marks[order] & (sizes[order, 0] <= 2.0 * np.maximum(sizes[order, 1], 1e-6))
    if keep.sum() >= 2:
        xs, ys = xs[keep], ys[keep]
    span = float(xs.max() - xs.min())
    if at_start:
        limit = xs.min() + max(TANGENT_SHARE * span, 1e-6)
        own = xs <= limit
        if own.sum() < TANGENT_LETTERS:
            own = np.zeros(xs.size, dtype=bool)
            own[: min(TANGENT_LETTERS, xs.size)] = True
        tip_x, tip_y = float(xs[0]), float(ys[0])
    else:
        limit = xs.max() - max(TANGENT_SHARE * span, 1e-6)
        own = xs >= limit
        if own.sum() < TANGENT_LETTERS:
            own = np.zeros(xs.size, dtype=bool)
            own[-min(TANGENT_LETTERS, xs.size) :] = True
        tip_x, tip_y = float(xs[-1]), float(ys[-1])
    slope = 0.0
    if own.sum() >= 2 and float(xs[own].max() - xs[own].min()) > 1e-6:
        slope = float(np.polyfit(xs[own], ys[own], 1)[0])
    # Наклон конца зажимается вокруг наклона всего куска: сам по себе он слишком шумен.
    if span > 1e-6 and xs.size >= 3:
        chord = float((ys[-1] - ys[0]) / span)
        limit = float(np.tan(np.radians(TANGENT_CLAMP_DEG)))
        slope = float(np.clip(slope, chord - limit, chord + limit))
    length = float(np.hypot(1.0, slope))
    direction = (-1.0 / length, -slope / length) if at_start else (1.0 / length, slope / length)
    return tip_x, tip_y, direction[0], direction[1]


def chord_slope(piece: Piece) -> float:
    """Наклон куска по хорде между крайними якорями: устойчивая мера против шума концов."""
    anchors = piece.anchors
    span = float(anchors[-1, 0] - anchors[0, 0])
    if anchors.shape[0] < 2 or abs(span) < 1e-6:
        return 0.0
    return float((anchors[-1, 1] - anchors[0, 1]) / span)


def pieces_of(
    stats: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    candidates: list[int],
    dpi: float = WORK_DPI,
    leaders: list | None = None,
) -> list[Piece]:
    """Куски строки страницы: сгустки-кандидаты со своими буквами, иксом и якорями.

    Args:
        stats: Статистика сгустков после смыкания.
        labels: Карта компонент после смыкания.
        mask: Маска глифов до смыкания.
        candidates: Индексы сгустков, годных в куски строки.
        dpi: Разрешение рабочей копии.
        leaders: Отточия страницы: по ним считается, сколько букв куска — точки отточий.

    Returns:
        Куски в том же порядке, что и ``candidates``; сгустки без букв пропускаются.
    """
    boxes, owners = letters_of(mask, labels)
    chosen = set(candidates)
    order: dict[int, list[int]] = {index: [] for index in candidates}
    for position, owner in enumerate(owners):
        if int(owner) in chosen:
            order[int(owner)].append(position)
    # Икс страницы — медиана высот всех букв: у куска из двух букв своя медиана ненадёжна.
    page_x_h = float(np.median(boxes[:, 3])) if boxes.shape[0] else mm_to_px(2.0, dpi)
    # Подъём якоря над базовой линией — ОДИН на всю полосу. Он сдвигает уровни всех кусков на
    # одну и ту же величину, поэтому их сравнение между собой не портит вовсе; возьми его от
    # собственного икса куска — и кусок из одних прописных снова уехал бы вверх, потому что его
    # «икс» (медиана высот его букв) равен высоте прописной, а не строчной.
    lift = ANCHOR_ABOVE_BASELINE_XH * page_x_h
    out: list[Piece] = []
    for index in candidates:
        own = order[index]
        if not own:
            continue
        piece_boxes = boxes[own]
        x_h = float(np.median(piece_boxes[:, 3])) if len(own) >= 3 else page_x_h
        x_h = max(x_h, 1.0)
        anchors, marks = anchors_of(piece_boxes, x_h, lift=lift)
        keep = np.argsort(anchors[:, 0])
        anchors, marks, piece_boxes = anchors[keep], marks[keep], piece_boxes[keep]
        x0 = float(stats[index, cv2.CC_STAT_LEFT])
        y0 = float(stats[index, cv2.CC_STAT_TOP])
        out.append(
            Piece(
                blobs=(int(index),),
                anchors=anchors,
                sizes=piece_boxes[:, 2:4].copy(),
                marks=marks,
                x0=x0,
                y0=y0,
                x1=x0 + float(stats[index, cv2.CC_STAT_WIDTH]),
                y1=y0 + float(stats[index, cv2.CC_STAT_HEIGHT]),
                x_h=x_h,
                letter_w=float(np.median(piece_boxes[:, 2])),
                leader_dots=_leader_dots(anchors, marks, piece_boxes, leaders),
            )
        )
    return out


def _leader_dots(anchors: np.ndarray, marks: np.ndarray, boxes: np.ndarray, leaders: list | None) -> int:
    """Сколько букв куска — точки отточий страницы (по готовым цепочкам ``leaders``)."""
    if not leaders or not marks.any():
        return 0
    count = 0
    for position in np.nonzero(marks)[0]:
        x = float(boxes[position, 0])
        y = float(boxes[position, 1])
        if any(leader.covers(x) and abs(leader.y - y) <= max(leader.thickness, 2.0) for leader in leaders):
            count += 1
    return count


def merged(first: Piece, second: Piece) -> Piece:
    """Слить два куска в один: буквы объединяются, ось потом строится заново по всем якорям."""
    left, right = (first, second) if first.x0 <= second.x0 else (second, first)
    anchors = np.vstack([left.anchors, right.anchors])
    sizes = np.vstack([left.sizes, right.sizes])
    marks = np.concatenate([left.marks, right.marks])
    order = np.argsort(anchors[:, 0])
    letters = anchors.shape[0]
    return Piece(
        blobs=tuple(sorted(set(left.blobs) | set(right.blobs))),
        anchors=anchors[order],
        sizes=sizes[order],
        marks=marks[order],
        x0=min(left.x0, right.x0),
        y0=min(left.y0, right.y0),
        x1=max(left.x1, right.x1),
        y1=max(left.y1, right.y1),
        # Икс — взвешенное по числу букв среднее: короткий кусок не должен сбивать меру длинного.
        x_h=(left.x_h * left.letters + right.x_h * right.letters) / max(letters, 1),
        letter_w=(left.letter_w * left.letters + right.letter_w * right.letters) / max(letters, 1),
        leader_dots=left.leader_dots + right.leader_dots,
    )


def axis_residual(piece: Piece) -> float:
    """Среднеквадратичный остаток якорей от оси куска (пиксели).

    По нему видно, что слияние свело в один кусок ДВЕ строки: перескок даёт ступеньку в полшага
    (на пак-1 это 11 px при остатке здорового куска меньше двух).
    """
    if piece.letters < 3:
        return 0.0
    coefficients, centre = piece.axis()
    resid = piece.anchors[:, 1] - np.polyval(coefficients, piece.anchors[:, 0] - centre)
    return float(np.sqrt(np.mean(resid**2)))


def pitch_of(pieces: list[Piece], dpi: float = WORK_DPI) -> float:
    """Межстрочный шаг страницы по самим кускам, без поля хода строк.

    Для каждого куска ищется ближайший по вертикали сосед, перекрытый с ним по x не меньше чем
    на ``PITCH_OVERLAP_SHARE`` более короткого, и берётся медиана таких расстояний. Оценка вне
    разумных пределов заменяется на ``PITCH_FALLBACK_XH`` высот строчной буквы.

    Args:
        pieces: Куски страницы.
        dpi: Разрешение рабочей копии.

    Returns:
        Шаг в пикселях рабочей копии.
    """
    x_h = float(np.median([piece.x_h for piece in pieces])) if pieces else mm_to_px(2.0, dpi)
    fallback = PITCH_FALLBACK_XH * max(x_h, 1.0)
    long_pieces = [piece for piece in pieces if piece.long]
    if len(long_pieces) < 3:
        return fallback
    xs0 = np.array([piece.x0 for piece in long_pieces])
    xs1 = np.array([piece.x1 for piece in long_pieces])
    ys = np.array([piece.cy for piece in long_pieces])
    gaps: list[float] = []
    for index in range(len(long_pieces)):
        overlap = np.minimum(xs1, xs1[index]) - np.maximum(xs0, xs0[index])
        shorter = np.minimum(xs1 - xs0, xs1[index] - xs0[index])
        own = overlap >= PITCH_OVERLAP_SHARE * np.maximum(shorter, 1e-6)
        own[index] = False
        if not own.any():
            continue
        distance = np.abs(ys[own] - ys[index])
        distance = distance[distance > 1.0]
        if distance.size:
            gaps.append(float(distance.min()))
    if not gaps:
        return fallback
    pitch = float(np.median(gaps))
    if not (mm_to_px(PITCH_MIN_MM, dpi) <= pitch <= mm_to_px(PITCH_MAX_MM, dpi)):
        return fallback
    return pitch


def page_x_height(pieces: list[Piece], dpi: float = WORK_DPI) -> float:
    """Высота строчной буквы страницы: медиана по кускам (мера всех порогов сцепки)."""
    if not pieces:
        return mm_to_px(2.0, dpi)
    return float(np.median([piece.x_h for piece in pieces]))


__all__ = [
    "Piece",
    "chord_slope",
    "anchors_of",
    "baselines_of",
    "axis_residual",
    "end_direction",
    "fit_axis",
    "is_low_mark",
    "letters_of",
    "merged",
    "page_x_height",
    "pieces_of",
    "pitch_of",
    "px_to_mm",
]
