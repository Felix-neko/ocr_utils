"""Разбор области с повёрнутым текстом: выпрямить вырезку, прогнать как есть, вернуть обратно.

Весь конвейер разбора горизонтален не порогом, а формулировкой: смыкание RLSA ядром 1×N, центр
масс краски ПО СТОЛБЦАМ, ось как функция ``y(x)``, ряды по ординатам, кромки блока как ``x(y)``.
Обобщать это на произвольную ориентацию значило бы переписать `segment`, `lines`, `blocks` и
`columns` и заново откалибровать все пороги.

Поэтому боковая область разбирается иначе: её вырезка ПОВОРАЧИВАЕТСЯ до прямого текста, через неё
проходит неизменённый конвейер, а координаты результата поворачиваются обратно. Ни один порог не
трогается, и на прямом тексте ничего не меняется вовсе.

Сторона поворота (90 против 270) для геометрии не нужна: оси и огибающие у них одни и те же,
различается только порядок чтения, которого здесь нет.

**Чего у боковой области нет.** Выключка (``alignment``) и боковые кромки огибающей (`left`,
`right`, `core_*`) — понятия читающего направления: «левый край блока» у лежащего текста это его
верх или низ, смотря куда читать, а сторону мы намеренно не определяем. Поэтому у блоков боковой
области остаётся замкнутый контур (`polygon`), а меры выключки не считаются.
"""

from __future__ import annotations

import numpy as np

from ocr_utils.curved_layout.hints import OrientedZone
from ocr_utils.scan_markup.rotation import rotate_cw as rotate_image


def upright(image: np.ndarray, zone: OrientedZone, scale: float) -> np.ndarray:
    """Вырезка области, повёрнутая до прямого текста.

    Args:
        image: Серый рендер страницы.
        zone: Область с ориентацией.
        scale: Во сколько раз изображение крупнее координат области (рендер против рабочей копии).

    Returns:
        Вырезка, повёрнутая на ``zone.rotate_cw`` по часовой.
    """
    x0, y0, x1, y1 = (int(round(value * scale)) for value in zone.box)
    crop = image[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
    if crop.size == 0 or zone.rotate_cw == 0:
        return crop
    return rotate_image(crop, zone.rotate_cw)


def back_points(points: np.ndarray, size: tuple[int, int], rotate_cw: int) -> np.ndarray:
    """Точки ``(N, 2)`` из выпрямленного кадра обратно в кадр вырезки.

    ``size`` — размер ИСХОДНОЙ (ещё не повёрнутой) вырезки ``(ширина, высота)``. Формулы —
    обратные к ``np.rot90``: поворот по часовой на 90 переводит пиксель ``(x, y)`` исходника в
    ``(H − 1 − y, x)`` повёрнутого кадра, значит обратно ``(x', y') → (y', H − 1 − x')``.

    Args:
        points: Точки в координатах выпрямленного кадра.
        size: Ширина и высота исходной вырезки.
        rotate_cw: На сколько кадр поворачивали по часовой.

    Returns:
        Точки в координатах исходной вырезки.
    """
    if points is None or len(points) == 0 or rotate_cw == 0:
        return points
    width, height = size
    xs, ys = np.asarray(points, dtype=np.float64)[:, 0], np.asarray(points, dtype=np.float64)[:, 1]
    if rotate_cw == 90:
        return np.column_stack([ys, (height - 1) - xs])
    if rotate_cw == 180:
        return np.column_stack([(width - 1) - xs, (height - 1) - ys])
    return np.column_stack([(width - 1) - ys, xs])  # 270


def back_axis(axis, size: tuple[int, int], zone: OrientedZone):
    """Ось строки из выпрямленного кадра в координаты страницы.

    Точки поворачиваются обратно, сортируются по x (после поворота порядок мог смениться) и
    сдвигаются на начало области. Меры формы строки (сагитта, наклон, изгиб) пересчитываются:
    они мерились в выпрямленном кадре и там верны, но знак наклона относится к другому кадру.
    """
    from dataclasses import replace

    from ocr_utils.curved_layout.lines import _shape_stats

    points = back_points(np.asarray(axis.points, dtype=np.float64), size, zone.rotate_cw)
    points = points[np.argsort(points[:, 0])]
    points = points + np.array([zone.box[0], zone.box[1]], dtype=np.float64)
    sagitta, slope, bend, resid = _shape_stats(points, axis.dpi, axis.mark_spans if not zone.sideways else ())
    return replace(
        axis,
        points=points,
        sagitta_mm=sagitta,
        slope_deg=slope,
        bend_mm=bend,
        resid_parabola_mm=resid,
        mark_spans=axis.mark_spans if not zone.sideways else (),
    )


def back_block(block, size: tuple[int, int], zone: OrientedZone):
    """Блок из выпрямленного кадра в координаты страницы.

    У боковой области переносятся замкнутые контуры и оси рядов; боковые кромки огибающей
    обнуляются вместе с мерами выключки — см. докстринг модуля.
    """
    from dataclasses import replace

    offset = np.array([zone.box[0], zone.box[1]], dtype=np.float64)

    def curve(points):
        if points is None or len(points) == 0:
            return points
        return back_points(np.asarray(points, dtype=np.float64), size, zone.rotate_cw) + offset

    def envelope(item):
        if item is None:
            return None
        polygon = curve(item.polygon)
        return replace(
            item,
            left=curve(item.left),
            right=curve(item.right),
            top=curve(item.top),
            bottom=curve(item.bottom),
            polygon=polygon,
            core_left=curve(item.core_left),
            core_right=curve(item.core_right),
            polygon_dilated=curve(item.polygon_dilated),
        )

    rows = tuple(_back_row(row, size, zone) for row in block.rows)
    polygon = np.asarray(envelope(block.envelope).polygon, dtype=np.float64)
    span = (int(polygon[:, 0].min()), int(polygon[:, 0].max()))
    return replace(
        block,
        rows=rows,
        span=span,
        envelope=envelope(block.envelope),
        envelope_coarse=envelope(block.envelope_coarse),
        envelope_ink=envelope(block.envelope_ink),
    )


def _back_row(row, size: tuple[int, int], zone: OrientedZone):
    """Ряд блока: оси и профили краски — обратным поворотом, края и уровень — из повёрнутых осей."""
    from dataclasses import replace

    offset = np.array([zone.box[0], zone.box[1]], dtype=np.float64)
    axes = tuple(back_axis(axis, size, zone) for axis in row.axes)
    points = np.vstack([np.asarray(axis.points, dtype=np.float64) for axis in axes]) if axes else None

    def profile(item):
        if item is None or len(item) == 0:
            return item
        return back_points(np.asarray(item, dtype=np.float64), size, zone.rotate_cw) + offset

    if points is None:
        return replace(row, axes=axes)
    return replace(
        row,
        axes=axes,
        y=float(np.median(points[:, 1])),
        x0=float(points[:, 0].min()),
        x1=float(points[:, 0].max()),
        top_edge=profile(row.top_edge),
        bottom_edge=profile(row.bottom_edge),
    )


def back_gutter(gutter, size: tuple[int, int], zone: OrientedZone):
    """Межколонник обратным поворотом: у боковой области он становится МЕЖСТРОЧНЫМ разрывом.

    Ломаная ``(y, x0, x1)`` в повёрнутом кадре после поворота перестаёт быть функцией ординаты,
    поэтому у боковой области межколонники наружу не отдаются вовсе — они своё дело сделали
    внутри, при сборке строк.
    """
    from dataclasses import replace

    if zone.sideways:
        return None
    if zone.rotate_cw == 180:
        return None  # перевёрнутая область: ломаная развернулась, наружу отдавать нечего
    return replace(
        gutter, points=tuple((y + zone.box[1], x0 + zone.box[0], x1 + zone.box[0]) for y, x0, x1 in gutter.points)
    )


def back_leader(leader, size: tuple[int, int], zone: OrientedZone):
    """Отточие обратным поворотом; у боковой области наружу не отдаётся по той же причине."""
    from dataclasses import replace

    if zone.sideways or zone.rotate_cw == 180:
        return None
    return replace(
        leader,
        x0=leader.x0 + zone.box[0],
        x1=leader.x1 + zone.box[0],
        y=leader.y + zone.box[1],
        points=tuple((x + zone.box[0], y + zone.box[1]) for x, y in leader.points),
    )


__all__ = ["back_axis", "back_block", "back_gutter", "back_leader", "back_points", "upright"]
