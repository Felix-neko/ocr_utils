"""Алгебра прямоугольников разметки: слияние, отсев мелочи, признак «во всю полосу».

Здесь нет ни одного обращения к пикселям — только арифметика над кортежами
``(x1, y1, x2, y2)``. Пиксельные детекторы лежат в ``dots`` (полутоновая печать) и
``cover`` (сплошная плашка обложки), а собирает их вместе ``regions``.

ВСЕ РАЗМЕРЫ ЗДЕСЬ — В ПИКСЕЛЯХ ОРИГИНАЛА. В прошлой версии детектор работал по копии 1/4,
и пороги были заданы в её пикселях; теперь основной счёт идёт по полному кадру, и разница
в четыре раза — ровно та ошибка, которую никто не заметит по числу найденных областей.
"""

import numpy as np

# Зазор, при котором два прямоугольника считаются одной областью. 48 px при 600 dpi — 2 мм.
# Нужен там, где к найденным областям добавляются блоки Surya: блок обводит вёрстку целиком
# и с найденной под ним областью должен дать ОДИН прямоугольник, а не два вложенных.
MERGE_GAP_PX = 48

# Минимальная доля площади полосы, ниже которой область не считается иллюстрацией.
# Полоса 3492x6051 при 600 dpi — это 14.8x25.6 см, то есть 379 см^2; 0.4% от неё — около
# 1.2x1.2 см. Всё, что мельче, на полосе журнала — виньетка, буквица или грязь. Замер по
# валидационной выборке: самая мелкая настоящая фотография занимает 2.0% полосы, так что
# запас двойной.
MIN_REGION_FRAC = 0.004

# Доля площади полосы, с которой область считается «во всю страницу» (обложка, вклейка).
# Такая полоса целиком уходит в PDF без распрямления строк.
FULL_PAGE_FRAC = 0.75

# Минимальная сторона области. 160 px при 600 dpi — 0.7 см: тоньше не бывает даже узкая
# колонка с фотографией, а вот тень разворота вдоль края кадра бывает именно такой.
MIN_REGION_SIDE_PX = 160


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    """Пересекаются ли прямоугольники, если каждый раздуть на ``gap``."""
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def merge_boxes(boxes: list[tuple[int, int, int, int]], gap: int = MERGE_GAP_PX) -> list[tuple[int, int, int, int]]:
    """Сливает прямоугольники, пересекающиеся или отстоящие меньше чем на ``gap``.

    Слияние повторяется до неподвижной точки: объединение двух прямоугольников может
    дотянуться до третьего, который поодиночке не доставал ни до одного из них.
    """
    result = [tuple(map(int, box)) for box in boxes]
    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int, int, int]] = []
        for box in result:
            for index, other in enumerate(merged):
                if _overlaps(box, other, gap):
                    merged[index] = (
                        min(box[0], other[0]),
                        min(box[1], other[1]),
                        max(box[2], other[2]),
                        max(box[3], other[3]),
                    )
                    changed = True
                    break
            else:
                merged.append(box)
        result = merged
    return result


def polygons_to_boxes(polygons: list[np.ndarray]) -> list[tuple[int, int, int, int]]:
    """Полигоны Surya (4 точки) -> охватывающие прямоугольники."""
    boxes = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        boxes.append(
            (
                int(np.floor(points[:, 0].min())),
                int(np.floor(points[:, 1].min())),
                int(np.ceil(points[:, 0].max())),
                int(np.ceil(points[:, 1].max())),
            )
        )
    return boxes


def keep_significant(
    boxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
    min_region_frac: float = MIN_REGION_FRAC,
    min_side: int = MIN_REGION_SIDE_PX,
) -> list[tuple[int, int, int, int]]:
    """Отсев тонких лоскутов и мелочи; ``width`` и ``height`` — размеры ПОЛОСЫ."""
    min_area = min_region_frac * width * height
    kept = []
    for box in boxes:
        box_w, box_h = box[2] - box[0], box[3] - box[1]
        if box_w >= min_side and box_h >= min_side and box_w * box_h >= min_area:
            kept.append(box)
    return kept


def is_full_page(box: tuple[int, int, int, int], width: int, height: int, frac: float = FULL_PAGE_FRAC) -> bool:
    """Занимает ли область почти всю полосу — то есть обложку или вклейку."""
    return (box[2] - box[0]) * (box[3] - box[1]) >= frac * width * height


def upscale_box(box: tuple[int, int, int, int], scale: int) -> tuple[int, int, int, int]:
    """Прямоугольник уменьшенной копии -> координаты оригинала.

    ``scale`` обязателен и умолчания не имеет намеренно. Пока детекция шла по копии 1/4,
    значение по умолчанию было верным всегда; теперь по копии считается только детектор
    обложки, а полутона — по полному кадру, и молчаливая четвёрка превратилась бы в
    четырёхкратный промах координат, который в базе выглядит совершенно правдоподобно.
    """
    return box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """Загоняет прямоугольник в границы кадра.

    Нужен после :func:`upscale_box`: размеры уменьшенной копии округлялись вниз, и
    умножение обратно вылезает за кадр на величину до ``scale - 1`` пикселя.
    """
    return max(0, box[0]), max(0, box[1]), min(box[2], width), min(box[3], height)
