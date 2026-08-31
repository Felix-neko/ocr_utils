"""Пересчёт координат между оригиналом и уменьшенной копией для CVAT.

Схема уменьшения (её же реализует ``cvat.images``)::

    divisor = round(dpi / CVAT_DPI)          600 dpi -> 8, 450 dpi -> 6
    crop_w  = (W // divisor) * divisor       обрезка справа и снизу
    crop_h  = (H // divisor) * divisor
    кадр CVAT = crop_w/divisor x crop_h/divisor

Обрезка нужна ради ТОЧНОГО масштаба. Без неё 3492 -> 436 означало бы коэффициент 8.0092,
и обратное умножение на 8 промахивалось бы тем сильнее, чем правее объект (на правом краю —
на 40 px). После обрезки соответствие точное на всей ширине кадра.

Цена обрезки — потерянная полоска шириной не более ``divisor - 1`` (<= 7 px) справа и
снизу, которой разметчик вообще не видел. Объект, доведённый им до самого края кадра, по
смыслу продолжается и в этой полоске, поэтому при пересчёте назад он РАСТЯГИВАЕТСЯ до
полного размера оригинала — см. :func:`expand_right_bottom`.
"""

import numpy as np

# Разрешение по умолчанию, до которого уменьшаются полосы для разметки (перекрывается
# опцией --cvat-dpi у to-cvat). Выбрано так, чтобы кадр 600 dpi (3492x6051) стал ~436x756:
# CVAT листает такие мгновенно, а печать и граница фотографии на них ещё различимы.
CVAT_DPI = 75


def divisor_for_dpi(dpi: int, cvat_dpi: int = CVAT_DPI) -> int:
    """Во сколько раз делить сторону: ``round(dpi / cvat_dpi)``, но не меньше 1."""
    return max(1, int(round(dpi / cvat_dpi)))


def crop_size(width: int, height: int, divisor: int) -> tuple[int, int]:
    """Размер после обрезки справа-снизу до кратности ``divisor``."""
    return (width // divisor) * divisor, (height // divisor) * divisor


def cvat_size(width: int, height: int, divisor: int) -> tuple[int, int]:
    """Размер кадра в CVAT."""
    crop_w, crop_h = crop_size(width, height, divisor)
    return crop_w // divisor, crop_h // divisor


def to_cvat_rect(
    x1: float, y1: float, x2: float, y2: float, divisor: int, cvat_width: int, cvat_height: int
) -> tuple[float, float, float, float]:
    """Прямоугольник оригинала -> координаты кадра CVAT, с клампом в границы кадра.

    Кламп обязателен: область могла быть найдена в той самой обрезанной полоске справа
    или снизу, и CVAT отвергает шейп, вылезающий за кадр.
    """
    return (
        min(max(x1 / divisor, 0.0), cvat_width),
        min(max(y1 / divisor, 0.0), cvat_height),
        min(max(x2 / divisor, 0.0), cvat_width),
        min(max(y2 / divisor, 0.0), cvat_height),
    )


def expand_right_bottom(value: int, crop_value: int, full_value: int) -> int:
    """Дотягивает правую/нижнюю границу до края оригинала, если она упёрлась в край кадра.

    ``value`` — уже умноженная на divisor координата, ``crop_value`` — размер после
    обрезки, ``full_value`` — размер оригинала. Разница между ними не больше 7 px, но
    именно в неё попадает край фотографии, свёрстанной в обрез.
    """
    return full_value if value >= crop_value else value


def rect_to_original(
    x1: float, y1: float, x2: float, y2: float, divisor: int, width: int, height: int
) -> tuple[int, int, int, int]:
    """Прямоугольник кадра CVAT -> координаты оригинала, с распространением в обрезанную полосу.

    ``width``/``height`` — размеры ОРИГИНАЛА. Возвращается целочисленный полуинтервал
    ``[x1, x2) x [y1, y2)``, пригодный для среза numpy.
    """
    crop_w, crop_h = crop_size(width, height, divisor)
    ox1 = min(max(int(round(x1 * divisor)), 0), width)
    oy1 = min(max(int(round(y1 * divisor)), 0), height)
    ox2 = min(max(int(round(x2 * divisor)), 0), width)
    oy2 = min(max(int(round(y2 * divisor)), 0), height)
    return ox1, oy1, expand_right_bottom(ox2, crop_w, width), expand_right_bottom(oy2, crop_h, height)


def mask_to_original(mask: np.ndarray, divisor: int, width: int, height: int) -> np.ndarray:
    """Маска кадра CVAT -> маска оригинала: апскейл nearest и распространение в обрезанную полосу.

    ``np.repeat`` по обеим осям — это в точности обращение пары ``crop`` + ``resize``:
    каждый пиксель разметки становится квадратом divisor x divisor. Ничего умнее делать
    нельзя, информации о том, где внутри этого квадрата проходила настоящая граница, в
    разметке просто нет.

    Затем последний столбец (если в нём есть единицы) копируется на всю потерянную при
    обрезке полоску справа, последняя строка — на полоску снизу. Логика та же, что у
    :func:`expand_right_bottom`: разметчик довёл кисть до края кадра, значит объект
    продолжается и дальше.
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    crop_w, crop_h = crop_size(width, height, divisor)
    upscaled = np.repeat(np.repeat(mask, divisor, axis=0), divisor, axis=1)
    if upscaled.shape != (crop_h, crop_w):
        raise ValueError(f"маска {mask.shape} при divisor={divisor} не даёт обрезанный кадр {(crop_h, crop_w)}")

    full = np.zeros((height, width), dtype=bool)
    full[:crop_h, :crop_w] = upscaled
    if width > crop_w:
        full[:crop_h, crop_w:] = upscaled[:, -1:]
    if height > crop_h:
        full[crop_h:, :crop_w] = upscaled[-1:, :]
    if width > crop_w and height > crop_h:
        # Угол: единица там, где к нему примыкают и правая полоска, и нижняя.
        full[crop_h:, crop_w:] = upscaled[-1, -1]
    return full
