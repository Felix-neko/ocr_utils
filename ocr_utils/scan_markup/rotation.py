"""Углы поворота полосы: единые обозначения для базы, CVAT, детекции и очистки.

ЕДИНАЯ ВАЛЮТА — угол ПО ЧАСОВОЙ СТРЕЛКЕ, на который надо повернуть полосу, чтобы стало
прямо. Не «на сколько она повёрнута», а именно «на сколько повернуть»: так его применяет
очистка и так его показывает CVAT, и любое второе соглашение по соседству кончилось бы
поворотом не в ту сторону.

Модуль намеренно не тянет за собой реестр детекторов: его импортируют и схема базы, и
работа с CVAT, и очистка сканов, а поднимать ради константы torch с surya незачем.
"""

from __future__ import annotations

import numpy as np

# Допустимые значения. 0 значит «поворот не нужен», а не «неизвестно»: «неизвестно» — это
# NULL в базе, и путать их нельзя, иначе непосчитанная полоса выдаст себя за проверенную.
ROTATIONS: tuple[int, ...] = (0, 90, 180, 270)

# Человеческие имена — для имён файлов, таблиц и меток CVAT.
ROTATION_NAMES: dict[int, str] = {0: "прямо", 90: "cw90", 180: "180", 270: "ccw90"}

# Набор по умолчанию: без 180. На паке-1 все 42 размеченные вручную боковые полосы требуют
# поворота по часовой, а 180 не встретилось ни разу — широкие рисунки в советских изданиях
# ставили одинаково. Сужение набора не украшение: ответ, которого в нём нет, не может быть
# дан в принципе, а арбитру достаётся меньше прогонов распознавания.
DEFAULT_ALLOWED: tuple[int, ...] = (0, 90, 270)


class RotationError(ValueError):
    """Неразборный или недопустимый набор углов."""


def parse_allowed(raw: str | None, default: tuple[int, ...] = DEFAULT_ALLOWED) -> tuple[int, ...]:
    """Разбирает «0,90,270» в набор углов. Пустая строка и None — значит взять умолчание.

    Хранится набор строкой через запятую, а не JSON: JSON в схеме не используется нигде, а
    прецедент списка в колонке уже есть — ``MaskAnnotation.rle``.
    """
    if raw is None or not raw.strip():
        return default
    try:
        angles = tuple(sorted({int(part.strip()) % 360 for part in raw.split(",") if part.strip()}))
    except ValueError as error:
        raise RotationError(f"не разобрать набор углов {raw!r}") from error
    return validate_allowed(angles)


def validate_allowed(angles: "tuple[int, ...]") -> tuple[int, ...]:
    """Проверяет набор: только кратные 90 и обязательно с нулём."""
    bad = [angle for angle in angles if angle not in ROTATIONS]
    if bad:
        raise RotationError(f"бывают только {ROTATIONS}, получено {sorted(angles)}")
    if not angles:
        raise RotationError("набор углов пуст")
    if 0 not in angles:
        # Без нуля полосу, которой поворот не нужен, некуда отнести — а таких подавляющее
        # большинство, и детектор обязан иметь возможность так ответить.
        raise RotationError("0 обязан быть в наборе")
    return tuple(sorted(angles))


def format_allowed(angles: "tuple[int, ...]") -> str:
    """Обратное к :func:`parse_allowed` — то, что ложится в колонку."""
    return ",".join(str(angle) for angle in validate_allowed(tuple(angles)))


def name(rotate_cw: int) -> str:
    """Человеческое имя угла; для неизвестных — сам угол, чтобы отчёт не падал."""
    return ROTATION_NAMES.get(rotate_cw, str(rotate_cw))


def rotate_cw(image: np.ndarray, degrees: int) -> np.ndarray:
    """Поворот по часовой на кратное 90. Через ``np.rot90``, без интерполяции.

    Именно без интерполяции: пересемплирование размыло бы штрих, а на скане 600 dpi это
    ровно то, ради чего скан и делался. Заодно синтетическая проверка детекторов меряет
    детектор, а не качество ресайза.
    """
    # np.rot90 крутит ПРОТИВ часовой, поэтому число четвертей берётся с обратным знаком.
    return np.ascontiguousarray(np.rot90(image, k=-((degrees // 90) % 4)))


def rotate_size(width: int, height: int, degrees: int) -> tuple[int, int]:
    """Размер кадра после поворота: у 90 и 270 стороны меняются местами."""
    return (height, width) if (degrees // 90) % 2 else (width, height)


def rotate_box(box: "tuple[int, int, int, int]", width: int, height: int, degrees: int) -> tuple[int, int, int, int]:
    """Прямоугольник ``(x1, y1, x2, y2)`` из кадра ``width x height`` — в повёрнутый кадр.

    Нужен всюду, где координаты разметки лежат в системе ОРИГИНАЛА, а картинка на диске уже
    повёрнута: разметка растровых областей, врезки иллюстраций в PDF. Без пересчёта врезка
    легла бы на совершенно другое место полосы, и заметить это можно было бы только глазами.
    """
    x1, y1, x2, y2 = box
    turns = (degrees // 90) % 4
    if turns == 0:
        return x1, y1, x2, y2
    if turns == 1:  # по часовой: верх кадра уходит вправо
        return height - y2, x1, height - y1, x2
    if turns == 2:
        return width - x2, height - y2, width - x1, height - y1
    return y1, width - x2, y2, width - x1
