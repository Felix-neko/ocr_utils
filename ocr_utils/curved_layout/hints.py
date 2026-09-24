"""Вспомогательная информация от внешних детекторов на входе разбора: маски, ориентации, границы.

Разбор страницы сам по себе видит только краску и не знает, где на полосе таблица, где схема и
где текст лежит боком. Всё это уже умеет `page_layout`, и подать его находки на вход дешевле и
надёжнее, чем угадывать заново:

* **маска разрешённого текста** — где текст искать можно, а где нельзя (растр, фотография, поле
  под печатью). Краска вне маски гасится до сегментации, и ни строк, ни блоков там не возникает;
* **области с ориентацией текста** — прямой текст, боковой, перевёрнутый. Строки сращиваются и
  блоки собираются только ВНУТРИ одной области: через границу ориентаций сшивать нечего;
* **границы таблиц и блок-схем** — рамки, через которые строка не тянется. Вертикальные рёбра
  ложатся к межколонникам (запрет сцепки), горизонтальные — к чертам (деление блока).

Углы поворота — целые градусы по часовой из `ocr_utils.scan_markup.rotation` (`ROTATIONS`), тот же
словарь, что у базы разметки и у детектора ориентации; второго источника истины не заводим.

Для ГЕОМЕТРИИ строки сторона поворота не важна: у 90 и 270 оси и огибающие одни и те же, разница
только в порядке чтения, которого здесь нет. Поэтому внутри области сводятся к двум случаям —
прямой (0, 180) и боковой (90, 270), см. :meth:`OrientedZone.sideways`.

Все координаты — пиксели РАБОЧЕЙ КОПИИ (``WORK_DPI``), как и весь остальной разбор.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ocr_utils.curved_layout import WORK_DPI
from ocr_utils.scan_markup.rotation import ROTATIONS


@dataclass(frozen=True)
class OrientedZone:
    """Область страницы с одной ориентацией текста.

    Args:
        box: Рамка ``(x0, y0, x1, y1)`` в пикселях рабочей копии; ``x1``/``y1`` — за последним
            пикселем, как срез.
        rotate_cw: На сколько повернуть область ПО ЧАСОВОЙ, чтобы текст стал прямым: 0, 90, 180
            или 270 (``ocr_utils.scan_markup.rotation.ROTATIONS``).
    """

    box: tuple[int, int, int, int]
    rotate_cw: int = 0

    def __post_init__(self) -> None:
        if self.rotate_cw not in ROTATIONS:
            raise ValueError(f"поворот {self.rotate_cw} не из {ROTATIONS}")

    @property
    def sideways(self) -> bool:
        """Лежит ли текст боком: у 90 и 270 строки идут по вертикали."""
        return self.rotate_cw in (90, 270)

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


@dataclass(frozen=True)
class LayoutHints:
    """Что внешние детекторы рассказали о полосе. Все поля необязательны.

    Args:
        text_allowed: Булева маска размера рабочей копии: ``True`` — текст искать можно. ``None``
            — можно везде.
        zones: Области с ориентацией текста. Пусто — вся полоса прямая, одной областью.
        barriers: Рамки таблиц и блок-схем ``(x0, y0, x1, y1)``: через их рёбра строка не
            собирается и блок не тянется.
        dpi: Разрешение, в котором заданы координаты; служит только проверкой на несовпадение.
    """

    text_allowed: np.ndarray | None = None
    zones: tuple[OrientedZone, ...] = ()
    barriers: tuple[tuple[int, int, int, int], ...] = ()
    dpi: float = WORK_DPI

    @property
    def empty(self) -> bool:
        """Нечего подсказывать: разбор пойдёт ровно как без подсказок."""
        return self.text_allowed is None and not self.zones and not self.barriers

    def zones_or_page(self, width: int, height: int) -> tuple[OrientedZone, ...]:
        """Области разбора: заданные подсказкой или одна на всю полосу.

        Прямые области (0 и 180) сводятся в одну, если подсказка их не разделяла: делить полосу
        на куски без нужды значит терять межколонники и ряды, идущие через всю страницу.
        """
        return self.zones or (OrientedZone((0, 0, width, height), 0),)


def barrier_separators(barriers: tuple[tuple[int, int, int, int], ...]) -> list[tuple[int, int, int, int]]:
    """Вертикальные рёбра рамок как полосы запрета сцепки ``(x0, x1, y0, y1)``.

    Формат тот же, что у межколонников (``columns.separators_for_segmentation``): через такую
    полосу ``segment._crosses`` строку не собирает. Ребро делается шириной в один пиксель — рамка
    таблицы и так проходит по краске, а широкая полоса съела бы крайнюю графу.
    """
    out: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in barriers:
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        out.append((int(x0), int(x0) + 1, int(y0), int(y1)))
        out.append((int(x1) - 1, int(x1), int(y0), int(y1)))
    return out


def barrier_rules(barriers: tuple[tuple[int, int, int, int], ...]) -> list:
    """Горизонтальные рёбра рамок как черты ``segment.Rule``: по ним делится блок."""
    from ocr_utils.curved_layout.segment import Rule

    out = []
    for x0, y0, x1, y1 in barriers:
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        out.append(Rule(int(x0), int(y0), int(x1), int(y0) + 1))
        out.append(Rule(int(x0), int(y1) - 1, int(x1), int(y1)))
    return out


def masked_ink(gray: np.ndarray, allowed: np.ndarray | None, paper: int = 255) -> np.ndarray:
    """Копия изображения, где краска вне разрешённой маски погашена до бумаги.

    Args:
        gray: Серое изображение (рендер или рабочая копия).
        allowed: Булева маска в пикселях РАБОЧЕЙ копии или ``None``.
        paper: Значение бумаги.

    Returns:
        Само ``gray``, если маски нет, иначе его копия с погашенной краской. Маска
        масштабируется под размер изображения — рендер и рабочая копия разного разрешения.
    """
    if allowed is None:
        return gray
    import cv2

    if allowed.shape != gray.shape[:2]:
        allowed = (
            cv2.resize(allowed.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        )
    out = gray.copy()
    out[~allowed] = paper
    return out


def zone_mask(shape: tuple[int, int], zone: OrientedZone) -> np.ndarray:
    """Булева маска области: ``True`` внутри её рамки."""
    out = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = zone.box
    out[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = True
    return out


__all__ = ["LayoutHints", "OrientedZone", "barrier_rules", "barrier_separators", "masked_ink", "zone_mask"]
