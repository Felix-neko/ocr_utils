"""Ячейки таблиц и их ориентация из кэша ``text_layer_fix`` — как области разбора.

`page_layout` отдаёт на таблицу ОДИН прямоугольник и намеренно не ищет повёрнутый текст внутри
таблиц (`rotated_text/detector.py`): их разбирает `rotated_text.tables` с tesseract. Из-за этого
разбор строк не знал ни рёбер ячеек — и блок накрывал полтаблицы, — ни того, что шапка графы
набрана боком: ось шла поперёк букв.

Всё недостающее уже посчитано по паку и лежит в кэше `text_layer_fix`: на ячейку есть бокс по
центрам линеек (``box``), внутренность без них (``inner``) и угол поворота (``rotate_cw``). Здесь
только чтение и перевод координат; ни одного нового детектора.

**Каждая ячейка с краской становится отдельной областью разбора** (`hints.OrientedZone`). Тогда
требование «блок не попадает на несколько ячеек» выполняется по построению: область разбирается
своим проходом по своей вырезке, и блок за её границы не выходит. Заодно решается и поворот —
область с боковым текстом выпрямляется, как любая другая (`orient.py`).

Пустые ячейки областью не делаются: на 1967/07 с.73 из 143 ячеек текст лишь в двух десятках, а
проход на ячейку не бесплатный.

Координаты кэша — пиксели растра страницы при ``raster_dpi`` (≈600), обратный поворот таблицы и
сдвиг там уже применены. Наружу отдаются пиксели рабочей копии.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ocr_utils.page_layout.text_blocks import WORK_DPI
from ocr_utils.page_layout.text_blocks.hints import LayoutHints, OrientedZone

# Ячейка считается пустой, если краски в ней меньше этой доли площади. Доля, а не число пикселей:
# ячейки одной таблицы различаются по площади в десятки раз. Замер по 1967/07 с.73 и 1973/08 с.19:
# у пустых графов доля ниже 0.002, у ячейки с одним числом — от 0.01.
MIN_CELL_INK = 0.005
# Область меньше этого в миллиметрах бумаги не заводится: разбирать в ней нечего, а проход стоит.
MIN_CELL_MM = 3.0
# Виды зон кэша, которые берутся областями ПОМИМО ячеек: повёрнутый текст вне таблиц. У них, в
# отличие от находок `page_layout`, сторона уже определена.
FREE_ZONE_KINDS = ("standalone",)


def hints_of(
    pdf_name: str,
    page_index: int,
    cache_dir: Path,
    work: np.ndarray,
    dpi: float = WORK_DPI,
    base: LayoutHints | None = None,
) -> LayoutHints:
    """Подсказки страницы, дополненные ячейками таблиц из кэша ``text_layer_fix``.

    Args:
        pdf_name: Имя файла PDF с расширением — подпапка кэша.
        page_index: Номер страницы С НУЛЯ.
        cache_dir: Корень кэша (``<прогон>/cache``).
        work: Рабочая копия страницы (серая) — по ней проверяется, есть ли в ячейке краска.
        dpi: Разрешение рабочей копии.
        base: Подсказки `page_layout` (растр, рамки, боковой текст вне таблиц); дополняются, а не
            заменяются. ``None`` — начинаем с пустых.

    Returns:
        :class:`hints.LayoutHints`. Если кэша нет, он чужой версии или с ошибкой разбора, ``base``
        возвращается как есть — разбор пойдёт как прежде.
    """
    from ocr_utils.text_layer_fix.cache import cache_path, load_page

    base = base or LayoutHints(dpi=dpi)
    payload = load_page(cache_path(cache_dir, pdf_name, page_index))
    if payload is None:
        return base
    scale = dpi / float(payload.get("raster_dpi") or 600.0)
    inked = _ink_mask(work)
    cells, barriers = _cells_of(payload, scale, inked, dpi)
    free = _free_zones(payload, scale, dpi)
    if not cells and not free:
        return base
    # Старшинство — у последующих областей (см. ``page._area_allowed``): полоса целиком, за ней
    # боковые зоны `page_layout`, и в самом конце ячейки — они перекрывают всё.
    height, width = work.shape[:2]
    zones = base.zones_or_page(width, height) + tuple(free) + tuple(cells)
    return LayoutHints(
        text_allowed=_allow_cells(base.text_allowed, cells),
        zones=zones,
        barriers=base.barriers + tuple(barriers),
        dpi=dpi,
    )


def _allow_cells(allowed: np.ndarray | None, cells: list[OrientedZone]) -> np.ndarray | None:
    """Вернуть ячейкам право на текст: `page_layout` запрещает таблицу целиком.

    Зона таблицы идёт у `from_layout` в запретную маску — внутри рисунка и таблицы строк не ищем.
    Но ячейка, ставшая областью разбора, — как раз то место, где их искать НАДО, иначе краска в
    ней погаснет до сегментации и разбор таблицы пропадёт вовсе. Разрешение возвращается только по
    боксам самих ячеек, а не по всей таблице: линейки, поля и подписи между графами остаются под
    запретом.

    Args:
        allowed: Маска разрешённого текста или ``None`` (запретов нет — и возвращать нечего).
        cells: Области-ячейки.

    Returns:
        Копия маски с ``True`` по боксам ячеек, либо ``None``/исходная маска, если делать нечего.
    """
    if allowed is None or not cells:
        return allowed
    out = allowed.copy()
    height, width = out.shape[:2]
    for zone in cells:
        x0, y0, x1, y1 = zone.box
        out[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = True
    return out


def _ink_mask(work: np.ndarray) -> np.ndarray:
    """Булева маска краски рабочей копии — порогом Оцу, как везде в подпакете."""
    import cv2

    threshold, _ = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return work <= threshold


def _cells_of(
    payload: dict, scale: float, inked: np.ndarray, dpi: float
) -> tuple[list[OrientedZone], list[tuple[int, int, int, int]]]:
    """Области-ячейки и рамки-запреты по их боксам.

    Областью становится ПОЛНЫЙ бокс ячейки (``box``, по центрам линеек), он же идёт рамкой-
    запретом: его рёбра лягут к межколонникам и чертам деления блока.

    Внутренность без линеек (``inner``) областью НЕ берётся, хотя напрашивается: буквы нередко
    выходят за неё и даже за сам бокс (1973/08 с.19, ячейка ``box`` 189..1041 при тексте до 1046),
    и по ``inner`` строка либо теряется целиком, либо её обрезки достаются полосе-области и та
    сцепляет их в ложную строку через всю таблицу. Замер по девяти полосам стенда: с ``box``
    покрытие краски выше на всех табличных полосах (1973/08 90.0 → 93.2 %, 1971/11 82.6 → 85.7 %),
    а блоков на нескольких ячейках — ноль. Линейки внутри вырезки разбору не мешают: их снимает
    своё же распознавание черт (``segment.rules_of``).
    """
    from ocr_utils.page_layout import mm_to_px

    limit = mm_to_px(MIN_CELL_MM, dpi)
    zones: list[OrientedZone] = []
    barriers: list[tuple[int, int, int, int]] = []
    for table in payload.get("tables") or []:
        for cell in table.get("cells") or []:
            box = _scaled(cell.get("box"), scale, inked.shape)
            if box is not None:
                barriers.append(box)
            # Пустая ячейка или нет — считается по ВНУТРЕННОСТИ: линейки самого бокса это тоже
            # краска, и по боксу непустыми оказались бы все ячейки подряд.
            inner = _scaled(cell.get("inner") or cell.get("box"), scale, inked.shape)
            if inner is None or min(inner[2] - inner[0], inner[3] - inner[1]) < limit:
                continue
            if _ink_share(inked, inner) < MIN_CELL_INK:
                continue
            zones.append(OrientedZone(box or inner, _rotation(cell.get("rotate_cw"))))
    return zones, barriers


def _free_zones(payload: dict, scale: float, dpi: float) -> list[OrientedZone]:
    """Повёрнутый текст ВНЕ таблиц из кэша: у него, в отличие от `page_layout`, есть сторона."""
    from ocr_utils.page_layout import mm_to_px

    limit = mm_to_px(MIN_CELL_MM, dpi)
    out: list[OrientedZone] = []
    for zone in payload.get("zones") or []:
        if zone.get("kind") not in FREE_ZONE_KINDS:
            continue
        box = _scaled(zone.get("box"), scale, None)
        if box is None or min(box[2] - box[0], box[3] - box[1]) < limit:
            continue
        out.append(OrientedZone(box, _rotation(zone.get("rotate_cw"))))
    return out


def _rotation(value) -> int:
    """Угол поворота ячейки; ``None`` («судить не по чему») считаем прямым текстом."""
    return int(value) if value in (0, 90, 180, 270) else 0


def _scaled(box, scale: float, shape) -> tuple[int, int, int, int] | None:
    """Бокс кэша в пиксели рабочей копии, обрезанный по кадру; ``None`` у вырожденного."""
    if not box or len(box) != 4:
        return None
    x0, y0, x1, y1 = (int(round(value * scale)) for value in box)
    if shape is not None:
        height, width = shape[:2]
        x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
        y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _ink_share(inked: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Доля краски внутри бокса."""
    x0, y0, x1, y1 = box
    piece = inked[y0:y1, x0:x1]
    return float(piece.mean()) if piece.size else 0.0


__all__ = ["FREE_ZONE_KINDS", "MIN_CELL_INK", "MIN_CELL_MM", "hints_of"]
