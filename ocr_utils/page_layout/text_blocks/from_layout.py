"""Находки ``page_layout`` как подсказки разбора: растр, таблицы, line art, повёрнутый текст.

Мост между общим детектором структуры полосы (`ocr_utils.page_layout`) и разбором строк. Он
намеренно тонкий: вся геометрия остаётся у `page_layout`, здесь только перевод в координаты
рабочей копии и в :class:`hints.LayoutHints`.

Что во что переходит:

* **растр** (фотографии, печати) — в запрет: краска там не текст, и строк в ней быть не должно;
* **таблицы и line art** — в рамки-запреты: через их рёбра строка не собирается, а блок делится.
  По запросу (``forbid_figures``) они идут ЕЩЁ И в запрет: внутри рисунка и таблицы строк не ищем
  вовсе. Запрет буквален — на 1973/06 с.65 детектор line art накрывает заголовок-вензель
  «ЭКОНОМИЧЕСКОЕ ОБРАЗОВАНИЕ КАДРОВ», и заголовок уходит из разбора вместе с зоной. Поэтому он и
  не по умолчанию: там, где разбирают САМУ таблицу или схему (ячейки из `from_text_layer`,
  боковые врезки блок-схем), запрет снимает ровно то, ради чего затевался разбор — замер по
  стенду `run_layout_hints.sh`: 1967/07 с.73 осей 58 → 1, 1970/06 с.62 — 97 → 8;
* **повёрнутый текст вне таблиц** — в боковые области. Сторона поворота детектором не
  определяется (см. `page_layout/rotated_text/detector.py`), и для геометрии она не нужна:
  берётся 90, а 270 дал бы те же оси и огибающие.

Порядок областей значим: полоса целиком идёт первой, боковые врезки за ней — область гасит у
себя всё, что забрали последующие (см. ``page.analyse_gray``).
"""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.text_blocks import WORK_DPI
from ocr_utils.page_layout.text_blocks.hints import LayoutHints, OrientedZone

# Боковая врезка тоньше этого в миллиметрах бумаги отдельной областью не делается: разбирать в
# ней нечего, а вырезка в три пикселя только мешает.
MIN_ZONE_MM = 4.0
# Припуск к рамке боковой врезки поперёк строк, с каждой стороны, мм: рамка детектора идёт впритык к цепочке глифов
# (2–3.7 мм поперёк на паке-1), и без припуска врезка не проходила ``MIN_ZONE_MM`` — подпись к фото длиной 120 мм
# (1971/03 IMG_0143_2R) терялась вместе с осью; из 20 полос с зонами вертикальные оси были только на 3.
ZONE_PAD_MM = 1.5


def hints_of(
    document,
    page_index: int,
    dpi: float = WORK_DPI,
    surya=None,
    width: int | None = None,
    height: int | None = None,
    variant=None,
    forbid_figures: bool = False,
) -> LayoutHints:
    """Подсказки для одной страницы PDF разбором ``page_layout``.

    Args:
        document: Открытый ``fitz.Document``.
        page_index: Номер страницы С НУЛЯ.
        dpi: Разрешение рабочей копии, в котором нужны координаты.
        surya: ``SuryaSource`` или ``None`` — тогда разбор по одним пикселям, без модели.
        width, height: Размер рабочей копии; нужны, чтобы область «вся полоса» точно совпала с
            кадром разбора. ``None`` — берётся размер кадра ``page_layout``.
        variant: Вариант картинки ``page_layout.image.Variant`` — ТОТ ЖЕ, что разбирается. Им
            задаётся и ключ кэша surya: с ``FR_NOGEO`` на geo-PDF кэш читался бы чужой.
            ``None`` — ``Variant.FR_NOGEO``.
        forbid_figures: Запрещать ли текст ВНУТРИ таблиц и line art, а не только резать по их
            рёбрам. Годится для набора сплошного текста, где таблица и схема — помеха; на
            табличных полосах гасит сам разбираемый материал.

    Returns:
        :class:`hints.LayoutHints` в пикселях рабочей копии. Промах кэша surya не ошибка: разбор
        повторяется по одним пикселям, без модели.
    """
    from ocr_utils.page_layout.analysis import Find, LayoutOptions, PageLayout
    from ocr_utils.page_layout.image import PageImage, Variant
    from ocr_utils.page_layout.surya.source import SuryaMissing

    image = PageImage.from_pdf_page(document, page_index, variant or Variant.FR_NOGEO)
    finds = {Find.RASTER, Find.TABLES, Find.LINE_ART, Find.ROTATED_TEXT}
    try:
        options = LayoutOptions(use_surya=surya is not None and surya.enabled)
        layout = PageLayout(image, finds, options).process(surya)
    except SuryaMissing:
        # Кэша на эту страницу нет, а модели в процессе тоже нет: детекторы умеют и по пикселям.
        layout = PageLayout(image, finds, LayoutOptions(use_surya=False)).process(None)
    scale = dpi / image.dpi
    page_width = width if width is not None else int(round(image.width * scale))
    page_height = height if height is not None else int(round(image.height * scale))
    # Рамки таблиц и схем режут сцепку всегда, а гасят краску внутри себя — только по запросу.
    barriers = [_box(region.box, scale) for region in layout.tables + layout.line_arts]
    forbidden = [_box(region.box, scale) for region in layout.raster_pics + layout.stamp_suspects]
    return build(
        forbidden=forbidden + (barriers if forbid_figures else []),
        barriers=barriers,
        sideways=[_box(region.box, scale) for region in layout.rotated_text_not_in_tables_regions],
        width=page_width,
        height=page_height,
        dpi=dpi,
        rules=[rule.scaled(scale).points for rule in layout.loose_rules],
    )


def build(
    forbidden: list[tuple[int, int, int, int]],
    barriers: list[tuple[int, int, int, int]],
    sideways: list[tuple[int, int, int, int]],
    width: int,
    height: int,
    dpi: float = WORK_DPI,
    rules: list | None = None,
) -> LayoutHints:
    """Собрать подсказки из готовых рамок (пиксели рабочей копии).

    Отделено от :func:`hints_of`, чтобы подавать рамки из чего угодно — из базы разметки, из
    кэша прогона, из теста — не поднимая ``page_layout``.

    Args:
        forbidden: Рамки, внутри которых текста быть не должно (растр, печати).
        barriers: Рамки таблиц и блок-схем.
        sideways: Рамки повёрнутого текста.
        width, height: Размер рабочей копии.
        dpi: Её разрешение.
        rules: Линейки-барьеры — ломаные ``((x, y), ...)`` в пикселях рабочей копии (непристроенные
            линейки детектора таблиц); ``None`` — нет.

    Returns:
        Подсказки: маска разрешённого текста, области (полоса целиком плюс боковые) и запреты.
    """
    from ocr_utils.page_layout import mm_to_px

    allowed: np.ndarray | None = None
    if forbidden:
        allowed = np.ones((height, width), dtype=bool)
        for x0, y0, x1, y1 in forbidden:
            allowed[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = False
    zones = [OrientedZone((0, 0, width, height), 0), *sideways_zones(sideways, width, height, dpi)]
    return LayoutHints(
        text_allowed=allowed,
        zones=tuple(zones) if len(zones) > 1 else (),
        barriers=tuple(barriers),
        rules=tuple(tuple((float(x), float(y)) for x, y in points) for points in (rules or ())),
        dpi=dpi,
    )


def sideways_zones(
    sideways: list[tuple[int, int, int, int]], width: int, height: int, dpi: float = WORK_DPI
) -> list[OrientedZone]:
    """Боковые области разбора из рамок повёрнутого текста: припуск поперёк строк, отсев тонких.

    Рамка расширяется на ``ZONE_PAD_MM`` с обеих сторон по короткой стороне (поперёк вертикальных строк) в пределах
    страницы; уже ``MIN_ZONE_MM`` после этого — выбрасывается.

    Args:
        sideways: Рамки повёрнутого текста ``(x0, y0, x1, y1)``, пиксели рабочей копии.
        width, height: Размер рабочей копии.
        dpi: Её разрешение.

    Returns:
        Области с поворотом 90.
    """
    from ocr_utils.page_layout import mm_to_px

    limit, pad = mm_to_px(MIN_ZONE_MM, dpi), int(round(mm_to_px(ZONE_PAD_MM, dpi)))
    out = []
    for x0, y0, x1, y1 in sideways:
        if x1 - x0 <= y1 - y0:
            x0, x1 = max(0, x0 - pad), min(width, x1 + pad)
        else:
            y0, y1 = max(0, y0 - pad), min(height, y1 + pad)
        if min(x1 - x0, y1 - y0) >= limit:
            out.append(OrientedZone((int(x0), int(y0), int(x1), int(y1)), 90))
    return out


def _box(box, scale: float) -> tuple[int, int, int, int]:
    """Рамка ``page_layout.geometry.Box`` в пиксели рабочей копии."""
    return (
        int(round(box.x0 * scale)),
        int(round(box.y0 * scale)),
        int(round(box.x1 * scale)),
        int(round(box.y1 * scale)),
    )


__all__ = ["MIN_ZONE_MM", "ZONE_PAD_MM", "build", "hints_of", "sideways_zones"]
