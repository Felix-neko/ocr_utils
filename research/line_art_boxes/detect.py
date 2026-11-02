"""Разбор line art вырезки страницы холста так же, как ``rectify.text_field.page_regions`` (без surya), с трассой роста рисунков детектора таблиц и вариантами прототипа."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import ocr_utils.page_layout.tables.detector as table_detector
from ocr_utils.page_layout import WORK_DPI
from ocr_utils.page_layout.analysis import Find, LayoutOptions, PageLayout
from ocr_utils.page_layout.image import PageImage, Variant
from research.line_art_boxes.fix import FixVariant, patched


@dataclass(frozen=True)
class Growth:
    """Один рост схемы или рисунка детектора таблиц (``grow_diagram``), px копии ``WORK_DPI``.

    Поля:
        seed: Затравка — рамка линеек ``(x0, y0, x1, y1)``.
        grown: Итог роста.
    """

    seed: tuple[int, int, int, int]
    grown: tuple[int, int, int, int]


@dataclass(frozen=True)
class ArtBox:
    """Рамка line art страницы.

    Поля:
        box: Рамка в px вырезки (родное разрешение холста) ``(x0, y0, x1, y1)``.
        kind: Тонкий вид (``схема``, ``рисунок``, ``штрих``, ``линейки``).
        sources: Источники кандидатов.
    """

    box: tuple[int, int, int, int]
    kind: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class PageArt:
    """Разбор line art страницы одним вариантом.

    Поля:
        variant: Вариант (``FixVariant``).
        boxes: Рамки line art.
        raster: Рамки растра (px вырезки).
        tables: Рамки таблиц (px вырезки).
        growths: Рост рисунков и схем детектора таблиц (px ``WORK_DPI``).
        scale: Во сколько раз вырезка крупнее копии ``WORK_DPI``.
    """

    variant: str
    boxes: tuple[ArtBox, ...]
    raster: tuple[tuple[int, int, int, int], ...]
    tables: tuple[tuple[int, int, int, int], ...]
    growths: tuple[Growth, ...] = field(default_factory=tuple)
    scale: float = 1.0


@contextmanager
def growth_trace():
    """Записывать каждый вызов ``tables.detector.grow_diagram`` (затравка → итог) в список, отдаваемый на входе.

    Yields:
        Список :class:`Growth`, пополняемый по ходу разбора.
    """
    original = table_detector.grow_diagram
    trace: list[Growth] = []

    def traced(art, box, dpi, barriers, figure=None):
        grown = original(art, box, dpi, barriers, figure)
        trace.append(Growth(box.as_tuple(), grown.as_tuple()))
        return grown

    table_detector.grow_diagram = traced
    try:
        yield trace
    finally:
        table_detector.grow_diagram = original


def _int_box(box) -> tuple[int, int, int, int]:
    """Рамка ``Box`` → кортеж целых."""
    return tuple(int(round(v)) for v in box.as_tuple())


def analyze_page(page: PageImage, variant: FixVariant, surya=None) -> PageArt:
    """Растр, таблицы и line art страницы ``page`` вариантом ``variant``.

    Args:
        page: Страница (вырезка холста или страница PDF).
        variant: Вариант прототипа (``FixVariant.BASE`` — боевой код как есть).
        surya: Источник блоков surya (``SuryaSource``, только кэш) или None — без surya, как в ``rectify``.

    Returns:
        Разбор страницы.
    """
    with growth_trace() as trace, patched(variant, page):
        layout = PageLayout(page, {Find.RASTER, Find.TABLES, Find.LINE_ART}, LayoutOptions(use_surya=surya is not None))
        layout.process(surya)
    arts = tuple(
        ArtBox(_int_box(r.box), str(r.info.get("kind", "")), tuple(r.info.get("sources", ()))) for r in layout.line_arts
    )
    return PageArt(
        variant.value,
        arts,
        tuple(_int_box(r.box) for r in layout.raster_pics),
        tuple(_int_box(r.box) for r in layout.tables),
        tuple(trace),
        page.scale_to_native(WORK_DPI),
    )


def analyze_crop(image: np.ndarray, dpi: float, variant: FixVariant) -> PageArt:
    """Растр, таблицы и line art вырезки страницы, как в ``page_regions`` (``use_surya=False``), вариантом ``variant``.

    Args:
        image: Вырезка страницы холста BGR.
        dpi: Плотность холста.
        variant: Вариант прототипа (``FixVariant.BASE`` — боевой код как есть).

    Returns:
        Разбор страницы.
    """
    page = PageImage.from_array(image, dpi=dpi, variant=Variant.SCAN, cache_name="line_art_boxes")
    return analyze_page(page, variant)


def save_page_art(path: Path, arts: dict[str, PageArt], meta: dict) -> None:
    """Записать разборы страницы всеми вариантами в JSON.

    Args:
        path: Файл.
        arts: Вариант → разбор.
        meta: Паспорт вырезки (``PageCrop`` как словарь).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "arts": {k: asdict(v) for k, v in arts.items()}}, ensure_ascii=False))


def load_page_art(path: Path) -> tuple[dict, dict[str, PageArt]]:
    """Прочитать JSON :func:`save_page_art`.

    Args:
        path: Файл.

    Returns:
        ``(паспорт, вариант → разбор)``.
    """
    data = json.loads(path.read_text())
    arts = {}
    for key, value in data["arts"].items():
        arts[key] = PageArt(
            value["variant"],
            tuple(ArtBox(tuple(b["box"]), b["kind"], tuple(b["sources"])) for b in value["boxes"]),
            tuple(tuple(b) for b in value["raster"]),
            tuple(tuple(b) for b in value["tables"]),
            tuple(Growth(tuple(g["seed"]), tuple(g["grown"])) for g in value["growths"]),
            value["scale"],
        )
    return data["meta"], arts
