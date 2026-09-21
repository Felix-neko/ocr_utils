"""Единый детектор line art: три источника затравок, одна пиксельная проверка, только вне растра и таблиц.

ЗАЧЕМ ОДИН. До него line art искали три детектора порознь: детектор таблиц (схема/рисунок —
по линейкам, с подсказкой surya), детектор крупного штриха (связные пятна и скопления
линеек, без surya) и surya сама по себе (Figure/Picture из кэша). На эталоне из 221 ручной
области (``text_layer_fix eval-lineart``) surya давала F1 0.89, пятна — 0.66, а объединение
всех источников — полноту 0.98. Здесь источники объединены, но решает по-прежнему не модель:
каждая затравка проходит те же пиксельные правила ``features.classify`` (заполнение рамки,
дыры растра, доля длинных прогонов, сплошная масса), а рамки сливаются как у детектора порчи
геометрии (``merge_boxes`` с зазором в шаг строк).

ПОРЯДОК ВАЖЕН. Растр и таблицы найдены раньше и подаются сюда как исключения: line art не
может лежать внутри фотографии или таблицы, а именно такие рамки давал детектор штриха,
запущенный без исключений (замечание пользователя по паку-2, 2026-09-21). Кандидат, накрытый
исключениями больше чем наполовину, отбрасывается.

ЗАТРАВКИ:

1. схема/рисунок детектора таблиц (``tables.detector.detect`` с видом не «таблица») — уже
   выращены по штриховой краске и проверены по линейкам;
2. блоки surya ``Figure``, ``Form``, ``Equation`` и ``Picture`` — последний только если его не
   подтвердил растр (штрих surya зовёт ``Picture`` на 28 полосах из 31);
3. связные пятна и скопления линеек ``features.analyse_gray`` по битональной копии.

Все три идут в один вызов ``analyse_gray``: (1) и (2) — как ``extra_boxes`` с меткой источника,
(3) — его собственный проход; исключения — ``exclude_boxes``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.page_layout.geometry import Box, TableBox, intersection
from ocr_utils.page_layout.line_art.features import (
    FULL_PAGE_FRAC,
    PageFindings,
    analyse_gray,
    members_of,
    params_for_dpi,
)
from ocr_utils.page_layout.regions import Region, RegionKind
from ocr_utils.page_layout.surya.blocks import EQUATION_LABELS, FIGURE_LABELS, FORM_LABELS, LayoutBlocks

# Тонкий вид рамки — по источнику самого «сильного» кандидата; пишется в detector_info.
FINE_KIND_BY_SOURCE = {
    "tables:схема": "схема",
    "tables:рисунок": "рисунок",
    "surya:Figure": "рисунок",
    "surya:Form": "бланк",
    "surya:Equation": "формула",
    "surya:Picture": "рисунок",
    "rules": "линейки",
    "ink": "штрих",
}

# Сколько ИСТОЧНИКОВ бывает у рамки максимум: таблицы, surya, пиксели. Уверенность — доля.
SOURCE_FAMILIES = ("tables", "surya", "pixels")

# Доля площади, при которой блок surya ``Picture`` считается подтверждённым растром и в
# затравки line art не идёт.
PICTURE_RASTER_COVER = 0.5


@dataclass(frozen=True)
class LineArtInputs:
    """Что подаётся детектору, всё в пикселях одной копии (``dpi``)."""

    bitonal: np.ndarray  # краска 0, бумага 255
    dpi: int
    raster: list[Box]  # растровые области — исключения
    tables: list[Box]  # таблицы — исключения
    table_drawings: list[TableBox]  # схема/рисунок детектора таблиц — затравки
    blocks: LayoutBlocks | None  # surya в пикселях той же копии или None


def _family(source: str) -> str:
    if source.startswith("tables:"):
        return "tables"
    if source.startswith("surya:"):
        return "surya"
    return "pixels"


def _covered(box: Box, others: list[Box], share: float) -> bool:
    """Накрыт ли ``box`` рамками ``others`` больше чем на ``share`` своей площади (сумма пересечений)."""
    if not others or box.area <= 0:
        return False
    covered = 0
    for other in others:
        width = min(box.x1, other.x1) - max(box.x0, other.x0)
        height = min(box.y1, other.y1) - max(box.y0, other.y0)
        if width > 0 and height > 0:
            covered += width * height
    return covered >= share * box.area


def seeds_from(inputs: LineArtInputs) -> list[tuple[tuple[int, int, int, int], str]]:
    """Затравки (1) и (2) как пары ``(рамка, метка источника)`` для ``analyse_gray(extra_boxes=...)``."""
    seeds: list[tuple[tuple[int, int, int, int], str]] = []
    for table in inputs.table_drawings:
        seeds.append((table.box.as_tuple(), f"tables:{table.kind}"))
    if inputs.blocks is not None:
        for block in inputs.blocks.by_label(FIGURE_LABELS + FORM_LABELS + EQUATION_LABELS):
            box = block.box.clipped(inputs.blocks.width, inputs.blocks.height)
            if block.label == "Picture" and _covered(box, inputs.raster, PICTURE_RASTER_COVER):
                continue  # это фотография, её уже забрал растр
            seeds.append((box.as_tuple(), f"surya:{block.label}"))
    return seeds


def find_line_art(inputs: LineArtInputs) -> PageFindings:
    """Прогон ``analyse_gray`` со всеми затравками и исключениями; рамки — в пикселях копии."""
    params = params_for_dpi(inputs.dpi)
    exclude = [box.as_tuple() for box in inputs.raster + inputs.tables]
    return analyse_gray(inputs.bitonal, params, exclude_boxes=exclude, extra_boxes=seeds_from(inputs))


def to_regions(findings: PageFindings, inputs: LineArtInputs) -> list[Region]:
    """Рамки детектора → области ``LINE_ART`` с уверенностью по числу семейств источников.

    Семейство засчитывается рамке и тогда, когда его затравку ``analyse_gray`` не считал
    отдельно (пиксели уже нашли то же пятно, и затравка пропущена как дубль): согласие
    источника — это факт, а не то, кто первым успел.

    Args:
        findings: Ответ ``analyse_gray``.
        inputs: То, что подавалось детектору (surya и схемы детектора таблиц — для сверки).
    """
    height, width = inputs.bitonal.shape[:2]
    page_area = width * height
    regions: list[Region] = []
    for box_tuple in findings.boxes:
        box = Box(*box_tuple)
        members = members_of(box_tuple, findings.candidates)
        sources = {c.source for c in members}
        strongest = max(members, key=lambda c: c.area).source if members else "ink"
        for table in inputs.table_drawings:
            if _agrees(box, table.box):
                sources.add(f"tables:{table.kind}")
        surya_conf = None
        if inputs.blocks is not None:
            hit = inputs.blocks.covering(box, FIGURE_LABELS + FORM_LABELS + EQUATION_LABELS, share=0.5)
            if hit is None:
                hit = _block_inside(box, inputs.blocks)
            if hit is not None:
                surya_conf = round(float(hit.confidence), 4)
                sources.add(f"surya:{hit.label}")
        families = {_family(s) for s in sources}
        info = {
            "kind": FINE_KIND_BY_SOURCE.get(strongest, "штрих"),
            "sources": sorted(sources),
            "candidates": len(members),
            "area_px": int(sum(c.area for c in members)),
            "surya_conf": surya_conf,
        }
        regions.append(
            Region(
                box,
                RegionKind.LINE_ART,
                confidence=round(len(families) / len(SOURCE_FAMILIES), 4),
                source="line_art",
                full_page=box.area >= FULL_PAGE_FRAC * max(1, page_area),
                info=info,
            )
        )
    return regions


def _agrees(box: Box, other: Box, share: float = 0.5) -> bool:
    """Согласны ли две рамки: пересечение не меньше половины МЕНЬШЕЙ из них."""
    common = intersection(box, other)
    return common is not None and common.area >= share * max(1, min(box.area, other.area))


def _block_inside(box: Box, blocks: LayoutBlocks):
    """Самый уверенный блок surya нужного вида, лежащий внутри рамки хотя бы наполовину своей площади."""
    best = None
    for block in blocks.by_label(FIGURE_LABELS + FORM_LABELS + EQUATION_LABELS):
        common = intersection(block.box, box)
        if common is None or common.area < 0.5 * max(1, block.box.area):
            continue
        if best is None or block.confidence > best.confidence:
            best = block
    return best


def detect_line_art(inputs: LineArtInputs) -> list[Region]:
    """Полный ход: затравки + пиксели → области line art в пикселях копии ``inputs.dpi``."""
    if inputs.bitonal is None or inputs.bitonal.size == 0:
        return []
    return to_regions(find_line_art(inputs), inputs)


__all__ = ["FINE_KIND_BY_SOURCE", "LineArtInputs", "detect_line_art", "find_line_art", "seeds_from", "to_regions"]
