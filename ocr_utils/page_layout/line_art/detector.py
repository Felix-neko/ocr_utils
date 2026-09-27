"""Единый детектор line art: затравки-подсказки любого происхождения, одна пиксельная проверка, вне исключений.

ЗАЧЕМ ОДИН. До него line art искали три детектора порознь: детектор таблиц (схема/рисунок —
по линейкам, с подсказкой surya), детектор крупного штриха (связные пятна и скопления
линеек, без surya) и surya сама по себе (Figure/Picture из кэша). На эталоне из 221 ручной
области (``text_layer_fix eval-lineart``) surya давала F1 0.89, пятна — 0.66, а объединение
всех источников — полноту 0.98. Здесь источники объединены, но решает по-прежнему не модель:
каждая затравка проходит те же пиксельные правила ``features.classify`` (заполнение рамки,
дыры растра, доля длинных прогонов, сплошная масса), а рамки сливаются как у детектора порчи
геометрии (``merge_boxes`` с зазором в шаг строк).

ВХОД ПЛОСКИЙ. Детектор не знает, откуда пришла рамка: он получает два списка ``Box``.

* ``excluded_boxes`` — где line art быть не может: растр, таблицы, всё, что вызывающий
  пометит сам. Кандидат, накрытый исключениями больше чем наполовину, отбрасывается. Растр и
  таблицы находятся раньше: без исключений детектор штриха давал рамки внутри фотографий и
  таблиц (замечание пользователя по паку-2, 2026-09-21). Блок surya ``Picture`` над фотографией
  отдельно не фильтруется — его гасит то же исключение растра.
* ``hinted_boxes`` — затравки: блоки surya, схемы и рисунки детектора таблиц, любые другие
  подозрения. Связная статистика их не видит (у схемы без сплошного штриха нет крупного
  пятна), поэтому каждая проверяется пикселями как отдельный кандидат. Подсказка может быть
  :class:`~ocr_utils.page_layout.geometry.HintBox` с меткой источника — тогда метка доезжает до
  ``info["sources"]`` и вида рамки; голая ``Box`` идёт как безымянная (``"hint"``).

Сборщики подсказок из привычных источников — :func:`hints_from_surya` (``Figure``, ``Form``,
``Picture``; ``Equation`` НЕ затравка: формула в словаре проекта — текст, метка CVAT «Схема или
line art» её не включает, а на эталоне из 221 области блоки ``Equation`` дали 19 ложных
областей на 200 контрольных страницах и ни одного совпадения) и :func:`hints_from_tables`
(схема/рисунок детектора таблиц — уже выращены по штриховой краске и проверены по линейкам).

Третий источник — собственный проход ``features.analyse_gray`` по битональной копии (связные
пятна и скопления линеек); подсказки идут в тот же вызов как ``extra_boxes``, исключения — как
``exclude_boxes``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.page_layout.geometry import Box, HintBox, TableBox, hint_likely, hint_source, intersection
from ocr_utils.page_layout.line_art.classes import ObjectClass
from ocr_utils.page_layout.line_art.expand import FIGURE_GROW_MM, FORMULA_GROW_MM, grow_to_components
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

# Семейства источников, по доле которых считается уверенность рамки: таблицы, surya, пиксели.
# Подсказка из другого семейства тоже засчитывается, но уверенность не выходит за единицу.
SOURCE_FAMILIES = ("tables", "surya", "pixels")

# Метки собственного прохода ``analyse_gray`` — семейство «пиксели».
PIXEL_SOURCES = ("ink", "rules")

# Семейство подсказок surya: у согласных с рамкой берётся уверенность блока (``surya_conf``).
SURYA_FAMILY = "surya"

# Блоки surya, которые идут в затравки line art (формулы — нет, см. докстринг модуля).
SEED_LABELS = FIGURE_LABELS + FORM_LABELS

# Вероятные классы объекта по метке блока surya и по виду находки детектора таблиц.
SURYA_LIKELY = {
    "Figure": (ObjectClass.DIAGRAM, ObjectClass.DRAWING),
    "Picture": (ObjectClass.DRAWING,),
    "Form": (ObjectClass.FORM,),
    "Equation": (ObjectClass.FORMULA,),
}
TABLES_LIKELY = {"схема": (ObjectClass.DIAGRAM,), "рисунок": (ObjectClass.DRAWING,)}

# Меньше этой доли краски в рамке формулы — пустое место, а не формула (surya иногда обводит
# пустую строку между абзацами).
FORMULA_MIN_INK = 0.01


@dataclass(frozen=True)
class LineArtInputs:
    """Что подаётся детектору, всё в пикселях одной копии (``dpi``).

    Attributes:
        bitonal: Битональная копия полосы: краска 0, бумага 255.
        dpi: Разрешение этой копии; от него пороги ``features.params_for_dpi``.
        excluded_boxes: Рамки, внутри которых line art не бывает (растр, таблицы, что ещё
            пометит вызывающий). Кандидат, накрытый ими больше чем наполовину, выбрасывается.
        hinted_boxes: Затравки — рамки, где line art подозревают другие детекторы. Голая
            ``Box`` или ``HintBox`` с меткой источника и уверенностью.
    """

    bitonal: np.ndarray
    dpi: int
    excluded_boxes: list[Box]
    hinted_boxes: list[Box]


def hints_from_surya(
    blocks: LayoutBlocks | None, ink: np.ndarray | None = None, dpi: int | None = None, barriers: list[Box] = ()
) -> list[HintBox]:
    """Блоки surya ``Figure``/``Picture``/``Form``/``Equation`` как подсказки с вероятными классами.

    Рамки surya срезают края объекта (у ``Equation`` низ срезан у 30 % блоков,
    ``reports/surya_equations.md``), поэтому при поданной краске каждая рамка достраивается
    наружу до краёв пятен, которые режет (:func:`expand.grow_to_components`): формула — до
    ``FORMULA_GROW_MM``, остальное — до ``FIGURE_GROW_MM``.

    Args:
        blocks: Разметка surya в пикселях той же копии, что пойдёт детектору, или ``None``.
        ink: Маска краски той же копии (ненулевое — краска); ``None`` — рамки как у surya.
        dpi: Разрешение копии (нужно вместе с ``ink``).
        barriers: Рамки, за которые достройка не заходит (исключения детектора).

    Returns:
        Подсказки с меткой ``"surya:<метка>"``, уверенностью блока и ``likely`` по
        ``SURYA_LIKELY``; пустой список без разметки.
    """
    if blocks is None:
        return []
    hints: list[HintBox] = []
    for block in blocks.by_label(SEED_LABELS + EQUATION_LABELS):
        box = block.box.clipped(blocks.width, blocks.height)
        if ink is not None and dpi is not None:
            limit = FORMULA_GROW_MM if block.label in EQUATION_LABELS else FIGURE_GROW_MM
            box = grow_to_components(box, ink, dpi, limit, list(barriers)).box
        hints.append(
            HintBox(
                *box.as_tuple(),
                source=f"surya:{block.label}",
                confidence=float(block.confidence),
                likely=SURYA_LIKELY.get(block.label, ()),
            )
        )
    return hints


def hints_from_tables(drawings: list[TableBox]) -> list[HintBox]:
    """Находки детектора таблиц вида «схема»/«рисунок» как подсказки line art.

    Args:
        drawings: Находки ``tables.detector`` (вызывающий сам отбирает не-таблицы).

    Returns:
        Подсказки с меткой ``"tables:<вид>"``, без уверенности, ``likely`` по ``TABLES_LIKELY``.
    """
    return [
        HintBox(*table.box.as_tuple(), source=f"tables:{table.kind}", likely=TABLES_LIKELY.get(table.kind, ()))
        for table in drawings
    ]


def is_formula_hint(hint: Box) -> bool:
    """Подсказка только на формулу: в затравки line art не идёт, её разбирает :func:`detect_formulas`."""
    return hint_likely(hint) == (ObjectClass.FORMULA,)


def _family(source: str) -> str:
    """Семейство источника: ``ink``/``rules`` — пиксели, иначе часть метки до «:» (или вся метка)."""
    if source in PIXEL_SOURCES:
        return "pixels"
    return source.split(":", 1)[0]


def seeds_from(inputs: LineArtInputs) -> list[tuple[tuple[int, int, int, int], str]]:
    """Подсказки как пары ``(рамка, метка источника)`` для ``analyse_gray(extra_boxes=...)``.

    Args:
        inputs: Вход детектора.

    Returns:
        По паре на каждую подсказку в порядке ``inputs.hinted_boxes``.
    """
    return [(hint.as_tuple(), hint_source(hint)) for hint in inputs.hinted_boxes if not is_formula_hint(hint)]


def find_line_art(inputs: LineArtInputs) -> PageFindings:
    """Прогон ``analyse_gray`` со всеми затравками и исключениями.

    Args:
        inputs: Вход детектора.

    Returns:
        Находки ``analyse_gray``; рамки — в пикселях копии ``inputs.dpi``.
    """
    params = params_for_dpi(inputs.dpi)
    exclude = [box.as_tuple() for box in inputs.excluded_boxes]
    return analyse_gray(inputs.bitonal, params, exclude_boxes=exclude, extra_boxes=seeds_from(inputs))


def to_regions(findings: PageFindings, inputs: LineArtInputs) -> list[Region]:
    """Рамки детектора → области ``LINE_ART`` с уверенностью по числу семейств источников.

    Подсказка засчитывается рамке и тогда, когда её затравку ``analyse_gray`` не считал
    отдельно (пиксели уже нашли то же пятно, и затравка пропущена как дубль): согласие
    источника — это факт, а не то, кто первым успел. Согласие — пересечение не меньше половины
    меньшей из двух рамок (так ловятся и блок, накрывший рамку, и блок внутри рамки).

    Args:
        findings: Ответ ``analyse_gray``.
        inputs: То, что подавалось детектору (подсказки — для сверки источников).

    Returns:
        Области line art в пикселях копии, ``info``: вид, источники, число кандидатов, площадь
        краски кандидатов и уверенность согласного блока surya (``surya_conf``).
    """
    height, width = inputs.bitonal.shape[:2]
    page_area = width * height
    regions: list[Region] = []
    for box_tuple in findings.boxes:
        box = Box(*box_tuple)
        members = members_of(box_tuple, findings.candidates)
        sources = {c.source for c in members}
        strongest = max(members, key=lambda c: c.area).source if members else "ink"
        # Сверка со всеми подсказками: источник засчитывается, даже если его затравку не считали.
        surya_conf = None
        for hint in inputs.hinted_boxes:
            # Формульные подсказки к line art отношения не имеют (формулы — отдельный выход).
            if is_formula_hint(hint) or not _agrees(box, hint):
                continue
            source = hint_source(hint)
            sources.add(source)
            confidence = hint.confidence if isinstance(hint, HintBox) else None
            if _family(source) == SURYA_FAMILY and confidence is not None:
                surya_conf = max(surya_conf or 0.0, round(float(confidence), 4))
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
                confidence=round(min(1.0, len(families) / len(SOURCE_FAMILIES)), 4),
                source="line_art",
                full_page=box.area >= FULL_PAGE_FRAC * max(1, page_area),
                info=info,
            )
        )
    return regions


def _covered(box: Box, others: list[Box], share: float) -> bool:
    """Накрыт ли ``box`` рамками ``others`` не меньше чем на ``share`` своей площади (сумма пересечений)."""
    if not others or box.area <= 0:
        return False
    covered = 0
    for other in others:
        common = intersection(box, other)
        if common is not None:
            covered += common.area
    return covered >= share * box.area


def _agrees(box: Box, other: Box, share: float = 0.5) -> bool:
    """Согласны ли две рамки: пересечение не меньше ``share`` площади МЕНЬШЕЙ из них."""
    common = intersection(box, other)
    return common is not None and common.area >= share * max(1, min(box.area, other.area))


def detect_formulas(inputs: LineArtInputs) -> list[Region]:
    """Формулы по подсказкам только на формулу (surya ``Equation``): отдельный вид области, не line art.

    Рамка — подсказки (уже достроенной сборщиком); отбрасывается формула, накрытая исключениями
    больше чем наполовину (формула в ячейке таблицы — часть таблицы), и пустая рамка (краски
    меньше ``FORMULA_MIN_INK``).

    Args:
        inputs: Вход детектора (``bitonal``: краска 0).

    Returns:
        Области ``RegionKind.FORMULA`` в пикселях копии; ``info``: вид, источник, доля краски.
    """
    if inputs.bitonal is None or inputs.bitonal.size == 0:
        return []
    formulas: list[Region] = []
    for hint in inputs.hinted_boxes:
        if not is_formula_hint(hint):
            continue
        box = Box(*hint.as_tuple())
        if box.area == 0 or _covered(box, inputs.excluded_boxes, 0.5):
            continue
        ink = float(np.count_nonzero(inputs.bitonal[box.slice] == 0)) / box.area
        if ink < FORMULA_MIN_INK:
            continue
        confidence = hint.confidence if isinstance(hint, HintBox) else None
        formulas.append(
            Region(
                box,
                RegionKind.FORMULA,
                confidence=None if confidence is None else round(float(confidence), 4),
                source="line_art",
                info={"kind": ObjectClass.FORMULA.value, "sources": [hint_source(hint)], "ink_frac": round(ink, 4)},
            )
        )
    return formulas


def detect_line_art(inputs: LineArtInputs) -> list[Region]:
    """Полный ход: затравки + пиксели → области line art.

    Args:
        inputs: Вход детектора.

    Returns:
        Области line art в пикселях копии ``inputs.dpi``; пусто для пустой картинки.
    """
    if inputs.bitonal is None or inputs.bitonal.size == 0:
        return []
    return to_regions(find_line_art(inputs), inputs)


__all__ = [
    "FINE_KIND_BY_SOURCE",
    "LineArtInputs",
    "detect_formulas",
    "detect_line_art",
    "is_formula_hint",
    "find_line_art",
    "hints_from_surya",
    "hints_from_tables",
    "seeds_from",
    "to_regions",
]
