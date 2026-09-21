"""Детектор таблиц и блок-схем: связные рёбра, вид объекта, граница, не режущая букв.

Это четвёртая версия детектора исследований (v4.2, ``research/legacy/table_processing``);
история версий и отвергнутые ходы — в README пакета. Что здесь чинится против третьей
версии (диагноз по 59 полосам, разложенным человеком, — пошаговой печатью конвейера v3):

1. Затравки — связные ядра линеек (``rules.cores``), а не склейка дилатацией: колонтитул,
   подчёркивание в соседней колонке, линии бланка и дробная черта к таблице не липнут.
2. Линейки собираются цепочками фрагментов и идут вслед за изгибом полосы (``rules``);
   ядро кляксы отсекается по тёмному соседству.
3. Рост таблицы — только по продолжениям её рёбер и по линейкам, которые их пересекают, с
   абсолютным вето по чужому тексту. В третьей версии клеилось всё в 12 мм, а вето перебивала
   проверка: рост с чужим текстом 0 → 0.74 проходил, потому что «это всё ещё таблица».
4. Обрезка (``refine.fit_rows``) работает с ПОДАННОЙ рамкой: третья пересобирала рамку из
   линеек всей полосы и откатывала всё, что нашёл рост, а могла и расширить рамку на бланк в
   соседней колонке. Заодно берутся строки без вертикалей — «Итого», «Расхождение норм».
5. Граница не пересекает ни одной компоненты краски (``refine.push_edges``): не «меньше
   процента краски под границей», а ноль разрезанных букв.
6. Находка несёт вид: таблица, схема, рисунок (``kind``). Схема растёт заливкой по
   штриховой краске с дилатацией — так берутся и блоки, связанные пунктиром.
7. При наличии разметки surya (``layout``) вид берётся из неё, рост таблицы допускается
   внутрь её Table-блока и запрещается в её Text-блоки, а схема добирает свой Figure-блок.

ПОРЯДОК: линейки → ядра → фильтры политики → вид и проверка → рост по виду (соседи — преграды)
→ строки по просветам → выталкивание границ → дедупликация.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import cv2
import numpy as np

from ocr_utils.page_layout.tables import kind as kind_module
from ocr_utils.page_layout.tables import quality, refine, rules
from ocr_utils.page_layout.tables.ruling import (
    WORK_DPI,
    ClusterPolicy,
    Lines,
    Segment,
    binarize,
    cluster_tables,
    find_lines,
    mm_to_px,
)
from ocr_utils.page_layout.geometry import KIND_DIAGRAM, KIND_DRAWING, KIND_TABLE, Box, TableBox, intersection
from ocr_utils.page_layout.surya.blocks import FIGURE_LABELS, FORM_LABELS, TABLE_LABELS, LayoutBlocks

logger = logging.getLogger(__name__)

# Политика размеров и сквозных линеек у ТАБЛИЦЫ — третьей версии. Старое правило «обе стороны
# не меньше 25 мм» остаётся: без него теряются небольшие квадратные таблицы. Пара «длинная 40,
# короткая 10» добавляет к нему широкие короткие шапки бланков — 126x25, 130x16, 91x21 мм.
# Сквозной считается линейка любой оси: у бланков, размеченных вертикалями граф без внешней
# рамки сверху и снизу, сквозная горизонталь одна, зато сквозных вертикалей 2–10.
TABLE_POLICY = ClusterPolicy(min_sides_mm=((25.0, 25.0), (40.0, 10.0)), long_rules_any_axis=True)

# Политика затравок: та же, но без требования «сквозных линеек ≥ 2». Это требование было
# отсевом рисунков ДО проверки; здесь рисунок и схема — не мусор, а находки со своим видом,
# поэтому сквозные считаются позже и только для таблиц.
POLICY = replace(TABLE_POLICY, min_long_rules=0)
MIN_LONG_RULES_FOR_TABLE = TABLE_POLICY.min_long_rules

# --- Рост таблицы ---------------------------------------------------------------

# Ореол вокруг рамки, в котором ищутся продолжения и пересекающие линейки: 20 мм, как в v3.
HALO_MM = 20.0

# Зазор, через который фрагмент считается продолжением ребра таблицы: 3 мм. Больше — и в
# продолжения лезут подчёркивания в соседней колонке (1966/05 с.29: 4 мм от угла).
CONTINUATION_GAP_MM = 3.0

# Общий потолок роста стороны, как в v3.
GROW_CAP_MM = 40.0
GROW_PASSES = 3

# Прирост доли чужого текста, выше которого рост отвергается. Вето абсолютное.
MAX_FOREIGN_GAIN = 0.02

# --- Рост схемы ------------------------------------------------------------------

# Дилатация заливки по штриховой краске: 5 мм. Пунктирные связи блок-схем рвутся на 1–2 мм,
# стрелка отстоит от коробки на 1–3 мм; 5 мм берёт их с запасом, а до соседнего абзаца
# (обычно 8–12 мм) не дотягивается.
DIAGRAM_GLUE_MM = 5.0
DIAGRAM_CAP_MM = 60.0

# Компонента краски размером с букву: не выше и не шире этого — текст, а не штрих схемы.
TEXT_MAX_HEIGHT_MM = 4.0
TEXT_MAX_WIDTH_MM = 12.0

# Сирая горизонталь не короче этой доли ширины полосы — линейка колонтитула, не штрих схемы.
PAGE_RULE_SPAN = 0.5

# Схема не выходит за Figure-блок surya дальше этого запаса. Потолок, а не граница: область
# surya неточна на миллиметры, а не на колонку текста.
FIGURE_SLACK_MM = 10.0

# --- Заголовок в рамке ----------------------------------------------------------------

# Буквы в таблице мелкие: у 98 размеченных таблиц девяностый процентиль высоты компоненты
# краски внутри рамки не выше 2.54 мм. У заголовка рубрики в рамке («ОПЫТ РАБОТЫ
# ТЕРРИТОРИАЛЬНЫХ УПРАВЛЕНИЙ», «КРИТИКА И БИБЛИОГРАФИЯ») — 3.2–4.1 мм: это и есть признак,
# по которому 34 таких рамки из 42 новых срабатываний по паку отделяются от таблиц. Правило
# третьей версии («внутренних горизонталей нет, ячеек не больше трёх») их не берёт: цепочки
# находят и рамку баннера, и стебли буквицы, и ячеек получается четыре.
BANNER_GLYPH_P90_MM = 3.0
BANNER_MIN_GLYPHS = 8
# Выше этого рамка с крупными буквами — уже не баннер, а обложка или плакат; их не трогаем —
# если только линеек в ней не больше, чем у одной рамки (``FRAME_MAX_RULES``): тогда это
# заголовок статьи в рамке при любой высоте.
BANNER_MAX_HEIGHT_MM = 35.0
FRAME_MAX_RULES = 4


def glyph_height_p90(components: np.ndarray, box: Box, dpi: int) -> float:
    """Девяностый процентиль высоты компонент краски внутри рамки, в миллиметрах."""
    if components.size == 0:
        return 0.0
    x0, y0, x1, y1 = components.T
    inside = (x0 >= box.x0) & (x1 <= box.x1) & (y0 >= box.y0) & (y1 <= box.y1)
    heights = (y1 - y0)[inside]
    heights = heights[heights >= mm_to_px(0.8, dpi)]
    if heights.size < BANNER_MIN_GLYPHS:
        return 0.0
    return float(np.percentile(heights, 90)) * 25.4 / dpi


# --- Surya ---------------------------------------------------------------------

# Блок surya накрывает рамку хотя бы на столько, чтобы решать её вид.
LAYOUT_COVER = 0.7

# Form у surya — и бланк (таблица), и блок-схема из коробок («Схема 1» 1970/06 с.62): вид
# решают линейки по нестрогому признаку схемы. Метка внутренняя, наружу не выходит.
KIND_FORM = "form"


def _extent(boxes: list[Box]) -> Box:
    return Box(min(b.x0 for b in boxes), min(b.y0 for b in boxes), max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def _inside(rule: Box, box: Box) -> bool:
    return rule.x0 >= box.x0 and rule.y0 >= box.y0 and rule.x1 <= box.x1 and rule.y1 <= box.y1


def _overlaps(first: Box, second: Box) -> bool:
    return first.x0 < second.x1 and second.x0 < first.x1 and first.y0 < second.y1 and second.y0 < first.y1


def _centre_inside(rule: Box, box: Box) -> bool:
    return box.x0 <= (rule.x0 + rule.x1) // 2 <= box.x1 and box.y0 <= (rule.y0 + rule.y1) // 2 <= box.y1


def _cap(grown: Box, start: Box, cap: int, shape: tuple[int, int]) -> Box:
    height, width = shape
    return Box(
        max(grown.x0, start.x0 - cap),
        max(grown.y0, start.y0 - cap),
        min(grown.x1, start.x1 + cap),
        min(grown.y1, start.y1 + cap),
    ).clipped(width, height)


def _stop_before(grown: Box, current: Box, barriers: list[Box]) -> Box:
    """Обрезать рост так, чтобы рамка не наехала на чужую находку (как в v3)."""
    result = grown
    for barrier in barriers:
        if not _overlaps(result, barrier):
            continue
        if barrier.y0 >= current.y1:
            result = Box(result.x0, result.y0, result.x1, min(result.y1, barrier.y0))
        elif barrier.y1 <= current.y0:
            result = Box(result.x0, max(result.y0, barrier.y1), result.x1, result.y1)
        elif barrier.x0 >= current.x1:
            result = Box(result.x0, result.y0, min(result.x1, barrier.x0), result.y1)
        elif barrier.x1 <= current.x0:
            result = Box(max(result.x0, barrier.x1), result.y0, result.x1, result.y1)
        else:
            return current
        if result.width <= 0 or result.height <= 0:
            return current
    return result


def _text_ink(binary: np.ndarray, lines: Lines) -> np.ndarray:
    fat = cv2.dilate(lines.mask, np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(binary, cv2.bitwise_not(fat))


# --- Таблица ----------------------------------------------------------------------


def grow_table(
    lines: Lines,
    fragments: tuple[list[rules.Fragment], list[rules.Fragment]],
    box: Box,
    ink: np.ndarray,
    dpi: int,
    barriers: list[Box],
    allowed: "Box | None" = None,
    forbidden: "list[Box] | None" = None,
) -> tuple[Box, list[Segment]]:
    """Дорастить таблицу по продолжениям рёбер и пересекающим линейкам.

    ``allowed`` — Table-блок surya: внутри него линейка полосы берётся и без связности (так
    достаётся графа, чью линейку изгиб полосы порвал). ``forbidden`` — Text-блоки surya: рост
    внутрь них запрещён. Без разметки оба пусты, и работает только связность.
    """
    height, width = ink.shape[:2]
    halo = mm_to_px(HALO_MM, dpi)
    cap = mm_to_px(GROW_CAP_MM, dpi)
    tolerance = mm_to_px(refine.CROSS_TOL_MM, dpi)
    start = box
    current = box
    owned = [s for s in list(lines.horizontal) + list(lines.vertical) if _centre_inside(s.box, box)]
    owned_boxes = {s.box for s in owned}

    for _ in range(GROW_PASSES):
        grown_halo = Box(current.x0 - halo, current.y0 - halo, current.x1 + halo, current.y1 + halo).clipped(
            width, height
        )
        candidates: list[Segment] = []
        for segment in list(lines.horizontal) + list(lines.vertical):
            if segment.box in owned_boxes or not _overlaps(segment.box, grown_halo):
                continue
            others = [o for o in owned if o.horizontal is not segment.horizontal]
            crossing = any(
                (
                    refine.crosses(segment, o, tolerance)
                    if not segment.horizontal
                    else refine.crosses(o, segment, tolerance)
                )
                for o in others
            )
            inside_allowed = allowed is not None and _inside(segment.box, allowed)
            if crossing or inside_allowed:
                candidates.append(segment)
        anchors = [rules._as_fragment(s) for s in owned]
        for axis_items in fragments:
            for fragment in axis_items:
                if _inside(fragment.box, current) or not _overlaps(fragment.box, grown_halo):
                    continue
                if any(
                    rules.continues(a, fragment, dpi, CONTINUATION_GAP_MM)
                    for a in anchors
                    if a.horizontal is fragment.horizontal
                ):
                    candidates.append(Segment(fragment.box, fragment.horizontal, fragment.angle_deg))
        candidates = [
            c
            for c in candidates
            if not any(_overlaps(c.box, b) for b in barriers) and not any(_overlaps(c.box, f) for f in forbidden or [])
        ]
        if not candidates:
            break
        proposed = _cap(_extent([current] + [c.box for c in candidates]), start, cap, (height, width))
        proposed = _stop_before(proposed, current, barriers)
        if proposed.area <= current.area:
            break
        with_new = Lines(
            list(lines.horizontal) + [c for c in candidates if c.horizontal],
            list(lines.vertical) + [c for c in candidates if not c.horizontal],
            lines.horizontal_mask,
            lines.vertical_mask,
        )
        before = quality.foreign_text(ink, with_new, current, dpi)
        after = quality.foreign_text(ink, with_new, proposed, dpi)
        if after > before + MAX_FOREIGN_GAIN:
            logger.debug(
                "рост %s → %s отвергнут: чужого %.3f → %.3f", current.as_tuple(), proposed.as_tuple(), before, after
            )
            break
        gained = proposed.area - current.area
        current = proposed
        for c in candidates:
            if _centre_inside(c.box, current) and c.box not in owned_boxes:
                owned.append(c)
                owned_boxes.add(c.box)
        if gained < mm_to_px(1.0, dpi) * max(current.width, current.height):
            break
    return current, owned


# --- Схема -------------------------------------------------------------------------


def line_art_mask(binary: np.ndarray, dpi: int, lines: "Lines | None" = None) -> np.ndarray:
    """Краска, которая не буква: компоненты крупнее буквы по высоте или ширине.

    Сирые линейки во всю полосу (колонтитул, линия под шапкой страницы) из маски вырезаются:
    заливка схемы с дилатацией 5 мм прилипала к линейке колонтитула в 4 мм над схемой и
    делала рамку шириной с полосу (1973/07 с.66). Сирая — ни с чем не пересекается; во всю
    полосу — не короче ``PAGE_RULE_SPAN`` её ширины. У таблиц такие линейки и так не в ядре.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(binary)
    max_h = mm_to_px(TEXT_MAX_HEIGHT_MM, dpi)
    max_w = mm_to_px(TEXT_MAX_WIDTH_MM, dpi)
    big = (stats[:, cv2.CC_STAT_HEIGHT] > max_h) | (stats[:, cv2.CC_STAT_WIDTH] > max_w)
    big[0] = False
    art = np.where(big[labels], 255, 0).astype(np.uint8)
    if lines is not None:
        width = binary.shape[1]
        tolerance = mm_to_px(refine.CROSS_TOL_MM, dpi)
        pad = mm_to_px(0.5, dpi)
        for segment in lines.horizontal:
            if segment.length < PAGE_RULE_SPAN * width:
                continue
            if any(refine.crosses(vertical, segment, tolerance) for vertical in lines.vertical):
                continue
            rule = segment.box
            art[max(0, rule.y0 - pad) : rule.y1 + pad, max(0, rule.x0 - pad) : rule.x1 + pad] = 0
    return art


def grow_diagram(art: np.ndarray, box: Box, dpi: int, barriers: list[Box], figure: "Box | None" = None) -> Box:
    """Дорастить схему заливкой по штриховой краске с дилатацией.

    Блоки схемы связаны стрелками и линиями, порой пунктирными; дилатация на ``DIAGRAM_GLUE_MM``
    сшивает разрывы, а компонента раздутой маски, в которой лежит затравка, и есть схема
    целиком. Чужие находки — преграды: их площадь из маски вырезается. ``figure`` — Figure-блок
    surya, накрывающий затравку: рамка добирает и его.
    """
    height, width = art.shape[:2]
    glue = mm_to_px(DIAGRAM_GLUE_MM, dpi)
    cap = mm_to_px(DIAGRAM_CAP_MM, dpi)
    # Потолок отсчитывается от затравки ВМЕСТЕ с Figure-блоком surya: у сетевого графика
    # 1967/05 с.66 затравка — нижняя треть схемы, и потолок в 60 мм от неё резал верх схемы,
    # который surya уже показала.
    anchor = _extent([box, figure]) if figure is not None else box
    window = _cap(Box(0, 0, width, height), anchor, cap + glue, (height, width))
    local = art[window.slice].copy()
    for barrier in barriers:
        clipped = barrier.padded(mm_to_px(1.0, dpi)).clipped(width, height)
        common = intersection(clipped, window)
        if common is not None:
            local[common.y0 - window.y0 : common.y1 - window.y0, common.x0 - window.x0 : common.x1 - window.x0] = 0
    seed = box.clipped(width, height)
    cv2.rectangle(
        local, (seed.x0 - window.x0, seed.y0 - window.y0), (seed.x1 - window.x0 - 1, seed.y1 - window.y0 - 1), 255, -1
    )
    glued = cv2.dilate(local, np.ones((glue, glue), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(glued, 8)
    centre = ((seed.y0 + seed.y1) // 2 - window.y0, (seed.x0 + seed.x1) // 2 - window.x0)
    label = int(labels[min(centre[0], labels.shape[0] - 1), min(centre[1], labels.shape[1] - 1)])
    if label == 0:
        grown = seed
    else:
        # Габарит компоненты раздутой маски шире самой краски на дилатацию — снимаем её обратно
        # по фактической краске внутри компоненты.
        region = labels == label
        ink_rows = np.nonzero((local > 0) & region)
        if ink_rows[0].size:
            grown = Box(
                window.x0 + int(ink_rows[1].min()),
                window.y0 + int(ink_rows[0].min()),
                window.x0 + int(ink_rows[1].max()) + 1,
                window.y0 + int(ink_rows[0].max()) + 1,
            )
        else:
            grown = seed
    if figure is not None:
        grown = _extent([grown, figure])
        grown = _cap(grown, figure, mm_to_px(FIGURE_SLACK_MM, dpi), (height, width))
    grown = _cap(grown, anchor, cap, (height, width))
    return _stop_before(grown, seed, barriers)


# --- Конвейер ------------------------------------------------------------------------


def _layout_kind(layout: "LayoutBlocks | None", box: Box) -> tuple[str, "Box | None"]:
    """Вид по surya и блок, который его дал; пусто — surya не решает."""
    if layout is None:
        return "", None
    block = layout.covering(box, FIGURE_LABELS, LAYOUT_COVER)
    if block is not None:
        return KIND_DIAGRAM, block.box
    block = layout.covering(box, TABLE_LABELS, LAYOUT_COVER)
    if block is not None:
        return KIND_TABLE, block.box
    block = layout.covering(box, FORM_LABELS, LAYOUT_COVER)
    if block is not None:
        return KIND_FORM, block.box
    return "", None


def detect(
    gray: np.ndarray, dpi: int = WORK_DPI, verify_findings: bool = True, layout: "LayoutBlocks | None" = None
) -> list[TableBox]:
    """Таблицы, схемы и рисунки четвёртой версии. Координаты — в пикселях поданного изображения.

    ``layout`` — разметка surya в тех же пикселях (см. ``layout.surya.load``); без неё вид
    решают признаки решётки, а рост — только связность.
    """
    from ocr_utils.page_layout.tables import verify as verification

    height, width = gray.shape[:2]
    binary = binarize(gray)
    fragments = rules.fragments(binary, dpi)
    raw = rules.find_rules(gray, dpi, binary, fragments)
    lines = refine.drop_border_rules(raw, (height, width), dpi)
    seeds = cluster_tables(lines, dpi, binary, POLICY, grouping="cores")
    if not seeds:
        return []
    if not verify_findings:
        return seeds

    ink = _text_ink(binary, lines)
    components = quality.glyph_components(ink)
    art: "np.ndarray | None" = None

    def signs_of(box: Box):
        crop = gray[box.slice]
        return verification.features(crop, find_lines(crop, dpi), dpi)

    # Первый проход: вид и проверка. Растить можно только то, что признано чем-то.
    accepted: list[tuple[TableBox, Box, str, object, dict, "Box | None"]] = []
    for seed in seeds:
        rule_box = seed.box.clipped(width, height)
        if rule_box.width < 8 or rule_box.height < 8:
            continue
        owned = [s for s in list(lines.horizontal) + list(lines.vertical) if _centre_inside(s.box, rule_box)]
        kind_signs = kind_module.features_of(owned, rule_box, dpi)
        signs = signs_of(rule_box)
        long_rules = float(seed.metrics.get("long_rules", 0.0))
        found_kind, why = kind_module.classify(kind_signs, signs, rule_box, dpi, long_rules)
        layout_kind, block = _layout_kind(layout, rule_box)
        if layout_kind == KIND_FORM:
            layout_kind = KIND_DIAGRAM if kind_module.is_diagram(kind_signs, rule_box, dpi, strict=False) else ""
        if layout_kind == KIND_DIAGRAM and found_kind == KIND_DRAWING:
            # Figure у surya — и схема, и график, и чертёж. Что из них — решает краска:
            # мало букв и пустые ячейки — рисунок, иначе схема.
            layout_kind = KIND_DRAWING
        conflict = float(bool(layout_kind) and layout_kind != found_kind)
        if layout_kind:
            found_kind = layout_kind
        glyph_p90 = glyph_height_p90(components, rule_box, dpi)
        frame_only = kind_signs.n_horizontal + kind_signs.n_vertical <= FRAME_MAX_RULES
        if glyph_p90 >= BANNER_GLYPH_P90_MM and (rule_box.height <= mm_to_px(BANNER_MAX_HEIGHT_MM, dpi) or frame_only):
            # Заголовок рубрики в рамке или заголовок статьи в рамке (1968/01 с.10 и с.20,
            # 1969/12 с.91: рамка 40–60 мм с буквами 5–8 мм, линеек — только сама рамка).
            # Проверяется до вида: surya зовёт его Figure, и без этой проверки он уходил в
            # находки схемой или рисунком. Графики и чертежи под правило не подпадают: линеек
            # у них много.
            logger.debug(
                "находка %s отклонена: буквы высотой %.1f мм — заголовок в рамке", rule_box.as_tuple(), glyph_p90
            )
            continue
        if found_kind == KIND_TABLE:
            good, reason = verification.is_table_v3(signs)
            if good and seed.metrics.get("long_rules", 0.0) < MIN_LONG_RULES_FOR_TABLE:
                good, reason = False, "сквозных линеек меньше двух — это не таблица"
            if not good:
                # Не таблица. Если это россыпь коробок (много линеек, редкие пересечения) — схема,
                # пусть и с высокими боковыми коробками; иначе рамка вокруг текста, баннер,
                # колонтитул — отбрасывается, как в третьей версии. Figure surya спасает как рисунок.
                if layout_kind == KIND_DIAGRAM:
                    found_kind = KIND_DRAWING
                elif kind_module.is_diagram(kind_signs, rule_box, dpi, strict=False):
                    found_kind = KIND_DIAGRAM
                else:
                    logger.debug("находка %s отклонена: %s", rule_box.as_tuple(), reason)
                    continue
        extra = {**kind_signs.as_row(), "kind_conflict": conflict, "glyph_p90_mm": round(glyph_p90, 2)}
        accepted.append((seed, rule_box, found_kind, signs, extra, block))
    if not accepted:
        return []

    # Второй проход: рост по виду. Соседи известны только теперь. Таблице преграда — любая
    # соседняя находка; схеме и рисунку — только таблицы: две затравки одной схемы (верх с
    # пунктирными связями и низ) обязаны срастись, а не резать друг друга (1974/12 с.42).
    neighbours = [box for _, box, _, _, _, _ in accepted]
    table_boxes = [box for _, box, k, _, _, _ in accepted if k == KIND_TABLE]
    forbidden = layout.text_blocks() if layout is not None else []
    kept: list[TableBox] = []
    for seed, rule_box, found_kind, signs, extra, block in accepted:
        barriers = [b for b in (neighbours if found_kind == KIND_TABLE else table_boxes) if b != rule_box]
        grown = rule_box
        owned: list[Segment] = []
        if found_kind == KIND_TABLE:
            grown, owned = grow_table(
                lines, fragments, rule_box, ink, dpi, barriers, allowed=block, forbidden=forbidden
            )
            if grown != rule_box:
                signs = signs_of(grown)
            fitted = refine.fit_rows(grown, _with(lines, owned), ink, dpi)
            fitted = _stop_before(fitted, grown, barriers)
        else:
            if art is None:
                # Линейки ДО отсева краевых: линейка колонтитула ближе 10 мм к обрезу
                # выбрасывается ``drop_border_rules``, а из штриховой краски её всё равно надо убрать.
                art = line_art_mask(binary, dpi, raw)
            figure = block if block is not None and found_kind != KIND_TABLE else None
            grown = grow_diagram(art, rule_box, dpi, barriers, figure)
            fitted = grown
        pushed = refine.push_edges(fitted, components, _with(lines, owned), dpi, (height, width), ink)
        final_kind = found_kind
        if grown != rule_box:
            final_signs = kind_module.features_of(
                [s for s in list(lines.horizontal) + list(lines.vertical) + owned if _centre_inside(s.box, grown)],
                grown,
                dpi,
            )
            extra.update(final_signs.as_row())
            # Кусок блок-схемы проходит проверку как маленькая таблица, а дорастает по связям до
            # целой схемы — и тогда линейки уже говорят «коробки». Вид пересчитывается по
            # выросшей рамке; surya, если она есть, своё слово уже сказала.
            if (
                found_kind == KIND_TABLE
                and block is None
                and kind_module.is_diagram(final_signs, grown, dpi, long_rules)
            ):
                final_kind = KIND_DIAGRAM
        metrics = {**seed.metrics, **{key: float(value) for key, value in signs.as_row().items()}, **extra}
        metrics["verified"] = 1.0
        metrics["push_failed"] = float(pushed.failed)
        for side, value in pushed.moved_mm.items():
            metrics[f"push_{side}"] = round(value, 2)
        metrics["crossed"] = float(sum(quality.crossed_glyphs(components, pushed.box).values()))
        metrics["kind_h"] = float(kind_signs.n_horizontal)
        kept.append(
            TableBox(
                box=pushed.box,
                score=seed.score,
                source="table_detection",
                skew_deg=seed.skew_deg,
                metrics=metrics,
                rule_box=grown,
                kind=final_kind,
            )
        )
    merged = _merge_diagrams(kept, dpi)
    if len(merged) != len(kept):
        # Сросшаяся схема получила новую рамку — граница снова обязана обойти буквы.
        repushed: list[TableBox] = []
        for table in merged:
            if table.metrics.get("merged"):
                pushed = refine.push_edges(table.box, components, lines, dpi, (height, width), ink)
                metrics = dict(table.metrics)
                metrics["crossed"] = float(sum(quality.crossed_glyphs(components, pushed.box).values()))
                table = TableBox(
                    pushed.box,
                    table.score,
                    table.source,
                    table.origin,
                    table.skew_deg,
                    metrics,
                    table.rules,
                    table.kind,
                )
            repushed.append(table)
        merged = repushed
    return _deduplicate(merged)


def _merge_diagrams(tables: list[TableBox], dpi: int) -> list[TableBox]:
    """Срастить куски одной схемы и втянуть в схему ряд коробок, принятый за таблицу.

    Две находки-схемы (или рисунка), которые перекрываются или стоят ближе зазора заливки, — одна
    схема, выросшая из двух затравок. «Таблица» без единой внутренней горизонтали (ряд коробок
    под общей линией, 1974/12 с.40), стоящая вплотную к схеме, — её нижний ярус: одна строка
    таблицей не бывает, а ряд коробок в блок-схеме — сплошь и рядом.
    """
    glue = mm_to_px(DIAGRAM_GLUE_MM, dpi)

    def near(first: Box, second: Box) -> bool:
        return _overlaps(first.padded(glue), second)

    def single_row(table: TableBox) -> bool:
        return table.kind == KIND_TABLE and table.metrics.get("kind_h", 99.0) <= 2

    diagrams = [t for t in tables if t.kind != KIND_TABLE]
    others = [t for t in tables if t.kind == KIND_TABLE]
    merged = True
    while merged:
        merged = False
        for a in range(len(diagrams)):
            for b in range(a + 1, len(diagrams)):
                # Схема и рисунок тоже сращиваются: у чертежа контейнера (1969/11 с.66) одна
                # затравка с надписями зовётся схемой, две другие — рисунком, а объект один.
                # Вид берёт бо́льшая по площади.
                if near(diagrams[a].box, diagrams[b].box):
                    first, second = sorted((diagrams[a], diagrams[b]), key=lambda t: -t.box.area)
                    diagrams[a] = _union(first, second)
                    del diagrams[b]
                    merged = True
                    break
            if merged:
                break
        if merged:
            continue
        for index, table in enumerate(others):
            if not single_row(table):
                continue
            for d_index, diagram in enumerate(diagrams):
                if near(diagram.box, table.box):
                    diagrams[d_index] = _union(diagram, table)
                    del others[index]
                    merged = True
                    break
            if merged:
                break
    return others + diagrams


def _union(first: TableBox, second: TableBox) -> TableBox:
    box = _extent([first.box, second.box])
    rules = _extent([first.rules, second.rules])
    metrics = dict(first.metrics)
    metrics["merged"] = metrics.get("merged", 0.0) + 1.0
    metrics["crossed"] = 0.0
    return TableBox(box, first.score, first.source, first.origin, first.skew_deg, metrics, rules, first.kind)


def _with(lines: Lines, extra: list[Segment]) -> Lines:
    if not extra:
        return lines
    known = {s.box for s in list(lines.horizontal) + list(lines.vertical)}
    fresh = [s for s in extra if s.box not in known]
    return Lines(
        list(lines.horizontal) + [s for s in fresh if s.horizontal],
        list(lines.vertical) + [s for s in fresh if not s.horizontal],
        lines.horizontal_mask,
        lines.vertical_mask,
    )


def _deduplicate(tables: list[TableBox]) -> list[TableBox]:
    """Убрать находки, съеденные соседкой того же вида после роста."""
    ordered = sorted(tables, key=lambda item: -item.box.area)
    kept: list[TableBox] = []
    for table in ordered:
        swallowed = False
        for bigger in kept:
            if bigger.kind != table.kind:
                continue
            overlap = intersection(bigger.box, table.box)
            if overlap is not None and overlap.area >= 0.8 * table.box.area:
                swallowed = True
                break
        if not swallowed:
            kept.append(table)
    return sorted(kept, key=lambda item: item.box.y0)
