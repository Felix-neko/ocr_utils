"""Меры качества РАМКИ находки — без разметки, а потому применимые ко всему паку.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. Разметка отвечает на вопрос «таблица ли это», и её 163 полосы
кончаются быстро. А вопросы «не режет ли рамка буквы», «не захватила ли она чужой абзац»,
«не осталась ли половина таблицы снаружи» — геометрические, и ответ на них считается по самой
полосе. Значит их можно мерить на всех 12 135 полосах и сравнивать версии детектора между
собой там, где размеченного эталона нет и не будет.

КРАСКА ЗДЕСЬ — ЭТО ``structure.ruling_grid.text_ink``: бинаризация минус раздутая маска
линеек. Именно минус линейки: рамка таблицы ОБЯЗАНА идти по линейке, и мера, считающая
линейку за краску, показала бы 100% «разрезанных» границ у всех находок подряд (проверено:
первый вариант замера так и дал 84% там, где на деле проблема у 11% сторон).

``verify.glyph_mask`` для этого не годится, хотя и соблазнителен: его пороги
размера глифа калиброваны на 300 dpi, а детектор работает на 150, где петит падает до 7 px и
буква перестаёт отличаться от точки над «й».
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.scan_markup.table_detection.ruling import Lines, mm_to_px
from ocr_utils.page_layout.geometry import Box

SIDES = ("сверху", "снизу", "слева", "справа")

# Ниже этой доли краски под границей сторона считается чистой. 1% ширины стороны — это
# примерно одна буква на строку в сто знаков, то есть уже не «граница режет текст», а пылинка.
CLEAN_SHARE = 0.01

# Насколько далеко искать чистый просвет. 12 мм — половина минимальной стороны таблицы:
# дальше искать бессмысленно, там уже другой элемент вёрстки.
SEARCH_MM = 12.0

# Ореол вокруг рамки, в котором ищутся не попавшие в неё линейки. 15 мм выбраны по замеру:
# у десяти полос, где детектор взял лишь кусок блок-схемы, зазоры до оставшихся фрагментов
# лежат в 6-15 мм.
HALO_MM = 15.0


@dataclass(frozen=True)
class BoxQuality:
    """Четыре меры одной рамки. Все — в долях или в миллиметрах, не в пикселях."""

    edge_ink: dict[str, float]  # доля краски-без-линеек под каждой стороной
    clearance_mm: dict[str, float]  # на сколько мм двигать сторону до чистого просвета
    outside_rules: int  # линеек в ореоле, не вошедших в рамку
    foreign_text: float  # доля краски внутри рамки, лежащей в полосах без единой линейки

    @property
    def dirty_sides(self) -> int:
        return sum(1 for value in self.edge_ink.values() if value > CLEAN_SHARE)

    def as_row(self) -> dict[str, float]:
        row: dict[str, float] = {f"edge_ink_{side}": round(value, 4) for side, value in self.edge_ink.items()}
        row.update({f"clearance_{side}": round(value, 2) for side, value in self.clearance_mm.items()})
        row["dirty_sides"] = float(self.dirty_sides)
        row["outside_rules"] = float(self.outside_rules)
        row["foreign_text"] = round(self.foreign_text, 4)
        return row


HEADER = tuple(
    [f"edge_ink_{side}" for side in SIDES]
    + [f"clearance_{side}" for side in SIDES]
    + ["dirty_sides", "outside_rules", "foreign_text"]
)


def _profiles(ink: np.ndarray, box: Box) -> dict[str, "tuple[np.ndarray, int, int]"]:
    """По стороне: профиль доли краски поперёк рамки, координата стороны и шаг наружу."""
    columns = ink[:, box.x0 : box.x1]
    rows = ink[box.y0 : box.y1, :]
    return {
        "сверху": (columns.mean(axis=1), box.y0, -1),
        "снизу": (columns.mean(axis=1), box.y1 - 1, +1),
        "слева": (rows.mean(axis=0), box.x0, -1),
        "справа": (rows.mean(axis=0), box.x1 - 1, +1),
    }


def edge_ink(ink: np.ndarray, box: Box) -> dict[str, float]:
    """Доля краски-без-линеек ровно под каждой из четырёх сторон рамки."""
    result: dict[str, float] = {}
    for side, (profile, position, _) in _profiles(ink > 0, box).items():
        result[side] = float(profile[position]) if 0 <= position < profile.size else 0.0
    return result


def clearance(ink: np.ndarray, box: Box, dpi: int) -> dict[str, float]:
    """На сколько миллиметров двигать сторону до ближайшего чистого места.

    Знак говорит куда: плюс — наружу, минус — внутрь. Ноль — сторона уже чиста. ``inf``
    (в виде ``SEARCH_MM``) — чистого места нет ни в ту, ни в другую сторону; такие случаи
    редки (один из восьмисот на замере), но их надо видеть, а не прятать в нуле.
    """
    limit = mm_to_px(SEARCH_MM, dpi)
    scale = 25.4 / dpi
    result: dict[str, float] = {}
    for side, (profile, position, outward) in _profiles(ink > 0, box).items():
        if not 0 <= position < profile.size or profile[position] <= CLEAN_SHARE:
            result[side] = 0.0
            continue
        out = _first_clean(profile, position, outward, limit)
        inside = _first_clean(profile, position, -outward, limit)
        if out is None and inside is None:
            result[side] = SEARCH_MM
        elif inside is None or (out is not None and out <= inside):
            result[side] = out * scale
        else:
            result[side] = -inside * scale
    return result


def _first_clean(profile: np.ndarray, start: int, step: int, limit: int) -> "int | None":
    for distance in range(1, limit + 1):
        index = start + step * distance
        if not 0 <= index < profile.size:
            return None
        if profile[index] <= CLEAN_SHARE:
            return distance
    return None


def outside_rules(lines: Lines, box: Box, dpi: int, halo_mm: float = HALO_MM) -> int:
    """Сколько линеек лежит в ореоле вокруг рамки, не попав в неё.

    Мера недобранной таблицы и обрезанной блок-схемы: у целой находки рядом пусто, у куска
    вокруг остались рёбра. Линейка считается «не попавшей», если её габарит пересекается с
    рамкой меньше чем на треть своей площади.
    """
    halo = mm_to_px(halo_mm, dpi)
    grown = Box(box.x0 - halo, box.y0 - halo, box.x1 + halo, box.y1 + halo)
    count = 0
    for segment in list(lines.horizontal) + list(lines.vertical):
        rule = segment.box
        if _overlap(rule, grown) <= 0:
            continue
        if _overlap(rule, box) < 0.3 * max(1, rule.area):
            count += 1
    return count


def outside_rules_page(lines: Lines, boxes: "list[Box]", dpi: int, halo_mm: float = HALO_MM) -> int:
    """То же, но по всей полосе сразу: линейка не в счёт, если попала ХОТЬ В ОДНУ рамку.

    Считать по каждой находке отдельно и складывать нельзя, и это не мелочь: когда детектор
    начинает находить на полосе три таблицы вместо одной, линейки второй и третьей становятся
    «снаружи» для первой, и мера растёт именно там, где стало лучше. На размеченных полосах
    такой подсчёт давал 178 против 214 — то есть говорил об ухудшении там, где число найденных
    таблиц выросло с 96 до 132.
    """
    if not boxes:
        return 0
    halo = mm_to_px(halo_mm, dpi)
    grown = [Box(box.x0 - halo, box.y0 - halo, box.x1 + halo, box.y1 + halo) for box in boxes]
    count = 0
    for segment in list(lines.horizontal) + list(lines.vertical):
        rule = segment.box
        if not any(_overlap(rule, item) > 0 for item in grown):
            continue
        if any(_overlap(rule, box) >= 0.3 * max(1, rule.area) for box in boxes):
            continue
        count += 1
    return count


def _overlap(first: Box, second: Box) -> int:
    width = min(first.x1, second.x1) - max(first.x0, second.x0)
    height = min(first.y1, second.y1) - max(first.y0, second.y0)
    return width * height if width > 0 and height > 0 else 0


def foreign_text(ink: np.ndarray, lines: Lines, box: Box, dpi: int) -> float:
    """Доля краски внутри рамки, лежащая в строках без единой вертикальной линейки.

    Это мера «захватили кусок текста рядом»: у таблицы текст стоит между вертикалями граф, а
    прихваченный абзац лежит там, где вертикалей нет вовсе. Считается по строкам, а не по
    колонкам, потому что чужое приезжает сверху и снизу — сбоку места нет.
    """
    inside = (ink[box.slice] > 0).astype(np.float32)
    total = float(inside.sum())
    if total <= 0:
        return 0.0
    supported = np.zeros(box.height, dtype=bool)
    for segment in lines.vertical:
        rule = segment.box
        centre = (rule.x0 + rule.x1) // 2
        if not box.x0 <= centre < box.x1:
            continue
        low = max(0, rule.y0 - box.y0)
        high = min(box.height, rule.y1 - box.y0)
        if high > low:
            supported[low:high] = True
    return float(inside[~supported].sum() / total)


def measure(gray: np.ndarray, lines: Lines, box: Box, dpi: int) -> BoxQuality:
    """Все четыре меры разом. ``gray`` — вся полоса, ``box`` — рамка в её координатах."""
    from ocr_utils.scan_markup.table_detection.grid import text_ink

    ink = text_ink(gray, lines)
    clipped = box.clipped(gray.shape[1], gray.shape[0])
    return BoxQuality(
        edge_ink=edge_ink(ink, clipped),
        clearance_mm=clearance(ink, clipped, dpi),
        outside_rules=outside_rules(lines, clipped, dpi),
        foreign_text=foreign_text(ink, lines, clipped, dpi),
    )


# --- 5. Компоненты краски, пересечённые границей ------------------------------------------

# Компонента мельче этого — пыль скана, а не буква; такие границе пересекать можно.
MIN_GLYPH_AREA_PX = 3


def glyph_components(ink: np.ndarray) -> np.ndarray:
    """Габариты компонент краски-без-линеек: массив ``(x0, y0, x1, y1)`` по строкам.

    Считается один раз на полосу и подаётся всем находкам: компоненты те же, меняется только
    рамка. Пыль мельче ``MIN_GLYPH_AREA_PX`` выброшена.
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    if count <= 1:
        return np.zeros((0, 4), dtype=np.int64)
    stats = stats[1:]
    keep = stats[:, cv2.CC_STAT_AREA] >= MIN_GLYPH_AREA_PX
    left = stats[keep, cv2.CC_STAT_LEFT]
    top = stats[keep, cv2.CC_STAT_TOP]
    return np.stack(
        [left, top, left + stats[keep, cv2.CC_STAT_WIDTH], top + stats[keep, cv2.CC_STAT_HEIGHT]], axis=1
    ).astype(np.int64)


def straddling(components: np.ndarray, box: Box, side: str) -> np.ndarray:
    """Компоненты, которые линия стороны рамки пересекает: лежат и внутри, и снаружи."""
    if components.size == 0:
        return components
    x0, y0, x1, y1 = components.T
    if side == "сверху":
        hit = (y0 < box.y0) & (y1 > box.y0) & (x1 > box.x0) & (x0 < box.x1)
    elif side == "снизу":
        hit = (y0 < box.y1) & (y1 > box.y1) & (x1 > box.x0) & (x0 < box.x1)
    elif side == "слева":
        hit = (x0 < box.x0) & (x1 > box.x0) & (y1 > box.y0) & (y0 < box.y1)
    else:
        hit = (x0 < box.x1) & (x1 > box.x1) & (y1 > box.y0) & (y0 < box.y1)
    return components[hit]


def crossed_glyphs(components: np.ndarray, box: Box) -> dict[str, int]:
    """Сколько компонент краски режет каждая сторона рамки.

    Это мера четвёртой версии вместо доли краски под границей: доля в 1 % — это три срезанные
    первые буквы на высокой таблице (1973/12 с.86), а человек считает недопустимой и одну.
    Ноль на всех сторонах — единственный приемлемый ответ.
    """
    return {side: int(straddling(components, box, side).shape[0]) for side in SIDES}


__all__ = [
    "BoxQuality",
    "HEADER",
    "SIDES",
    "clearance",
    "crossed_glyphs",
    "edge_ink",
    "foreign_text",
    "glyph_components",
    "measure",
    "outside_rules",
    "straddling",
]
