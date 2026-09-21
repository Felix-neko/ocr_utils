"""Блоки surya layout: метка, уверенность, рамка и полигон в пикселях кадра, который подавали модели.

ЗАЧЕМ. Детектор по линейкам точен в геометрии, но не знает, ЧТО обвёл: кусок блок-схемы
для него — маленькая таблица. Surya смотрит на полосу целиком и отвечает на другой вопрос —
где здесь таблица, где рисунок, где текст. Эксперимент на 190 размеченных полосах: все
двенадцать «кусков блок-схемы» и все диаграммы получили целый Figure/Form-блок, на
перекошенных таблицах Table-блок шире нашей рамки ровно там, где нам не хватало графы, на
«тексте рядом» он кончается на 15–60 мм выше. Слабости тоже замерены: 10 из 38 полос «не
таблицы» помечены Table, бланк целиком бывает одним Form без таблицы внутри, границы ±1–2 мм;
штриховой рисунок surya зовёт ``Picture`` на 28 полосах из 31. Поэтому surya — источник ВИДА и
подсказки протяжённости, а не точных границ; блок — предложение, которое проверяют пиксели.

Здесь только ТИПЫ и разбор сырого ответа модели; сама модель — :mod:`.model`, кэш — :mod:`.cache`.
Модуль не импортирует surya: блоки читаются из кэша в воркерах пула, где torch не нужен.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ocr_utils.page_layout.geometry import Box, intersection

# Классы surya по назначению. ``Picture`` — фотография (растр), но и штрих она зовёт так же;
# ``Figure`` — графики, схемы, чертежи; ``Form`` — бланк; ``Equation`` — формула.
PICTURE_LABELS = ("Picture",)
FIGURE_LABELS = ("Figure", "Picture")
FORM_LABELS = ("Form",)
TABLE_LABELS = ("Table",)
EQUATION_LABELS = ("Equation",)
TEXT_LABELS = ("Text", "ListItem", "Footnote", "Caption", "SectionHeader", "PageHeader", "PageFooter", "TextInlineMath")

# Ниже этой уверенности блоку не верим ни как виду, ни как подсказке границ.
MIN_CONFIDENCE = 0.3

Point = tuple[float, float]


@dataclass(frozen=True)
class Block:
    """Один блок: метка surya, уверенность 0..1, рамка и четырёхугольник в пикселях кадра."""

    label: str
    confidence: float
    box: Box
    polygon: tuple[Point, Point, Point, Point] | None = None

    @property
    def is_picture(self) -> bool:
        return self.label in PICTURE_LABELS

    @property
    def is_figure(self) -> bool:
        return self.label in FIGURE_LABELS

    @property
    def is_form(self) -> bool:
        return self.label in FORM_LABELS

    @property
    def is_table(self) -> bool:
        return self.label in TABLE_LABELS

    @property
    def is_equation(self) -> bool:
        return self.label in EQUATION_LABELS

    @property
    def is_text(self) -> bool:
        return self.label in TEXT_LABELS

    def scaled(self, factor: float) -> "Block":
        polygon = None
        if self.polygon is not None:
            polygon = tuple((x * factor, y * factor) for x, y in self.polygon)  # type: ignore[assignment]
        return Block(self.label, self.confidence, self.box.scaled(factor), polygon)

    def to_json(self) -> dict:
        payload = {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "box": list(self.box.as_tuple()),
        }
        if self.polygon is not None:
            payload["polygon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in self.polygon]
        return payload

    @classmethod
    def from_json(cls, payload: dict) -> "Block":
        polygon = payload.get("polygon")
        return cls(
            str(payload["label"]),
            float(payload["confidence"]),
            Box(*(int(v) for v in payload["box"])),
            tuple((float(x), float(y)) for x, y in polygon) if polygon else None,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class LayoutBlocks:
    """Блоки одной страницы в пикселях кадра ``width`` × ``height``, который подавался модели."""

    blocks: tuple[Block, ...]
    width: int
    height: int

    def scaled(self, factor: float) -> "LayoutBlocks":
        return LayoutBlocks(
            tuple(block.scaled(factor) for block in self.blocks),
            round(self.width * factor),
            round(self.height * factor),
        )

    def scaled_to(self, width: int, height: int) -> "LayoutBlocks":
        """Блоки в пикселях кадра ``width`` × ``height`` (копии одной страницы расходятся на пиксель)."""
        if (self.width, self.height) == (width, height):
            return self
        return self.scaled(width / max(1, self.width))

    def by_label(self, labels: Sequence[str], min_confidence: float = MIN_CONFIDENCE) -> list[Block]:
        """Блоки заданных меток не ниже ``min_confidence``."""
        return [b for b in self.blocks if b.label in labels and b.confidence >= min_confidence]

    def boxes(self, labels: Sequence[str], min_confidence: float = MIN_CONFIDENCE) -> list[Box]:
        """Рамки блоков заданных меток, зажатые в кадр."""
        return [b.box.clipped(self.width, self.height) for b in self.by_label(labels, min_confidence)]

    def covering(self, box: Box, labels: Sequence[str], share: float = 0.7) -> "Block | None":
        """Самый уверенный блок нужного вида, накрывающий рамку не меньше чем на ``share``."""
        best: Block | None = None
        for block in self.blocks:
            if block.label not in labels or block.confidence < MIN_CONFIDENCE:
                continue
            common = intersection(block.box, box)
            if common is None or common.area < share * max(1, box.area):
                continue
            if best is None or block.confidence > best.confidence:
                best = block
        return best

    def text_blocks(self) -> list[Box]:
        return [b.box for b in self.blocks if b.is_text and b.confidence >= MIN_CONFIDENCE]

    def to_json(self) -> dict:
        return {"width": self.width, "height": self.height, "blocks": [b.to_json() for b in self.blocks]}

    @classmethod
    def from_json(cls, payload: dict) -> "LayoutBlocks":
        return cls(
            tuple(Block.from_json(item) for item in payload.get("blocks", [])),
            int(payload["width"]),
            int(payload["height"]),
        )


def from_surya_result(result: object, width: int, height: int) -> LayoutBlocks:
    """Сырой ответ surya (``LayoutResult``) → блоки в пикселях кадра ``width`` × ``height``.

    Рамки зажимаются в кадр: surya отдаёт координаты в пикселях поданной картинки, но на
    полпикселя за край выходит регулярно, а ``Box`` вывернутой рамки не терпит. Полигон
    сохраняется как есть — он нужен защите контента при кадрировании.
    """
    blocks = []
    for item in getattr(result, "bboxes", ()):
        x0, y0, x1, y1 = (int(round(v)) for v in item.bbox)
        x0, y0 = max(0, min(x0, width)), max(0, min(y0, height))
        x1, y1 = max(x0, min(x1, width)), max(y0, min(y1, height))
        polygon = getattr(item, "polygon", None)
        points = tuple((float(x), float(y)) for x, y in polygon) if polygon is not None and len(polygon) == 4 else None
        blocks.append(Block(str(item.label), float(item.confidence), Box(x0, y0, x1, y1), points))  # type: ignore[arg-type]
    return LayoutBlocks(tuple(blocks), width, height)


def from_boxes(items: Iterable[tuple[str, float, Box]], width: int, height: int) -> LayoutBlocks:
    """Блоки из троек ``(метка, уверенность, рамка)`` — для тестов и импорта старых кэшей."""
    return LayoutBlocks(tuple(Block(label, confidence, box) for label, confidence, box in items), width, height)


__all__ = [
    "Block",
    "EQUATION_LABELS",
    "FIGURE_LABELS",
    "FORM_LABELS",
    "LayoutBlocks",
    "MIN_CONFIDENCE",
    "PICTURE_LABELS",
    "TABLE_LABELS",
    "TEXT_LABELS",
    "from_boxes",
    "from_surya_result",
]
