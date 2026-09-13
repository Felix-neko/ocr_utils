"""Разметка surya layout: блоки полосы с видом (Table, Figure, Form, Text…).

ЗАЧЕМ. Детектор по линейкам точен в геометрии, но не знает, ЧТО обвёл: кусок блок-схемы
для него — маленькая таблица. Surya смотрит на полосу целиком и отвечает на другой вопрос —
где здесь таблица, где рисунок, где текст. Эксперимент на 190 размеченных полосах: все
двенадцать «кусков блок-схемы» и все диаграммы получили целый Figure/Form-блок, на
перекошенных таблицах Table-блок шире нашей рамки ровно там, где нам не хватало графы, на
«тексте рядом» он кончается на 15–60 мм выше. Слабости тоже замерены: 10 из 38 полос «не
таблицы» помечены Table, бланк целиком бывает одним Form без таблицы внутри, границы ±1–2 мм.
Поэтому surya — источник ВИДА и подсказки протяжённости, а не точных границ.

Здесь только ТИПЫ и разбор сырого ответа модели. Саму модель зовёт этап ``detect``
(``background_smoothing.layout.LayoutDetector``, GPU только в родителе), а кэш ответов на
диске держит ``scan_markup.detection.layout_cache``: детектор таблиц получает готовую
``PageLayout`` в пикселях той же копии полосы, по которой ищет линейки, или ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ocr_utils.scan_markup.table_detection.geometry import Box, intersection

# Метки surya, которые означают «это не текст, а нечто с линиями»: рисунок, схема, бланк.
FIGURE_LABELS = ("Figure", "Picture")
FORM_LABELS = ("Form",)
TABLE_LABELS = ("Table",)
TEXT_LABELS = ("Text", "ListItem", "Footnote", "Caption", "SectionHeader", "PageHeader", "PageFooter", "TextInlineMath")

# Ниже этой уверенности блоку не верим ни как виду, ни как подсказке границ.
MIN_CONFIDENCE = 0.3

# Сколько полос подаётся модели за раз.
BATCH = 8


@dataclass(frozen=True)
class Block:
    label: str
    confidence: float
    box: Box

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
    def is_text(self) -> bool:
        return self.label in TEXT_LABELS


@dataclass(frozen=True)
class PageLayout:
    """Блоки одной полосы в пикселях той копии, что подавалась модели (``width`` × ``height``)."""

    blocks: tuple[Block, ...]
    width: int
    height: int

    def scaled(self, factor: float) -> "PageLayout":
        return PageLayout(
            tuple(Block(b.label, b.confidence, b.box.scaled(factor)) for b in self.blocks),
            round(self.width * factor),
            round(self.height * factor),
        )

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
        return {
            "width": self.width,
            "height": self.height,
            "blocks": [
                {"label": b.label, "confidence": round(b.confidence, 4), "box": list(b.box.as_tuple())}
                for b in self.blocks
            ],
        }

    @staticmethod
    def from_json(payload: dict) -> "PageLayout":
        return PageLayout(
            tuple(
                Block(str(item["label"]), float(item["confidence"]), Box(*(int(v) for v in item["box"])))
                for item in payload.get("blocks", [])
            ),
            int(payload["width"]),
            int(payload["height"]),
        )


def from_surya_result(result: object, width: int, height: int) -> PageLayout:
    """Сырой ответ surya (``LayoutResult``) -> наша разметка в пикселях картинки ``width`` x ``height``.

    Рамки зажимаются в кадр: surya отдаёт координаты в пикселях поданной картинки, но на
    полпикселя за край выходит регулярно, а ``Box`` вывернутой рамки не терпит.
    """
    blocks = []
    for item in getattr(result, "bboxes", ()):
        x0, y0, x1, y1 = (int(round(v)) for v in item.bbox)
        x0, y0 = max(0, min(x0, width)), max(0, min(y0, height))
        x1, y1 = max(x0, min(x1, width)), max(y0, min(y1, height))
        blocks.append(Block(str(item.label), float(item.confidence), Box(x0, y0, x1, y1)))
    return PageLayout(tuple(blocks), width, height)


__all__ = ["Block", "PageLayout", "from_surya_result"]
