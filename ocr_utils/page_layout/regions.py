"""Результат разбора: область страницы с видом, уверенностью, источником и подробностями детектора."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from ocr_utils.page_layout.geometry import Box


class RegionKind(str, Enum):
    """Виды областей; значения — те же строки, что в колонке ``kind`` базы разметки (``db.models``)."""

    COLOR = "color"  # цветной растр (фотография, цветная плашка)
    GRAYSCALE = "grayscale"  # серый растр
    STAMP_SUSPECT = "stamp_suspect"  # подозрение на библиотечную печать
    COLOR_TEXT = "color_text"  # цветной набор (только руками)
    TABLE = "table"  # таблица с линейками
    LINE_ART = "line_art_schema"  # схема, чертёж, график, рисунок штрихом
    ROTATED_TEXT = "rotated_text"  # повёрнутый текст вне таблиц

    @property
    def is_raster(self) -> bool:
        return self in (RegionKind.COLOR, RegionKind.GRAYSCALE, RegionKind.STAMP_SUSPECT, RegionKind.COLOR_TEXT)


@dataclass(frozen=True)
class Region:
    """Область в пикселях кадра, по которому шёл разбор (родное разрешение ``PageImage``).

    ``confidence`` — 0..1 там, где детектор её даёт (доля источников, уверенность surya), иначе
    ``None``; ``source`` — кто нашёл (``raster``, ``tables``, ``line_art``, ``rotated_text``);
    ``info`` — подробности детектора, уходят в ``detector_info`` базы как JSON.
    """

    box: Box
    kind: RegionKind
    confidence: float | None = None
    source: str = ""
    full_page: bool = False
    info: dict = field(default_factory=dict)

    def at_scale(self, factor: float) -> "Region":
        return Region(self.box.scaled(factor), self.kind, self.confidence, self.source, self.full_page, self.info)

    def detector_info_json(self) -> str:
        payload = dict(self.info)
        if self.confidence is not None:
            payload.setdefault("confidence", round(float(self.confidence), 4))
        return json.dumps(payload, ensure_ascii=False)


def boxes_of(regions: Iterable[Region]) -> list[Box]:
    return [region.box for region in regions]


__all__ = ["Region", "RegionKind", "boxes_of"]
