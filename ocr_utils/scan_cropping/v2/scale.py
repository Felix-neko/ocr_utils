"""Масштабы v2 от плотности съёмки: px/мм кадра → масштаб сегментации и защиты глифов, припуски в px кадра."""

from __future__ import annotations

from dataclasses import dataclass

from ocr_utils.scan_cropping.v2.background import BG_SCALE
from ocr_utils.scan_cropping.v2.glyphs import GLYPH_SCALE

# Плотность кадров, на которых калиброваны все размеры v2 (подшивки МТО 1990, пак-6, кадр 5875×4570): ~6.9 px/мм
# на масштабе сегментации ½ (``research/bound_volumes/book_mask.py``), то есть ~13.8 px/мм в кадре.
REF_PX_PER_MM = 13.8


@dataclass(frozen=True)
class Scale:
    """Масштабы прогона по плотности кадра.

    Все размеры сегментации (окна, радиусы, полосы в px) заданы на масштабе ½ кадров МТО 1990. Чтобы не пересчитывать
    десятки констант, кадр другой плотности уменьшается так, чтобы на масштабе сегментации px/мм совпали с
    калибровочными; так же — копия для защиты глифов. Припуски закраса пальцев в px кадра умножаются на отношение.

    Поля:
        px_per_mm: Плотность кадра, px/мм бумаги (мерить заранее и задавать в run-скрипте).
    """

    px_per_mm: float = REF_PX_PER_MM

    @property
    def ratio(self) -> float:
        """Отношение плотности кадра к калибровочной."""
        return self.px_per_mm / REF_PX_PER_MM

    @property
    def seg(self) -> float:
        """Масштаб сегментации относительно кадра (для калибровочных кадров — ½)."""
        return BG_SCALE / self.ratio

    @property
    def glyph(self) -> float:
        """Масштаб копии для защиты глифов относительно кадра (для калибровочных кадров — ½)."""
        return GLYPH_SCALE / self.ratio

    def frame_px(self, px: float) -> int:
        """Размер в px калибровочного кадра → px этого кадра.

        Args:
            px: Размер на калибровочном кадре, px.

        Returns:
            Размер на этом кадре, px (не меньше 1).
        """
        return max(1, int(round(px * self.ratio)))
