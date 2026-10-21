"""Масштабы v2 от плотности съёмки: px/мм кадра → масштаб сегментации и защиты глифов, припуски в px кадра.

ЕДИНИЦЫ. Размеры v2 заданы либо в мм бумаги (опции CLI, ``border_fill``, ``finger_shadow``, ``side_fit``), либо в px
КАЛИБРОВОЧНОГО кадра (``REF_PX_PER_MM`` = 13.8 px/мм) или копий, приведённых к его плотности (сегментация, защита
глифов, копия закраса ¼). Второе — та же независимая от разрешения единица: px калибровочного кадра = мм × 13.8, и в
px кадра любой плотности (350, 450, 600 dpi) она переводится через :meth:`Scale.frame_px`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ocr_utils.scan_cropping.v2.background import BG_SCALE
from ocr_utils.scan_cropping.v2.glyphs import GLYPH_SCALE
from ocr_utils.units import MM_PER_INCH

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
        px_per_mm: Плотность кадра, px/мм бумаги (мерить заранее и задавать в run-скрипте: ``--px-per-mm`` или
            ``--dpi``).
    """

    px_per_mm: float = REF_PX_PER_MM

    @classmethod
    def from_dpi(cls, dpi: float) -> "Scale":
        """Масштабы по разрешению бумаги на кадре.

        Args:
            dpi: Сколько пикселей кадра приходится на дюйм бумаги (450 у новых паков).

        Returns:
            :class:`Scale` с ``px_per_mm = dpi / 25.4``.
        """
        return cls(px_per_mm=dpi / MM_PER_INCH)

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

    def frame_odd_px(self, px: float) -> int:
        """То же, что :meth:`frame_px`, но нечётное и не меньше 3 — для ядер медианы и эллипса OpenCV.

        Args:
            px: Нечётная сторона ядра на калибровочном кадре, px.

        Returns:
            Нечётная сторона на этом кадре, px (на калибровочном кадре — ровно ``px``).
        """
        side = max(3, int(round(px * self.ratio)))
        return side if side % 2 else side + 1

    def mm_px(self, mm: float, minimum: int = 1) -> int:
        """Миллиметры бумаги → px этого кадра.

        Args:
            mm: Длина на бумаге, мм.
            minimum: Нижняя граница результата (0 — для опций, где ноль значит «выключено»).

        Returns:
            Длина в px кадра.
        """
        return max(minimum, int(round(mm * self.px_per_mm)))

    def mm_px_float(self, mm: float) -> float:
        """Миллиметры бумаги → px этого кадра без округления (для сигм размытия).

        Args:
            mm: Длина на бумаге, мм.

        Returns:
            Длина в px кадра.
        """
        return mm * self.px_per_mm

    def seg_px(self, mm: float) -> int:
        """Миллиметры бумаги → px масштаба сегментации (его плотность постоянна: ``REF_PX_PER_MM × BG_SCALE``).

        Args:
            mm: Длина на бумаге, мм.

        Returns:
            Длина в px масштаба сегментации (не меньше 1).
        """
        return max(1, int(round(mm * REF_PX_PER_MM * BG_SCALE)))
