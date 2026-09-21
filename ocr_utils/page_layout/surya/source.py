"""Откуда детекторы берут блоки surya: кэш, модель или ничего — и что делать при промахе.

Три режима, и они не смешиваются молча:

* ``SuryaSource(cache, model)`` — родительский процесс: промах кэша досчитывается моделью и
  пишется в кэш. Так работает ``page_layout analyze`` и однопроцессные прогоны.
* ``SuryaSource(cache, on_miss=OnMiss.FAIL)`` — воркер пула: кэш только для чтения, промах —
  исключение :class:`SuryaMissing`. Тихо разбирать страницу без surya нельзя: результат был
  бы другого детектора, а никто бы этого не заметил. Кэш перед пулом набивает ``prefill``.
* ``SuryaSource(None)`` или ``on_miss=OnMiss.SKIP`` — осознанный отказ от surya (``--no-surya``,
  тесты): детекторы работают по одним пикселям и помечают результат ``surya_used=False``.
"""

from __future__ import annotations

from enum import Enum

from ocr_utils.page_layout.image import PageImage
from ocr_utils.page_layout.surya.blocks import LayoutBlocks
from ocr_utils.page_layout.surya.cache import SuryaCache


class OnMiss(str, Enum):
    """Что делать, когда блоков нет ни в кэше, ни у модели."""

    FAIL = "fail"  # исключение: кэш должен был быть набит заранее
    SKIP = "skip"  # работать без surya, помечая это в результате


class SuryaMissing(RuntimeError):
    """В кэше нет блоков для страницы, а модели в этом процессе нет — кэш не набит."""


class SuryaSource:
    """Источник блоков surya для одной страницы: кэш → модель → политика промаха."""

    def __init__(self, cache: SuryaCache | None = None, model=None, on_miss: OnMiss = OnMiss.FAIL) -> None:
        """
        Args:
            cache: Кэш на диске или ``None``.
            model: :class:`~ocr_utils.page_layout.surya.model.SuryaLayoutModel` или ``None`` (воркер пула).
            on_miss: Политика, когда ни кэша, ни модели: см. :class:`OnMiss`.
        """
        self.cache = cache
        self.model = model
        self.on_miss = OnMiss(on_miss)

    @property
    def enabled(self) -> bool:
        """Есть ли откуда взять блоки вообще (иначе surya выключена осознанно)."""
        return self.cache is not None or self.model is not None

    def resolve(self, image: PageImage) -> LayoutBlocks | None:
        """Блоки страницы в пикселях её кадра surya; ``None`` — surya выключена или промах при ``SKIP``.

        Raises:
            SuryaMissing: Промах кэша без модели при политике ``FAIL``.
        """
        if not self.enabled:
            return None
        if self.cache is not None:
            blocks = self.cache.load(image)
            if blocks is not None:
                return blocks
        if self.model is not None:
            blocks = self.model.predict_one(image.surya_frame)
            if self.cache is not None:
                self.cache.save(image, blocks, self.model.name())
            return blocks
        if self.on_miss is OnMiss.SKIP:
            return None
        raise SuryaMissing(
            f"{image.variant.value}/{image.cache_name}: блоков surya нет в кэше, а модели в этом процессе нет — "
            "набить кэш заранее (page_layout prefill-surya) или разбирать в родителе"
        )


__all__ = ["OnMiss", "SuryaMissing", "SuryaSource"]
