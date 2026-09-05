"""Сменные заливщики ROI: LaMa и Stable Diffusion за одним интерфейсом.

Интерфейс — ``fill_roi(roi, roi_mask, bounds) -> roi``: вернуть ROI того же размера
с заполненной дырой. Никакого состояния у заливщика нет, сети живут в
:class:`ocr_utils.scan_cropping.gpu_models.GpuModels` — единственном владельце
видеопамяти в процессе.

``bounds`` заливщику передаётся не для геометрии (ROI уже вырезан), а чтобы он мог
восстановить положение зоны на КАДРЕ. Это нужно Stable Diffusion: промпт зависит от
того, куда на полосе попала зона — на чистое поле, на цветную обложку или на
полутоновую фотографию. Само правило выбора этому модулю неизвестно: оно приходит
колбэком ``prompts``, а формулировки живут там, где знают предметную область
(``scan_cleanup.prompts``).

Промпт выбирается по РАМКЕ САМОЙ ЗОНЫ, а не по ROI: ROI вдвое больше и у зоны на
краю иллюстрации почти всегда вылезает на бумагу, так что по нему любая
приграничная печать считалась бы стоящей на поле. Рамка зоны восстанавливается из
маски ROI — точно и без лишних параметров.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

from ocr_utils.inpainting.roi import mask_bbox


BACKEND_LAMA = "lama"
BACKEND_SD = "sd"
BACKENDS = (BACKEND_LAMA, BACKEND_SD)

# Модель diffusers по умолчанию. Та же, что была в legacy-обвязке: инпейнт-вариант
# SD 1.5, нативное разрешение 512.
DEFAULT_SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"


def zone_box(roi_mask: np.ndarray, bounds: "tuple[int, int, int, int]") -> "tuple[int, int, int, int]":
    """Рамка зоны В КООРДИНАТАХ КАДРА по её маске внутри ROI.

    Пустая маска (так не бывает, но проверять дешевле, чем ловить) → сам ROI.
    """
    box = mask_bbox(roi_mask)
    if box is None:
        return bounds
    x1, y1, x2, y2 = box
    return bounds[0] + x1, bounds[1] + y1, bounds[0] + x2, bounds[1] + y2


class RoiFiller(Protocol):
    """Заливка одного ROI. ``bounds`` — (x1, y1, x2, y2) ROI в координатах кадра."""

    def __call__(self, roi: np.ndarray, roi_mask: np.ndarray, bounds: "tuple[int, int, int, int]") -> np.ndarray: ...


@dataclass(frozen=True)
class SdParams:
    """Параметры прогона Stable Diffusion.

    ``size`` — длинная сторона, к которой приводится ROI перед сетью (у SD 1.5
    нативные 512). ``seed`` фиксируется, иначе два прогона сравнения дали бы разные
    картинки и сравнивать было бы нечего.
    """

    model: str = DEFAULT_SD_MODEL
    steps: int = 30
    guidance: float = 7.0
    size: int = 512
    seed: int = 0


@dataclass
class LamaFiller:
    """Заливка LaMa. Промпта у неё нет, ``bounds`` не используется.

    Уменьшать ROI перед сетью можно двумя способами, и они взаимоисключающие:
    по длинной стороне ROI (``roi_max_side``, так исторически делалось на пальцах)
    либо по размеру ДЫРЫ (``hole_max_px``, он и есть настоящая величина — см.
    ``gpu_models.lama_fill_roi``). Заданный ``hole_max_px`` имеет приоритет.
    """

    models: object
    roi_max_side: int = 512
    hole_max_px: Optional[int] = None

    def __call__(self, roi: np.ndarray, roi_mask: np.ndarray, bounds: "tuple[int, int, int, int]") -> np.ndarray:
        return self.models.lama_fill_roi(roi, roi_mask, max_side=self.roi_max_side, hole_max_px=self.hole_max_px)


@dataclass
class SdFiller:
    """Заливка Stable Diffusion с промптом, выбранным по положению зоны на полосе.

    ``prompts`` — ``(рамка зоны, ROI) -> (промпт, негативный промпт)``. ROI передаётся
    потому, что по одной только разметке фон под зоной определяется не всегда: у
    обложки прямоугольник иллюстрации накрывает и поля, — и тогда решает замер по
    пикселям (см. ``scan_cleanup.prompts``).

    ``on_prompt`` (если задан) получает ``(рамка зоны, промпт)`` — этим пользуются лог
    и подпись на сравнительной врезке: без записи, каким промптом получена картинка,
    сравнивать варианты бессмысленно.
    """

    models: object
    prompts: Callable[["tuple[int, int, int, int]", np.ndarray], "tuple[str, str]"]
    params: SdParams = field(default_factory=SdParams)
    on_prompt: Optional[Callable[["tuple[int, int, int, int]", str], None]] = None

    def __call__(self, roi: np.ndarray, roi_mask: np.ndarray, bounds: "tuple[int, int, int, int]") -> np.ndarray:
        box = zone_box(roi_mask, bounds)
        prompt, negative = self.prompts(box, roi)
        if self.on_prompt is not None:
            self.on_prompt(box, prompt)
        return self.models.sd_fill_roi(roi, roi_mask, prompt, negative, self.params)


def make_filler(
    backend: str,
    models: object,
    *,
    roi_max_side: int = 512,
    hole_max_px: Optional[int] = None,
    prompts: Optional[Callable[["tuple[int, int, int, int]", np.ndarray], "tuple[str, str]"]] = None,
    sd: Optional[SdParams] = None,
    on_prompt: Optional[Callable[["tuple[int, int, int, int]", str], None]] = None,
) -> RoiFiller:
    """Заливщик по имени бэкенда.

    Для ``sd`` обязателен ``prompts``: промпт по умолчанию тут заводить нельзя —
    он и есть главный параметр качества, и молча подставленный «какой-нибудь»
    сделал бы результат необъяснимым.
    """
    if backend == BACKEND_LAMA:
        return LamaFiller(models, roi_max_side=roi_max_side, hole_max_px=hole_max_px)
    if backend == BACKEND_SD:
        if prompts is None:
            raise ValueError("для бэкенда 'sd' нужен выбор промпта (аргумент prompts)")
        return SdFiller(models, prompts, sd or SdParams(), on_prompt)
    raise ValueError(f"неизвестный бэкенд закраса: {backend!r} (доступны {BACKENDS})")
