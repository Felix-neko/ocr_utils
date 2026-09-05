"""Операция 1: закрас размеченных в CVAT областей (печати, надписи, прочее).

Маски приходят из базы уже проверенными человеком, поэтому никакой фильтрации по
площади, положению или правдоподобию здесь нет — в отличие от закраса пальцев, где
маску строит детектор и половина кода уходит на отсев его выдумок. Здесь маске
верят полностью.

ПОРЯДОК РАБОТЫ на полосе:

1. Виды разметки обходятся ПО ОТДЕЛЬНОСТИ и в фиксированном порядке ``MASK_KINDS``.
   По отдельности — потому что группировать печать с рукописной надписью не за чем:
   это разные объекты, у них разная цена ошибки, и подавать их одной операцией
   значило бы раздувать ROI без пользы. В фиксированном порядке — потому что каждый
   следующий вид видит результат предыдущего, и без него прогон не воспроизводился бы.
2. Строки одного вида сливаются в одну карту (их бывает несколько на полосу).
3. Карта дилатируется на ``mask_dilate_px``: маску рисовали на копии 1/divisor, и её
   край известен лишь с точностью до этого делителя (при 600 dpi — до 8 px).
   Дилатация идёт ДО группировки, чтобы области, которые и так сливает эта
   погрешность, группировались естественно.
4. Связные области группируются правилом «раздуть на 1/3, пересеклись — вместе»
   (см. :mod:`ocr_utils.inpainting.grouping`).
5. Каждая группа закрашивается одной операцией в ROI, вдвое большем её рамки.
"""

import logging
from dataclasses import dataclass, field
from functools import partial

import cv2
import numpy as np

from ocr_utils.inpainting.apply import inpaint_by_groups
from ocr_utils.inpainting.backends import BACKEND_LAMA, SdParams, make_filler
from ocr_utils.inpainting.grouping import DEFAULT_GROUP_DILATE_FRAC, MIN_ZONE_AREA, group_masks
from ocr_utils.scan_cleanup.prompts import PromptSet, prompt_chooser
from ocr_utils.scan_cleanup.source import PageMarkup, decode_mask_rows
from ocr_utils.scan_cropping.gpu_models import LAMA_ROI_MAX_SIDE
from ocr_utils.scan_cropping.morphology import dilate_disk
from ocr_utils.scan_markup.db.models import MASK_KINDS

logger = logging.getLogger(__name__)

# Припуск к маске перед закраской, пикс. Примерно два делителя уменьшения: разметку
# рисовали на копии 1/8, поэтому её край и без того известен с точностью до 8 px, а
# у оттиска печати вокруг видимой краски остаётся ещё и бледный ореол.
MASK_DILATE_PX = 16

# Во сколько раз ROI больше рамки группы — «примерно 2x по ширине и по высоте».
GROUP_ROI_SCALE = 2.0

# Контекстное поле вокруг рамки группы до растяжения, пикс.
ROI_PADDING = 64

# Ширина растушёвки шва при вклейке, пикс. (уходит внутрь маски).
FEATHER_PX = 9


@dataclass
class InpaintOptions:
    """Настройки закраса."""

    backend: str = BACKEND_LAMA
    kinds: "tuple[str, ...]" = field(default=MASK_KINDS)
    mask_dilate_px: float = MASK_DILATE_PX
    group_dilate_frac: float = DEFAULT_GROUP_DILATE_FRAC
    min_zone_area: int = MIN_ZONE_AREA
    roi_scale: float = GROUP_ROI_SCALE
    roi_padding: int = ROI_PADDING
    feather: int = FEATHER_PX
    lama_roi_max_side: int = LAMA_ROI_MAX_SIDE
    sd: SdParams = field(default_factory=SdParams)
    prompts: PromptSet = field(default_factory=PromptSet)


@dataclass
class InpaintReport:
    """Что и чем закрашено — для оверлея, лога и отчёта."""

    rois: "list[tuple[str, tuple[int, int, int, int]]]" = field(default_factory=list)
    masks: "dict[str, np.ndarray]" = field(default_factory=dict)
    prompts: "list[tuple[str, str]]" = field(default_factory=list)

    @property
    def zones(self) -> int:
        return len(self.rois)

    def counts(self) -> "dict[str, int]":
        """Сколько операций закраса пришлось на каждый вид разметки."""
        out: "dict[str, int]" = {}
        for kind, _roi in self.rois:
            out[kind] = out.get(kind, 0) + 1
        return out


def kind_mask(markup: PageMarkup, kind: str, dilate_px: float) -> np.ndarray:
    """Маска одного вида разметки на полосе, уже с припуском (uint8 0/255)."""
    rows = markup.masks_of(kind)
    if not rows:
        return np.zeros((markup.height, markup.width), np.uint8)
    mask = decode_mask_rows(rows, markup.width, markup.height).astype(np.uint8) * 255
    return dilate_disk(mask, dilate_px)


def inpaint_page(
    bgr: np.ndarray, markup: PageMarkup, opts: "InpaintOptions | None" = None, models=None
) -> "tuple[np.ndarray, InpaintReport]":
    """Закрашивает все размеченные области полосы. Возвращает кадр BGR и отчёт.

    ``models`` — ``scan_cropping.gpu_models.GpuModels``; для бэкенда ``sd`` он
    должен быть создан с ``sd_model=...``.

    Работа идёт в RGB (сети ждут именно его), на входе и выходе — BGR, как во всём
    остальном пайплайне.
    """
    opts = opts or InpaintOptions()
    report = InpaintReport()
    if not markup.masks:
        return bgr, report

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for kind in opts.kinds:
        mask = kind_mask(markup, kind, opts.mask_dilate_px)
        if not mask.any():
            continue

        chooser = prompt_chooser(markup, kind, opts.prompts)
        filler = make_filler(
            opts.backend,
            models,
            roi_max_side=opts.lama_roi_max_side,
            prompts=chooser,
            sd=opts.sd,
            on_prompt=lambda _box, prompt, kind=kind: report.prompts.append((kind, prompt)),
        )
        rgb, rois = inpaint_by_groups(
            rgb,
            mask,
            filler,
            groups=partial(group_masks, dilate_frac=opts.group_dilate_frac, min_area=opts.min_zone_area),
            padding=opts.roi_padding,
            feather=opts.feather,
            roi_scale=opts.roi_scale,
        )
        report.masks[kind] = mask
        report.rois.extend((kind, roi) for roi in rois)
        logger.debug("%s: %s — зон %d", markup.rel_path, kind, len(rois))

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), report
