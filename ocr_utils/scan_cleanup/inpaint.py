"""Операция 1: закрас размеченных в CVAT областей (печати, надписи, прочее).

Маски приходят из базы уже проверенными человеком, поэтому никакой фильтрации по
площади, положению или правдоподобию здесь нет — в отличие от закраса пальцев, где
маску строит детектор и половина кода уходит на отсев его выдумок. Здесь маске
верят полностью.

ПОРЯДОК РАБОТЫ на полосе:

1. Маски ВСЕХ видов сливаются в одну карту. Именно всех вместе, а не по видам:
   заливщику всё равно, почему объект убирают, а вот соседство важно. Рукописная
   пометка рядом с библиотечной печатью, закрашиваемая отдельной операцией, попадает
   в контекстное поле соседней зоны, и сеть честно затягивает её штрихи в дыру
   печати. Одной операцией обе дыры закрываются разом, и затекать нечему.
2. Связные области группируются правилом «раздуть на 1/3 своего размера, пересеклись —
   вместе», с нижней границей припуска ``GROUP_MIN_DILATE_PX`` (см.
   :mod:`ocr_utils.inpainting.grouping`).
3. Раздутая версия решает ТОЛЬКО, что с чем объединять. В сеть уходит объединённый
   набор ИСХОДНЫХ областей, БЕЗ всякого припуска: закрашивается ровно то, что обвёл
   человек, а зазор между слитыми областями остаётся нетронутым.
4. Каждая группа закрашивается одной операцией в ROI, вдвое большем её рамки.

Заливщик — LaMa: на сравнении по 30 полосам Stable Diffusion выдумывала на месте
удалённого объекта псевдотекст и цветные фигуры, а LaMa клала чистую бумагу и ровный
цвет плашки. SD оставлена доступной опцией ради сравнений, но в этой обработке не
используется.
"""

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from ocr_utils.inpainting.apply import inpaint_by_groups, single_group
from ocr_utils.inpainting.backends import BACKEND_LAMA, SdParams, make_filler
from ocr_utils.inpainting.grouping import DEFAULT_GROUP_DILATE_FRAC, MIN_ZONE_AREA, group_masks
from ocr_utils.scan_cleanup.prompts import PromptSet, prompt_chooser
from ocr_utils.scan_cleanup.source import PageMarkup, decode_mask_rows
from ocr_utils.scan_markup.db.models import MASK_KINDS

logger = logging.getLogger(__name__)

# Нижняя граница припуска при ГРУППИРОВКЕ, пикс. Правило «раздуть на 1/3 своего
# размера» у мелких областей вырождается: треть от пяти пикселей — это полтора, и две
# соседние крошки не склеятся. Шестнадцать — примерно два делителя уменьшения: разметку
# рисовали на копии 1/8, и её край и без того известен лишь с точностью до 8 px, так
# что области, разделённые таким зазором, разумно считать одной.
#
# ЭТО ТОЛЬКО ДЛЯ РЕШЕНИЯ О СЛИЯНИИ. В сеть маска идёт БЕЗ припуска: закрашивается ровно
# то, что обвёл человек. Он и так обводит с запасом (видно по любой маске в CVAT: кисть
# заходит далеко за края оттиска), а лишний припуск съедал бы соседнее содержимое.
GROUP_MIN_DILATE_PX = 16

# Во сколько раз ROI больше рамки группы — «примерно 2x по ширине и по высоте».
GROUP_ROI_SCALE = 2.0

# Контекстное поле вокруг рамки группы до растяжения, пикс.
ROI_PADDING = 64

# Ширина растушёвки шва при вклейке, пикс. (уходит внутрь маски).
FEATHER_PX = 9

# Предел размера ДЫРЫ в масштабе сети, пикс. Здесь ограничивается именно дыра, а не
# длинная сторона ROI (``LAMA_ROI_MAX_SIDE``, как на пальцах): предел по ROI даёт на
# разных зонах разную дыру, а от неё-то всё и зависит — см. ``gpu_models.lama_fill_roi``.
# Промежуточное «1024 по ROI» тем и обожглось: на печати 632x464 (1976/02 IMG_0054_2R)
# оно оставляло дыру 539 px и рисовало голубую кляксу, тогда как та же зона при пределе
# дыры в 300 px заливается чистой бумагой.
DEFAULT_LAMA_HOLE_MAX_PX = 300


@dataclass
class InpaintOptions:
    """Настройки закраса."""

    backend: str = BACKEND_LAMA
    kinds: "tuple[str, ...]" = field(default=MASK_KINDS)
    group_dilate_frac: float = DEFAULT_GROUP_DILATE_FRAC
    group_min_dilate_px: int = GROUP_MIN_DILATE_PX
    min_zone_area: int = MIN_ZONE_AREA
    roi_scale: float = GROUP_ROI_SCALE
    roi_padding: int = ROI_PADDING
    feather: int = FEATHER_PX
    lama_hole_max_px: int = DEFAULT_LAMA_HOLE_MAX_PX
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


def kind_mask(markup: PageMarkup, kind: str) -> np.ndarray:
    """Маска одного вида разметки на полосе, как её нарисовали (uint8 0/255).

    Без всякого припуска: припуск нужен только для решения о слиянии областей, а
    закрашивается ровно обведённое.
    """
    rows = markup.masks_of(kind)
    if not rows:
        return np.zeros((markup.height, markup.width), np.uint8)
    return decode_mask_rows(rows, markup.width, markup.height).astype(np.uint8) * 255


def page_masks(markup: PageMarkup, kinds: "tuple[str, ...]") -> "dict[str, np.ndarray]":
    """Маски полосы по видам разметки; виды без разметки пропускаются."""
    out: "dict[str, np.ndarray]" = {}
    for kind in kinds:
        mask = kind_mask(markup, kind)
        if mask.any():
            out[kind] = mask
    return out


def zone_kinds(zone: np.ndarray, masks: "dict[str, np.ndarray]") -> "list[str]":
    """Виды разметки, попавшие в зону, по убыванию занятой площади.

    Зона теперь строится по всем видам сразу, поэтому в одну могут войти и печать, и
    надпись рядом с ней. Порядок по площади нужен, чтобы у отчёта и промпта был
    определённый «главный» вид, а не какой придётся.
    """
    inside = zone > 0
    hit = [(int(np.count_nonzero(inside & (mask > 0))), kind) for kind, mask in masks.items()]
    return [kind for area, kind in sorted(hit, reverse=True) if area]


def inpaint_page(
    bgr: np.ndarray, markup: PageMarkup, opts: "InpaintOptions | None" = None, models=None
) -> "tuple[np.ndarray, InpaintReport]":
    """Закрашивает все размеченные области полосы. Возвращает кадр BGR и отчёт.

    ``models`` — ``scan_cropping.gpu_models.GpuModels``; для бэкенда ``sd`` он
    должен быть создан с ``sd_model=...``.

    Виды разметки НЕ разделяются: связные области всех видов группируются вместе, и
    соседние печать с надписью уходят в закрас одной операцией (зачем — см. докстринг
    модуля). Каждая зона закрашивается отдельным вызовом, а не одним общим: так у
    неё свой ROI и свой промпт, а следующая зона видит уже закрашенные предыдущие.

    Работа идёт в RGB (сети ждут именно его), на входе и выходе — BGR, как во всём
    остальном пайплайне.
    """
    opts = opts or InpaintOptions()
    report = InpaintReport()
    if not markup.masks:
        return bgr, report

    masks = page_masks(markup, opts.kinds)
    if not masks:
        return bgr, report
    report.masks = masks

    union = np.zeros((markup.height, markup.width), np.uint8)
    for mask in masks.values():
        union = cv2.bitwise_or(union, mask)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    zones = group_masks(
        union, opts.group_dilate_frac, min_dilate_px=opts.group_min_dilate_px, min_area=opts.min_zone_area
    )
    for zone in zones:
        kinds = zone_kinds(zone, masks)
        label = "+".join(kinds) or "?"
        filler = make_filler(
            opts.backend,
            models,
            hole_max_px=opts.lama_hole_max_px,
            prompts=prompt_chooser(markup, kinds[0] if kinds else "", opts.prompts),
            sd=opts.sd,
            on_prompt=lambda _box, prompt, label=label: report.prompts.append((label, prompt)),
        )
        rgb, rois = inpaint_by_groups(
            rgb,
            zone,
            filler,
            groups=single_group,
            padding=opts.roi_padding,
            feather=opts.feather,
            roi_scale=opts.roi_scale,
        )
        report.rois.extend((label, roi) for roi in rois)

    logger.debug("%s: зон закраса %d по видам %s", markup.rel_path, report.zones, sorted(masks))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), report
