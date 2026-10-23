"""Проход кадра со вторым преобразованием: области страниц по кэшу v2 → выравнивание освещения не-цветных страниц итога v2 → выпрямление Кунсом → разбор текста страниц → поле по осям строк и сторонам блоков → составная карта → одна выборка из итога v2 → разбор итога для мер."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.colour_pages import Colour, chroma_of, page_colour
from ocr_utils.units import mm_to_px_float
from research.page_rectify.coons import Layout, Patch, page_patch, plan_layout, rectify_image, rectify_maps
from research.page_rectify.frames import CachedFrames
from research.page_rectify.illumination import Background, apply_shift, brightness_shift
from research.page_rectify.measure import TextMeasures, text_measures
from research.page_rectify.overlay import patch_outline
from research.page_rectify.text_field import (
    CROP_MARGIN_MM,
    SCALE_WEIGHT,
    FieldMode,
    PageText,
    Samples,
    TextField,
    analyze_page,
    compose_maps,
    crop_box,
    far_zeros,
    field_mode,
    fit_field,
    line_samples,
    page_regions,
    equation_regions,
    rule_samples,
    side_samples,
    zero_samples,
)

# Доля насыщенных пикселей страницы, с которой она считается цветной (между 0.09 у ELEMENTS и 0.45 у обложек).
COLOUR_SHARE = 0.2
# Переход правки освещения от края страницы наружу, мм: на полях итога нет ступеньки яркости.
ILLUMINATION_FEATHER_MM = 3.0


@dataclass(frozen=True)
class TextParams:
    """Параметры второго преобразования.

    Поля:
        tangent_mm: Отрезок у конца стороны для касательной продолжения (Кунс), мм.
        field_step_mm: Шаг узлов поля, мм.
        field_smooth_mm: Длина изгиба поля в теле страницы, мм.
        gutter_smooth_mm: Длина изгиба поля у корешка, мм.
        across_smooth_mm: Длина изгиба поля поперёк строк, мм.
        illumination: Выравнивать освещение не-цветных страниц.
        passes: Число проходов «разбор текста → поле» (смещения складываются).
        scale_weight: Вес штрафа за изменение масштаба по вертикали (цели строк свободны); 0 — жёсткие медианы.
    """

    tangent_mm: float
    field_step_mm: float
    field_smooth_mm: float
    gutter_smooth_mm: float
    across_smooth_mm: float
    illumination: bool
    passes: int = 3
    scale_weight: float = SCALE_WEIGHT


@dataclass(frozen=True)
class PageState:
    """Страница кадра.

    Поля:
        mask: Область страницы на итоге v2 (bool).
        colour: Уровень цветности (``colour_pages.Colour``).
        background: Оценка фона или None (цветная или без выравнивания).
        before: Разбор текста выпрямленного кадра (после Кунса) или None — цветная.
        after: Разбор текста итога или None.
        lines: Строк в выборках поля.
        mode: Как страница участвует в поле (:class:`FieldMode`).
    """

    mask: np.ndarray
    colour: Colour
    background: Background | None
    before: PageText | None
    after: PageText | None
    lines: int
    mode: FieldMode


@dataclass(frozen=True)
class TextResult:
    """Итог кадра со вторым преобразованием и всё, что нужно листам.

    Поля:
        image: Итог v2 (вход).
        equalized: Итог v2 после выравнивания освещения.
        first: Выпрямленный кадр (только Кунс, по выровненному итогу).
        final: Итог (Кунс + поле).
        patches: Области страниц.
        layout: Расклад холста.
        field: Поле первого прохода (выборки и веса — для листа 02).
        dense: Накопленное смещение всех проходов в каждом пикселе ``(dx, dy)``.
        pages: Страницы.
        before: Меры текста после первого преобразования.
        after: Меры текста итога.
        passes: Меры после каждого прохода поля (нулевой — до поля).
    """

    image: np.ndarray
    equalized: np.ndarray
    first: np.ndarray
    final: np.ndarray
    patches: tuple[Patch, Patch]
    layout: Layout
    field: TextField
    dense: tuple[np.ndarray, np.ndarray]
    pages: list[PageState]
    before: TextMeasures
    after: TextMeasures
    passes: list[TextMeasures]


def page_mask(patch: Patch, shape: tuple[int, int]) -> np.ndarray:
    """Область страницы на итоге v2 (заливка контура поверхности Кунса)."""
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(patch_outline(patch)).astype(np.int32)], 1)
    return mask > 0


def equalize_pages(
    image: np.ndarray, masks: list[np.ndarray], skip: list[bool], dpi: float
) -> tuple[np.ndarray, list[Background | None]]:
    """Выравнивание освещения страниц итога v2: сдвиги яркости страниц сводятся с весами-переходами.

    Внутри страницы вес 1, наружу — спад до 0 на :data:`ILLUMINATION_FEATHER_MM`. Где веса двух страниц
    перекрываются (у корешка), сдвиг — их средневзвешенное, поэтому на корешке нет ступеньки.

    Args:
        image: Итог v2 BGR.
        masks: Области страниц.
        skip: Не выравнивать страницу (цветная).
        dpi: Разрешение.

    Returns:
        Выровненная картинка и фоны страниц (None — пропущена).
    """
    feather = mm_to_px_float(ILLUMINATION_FEATHER_MM, dpi)
    total = np.zeros(image.shape[:2], np.float32)
    weights = np.zeros_like(total)
    backgrounds: list[Background | None] = []
    for mask, off in zip(masks, skip):
        if off or not mask.any():
            backgrounds.append(None)
            continue
        shift, background = brightness_shift(image, mask)
        # Расстояние до страницы снаружи → вес перехода.
        dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        weight = np.clip(1.0 - dist / feather, 0.0, 1.0)
        total += shift * weight
        weights += weight
        backgrounds.append(background)
    shift = total / np.maximum(weights, 1.0)
    return apply_shift(image, shift), backgrounds


def text_frame(image: np.ndarray, cf: CachedFrames, dpi: float, params: TextParams) -> TextResult:
    """Второе преобразование одного кадра.

    Args:
        image: Итог v2 BGR.
        cf: Области страниц и рамка вырезки (по кэшу v2), годные для этого итога.
        dpi: Разрешение итога.
        params: Параметры.

    Returns:
        Итог и промежуточные данные.
    """
    tangent_px = mm_to_px_float(params.tangent_mm, dpi)
    patches = (page_patch(cf.frames[0], True, cf.to_output, tangent_px),
               page_patch(cf.frames[1], False, cf.to_output, tangent_px))  # fmt: skip
    layout = plan_layout(patches, image.shape[:2])
    masks = [page_mask(p, image.shape[:2]) for p in patches]
    chroma = chroma_of(image)
    measured = [page_colour(chroma, m) for m in masks]
    colours = [c.level for c in measured]
    # Цветная — без выравнивания освещения и без поля: целиком цветная (FULL) или с большой долей насыщенного цвета.
    # Обложки ХиП 1991 с белыми плашками названия: медиана цветности 8–11 (ниже FULL), а насыщенного — 45–47 %;
    # у страниц с цветными фото и плашками (ELEMENTS) — до 9 % (546 страниц: обложки, середины выпусков, 2026-10-22).
    full = [c.level is Colour.FULL or c.saturated_share >= COLOUR_SHARE for c in measured]
    equalized, backgrounds = image, [None, None]
    if params.illumination:
        equalized, backgrounds = equalize_pages(image, masks, full, dpi)
    first_maps = rectify_maps(patches, layout)
    first = rectify_image(equalized, first_maps, dpi)
    margin = mm_to_px_float(CROP_MARGIN_MM, dpi)
    boxes = [crop_box(rect, first.shape[:2], margin, layout.spine_x, i == 0) for i, rect in enumerate(layout.rects)]
    # Итерации: разбор текста на текущем итоге → поле-приращение → сложение с накопленным смещением. На сильно
    # изогнутых участках разбор с первого раза прокладывает ось напрямую через соседние строки (ХиП 1991/06 IMG_0412,
    # волна строк с наклоном 3° внизу правой страницы), на уже подправленной картинке — по тексту.
    # Растр, таблицы и line art — один раз, по кадру после Кунса: поле сдвигает их на миллиметры, рамки грубее.
    # Формулы — блоки surya ``Equation`` кэша v2, перенесённые на холст обратной картой Кунса.
    regions = [() if is_full else page_regions(first, box, dpi) + equation_regions(cf.equations, cf.frame_to_output,
                                                                                    first_maps, rect)
               for box, rect, is_full in zip(boxes, layout.rects, full)]  # fmt: skip
    total = (np.zeros((layout.height, layout.width), np.float32), np.zeros((layout.height, layout.width), np.float32))
    current, analyses, measures = first, [], []
    field = None
    for index in range(params.passes + 1):
        pages_text = [None if is_full else analyze_page(current, box, dpi, rect, regs)
                      for box, rect, regs, is_full in zip(boxes, layout.rects, regions, full)]  # fmt: skip
        analyses.append(pages_text)
        measures.append(text_measures([p for p in pages_text if p is not None], dpi))
        if index == params.passes:
            break
        h_samples, v_samples, counts, used = pass_samples(pages_text, layout, dpi)
        increment = fit_field(h_samples, v_samples, layout, dpi, params.field_step_mm, params.field_smooth_mm,
                              params.gutter_smooth_mm, params.across_smooth_mm, params.scale_weight)  # fmt: skip
        if field is None:
            # Листы выборок (02) и подписи страниц — по первому проходу.
            field, line_counts, modes = increment, counts, used
        total = add_displacement(total, increment.dense(layout.width, layout.height))
        current = rectify_image(equalized, compose_maps(patches, layout, *total), dpi)
    pages = [PageState(m, c, b, pb, pa, n, f)
             for m, c, b, pb, pa, n, f in zip(masks, colours, backgrounds, analyses[0], analyses[-1], line_counts,
                                               modes)]  # fmt: skip
    return TextResult(image, equalized, first, current, patches, layout, field, total, pages, measures[0],
                      measures[-1], measures)  # fmt: skip


def pass_samples(
    pages_text: list[PageText | None], layout: Layout, dpi: float
) -> tuple[Samples, Samples, list[int], list[FieldMode]]:
    """Выборки поля одного прохода по разборам страниц.

    Режим страницы — :func:`field_mode`: полный (строк не меньше :data:`MIN_LINES`) — все опоры; разреженный — те же
    опоры и нули вдали от них (:func:`far_zeros`); без поля (цветная, опор мало) — только выборки «смещение 0».

    Args:
        pages_text: Разборы левой и правой страницы (None — цветная).
        layout: Расклад холста.
        dpi: Разрешение.

    Returns:
        Выборки ``dx``, выборки ``dy``, число строк в выборках по страницам, режимы страниц.
    """
    h_samples, v_samples = Samples.empty(), Samples.empty()
    counts, modes = [], []
    for rect, page in zip(layout.rects, pages_text):
        lines, count = (Samples.empty(), 0) if page is None else line_samples(page, dpi)
        rules_h, rules_v, rule_count = (
            (Samples.empty(), Samples.empty(), 0) if page is None else rule_samples(page, dpi)
        )
        # Горизонтальные линейки (таблицы, одинокие черты) — тоже опоры поля, наравне со строками.
        count += rule_count
        supports_v = lines + rules_v
        mode = FieldMode.NONE if page is None else field_mode(count, supports_v, dpi)
        zeros = zero_samples(rect, dpi)
        if mode is FieldMode.NONE:
            h_samples, v_samples = h_samples + zeros, v_samples + zeros
        else:
            supports_h = side_samples(page, dpi) + rules_h
            h_samples, v_samples = h_samples + supports_h, v_samples + supports_v
            if mode is FieldMode.SPARSE:
                h_samples = h_samples + far_zeros(zeros, supports_h, dpi)
                v_samples = v_samples + far_zeros(zeros, supports_v, dpi)
        counts.append(count)
        modes.append(mode)
    return h_samples, v_samples, counts, modes


def add_displacement(
    total: tuple[np.ndarray, np.ndarray], increment: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Сложение смещений проходов: пиксель итога ``X`` берётся из ``X + d(X)`` текущего кадра, а тот — из
    ``X + d(X) + D(X + d(X))`` выпрямленного, то есть ``D'(X) = d(X) + D(X + d(X))``.

    Args:
        total: Накопленное смещение ``(D_x, D_y)``, float32 размера холста.
        increment: Приращение этого прохода ``(d_x, d_y)``.

    Returns:
        Новое накопленное смещение.
    """
    h, w = total[0].shape
    xs = np.arange(w, dtype=np.float32)[None, :] + increment[0]
    ys = np.arange(h, dtype=np.float32)[:, None] + increment[1]
    return tuple(inc + cv2.remap(acc, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                 for acc, inc in zip(total, increment))  # fmt: skip
