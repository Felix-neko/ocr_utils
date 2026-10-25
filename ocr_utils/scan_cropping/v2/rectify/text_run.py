"""Выпрямление разворота в два преобразования: выравнивание освещения страниц итога v2 → Кунс по подгоночным областям → проходы «разбор текста → поле по осям строк и сторонам блоков» → составная карта → одна выборка из итога v2."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.gutter_decompression.apply import page_plans
from ocr_utils.scan_cropping.v2.gutter_decompression.widen import (
    WidenLimits,
    WidenPlan,
    add_wide_displacement,
    widen_displacement,
)
from ocr_utils.scan_cropping.v2.colour_pages import TINT_MIN_SHARE, Colour, chroma_of, page_colour, tinted_share
from ocr_utils.scan_cropping.v2.rectify.coons import (
    TANGENT_MM,
    Layout,
    Patch,
    page_patch,
    patch_outline,
    plan_layout,
    rectify_image,
    rectify_maps,
)
from ocr_utils.scan_cropping.v2.rectify.geometry import SpreadGeometry
from ocr_utils.scan_cropping.v2.rectify.illumination import Background, apply_shift, brightness_shift
from ocr_utils.scan_cropping.v2.rectify.measure import TextMeasures, text_measures
from ocr_utils.scan_cropping.v2.rectify.text_field import (
    ACROSS_SMOOTH_MM,
    CROP_MARGIN_MM,
    FIELD_SMOOTH_MM,
    FIELD_STEP_MM,
    GUTTER_SMOOTH_MM,
    SCALE_WEIGHT,
    FieldMode,
    PageRegion,
    PageText,
    Samples,
    TextField,
    analyze_page,
    band_fits_page,
    band_profile,
    band_samples,
    compose_maps,
    crop_box,
    equation_regions,
    far_zeros,
    field_mode,
    fit_field,
    line_samples,
    page_regions,
    PHOTO_NEAR_MM,
    PHOTO_NEAR_WEIGHT,
    limit_photo_weights,
    photo_samples,
    rule_samples,
    side_samples,
    zero_samples,
)
from ocr_utils.units import MM_PER_INCH, mm_to_px_float

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
        passes: Число проходов «разбор текста → поле» (смещения складываются); 0 — только Кунс.
        scale_weight: Вес штрафа за изменение масштаба по вертикали (цели строк свободны); 0 — жёсткие медианы.
        photos: Растр — до фото (рамка по краю снимка, у чёткого края — кривой четырёхугольник как преграда разбора и
            зона отсева выборок), стороны фото — опоры поля; False — рамки растра детектора, опор в фото нет.
        photo_near_mm: Ближе этого к опоре текста вес выборки стороны фото — ``photo_near_weight``, мм.
        photo_near_weight: Вес выборки стороны фото рядом с текстом (доля полного; 1 — без ограничения).
        gutter_decompression: Растягивать сжатые у корешка буквы разворота по модели, подогнанной к ширине букв
            (``gutter_decompression``): страница становится шире со стороны корешка, холст итога — шире раскладки
            Кунса. Только при проходах поля и у разворота; страницы без поля (цветные, игнорируемые,
            ``no-text-rectify``) не растягиваются.
        gutter_limits: Пределы и порог применения растяжения у корешка.
    """

    tangent_mm: float = TANGENT_MM
    field_step_mm: float = FIELD_STEP_MM
    field_smooth_mm: float = FIELD_SMOOTH_MM
    gutter_smooth_mm: float = GUTTER_SMOOTH_MM
    across_smooth_mm: float = ACROSS_SMOOTH_MM
    illumination: bool = True
    passes: int = 3
    scale_weight: float = SCALE_WEIGHT
    photos: bool = True
    photo_near_mm: float = PHOTO_NEAR_MM
    photo_near_weight: float = PHOTO_NEAR_WEIGHT
    gutter_decompression: bool = True
    gutter_limits: WidenLimits = WidenLimits()


@dataclass(frozen=True)
class PageControl:
    """Что разрешено делать со страницей (из флагов кадра).

    Поля:
        equalize: Выравнивать освещение (если страница не цветная).
        field: Строить поле по тексту; False — страница прижата к нулю (игнорируемая страница: карточка-вклейка
            или поднятый лист не должны тянуть поле).
    """

    equalize: bool = True
    field: bool = True


@dataclass(frozen=True)
class PageState:
    """Страница кадра.

    Поля:
        mask: Область страницы на итоге v2 (bool).
        colour: Уровень цветности (``colour_pages.Colour``).
        background: Оценка фона или None (цветная или без выравнивания).
        before: Разбор текста выпрямленного кадра (после Кунса) или None — без поля.
        after: Разбор текста итога или None.
        lines: Строк в выборках поля.
        mode: Как страница участвует в поле (:class:`FieldMode`).
        tint: Доля тонированного фона (``colour_pages.tinted_share``).
        plain: Страница выравнивалась и шла в поле по автоматике (не цветная и не тонированная).
    """

    mask: np.ndarray
    colour: Colour
    background: Background | None
    before: PageText | None
    after: PageText | None
    lines: int
    mode: FieldMode
    tint: float = 0.0
    plain: bool = True


@dataclass(frozen=True)
class TextResult:
    """Итог выпрямления кадра и промежуточные данные.

    Поля:
        image: Итог v2 (вход).
        equalized: Итог v2 после выравнивания освещения.
        first: Выпрямленный кадр (только Кунс, по выровненному итогу).
        final: Итог (Кунс + поле).
        patches: Области страниц (одна — у кадра с одной страницей).
        layout: Расклад холста.
        field: Поле первого прохода (выборки и веса — для листов) или None — проходов 0.
        dense: Накопленное смещение всех проходов в каждом пикселе ``(dx, dy)`` холста.
        pages: Страницы.
        analyses: Разборы страниц на каждом проходе (нулевой — до поля, последний — итог); None — без поля.
        passes: Меры прямизны текста после каждого прохода (нулевой — после Кунса, последний — итог).
        maps: Составная карта итога ``(map_x, map_y)``: пиксель холста → точка итога v2 (тем же переносятся маски).
        canvas: Раскладка холста итога: при растяжении у корешка шире ``layout`` (левая страница растёт к корешку,
            правая сдвинута и растёт), иначе ``layout``. ``dense`` и ``maps`` — размера этого холста; ``dense`` ведёт
            в координаты холста Кунса (``layout``), поэтому принадлежность пикселя странице и линия разреза
            по-прежнему считаются по прямоугольникам и корешку ``layout``.
        gutter: Планы растяжения у корешка левой и правой страницы или None — растяжения не было (выключено, одна
            страница, нет проходов поля).
        unstretched: Итог без растяжения у корешка (холст ``layout``) — для листа «было | стало»; None — растяжение
            не применено ни к одной странице.
    """

    image: np.ndarray
    equalized: np.ndarray
    first: np.ndarray
    final: np.ndarray
    patches: tuple[Patch, ...]
    layout: Layout
    field: TextField | None
    dense: tuple[np.ndarray, np.ndarray]
    pages: list[PageState]
    analyses: list[list[PageText | None]]
    passes: list[TextMeasures]
    maps: tuple[np.ndarray, np.ndarray]
    canvas: Layout
    gutter: tuple[WidenPlan, WidenPlan] | None = None
    unstretched: np.ndarray | None = None

    @property
    def before(self) -> TextMeasures:
        """Меры текста после первого преобразования (Кунс)."""
        return self.passes[0]

    @property
    def after(self) -> TextMeasures:
        """Меры текста итога."""
        return self.passes[-1]


def page_mask(patch: Patch, shape: tuple[int, int]) -> np.ndarray:
    """Область страницы на итоге v2 (заливка контура поверхности Кунса).

    Args:
        patch: Область страницы.
        shape: ``(H, W)`` итога.

    Returns:
        Маска bool.
    """
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
        skip: Не выравнивать страницу (цветная или запрещено флагом кадра).
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


def text_frame(
    image: np.ndarray,
    geometry: SpreadGeometry,
    dpi: float,
    params: TextParams,
    controls: tuple[PageControl, ...] = (PageControl(), PageControl()),
) -> TextResult:
    """Выпрямление одного кадра в два преобразования.

    Разворот — две области (левая и правая страница, разрез по корешку); кадр с одной страницей (обложка, крышка,
    страница без полоски соседней) — одна область, выпрямляемая как левая страница без корешка (``Layout.single``).

    Args:
        image: Итог v2 BGR.
        geometry: Области страниц и рамка вырезки, годные для этого итога (``frames`` не None: одна или две).
        dpi: Разрешение итога.
        params: Параметры.
        controls: Что разрешено каждой странице (флаги кадра), по числу областей.

    Returns:
        Итог и промежуточные данные.
    """
    tangent_px = mm_to_px_float(params.tangent_mm, dpi)
    # Левая страница (и единственная) — внешняя сторона слева, правая — справа.
    patches = tuple(
        page_patch(frame, i == 0, geometry.to_output, tangent_px) for i, frame in enumerate(geometry.frames)
    )
    n = len(patches)
    if len(controls) != n:
        raise ValueError(f"запретов страниц {len(controls)}, а областей {n}")
    layout = plan_layout(patches, image.shape[:2])
    masks = [page_mask(p, image.shape[:2]) for p in patches]
    chroma = chroma_of(image)
    measured = [page_colour(chroma, m) for m in masks]
    colours = [c.level for c in measured]
    # Цветная — без выравнивания освещения и без поля: целиком цветная (FULL) или с большой долей насыщенного цвета.
    # Обложки ХиП 1991 с белыми плашками названия: медиана цветности 8–11 (ниже FULL), а насыщенного — 45–47 %;
    # у страниц с цветными фото и плашками (ELEMENTS) — до 9 % (546 страниц: обложки, середины выпусков, 2026-10-22).
    # Тонированная (оливковая, синяя обложка с белыми плашками) — тоже: хрома её не отличает (ПХ 1974/01 IMG_0003),
    # а выравнивание освещения поднимало её фон до белого.
    tints = [tinted_share(image, m, dpi / MM_PER_INCH) for m in masks]
    full = [c.level is Colour.FULL or c.saturated_share >= COLOUR_SHARE or t >= TINT_MIN_SHARE
            for c, t in zip(measured, tints)]  # fmt: skip
    # Флаги кадра добавляют запреты к автоматике: без освещения — по флагу, без поля — игнорируемая страница.
    no_equalize = [f or not c.equalize for f, c in zip(full, controls)]
    no_field = [f or not c.field for f, c in zip(full, controls)]
    equalized, backgrounds = image, [None] * n
    if params.illumination:
        equalized, backgrounds = equalize_pages(image, masks, no_equalize, dpi)
    first_maps = rectify_maps(patches, layout)
    first = rectify_image(equalized, first_maps, dpi)
    zero = (np.zeros((layout.height, layout.width), np.float32), np.zeros((layout.height, layout.width), np.float32))
    if params.passes <= 0:
        pages = [PageState(m, c, b, None, None, 0, FieldMode.NONE, t, not f)
                 for m, c, b, t, f in zip(masks, colours, backgrounds, tints, full)]  # fmt: skip
        return TextResult(image, equalized, first, first, patches, layout, None, zero, pages, [[None] * n],
                          [TextMeasures()], first_maps, layout)  # fmt: skip
    margin = mm_to_px_float(CROP_MARGIN_MM, dpi)
    boxes = [crop_box(rect, first.shape[:2], margin, layout.spine_x, i == 0) for i, rect in enumerate(layout.rects)]
    # Итерации: разбор текста на текущем итоге → поле-приращение → сложение с накопленным смещением. На сильно
    # изогнутых участках разбор с первого раза прокладывает ось напрямую через соседние строки (ХиП 1991/06 IMG_0412,
    # волна строк с наклоном 3° внизу правой страницы), на уже подправленной картинке — по тексту.
    # Растр, таблицы и line art — один раз, по кадру после Кунса: поле сдвигает их на миллиметры, рамки грубее.
    # Формулы — блоки surya ``Equation`` кэша v2, перенесённые на холст обратной картой Кунса.
    regions = [() if off else page_regions(first, box, dpi, params.photos) + equation_regions(geometry.equations,
                                                                                geometry.frame_to_output,
                                                                                first_maps, rect)
               for box, rect, off in zip(boxes, layout.rects, no_field)]  # fmt: skip
    total = zero
    current, analyses, maps = first, [], first_maps
    field, line_counts, modes = None, [0] * n, [FieldMode.NONE] * n
    for index in range(params.passes + 1):
        pages_text = [None if off else analyze_page(current, box, dpi, rect, regs)
                      for box, rect, regs, off in zip(boxes, layout.rects, regions, no_field)]  # fmt: skip
        analyses.append(pages_text)
        if index == params.passes:
            break
        h_samples, v_samples, counts, used = pass_samples(pages_text, layout, dpi, params.photo_near_mm,
                                                          params.photo_near_weight)  # fmt: skip
        increment = fit_field(h_samples, v_samples, layout, dpi, params.field_step_mm, params.field_smooth_mm,
                              params.gutter_smooth_mm, params.across_smooth_mm, params.scale_weight)  # fmt: skip
        if field is None:
            # Листы выборок и подписи страниц — по первому проходу.
            field, line_counts, modes = increment, counts, used
        total = add_displacement(total, increment.dense(layout.width, layout.height))
        maps = compose_maps(patches, layout, *total)
        current = rectify_image(equalized, maps, dpi)
    canvas, gutter, unstretched = layout, None, None
    if params.gutter_decompression and not layout.single:
        # Растяжение у корешка — по итогу поля (строки горизонтальны): буквы → модели → планы с расширением.
        px_per_mm = dpi / MM_PER_INCH
        gutter = page_plans(current, layout, px_per_mm, tuple(not off for off in no_field), params.gutter_limits)
        if gutter is not None and any(plan.applied for plan in gutter):
            unstretched = current
            canvas, shift = widen_displacement(gutter, layout, px_per_mm)
            total = add_wide_displacement(total, shift)
            # Кунс — по прежней раскладке: точки X' + D' лежат на холсте Кунса; ширина карты — расширенного холста.
            maps = compose_maps(patches, replace(layout, width=canvas.width), *total)
            current = rectify_image(equalized, maps, dpi)
            # Итог разбирается заново на расширенном холсте (меры итога, оси строк листов); преграды правой
            # страницы сдвигаются на сдвиг её внешнего края.
            analyses[-1] = wide_analyses(current, layout, canvas, regions, no_field, dpi)
    pages = [PageState(m, c, b, pb, pa, n, f, t, not off)
             for m, c, b, pb, pa, n, f, t, off in zip(masks, colours, backgrounds, analyses[0], analyses[-1],
                                                      line_counts, modes, tints, full)]  # fmt: skip
    # Меры прямизны строк и сторон на каждом проходе — для шапки листа и сводки (дёшево против разбора).
    passes = [text_measures([p for p in pages_text if p is not None], dpi) for pages_text in analyses]
    return TextResult(image, equalized, first, current, patches, layout, field, total, pages, analyses, passes,
                      maps, canvas, gutter, unstretched)  # fmt: skip


def shifted_region(region: PageRegion, dx: float) -> PageRegion:
    """Область-преграда разбора, сдвинутая по x на ``dx`` px (рамка и многоугольник)."""
    x0, y0, x1, y1 = region.box
    polygon = None if region.polygon is None else np.asarray(region.polygon, np.float64) + np.array([dx, 0.0])
    return replace(region, box=(x0 + dx, y0, x1 + dx, y1), polygon=polygon)


def wide_analyses(image: np.ndarray, layout: Layout, canvas: Layout, regions: list[tuple[PageRegion, ...]],
                  no_field: list[bool], dpi: float) -> list[PageText | None]:  # fmt: skip
    """Разбор текста итога, растянутого у корешка, на расширенном холсте.

    Args:
        image: Итог на расширенном холсте.
        layout: Раскладка Кунса.
        canvas: Расширенная раскладка.
        regions: Преграды разбора страниц на холсте Кунса.
        no_field: Страницы без поля (не разбираются).
        dpi: Разрешение.

    Returns:
        Разборы страниц (None — страница без поля).
    """
    margin = mm_to_px_float(CROP_MARGIN_MM, dpi)
    out = []
    for i, (rect, regs, off) in enumerate(zip(canvas.rects, regions, no_field)):
        if off:
            out.append(None)
            continue
        # Левая страница растёт к корешку (внешний край на месте), правая сдвинута целиком на W_L и растёт на W_R.
        dx = 0.0 if i == 0 else rect.x1 - layout.rects[1].x1
        moved = tuple(shifted_region(r, dx) for r in regs) if dx else regs
        box = crop_box(rect, image.shape[:2], margin, canvas.spine_x, i == 0)
        out.append(analyze_page(image, box, dpi, rect, moved))
    return out


def pass_samples(
    pages_text: list[PageText | None],
    layout: Layout,
    dpi: float,
    photo_near_mm: float = PHOTO_NEAR_MM,
    photo_near_weight: float = PHOTO_NEAR_WEIGHT,
) -> tuple[Samples, Samples, list[int], list[FieldMode]]:
    """Выборки поля одного прохода по разборам страниц.

    Режим страницы — :func:`field_mode`: полный (строк не меньше :data:`MIN_LINES`) — все опоры; разреженный — те же
    опоры и нули вдали от них (:func:`far_zeros`); полоса (одна строка или несколько рядом) — опоры, вдали от них
    по ``dy`` — их изгиб с затуханием (:func:`band_samples`), по ``dx`` — нули по всей странице; полоса, не прошедшая
    :func:`band_fits_page`, — без поля; без поля (цветная, игнорируемая, опор
    нет или они не похожи на строки) — только выборки «смещение 0».

    Args:
        pages_text: Разборы левой и правой страницы (None — без поля).
        layout: Расклад холста.
        dpi: Разрешение.
        photo_near_mm: Ближе этого к опоре текста вес стороны фото снижен, мм (:func:`limit_photo_weights`).
        photo_near_weight: Вес стороны фото рядом с текстом.

    Returns:
        Выборки ``dx``, выборки ``dy``, число строк в выборках по страницам, режимы страниц.
    """
    h_samples, v_samples = Samples.empty(), Samples.empty()
    counts, modes = [], []
    for rect, page in zip(layout.rects, pages_text):
        lines, text_count = (Samples.empty(), 0) if page is None else line_samples(page, dpi)
        rules_h, rules_v, rule_count = (
            (Samples.empty(), Samples.empty(), 0) if page is None else rule_samples(page, dpi)
        )
        if page is not None and page.photos:
            # Стороны фото с чётким краем — опоры наравне с линейками: горизонтальные идут в число опор
            # (страница с фото и подписью без текста получает режим полосы или разреженный, а не «без поля»).
            # Рядом с опорами текста вес сторон фото снижен: текст главнее (МТС 1990/03 IMG_1101).
            photo_h, photo_v, photo_count = photo_samples(page, dpi)
            photo_h = limit_photo_weights(photo_h, side_samples(page, dpi) + rules_h, dpi, photo_near_mm,
                                          photo_near_weight)  # fmt: skip
            photo_v = limit_photo_weights(photo_v, lines + rules_v, dpi, photo_near_mm, photo_near_weight)
            rules_h, rules_v, rule_count = rules_h + photo_h, rules_v + photo_v, rule_count + photo_count
        # Горизонтальные линейки (таблицы, одинокие черты) — тоже опоры поля, наравне со строками.
        count = text_count + rule_count
        supports_v = lines + rules_v
        mode = FieldMode.NONE if page is None else field_mode(count, supports_v, dpi, text_count)
        if mode is FieldMode.BAND and not band_fits_page(page, supports_v, rect, dpi):
            # Строка у края листа или на странице есть то, чего она не описывает (схема, повёрнутый текст, карточка).
            mode = FieldMode.NONE
        zeros = zero_samples(rect, dpi)
        if mode is FieldMode.NONE:
            h_samples, v_samples = h_samples + zeros, v_samples + zeros
        elif mode is FieldMode.BAND:
            # Поперёк строк — нули по всей странице: стороны блоков при одной-двух строках ложны (у повёрнутого блока
            # 09/IMG_0016 левой — сдвиг до 1.5 мм), вдали от строки — её наклон и прогиб с затуханием по высоте.
            h_samples = h_samples + zeros
            v_samples = v_samples + supports_v + band_samples(zeros, supports_v, band_profile(supports_v, dpi), dpi)
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
