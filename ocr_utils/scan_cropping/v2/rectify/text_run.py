"""Выпрямление разворота в два преобразования: выравнивание освещения страниц итога v2 → Кунс по подгоночным областям → проходы «разбор текста → поле по осям строк и сторонам блоков» → составная карта → одна выборка из итога v2."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.gutter_decompression.apply import page_plans
from ocr_utils.scan_cropping.v2.gutter_decompression.widen import (
    RAGGED_START_QUANTILE,
    BlockAnchor,
    WidenLimits,
    WidenPlan,
    add_wide_displacement,
    widen_displacement,
)
from ocr_utils.scan_cropping.v2.border_fill import SEAM_MM
from ocr_utils.scan_cropping.v2.fill import FillMethod
from ocr_utils.scan_cropping.v2.colour_pages import TINT_MIN_SHARE, Colour, chroma_of, page_colour, tinted_share
from ocr_utils.scan_cropping.v2.rectify.coons import (
    TANGENT_MM,
    Layout,
    PageRect,
    Patch,
    page_patch,
    patch_outline,
    plan_layout,
    rectify_image,
    rectify_maps,
)
from ocr_utils.scan_cropping.v2.rectify.fold import own_sides, own_view
from ocr_utils.scan_cropping.v2.rectify.page_scale import PageScale, ScaleParams, page_scale, scaled_patch
from ocr_utils.scan_cropping.v2.rectify.geometry import SpreadGeometry
from ocr_utils.scan_cropping.v2.rectify.illumination import Background, apply_shift, brightness_shift
from ocr_utils.page_layout.text_blocks.leaders import inside_spans
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
    vertical_samples,
    zero_samples,
)
from ocr_utils.units import MM_PER_INCH, mm_to_px_float

# Доля насыщенных пикселей страницы, с которой она считается цветной (между 0.09 у ELEMENTS и 0.45 у обложек).
COLOUR_SHARE = 0.2
# Переход правки освещения от края страницы наружу, мм: на полях итога нет ступеньки яркости.
ILLUMINATION_FEATHER_MM = 3.0
# Перенос областей за содержимым (:func:`carried_region`): шаг точек на сторонах рамки, мм, и число итераций обращения
# смещения (поле гладкое — 3 итерации дают доли пикселя).
CARRY_STEP_MM = 2.0
CARRY_ITERATIONS = 3


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
        rotated_text: Искать повёрнутый (вертикальный) текст, в таблицах тоже (``text_field.page_regions``), и
            разбирать его боковыми областями: вертикальные строки не разбираются как горизонтальные.
        photo_near_mm: Ближе этого к опоре текста вес выборки стороны фото — ``photo_near_weight``, мм.
        photo_near_weight: Вес выборки стороны фото рядом с текстом (доля полного; 1 — без ограничения).
        gutter_decompression: Растягивать сжатые у корешка буквы разворота по модели, подогнанной к ширине букв
            (``gutter_decompression``): страница становится шире со стороны корешка, холст итога — шире раскладки
            Кунса. Только при проходах поля и у разворота; страницы без поля (цветные, игнорируемые,
            ``no-text-rectify``) не растягиваются.
        gutter_limits: Пределы и порог применения растяжения у корешка.
        carry_regions: Переносить области (растр, таблицы, line art, формулы) за содержимым после каждого прохода поля
            и растяжения у корешка (:func:`carry_regions`); False — области стоят на месте холста Кунса (как до
            2026-10-25: на итоге рамки отставали от рисунков на 0.5–2.2 мм).
        own_pages: Перед разбором итога (проходы после первого и итог растяжения у корешка) закрашивать в вырезке
            страницы чужое — содержимое с другой стороны корешка и полосу повтора концов строк (``fold.own_view``):
            иначе оси правой страницы пришивали повторённые концы строк левой (ХиП 1991/01 IMG_0982 R).
        fill: Способ закраса чужого — тот же, что докрас страниц до прямоугольника (``--fill``).
        fill_seam_mm: Ширина мягкого шва для ``pull_push_seam``, мм.
        page_scale: Ширина прямоугольника страницы первого преобразования и параметр Кунса вдоль страницы — по
            перспективному масштабу её высоты на снимке (``page_scale``); False — по длине дуги верха и низа (до
            2026-10-27: страница, снятая ближе к камере, выходила шире — ХиП 1991/01 IMG_0993 R 122.6 мм при левой
            110.6 мм, и растяжение у корешка принимало перепад масштаба за сжатие).
        scale_params: Параметры меры масштаба.
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
    rotated_text: bool = True
    photo_near_mm: float = PHOTO_NEAR_MM
    photo_near_weight: float = PHOTO_NEAR_WEIGHT
    gutter_decompression: bool = True
    gutter_limits: WidenLimits = WidenLimits()
    carry_regions: bool = True
    own_pages: bool = True
    fill: FillMethod = FillMethod.PULL_PUSH_SEAM
    fill_seam_mm: float = SEAM_MM
    page_scale: bool = True
    scale_params: ScaleParams = ScaleParams()


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
        scales: Меры перспективного масштаба страниц (``page_scale``), по числу областей; пусто — выключено.
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
    scales: tuple[PageScale, ...] = ()

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
    scales: tuple[PageScale, ...] = ()
    if params.page_scale:
        # Ширина прямоугольника и параметр вдоль страницы — по масштабу высоты страницы на снимке; высота
        # прямоугольника (у разворота — сторона у корешка) прежняя.
        sides = (None,) if n == 1 else (True, False)
        centre = geometry.principal_point(image.shape[:2])
        scales = tuple(page_scale(p, r.height, side, frame.pose, centre, dpi / MM_PER_INCH, params.scale_params)
                       for p, r, side, frame in zip(patches, layout.rects, sides, geometry.frames))  # fmt: skip
        patches = tuple(scaled_patch(p, sc) for p, sc in zip(patches, scales))
        layout = plan_layout(patches, image.shape[:2], widths=tuple(sc.target_width for sc in scales))
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
                          [TextMeasures()], first_maps, layout, scales=scales)  # fmt: skip
    margin = mm_to_px_float(CROP_MARGIN_MM, dpi)
    boxes = [crop_box(rect, first.shape[:2], margin, layout.spine_x, i == 0) for i, rect in enumerate(layout.rects)]
    # Итерации: разбор текста на текущем итоге → поле-приращение → сложение с накопленным смещением. На сильно
    # изогнутых участках разбор с первого раза прокладывает ось напрямую через соседние строки (ХиП 1991/06 IMG_0412,
    # волна строк с наклоном 3° внизу правой страницы), на уже подправленной картинке — по тексту.
    # Растр, таблицы и line art — один раз, по кадру после Кунса: поле сдвигает их на миллиметры, рамки грубее.
    # Формулы — блоки surya ``Equation`` кэша v2, перенесённые на холст обратной картой Кунса.
    regions = [() if off else page_regions(first, box, dpi, params.photos, params.rotated_text) + equation_regions(geometry.equations,
                                                                                geometry.frame_to_output,
                                                                                first_maps, rect)
               for box, rect, off in zip(boxes, layout.rects, no_field)]  # fmt: skip
    total = zero
    current, analyses, maps = first, [], first_maps
    field, line_counts, modes = None, [0] * n, [FieldMode.NONE] * n
    for index in range(params.passes + 1):
        # Области — за содержимым: поле прохода сдвигает рисунки и фото на миллиметры, а рамки найдены по холсту Кунса.
        moved = carry_regions(regions, total, dpi) if params.carry_regions and index else regions
        # Своя сторона страниц — после первого прохода поля (до него чужого за корешком нет: вырезки по корешку).
        own = own_sides(layout, total) if params.own_pages and index else None
        pages_text = [None if off else analyze_page(page_view(current, own, i, box, params, dpi), box, dpi, rect, regs)
                      for i, (box, rect, regs, off) in enumerate(zip(boxes, layout.rects, moved, no_field))]  # fmt: skip
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
        # Начало зоны растяжения — по блокам разбора итога (выровненная сторона у корешка остаётся прямой).
        anchors = tuple(block_anchors(text, rect, layout.spine_x, i == 0, px_per_mm, params.gutter_limits.reach_mm)
                        for i, (text, rect) in enumerate(zip(analyses[-1], layout.rects)))  # fmt: skip
        gutter = page_plans(current, layout, px_per_mm, tuple(not off for off in no_field), params.gutter_limits,
                            anchors=anchors)  # fmt: skip
        if gutter is not None and any(plan.applied for plan in gutter):
            unstretched = current
            canvas, shift = widen_displacement(gutter, layout, px_per_mm)
            total = add_wide_displacement(total, shift)
            # Кунс — по прежней раскладке: точки X' + D' лежат на холсте Кунса; ширина карты — расширенного холста.
            maps = compose_maps(patches, replace(layout, width=canvas.width), *total)
            current = rectify_image(equalized, maps, dpi)
            # Итог разбирается заново на расширенном холсте (меры итога, оси строк листов); преграды — за содержимым
            # по накопленному смещению с растяжением (или, без переноса, на сдвиг внешнего края правой страницы).
            analyses[-1] = wide_analyses(current, layout, canvas, regions, no_field, dpi,
                                         total if params.carry_regions else None,
                                         own_sides(layout, total) if params.own_pages else None, params)  # fmt: skip
    pages = [PageState(m, c, b, pb, pa, n, f, t, not off)
             for m, c, b, pb, pa, n, f, t, off in zip(masks, colours, backgrounds, analyses[0], analyses[-1],
                                                      line_counts, modes, tints, full)]  # fmt: skip
    # Меры прямизны строк и сторон на каждом проходе — для шапки листа и сводки (дёшево против разбора).
    passes = [text_measures([p for p in pages_text if p is not None], dpi) for pages_text in analyses]
    return TextResult(image, equalized, first, current, patches, layout, field, total, pages, analyses, passes,
                      maps, canvas, gutter, unstretched, scales)  # fmt: skip


def carried_points(points: np.ndarray, total: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Где на текущем холсте оказались точки холста Кунса: обратное к ``итог(P) = Кунс(P + D(P))``.

    Текущий холст берёт пиксель ``P`` из точки ``P + D(P)`` холста Кунса, поэтому точка ``Q`` холста Кунса стоит в
    ``P``, решающей ``P + D(P) = Q``. Решение — простой итерацией ``P ← Q − D(P)`` (:data:`CARRY_ITERATIONS` раз):
    поле гладкое, его производная много меньше единицы, итерация сходится за 2–3 шага до долей пикселя. На холсте
    растяжения у корешка ``D`` включает и растяжение (``add_wide_displacement``) — сходится так же: растяжение плавное.

    Args:
        points: Точки холста Кунса ``(N, 2)`` px.
        total: Накопленное смещение ``(D_x, D_y)`` текущего холста (H, W).

    Returns:
        Точки текущего холста ``(N, 2)`` px.
    """
    from scipy.ndimage import map_coordinates

    target = np.asarray(points, np.float64)
    height, width = total[0].shape
    current = target.copy()
    for _ in range(CARRY_ITERATIONS):
        # Смещение в текущей оценке (билинейно, за краем — ближайший край холста).
        xs = np.clip(current[:, 0], 0, width - 1)
        ys = np.clip(current[:, 1], 0, height - 1)
        dx = map_coordinates(total[0], [ys, xs], order=1, mode="nearest")
        dy = map_coordinates(total[1], [ys, xs], order=1, mode="nearest")
        current = target - np.stack([dx, dy], axis=1)
    return current


def carried_region(region: PageRegion, total: tuple[np.ndarray, np.ndarray], dpi: float) -> PageRegion:
    """Область, перенесённая за содержимым на текущий холст (рамка — по точкам её сторон, многоугольник — по вершинам).

    Рамка переносится не по одной точке: стороны отбираются через :data:`CARRY_STEP_MM` (не меньше четырёх точек на
    сторону), и новая рамка — охват перенесённых точек, так что растяжение у корешка и изгиб поля внутри рамки её
    расширяют или сжимают, а не только сдвигают. Найденное фото (``photo``) не переносится: стороны фото ищутся
    заново на холсте каждого прохода по рамке-затравке (``analyze_page``), ему нужна только рамка на своём месте.

    Args:
        region: Область на холсте Кунса.
        total: Накопленное смещение текущего холста.
        dpi: Разрешение холста.

    Returns:
        Область на текущем холсте.
    """
    x0, y0, x1, y1 = region.box
    step = mm_to_px_float(CARRY_STEP_MM, dpi)
    nx = max(4, int(np.ceil((x1 - x0) / step)) + 1)
    ny = max(4, int(np.ceil((y1 - y0) / step)) + 1)
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    border = np.concatenate([np.stack([xs, np.full(nx, y0)], 1), np.stack([xs, np.full(nx, y1)], 1),
                             np.stack([np.full(ny, x0), ys], 1), np.stack([np.full(ny, x1), ys], 1)])  # fmt: skip
    moved = carried_points(border, total)
    polygon = None
    if region.polygon is not None:
        polygon = carried_points(np.asarray(region.polygon, np.float64), total)
        if len(polygon) >= 3:
            # Рамка фото с многоугольником — охват многоугольника (как в page_regions).
            moved = polygon
    box = (float(moved[:, 0].min()), float(moved[:, 1].min()), float(moved[:, 0].max()), float(moved[:, 1].max()))
    return replace(region, box=box, polygon=polygon)


def carry_regions(
    regions: list[tuple[PageRegion, ...]], total: tuple[np.ndarray, np.ndarray], dpi: float
) -> list[tuple[PageRegion, ...]]:
    """Области всех страниц, перенесённые за содержимым на текущий холст (:func:`carried_region`).

    Args:
        regions: Области страниц на холсте Кунса.
        total: Накопленное смещение текущего холста (после проходов поля; на расширенном холсте — с растяжением).
        dpi: Разрешение холста.

    Returns:
        Области страниц на текущем холсте.
    """
    if not np.any(total[0]) and not np.any(total[1]):
        return regions
    return [tuple(carried_region(r, total, dpi) for r in regs) for regs in regions]


def shifted_region(region: PageRegion, dx: float) -> PageRegion:
    """Область-преграда разбора, сдвинутая по x на ``dx`` px (рамка и многоугольник)."""
    x0, y0, x1, y1 = region.box
    polygon = None if region.polygon is None else np.asarray(region.polygon, np.float64) + np.array([dx, 0.0])
    return replace(region, box=(x0 + dx, y0, x1 + dx, y1), polygon=polygon)


def page_view(
    image: np.ndarray,
    own: tuple[np.ndarray, np.ndarray] | None,
    index: int,
    box: tuple[int, int, int, int],
    params: TextParams,
    dpi: float,
) -> np.ndarray:
    """Картинка для разбора страницы: итог, где в вырезке страницы закрашено чужое (``fold.own_view``).

    Args:
        image: Итог текущего прохода BGR.
        own: Своя сторона левой и правой страницы (``fold.own_sides``) или None — закрашивать нечего (одна страница,
            первый проход, ``own_pages`` выключен).
        index: Номер страницы (0 — левая).
        box: Вырезка разбора страницы.
        params: Параметры (способ закраса и шов).
        dpi: Разрешение итога.

    Returns:
        ``image`` или его копия с закрашенным чужим в вырезке.
    """
    if own is None:
        return image
    return own_view(image, own[index], box, params.fill, dpi / MM_PER_INCH, params.fill_seam_mm)


def block_anchors(text: PageText | None, rect: PageRect, spine_x: float, left_page: bool, px_per_mm: float,
                  reach_mm: float) -> list[BlockAnchor]:  # fmt: skip
    """Начала зоны растяжения у корешка по текстовым блокам страницы (``gutter_decompression.widen.BlockAnchor``).

    Сторона блока у корешка — правая у левой страницы, левая у правой. Выровненная (``Alignment.<сторона>.aligned``)
    — начало зоны на медиане её линии (``envelope.core_*`` без недостоверных участков, как в мере сторон), иначе —
    перцентиль :data:`RAGGED_START_QUANTILE` концов строк блока со стороны корешка. Блоки, чьи строки начинаются
    дальше ``reach_mm`` от корешка, зоны у корешка не задают.

    Args:
        text: Разбор страницы на холсте до растяжения или None — без поля (блоков нет).
        rect: Прямоугольник страницы на холсте.
        spine_x: X линии корешка на холсте.
        left_page: Левая страница (корешок справа).
        px_per_mm: Плотность холста, px/мм.
        reach_mm: Зона растяжения, мм.

    Returns:
        Начала по блокам (пусто — блоков нет: начало зоны по огибающей строк).
    """
    if text is None:
        return []
    side = "right" if left_page else "left"
    out = []
    for block, alignment in zip(text.analysis.blocks, text.analysis.alignments):
        if not block.rows:
            continue
        # Концы строк со стороны корешка → расстояние от линии корешка, мм.
        ends = np.array([[r.x1 if left_page else r.x0, r.y] for r in block.rows], np.float64)
        ends = text.to_canvas(ends)
        dist = np.abs(ends[:, 0] - spine_x) / px_per_mm
        start = float(np.quantile(dist, RAGGED_START_QUANTILE))
        stats = getattr(alignment, side)
        aligned = False
        core = getattr(block.envelope, f"core_{side}")
        if stats.aligned and core is not None and len(core) >= 2:
            pts = np.asarray(core, np.float64)
            unreliable = getattr(block.envelope, f"unreliable_{side}", ())
            if unreliable:
                pts = pts[~inside_spans(pts[:, 1], list(unreliable))]
            if len(pts) >= 2:
                start = float(np.median(np.abs(text.to_canvas(pts)[:, 0] - spine_x))) / px_per_mm
                aligned = True
        if start > reach_mm:
            continue
        # Высота блока — от верха первого ряда до низа последнего, доли прямоугольника страницы.
        half = np.array([r.height for r in block.rows]) / 2 * text.scale
        y0 = (float(np.min(ends[:, 1] - half)) - rect.y0) / max(rect.height, 1.0)
        y1 = (float(np.max(ends[:, 1] + half)) - rect.y0) / max(rect.height, 1.0)
        out.append(BlockAnchor(y0, y1, start, aligned))
    return out


def wide_analyses(image: np.ndarray, layout: Layout, canvas: Layout, regions: list[tuple[PageRegion, ...]],
                  no_field: list[bool], dpi: float, total: tuple[np.ndarray, np.ndarray] | None = None,
                  own: tuple[np.ndarray, np.ndarray] | None = None,
                  params: TextParams = TextParams()) -> list[PageText | None]:  # fmt: skip
    """Разбор текста итога, растянутого у корешка, на расширенном холсте.

    Args:
        image: Итог на расширенном холсте.
        layout: Раскладка Кунса.
        canvas: Расширенная раскладка.
        regions: Преграды разбора страниц на холсте Кунса.
        no_field: Страницы без поля (не разбираются).
        dpi: Разрешение.
        total: Накопленное смещение расширенного холста (поле и растяжение, ``add_wide_displacement``): области
            переносятся за содержимым по нему (:func:`carry_regions`); None — прежний постоянный сдвиг правой страницы.
        own: Своя сторона страниц на расширенном холсте (``fold.own_sides``): чужое в вырезке закрашивается; None —
            вырезка как есть.
        params: Параметры (способ закраса чужого).

    Returns:
        Разборы страниц (None — страница без поля).
    """
    margin = mm_to_px_float(CROP_MARGIN_MM, dpi)
    carried = carry_regions(regions, total, dpi) if total is not None else None
    out = []
    for i, (rect, regs, off) in enumerate(zip(canvas.rects, regions, no_field)):
        if off:
            out.append(None)
            continue
        if carried is not None:
            moved = carried[i]
        else:
            # Левая страница растёт к корешку (внешний край на месте), правая сдвинута целиком на W_L и растёт на W_R.
            dx = 0.0 if i == 0 else rect.x1 - layout.rects[1].x1
            moved = tuple(shifted_region(r, dx) for r in regs) if dx else regs
        box = crop_box(rect, image.shape[:2], margin, canvas.spine_x, i == 0)
        out.append(analyze_page(page_view(image, own, i, box, params, dpi), box, dpi, rect, moved))
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
            photo_h = limit_photo_weights(photo_h, text_sides(page, dpi) + rules_h, dpi, photo_near_mm,
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
            supports_h = text_sides(page, dpi) + rules_h
            h_samples, v_samples = h_samples + supports_h, v_samples + supports_v
            if mode is FieldMode.SPARSE:
                h_samples = h_samples + far_zeros(zeros, supports_h, dpi)
                v_samples = v_samples + far_zeros(zeros, supports_v, dpi)
        counts.append(count)
        modes.append(mode)
    return h_samples, v_samples, counts, modes


def text_sides(page: PageText, dpi: float) -> Samples:
    """Опоры текста составляющей ``dx``: выровненные стороны блоков и базы вертикальных строк.

    Args:
        page: Разбор страницы.
        dpi: Разрешение холста.

    Returns:
        Выборки ``dx`` (:func:`side_samples` + :func:`vertical_samples`).
    """
    return side_samples(page, dpi) + vertical_samples(page, dpi)


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
