"""Оркестрация: настройки прогона, обработка одного кадра, обход пачки файлов.

Здесь собирается пайплайн из модулей пакета; сама логика обработки изображений
живёт в них (``page_detection``, ``geometry``, ``background_fill``, ``cropping``,
``levels``, ``layout_filtering``, ``overlay``, ``image_io``), а CLI (``cli``)
только строит :class:`CropParams` и зовёт :func:`run_batch`.

Порядок на каждый кадр:
  1. удаление придерживающего страницу пальца (``finger_removal.remove_fingers``) —
     ДО детекции разворота: палец искажает силуэт книги и попадает в кроп;
  2. ``page_detection.page_mask`` → криволинейный силуэт разворота (E1);
  3. ``geometry.min_area_rotated_bbox`` → «правильный поворот» (угол + осевой bbox
     в повёрнутой системе координат);
  4. ``geometry.trim_cover_fragments`` → область КОПИРОВАНИЯ (E2): E1 без
     периферийных фрагментов тёмной обложки;
  5. припуски по сторонам + расширение под блоки layout → финальная crop-зона;
  6. вырезка способом ``--crop-mode`` и запись результата;
  7. при ``--debug-dir`` — оверлей поверх ИСХОДНОГО кадра.
"""

import logging
import timeit
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from ocr_utils.scan_cropping.background_fill import (
    BG_FILL_AVERAGE,
    CROP_FILL_REPLICATE,
    book_mean_color,
    fill_outside_mask,
)
from ocr_utils.scan_cropping.cropping import (
    CROP_FILL_BLUR_PX,
    CROP_FILL_FADE,
    CROP_MODE_PIXEL_EXACT,
    CROP_MODE_ROTATE,
    crop_pixel_exact,
    crop_rotated,
)
from ocr_utils.scan_cropping.finger_removal.finger_shadow import correct_finger_shadow
from ocr_utils.scan_cropping.finger_removal.removal import FINGER_DILATE_PX, FINGER_ZONE_LIGHT_INCREMENT, remove_fingers
from ocr_utils.scan_cropping.finger_removal.asymmetric_dilation import DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO
from ocr_utils.scan_cropping.finger_removal.text_protection import (
    DEFAULT_LAYOUT_PAD_PX,
    PROTECT_LIMIT_LAMA,
    layout_polygons,
    polygons_to_mask,
)
from ocr_utils.scan_cropping.geometry import (
    EXTRA_EROSION_PX,
    crop_ext_with_layout,
    ext_to_mask,
    ext_with_margins,
    layout_ext_bounds,
    min_area_rotated_bbox,
    trim_cover_fragments,
)
from ocr_utils.scan_cropping.gpu_models import LAMA_ROI_MAX_SIDE
from ocr_utils.scan_cropping.image_io import imwrite_params, resolve_output_suffix, write_image
from ocr_utils.scan_cropping.layout_filtering import classify_parasitic_layouts
from ocr_utils.scan_cropping.levels import compensate_levels
from ocr_utils.scan_cropping.overlay import draw_overlay
from ocr_utils.scan_cropping.page_detection import page_mask
from ocr_utils.timing import log_timing

logger = logging.getLogger(__name__)


@dataclass
class CropParams:
    """Все настройки прогона (значения опций CLI в одном месте).

    Собирается один раз в ``cli.main`` и дальше только читается, поэтому
    :func:`process_frame` не тащит два десятка отдельных аргументов.
    """

    input_dir: Path
    output_dir: Path
    debug_dir: Optional[Path] = None

    # Припуски crop-зоны по сторонам: (left, top, right, bottom), пикс.
    margins: "tuple[int, int, int, int]" = (0, 0, 0, 0)
    recursive: bool = False
    skip_if_exists: bool = True
    output_format: Optional[str] = None
    force_dpi: Optional[int] = None

    # Компенсация уровней и обрезка краёв силуэта книги
    compensate_levels: bool = False
    extra_erosion_px: int = EXTRA_EROSION_PX

    # Вырезка
    crop_mode: str = CROP_MODE_ROTATE
    upscale: Optional[float] = None
    crop_fill_method: str = CROP_FILL_REPLICATE
    crop_fill_blur_px: float = CROP_FILL_BLUR_PX
    crop_fill_fade: float = CROP_FILL_FADE
    bg_fill_method: str = BG_FILL_AVERAGE
    bg_fill_blur_px: float = 0.0

    # Удаление пальцев
    remove_fingers: bool = True
    finger_dilate_px: int = FINGER_DILATE_PX
    finger_zone_light_increment: "tuple[float, float]" = (FINGER_ZONE_LIGHT_INCREMENT, FINGER_ZONE_LIGHT_INCREMENT)
    asymmetric_dilation_ratio: float = DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO
    lama_roi_max_side: int = LAMA_ROI_MAX_SIDE
    shadow_method: str = "none"

    # Защита контента от закраски
    protect_text_layout: bool = False
    text_protect_mode: str = PROTECT_LIMIT_LAMA
    layout_pad_px: "tuple[int, int]" = field(default=(DEFAULT_LAYOUT_PAD_PX, DEFAULT_LAYOUT_PAD_PX))

    # Детекция разворота
    detect_pad_tb_px: int = 250
    # Радиус смыкания силуэта разворота, пикс.; None — по размеру кадра
    page_close_px: Optional[int] = None


def process_frame(path: Path, params: CropParams, models) -> None:
    """Обрабатывает один кадр: детекция → закраска пальца → кроп → запись.

    ``path`` — путь к исходному изображению (внутри ``params.input_dir``),
    ``models`` — ``gpu_models.GpuModels``. Ничего не возвращает: результат
    пишется в ``params.output_dir`` (и оверлей в ``params.debug_dir``).
    Сообщения о пропусках и находках печатаются через ``tqdm.write``, чтобы не
    ломать прогресс-бар.
    """
    # Путь результата (при recursive зеркалим подкаталоги; формат — из
    # --output-format либо как у входа). Считаем его ДО загрузки картинки,
    # чтобы --skip-if-exists мог пропустить файл, не тратя время на imread.
    rel = path.relative_to(params.input_dir)
    out_suffix = resolve_output_suffix(path.suffix, params.output_format)
    out_path = (params.output_dir / rel).with_suffix(out_suffix)
    # Докачка прерванного прогона: пропускаем файл, только если готов
    # OUTPUT-файл. debug-оверлей при фактической обработке переписывается
    # всегда (его наличие на решение о пропуске не влияет).
    if params.skip_if_exists and out_path.exists():
        logger.info("Пропуск (результат уже есть): %s", rel)
        return
    write_params = imwrite_params(out_suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with log_timing("imread", path.name):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        tqdm.write(f"  Не удалось загрузить: {path.name}")
        return

    bgr_orig = bgr  # для debug-оверлея: без удаления пальцев и без компенсации уровней
    finger_mask: Optional[np.ndarray] = None
    lama_roi_bboxes: Optional[list] = None
    finger_boxes: Optional[np.ndarray] = None
    finger_mask_predilate: Optional[np.ndarray] = None
    layout_polys: Optional[list] = None
    if params.remove_fingers:
        with log_timing("remove_fingers", path.name):
            (bgr, finger_mask, lama_roi_bboxes, finger_boxes, finger_info, finger_mask_predilate, layout_polys) = (
                remove_fingers(
                    bgr,
                    models,
                    want_boxes=params.debug_dir is not None,
                    dilate_px=params.finger_dilate_px,
                    light_increment=params.finger_zone_light_increment,
                    asymmetric_dilation_ratio=params.asymmetric_dilation_ratio,
                    protect_text=params.protect_text_layout,
                    protect_mode=params.text_protect_mode,
                    layout_pad_px=params.layout_pad_px,
                    lama_roi_max_side=params.lama_roi_max_side,
                    log_name=path.name,
                )
            )
        if int(np.count_nonzero(finger_mask)) > 0:
            tqdm.write(f"  Пальцы: {finger_info} ({path.name})")

    with log_timing("page_mask", path.name):
        # E1 — силуэт разворота (светлые страницы + куски обложки)
        mask = page_mask(bgr, models, pad_tb_px=params.detect_pad_tb_px, close_px=params.page_close_px)

    # Коррекция теневой зоны вокруг пальца (после зарисовки, до кропа/уровней)
    if params.shadow_method != "none" and finger_mask is not None:
        with log_timing("correct_finger_shadow", path.name):
            bgr = correct_finger_shadow(bgr, finger_mask, mask, params.shadow_method, models)

    with log_timing("min_area_rotated_bbox", path.name):
        geom = min_area_rotated_bbox(mask)  # B1/B2 строим по E1
    # E2 — область копирования: E1 с обрезанными периферийными фрагментами обложки
    with log_timing("trim_cover_fragments", path.name):
        copy_mask = trim_cover_fragments(mask, params.extra_erosion_px)

    # Блоки layout нужны и для расширения crop-зоны, и для debug-оверлея.
    # При удалении пальцев с защитой текста они уже посчитаны в remove_fingers;
    # иначе, при включённой --protect-text-layout, считаем здесь.
    if layout_polys is None and params.protect_text_layout:
        with log_timing("layout_polygons", path.name):
            layout_polys = layout_polygons(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), models)
    # Отсев паразитных layout: теперь, когда известна область книги (mask),
    # исключаем артефактные блоки (целая пустая страница и т.п.) из расчёта
    # crop-зоны — иначе они сильно раздувают вырезаемую область. Для crop-зоны
    # и её оверлея берём только «хорошие» блоки; паразитные рисуем пунктиром.
    #
    # TODO (на будущее): если паразитный блок пересекался с первичной зоной
    # пальца, то закраска этого пальца шла НЕКОРРЕКТНО (блок ложно защитил
    # контент — на самом деле там пусто). После отсева таких блоков стоило бы
    # ПОВТОРИТЬ закраску именно пересекавшихся с ними пальцев. А при включённом
    # copy-back — заново скопировать обратно оставшиеся (не-паразитные) блоки,
    # пересекающиеся с зоной пальца. Сейчас отсев влияет только на crop-зону и
    # оверлей: закраска/защита считаются раньше (до page_mask), где область
    # книги ещё неизвестна.
    good_polys = layout_polys
    parasitic_polys: list = []
    if layout_polys:
        with log_timing("classify_parasitic_layouts", path.name):
            par_flags = classify_parasitic_layouts(layout_polys, mask, params.layout_pad_px)
        good_polys = [p for p, par in zip(layout_polys, par_flags) if not par]
        parasitic_polys = [p for p, par in zip(layout_polys, par_flags) if par]
        if parasitic_polys:
            tqdm.write(f"  Паразитных layout: {len(parasitic_polys)} (исключены из crop) ({path.name})")
    with log_timing("polygons_to_mask", path.name):
        layout_mask = polygons_to_mask(mask.shape, good_polys, params.layout_pad_px) if good_polys else None

    crop_ext: Optional[tuple] = None
    if geom is None:
        # Разворот не найден — кладём оригинал, чтобы не терять файл в пайплайне.
        # Кропа не было, поэтому уровни (если просили) считаем по области книги:
        # гистограмма всего кадра здесь включала бы чёрный фон стола.
        tqdm.write(f"  Разворот не найден, сохраняю оригинал: {rel}")
        out_img = bgr
        if params.compensate_levels:
            with log_timing("compensate_levels", path.name):
                out_img = compensate_levels(bgr, copy_mask)
        with log_timing("write_image", path.name):
            write_image(out_path, out_img, write_params, params.force_dpi)
    else:
        with log_timing("crop_geometry", path.name):
            cx, cy, angle, ext = geom
            # Финальная crop-зона: ext с припусками, расширенный так, чтобы целиком
            # вместить блоки layout (иначе отриц. припуски срезают часть обложки).
            margined = ext_with_margins(ext, params.margins)
            crop_ext = crop_ext_with_layout(ext, params.margins, layout_ext_bounds(cx, cy, angle, layout_mask))
            # Область копирования (E2) расширяем до расширенного crop-bbox: в кольце
            # между bbox с припусками и расширенным под layout bbox копируем контент
            # страницы (E1), иначе fill_outside_mask замажет там обложку фоном — ту
            # самую, ради которой crop и расширяли.
            if crop_ext != margined:
                ring = cv2.bitwise_and(
                    ext_to_mask(mask.shape, cx, cy, angle, crop_ext),
                    cv2.bitwise_not(ext_to_mask(mask.shape, cx, cy, angle, margined)),
                )
                copy_mask = cv2.bitwise_or(copy_mask, cv2.bitwise_and(mask, ring))
        with log_timing(f"crop[{params.crop_mode}]", path.name):
            if params.crop_mode == CROP_MODE_PIXEL_EXACT:
                # fill_outside_mask здесь НЕ нужен: crop_pixel_exact заполняет всю
                # зону вне E2 сам — в осях crop-зоны, продолжая линию корешка прямо
                # (в fill_outside_mask этих осей нет, и Вороной её загибает).
                fade_color = book_mean_color(bgr, copy_mask)
                crop = crop_pixel_exact(
                    bgr,
                    cx,
                    cy,
                    angle,
                    crop_ext,
                    fade_color,
                    params.crop_fill_blur_px,
                    params.crop_fill_fade,
                    params.crop_fill_method,
                    copy_mask,
                )
            else:
                # Копируем только E2 ∩ B2: всё в B2 вне E2 заливаем цветом края
                with log_timing(f"fill_outside_mask[{params.bg_fill_method}]", path.name):
                    bgr_for_crop = fill_outside_mask(
                        bgr, copy_mask, method=params.bg_fill_method, blur_px=params.bg_fill_blur_px
                    )
                crop = crop_rotated(bgr_for_crop, cx, cy, angle, crop_ext, params.upscale)
        # Уровни — уже по вырезанному кадру: он весь состоит из книги и продлённого
        # от неё цвета, поэтому гистограмма по нему и есть гистограмма результата
        # (см. compensate_levels). Стретч монотонный и общий для всего кадра, так
        # что шов между содержимым и заливкой «ушей» остаётся незаметным.
        if params.compensate_levels:
            with log_timing("compensate_levels", path.name):
                crop = compensate_levels(crop)
        with log_timing("write_image", path.name):
            write_image(out_path, crop, write_params, params.force_dpi)

    if params.debug_dir is not None:
        dbg_path = (params.debug_dir / rel).with_suffix(".jpg")
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        with log_timing("draw_overlay", path.name):
            overlay = draw_overlay(
                bgr_orig,
                mask,
                geom,
                params.margins,
                finger_mask,
                lama_roi_bboxes,
                finger_boxes,
                copy_mask=copy_mask,
                finger_mask_predilate=finger_mask_predilate,
                layout_polygons=good_polys,
                parasitic_layout_polygons=parasitic_polys,
                layout_pad_px=params.layout_pad_px,
                crop_ext=crop_ext,
            )
        with log_timing("write_debug_overlay", path.name):
            cv2.imwrite(str(dbg_path), overlay, imwrite_params(".jpg"))


def run_batch(files: "list[Path]", params: CropParams, models) -> None:
    """Прогоняет :func:`process_frame` по всем ``files`` с прогресс-баром.

    Ошибка на отдельном кадре печатается с трейсбеком и не прерывает пачку:
    прогон по нескольким сотням файлов не должен падать целиком из-за одного
    битого снимка.
    """
    for path in tqdm(files, desc="Crop", unit="img"):
        t_frame = timeit.default_timer()
        try:
            process_frame(path, params, models)
            logger.info("%7.0f мс: ИТОГО кадр (%s)", (timeit.default_timer() - t_frame) * 1000.0, path.name)
        except Exception as e:
            import traceback

            tqdm.write(f"  Ошибка {path.name}: {e}")
            tqdm.write(traceback.format_exc())
