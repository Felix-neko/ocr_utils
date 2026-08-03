"""Обход пачки файлов, обработка кадра и запись результата.

Весь расчёт по кадру живёт в ``processing``; здесь — только ввод-вывод, порядок
шагов и debug-оверлей. CLI (``cli``) собирает :class:`SmoothParams` и зовёт
:func:`run_batch`.
"""

import logging
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from ocr_utils.background_smoothing.layout import analysis_roi, make_detector, polygons_mask, raster_regions
from ocr_utils.background_smoothing.processing import (
    DEFAULT_BLUR_MULT,
    DEFAULT_SAUVOLA_K,
    DEFAULT_THRESHOLD_BIAS,
    METHOD_OTSU,
    compose,
    dilate_radius,
    has_content,
    has_halftone,
    normalized_blur,
    primary_mask,
)
from ocr_utils.scan_cropping.image_io import imwrite_params, read_dpi, resolve_output_suffix, write_image
from ocr_utils.scan_cropping.morphology import dilate_disk
from ocr_utils.timing import log_timing

logger = logging.getLogger(__name__)

# Режимы построения размытого фона (значения --blur-mode).
BLUR_MODE_MASKED = "masked"
BLUR_MODE_PLAIN = "plain"
BLUR_MODES = (BLUR_MODE_MASKED, BLUR_MODE_PLAIN)

# Цвета заливки масок на debug-оверлее (BGR) и их прозрачность.
COLOR_DILATED = (0, 255, 255)  # жёлтый — защитная маска M_dilated (что НЕ размывается)
COLOR_PRIMARY = (0, 0, 255)  # красный — первичная маска M_primary (найденный контент)
ALPHA_DILATED = 0.25
ALPHA_PRIMARY = 0.45

# Рамка вокруг блоков-иллюстраций Surya (--use-surya-layout): ярко-сиреневая, без
# заливки — под ней должно быть видно саму фотографию, чтобы по оверлею судить,
# насколько точно сеть её обвела.
COLOR_PICTURE = (255, 0, 200)
# Ярко-зелёная рамка — связные растровые области, добавленные к блокам Surya
# (``layout.raster_regions``). Отдельный цвет затем и нужен, чтобы на оверлее было
# видно, где сеть промахнулась и насколько её поправил детектор растра.
COLOR_RASTER = (0, 255, 0)
# Толщина рамки как доля длинной стороны кадра (~5 px при 6200 px): константа в
# пикселях была бы невидима на 21-Мп кадре и жирной на превью.
PICTURE_OUTLINE_FRAC = 0.0008


@dataclass
class SmoothParams:
    """Все настройки прогона (значения опций CLI в одном месте).

    Собирается один раз в ``cli.main`` и дальше только читается, поэтому
    :func:`process_frame` не тащит полтора десятка отдельных аргументов.
    """

    input_dir: Path
    output_dir: Path
    debug_dir: Optional[Path] = None

    # Обход и вывод
    recursive: bool = True
    skip_if_exists: bool = True
    output_format: Optional[str] = None
    to_gray: bool = False

    # Первичная маска
    method: str = METHOD_OTSU
    threshold_bias: float = DEFAULT_THRESHOLD_BIAS
    sauvola_k: float = DEFAULT_SAUVOLA_K
    sauvola_window: Optional[int] = None

    # Защитный припуск и размытие фона
    dilate_px: Optional[float] = None
    blur_mult: float = DEFAULT_BLUR_MULT
    blur_mode: str = BLUR_MODE_MASKED

    # Разметка страницы нейросетью (защита растровых иллюстраций)
    use_surya_layout: bool = False


def draw_overlay(
    bgr: np.ndarray,
    m_primary: np.ndarray,
    m_dilated: np.ndarray,
    picture_polys: "list[np.ndarray] | None" = None,
    raster_polys: "list[np.ndarray] | None" = None,
) -> np.ndarray:
    """Исходный кадр с полупрозрачной заливкой обеих масок.

    Показывает именно ВХОД обработки: по оверлею видно, что было найдено как
    контент (красный) и что за счёт припуска попало под защиту (жёлтый).
    Результат размытия сюда не подмешивается — иначе не отличить, где фон
    сгладился, а где просто не было контента.

    Порядок важен: ``M_dilated`` крупнее и кладётся первой, ``M_primary`` — поверх.

    ``picture_polys`` (при ``--use-surya-layout``) обводятся ярко-сиреневой рамкой
    поверх всего: заливка показала бы, что блок защищён, но не показала бы, ГДЕ
    именно сеть провела границу иллюстрации, а это и есть то, что нужно проверять.
    ``raster_polys`` — ярко-зелёным: это связные растровые области, добавленные к
    блокам, и по расхождению двух рамок сразу видно, где Surya промахнулась.
    """
    out = bgr.copy()
    for mask, color, alpha in ((m_dilated, COLOR_DILATED, ALPHA_DILATED), (m_primary, COLOR_PRIMARY, ALPHA_PRIMARY)):
        sel = mask > 0
        if not sel.any():
            continue
        # Смешиваем только под маской: cv2.addWeighted по всему кадру приглушил бы
        # и незакрашенные области, а нам нужен неизменный исходник рядом с заливкой.
        # Цвет подмешивается как константа, без полноразмерного массива-заливки:
        # на кадре в 21 Мп это лишние 63 МБ на каждую из двух масок.
        tint = np.asarray(color, dtype=np.float32) * alpha
        out[sel] = (out[sel] * (1.0 - alpha) + tint).astype(np.uint8)

    thickness = max(1, int(round(PICTURE_OUTLINE_FRAC * max(bgr.shape[0], bgr.shape[1]))))
    for polys, color in ((picture_polys, COLOR_PICTURE), (raster_polys, COLOR_RASTER)):
        if polys:
            cv2.polylines(out, [np.round(p).astype(np.int32) for p in polys], True, color, thickness, cv2.LINE_AA)
    return out


def _write_overlay(
    path: Path,
    params: SmoothParams,
    rel: Path,
    bgr: np.ndarray,
    m_primary: np.ndarray,
    m_dilated: np.ndarray,
    picture_polys: "list[np.ndarray] | None" = None,
    raster_polys: "list[np.ndarray] | None" = None,
) -> None:
    """Пишет debug-оверлей, если задан ``debug_dir``; иначе ничего не делает.

    Пишется и для кадров, оставленных без обработки: пустой оверлей — это тоже
    ответ, по нему видно, что контент не выделился, а не что файл потерялся.
    """
    if params.debug_dir is None:
        return
    dbg_path = (params.debug_dir / rel).with_suffix(".jpg")
    dbg_path.parent.mkdir(parents=True, exist_ok=True)
    with log_timing("overlay", path.name):
        overlay = draw_overlay(bgr, m_primary, m_dilated, picture_polys, raster_polys)
        cv2.imwrite(str(dbg_path), overlay, imwrite_params(".jpg"))


def process_frame(path: Path, params: SmoothParams, detector=None) -> None:
    """Обрабатывает один кадр и пишет результат (и оверлей, если задан ``debug_dir``).

    Кадр, который трогать нельзя (чистый лист, обложка, растровая вкладка), копируется
    на выход без изменений — см. :func:`has_content` и :func:`has_halftone`.

    ``detector`` — ``layout.LayoutDetector`` при ``--use-surya-layout``, иначе ``None``.
    Порядок при включённом флаге такой:

    1. Surya находит блоки-иллюстрации, к ним добавляются граничащие связные
       растровые области (``layout.raster_regions`` — блок Surya это визуальный
       блок, он может и прихватить пустую бумагу, и срезать край фотографии);
       всё остальное поле кадра становится областью анализа ``roi``.
    2. Проверки «есть ли контент» и «нет ли растра» идут ПО ``roi`` — то есть по
       остатку страницы. Детектор растра при этом остаётся страховкой: если растр
       нашёлся и ВНЕ найденных иллюстраций (полосная фотография, обложка, которую
       layout не разметил), кадр по-прежнему копируется как есть.
    3. Только если остаток прошёл обе проверки, строится маска и размывается фон.
       Иллюстрации входят в защитную маску с тем же припуском, что и текст.

    Сообщения печатаются через ``tqdm.write``, чтобы не разрывать прогресс-бар.
    """
    rel = path.relative_to(params.input_dir)
    out_suffix = resolve_output_suffix(path.suffix, params.output_format)
    out_path = (params.output_dir / rel).with_suffix(out_suffix)

    # Проверяем ДО imread: чтение 20-40 Мп TIFF заметно дороже, чем stat файла.
    if params.skip_if_exists and out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        tqdm.write(f"  Не читается, пропускаю: {path.name}")
        return

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    src = gray if params.to_gray else bgr

    picture_polys: "list[np.ndarray]" = []
    raster_polys: "list[np.ndarray]" = []
    if detector is not None:
        with log_timing("surya_layout", path.name):
            picture_polys = detector.picture_polygons(bgr, gray)
            # Блоки Surya — это визуальные блоки, а не контуры растра: к ним
            # добавляются граничащие связные растровые области (см. layout).
            raster_polys = raster_regions(gray, picture_polys)
    m_picture = polygons_mask(gray.shape, picture_polys + raster_polys)
    roi = analysis_roi(m_picture, picture_polys)

    # Два случая, когда кадр трогать нельзя, и оба кончаются копированием как есть
    # (пропустить файл совсем значило бы оставить дыру в паке):
    #   * контент не выделяется — Оцу режет собственное зерно, маске верить нельзя;
    #   * есть крупная растровая область — обложка или полутоновая вкладка, там зерно
    #     и есть содержимое, а размытие выело бы его островами.
    # Обе проверки — по области анализа ``roi``, то есть без уже опознанных иллюстраций.
    skip_reason = ""
    if not has_content(gray, roi):
        skip_reason = "контент не выделяется (чистый лист?)"
    elif has_halftone(gray, roi=roi):
        skip_reason = "крупная растровая область (обложка или вкладка?)"
    if skip_reason:
        tqdm.write(f"  {skip_reason}, копирую без изменений: {path.name}")
        write_image(out_path, src, imwrite_params(out_suffix), read_dpi(path))
        zeros = np.zeros(gray.shape, np.uint8)
        _write_overlay(path, params, rel, bgr, zeros, zeros, picture_polys, raster_polys)
        return

    with log_timing("primary_mask", path.name):
        m_primary = primary_mask(
            gray,
            method=params.method,
            bias=params.threshold_bias,
            sauvola_k=params.sauvola_k,
            window=params.sauvola_window,
            roi=roi,
        )

    radius = dilate_radius(gray.shape, params.dilate_px)
    with log_timing("dilate", path.name):
        m_dilated = dilate_disk(m_primary, radius)
        # Иллюстрации присоединяются к защитной маске ПОСЛЕ отдельной дилатации тем же
        # радиусом: припуск у них такой же, как у текста, а объединять маски до
        # дилатации нельзя — на оверлее фотография залилась бы красным как «контент».
        if picture_polys:
            m_dilated = cv2.bitwise_or(m_dilated, dilate_disk(m_picture, radius))

    with log_timing(f"blur[{params.blur_mode}]", path.name):
        weight = None if params.blur_mode == BLUR_MODE_PLAIN else (m_dilated == 0).astype(np.uint8)
        blurred = normalized_blur(src, weight, radius * params.blur_mult)

    with log_timing("compose", path.name):
        final = compose(src, blurred, m_dilated)

    write_image(out_path, final, imwrite_params(out_suffix), read_dpi(path))
    _write_overlay(path, params, rel, bgr, m_primary, m_dilated, picture_polys, raster_polys)


def run_batch(files: "list[Path]", params: SmoothParams) -> None:
    """Прогоняет :func:`process_frame` по всем ``files`` с прогресс-баром.

    Ошибка на отдельном кадре печатается с трейсбеком и не прерывает пачку:
    на прогоне в сотню сканов обиднее потерять всё из-за одного битого файла.

    Детектор layout создаётся здесь один раз на всю пачку (веса Surya грузятся
    при первом кадре) и передаётся в :func:`process_frame`.
    """
    detector = make_detector(params.use_surya_layout)
    for path in tqdm(files, desc="BgSmooth", unit="img"):
        t_frame = timeit.default_timer()
        try:
            process_frame(path, params, detector)
            logger.info("%7.0f мс: ИТОГО кадр (%s)", (timeit.default_timer() - t_frame) * 1000.0, path.name)
        except Exception as e:
            import traceback

            tqdm.write(f"  Ошибка {path.name}: {e}")
            tqdm.write(traceback.format_exc())
