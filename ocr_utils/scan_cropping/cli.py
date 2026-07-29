"""CLI вырезки сканов: ``python -m ocr_utils.scan_cropping``.

Здесь ТОЛЬКО click: объявление опций, их разбор и сборка :class:`CropParams`.
Вся обработка изображений — в остальных модулях пакета, обход пачки файлов —
в :func:`ocr_utils.scan_cropping.pipeline.run_batch`.

    uv run python -m ocr_utils.scan_cropping \\
        --input-dir IN --output-dir OUT --debug-dir DBG \\
        --left-margin -150 --top-margin -150 --right-margin -150 --bottom-margin -150
"""

import logging
from pathlib import Path
from typing import Optional

import click

from ocr_utils.scan_cropping.background_fill import (
    BG_FILL_AVERAGE,
    BG_FILL_METHODS,
    CROP_FILL_METHODS,
    CROP_FILL_REPLICATE,
)
from ocr_utils.scan_cropping.cropping import (
    CROP_FILL_BLUR_PX,
    CROP_FILL_FADE,
    CROP_MODE_PIXEL_EXACT,
    CROP_MODE_ROTATE,
    CROP_MODES,
)
from ocr_utils.scan_cropping.finger_removal.asymmetric_dilation import DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO
from ocr_utils.scan_cropping.finger_removal.finger_shadow import SHADOW_METHODS, shadow_variant
from ocr_utils.scan_cropping.finger_removal.removal import FINGER_DILATE_PX, FINGER_ZONE_LIGHT_INCREMENT
from ocr_utils.scan_cropping.finger_removal.text_protection import (
    DEFAULT_LAYOUT_PAD_PX,
    PROTECT_LIMIT_LAMA,
    PROTECT_MODES,
)
from ocr_utils.scan_cropping.geometry import EXTRA_EROSION_PX
from ocr_utils.scan_cropping.gpu_models import GpuModels
from ocr_utils.scan_cropping.image_io import collect_images
from ocr_utils.scan_cropping.pipeline import CropParams, run_batch

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_light_increment(ctx, param, value: str) -> "tuple[float, float]":
    """Парсит ``--finger-zone-light-increment``: 'N' → (N, N), 'L,R' → (L, R)."""
    parts = [p.strip() for p in str(value).split(",")]
    try:
        if len(parts) == 1:
            v = float(parts[0])
            return (v, v)
        if len(parts) == 2:
            return (float(parts[0]), float(parts[1]))
    except ValueError:
        pass
    raise click.BadParameter("ожидается число ('20') или пара 'слева,справа' ('15,30')")


def _parse_layout_pad(ctx, param, value) -> "tuple[int, int]":
    """Парсит ``--layout-pad-px``: 'N' → (N, N), 'X,Y' → (X, Y) (запас по X и по Y)."""
    parts = [p.strip() for p in str(value).split(",")]
    try:
        if len(parts) == 1:
            v = int(parts[0])
            return (v, v)
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except ValueError:
        pass
    raise click.BadParameter("ожидается число ('12') или пара 'по_x,по_y' ('12,24')")


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Каталог с исходными изображениями (JPG/PNG)",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Куда сохранять повёрнутые и обрезанные развороты (имя файла сохраняется)",
)
@click.option(
    "--debug-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Если задана — сюда кадр с оверлеями (граница/min-bbox/crop-зона)",
)
@click.option("--left-margin", default=0, show_default=True, help="Припуск crop-зоны слева, пикс. (>0 шире, <0 уже)")
@click.option("--top-margin", default=0, show_default=True, help="Припуск crop-зоны сверху, пикс. (>0 шире, <0 уже)")
@click.option("--right-margin", default=0, show_default=True, help="Припуск crop-зоны справа, пикс. (>0 шире, <0 уже)")
@click.option("--bottom-margin", default=0, show_default=True, help="Припуск crop-зоны снизу, пикс. (>0 шире, <0 уже)")
@click.option("--recursive", is_flag=True, default=False, help="Рекурсивно обходить подкаталоги в поисках картинок")
@click.option(
    "--skip-if-exists/--no-skip-if-exists",
    "skip_if_exists",
    default=True,
    show_default=True,
    help="Пропускать файл, если соответствующий результат уже есть в OUTPUT_DIR (докачка "
    "прерванного прогона). Проверяется только output-файл; debug-оверлей при обработке файла "
    "перезаписывается всегда. При --no-skip-if-exists всё пересчитывается заново.",
)
@click.option(
    "--output-format",
    type=click.Choice(["png", "tiff"], case_sensitive=False),
    default=None,
    help="Формат файлов в output-dir (по умолчанию — как у входного файла)",
)
@click.option(
    "--compensate-levels/--no-compensate-levels",
    "do_compensate_levels",
    default=False,
    show_default=True,
    help="Растягивать уровни по перцентилям внутри маски страницы (минус эрозия)",
)
@click.option(
    "--extra-erosion-px",
    default=EXTRA_EROSION_PX,
    show_default=True,
    help="Доп. обрезка краёв силуэта книги перед копированием, пикс. (диляция на extra + "
    "эрозия на 2*extra → срезает тёмные фрагменты обложки в углах; 0 — выкл.)",
)
@click.option(
    "--upscale",
    default=None,
    type=float,
    show_default=True,
    help="Апскейл выходного изображения перед поворотом/кропом (по умолчанию — без апскейла)",
)
@click.option(
    "--crop-mode",
    type=click.Choice(CROP_MODES, case_sensitive=False),
    default=CROP_MODE_ROTATE,
    show_default=True,
    help="Способ вырезки crop-зоны: rotate — повернуть кадр на найденный угол и вырезать "
    "выпрямленный прямоугольник (интерполяция всего кадра; на скромном разрешении и большом "
    "угле слегка мылит); pixel-exact — скопировать crop-зону пиксель-в-пиксель в минимальный "
    "осевой холст, книга остаётся наклонённой (выпрямлять снаружи, напр. в ScanTailor), "
    "«уши» по углам заполняются заливкой (см. --crop-fill-blur-px / --crop-fill-fade). "
    "В режиме pixel-exact --upscale не применяется (он бы вернул интерполяцию)",
)
@click.option(
    "--crop-fill-method",
    type=click.Choice(CROP_FILL_METHODS, case_sensitive=False),
    default=CROP_FILL_REPLICATE,
    show_default=True,
    help="Способ заливки «ушей» (--crop-mode=pixel-exact): replicate — продлить краевые пиксели "
    "crop-зоны наружу по нормали к её сторонам (clamp-to-edge): линия корешка, выходящая из "
    "зоны, продолжается прямо и ScanTailor находит по ней разрез; voronoi — цвет ближайшей "
    "точки границы зоны: у углов расходится веером и загибает такие линии",
)
@click.option(
    "--crop-fill-blur-px",
    default=CROP_FILL_BLUR_PX,
    type=float,
    show_default=True,
    help="Макс. σ размытия заливки «ушей» (--crop-mode=pixel-exact): у границы crop-зоны "
    "размытия нет, вдали растёт до этого значения; 0 — не размывать",
)
@click.option(
    "--crop-fill-fade",
    default=CROP_FILL_FADE,
    type=float,
    show_default=True,
    help="Доля выцветания заливки «ушей» к среднему цвету книги (--crop-mode=pixel-exact) на "
    "самом дальнем от crop-зоны пикселе: 1 — полностью уходит в средний цвет, 0 — не выцветать",
)
@click.option(
    "--remove-fingers/--no-remove-fingers",
    "do_remove_fingers",
    default=True,
    show_default=True,
    help="Детектировать и закрашивать пальцы (finger_removal) перед детекцией книги/кропом",
)
@click.option(
    "--finger-dilate-px",
    default=FINGER_DILATE_PX,
    show_default=True,
    help="Дилатация маски пальца, пикс. (шире — надёжнее докрашивает полутона на краю силуэта)",
)
@click.option(
    "--finger-zone-light-increment",
    "finger_zone_light_increment",
    default=str(FINGER_ZONE_LIGHT_INCREMENT),
    show_default=True,
    callback=_parse_light_increment,
    help="Осветление зоны пальца перед закраской: одно число (на весь кадр) "
    "либо 'слева,справа' (напр. 15,30) — если свет в кадре падает не симметрично",
)
@click.option(
    "--force-dpi",
    default=None,
    type=int,
    show_default=True,
    help="Принудительно прописать выходным изображениям указанный DPI (по умолчанию — не трогать)",
)
@click.option(
    "--max-asymmetric-dilation-ratio",
    "asymmetric_dilation_ratio",
    default=DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO,
    type=float,
    show_default=True,
    help="Максимальная ДОБАВКА к коэффициенту дилатации маски пальца по «выгодной» оси: "
    "боковой палец растёт по Y в (1 + ratio) раз, верхний/нижний — по X, угловой — в "
    "(1 + ratio/2) по обеим. 0 — прежняя круговая дилатация",
)
@click.option(
    "--protect-text-layout/--no-protect-text-layout",
    "protect_text_layout",
    default=False,
    show_default=True,
    help="Прогонять кадр через Surya layout ДО закраски и защищать найденные блоки (текст, "
    "заголовки, картинки, таблицы) от закраски пальца — позволяет ставить щедрую дилатацию "
    "под тень, не портя контент. Способ защиты — см. --text-protect-mode",
)
@click.option(
    "--text-protect-mode",
    type=click.Choice(PROTECT_MODES, case_sensitive=False),
    default=PROTECT_LIMIT_LAMA,
    show_default=True,
    help="Способ защиты блоков layout (--protect-text-layout): limit-lama-zone — вычесть блоки из "
    "зоны закраски; copy-back-layout-zones — закрасить всё, а потом скопировать пересекающиеся "
    "с зоной закраски блоки обратно с оригинала (лучше вычищает тень между блоками)",
)
@click.option(
    "--layout-pad-px",
    default=str(DEFAULT_LAYOUT_PAD_PX),
    show_default=True,
    callback=_parse_layout_pad,
    help="Запас вокруг блока layout, пикс. (--protect-text-layout): Surya обводит блок впритык, "
    "и без запаса закраска подъедает край строки. Число ('12') — одинаково по обеим осям, пара "
    "'по_x,по_y' ('12,24') — раздельно. Больше запас — целее контент, но хуже вычищается тень у "
    "самого блока; 0 — строго по границе блока",
)
@click.option(
    "--shadow-method",
    type=click.Choice(SHADOW_METHODS, case_sensitive=False),
    default="none",
    show_default=True,
    help="Коррекция теневой зоны вокруг пальца после зарисовки: none | classic | retinex | "
    "docshadow-sd7k | docshadow-kligler | docshadow-jung (нейросеть DocShadow)",
)
@click.option(
    "--bg-fill-method",
    type=click.Choice(BG_FILL_METHODS, case_sensitive=False),
    default=BG_FILL_AVERAGE,
    show_default=True,
    help="Способ заливки внешней зоны (в углы повёрнутого кропа за краем страницы): "
    "average — один усреднённый цвет по всей странице (старый); nearest — цвет ближайшей "
    "точки границы зоны копирования (Вороной): учитывает неравномерный свет и цветную "
    "обложку, можно сгладить через --bg-fill-blur-px",
)
@click.option(
    "--bg-fill-blur-px",
    type=float,
    default=0.0,
    show_default=True,
    help="Размытие зоны заливки (только для локальных методов заливки): 0 — выкл; иначе σ "
    "растёт от 0 у границы зоны копирования до этого значения вдали. Зона копирования не "
    "затрагивается",
)
@click.option(
    "--detect-pad-tb-px",
    type=int,
    default=250,
    show_default=True,
    help="Перед детекцией разворота добавить чёрную рамку сверху и снизу на N пикселей (маска "
    "возвращается в координатах кадра, рамка срезается обратно). Помогает на снимках, где книга "
    "занимает кадр целиком по вертикали и детектится «вся область как разворот»: чёрная рамка "
    "даёт SAM тёмную границу и отодвигает боксы от краёв. 0 — выкл",
)
@click.option(
    "--page-close-px",
    type=int,
    default=None,
    show_default=True,
    help="Радиус смыкания силуэта разворота, пикс. По умолчанию считается по размеру кадра "
    "(≈30 px при 300 DPI, ≈45 px при 450 DPI). Смыкание собирает страницу обратно, когда SAM "
    "вернул силуэт не листа, а напечатанного текста (маска идёт по глифам), поэтому радиус "
    "должен перекрывать межстрочный интервал и растёт вместе с разрешением съёмки. Мало — верх "
    "страницы выпадает из маски и затирается заливкой при кропе; много — лишнего не съедает, "
    "но и не помогает",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="WARNING",
    show_default=True,
    help="Уровень логирования. INFO — печатать тайминги ключевых операций (timeit) по каждому кадру",
)
def main(
    input_dir: Path,
    output_dir: Path,
    debug_dir: Optional[Path],
    left_margin: int,
    top_margin: int,
    right_margin: int,
    bottom_margin: int,
    recursive: bool,
    skip_if_exists: bool,
    output_format: Optional[str],
    do_compensate_levels: bool,
    extra_erosion_px: int,
    upscale: Optional[float],
    crop_mode: str,
    crop_fill_method: str,
    crop_fill_blur_px: float,
    crop_fill_fade: float,
    do_remove_fingers: bool,
    finger_dilate_px: int,
    finger_zone_light_increment: "tuple[float, float]",
    force_dpi: Optional[int],
    asymmetric_dilation_ratio: float,
    protect_text_layout: bool,
    text_protect_mode: str,
    layout_pad_px: "tuple[int, int]",
    shadow_method: str,
    bg_fill_method: str,
    bg_fill_blur_px: float,
    detect_pad_tb_px: int,
    page_close_px: Optional[int],
    log_level: str,
) -> None:
    """Находит разворот и вырезает crop-зону в OUTPUT_DIR (способ — см. --crop-mode)."""
    logging.getLogger().setLevel(log_level.upper())
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    params = CropParams(
        input_dir=input_dir,
        output_dir=output_dir,
        debug_dir=debug_dir,
        # Припуски crop-зоны: (left, top, right, bottom) — по одному на сторону
        margins=(left_margin, top_margin, right_margin, bottom_margin),
        recursive=recursive,
        skip_if_exists=skip_if_exists,
        output_format=output_format,
        force_dpi=force_dpi,
        compensate_levels=do_compensate_levels,
        extra_erosion_px=extra_erosion_px,
        crop_mode=crop_mode,
        upscale=upscale,
        crop_fill_method=crop_fill_method,
        crop_fill_blur_px=crop_fill_blur_px,
        crop_fill_fade=crop_fill_fade,
        bg_fill_method=bg_fill_method,
        bg_fill_blur_px=bg_fill_blur_px,
        remove_fingers=do_remove_fingers,
        finger_dilate_px=finger_dilate_px,
        finger_zone_light_increment=finger_zone_light_increment,
        asymmetric_dilation_ratio=asymmetric_dilation_ratio,
        shadow_method=shadow_method,
        protect_text_layout=protect_text_layout,
        text_protect_mode=text_protect_mode,
        layout_pad_px=layout_pad_px,
        detect_pad_tb_px=detect_pad_tb_px,
        page_close_px=page_close_px,
    )

    files = collect_images(input_dir, recursive)
    if not files:
        logger.warning("Изображения не найдены в %s", input_dir)
        return

    if crop_mode == CROP_MODE_PIXEL_EXACT and upscale is not None:
        logger.warning("--upscale игнорируется при --crop-mode=%s: он вернул бы интерполяцию", CROP_MODE_PIXEL_EXACT)

    # Модели грузятся ОДИН раз на весь прогон: Surya и DocShadow — только если их
    # действительно просят соответствующие опции (обе стоят дорого и по VRAM, и по
    # времени старта).
    with GpuModels(with_layout=protect_text_layout, shadow_variant=shadow_variant(shadow_method)) as models:
        logger.info(
            "Файлов: %d | устройство: %s | margins: left=%d top=%d right=%d bottom=%d | recursive: %s | "
            "skip-if-exists: %s | "
            "output-format: %s | compensate-levels: %s | extra-erosion-px=%d | upscale: %s | "
            "crop-mode: %s (fill=%s, fill-blur-px=%g, fill-fade=%g) | "
            "remove-fingers: %s (dilate-px=%d, light-increment=слева=%g,справа=%g) | force-dpi: %s | "
            "max-asymmetric-dilation-ratio: %g | protect-text-layout: %s (mode=%s, pad-px=x=%d,y=%d) | "
            "shadow-method: %s | bg-fill-method: %s (blur-px=%g) | detect-pad-tb-px: %d | "
            "page-close-px: %s",
            len(files),
            models.device,
            left_margin,
            top_margin,
            right_margin,
            bottom_margin,
            recursive,
            skip_if_exists,
            output_format or "как у входа",
            do_compensate_levels,
            extra_erosion_px,
            upscale if upscale is not None else "без апскейла",
            crop_mode,
            crop_fill_method,
            crop_fill_blur_px,
            crop_fill_fade,
            do_remove_fingers,
            finger_dilate_px,
            finger_zone_light_increment[0],
            finger_zone_light_increment[1],
            force_dpi if force_dpi is not None else "не трогать",
            asymmetric_dilation_ratio,
            protect_text_layout,
            text_protect_mode,
            layout_pad_px[0],
            layout_pad_px[1],
            shadow_method,
            bg_fill_method,
            bg_fill_blur_px,
            detect_pad_tb_px,
            page_close_px if page_close_px is not None else "по размеру кадра",
        )
        run_batch(files, params, models)

    logger.info("Готово. Crop → %s%s", output_dir, f" | debug → {debug_dir}" if debug_dir else "")


if __name__ == "__main__":
    main()
