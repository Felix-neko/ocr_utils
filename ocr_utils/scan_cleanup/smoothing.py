"""Операция 2: размытие фона по защитной маске из базы.

Тонкая обёртка над ``background_smoothing.processing.smooth_frame``: весь расчёт
там, здесь — только настройки и подстановка размеченных областей вместо Surya.

ЧЕМ ОТЛИЧАЕТСЯ ОТ ``background_smoothing``:

* иллюстрации не ищутся сетью, а берутся из базы — точными прямоугольниками,
  выверенными человеком в CVAT;
* умолчание бинаризации — Sauvola, а не Otsu. Ветка Sauvola в ``primary_mask``
  ОБЪЕДИНЯЕТСЯ с глобальным порогом, а не заменяет его, то есть может только
  добрать контент. Систематической потери букв у Otsu на паке-1 не нашлось (2
  тайла из 1562), но размытие здесь идёт сильнее, и цена пропуска выросла, а цена
  лишнего — всего лишь недоразмытая крупинка;
* предохранитель «крупная растровая область» остаётся включённым как страховка от
  промаха разметки, но срабатывать должен редко: обложки и вкладки отсекаются
  раньше, по флагу ``full_page`` из базы.
"""

from dataclasses import dataclass

import numpy as np

from ocr_utils.paper import INK_LEVEL, PAPER_BLUR_MM, PAPER_DILATE_MM
from ocr_utils.background_smoothing.processing import (
    BLUR_MODE_MASKED,
    DEFAULT_BLUR_MULT,
    DEFAULT_SAUVOLA_K,
    DEFAULT_THRESHOLD_BIAS,
    METHOD_SAUVOLA,
    SmoothResult,
    smooth_frame,
)
from ocr_utils.units import mm2_to_px2, mm_to_px

# РАЗМЕРЫ — В ММ БУМАГИ, в пиксели переводятся по dpi полосы из базы (``Page.dpi``). Числа
# замерены на паке-1 при 600 dpi; в комментариях — прежние пиксели при 600 dpi, которые они
# дают ровно. На паках 450 dpi те же мм дают втрое-вчетверо меньше пикселей, как и должно.

# Радиус защитного припуска по умолчанию: 25 px при 600 dpi. Замер по паку-1 при 600 dpi:
# шаг строк 89 px, кегль ≈52 px, межстрочный просвет ≈37 px. Радиус 25 даёт диаметр 50
# — то есть ядро размером с целую букву, — и как раз смыкает межстрочник, отчего
# текстовый блок становится сплошной защищённой плашкой (60% кадра против 47% при
# прежних 15 px). Дальше радиус букв уже не защищает: 35 px дают 64%, 45 px — 66%,
# и весь прирост идёт за счёт полей и межколонников, то есть той самой бумаги,
# ради которой размытие и затевалось.
DEFAULT_DILATE_MM = 1.058

# Окно Саволы: 101 px при 600 dpi — порядка двух-трёх высот строки. Раньше по умолчанию
# считалось долей длинной стороны кадра (``SAUVOLA_WINDOW_FRAC``), но доля ломается на
# обрезанных полосах, а run-скрипты всё равно передавали 101.
SAUVOLA_WINDOW_MM = 4.276

# Минимальная площадь связной области первичной маски: 34 px² при 600 dpi
# (``background_smoothing.processing.MIN_GLYPH_AREA``).
MIN_GLYPH_AREA_MM2 = 0.0609

# Площадь, с которой область подтверждается сама по себе: 500 px² при 600 dpi
# (``background_smoothing.processing.SURE_GLYPH_AREA``).
SURE_GLYPH_AREA_MM2 = 0.896


@dataclass
class SmoothOptions:
    """Настройки размытия фона; размеры — в мм бумаги, площади — в мм²."""

    method: str = METHOD_SAUVOLA
    threshold_bias: float = DEFAULT_THRESHOLD_BIAS
    sauvola_k: float = DEFAULT_SAUVOLA_K
    sauvola_window_mm: float = SAUVOLA_WINDOW_MM
    # 0 — не чистить первичную маску вовсе.
    min_glyph_area_mm2: float = MIN_GLYPH_AREA_MM2
    ink_level: "float | None" = INK_LEVEL
    sure_glyph_area_mm2: float = SURE_GLYPH_AREA_MM2
    paper_dilate_mm: float = PAPER_DILATE_MM
    paper_blur_mm: float = PAPER_BLUR_MM
    trust_strong: bool = False
    dilate_mm: float = DEFAULT_DILATE_MM
    # None — радиус размытия = радиус припуска × ``blur_mult``.
    blur_mm: "float | None" = None
    blur_mult: float = DEFAULT_BLUR_MULT
    blur_mode: str = BLUR_MODE_MASKED
    # Предохранитель ``has_halftone`` здесь ВЫКЛЮЧЕН, в отличие от background_smoothing.
    # Там он был единственной защитой от обложек: иллюстрации искала Surya, и ошибиться
    # было легко. Здесь растровые области размечены руками и выверены в CVAT, полосные
    # отсекаются раньше по базе (``is_full_page``), а размеченные исключены из области
    # анализа — то есть искать растр предохранителю остаётся ровно там, где его нет.
    # Замер на 1976 году: сработал на 3 полосах из 1194, и все три — ложные (1976/01
    # IMG_0005_1L, 1976/02 IMG_0095_2R, 1976/03 IMG_0138_1L). Ловил он тёмную кромку у
    # края скана и пятно на поле: размыкание 15 px по копии 1/4 — это 60 px кадра,
    # всего одна буква при 600 dpi, и вертикальная полоска тени его переживает. Цена
    # ошибки высока и незаметна: полоса целиком уходит в копию без размытия.
    halftone_guard: bool = False


def smooth_page(
    src: np.ndarray,
    gray: np.ndarray,
    protect_mask: "np.ndarray | None",
    roi: "np.ndarray | None",
    opts: "SmoothOptions | None",
    dpi: int,
) -> SmoothResult:
    """Размывает фон полосы, оставляя контент и размеченные иллюстрации нетронутыми.

    Args:
        src: Полоса BGR.
        gray: Её серая копия.
        protect_mask: Размеченные иллюстрации (защищаются целиком) или ``None``.
        roi: Область анализа или ``None`` — весь кадр.
        opts: Настройки; ``None`` — умолчания.
        dpi: Разрешение полосы: по нему размеры из ``opts`` переводятся в пиксели.

    Returns:
        ``SmoothResult`` — кадр, маски и фактические радиусы в пикселях.
    """
    opts = opts or SmoothOptions()
    return smooth_frame(
        src,
        gray,
        protect_mask=protect_mask,
        roi=roi,
        method=opts.method,
        bias=opts.threshold_bias,
        sauvola_k=opts.sauvola_k,
        sauvola_window=mm_to_px(opts.sauvola_window_mm, dpi),
        min_glyph_area=mm2_to_px2(opts.min_glyph_area_mm2, dpi, minimum=0),
        ink_level=opts.ink_level,
        sure_glyph_area=mm2_to_px2(opts.sure_glyph_area_mm2, dpi, minimum=0),
        paper_dilate_px=mm_to_px(opts.paper_dilate_mm, dpi),
        paper_blur_px=mm_to_px(opts.paper_blur_mm, dpi),
        trust_strong=opts.trust_strong,
        dilate_px=float(mm_to_px(opts.dilate_mm, dpi, minimum=0)),
        blur_px=None if opts.blur_mm is None else float(mm_to_px(opts.blur_mm, dpi, minimum=0)),
        blur_mult=opts.blur_mult,
        blur_mode=opts.blur_mode,
        check_halftone=opts.halftone_guard,
    )
