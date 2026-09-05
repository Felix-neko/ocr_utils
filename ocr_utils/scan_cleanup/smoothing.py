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

from ocr_utils.paper import INK_LEVEL, PAPER_BLUR_PX, PAPER_DILATE_PX
from ocr_utils.background_smoothing.processing import (
    BLUR_MODE_MASKED,
    DEFAULT_BLUR_MULT,
    DEFAULT_SAUVOLA_K,
    DEFAULT_THRESHOLD_BIAS,
    METHOD_SAUVOLA,
    PROTECT_DILATE_FRAC,
    MIN_GLYPH_AREA,
    SURE_GLYPH_AREA,
    SmoothResult,
    smooth_frame,
)

# Радиус защитного припуска по умолчанию, пикс. Замер по паку-1 при 600 dpi: шаг
# строк 89 px, кегль ≈52 px, межстрочный просвет ≈37 px. Радиус 25 даёт диаметр 50
# — то есть ядро размером с целую букву, — и как раз смыкает межстрочник, отчего
# текстовый блок становится сплошной защищённой плашкой (60% кадра против 47% при
# прежних 15 px). Дальше радиус букв уже не защищает: 35 px дают 64%, 45 px — 66%,
# и весь прирост идёт за счёт полей и межколонников, то есть той самой бумаги,
# ради которой размытие и затевалось.
DEFAULT_DILATE_PX = 25.0


@dataclass
class SmoothOptions:
    """Настройки размытия фона."""

    method: str = METHOD_SAUVOLA
    threshold_bias: float = DEFAULT_THRESHOLD_BIAS
    sauvola_k: float = DEFAULT_SAUVOLA_K
    sauvola_window: "int | None" = None
    min_glyph_area: int = MIN_GLYPH_AREA
    ink_level: "float | None" = INK_LEVEL
    sure_glyph_area: int = SURE_GLYPH_AREA
    paper_dilate_px: int = PAPER_DILATE_PX
    paper_blur_px: int = PAPER_BLUR_PX
    trust_strong: bool = False
    dilate_px: "float | None" = DEFAULT_DILATE_PX
    dilate_frac: float = PROTECT_DILATE_FRAC
    blur_px: "float | None" = None
    blur_frac: "float | None" = None
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
    opts: "SmoothOptions | None" = None,
) -> SmoothResult:
    """Размывает фон полосы, оставляя контент и размеченные иллюстрации нетронутыми."""
    opts = opts or SmoothOptions()
    return smooth_frame(
        src,
        gray,
        protect_mask=protect_mask,
        roi=roi,
        method=opts.method,
        bias=opts.threshold_bias,
        sauvola_k=opts.sauvola_k,
        sauvola_window=opts.sauvola_window,
        min_glyph_area=opts.min_glyph_area,
        ink_level=opts.ink_level,
        sure_glyph_area=opts.sure_glyph_area,
        paper_dilate_px=opts.paper_dilate_px,
        paper_blur_px=opts.paper_blur_px,
        trust_strong=opts.trust_strong,
        dilate_px=opts.dilate_px,
        dilate_frac=opts.dilate_frac,
        blur_px=opts.blur_px,
        blur_frac=opts.blur_frac,
        blur_mult=opts.blur_mult,
        blur_mode=opts.blur_mode,
        check_halftone=opts.halftone_guard,
    )
