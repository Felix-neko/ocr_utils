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

from ocr_utils.background_smoothing.processing import (
    BLUR_MODE_MASKED,
    DEFAULT_BLUR_MULT,
    DEFAULT_SAUVOLA_K,
    DEFAULT_THRESHOLD_BIAS,
    METHOD_SAUVOLA,
    PROTECT_DILATE_FRAC,
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
    dilate_px: "float | None" = DEFAULT_DILATE_PX
    dilate_frac: float = PROTECT_DILATE_FRAC
    blur_px: "float | None" = None
    blur_frac: "float | None" = None
    blur_mult: float = DEFAULT_BLUR_MULT
    blur_mode: str = BLUR_MODE_MASKED
    halftone_guard: bool = True


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
        dilate_px=opts.dilate_px,
        dilate_frac=opts.dilate_frac,
        blur_px=opts.blur_px,
        blur_frac=opts.blur_frac,
        blur_mult=opts.blur_mult,
        blur_mode=opts.blur_mode,
        check_halftone=opts.halftone_guard,
    )
