"""Закрас пальцев «inkback»: вся раздутая зона пальца по контексту без блоков текста (LaMa), затем возврат краски с оригинала."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.glyphs import GLYPH_SCALE, glyph_protect
from ocr_utils.scan_cropping.v2.scale import REF_PX_PER_MM, Scale

# ЕДИНИЦЫ. Размеры «px кадра» ниже — px КАЛИБРОВОЧНОГО кадра (13.8 px/мм, подшивки МТО 1990): на кадре другой
# плотности они пересчитываются через ``Scale.frame_px`` (``px_per_mm`` кадра передаётся в функции), то есть
# заданы фактически в мм бумаги (px / 13.8). Размеры «px ¼ кадра» — на копии, приведённой к калибровочной плотности.

# Припуск зоны закраса вокруг пальца, px калибровочного кадра (2.9 мм). Отсмотр колец 24/48/96 px
# (2026-10-09): тёмный контур кромки ≤ 5 px от ядра SAM, розовая кайма и блик ≤ 15–20 px, тёплый отсвет кожи на
# бумаге затухает к ~48 px; краска в зоне возвращается с оригинала, так что 40 px текст не стоит.
TIGHT_DILATE_PX = 40
# Наибольшее отношение сторон асимметричной дилатации пальца (как в боевом scan_cropping).
ASYM_RATIO = 2.0
# Зазор защиты глифов, в пределах которой возвращается краска, px масштаба ``finger_protect`` (¼ кадра).
INK_GUARD_PX = 10
# Зазор блоков текста для контекста LaMa (``text_free_context``), px масштаба ``finger_protect`` (¼ кадра).
TEXT_BLOCK_GUARD_PX = 3
# Полоса у ядра пальца, где краска не возвращается (тёмный контур кромки пальца малонасыщен), px кадра.
INK_CORE_BAND_PX = 16
# Возврат краски: окно фона бумаги (закрытие и медиана), px калибровочного кадра (3.0 и 1.5 мм); рампа «насколько
# темнее фона» (уровни серого:
# ниже нижней — не краска, выше верхней — краска целиком); предел насыщенности краски (хрома Lab в 8-битных единицах
# OpenCV): кожа пальца розовая и не возвращается.
INK_BG_CLOSE_PX = 41
INK_BG_MEDIAN_PX = 21
# Сигма размытия доли краски (без ступенек по краю букв), px калибровочного кадра.
INK_ALPHA_SIGMA_PX = 0.8
# Контекст вокруг рамки пальца для LaMa и растушёвка шва вклейки, px калибровочного кадра (4.6 и 0.65 мм) —
# прежние ``removal.FINGER_PADDING`` и ``gpu_models.inpaint(feather=9)``, теперь с пересчётом по плотности кадра.
LAMA_PADDING_PX = 64
LAMA_FEATHER_PX = 9
# Масштаб копии для Telea в ``text_free_context`` на калибровочном кадре (¼); на кадре другой плотности копия
# приводится к той же плотности, иначе радиус Telea (px копии) значил бы разное на бумаге.
TEXT_FREE_SCALE = 0.25
TEXT_FREE_TELEA_RADIUS = 9
INK_DARK_RAMP = (10, 35)
INK_MAX_CHROMA = 16
# Предел длинной стороны рамки дыры в масштабе LaMa, px (``GpuModels.lama_fill_roi``): дыра больше — окно
# ужимается так, чтобы рамка дыры стала ровно этой; меньше — идёт нативно (``research/lama_params``: разметка — 300).
LAMA_HOLE_MAX_PX = 300
# Высветление зоны пальца перед LaMa, уровни яркости (слева, справа) — как в v1
# (``finger_removal.removal.FINGER_ZONE_LIGHT_INCREMENT``, в run-скриптах МТС ``--finger-zone-light-increment=20,20``):
# LaMa заливает дыру на 25–35 уровней темнее окружающей бумаги.
LIGHT_INCREMENT = (20.0, 20.0)


@dataclass(frozen=True)
class FingerResult:
    """Итог закраса кадра.

    Поля:
        clean: Кадр BGR без пальцев.
        paint: Зона закраса (uint8 0/255).
        core: Ядро пальца до дилатации (uint8 0/255).
        protect: Защита глифов, в пределах которой возвращалась краска (uint8 0/255).
        info: Строка для лога.
    """

    clean: np.ndarray
    paint: np.ndarray
    core: np.ndarray
    protect: np.ndarray
    info: str


def text_free_context(
    bgr: np.ndarray, text: np.ndarray, finger: np.ndarray, px_per_mm: float = REF_PX_PER_MM
) -> np.ndarray:
    """Контекст для LaMa: блоки текста заменены тоном соседней бумаги (Telea на ¼ масштаба), всё остальное как есть.

    Палец на тексте LaMa заливает продолжением обрезанных строк, если видит строки рядом с дырой. Здесь строки
    (сгустки глифов, сомкнутые в колонки) убираются из контекста, а линейки, края страниц, торец и коврик остаются —
    их LaMa продолжает под пальцем как прежде.

    Args:
        bgr: Кадр BGR.
        text: Маска блоков текста (bool) размера кадра.
        finger: Зона закраса пальца (bool): для Telea она тоже неизвестна, иначе цвет кожи растечётся в бумагу.
        px_per_mm: Плотность кадра, px/мм: копия для Telea приводится к ¼ калибровочной плотности.

    Returns:
        Кадр BGR без текста.
    """
    h, w = text.shape
    # ¼ на калибровочном кадре; на кадре другой плотности — та же плотность копии (при 13.8 px/мм — ровно w // 4).
    factor = TEXT_FREE_SCALE * (REF_PX_PER_MM / px_per_mm)
    size = (max(1, int(w * factor)), max(1, int(h * factor)))
    small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    hole = cv2.resize((text | finger).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, hole, TEXT_FREE_TELEA_RADIUS, cv2.INPAINT_TELEA)
    filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
    out = bgr.copy()
    out[text] = filled[text]
    return out


def copy_back_ink(
    filled: np.ndarray, original: np.ndarray, region: np.ndarray, px_per_mm: float = REF_PX_PER_MM
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает с оригинала краску (буквы, линейки) в пределах ``region`` поверх закраса LaMa.

    Краска — пиксель заметно темнее локального фона бумаги (серое закрытие окном больше штриха и медиана) и
    малонасыщенный. Кожа пальца, тень пальца и розовая кайма не проходят по насыщенности или по яркости и
    остаются закрашенными. Переход мягкий: доля оригинала растёт по рампе :data:`INK_DARK_RAMP`.

    Args:
        filled: Кадр BGR после LaMa.
        original: Исходный кадр BGR.
        region: Где возвращать (bool): зона закраса ∩ защита глифов.
        px_per_mm: Плотность кадра, px/мм: по ней пересчитываются окна фона и сигма размытия доли.

    Returns:
        ``(кадр BGR, доля оригинала (H, W) float32)``.
    """
    scale = Scale(px_per_mm)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Фон бумаги: закрытие убирает буквы, медиана — остатки штрихов.
    close = scale.frame_odd_px(INK_BG_CLOSE_PX)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    paper = cv2.medianBlur(
        cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE, kernel), scale.frame_odd_px(INK_BG_MEDIAN_PX)
    )
    # Насколько пиксель темнее фона — в долю краски по рампе.
    lo, hi = INK_DARK_RAMP
    dark = np.clip((paper.astype(np.float32) - gray - lo) / (hi - lo), 0, 1)
    # Насыщенное (кожа, розовая кайма) краской не считается.
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    alpha = (dark * (chroma <= INK_MAX_CHROMA) * region).astype(np.float32)
    # Лёгкое размытие доли — без ступенек по краю букв.
    alpha = cv2.GaussianBlur(alpha, (0, 0), INK_ALPHA_SIGMA_PX * scale.ratio)[..., None]
    out = (filled * (1 - alpha) + original * alpha).astype(np.uint8)
    return out, alpha[..., 0]


def remove_fingers_inkback(
    bgr: np.ndarray,
    models,
    dilate_px: int = TIGHT_DILATE_PX,
    hole_max_px: int | None = LAMA_HOLE_MAX_PX,
    guard_px: int = INK_GUARD_PX,
    core_band_px: int = INK_CORE_BAND_PX,
    glyph_scale: float = GLYPH_SCALE,
    light_increment: tuple[float, float] = LIGHT_INCREMENT,
    log_name: str = "",
    px_per_mm: float = REF_PX_PER_MM,
) -> FingerResult:
    """Закрас пальцев кадра в варианте ``inkback`` стенда подшивок.

    1. Маска пальца как в боевом ``remove_fingers`` (YOLO-World → SAM → асимметричная дилатация на ``dilate_px``,
       отсев ядер не у края), без защиты: красится вся раздутая зона.
    2. LaMa по контексту без блоков текста (``text_free_context``): палец на тексте закрашивается бумагой, а не
       продолжением обрезанных строк; ``hole_max_px`` ограничивает рамку дыры в масштабе сети.
    3. В пределах защиты глифов (без полосы у ядра пальца) с оригинала возвращается только краска (``copy_back_ink``).

    Перед LaMa зона пальца высветляется, как в v1 (``brighten_finger_zone``: полный прирост внутри зоны, спад до нуля
    на ``2 × dilate_px`` наружу); итог LaMa вставляется в высветленный кадр, так что кайма спада тоже остаётся
    высветленной и шва на границе зоны нет.

    Args:
        bgr: Кадр BGR в исходном разрешении.
        models: ``GpuModels`` с детекцией и LaMa.
        dilate_px: Припуск зоны закраса вокруг пальца, px кадра.
        hole_max_px: Предел длинной стороны рамки дыры для LaMa, px; None — прежнее ужатие ROI до 512 px.
        guard_px: Зазор защиты глифов, в пределах которого возвращается краска, px ¼ кадра.
        core_band_px: Полоса у ядра пальца, где краска не возвращается, px кадра.
        glyph_scale: Масштаб копии для защиты глифов (``Scale.glyph``).
        light_increment: Высветление зоны пальца перед LaMa, уровни (слева, справа); (0, 0) — без высветления.
        log_name: Имя кадра для логов.
        px_per_mm: Плотность кадра, px/мм: по ней пересчитываются контекст и растушёвка LaMa, окна фона при возврате
            краски и копия Telea без текста (``dilate_px`` и ``core_band_px`` вызывающий пересчитывает сам).

    Returns:
        Кадр без пальцев и маски.
    """
    from ocr_utils.scan_cropping.finger_removal.removal import FINGER_ROI_SCALE, brighten_finger_zone, remove_fingers
    from ocr_utils.scan_cropping.gpu_models import LAMA_ROI_MAX_SIDE

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Маски без закраса: красит этот модуль по своему контексту.
    _, paint, _, _, info, core, _ = remove_fingers(
        bgr,
        models,
        want_boxes=True,
        dilate_px=dilate_px,
        asymmetric_dilation_ratio=ASYM_RATIO,
        light_increment=0,
        log_name=log_name,
        inpaint=False,
    )
    core = np.zeros(paint.shape, np.uint8) if core is None else core
    protect = glyph_protect(rgb, guard_px, glyph_scale)
    if not paint.any():
        return FingerResult(bgr, paint, core, protect, info)
    # Высветление зоны пальца со спадом наружу (прибавка поровну ко всем каналам — порядок BGR не важен).
    bright = brighten_finger_zone(bgr, paint, tuple(light_increment), 2 * dilate_px)
    # LaMa видит кадр без блоков текста; в итог берётся только зона закраса.
    scale = Scale(px_per_mm)
    context = text_free_context(bright, glyph_protect(rgb, TEXT_BLOCK_GUARD_PX, glyph_scale) > 0, paint > 0, px_per_mm)
    filled = models.inpaint(
        cv2.cvtColor(context, cv2.COLOR_BGR2RGB),
        paint,
        padding=scale.frame_px(LAMA_PADDING_PX),
        feather=scale.frame_px(LAMA_FEATHER_PX),
        roi_scale=FINGER_ROI_SCALE,
        roi_max_side=LAMA_ROI_MAX_SIDE,
        hole_max_px=hole_max_px,
    )
    # Вне зоны закраса — высветленный кадр (кайма спада), как в v1.
    clean = bright.copy()
    clean[paint > 0] = cv2.cvtColor(filled, cv2.COLOR_RGB2BGR)[paint > 0]
    # Краска возвращается в защите глифов, но не у самого ядра пальца (тёмный контур кромки).
    k = 2 * core_band_px + 1
    band = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    clean, _ = copy_back_ink(clean, bgr, (paint > 0) & (protect > 0) & ~band, px_per_mm)
    return FingerResult(clean, paint, core, protect, info)
