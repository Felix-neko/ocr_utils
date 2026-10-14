"""Второй проход закраса пальца: LaMa только по «зона пальца ∩ книга» после того, как известны бумага и защита, с ROI вокруг этих кусков (меньше коврика в окне)."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.finger_removal.removal import FINGER_PADDING, FINGER_ROI_SCALE
from ocr_utils.scan_cropping.gpu_models import LAMA_ROI_MAX_SIDE
from ocr_utils.scan_cropping.v2.border_fill import pull_push
from ocr_utils.scan_cropping.v2.fill import fill_non_paper_pull_push
from ocr_utils.scan_cropping.finger_removal.asymmetric_dilation import dilate_zones_elliptical
from ocr_utils.scan_cropping.finger_removal.removal import brighten_finger_zone
from ocr_utils.scan_cropping.v2.fingers import (
    ASYM_RATIO,
    INK_CORE_BAND_PX,
    INK_GUARD_PX,
    LAMA_HOLE_MAX_PX,
    LIGHT_INCREMENT,
    TEXT_BLOCK_GUARD_PX,
    TIGHT_DILATE_PX,
    copy_back_ink,
    text_free_context,
)
from ocr_utils.scan_cropping.v2.glyphs import glyph_protect
from ocr_utils.scan_cropping.v2.pipeline import WorkPaths, read_bgr, write_png
from ocr_utils.scan_cropping.v2.scale import Scale
from research.border_fill.cases import spine_in_box
from research.border_fill.frames import load_masks
from research.finger_repaint.measure import ink_mask

# Расширение «книги» (бумага ∪ защита) перед пересечением с зоной пальца, мм: шов закраса уходит за край листа, и
# край бумаги второго прохода не граничит с закрасом первого прохода.
BOOK_GROW_MM = 1.0
# Тесный ROI варианта C: поле вокруг компоненты дыры, px кадра, и растяжение рамки.
TIGHT_PADDING = 32
TIGHT_ROI_SCALE = 1.0
# Тоновая поправка (E, F): масштаб расчёта; кольцо бумаги вокруг дыры, по которому берётся тон, мм; σ низких частот
# закраса и опоры, мм; рампа поправки от края дыры внутрь, мм (у края LaMa и так сходится с бумагой).
TONE_SCALE = 0.25
TONE_RING_MM = 15.0
TONE_SIGMA_MM = 4.0
TONE_RAMP_MM = 3.0
# Круговая дилатация ядра пальца (H, I): радиус в долях припуска первого прохода. 3 = 1 + ASYM_RATIO — наибольшая
# полуось асимметричного эллипса, т.е. круг накрывает нынешнюю зону целиком.
SYM_RADIUS_RATIO = {
    "H_sym_hole": 1.0 + ASYM_RATIO,
    "I_sym_hole_2x": 2.0,
    "J_sym_paste_all": 1.0 + ASYM_RATIO,
    "K_sym_nolight": 1.0 + ASYM_RATIO,
}
# Рампа вклейки закраса H, I внутрь асимметричной зоны, мм: у края зоны снаружи остаётся кадр (с тенью), закрас к нему
# не привязан — без рампы был бы шов.
PASTE_RAMP_MM = 1.5
# Мелкая тоновая поправка (G): σ и рампа меньше — ловит и край призрака пальца у границы дыры.
TONE_FINE_SIGMA_MM = 2.0
TONE_FINE_RAMP_MM = 1.0


class Variant(str, Enum):
    """Вариант второго прохода (код стенда; A — итог первого прохода как есть)."""

    # B — вариант пользователя: LaMa по «палец ∩ книга», ROI как в первом проходе (поле 64 px, растяжение 1.5).
    REPAINT = "B_repaint"
    # C — то же с тесным ROI (поле 32 px, без растяжения): коврика в окне ещё меньше.
    TIGHT = "C_tight"
    # D — B, но коврик и всё вне книги в контексте LaMa заменены цветом бумаги (pull-push от края): чёрного сеть не видит.
    NO_MAT = "D_no_mat"
    # E — без второго LaMa: низкие частоты закраса первого прохода подтягиваются к тону бумаги вокруг (pull-push по
    # кольцу бумаги), фактура LaMa и возвращённая краска остаются.
    TONE = "E_tone"
    # F — B, затем тоновая поправка E.
    REPAINT_TONE = "F_repaint_tone"
    # G — как E, но σ 2 мм и рампа 1 мм.
    TONE_FINE = "G_tone_fine"
    # H — первый проход заново: LaMa по круговой дилатации ядра радиусом 3× припуска (тень в дыре, не в контексте),
    # вклеивается только нынешняя асимметричная зона с рампой внутрь неё.
    SYM_HOLE = "H_sym_hole"
    # I — как H, радиус 2× припуска.
    SYM_HOLE_2X = "I_sym_hole_2x"
    # J — как H, но закрас вклеивается во всю круговую зону (рампа у её края), буквы возвращаются ``copy_back_ink``.
    SYM_PASTE_ALL = "J_sym_paste_all"
    # K — как J, но без высветления зоны пальца перед LaMa (тень снимается потом картой тени).
    SYM_NO_LIGHT = "K_sym_nolight"


@dataclass(frozen=True)
class RepaintInput:
    """Вход второго прохода по кадру (всё — размера кадра).

    Поля:
        original: Исходный кадр BGR (с пальцами).
        clean: Кадр после первого прохода (``clean.png``), BGR.
        paint: Зона закраса пальца первого прохода (bool).
        core: Ядро пальца (bool).
        book: Бумага ∪ защита контента (bool) — что в итоге не закрашивается до рамки.
        spine_x: Абсцисса стыка страниц, px кадра, или None.
    """

    original: np.ndarray
    clean: np.ndarray
    paint: np.ndarray
    core: np.ndarray
    book: np.ndarray
    spine_x: float | None


def load_input(src: Path, cache: Path) -> RepaintInput:
    """Вход второго прохода из кэша прогона v2 с ``--save-fingers``.

    Args:
        src: Исходный кадр.
        cache: Папка кадра в кэше (``clean.png``, ``fingers.npz``, ``paper.npz``, ``pages.json``).

    Returns:
        Вход.
    """
    work = WorkPaths(cache)
    clean = read_bgr(work.clean)
    with np.load(work.fingers) as data:
        paint, core = data["paint"], data["core"]
    masks = load_masks(cache)
    with np.load(cache / "paper.npz") as data:
        seg_h = data["paper"].shape[0]
    spine = spine_in_box(cache, clean.shape[0] / seg_h, 0)
    return RepaintInput(read_bgr(src), clean, paint, core, masks.paper | masks.protected, spine)


def repaint_hole(inp: RepaintInput, scale: Scale) -> np.ndarray:
    """Дыра второго прохода: зона пальца ∩ книга, расширенная на ``BOOK_GROW_MM`` (bool)."""
    k = 2 * max(1, int(round(BOOK_GROW_MM * scale.px_per_mm))) + 1
    grown = cv2.dilate(inp.book.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    return inp.paint & grown


def repaint(inp: RepaintInput, variant: Variant, models, scale: Scale) -> tuple[np.ndarray, float]:
    """Второй проход закраса пальца вариантом стенда.

    Контекст LaMa — кадр первого прохода (уже высветленный в зоне пальца) без блоков текста (``text_free_context``);
    дыра — ``repaint_hole``; затем, как в первом проходе, с оригинала возвращается краска в защите глифов без полосы у
    ядра пальца (``copy_back_ink``). Вне дыры кадр не меняется.

    Args:
        inp: Вход.
        variant: Вариант.
        models: ``GpuModels`` с LaMa.
        scale: Масштабы прогона.

    Returns:
        ``(кадр BGR после второго прохода, время LaMa, с)``.
    """
    if variant in (Variant.SYM_HOLE, Variant.SYM_HOLE_2X, Variant.SYM_PASTE_ALL, Variant.SYM_NO_LIGHT):
        paste_all = variant in (Variant.SYM_PASTE_ALL, Variant.SYM_NO_LIGHT)
        light = (0.0, 0.0) if variant == Variant.SYM_NO_LIGHT else LIGHT_INCREMENT
        return symmetric_first_pass(inp, SYM_RADIUS_RATIO[variant.value], models, scale, paste_all, light)
    hole = repaint_hole(inp, scale)
    if not hole.any():
        return inp.clean, 0.0
    if variant == Variant.TONE:
        return tone_correct(inp.clean, hole, inp.book, inp.paint, scale), 0.0
    if variant == Variant.TONE_FINE:
        return tone_correct(inp.clean, hole, inp.book, inp.paint, scale, TONE_FINE_SIGMA_MM, TONE_FINE_RAMP_MM), 0.0
    if variant == Variant.REPAINT_TONE:
        out, seconds = repaint(inp, Variant.REPAINT, models, scale)
        return tone_correct(out, hole, inp.book, inp.paint, scale), seconds
    rgb = cv2.cvtColor(inp.original, cv2.COLOR_BGR2RGB)
    base = inp.clean
    if variant == Variant.NO_MAT:
        # Вне книги — цвет бумаги от края (как закрас до прямоугольника), чёрного коврика в окне LaMa нет.
        base = fill_non_paper_pull_push(inp.clean, inp.book, scale.px_per_mm, inp.spine_x)
    text = glyph_protect(rgb, TEXT_BLOCK_GUARD_PX, scale.glyph) > 0
    context = text_free_context(base, text, hole)
    padding, roi_scale = (
        (TIGHT_PADDING, TIGHT_ROI_SCALE) if variant == Variant.TIGHT else (FINGER_PADDING, FINGER_ROI_SCALE)
    )
    t0 = time.perf_counter()
    filled = models.inpaint(
        cv2.cvtColor(context, cv2.COLOR_BGR2RGB),
        hole.astype(np.uint8) * 255,
        padding=padding,
        roi_scale=roi_scale,
        roi_max_side=LAMA_ROI_MAX_SIDE,
        hole_max_px=LAMA_HOLE_MAX_PX,
    )
    seconds = time.perf_counter() - t0
    out = inp.clean.copy()
    out[hole] = cv2.cvtColor(filled, cv2.COLOR_RGB2BGR)[hole]
    # Краска с оригинала — в защите глифов, но не у самого ядра пальца (как ``remove_fingers_inkback``).
    protect = glyph_protect(rgb, INK_GUARD_PX, scale.glyph) > 0
    k = 2 * scale.frame_px(INK_CORE_BAND_PX) + 1
    band = cv2.dilate(inp.core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    out, _ = copy_back_ink(out, inp.original, hole & protect & ~band)
    return out, seconds


def symmetric_hole(inp: RepaintInput, radius_ratio: float, scale: Scale) -> np.ndarray:
    """Дыра H, I: ядро пальца, раздутое кругом радиусом ``radius_ratio`` × припуск первого прохода, ∪ нынешняя зона.

    Берутся только компоненты, задевающие нынешнюю зону закраса (ядра, отсеянные первым проходом, не красятся).

    Args:
        inp: Вход кадра (``core``, ``paint``).
        radius_ratio: Радиус круга в долях припуска ``TIGHT_DILATE_PX`` (в px кадра).
        scale: Масштабы прогона.

    Returns:
        Дыра (bool) размера кадра.
    """
    radius = max(1, int(round(scale.frame_px(TIGHT_DILATE_PX) * radius_ratio)))
    big = dilate_zones_elliptical(inp.core.astype(np.uint8) * 255, lambda _bbox: (radius, radius)) > 0
    num, labels = cv2.connectedComponents((big | inp.paint).astype(np.uint8), connectivity=8)
    keep = np.unique(labels[inp.paint])
    return np.isin(labels, keep[keep > 0])


def symmetric_first_pass(
    inp: RepaintInput,
    radius_ratio: float,
    models,
    scale: Scale,
    paste_all: bool = False,
    light: tuple[float, float] = LIGHT_INCREMENT,
) -> tuple[np.ndarray, float]:
    """Первый проход заново (H, I): LaMa по круговой дыре ``symmetric_hole``, вклейка только в асимметричную зону.

    Повторяет ``fingers.remove_fingers_inkback`` (высветление по нынешней зоне, контекст без блоков текста, LaMa,
    возврат краски), меняется только дыра LaMa: тень пальца у края нынешней зоны теперь внутри дыры, а не в
    контексте. В кадр закрас кладётся в пределах нынешней зоны ``paint`` с рампой ``PASTE_RAMP_MM`` внутрь от её
    края; вне ``paint`` кадр остаётся высветленным оригиналом, как в первом проходе.

    Args:
        inp: Вход кадра (оригинал, ``paint``, ``core``).
        radius_ratio: Радиус круговой дилатации в долях припуска.
        models: ``GpuModels`` с LaMa.
        scale: Масштабы прогона.
        paste_all: Вклеивать закрас во всю круговую дыру (J), а не только в ``paint``; краска с оригинала тогда
            возвращается по всей дыре.
        light: Высветление зоны пальца перед LaMa, уровни (слева, справа); ``(0, 0)`` — без высветления (K).

    Returns:
        ``(кадр BGR — замена clean.png, время LaMa, с)``.
    """
    rgb = cv2.cvtColor(inp.original, cv2.COLOR_BGR2RGB)
    paint_u8 = inp.paint.astype(np.uint8) * 255
    dilate_px = scale.frame_px(TIGHT_DILATE_PX)
    # Высветление — по нынешней зоне, как в первом проходе: вне неё в итог идёт именно этот кадр.
    bright = brighten_finger_zone(inp.original, paint_u8, light, 2 * dilate_px)
    hole = symmetric_hole(inp, radius_ratio, scale)
    context = text_free_context(bright, glyph_protect(rgb, TEXT_BLOCK_GUARD_PX, scale.glyph) > 0, hole)
    t0 = time.perf_counter()
    filled = models.inpaint(
        cv2.cvtColor(context, cv2.COLOR_BGR2RGB),
        hole.astype(np.uint8) * 255,
        padding=FINGER_PADDING,
        roi_scale=FINGER_ROI_SCALE,
        roi_max_side=LAMA_ROI_MAX_SIDE,
        hole_max_px=LAMA_HOLE_MAX_PX,
    )
    seconds = time.perf_counter() - t0
    # Рампа внутрь вклеиваемой зоны: 0 на её краю, 1 на глубине PASTE_RAMP_MM.
    ramp_px = max(1.0, PASTE_RAMP_MM * scale.px_per_mm)
    paste = hole if paste_all else inp.paint
    alpha = np.clip(cv2.distanceTransform(paste.astype(np.uint8), cv2.DIST_L2, 5) / ramp_px, 0, 1)[..., None]
    filled_bgr = cv2.cvtColor(filled, cv2.COLOR_RGB2BGR).astype(np.float32)
    out = np.clip(alpha * filled_bgr + (1 - alpha) * bright.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
    # Краска с оригинала — как в первом проходе: защита глифов без полосы у ядра.
    protect = glyph_protect(rgb, INK_GUARD_PX, scale.glyph) > 0
    k = 2 * scale.frame_px(INK_CORE_BAND_PX) + 1
    band = cv2.dilate(inp.core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    out, _ = copy_back_ink(out, inp.original, paste & protect & ~band)
    return out, seconds


def normalized_blur(img: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Нормированная свёртка: гаусс от ``w·I``, делённый на гаусс от ``w`` (пиксели с весом 0 в среднее не входят).

    Args:
        img: Картинка float32 (H, W, C).
        weight: Вес float32 (H, W).
        sigma: σ гаусса, px.

    Returns:
        Картинка float32 (H, W, C).
    """
    num = cv2.GaussianBlur(img * weight[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)[..., None]


def tone_correct(
    clean: np.ndarray,
    hole: np.ndarray,
    book: np.ndarray,
    paint: np.ndarray,
    scale: Scale,
    sigma_mm: float = TONE_SIGMA_MM,
    ramp_mm: float = TONE_RAMP_MM,
) -> np.ndarray:
    """Тоновая поправка закраса пальца: низкие частоты внутри дыры — к тону бумаги вокруг, фактура закраса остаётся.

    На ¼ копии: опора — pull-push по кольцу бумаги ``TONE_RING_MM`` вокруг дыры (без зоны пальца и без краски);
    низкие частоты закраса — нормированная свёртка дыры без краски (σ ``TONE_SIGMA_MM``). Поправка «опора − закрас»
    нарастает от края дыры на ``TONE_RAMP_MM`` и прибавляется к кадру внутри дыры в полном масштабе.

    Args:
        clean: Кадр после закраса пальца, BGR uint8.
        hole: Дыра (bool) размера кадра.
        book: Бумага ∪ защита (bool).
        paint: Зона закраса пальца (bool).
        scale: Масштабы прогона.
        sigma_mm: σ низких частот закраса и опоры, мм.
        ramp_mm: Рампа поправки от края дыры внутрь, мм.

    Returns:
        Кадр BGR uint8.
    """
    h, w = hole.shape
    small = cv2.resize(clean, None, fx=TONE_SCALE, fy=TONE_SCALE, interpolation=cv2.INTER_AREA)
    size = (small.shape[1], small.shape[0])
    mm = scale.px_per_mm * TONE_SCALE
    hole_s = cv2.resize(hole.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    book_s = cv2.resize(book.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.99
    paint_s = cv2.resize(paint.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    ink = ink_mask(small, mm)
    img = small.astype(np.float32)
    # Кольцо бумаги вокруг дыры: опора тона.
    outside = cv2.distanceTransform((~hole_s).astype(np.uint8), cv2.DIST_L2, 5)
    ring = book_s & ~paint_s & ~ink & (outside <= TONE_RING_MM * mm)
    if ring.sum() < 50:
        return clean
    ref = cv2.GaussianBlur(pull_push(img, ring.astype(np.float32)), (0, 0), sigma_mm * mm)
    low = normalized_blur(img, (hole_s & ~ink).astype(np.float32), sigma_mm * mm)
    # Рампа от края дыры внутрь.
    inside = cv2.distanceTransform(hole_s.astype(np.uint8), cv2.DIST_L2, 5)
    ramp = np.clip(inside / (ramp_mm * mm), 0, 1)[..., None]
    corr = (ref - low) * ramp * hole_s[..., None]
    big = cv2.resize(corr, (w, h), interpolation=cv2.INTER_CUBIC)
    out = clean.astype(np.float32)
    out[hole] += big[hole]
    return np.clip(out, 0, 255).astype(np.uint8)


def stage_cache(base_cache: Path, variant_cache: Path, clean: np.ndarray) -> None:
    """Кэш кадра варианта: копия кэша базы с подменённым ``clean.png`` (финал v2 возьмёт кадр второго прохода).

    Args:
        base_cache: Папка кадра в кэше базового прогона.
        variant_cache: Папка кадра в кэше варианта (перезаписывается).
        clean: Кадр после второго прохода.
    """
    if variant_cache.exists():
        shutil.rmtree(variant_cache)
    shutil.copytree(base_cache, variant_cache, ignore=shutil.ignore_patterns("clean.png"))
    write_png(WorkPaths(variant_cache).clean, clean)
