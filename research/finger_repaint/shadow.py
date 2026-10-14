"""Карта тени пальца на бумаге и осветление итога по ней: область поиска вокруг пальца в книге, оценщики «бумага без тени» и «бумага как есть», рампа от краёв, поправка коэффициентом или прибавкой, меры остатка тени."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.border_fill import pull_push
from research.border_fill.frames import key_of, load_masks, read_image
from research.finger_repaint.measure import ink_mask

# Масштаб, на котором оценивается карта (как мера пятна); поправка затем растягивается до полного итога.
SHADOW_SCALE = 0.25
# Досягаемость тени от ядра пальца вглубь книги, мм (радиус круга области поиска). Подобрано по профилю тени
# (команда ``reach``, 28 кадров dark_spots): провал L бумаги вне зоны закраса 1–1.6 медиана (90 % — до 3.5) на 9–20 мм
# от ядра, к ~30 мм сходит на нет.
SHADOW_REACH_MM = 25.0
# Кольцо бумаги снаружи области, по которому берётся опора «бумага без тени», мм от края области.
REF_RING_MM = 10.0
# Кожа — отход цвета от окрестной бумаги в красное: Δa* Lab (8-битные единицы OpenCV) больше SKIN_DA, направление
# сдвига — в SKIN_ANGLE. Пак-6: у бумаги Δa* ≤ ~4 (99-й перцентиль), у кожи 5-й перцентиль a* 4–9, медиана 10–16.
SKIN_DA = 5.0
# Направление сдвига цвета от бумаги в плоскости (Δa*, Δb*), градусы: кожа — в красное, с жёлтым или без
# (на белой бумаге ~55°, на жёлтой ПХ ~10°); фиолетовая и синяя печать — ниже −20°, пожелтение — выше 70°.
SKIN_ANGLE = (-20.0, 70.0)
# Кожа цветная и тёплая сама по себе: a* больше SKIN_MIN_A и хрома больше SKIN_MIN_CHROMA (у кожи пака-6 a* 4–18, хрома
# 10–30; чёрная и серая печать — хрома < 5).
SKIN_MIN_A = 3.0
SKIN_MIN_CHROMA = 8.0
# Отсев не-кожи: открытие окном (мм) и наименьшее пятно кожи (мм²).
SKIN_OPEN_MM = 1.0
SKIN_MIN_MM2 = 20.0
# Расширение маски кожи, мм: кайма пальца темнее и чуть менее красная.
SKIN_GROW_MM = 1.0
# Крупная печать, которую маска краски не ловит: темнее медианы бумаги кольца больше чем на CONTENT_DROP_L (L 0..100)
# — тень пальца по замерам стенда до ~20 L; расширение такой маски, мм.
CONTENT_DROP_L = 25.0
CONTENT_GROW_MM = 1.0
# Сглаживание «бумаги как есть» (B) и опоры (R), мм: тень плавная, буквы и фактура — мельче.
SMOOTH_SIGMA_MM = 1.5
# Блок робастного фона (оценщик Q), мм, и перцентиль яркости бумаги в блоке.
BLOCK_MM = 4.0
BLOCK_PERCENTILE = 80
# Наименьшая доля «бумаги без краски» в блоке Q, чтобы блок считался.
BLOCK_MIN_FRACTION = 0.3
# Рампа карты от края области (там, где снаружи — бумага), мм: на краю поправка 0 и растёт плавно.
EDGE_RAMP_MM = 4.0
# Итоговое сглаживание логарифма коэффициента внутри области, мм (снимает всплески).
POST_SIGMA_MM = 1.0
# Наибольший коэффициент осветления (защита от выбросов опоры).
GAIN_MAX = 1.6
# Насколько поправка продолжается в закрас вне книги (там закрас продолжил тёмный край листа), мм, с рампой до 0.
FILL_REACH_MM = 10.0
# Перцентиль, которым карта ограничивается сверху внутри области (отсечка одиночных выбросов).
CAP_PERCENTILE = 99.5
# Мера остатка: кольцо опоры снаружи области (мм), провал/пересвет L (0..100), наименьшее пятно, мм².
MEASURE_RING_MM = (3.0, 10.0)
MEASURE_DROP_L = 8.0
MEASURE_MIN_MM2 = 4.0


class ShadowMethod(str, Enum):
    """Оценщик опоры «бумага без тени» R и фона «бумага как есть» B (коды стенда)."""

    # T1 — вариант пользователя: Telea по области поиска (с краской и не-бумагой), затем сглаживание; B — нормированное.
    TELEA = "T1_telea"
    # T2 — то же, LaMa вместо Telea (GPU).
    LAMA = "T2_lama"
    # P — pull-push по кольцу бумаги снаружи области; B — нормированное сглаживание.
    PULL_PUSH = "P_pull_push"
    # Q — опора как P; B — робастный фон блоками (верхний перцентиль бумаги без краски, по Bako et al. 2016).
    ROBUST = "Q_robust"
    # S — опора — гладкая поверхность 2-й степени по кольцу (Хубер), по компонентам области; B — как Q.
    SURFACE = "S_surface"


class LightenMode(str, Enum):
    """Как осветлять: коэффициентом по каналам (поправка освещения) или прибавкой."""

    GAIN = "gain"
    ADD = "add"


@dataclass(frozen=True)
class ShadowInputs:
    """Итог кадра и маски в его координатах (полный масштаб).

    Поля:
        key: Ключ кадра.
        img: Итог BGR uint8.
        paint: Зона закраса пальца (bool).
        core: Ядро пальца (bool).
        book: Бумага после доводки ∪ защита контента (bool) — книга в итоге.
        original: Где в итоге пиксели оригинала (bool): вне зоны закраса и в copy back. Только там может быть кожа:
            внутри закраса LaMa кожи нет, а розово-бурое пятно LaMa похоже на неё цветом.
    """

    key: str
    img: np.ndarray
    paint: np.ndarray
    core: np.ndarray
    book: np.ndarray
    original: np.ndarray


@dataclass(frozen=True)
class ShadowMap:
    """Карта тени на масштабе ``SHADOW_SCALE``.

    Поля:
        area: Область поиска тени (bool).
        book: Книга (bool).
        skin: Кожа в итоге (bool), в оценку и осветление не входит.
        ink: Краска (bool).
        content: Крупная печать — темнее бумаги кольца больше чем на ``CONTENT_DROP_L`` (bool).
        log_gain: Логарифм коэффициента осветления по каналам BGR, float32 (H, W, 3); 0 — без поправки.
        delta: Прибавка по каналам BGR (R − B), float32 (H, W, 3), уже с рампой.
        seconds: Время оценки, с.
    """

    area: np.ndarray
    book: np.ndarray
    skin: np.ndarray
    ink: np.ndarray
    content: np.ndarray
    log_gain: np.ndarray
    delta: np.ndarray
    seconds: float


def load_shadow_inputs(path: Path, output_root: Path, cache_root: Path) -> ShadowInputs:
    """Итог кадра прогона v2 и маски пальца и книги в его координатах.

    Args:
        path: Итоговый TIFF.
        output_root: Корень итогов прогона (для ключа).
        cache_root: Корень кэша (``fingers.npz``, ``paper.npz``).

    Returns:
        Вход.

    Raises:
        ValueError: Бумаги нет или размер итога не совпал с рамкой.
    """
    from research.finger_repaint.cli import copy_back_region

    key = key_of(path.relative_to(output_root))
    masks = load_masks(cache_root / key)
    if masks.box is None:
        raise ValueError("нет бумаги")
    with np.load(cache_root / key / "fingers.npz") as data:
        paint, core = data["paint"], data["core"]
    img = read_image(path)
    book = masks.crop(masks.paper | masks.protected)
    if img.shape[:2] != book.shape:
        raise ValueError(f"размер {img.shape[:2]} ≠ рамке {book.shape}")
    original = ~paint | copy_back_region(cache_root / key, paint)
    return ShadowInputs(key, img, masks.crop(paint), masks.crop(core), book, masks.crop(original))


def disk(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Маска, раздутая кругом радиуса ``radius_px`` (через расстояние, быстро на больших радиусах)."""
    if not mask.any():
        return mask.copy()
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) <= radius_px


def search_area(paint: np.ndarray, core: np.ndarray, book: np.ndarray, mm: float, reach_mm: float) -> np.ndarray:
    """Область поиска тени: ядро пальца, раздутое кругом ``reach_mm``, ∪ зона закраса, ∩ книга.

    Берутся только компоненты, задевающие зону закраса (ядра, отсеянные первым проходом, не в счёт).

    Args:
        paint: Зона закраса (bool).
        core: Ядро пальца (bool).
        book: Книга (bool).
        mm: Плотность, px/мм.
        reach_mm: Радиус круга, мм.

    Returns:
        Область (bool).
    """
    grown = disk(core, reach_mm * mm) | paint
    n, labels = cv2.connectedComponents(grown.astype(np.uint8), connectivity=8)
    keep = np.unique(labels[paint])
    return np.isin(labels, keep[keep > 0]) & book


def skin_mask(
    bgr: np.ndarray, mm: float, book: np.ndarray, near: np.ndarray | None = None, allowed: np.ndarray | None = None
) -> np.ndarray:
    """Кожа пальца в итоге — крупное сплошное пятно, отошедшее от цвета окрестной бумаги в красное.

    Опора — медиана a*, b* светлой бумаги (ярче медианы L, без краски) в ``book ∩ near``: местная, потому что
    страницы разворота бывают разного оттенка («Экономист» — белая и розоватая). Кожа: ``Δa* > SKIN_DA`` и
    направление сдвига ``atan2(Δb*, Δa*)`` в ``SKIN_ANGLE`` (в красное; фиолетовая печать уходит вниз по b*) — пожелтевшая или порыжевшая бумага (ПХ, b* до +18) сдвинута по b* и сама с собой даёт
    Δ ≈ 0, тень на ней уходит к серому. Затем открытие окном ``SKIN_OPEN_MM`` и пятна от ``SKIN_MIN_MM2``: цветные
    каймы букв, штампы и линии у стыка страниц тонкие и мелкие, палец — нет. Расширено на ``SKIN_GROW_MM``.

    Args:
        bgr: Картинка BGR uint8.
        mm: Плотность, px/мм.
        book: Книга (bool).
        near: Где брать опору бумаги (bool) — окрестность пальца; None — вся книга.
        allowed: Где кожа возможна (bool) — пиксели оригинала в итоге; None — везде.

    Returns:
        Маска кожи (bool).
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    light, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    sel = book & ~ink_mask(bgr, mm)
    if near is not None and (sel & near).sum() >= 100:
        sel = sel & near
    if sel.sum() < 100:
        return np.zeros(book.shape, bool)
    bright = sel & (light >= np.median(light[sel]))
    da, db = a - np.median(a[bright]), b - np.median(b[bright])
    angle = np.degrees(np.arctan2(db, da))
    skin = (da > SKIN_DA) & (angle >= SKIN_ANGLE[0]) & (angle <= SKIN_ANGLE[1])
    skin &= (a > SKIN_MIN_A) & (np.hypot(a, b) > SKIN_MIN_CHROMA)
    if allowed is not None:
        skin &= allowed
    skin = skin.astype(np.uint8)
    k = max(3, int(round(SKIN_OPEN_MM * mm)) | 1)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, comp, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    small = np.nonzero(stats[:, cv2.CC_STAT_AREA] < SKIN_MIN_MM2 * mm * mm)[0]
    skin = (skin > 0) & ~np.isin(comp, small)
    return disk(skin, SKIN_GROW_MM * mm)


def smooth_fill(img: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Плавная поверхность по пикселям с весом: pull-push (дыры заполняются с соседей), затем гаусс ``sigma`` px."""
    return cv2.GaussianBlur(pull_push(img, weight.astype(np.float32)), (0, 0), sigma)


def robust_background(img: np.ndarray, weight: np.ndarray, mm: float) -> np.ndarray:
    """Фон бумаги блоками (оценщик Q, по Bako et al. 2016): в блоке ``BLOCK_MM`` — средний цвет самых светлых пикселей.

    В блоке берутся пиксели с весом (бумага без краски и кожи) ярче ``BLOCK_PERCENTILE``-го перцентиля серого, их
    средний цвет — фон блока. Краска, которую маска не поймала, в верхний перцентиль не попадает. Блоки с долей
    бумаги меньше ``BLOCK_MIN_FRACTION`` пропускаются, сетка блоков заполняется pull-push и растягивается.

    Args:
        img: Картинка float32 BGR (¼ масштаба).
        weight: Вес бумаги (bool).
        mm: Плотность, px/мм.

    Returns:
        Фон float32 BGR того же размера.
    """
    b = max(4, int(round(BLOCK_MM * mm)))
    h, w = weight.shape
    gh, gw = -(-h // b), -(-w // b)
    grid = np.zeros((gh, gw, 3), np.float32)
    known = np.zeros((gh, gw), np.float32)
    gray = img.mean(axis=2)
    ys, xs = np.nonzero(weight)
    if ys.size == 0:
        return img.copy()
    by, bx = ys // b, xs // b
    order = np.lexsort((bx, by))
    ys, xs, by, bx = ys[order], xs[order], by[order], bx[order]
    cell = by * gw + bx
    starts = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
    ends = np.r_[starts[1:], cell.size]
    for s, e in zip(starts, ends):
        if e - s < BLOCK_MIN_FRACTION * b * b:
            continue
        py, px = ys[s:e], xs[s:e]
        g = gray[py, px]
        top = g >= np.percentile(g, BLOCK_PERCENTILE)
        grid[by[s], bx[s]] = img[py[top], px[top]].mean(axis=0)
        known[by[s], bx[s]] = 1
    grid = pull_push(grid, known)
    big = cv2.resize(grid, (gw * b, gh * b), interpolation=cv2.INTER_LINEAR)[:h, :w]
    return cv2.GaussianBlur(big, (0, 0), SMOOTH_SIGMA_MM * mm)


def surface_reference(img: np.ndarray, ring: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Опора S: по каждой компоненте области — поверхность 2-й степени по кольцу бумаги (Хубер, 5 итераций IRLS).

    Args:
        img: Картинка float32 BGR.
        ring: Кольцо опоры (bool).
        area: Область поиска (bool).

    Returns:
        Опора float32 BGR (вне области — исходная картинка).
    """
    out = img.copy()
    n, labels = cv2.connectedComponents(area.astype(np.uint8), connectivity=8)
    dist, (iy, ix) = _nearest(area)
    owner = labels[iy, ix]
    for k in range(1, n):
        own = ring & (owner == k)
        ys, xs = np.nonzero(own)
        if ys.size < 30:
            continue
        step = max(1, ys.size // 4000)
        ys, xs = ys[::step], xs[::step]
        cy, cx, sc = ys.mean(), xs.mean(), max(1.0, float(np.ptp(ys) + np.ptp(xs)) / 2)
        design = _quad((ys - cy) / sc, (xs - cx) / sc)
        py, px = np.nonzero(labels == k)
        target = _quad((py - cy) / sc, (px - cx) / sc)
        for c in range(3):
            v = img[ys, xs, c]
            wts = np.ones_like(v)
            for _ in range(5):
                coef = np.linalg.lstsq(design * wts[:, None], v * wts, rcond=None)[0]
                res = np.abs(v - design @ coef)
                scale = max(1.0, 1.4826 * float(np.median(res)))
                wts = np.sqrt(np.minimum(1.0, 1.345 * scale / np.maximum(res, 1e-6)))
            out[py, px, c] = target @ coef
    return out


def _quad(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Матрица полинома 2-й степени по (y, x): 1, y, x, y², xy, x²."""
    return np.stack([np.ones_like(y), y, x, y * y, x * y, x * x], axis=1).astype(np.float64)


def _nearest(mask: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Расстояние до маски и индексы ближайшего её пикселя (scipy EDT)."""
    from scipy import ndimage

    return ndimage.distance_transform_edt(~mask, return_indices=True)


def inpaint_reference(small: np.ndarray, hole: np.ndarray, method: ShadowMethod, models, mm: float) -> np.ndarray:
    """Опора T1/T2 (вариант пользователя): закрас области поиска вместе с краской и не-бумагой, затем сглаживание.

    Args:
        small: Итог BGR uint8 на ¼.
        hole: Что закрасить (bool): область ∪ краска ∪ кожа ∪ не-книга.
        method: ``TELEA`` или ``LAMA``.
        models: ``GpuModels`` с LaMa (для ``LAMA``) или None.
        mm: Плотность, px/мм.

    Returns:
        Опора float32 BGR.
    """
    mask = hole.astype(np.uint8) * 255
    if method == ShadowMethod.TELEA:
        filled = cv2.inpaint(small, mask, 9, cv2.INPAINT_TELEA)
    else:
        rgb = models.inpaint(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), mask, padding=64, roi_scale=1.5, hole_max_px=300)
        filled = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.GaussianBlur(filled.astype(np.float32), (0, 0), SMOOTH_SIGMA_MM * mm)


def estimate(inp: ShadowInputs, method: ShadowMethod, px_per_mm: float, models=None) -> ShadowMap:
    """Карта тени кадра оценщиком ``method`` с общей постобработкой (рампа, сглаживание, отсечка).

    Шаги (на ¼): область поиска; краска, кожа; B — «бумага как есть» по бумаге области без краски и кожи; R — «бумага
    без тени» по кольцу бумаги снаружи области (или закрасом области для T1/T2); логарифм коэффициента ``log(R/B)`` и
    прибавка ``R − B`` не меньше 0; сглаживание по области, отсечка сверху, рампа от края области, где снаружи бумага.

    Args:
        inp: Итог и маски.
        method: Оценщик.
        px_per_mm: Плотность итога, px/мм.
        models: ``GpuModels`` с LaMa — только для ``LAMA``.

    Returns:
        Карта.
    """
    t0 = time.perf_counter()
    s = SHADOW_SCALE
    small = cv2.resize(inp.img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    size = (small.shape[1], small.shape[0])
    mm = px_per_mm * s
    nearest = lambda m: cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0  # noqa: E731
    paint, core, original = nearest(inp.paint), nearest(inp.core), nearest(inp.original)
    book = cv2.resize(inp.book.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.99
    area = search_area(paint, core, book, mm, SHADOW_REACH_MM)
    ink = ink_mask(small, mm)
    skin = skin_mask(small, mm, book, disk(area, REF_RING_MM * mm) & ~area, original)
    img = small.astype(np.float32)
    zeros = np.zeros_like(img)
    if not area.any():
        return ShadowMap(area, book, skin, ink, np.zeros_like(area), zeros, zeros, time.perf_counter() - t0)
    clean = book & ~ink & ~skin
    outside = cv2.distanceTransform((~area).astype(np.uint8), cv2.DIST_L2, 5)
    ring = clean & ~area & ~paint & (outside <= REF_RING_MM * mm)
    # Крупная печать (логотипы, заливки, штрихи толще окна маски краски): темнее бумаги кольца больше, чем бывает тень.
    light = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * (100 / 255)
    content = np.zeros_like(area)
    if ring.sum() >= 50:
        # Только пиксели оригинала: в закрасе LaMa печати нет (буквы возвращает copy_back_ink — их ловит маска краски),
        # а глубокое пятно LaMa темнее порога — его как раз и надо осветлить.
        content = (light < np.median(light[ring]) - CONTENT_DROP_L) & original
        content = disk(content, CONTENT_GROW_MM * mm)
        clean &= ~content
        ring &= ~content
    sigma = SMOOTH_SIGMA_MM * mm
    # Рабочее окно — рамка области с запасом кольца (быстрее, чем весь кадр).
    y0, y1, x0, x1 = _window(area, int((REF_RING_MM + 3 * SMOOTH_SIGMA_MM) * mm) + 2)
    win = (slice(y0, y1), slice(x0, x1))
    w_img, w_area, w_ring, w_clean = img[win], area[win], ring[win], clean[win]
    # B — бумага как есть.
    weight_b = w_area & w_clean
    if method in (ShadowMethod.ROBUST, ShadowMethod.SURFACE):
        back = robust_background(w_img, weight_b | w_ring, mm)
    else:
        back = smooth_fill(w_img, weight_b | w_ring, sigma)
    # R — бумага без тени.
    if method in (ShadowMethod.TELEA, ShadowMethod.LAMA):
        hole = w_area | ~w_clean
        ref = inpaint_reference(small[win], hole, method, models, mm)
    elif method == ShadowMethod.SURFACE:
        ref = cv2.GaussianBlur(surface_reference(w_img, w_ring, w_area), (0, 0), sigma)
    else:
        ref = smooth_fill(w_img, w_ring, sigma)
    log_gain = np.log(np.maximum(ref, 1.0) / np.maximum(back, 1.0))
    delta = ref - back
    log_gain, delta = (_post(x, w_area, book[win], mm) for x in (np.maximum(log_gain, 0), np.maximum(delta, 0)))
    full_gain, full_delta = zeros.copy(), zeros.copy()
    full_gain[win], full_delta[win] = log_gain, delta
    return ShadowMap(area, book, skin, ink, content, full_gain, full_delta, time.perf_counter() - t0)


def _window(mask: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    """Рамка маски ``(y0, y1, x0, x1)`` с запасом ``margin``, обрезанная по картинке."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return (
        max(0, ys.min() - margin),
        min(h, ys.max() + margin + 1),
        max(0, xs.min() - margin),
        min(w, xs.max() + margin + 1),
    )


def _post(value: np.ndarray, area: np.ndarray, book: np.ndarray, mm: float) -> np.ndarray:
    """Общая постобработка карты: отсечка сверху, сглаживание по области, рампа от края, продолжение в закрас вне книги.

    Args:
        value: Карта (логарифм коэффициента или прибавка) по каналам, ≥ 0.
        area: Область поиска (bool).
        book: Книга (bool).
        mm: Плотность, px/мм.

    Returns:
        Карта с нулями вне области и вне продолжения в закрас.
    """
    inside = value[area]
    cap = np.percentile(inside, CAP_PERCENTILE, axis=0) if inside.size else np.zeros(3)
    value = np.minimum(value, cap)
    # Сглаживание только по области (нормированная свёртка): снаружи нули в среднее не входят.
    wts = area.astype(np.float32)
    num = cv2.GaussianBlur(value * wts[..., None], (0, 0), POST_SIGMA_MM * mm)
    den = cv2.GaussianBlur(wts, (0, 0), POST_SIGMA_MM * mm)
    value = num / np.maximum(den, 1e-6)[..., None]
    # Рампа: 0 там, где снаружи области бумага (край тени), растёт smoothstep на EDGE_RAMP_MM. У края листа (снаружи
    # не книга) рампы нет: там поправка продолжается в закрас вне книги.
    paper_out = book & ~area
    d = cv2.distanceTransform((~paper_out).astype(np.uint8), cv2.DIST_L2, 5) / (EDGE_RAMP_MM * mm)
    ramp = _smoothstep(np.clip(d, 0, 1))
    value = value * ramp[..., None] * area[..., None]
    # Продолжение в закрас вне книги: закрас до рамки продолжил тёмный край листа — та же поправка, со спадом.
    fill = ~book
    if fill.any() and area.any():
        dist, (iy, ix) = _nearest(area)
        near = fill & (dist <= FILL_REACH_MM * mm)
        fade = _smoothstep(np.clip(1 - dist / (FILL_REACH_MM * mm), 0, 1))
        value[near] = value[iy[near], ix[near]] * fade[near][:, None]
    return value


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Гладкая ступенька 3t² − 2t³ (нулевая производная на концах)."""
    return t * t * (3 - 2 * t)


def lighten(img: np.ndarray, smap: ShadowMap, mode: LightenMode) -> np.ndarray:
    """Осветление итога по карте: коэффициент ``exp(log_gain)`` или прибавка ``delta``, кожа не трогается.

    Args:
        img: Итог BGR uint8 (полный масштаб).
        smap: Карта на ¼.
        mode: Коэффициент или прибавка.

    Returns:
        Осветлённый итог BGR uint8.
    """
    h, w = img.shape[:2]
    keep = (~smap.skin).astype(np.float32)
    if mode == LightenMode.GAIN:
        small = np.exp(np.minimum(smap.log_gain, np.log(GAIN_MAX))) - 1
    else:
        small = smap.delta
    small = small * keep[..., None]
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    out = img.astype(np.float32)
    out = out * (1 + big) if mode == LightenMode.GAIN else out + big
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class ShadowScore:
    """Мера остатка тени в области поиска (на ¼).

    Поля:
        residual_mm2: Площадь пятен: бумага области без краски и кожи темнее опоры кольца на ``MEASURE_DROP_L``.
        residual_drop: Средний провал L по пятнам.
        over_mm2: Площадь пересвета: светлее опоры на ``MEASURE_DROP_L`` там, где осветление прибавило больше 2 L.
        max_grad: Наибольший (99.9 %) градиент поправки L, единиц L на мм — плавность.
        ink_contrast: Медиана контраста краски к фону в области после, делённая на до (1 — не изменился).
        area_mm2: Площадь области, мм².
    """

    residual_mm2: float
    residual_drop: float
    over_mm2: float
    max_grad: float
    ink_contrast: float
    area_mm2: float


def shadow_score(before: np.ndarray, after: np.ndarray, smap: ShadowMap, px_per_mm: float) -> ShadowScore:
    """Мера итога ``after`` в области поиска карты: остаток тени, пересвет, плавность поправки, контраст букв.

    Опора — медиана L кольца бумаги ``MEASURE_RING_MM`` снаружи области (вне области итоги вариантов совпадают),
    по компонентам области.

    Args:
        before: Итог до осветления (A) BGR uint8.
        after: Итог варианта BGR uint8.
        smap: Карта (маски области, книги, кожи, краски).
        px_per_mm: Плотность итога.

    Returns:
        Мера.
    """
    s = SHADOW_SCALE
    mm = px_per_mm * s
    to_l = lambda im: cv2.cvtColor(  # noqa: E731
        cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB
    )[..., 0].astype(np.float32) * (100 / 255)
    l0, l1 = to_l(before), to_l(after)
    area, book, skin, ink = smap.area, smap.book, smap.skin, smap.ink
    if not area.any():
        return ShadowScore(0, 0, 0, 0, 1, 0)
    dist, (iy, ix) = _nearest(area)
    n, labels = cv2.connectedComponents(area.astype(np.uint8), connectivity=8)
    owner = labels[iy, ix]
    ring = (
        book
        & ~ink
        & ~skin
        & ~smap.content
        & ~area
        & (dist >= MEASURE_RING_MM[0] * mm)
        & (dist <= MEASURE_RING_MM[1] * mm)
    )
    ref = np.full(area.shape, np.nan, np.float32)
    for k in range(1, n):
        own = ring & (owner == k)
        if own.sum() >= 50:
            ref[labels == k] = np.median(l0[own])
    zone = area & ~ink & ~skin & ~smap.content & np.isfinite(ref)
    drop = np.where(zone, ref - l1, 0)
    dark = _blobs(zone & (drop > MEASURE_DROP_L), mm)
    # Пересвет — только там, где осветление что-то прибавило (светлое место в самом A — не пересвет).
    bright = _blobs(zone & (-drop > MEASURE_DROP_L) & (l1 - l0 > 2), mm)
    # Плавность: градиент сглаженной (σ 1 мм) поправки L.
    change = cv2.GaussianBlur(l1 - l0, (0, 0), mm)
    gy, gx = np.gradient(change)
    grad = np.hypot(gx, gy)[area] * mm
    # Контраст букв: фон (закрытие окном 3 мм) минус краска, медиана по краске области, после к до.
    contrast = []
    for im in (l0, l1):
        k = max(3, int(round(3 * mm)) | 1)
        bg = cv2.morphologyEx(im, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        sel = area & ink
        contrast.append(float(np.median((bg - im)[sel])) if sel.sum() > 20 else np.nan)
    ratio = contrast[1] / contrast[0] if contrast[0] and np.isfinite(contrast[0]) else 1.0
    return ShadowScore(
        float(dark.sum() / mm**2),
        float(drop[dark].mean()) if dark.any() else 0.0,
        float(bright.sum() / mm**2),
        float(np.percentile(grad, 99.9)) if grad.size else 0.0,
        float(ratio),
        float(area.sum() / mm**2),
    )


def _blobs(mask: np.ndarray, mm: float) -> np.ndarray:
    """Маска без мелочи: открытие 3×3 и компоненты не меньше ``MEASURE_MIN_MM2``."""
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    small = np.nonzero(stats[:, cv2.CC_STAT_AREA] < MEASURE_MIN_MM2 * mm * mm)[0]
    return mask & ~np.isin(comp, small)


def reach_profile(inp: ShadowInputs, px_per_mm: float, max_mm: float = 40.0) -> np.ndarray:
    """Профиль тени по расстоянию от ядра: медиана L бумаги без краски и кожи в кольцах по 1 мм от ядра пальца.

    Только бумага книги в пределах ``max_mm`` от ядра; зона закраса исключена (там закрас, не тень).

    Args:
        inp: Итог и маски.
        px_per_mm: Плотность итога.
        max_mm: Наибольшее расстояние, мм.

    Returns:
        Медиана L по кольцам 0..max_mm (NaN — нет пикселей), float32.
    """
    s = SHADOW_SCALE
    small = cv2.resize(inp.img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    size = (small.shape[1], small.shape[0])
    mm = px_per_mm * s
    core = cv2.resize(inp.core.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    paint = cv2.resize(inp.paint.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    book = cv2.resize(inp.book.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.99
    light = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * (100 / 255)
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5) / mm
    sel = book & ~paint & ~ink_mask(small, mm) & ~skin_mask(small, mm, book) & (dist < max_mm)
    bins = dist[sel].astype(int)
    out = np.full(int(max_mm), np.nan, np.float32)
    for b in range(int(max_mm)):
        v = light[sel][bins == b]
        if v.size >= 30:
            out[b] = np.median(v)
    return out
