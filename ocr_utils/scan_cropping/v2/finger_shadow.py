"""Тень пальца на бумаге после закраса и copy back: карта затемнения в области вокруг пальца (pull-push по кольцу бумаги против сглаженной бумаги без краски, кожи и печати) и осветление итога прибавкой."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.border_fill import pull_push

# Стенд и выбор метода: research/finger_repaint (reports/finger_repaint.md, вариант «A0 + P»): на 28 кадрах с пятном
# остаток тени 17.8 тыс. → 3.2 тыс. мм², на 15 контрольных 338 → 9 мм², ~0.5 с CPU на кадр.

# Масштаб, на котором оценивается карта; поправка затем растягивается до полного итога.
SHADOW_SCALE = 0.25
# Досягаемость тени от ядра пальца вглубь книги, мм (радиус круга области поиска). Профиль по 28 кадрам пака-6:
# провал L бумаги вне зоны закраса 1–1.6 медианой (90 % — до 3.5) на 9–20 мм от ядра, к ~30 мм сходит на нет; на стенде
# было 25 мм, по решению пользователя (2026-10-14) — 15 мм: дальше тень слабая (1–2 L), а широкая зона захватывает
# естественное затемнение у края листа.
SHADOW_REACH_MM = 15.0
# Кольцо бумаги, по которому берётся опора «бумага без тени», мм от ядра пальца (от, до). Решение пользователя
# 2026-10-14: зона осветления — 15 мм, а опора — дальше, за тенью: с кольцом сразу за зоной 15 мм (15–25 мм) опора сама
# в тени (провал 1–1.6 L), и на 43 кадрах стенда остаток тени в зоне 15 мм был 3 246 мм² против 492 у зоны 25 мм.
REF_RING_MM = (25.0, 35.0)
# Краска: темнее закрытия серого окном INK_CLOSE_MM на INK_DARK уровней и малонасыщенная (хрома ≤ INK_MAX_CHROMA).
INK_CLOSE_MM = 3.0
INK_DARK = 20
INK_MAX_CHROMA = 16
# Кожа — пиксель оригинала, ушедший от окрестной бумаги в красное: Δa* Lab (8-битные единицы OpenCV) больше SKIN_DA,
# направление сдвига в плоскости (Δa*, Δb*) — в SKIN_ANGLE, градусы (кожа на белой бумаге ~55°, на жёлтой ПХ ~10°;
# фиолетовая и синяя печать — ниже −20°, пожелтение — выше 70°). Пак-6: у бумаги Δa* ≤ ~4, у кожи медиана a* 10–16.
SKIN_DA = 5.0
SKIN_ANGLE = (-20.0, 70.0)
# Кожа цветная и тёплая сама по себе: a* больше SKIN_MIN_A и хрома больше SKIN_MIN_CHROMA (у кожи пака-6 a* 4–18, хрома
# 10–30; чёрная и серая печать — хрома < 5).
SKIN_MIN_A = 3.0
SKIN_MIN_CHROMA = 8.0
# Отсев не-кожи (каймы цветных букв, штампы, линия у стыка страниц разного оттенка): открытие окном, мм, и наименьшее
# пятно, мм²; расширение маски кожи, мм (кайма пальца темнее и чуть менее красная).
SKIN_OPEN_MM = 1.0
SKIN_MIN_MM2 = 20.0
SKIN_GROW_MM = 1.0
# Крупная печать, которую маска краски не ловит (логотипы, заливки): пиксель оригинала темнее медианы бумаги кольца
# больше чем на CONTENT_DROP_L (L 0..100) — тень пальца по замерам до ~20 L; расширение, мм.
CONTENT_DROP_L = 25.0
CONTENT_GROW_MM = 1.0
# Сглаживание «бумаги как есть» и опоры, мм: край призрака LaMa резкий, σ 3 мм оставлял кайму.
SMOOTH_SIGMA_MM = 1.5
# Итоговое сглаживание карты внутри области, мм.
POST_SIGMA_MM = 1.0
# Рампа карты от края области (там, где снаружи бумага), мм: на краю поправка 0 и растёт плавно (smoothstep).
EDGE_RAMP_MM = 4.0
# Полоса закраса вне книги у области, мм: карта мерится и на ней (закрас темнее опоры — осветляется), со спадом до 0.
FILL_REACH_MM = 10.0
# Перцентиль, которым карта ограничивается сверху внутри области (отсечка одиночных выбросов).
CAP_PERCENTILE = 99.5


@dataclass(frozen=True)
class FingerShadow:
    """Карта тени пальца в координатах итога, на масштабе ``SHADOW_SCALE``.

    Поля:
        area: Область поиска тени (bool): ядро пальца + ``SHADOW_REACH_MM`` ∪ зона закраса, ∩ книга.
        skin: Кожа (bool) — в оценку и осветление не входит.
        ink: Краска (bool) — в оценку не входит.
        content: Крупная печать (bool) — в оценку не входит.
        delta: Прибавка по каналам BGR, float32 (H, W, 3), ≥ 0; ноль вне области и полосы закраса у неё.
        fill: Полоса закраса вне книги у области (bool), где карта тоже мерится; None — пустая.
    """

    area: np.ndarray
    skin: np.ndarray
    ink: np.ndarray
    content: np.ndarray
    delta: np.ndarray
    fill: np.ndarray | None = None


def disk(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Маска, раздутая кругом радиуса ``radius_px`` (через расстояние — быстро на больших радиусах)."""
    if not mask.any():
        return mask.copy()
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) <= radius_px


def smoothstep(t: np.ndarray) -> np.ndarray:
    """Гладкая ступенька 3t² − 2t³ (нулевая производная на концах)."""
    return t * t * (3 - 2 * t)


def ink_mask(bgr: np.ndarray, mm: float) -> np.ndarray:
    """Краска (буквы, линейки): темнее закрытия серого окном ``INK_CLOSE_MM`` на ``INK_DARK`` и малонасыщенная.

    Args:
        bgr: Картинка BGR uint8.
        mm: Плотность картинки, px/мм.

    Returns:
        Маска краски (bool), расширенная на 1 px.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = max(3, int(round(INK_CLOSE_MM * mm)) | 1)
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    ink = (paper.astype(np.int16) - gray > INK_DARK) & (chroma <= INK_MAX_CHROMA)
    return cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def search_area(paint: np.ndarray, core: np.ndarray, book: np.ndarray, mm: float) -> np.ndarray:
    """Область поиска тени: ядро пальца, раздутое кругом ``SHADOW_REACH_MM``, ∪ зона закраса, ∩ книга.

    Берутся только компоненты, задевающие зону закраса (ядра, отсеянные шагом пальцев, не в счёт).

    Args:
        paint: Зона закраса пальца (bool).
        core: Ядро пальца (bool).
        book: Книга — бумага после доводки ∪ защита контента (bool).
        mm: Плотность, px/мм.

    Returns:
        Область (bool).
    """
    grown = disk(core, SHADOW_REACH_MM * mm) | paint
    _, labels = cv2.connectedComponents(grown.astype(np.uint8), connectivity=8)
    keep = np.unique(labels[paint])
    return np.isin(labels, keep[keep > 0]) & book


def skin_mask(bgr: np.ndarray, mm: float, book: np.ndarray, near: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Кожа пальца в итоге — крупное сплошное пятно пикселей оригинала, ушедшее от окрестной бумаги в красное.

    Опора — медиана a*, b* светлой бумаги (ярче медианы L, без краски) в ``book ∩ near``: местная, потому что страницы
    разворота бывают разного оттенка. Пожелтевшая или порыжевшая бумага сдвинута по b* сама по себе и с собой даёт
    Δ ≈ 0, тень на ней уходит к серому. Кожа возможна только среди пикселей оригинала: розово-бурое пятно LaMa похоже на
    неё цветом, но его как раз надо осветлить.

    Args:
        bgr: Картинка BGR uint8.
        mm: Плотность, px/мм.
        book: Книга (bool).
        near: Где брать опору бумаги (bool) — окрестность области поиска; если там мало бумаги — вся книга.
        allowed: Пиксели оригинала в итоге (bool): вне зоны закраса и в copy back.

    Returns:
        Маска кожи (bool).
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    light, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    sel = book & ~ink_mask(bgr, mm)
    if (sel & near).sum() >= 100:
        sel = sel & near
    if sel.sum() < 100:
        return np.zeros(book.shape, bool)
    bright = sel & (light >= np.median(light[sel]))
    da, db = a - np.median(a[bright]), b - np.median(b[bright])
    angle = np.degrees(np.arctan2(db, da))
    # Сама по себе кожа цветная и тёплая: чёрная и серая печать на цветной бумаге (голубая обложка МТС 1990/01) тоже
    # «краснее» бумаги относительно, но бесцветна.
    warm = (a > SKIN_MIN_A) & (np.hypot(a, b) > SKIN_MIN_CHROMA)
    skin = ((da > SKIN_DA) & (angle >= SKIN_ANGLE[0]) & (angle <= SKIN_ANGLE[1]) & warm & allowed).astype(np.uint8)
    # Без тонкого и мелкого: каймы цветных букв, штампы, линии — не палец.
    k = max(3, int(round(SKIN_OPEN_MM * mm)) | 1)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    _, comp, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    small = np.nonzero(stats[:, cv2.CC_STAT_AREA] < SKIN_MIN_MM2 * mm * mm)[0]
    return disk((skin > 0) & ~np.isin(comp, small), SKIN_GROW_MM * mm)


def smooth_fill(img: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Плавная поверхность по пикселям с весом: pull-push (дыры заполняются с соседей), затем гаусс ``sigma`` px."""
    return cv2.GaussianBlur(pull_push(img, weight.astype(np.float32)), (0, 0), sigma)


def window_of(mask: np.ndarray, margin: int) -> tuple[slice, slice]:
    """Рамка маски с запасом ``margin`` px, обрезанная по картинке, — срезы ``(строки, столбцы)``."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return (
        slice(max(0, ys.min() - margin), min(h, ys.max() + margin + 1)),
        slice(max(0, xs.min() - margin), min(w, xs.max() + margin + 1)),
    )


def postprocess(delta: np.ndarray, area: np.ndarray, fill: np.ndarray, book: np.ndarray, mm: float) -> np.ndarray:
    """Карта после оценки: отсечка сверху, сглаживание по области, рампа от края, спад в закрасе вне книги.

    Args:
        delta: Прибавка по каналам, ≥ 0 (оценена и на области, и на закрасе вне книги у неё).
        area: Область поиска (bool).
        fill: Закрас вне книги в пределах ``FILL_REACH_MM`` от области (bool) — там карта мерится так же, как в области.
        book: Книга (bool).
        mm: Плотность, px/мм.

    Returns:
        Карта; ноль вне области и полосы закраса у неё.
    """
    where = area | fill
    inside = delta[where]
    cap = np.percentile(inside, CAP_PERCENTILE, axis=0) if inside.size else np.zeros(3)
    delta = np.minimum(delta, cap)
    # Сглаживание по области и по полосе закраса порознь (нормированная свёртка): через край листа карта не перетекает —
    # большая поправка тёмного края листа не ложится на светлый закрас.
    smoothed = np.zeros_like(delta)
    for part in (area, fill):
        wts = part.astype(np.float32)
        num = cv2.GaussianBlur(delta * wts[..., None], (0, 0), POST_SIGMA_MM * mm)
        den = cv2.GaussianBlur(wts, (0, 0), POST_SIGMA_MM * mm)
        smoothed[part] = (num / np.maximum(den, 1e-6)[..., None])[part]
    delta = smoothed
    # Рампа: 0 там, где снаружи области бумага (край тени), растёт smoothstep на EDGE_RAMP_MM. У края листа (снаружи
    # не книга) рампы нет: там продолжается полоса закраса.
    paper_out = book & ~area
    d = cv2.distanceTransform((~paper_out).astype(np.uint8), cv2.DIST_L2, 5) / (EDGE_RAMP_MM * mm)
    delta = delta * smoothstep(np.clip(d, 0, 1))[..., None] * where[..., None]
    # Полоса закраса: карта там своя (закрас темнее опоры — осветляется, светлый — нет) и спадает к внешнему краю полосы.
    if fill.any():
        dist = cv2.distanceTransform((~area).astype(np.uint8), cv2.DIST_L2, 5)
        fade = smoothstep(np.clip(1 - dist / (FILL_REACH_MM * mm), 0, 1))
        delta[fill] *= fade[fill][:, None]
    return delta


def estimate_finger_shadow(
    img: np.ndarray, paint: np.ndarray, core: np.ndarray, book: np.ndarray, original: np.ndarray, px_per_mm: float
) -> FingerShadow:
    """Карта тени пальца на итоге (после закраса, copy back и растяжки уровней).

    Шаги (на ¼):
    1. Область поиска — ``search_area``.
    2. Из оценки исключены краска, кожа (``skin_mask``) и крупная печать (пиксели оригинала темнее бумаги кольца на
       ``CONTENT_DROP_L``).
    3. B — «бумага как есть»: pull-push по бумаге до внешнего края кольца без исключённого, гаусс ``SMOOTH_SIGMA_MM``.
    4. R — «бумага без тени»: pull-push только по кольцу бумаги ``REF_RING_MM`` от ядра пальца (за тенью), тот же гаусс;
       между зоной и кольцом R продолжается гладко.
    5. Карта ``max(0, R − B)`` по каналам, затем ``postprocess``.

    Args:
        img: Итог BGR uint8 (полный масштаб).
        paint: Зона закраса пальца в координатах итога (bool).
        core: Ядро пальца в координатах итога (bool).
        book: Бумага после доводки ∪ защита контента в координатах итога (bool).
        original: Где в итоге пиксели оригинала (bool): вне зоны закраса и возвращённое copy back.
        px_per_mm: Плотность итога, px/мм.

    Returns:
        Карта.
    """
    s = SHADOW_SCALE
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    size = (small.shape[1], small.shape[0])
    mm = px_per_mm * s
    paint_s, core_s, original_s = (
        cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0 for m in (paint, core, original)
    )
    book_s = cv2.resize(book.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.99
    area = search_area(paint_s, core_s, book_s, mm)
    empty = np.zeros(area.shape, bool)
    zeros = np.zeros(small.shape, np.float32)
    if not area.any():
        return FingerShadow(area, empty, empty, empty, zeros)
    ink = ink_mask(small, mm)
    # Расстояние от ядер своих пальцев (задевающих зону), мм: кольцо опоры и окрестность для опоры цвета кожи.
    own_core = core_s & disk(area, SHADOW_REACH_MM * mm)
    from_core = cv2.distanceTransform((~own_core).astype(np.uint8), cv2.DIST_L2, 5) / mm
    near = (from_core <= REF_RING_MM[1]) & ~area
    skin = skin_mask(small, mm, book_s, near, original_s)
    clean = book_s & ~ink & ~skin
    ring = clean & ~area & ~paint_s & (from_core >= REF_RING_MM[0]) & (from_core <= REF_RING_MM[1])
    content = empty
    if ring.sum() >= 50:
        # Крупная печать — только среди пикселей оригинала: глубокое пятно LaMa темнее порога, его надо осветлить.
        light = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * (100 / 255)
        content = disk((light < np.median(light[ring]) - CONTENT_DROP_L) & original_s, CONTENT_GROW_MM * mm)
        clean &= ~content
        ring &= ~content
    # Рабочее окно — рамка области с запасом кольца и сглаживания.
    win = window_of(area, int((REF_RING_MM[1] - SHADOW_REACH_MM + 3 * SMOOTH_SIGMA_MM) * mm) + 2)
    w_img = small[win].astype(np.float32)
    sigma = SMOOTH_SIGMA_MM * mm
    # Закрас до рамки вне книги у области: карта мерится и на нём — его тон задал pull-push от чистой полосы бумаги, и
    # он бывает уже светлым там, где край листа тёмный (продолжение поправки с края переосветляло, МТС 1990/03 IMG_0972).
    fill = ~book_s & disk(area, FILL_REACH_MM * mm)
    # «Как есть» — порознь для листа и для полосы закраса: сглаживание через край листа смешало бы их тон.
    back_book = smooth_fill(w_img, clean[win] & (from_core[win] <= REF_RING_MM[1]), sigma)
    back_fill = smooth_fill(w_img, fill[win], sigma) if fill[win].any() else back_book
    back = np.where(fill[win][..., None], back_fill, back_book)
    ref = smooth_fill(w_img, ring[win], sigma)
    delta = zeros.copy()
    delta[win] = postprocess(np.maximum(ref - back, 0), area[win], fill[win], book_s[win], mm)
    return FingerShadow(area, skin, ink, content, delta, fill)


def lighten_finger_shadow(img: np.ndarray, shadow: FingerShadow) -> np.ndarray:
    """Осветление итога прибавкой карты (растянутой до полного масштаба); кожа не трогается.

    Прибавка, а не коэффициент по каналам: на глубоком буром пятне LaMa коэффициент упирается в предел и сдвигает цвет
    (РИСК 1993/04-05 IMG_0558 — жёлто-оранжевое пятно).

    Args:
        img: Итог BGR uint8.
        shadow: Карта.

    Returns:
        Осветлённый итог BGR uint8.
    """
    small = shadow.delta * (~shadow.skin).astype(np.float32)[..., None]
    nonzero = small.max(axis=2) > 0
    if not nonzero.any():
        return img
    # Только в рамке ненулевой карты (с запасом на интерполяцию): полный кадр во float32 — сотни МБ на воркер.
    rows, cols = window_of(nonzero, 2)
    h, w = img.shape[:2]
    ky, kx = h / small.shape[0], w / small.shape[1]
    y0, y1 = int(rows.start * ky), min(h, int(np.ceil(rows.stop * ky)))
    x0, x1 = int(cols.start * kx), min(w, int(np.ceil(cols.stop * kx)))
    # Растягивается только рамка карты, отдельно по области на листе и по полосе закраса, с нормировкой по маске части:
    # кубическая интерполяция через край листа переносила большую поправку тёмного края на светлый закрас (+20 уровней
    # на первом пикселе) и давала «звон» с обратной стороны.
    size = (x1 - x0, y1 - y0)
    add = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    fill = shadow.fill if shadow.fill is not None else np.zeros_like(shadow.area)
    for region in (shadow.area & ~fill, fill):
        part = region[rows, cols]
        if not part.any():
            continue
        weight = part.astype(np.float32)
        num = cv2.resize(small[rows, cols] * weight[..., None], size, interpolation=cv2.INTER_LINEAR)
        den = cv2.resize(weight, size, interpolation=cv2.INTER_LINEAR)
        inside = (cv2.resize(part.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0) & (den > 1e-6)
        add[inside] = num[inside] / den[inside][:, None]
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(img[y0:y1, x0:x1].astype(np.float32) + add + 0.5, 0, 255).astype(np.uint8)
    return out
