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
# «Бумага как есть» в зоне закраса пальца (и SHARP_GROW_MM вокруг) и в полосе закраса вне книги — почти без сглаживания:
# у пятна LaMa резкий край, и B с гауссом SMOOTH_SIGMA_MM недоосветляла его изнутри (тёмная дуга) и переосветляла
# снаружи (светлый ореол) — «вмятина» у МТС 1991/02 IMG_0622, IMG_0634. Там B — pull-push с гауссом SHARP_SIGMA_MM и
# медиана окна SHARP_MEDIAN_MM (гасит зерно, край пятна держит); переход к обычной B — гаусс маски SHARP_BLEND_MM; карта
# там сглаживается SHARP_POST_SIGMA_MM.
SHARP_GROW_MM = 1.0
SHARP_SIGMA_MM = 0.3
SHARP_MEDIAN_MM = 1.0
SHARP_BLEND_MM = 0.5
SHARP_POST_SIGMA_MM = 0.3
# Опора у края листа — профиль кольца опоры по расстоянию от края книги: полоса у края (торец, завал страницы) темнее
# бумаги в глубине, и pull-push кольца высветлял её до белого диска (МТС 1991/02 IMG_0669, IMG_0666). В полосе до
# EDGE_PROFILE_MM от края опора — не светлее медианы кольца на том же расстоянии (корзины EDGE_BIN_MM, в корзине не
# меньше EDGE_BIN_MIN_PX пикселей на масштабе карты).
EDGE_PROFILE_MM = 15.0
EDGE_BIN_MM = 0.5
EDGE_BIN_MIN_PX = 10
# Отсечка карты сверху — мягкая: до cap карта как есть, выше гладко подходит к (1 + CAP_SOFT_ROOM)·cap (без плато с
# резким краем). Колено ниже cap (0.8·cap) недоосветляло самую глубокую часть пятна LaMa кольцами по изолиниям.
CAP_SOFT_ROOM = 0.25
# Кожа не осветляется, вокруг неё прибавка нарастает от нуля на SKIN_FEATHER_MM (без выреза с резким краем).
SKIN_FEATHER_MM = 1.0
# Опора чуть за краем листа (решение пользователя 2026-10-15): в кольцо входит и полоса закраса до RING_OUT_MM за краем
# книги, но только пиксели не темнее медианы кольца на листе той же зоны больше чем на RING_OUT_MAX_DROP (L 0..100):
# закрас — pull-push от края листа и продолжает тень у края (РИСК 1993/01 IMG_0258: без защиты прибавка зоны 38.9 → 5.7).
RING_OUT_MM = 5.0
# Опора зоны надёжна, если своего цвета в кольце-кандидате не меньше этой доли (стенд, 114 зон: у 112 — 0.78 и больше,
# у зон на стыке белой страницы и голубой плашки — 0.28 и 0.46, МТС 1990/09 IMG_0310, IMG_0248).
RING_MIN_SHARE = 0.5
RING_OUT_MAX_DROP = 2.0
# Цвет пикселя: пиксель оригинала осветляется, только если его «бумага как есть» близка по a*b* к опоре — вес 1 до
# COLOUR_FULL_AB, к нулю на ZONE_MAX_AB_GAP. Тень меняет яркость, а не цвет: голубое у белой опоры не уходит в белое,
# белый обрез у голубой — не розовеет. Стенд (52 кадра): у тени на пикселях оригинала расстояние B–R 99 % ≤ 12.6, у
# голубого против белого — 34–44. Закрас LaMa от правила свободен: бурые и тёмно-синие пятна (РИСК 1993/04-05 IMG_0558,
# МТС 1991/02 IMG_0622) другого оттенка, их как раз снимать.
COLOUR_FULL_AB = 10.0
# Потолок осветления (решение пользователя 2026-10-15: чёрное не должно уходить в белое): прибавка светлоты L (0..100)
# мягко ограничена — до MAX_ADD_KNEE_L как есть, выше гладко к MAX_ADD_L. Стенд: нужная прибавка 99 % по кадру — медиана
# 17, 90 % — 29, наибольшая 62 (IMG_0622: тёмно-синее пятно на голубом), чёрное → белое — 80–90. Было 55/65 — на IMG_0622
# оставался след пятна; по решению пользователя 2026-10-15 — 60/70 (чёрное — до светло-серого, около L 75, не белого).
# И итог не светлее 90-го перцентиля L кольца своей зоны + RING_LIGHT_MARGIN.
MAX_ADD_KNEE_L = 60.0
MAX_ADD_L = 70.0
RING_LIGHT_MARGIN = 2.0
# Рампа карты от края области (там, где снаружи бумага), мм: на краю поправка 0 и растёт плавно (smoothstep).
EDGE_RAMP_MM = 4.0
# Полоса закраса вне книги у области, мм: карта мерится и на ней (закрас темнее опоры — осветляется) с той же опорой и
# резкой B; в зоне закраса пальца без спада, дальше — спад до 0 на FILL_REACH_MM.
FILL_REACH_MM = 10.0
# Перцентиль, которым карта ограничивается сверху внутри области (отсечка одиночных выбросов).
CAP_PERCENTILE = 99.5
# Опора — та же поверхность, что вокруг пальца: тень меняет яркость, а не цвет. Если медианный цвет (a*, b*, 8-битные
# единицы OpenCV Lab) кольца опоры зоны отличается от цвета поверхности у пальца больше чем на ZONE_MAX_AB_GAP, опора
# взята с чужой поверхности, и зона не осветляется: палец на голубой обложке у края, а в кольце — белый торец и закрас
# (МТС 1990/04 IMG_1112: прибавка 101 уровень в среднем, обложка высветлилась до белого). Пак-6, 5398 зон по оригиналам
# кадров: у 5395 расхождение ≤ 10.8 (99 % — ≤ 6.3), у трёх — 31–35 (IMG_1112, МТС 1990/09 IMG_0248, IMG_0310: палец на
# стыке белой страницы и голубой обложки).
ZONE_MAX_AB_GAP = 20.0
# Наименьшее число пикселей своего кольца опоры у зоны (на масштабе карты, ¼): меньше — опоры у зоны нет, не осветлять.
ZONE_MIN_RING_PX = 20


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
        gaps: Расхождение цвета опоры и поверхности у пальца по зонам (связным частям ``area``, по порядку меток), a*b*;
            зоны с расхождением больше :data:`ZONE_MAX_AB_GAP` не осветляются.
        ref: Опора R «бумага без тени» по каналам BGR, float32 (H, W, 3); ноль вне рабочего окна. Для стенда и отладки.
        back: «Бумага как есть» B на масштабе карты, float32 (H, W, 3); ноль вне рабочего окна.
        sharp: Вес резкой B (0..1, float32): зона закраса пальца и полоса закраса — там осветление уточняется на полном
            масштабе (:func:`lighten_finger_shadow`).
        px_per_mm: Плотность итога, px/мм (полный масштаб).
        ring: Кольцо опоры — пиксели pull-push R (bool), включая полосу закраса за краем листа; для оверлея.
        ring_dropped: Пиксели кольца-кандидата, отсеянные по цвету или как тёмный закрас за краем (bool); для оверлея.
        paper: «Бумага как есть» — пиксели, по которым строится B (bool); для оверлея.
        book: Книга на масштабе карты (bool); для оверлея (кольцо на листе и за краем).
    """

    area: np.ndarray
    skin: np.ndarray
    ink: np.ndarray
    content: np.ndarray
    delta: np.ndarray
    fill: np.ndarray | None = None
    gaps: tuple[float, ...] = ()
    ref: np.ndarray | None = None
    back: np.ndarray | None = None
    sharp: np.ndarray | None = None
    px_per_mm: float = 0.0
    ring: np.ndarray | None = None
    ring_dropped: np.ndarray | None = None
    paper: np.ndarray | None = None
    book: np.ndarray | None = None


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


def soft_cap(delta: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Мягкая отсечка сверху: до ``cap`` карта как есть, выше — гладко к ``(1 + CAP_SOFT_ROOM)·cap`` (без плато с краем).

    Args:
        delta: Карта по каналам, ≥ 0.
        cap: Порог по каналам, shape (3,).

    Returns:
        Карта, не больше ``(1 + CAP_SOFT_ROOM)·cap``; производная непрерывна в ``cap``.
    """
    room = np.maximum(CAP_SOFT_ROOM * cap, 1e-6)
    over = np.maximum(delta - cap, 0)
    # Выше порога — экспонента к пределу: наклон 1 в cap, предел cap + room на бесконечности.
    return np.where(delta > cap, cap + room * (1 - np.exp(-over / room)), delta)


def postprocess(
    delta: np.ndarray,
    area: np.ndarray,
    fill: np.ndarray,
    book: np.ndarray,
    sharp: np.ndarray,
    paint: np.ndarray,
    mm: float,
) -> np.ndarray:
    """Карта после оценки: мягкая отсечка сверху, сглаживание по области и полосе закраса порознь, рампа, спад полосы.

    Args:
        delta: Прибавка по каналам, ≥ 0 (оценена и на области, и на закрасе вне книги у неё).
        area: Область поиска (bool).
        fill: Закрас вне книги в пределах ``FILL_REACH_MM`` от области (bool) — там карта мерится так же, как в области.
        book: Книга (bool).
        sharp: Вес «резкой» B (0..1): зона закраса пальца и полоса закраса — там карта сглаживается
            ``SHARP_POST_SIGMA_MM``, вне — ``POST_SIGMA_MM``.
        paint: Откуда отсчитывается спад полосы закраса (bool): область ∪ зона закраса пальца у неё — тот же отсчёт, что
            у самой полосы (иначе прибавка обрывается на её границе ступенькой).
        mm: Плотность, px/мм.

    Returns:
        Карта; ноль вне области и полосы закраса у неё.
    """
    where = area | fill
    inside = delta[where]
    if inside.size:
        delta = soft_cap(delta, np.percentile(inside, CAP_PERCENTILE, axis=0))
    # Сглаживание по области и по полосе закраса порознь (нормированная свёртка): через край листа карта не перетекает —
    # большая поправка тёмного края листа не ложится на светлый закрас (МТС 1990/03 IMG_0972). В зоне резкой B — своё,
    # малое сглаживание.
    smoothed = np.zeros_like(delta)
    for part in (area, fill):
        wts = part.astype(np.float32)
        pair = []
        for sigma_mm in (POST_SIGMA_MM, SHARP_POST_SIGMA_MM):
            num = cv2.GaussianBlur(delta * wts[..., None], (0, 0), sigma_mm * mm)
            den = cv2.GaussianBlur(wts, (0, 0), sigma_mm * mm)
            pair.append(num / np.maximum(den, 1e-6)[..., None])
        mixed = (1 - sharp[..., None]) * pair[0] + sharp[..., None] * pair[1]
        smoothed[part] = mixed[part]
    delta = smoothed
    # Рампа: 0 там, где снаружи области бумага (край тени), растёт smoothstep на EDGE_RAMP_MM. У края листа (снаружи
    # не книга) рампы нет: там продолжается полоса закраса.
    paper_out = book & ~area
    d = cv2.distanceTransform((~paper_out).astype(np.uint8), cv2.DIST_L2, 5) / (EDGE_RAMP_MM * mm)
    delta = delta * smoothstep(np.clip(d, 0, 1))[..., None] * where[..., None]
    # Полоса закраса: в зоне закраса пальца — как область, дальше спадает к внешнему краю полосы.
    if fill.any():
        dist = cv2.distanceTransform((~(area | paint)).astype(np.uint8), cv2.DIST_L2, 5)
        fade = smoothstep(np.clip(1 - dist / (FILL_REACH_MM * mm), 0, 1))
        delta[fill] *= fade[fill][:, None]
    return delta


def sharp_paper(img: np.ndarray, weight: np.ndarray, mm: float, blur: bool = True) -> np.ndarray:
    """Резкая «бумага как есть» по пикселям ``weight``: pull-push (с гауссом ``SHARP_SIGMA_MM`` или без) и медиана.

    Args:
        img: Окно итога BGR float32.
        weight: Пиксели-опоры (bool).
        mm: Плотность, px/мм.
        blur: Гаусс ``SHARP_SIGMA_MM`` после pull-push; без него (полный масштаб) край пятна остаётся ступенькой —
            медиана ``SHARP_MEDIAN_MM`` ступеньку не размывает, а зерно гасит.

    Returns:
        B по каналам, float32.
    """
    raw = smooth_fill(img, weight, SHARP_SIGMA_MM * mm) if blur else pull_push(img, weight.astype(np.float32))
    k = max(3, int(round(SHARP_MEDIAN_MM * mm)) | 1)
    # Медиана — по каналам uint8 (окно больше 5 px у OpenCV только для uint8).
    return cv2.medianBlur(np.clip(raw + 0.5, 0, 255).astype(np.uint8), k).astype(np.float32)


def paper_as_is(
    img: np.ndarray, book_weight: np.ndarray, fill: np.ndarray, sharp_zone: np.ndarray, mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """«Бумага как есть» B: гладкая вне зоны закраса, почти без сглаживания в ней и в полосе закраса вне книги.

    Лист и полоса закраса считаются порознь: сглаживание через край листа смешало бы их тон (тёмный край листа
    затемнял B первого столбца закраса, и тот осветлялся сверх бумаги — МТС 1990/03 IMG_0972).

    Args:
        img: Окно итога BGR float32 на масштабе карты.
        book_weight: Чистая бумага книги (bool) — опоры B на листе.
        fill: Полоса закраса вне книги (bool) — опоры B на ней.
        sharp_zone: Где B на листе резкая (bool): зона закраса пальца + ``SHARP_GROW_MM``.
        mm: Плотность, px/мм.

    Returns:
        ``(B, вес резкой B 0..1)`` — вес нужен и сглаживанию карты; в полосе закраса вес 1.
    """
    soft = smooth_fill(img, book_weight, SMOOTH_SIGMA_MM * mm)
    weight = np.zeros(fill.shape, np.float32)
    back = soft
    if sharp_zone.any():
        sharp = sharp_paper(img, book_weight, mm)
        weight = np.clip(cv2.GaussianBlur(sharp_zone.astype(np.float32), (0, 0), SHARP_BLEND_MM * mm), 0, 1)
        # Внутри зоны вес ровно 1: размытие маски только расширяет переход наружу.
        weight = np.maximum(weight, sharp_zone.astype(np.float32))
        back = weight[..., None] * sharp + (1 - weight[..., None]) * soft
    if fill.any():
        back = np.where(fill[..., None], sharp_paper(img, fill, mm), back)
        weight = np.where(fill, 1.0, weight).astype(np.float32)
    return back, weight


def fill_inner_holes(mask: np.ndarray) -> np.ndarray:
    """Маска с залитыми внутренними дырами: «не маска», не связанная с краем картинки, становится маской.

    Args:
        mask: Маска (bool).

    Returns:
        Маска без внутренних дыр (bool).
    """
    outside = (~mask).astype(np.uint8)
    count, labels = cv2.connectedComponents(outside, connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    return mask | ((labels > 0) & ~np.isin(labels, border))


def edge_profile_ref(
    ref: np.ndarray, img: np.ndarray, ring: np.ndarray, zones: np.ndarray, book: np.ndarray, mm: float
) -> np.ndarray:
    """Опора у края листа: не светлее кольца опоры на том же расстоянии от края книги (профиль по зонам).

    Полоса у края (торец, завал страницы) темнее бумаги в глубине. Pull-push кольца берёт тон в основном из глубины и
    высветлял полосу; профиль медиан кольца по расстоянию от края оставляет полосу опорой, и осветляется только то,
    что темнее неё на том же расстоянии (сама тень и пятно LaMa).

    Args:
        ref: Опора R по каналам (pull-push кольца), float32.
        img: Итог BGR float32 (то же окно).
        ring: Кольцо опоры (bool).
        zones: Метки зон (связных частей области поиска), 0 — вне зон.
        book: Книга (bool).
        mm: Плотность, px/мм.

    Returns:
        Опора, ограниченная профилем в полосе ``EDGE_PROFILE_MM`` от края; вне полосы и без профиля — прежняя.
    """
    # Край листа — только внешний: дыры маски бумаги внутри книги (выемка у кончика пальца, МТС 1991/02 IMG_0669) не
    # край, иначе опору вокруг них опускало до тона полосы у края — ступеньками по корзинам.
    solid = fill_inner_holes(book)
    if solid.all():
        return ref
    # Расстояние от края книги, мм; вне книги (полоса закраса) — 0.
    dist = cv2.distanceTransform(solid.astype(np.uint8), cv2.DIST_L2, 5) / mm
    bins = int(np.ceil(EDGE_PROFILE_MM / EDGE_BIN_MM))
    centers = (np.arange(bins) + 0.5) * EDGE_BIN_MM
    reach = (REF_RING_MM[1] - SHADOW_REACH_MM) * mm + 2
    out = ref.copy()
    for k in range(1, int(zones.max()) + 1):
        zone = zones == k
        near = cv2.distanceTransform((~zone).astype(np.uint8), cv2.DIST_L2, 5)
        own = ring & (near <= reach) & (dist < EDGE_PROFILE_MM)
        idx = np.minimum((dist[own] / EDGE_BIN_MM).astype(int), bins - 1)
        values = img[own]
        good, medians = [], []
        for i in range(bins):
            sel = idx == i
            if sel.sum() >= EDGE_BIN_MIN_PX:
                good.append(centers[i])
                medians.append(np.median(values[sel], axis=0))
        # Профиль нужен от самого края: без корзин у края (кольцо не доходит до полосы) опора не трогается.
        if len(good) < 3 or good[0] > 3 * EDGE_BIN_MM:
            continue
        good, medians = np.asarray(good), np.asarray(medians)
        # Профиль сглаживается по соседним корзинам (окно 3): медианы корзин скачут на 1–3 уровня, и опора шла ступеньками.
        if len(good) >= 3:
            padded = np.concatenate([medians[:1], medians, medians[-1:]])
            medians = (padded[:-2] + padded[1:-1] + padded[2:]) / 3
        # Где профиль: зона и полоса закраса у неё, ближе к краю, чем последняя корзина профиля.
        target = (near <= FILL_REACH_MM * mm + 2) & (dist <= good[-1])
        d = dist[target]
        limit = np.stack([np.interp(d, good, medians[:, c]) for c in range(3)], axis=1)
        # К концу профиля ограничение гладко сходит на нет (последние 2 мм) — без ступеньки на его границе.
        fade = smoothstep(np.clip((good[-1] - d) / 2.0, 0, 1))[:, None]
        cur = ref[target]
        out[target] = np.minimum(cur, fade * limit + (1 - fade) * cur)
    return out


def zone_rings(
    small: np.ndarray,
    area: np.ndarray,
    clean: np.ndarray,
    candidates: np.ndarray,
    book: np.ndarray,
    paint: np.ndarray,
    original: np.ndarray,
    mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Кольцо опоры по зонам (связным частям области) и крупная печать.

    Для каждой зоны:
    1. кандидаты кольца рядом с зоной;
    2. цвет опоры — медиана a*b* поверхности у пальца (чистая бумага зоны вне закраса), не видно — преобладающий в
       кольце; кандидаты дальше ``ZONE_MAX_AB_GAP`` от него отсеиваются (белый обрез у голубой обложки, МТС 1990/04
       IMG_1112; голубая плашка у белой страницы);
    3. пиксели закраса за краем листа — только не темнее кольца на листе (``RING_OUT_MAX_DROP``).

    Крупная печать — пиксели оригинала темнее медианы колец всех зон того же цвета на ``CONTENT_DROP_L``: тень руки
    бывает шире кольца одной зоны (РИСК 1993/01 IMG_0258), а зоны на разных поверхностях не смешиваются (у IMG_1112
    порог по белой бумаге левой страницы забрал в «печать» всю голубую обложку).

    Args:
        small: Итог BGR на масштабе карты.
        area: Область поиска (bool).
        clean: Чистая бумага книги — без краски и кожи (bool).
        candidates: Кольцо-кандидат: чистая бумага и закрас у края листа в полосе ``REF_RING_MM`` от ядра (bool).
        book: Книга (bool).
        paint: Зона закраса пальца (bool).
        original: Пиксели оригинала (bool).
        mm: Плотность на масштабе карты, px/мм.

    Returns:
        ``(кольцо, отсеянное из кандидатов, крупная печать, свой цвет)`` — bool. «Свой цвет» — пиксели, близкие по a*b*
        к цвету опоры своей зоны: только по ним строится гладкая «бумага как есть» (резкий переход голубой обложки и
        белого обреза не размывается в прибавку).
    """
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    light, ab = lab[..., 0] * (100 / 255), lab[..., 1:]
    count, zones = cv2.connectedComponents(area.astype(np.uint8), connectivity=8)
    reach = (REF_RING_MM[1] - SHADOW_REACH_MM) * mm + 2
    rings, colours, nears = [], [], []
    for k in range(1, count):
        zone = zones == k
        near = cv2.distanceTransform((~zone).astype(np.uint8), cv2.DIST_L2, 5) <= reach
        ring = candidates & near
        if ring.sum() < ZONE_MIN_RING_PX:
            continue
        surface = clean & zone & ~paint
        source = surface if surface.sum() >= ZONE_MIN_RING_PX else ring
        colour = np.median(ab[source], axis=0)
        # За краем листа допуск строже (COLOUR_FULL_AB): закрас у цветной плашки рядом подкрашен её цветом (МТС 1990/09
        # IMG_0310: голубоватый закрас под голубой плашкой давал голубой налёт на осветлённой полосе).
        gap = np.linalg.norm(ab - colour, axis=2)
        ring &= np.where(book, gap <= ZONE_MAX_AB_GAP, gap <= COLOUR_FULL_AB)
        on_book = ring & book
        if on_book.sum() >= ZONE_MIN_RING_PX:
            ring &= book | (light >= np.median(light[on_book]) - RING_OUT_MAX_DROP)
        if ring.sum() < ZONE_MIN_RING_PX:
            continue
        rings.append(ring)
        colours.append(np.median(ab[ring], axis=0))
        nears.append(near)
    kept = np.zeros(area.shape, bool)
    content = np.zeros(area.shape, bool)
    # Свой цвет: пиксель близок по a*b* к цвету опоры своей зоны (первой из задевающих его окрестностей); вне
    # окрестностей зон — без ограничения.
    own_colour = np.ones(area.shape, bool)
    taken = np.zeros(area.shape, bool)
    for colour, near in zip(colours, nears):
        part = near & ~taken
        own_colour[part] = (np.linalg.norm(ab - colour, axis=2) <= ZONE_MAX_AB_GAP)[part]
        taken |= near
    for ring, colour, near in zip(rings, colours, nears):
        pool = np.zeros(area.shape, bool)
        for other, other_colour in zip(rings, colours):
            if np.linalg.norm(other_colour - colour) <= ZONE_MAX_AB_GAP:
                pool |= other
        printed = disk((light < np.median(light[pool]) - CONTENT_DROP_L) & original & near, CONTENT_GROW_MM * mm)
        content |= printed
        kept |= ring & ~printed
    return kept, candidates & ~kept & ~content, content, own_colour


def colour_weight(back: np.ndarray, ref: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Вес осветления по цвету: пиксель оригинала — только если его цвет близок по a*b* к опоре R (``COLOUR_FULL_AB``).

    Args:
        back: Цвет пикселя по каналам (картинка после медианы), float32.
        ref: R по каналам, float32.
        original: Пиксели оригинала вне зоны закраса (bool); остальным вес 1.

    Returns:
        Вес 0..1, float32.
    """
    ab_b = cv2.cvtColor(np.clip(back + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 1:].astype(np.float32)
    ab_r = cv2.cvtColor(np.clip(ref + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 1:].astype(np.float32)
    dist = np.linalg.norm(ab_b - ab_r, axis=2)
    weight = 1 - smoothstep(np.clip((dist - COLOUR_FULL_AB) / (ZONE_MAX_AB_GAP - COLOUR_FULL_AB), 0, 1))
    return np.where(original, weight, 1.0).astype(np.float32)


def cap_lightness(base: np.ndarray, add: np.ndarray, limit: np.ndarray | None = None) -> np.ndarray:
    """Прибавка, ужатая так, чтобы светлота росла не больше мягкого потолка ``MAX_ADD_L`` и не выше ``limit``.

    Прибавка масштабируется целиком (по всем каналам одним множителем): цвет сдвига сохраняется.

    Args:
        base: Картинка до осветления, BGR float32.
        add: Прибавка по каналам, float32, ≥ 0.
        limit: Наибольшая светлота итога по пикселям (L 0..100) или None.

    Returns:
        Прибавка того же размера.
    """
    before, after = (
        cv2.cvtColor(np.clip(x + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        * (100 / 255)
        for x in (base, base + add)
    )
    gain = np.maximum(after - before, 0)
    room = MAX_ADD_L - MAX_ADD_KNEE_L
    # Мягкий потолок: до колена как есть, выше — экспонента к MAX_ADD_L.
    target = np.where(
        gain > MAX_ADD_KNEE_L, MAX_ADD_KNEE_L + room * (1 - np.exp(-(gain - MAX_ADD_KNEE_L) / room)), gain
    )
    if limit is not None:
        target = np.minimum(target, np.maximum(limit - before, 0))
    scale = np.where(gain > 1e-3, np.clip(target / np.maximum(gain, 1e-3), 0, 1), 1.0)
    return add * scale[..., None]


def ring_light_limit(small: np.ndarray, ring: np.ndarray, zones: np.ndarray, fill: np.ndarray, mm: float) -> np.ndarray:
    """Наибольшая светлота итога по пикселям: 90-й перцентиль L кольца своей зоны + ``RING_LIGHT_MARGIN``.

    Args:
        small: Итог BGR на масштабе карты.
        ring: Кольцо опоры (bool).
        zones: Метки зон (0 — вне зон).
        fill: Полоса закраса вне книги (bool) — предел ближайшей зоны.
        mm: Плотность на масштабе карты, px/мм.

    Returns:
        Предел L (0..100), float32; вне зон и их полос — 100.
    """
    light = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * (100 / 255)
    limit = np.full(ring.shape, 100.0, np.float32)
    reach = (REF_RING_MM[1] - SHADOW_REACH_MM) * mm + 2
    for k in range(1, int(zones.max()) + 1):
        zone = zones == k
        dist = cv2.distanceTransform((~zone).astype(np.uint8), cv2.DIST_L2, 5)
        own = ring & (dist <= reach)
        if own.sum() < ZONE_MIN_RING_PX:
            continue
        where = zone | (fill & (dist <= FILL_REACH_MM * mm + 2))
        limit[where] = np.minimum(limit[where], np.percentile(light[own], 90) + RING_LIGHT_MARGIN)
    return limit


def zone_colour_gaps(
    small: np.ndarray,
    area: np.ndarray,
    surface: np.ndarray,
    ring: np.ndarray,
    mm: float,
    dropped: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Расхождение цвета опоры и поверхности у пальца по зонам — связным частям области поиска.

    Args:
        small: Итог BGR на масштабе карты.
        area: Область поиска (bool).
        surface: Поверхность у пальца, по которой мерится её цвет (bool): чистая бумага области вне зоны закраса.
        ring: Кольцо опоры (bool).
        mm: Плотность на масштабе карты, px/мм.
        dropped: Отсеянные по цвету пиксели кольца-кандидата (bool) или None: если своего цвета в кольце меньше доли
            ``RING_MIN_SHARE``, опора зоны ненадёжна.

    Returns:
        ``(метки зон, расхождения a*b* по зонам 1..N)``; зона без своего кольца опоры или с ненадёжной — бесконечность
        (не осветлять), без пикселей поверхности — 0.
    """
    count, labels = cv2.connectedComponents(area.astype(np.uint8), connectivity=8)
    ab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 1:].astype(np.float32)
    # Кольцо зоны — пиксели кольца не дальше внешнего радиуса кольца от самой зоны.
    reach = (REF_RING_MM[1] - SHADOW_REACH_MM) * mm + 2
    gaps = []
    for k in range(1, count):
        zone = labels == k
        near = cv2.distanceTransform((~zone).astype(np.uint8), cv2.DIST_L2, 5) <= reach
        own_ring = ring & near
        own_surface = surface & zone
        if own_ring.sum() < ZONE_MIN_RING_PX:
            # Своей опоры нет — R в зоне достроен pull-push издалека, с чужой поверхности (МТС 1990/04 IMG_1112: голубая
            # обложка целиком ушла в «крупную печать» против белого кольца соседней зоны): зона не осветляется.
            gaps.append(float("inf"))
            continue
        if dropped is not None and own_ring.sum() < RING_MIN_SHARE * (own_ring.sum() + (dropped & near).sum()):
            # Своего цвета в кольце меньше половины: палец на стыке поверхностей (белая страница у голубой плашки), опора
            # — обрывки по краю и тянет тон с них (МТС 1990/09 IMG_0310: бурый клин в закрасе) — не осветлять.
            gaps.append(float("inf"))
            continue
        if own_surface.sum() < ZONE_MIN_RING_PX:
            # Поверхности у пальца не видно (всё — закрас или печать): сравнить не с чем, опора своя — зона осветляется.
            gaps.append(0.0)
            continue
        gaps.append(float(np.linalg.norm(np.median(ab[own_ring], axis=0) - np.median(ab[own_surface], axis=0))))
    return labels, tuple(gaps)


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
    # Кольцо-кандидат: чистая бумага книги и полоса закраса чуть за краем листа (RING_OUT_MM) за тенью пальца.
    band = (from_core >= REF_RING_MM[0]) & (from_core <= REF_RING_MM[1])
    outside = ~book_s & disk(book_s, RING_OUT_MM * mm)
    candidates = (clean | outside) & ~area & ~paint_s & band
    ring, ring_dropped, content, own_colour = zone_rings(
        small, area, clean, candidates, book_s, paint_s, original_s, mm
    )
    clean &= ~content
    # Поверхность у пальца — чистая бумага области вне зоны закраса; если её нет, вся чистая бумага области.
    surface = clean & area & ~paint_s
    if surface.sum() < 20:
        surface = clean & area
    zones, gaps = zone_colour_gaps(small, area, surface, ring, mm, ring_dropped)
    # Рабочее окно — рамка области с запасом кольца и сглаживания.
    win = window_of(area, int((REF_RING_MM[1] - SHADOW_REACH_MM + 3 * SMOOTH_SIGMA_MM) * mm) + 2)
    w_img = small[win].astype(np.float32)
    # Закрас до рамки вне книги у области: карта мерится и на нём с той же опорой — его тон задал pull-push от чистой
    # полосы бумаги, и он бывает уже светлым там, где край листа тёмный (МТС 1990/03 IMG_0972): резкая B там следует за
    # ним, и светлый закрас не осветляется.
    # Полоса — в FILL_REACH_MM от «область ∪ зона закраса»: тот же отсчёт, что у спада прибавки в ней (обрезанная раньше
    # спада полоса давала ступеньку в закрасе, МТС 1990/04 IMG_1112).
    fill_origin = area | (paint_s & disk(area, FILL_REACH_MM * mm))
    fill = ~book_s & disk(fill_origin, FILL_REACH_MM * mm)
    # Гладкая B — только по бумаге цвета своей зоны: гаусс 1.5 мм через резкий переход (белый обрез у голубой обложки)
    # давал ложную прибавку по каналам до 68 уровней на голубой стороне перехода.
    soft_weight = clean[win] & own_colour[win] & (from_core[win] <= REF_RING_MM[1])
    sharp_zone = disk(paint_s, SHARP_GROW_MM * mm) & book_s
    back, sharp = paper_as_is(w_img, soft_weight, fill[win], sharp_zone[win], mm)
    ref = smooth_fill(w_img, ring[win], SMOOTH_SIGMA_MM * mm)
    ref = edge_profile_ref(ref, w_img, ring[win], zones[win], book_s[win], mm)
    # Цвет пикселя сравнивается с опорой по самой картинке (медиана 1 мм), а не по размытой B.
    k = max(3, int(round(SHARP_MEDIAN_MM * mm)) | 1)
    own = cv2.medianBlur(small[win], k).astype(np.float32)
    raw = np.maximum(ref - back, 0) * colour_weight(own, ref, (original_s & ~paint_s)[win])[..., None]
    delta = zeros.copy()
    delta[win] = postprocess(raw, area[win], fill[win], book_s[win], sharp, fill_origin[win], mm)
    # Потолок: мягкий по прибавке светлоты и не светлее 90-го перцентиля кольца своей зоны (+ запас).
    delta[win] = cap_lightness(back, delta[win], ring_light_limit(small, ring, zones, fill, mm)[win])
    # Зоны, опора которых взята с другой поверхности (цвет кольца не тот, что у пальца), не осветляются — вместе с полосой
    # закраса у них.
    for k, gap in enumerate(gaps, start=1):
        if gap > ZONE_MAX_AB_GAP:
            delta[disk(zones == k, FILL_REACH_MM * mm + 2)] = 0
    # Кожа не осветляется; вокруг неё прибавка нарастает от нуля на SKIN_FEATHER_MM — без выреза с резким краем.
    if skin.any():
        from_skin = cv2.distanceTransform((~skin).astype(np.uint8), cv2.DIST_L2, 5) / (SKIN_FEATHER_MM * mm)
        delta *= smoothstep(np.clip(from_skin, 0, 1))[..., None]
    ref_full, back_full = zeros.copy(), zeros.copy()
    ref_full[win], back_full[win] = ref, back
    sharp_full = np.zeros(area.shape, np.float32)
    sharp_full[win] = sharp
    return FingerShadow(
        area, skin, ink, content, delta, fill, gaps, ref_full, back_full, sharp_full, px_per_mm, ring, ring_dropped,
        clean & (from_core <= REF_RING_MM[1]), book_s,
    )  # fmt: skip


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
    add = add + sharp_detail(img[y0:y1, x0:x1], shadow, (rows, cols), add)
    # Потолок светлоты и на полном масштабе: уточнение у края пятна добавляет прибавку уже после потолка на ¼.
    add = cap_lightness(img[y0:y1, x0:x1].astype(np.float32), add)
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(img[y0:y1, x0:x1].astype(np.float32) + add + 0.5, 0, 255).astype(np.uint8)
    return out


def sharp_detail(roi: np.ndarray, shadow: FingerShadow, cut: tuple[slice, slice], add: np.ndarray) -> np.ndarray:
    """Уточнение прибавки на полном масштабе в зоне резкой B: край пятна LaMa резче шага карты (¼).

    Прибавка на ¼ — R − B; растянутая билинейно, она размывает край пятна на несколько пикселей, и по контуру пятна
    остаются тонкие кольца. Полномасштабная прибавка — R − B_полн = (R − B)↑ + (B↑ − B_полн): к растянутой карте
    добавляется разность растянутой B и B полного масштаба (тот же pull-push с медианой, по бумаге без краски, кожи и
    печати, порознь по листу и полосе закраса). Только там, где карта не нулевая (зоны, отключённые правилами, и рампы
    остаются как есть).

    Args:
        roi: Рамка итога BGR uint8 (полный масштаб).
        shadow: Карта (с ``back``, ``sharp`` и ``px_per_mm``).
        cut: Срезы рамки на масштабе карты ``(строки, столбцы)``.
        add: Растянутая прибавка в рамке, float32 (h, w, 3).

    Returns:
        Добавка к ``add`` в рамке, float32; итоговая прибавка не опускается ниже нуля.
    """
    if shadow.back is None or shadow.sharp is None or shadow.px_per_mm <= 0 or not shadow.sharp[cut].any():
        return np.zeros_like(add)
    size = (roi.shape[1], roi.shape[0])
    sharp_up = cv2.resize(shadow.sharp[cut], size, interpolation=cv2.INTER_LINEAR)
    back_up = cv2.resize(shadow.back[cut], size, interpolation=cv2.INTER_LINEAR)
    fill = shadow.fill if shadow.fill is not None else np.zeros_like(shadow.area)
    # B на ¼ растягивается по частям с нормировкой, как и карта: билинейно через край листа светлая B полосы закраса
    # подмешивалась к краю листа, и поправка там высветляла до белого.
    for region in (shadow.area & ~fill, fill):
        weight = region[cut].astype(np.float32)
        if not weight.any():
            continue
        num = cv2.resize(shadow.back[cut] * weight[..., None], size, interpolation=cv2.INTER_LINEAR)
        den = cv2.resize(weight, size, interpolation=cv2.INTER_LINEAR)
        part = (cv2.resize(region[cut].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0) & (den > 1e-6)
        back_up[part] = num[part] / den[part][:, None]
    # Кожа и крупная печать — с карты; краска — заново на полном масштабе (см. ниже).
    excluded = (
        cv2.resize((shadow.content | shadow.skin)[cut].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    )
    img = roi.astype(np.float32)
    back_full = back_up.copy()
    parts = [
        cv2.resize(region[cut].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        for region in (shadow.area & ~fill, fill)
    ]
    # Дорогая часть (краска, pull-push, медиана на полном масштабе) — только в рамках связных частей зоны резкой B с
    # запасом на окно закрытия краски: рамка всей карты охватывает оба пальца и в разы больше.
    margin = int(np.ceil((INK_CLOSE_MM + SHARP_MEDIAN_MM) * shadow.px_per_mm))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((sharp_up > 0).astype(np.uint8), connectivity=8)
    for k in range(1, count):
        x, y, w, h = stats[k, :4]
        box = (
            slice(max(0, y - margin), min(size[1], y + h + margin)),
            slice(max(0, x - margin), min(size[0], x + w + margin)),
        )
        # Лист и полоса закраса порознь — как у B на ¼: сглаживание через край листа смешало бы их тон.
        for part_full in parts:
            part = part_full[box]
            if not part.any():
                continue
            # Краска своей части: чужая сторона заполнена pull-push своей. На ¼ и через край листа закрытие окном 3 мм
            # заливало узкий серп пятна LaMa у края (за краем светлый закрас) — серп считался краской, B на нём
            # интерполировалась с обеих сторон, и по ободу пятна оставалась дуга (МТС 1991/02 IMG_0622).
            own = np.clip(pull_push(img[box], part.astype(np.float32)) + 0.5, 0, 255).astype(np.uint8)
            clean = part & ~excluded[box] & ~ink_mask(own, shadow.px_per_mm)
            if clean.sum() < 100:
                continue
            sub = back_full[box]
            sub[part] = sharp_paper(img[box], clean, shadow.px_per_mm, blur=False)[part]
    # Только на ненулевой карте: уровень карты ≥ 2 — полная поправка, к нулю — гладко на нет.
    level = smoothstep(np.clip(add.max(axis=2) / 2.0, 0, 1))
    detail = (sharp_up * level)[..., None] * (back_up - back_full)
    return np.maximum(add + detail, 0) - add
