"""Кромка верхних страниц по полоскам вдоль края книги: от своей печати наружу до первой «не бумаги» (полоски обреза, обложка).

Так край ищет человек: примерная граница разворота известна (маска книги SAM2); у этой границы торчащий обрез виден
как пачка узких полосок вдоль неё (по полоске на лист), край обложки — как полоса шире и другого цвета. Бумага верхней
страницы между печатью и кромкой гладкая и одного цвета. Поэтому от каждой точки края книги берётся луч внутрь,
полоса лучей разворачивается в прямоугольник (точка края × глубина), и по нему, идя от печати к краю, ищется первое
место, где кончается гладкая бумага.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.book import (  # noqa: F401 — ядро, ре-экспорт для стенда
    EDGE_LINE_ASPECT,
    EDGE_LINE_DEG,
    drop_edge_lines,
    content_hulls,
)

# Масштаб поиска относительно кадра: ½ (5813 → 2906 px, ~175 dpi) — полоски обреза ещё различимы.
STRIPE_SCALE = 0.5
# Длина луча внутрь, px масштаба поиска (~170 dpi: 240 px ≈ 35 мм — шире любой стопки торцов и крышки).
RAY_PX = 420
# Сглаживание контура (точек ±) и вынос начала луча наружу, px масштаба поиска.
SMOOTH_HALF = 15
OUTSET_PX = 15
# Шаг точек края книги, px масштаба поиска.
STRIDE = 3
# Окно усреднения вдоль края (точек ±): полоски вдоль края при усреднении остаются, текст и шум — гаснут.
ALONG_HALF = 8
# Полосатость: модуль второй производной серого по глубине после усреднения вдоль края, сглаженный по глубине.
STRIPE_SIGMA = 1.0
# Порог полосатости, уровней серого / px² (подбирается по распределению на бумаге у печати).
STRIPE_THR = 2.5
# Порог цвета: отличие от бумаги своей страницы, ΔE (Lab OpenCV, 8 бит).
COLOUR_THR = 1e9
# Сколько px подряд гладкой бумаги нужно, чтобы считать, что началась верхняя страница (~1.5 мм).
PAPER_RUN_PX = 20
# Опорная бумага страницы — глубже этого от края книги, px масштаба поиска (~25 мм).
REF_DEPTH_PX = 170
# Медиана глубины кромки вдоль края, точек.
EDGE_MEDIAN = 31


@dataclass(frozen=True)
class StripeCut:
    """Итог движка полосок (всё — в масштабе поиска).

    Поля:
        book: Маска книги.
        points: Точки края ``(N, 2)``.
        normals: Внутренние нормали ``(N, 2)``.
        start: Глубина первой печати на луче (или длина луча) ``(N,)``.
        raw_edge: Глубина кромки по лучам до сглаживания ``(N,)``.
        edge: Глубина кромки после медианы ``(N,)``.
        stripe: Полосатость по полосе ``(N, L)``.
        colour: Отличие цвета от бумаги ``(N, L)``.
        band: Развёрнутая полоса BGR ``(N, L, 3)`` — для оверлея.
        top: Маска верхних страниц.
    """

    book: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    start: np.ndarray
    raw_edge: np.ndarray
    edge: np.ndarray
    stripe: np.ndarray
    colour: np.ndarray
    band: np.ndarray
    top: np.ndarray


def contour_rays(book: np.ndarray, stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """Точки внешнего контура книги и внутренние нормали к нему (по градиенту сглаженного расстояния до края).

    Args:
        book: Маска книги.
        stride: Шаг точек контура.

    Returns:
        ``(точки (N, 2) float32, нормали (N, 2) float32)``.
    """
    h, w = book.shape
    contours, _ = cv2.findContours(book.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(contours, key=len)[:, 0, :][::stride].astype(np.float32)
    dist = cv2.GaussianBlur(cv2.distanceTransform(book.astype(np.uint8), cv2.DIST_L2, 5), (0, 0), 6)
    gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=5)
    xi = np.clip(pts[:, 0].astype(int), 0, w - 1)
    yi = np.clip(pts[:, 1].astype(int), 0, h - 1)
    normals = np.stack([gx[yi, xi], gy[yi, xi]], axis=1)
    # Нормали сглаживаются вдоль контура: иначе на зубцах маски соседние лучи перекрещиваются.
    k = 15
    padded = np.r_[normals[-k:], normals, normals[:k]]
    normals = np.stack([np.convolve(padded[:, c], np.ones(2 * k + 1), "valid") for c in range(2)], axis=1)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)
    # Зубцы маски SAM сбивают полоски обреза в развёрнутой полосе: начало луча — сглаженный контур,
    # отодвинутый наружу на OUTSET_PX (луч начинается вне книги и пересекает её край целиком).
    k = SMOOTH_HALF
    padded = np.r_[pts[-k:], pts, pts[:k]]
    pts = np.stack([np.convolve(padded[:, c], np.ones(2 * k + 1) / (2 * k + 1), "valid") for c in range(2)], axis=1)
    pts = (pts - normals * OUTSET_PX).astype(np.float32)
    return pts, normals.astype(np.float32)


def along_mean(a: np.ndarray, half: int) -> np.ndarray:
    """Скользящее среднее по первой оси (вдоль замкнутого контура).

    Args:
        a: Массив ``(N, ...)``.
        half: Полуширина окна.

    Returns:
        Массив той же формы.
    """
    padded = np.concatenate([a[-half:], a, a[:half]], axis=0)
    c = np.cumsum(padded, axis=0, dtype=np.float64)
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), c], axis=0)
    return ((c[2 * half + 1 :] - c[: -2 * half - 1]) / (2 * half + 1)).astype(np.float32)


def stripe_cut(bgr: np.ndarray, book: np.ndarray, content: np.ndarray) -> StripeCut:
    """Кромка верхних страниц: от печати наружу до первой полоски обреза или полосы другого цвета.

    Args:
        bgr: Кадр BGR в масштабе поиска.
        book: Маска книги (bool) в масштабе поиска.
        content: Своя печать с зазором (bool) в масштабе поиска.

    Returns:
        Итог движка.
    """
    h, w = book.shape
    pts, normals = contour_rays(book)
    t = np.arange(RAY_PX, dtype=np.float32)
    xs = pts[:, :1] + normals[:, :1] * t
    ys = pts[:, 1:] + normals[:, 1:] * t
    xs, ys = xs.astype(np.float32), ys.astype(np.float32)
    band = cv2.remap(bgr, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # Опорная бумага — по страницам (левая и правая половины книги): медиана Lab кадра без краски (серое закрытие)
    # в глубине страницы, дальше REF_DEPTH_PX от края книги. Опора у печати или у соседей бралась с белой пачки торца.
    clean = cv2.morphologyEx(bgr, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    lab_img = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB).astype(np.float32)
    deep = cv2.distanceTransform(book.astype(np.uint8), cv2.DIST_L2, 5) > REF_DEPTH_PX
    xs_book = np.nonzero(book.any(axis=0))[0]
    mid_x = (xs_book[0] + xs_book[-1]) / 2
    col = np.arange(w)[None, :]
    # Только оттенок (a, b): яркость бумаги у края падает от изгиба и виньетки, оттенок держится.
    lab_img[..., 0] = 0
    refs = [np.median(lab_img[deep & side], axis=0) for side in (col < mid_x, col >= mid_x)]
    ref = np.where((pts[:, :1] < mid_x), refs[0], refs[1]).astype(np.float32)
    lab_band = cv2.remap(lab_img, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    bg_colour = np.linalg.norm(along_mean(lab_band, ALONG_HALF) - ref[:, None, :], axis=2)
    # Печать считается своей, только если фон вокруг — бумага страницы: полоски обреза защита глифов принимает за строки.
    hits = cv2.remap(content.astype(np.uint8), xs, ys, cv2.INTER_NEAREST, borderValue=0) > 0
    hits &= bg_colour < COLOUR_THR
    inside = cv2.remap(book.astype(np.uint8), xs, ys, cv2.INTER_NEAREST, borderValue=0) > 0
    start = np.where(hits.any(axis=1), hits.argmax(axis=1), RAY_PX).astype(np.int64)
    n = len(pts)
    # Полосатость: серое, усреднённое вдоль края, вторая производная по глубине.
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_gray = along_mean(gray, ALONG_HALF)
    smooth = cv2.GaussianBlur(mean_gray, (0, 0), sigmaX=STRIPE_SIGMA, sigmaY=0.01)
    d2 = np.abs(smooth[:, 2:] - 2 * smooth[:, 1:-1] + smooth[:, :-2])
    stripe = np.pad(d2, ((0, 0), (1, 1)))
    colour = bg_colour
    nonpaper = (stripe > STRIPE_THR) | (colour > COLOUR_THR) | ~inside
    # Снаружи внутрь: через сплошную «не бумагу» (коврик, обложка, полоски обреза) до первого участка гладкой бумаги
    # длиной PAPER_RUN_PX; кромка — его начало. От печати наружу искать нельзя: колонтитул и линейки над текстом —
    # тоже полоски вдоль края, и кромка вставала на них (08/IMG_0188).
    raw = np.zeros(n, np.float32)
    kernel = np.ones(PAPER_RUN_PX)
    for i in range(n):
        paper = ~nonpaper[i, : max(start[i], PAPER_RUN_PX)]
        runs = np.convolve(paper.astype(float), kernel, "valid") >= PAPER_RUN_PX
        where = np.nonzero(runs)[0]
        raw[i] = where[0] if len(where) else start[i]
    k = EDGE_MEDIAN // 2
    padded = np.r_[raw[-k:], raw, raw[:k]]
    edge = np.array([np.median(padded[i : i + EDGE_MEDIAN]) for i in range(n)], np.float32)
    # Маска: многоугольник из точек кромки (начало луча + нормаль × глубина) ∩ книга, плюс своя печать.
    poly = (pts + normals * edge[:, None]).round().astype(np.int32)
    top = np.zeros((h, w), np.uint8)
    cv2.fillPoly(top, [poly], 1)
    top = (top > 0) & book
    top |= content & book
    from research.bound_volumes.layers import largest_filled

    top = largest_filled(top)
    return StripeCut(book, pts, normals, start, raw, edge, stripe, colour, band, top)


# Признак «скачок поверхности» (по разметке пользователя, 7 кадров): на боковой кромке верхней страницы яркость или
# желтизна (b Lab) меняется скачком на 4–8 px — торец белее и холоднее (b ~0 против ~10) либо в тени (150 против 190),
# часто с тонкой тёмной линией по самой кромке; поле страницы между текстом и кромкой желтеет плавно.
# Окно сравнения по обе стороны точки, px масштаба поиска; зазор между окнами, px.
STEP_WIN = 6
STEP_GAP = 1
# Пороги скачка: серое, уровней; b Lab, единиц OpenCV (8 бит).
STEP_GRAY = 10
STEP_B = 3
# Не ближе к своей печати, px (зазор глифов уже входит в печать).
STEP_MIN_FROM_PRINT = 12
# Расширение маски книги перед поиском, px масштаба поиска.
BOOK_GROW_PX = 12
# Кромка по цвету: длина опоры поля у печати, px; отклонение b, единиц; длина подряд, px.
COLOUR_REF_PX = 20
COLOUR_DEV = 3.5
COLOUR_RUN = 8


def step_profile(profile: np.ndarray, win: int = STEP_WIN, gap: int = STEP_GAP) -> np.ndarray:
    """Скачок профиля: модуль разности средних в окнах ``win`` по обе стороны точки (с зазором ``gap``).

    Args:
        profile: Профили ``(N, L)`` (по лучам).
        win: Ширина окна.
        gap: Зазор от точки до окна.

    Returns:
        ``(N, L)`` float32, на краях — 0.
    """
    c = np.cumsum(np.pad(profile, ((0, 0), (1, 0))), axis=1)
    n, length = profile.shape
    out = np.zeros((n, length), np.float32)
    t = np.arange(win + gap, length - win - gap)
    left = (c[:, t - gap] - c[:, t - gap - win]) / win
    right = (c[:, t + gap + win + 1] - c[:, t + gap + 1]) / win
    out[:, t] = np.abs(right - left)
    return out


def step_cut(bgr: np.ndarray, book: np.ndarray, content: np.ndarray) -> StripeCut:
    """Кромка верхних страниц: самый внутренний (ближний к печати) скачок яркости или желтизны на луче.

    Профили усредняются вдоль края (``along_mean``), на каждом луче от первой своей печати наружу ищется первая точка,
    где скачок серого не меньше :data:`STEP_GRAY` или скачок b не меньше :data:`STEP_B`. Кромка — в этой точке; луч без
    печати (край книги без текста рядом) берёт самый внутренний скачок на всей длине.

    Args:
        bgr: Кадр BGR в масштабе поиска.
        book: Маска книги (bool) в масштабе поиска.
        content: Своя печать с зазором (bool) в масштабе поиска.

    Returns:
        Итог движка (поля ``stripe`` — скачок серого, ``colour`` — скачок b).
    """
    # Маска книги SAM считается на ¼ кадра, её край рваный и местами внутри страницы: расширяем, кромку к коврику
    # найдёт скачок.
    k = 2 * BOOK_GROW_PX + 1
    book = cv2.dilate(book.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    h, w = book.shape
    pts, normals = contour_rays(book)
    t = np.arange(RAY_PX, dtype=np.float32)
    xs = (pts[:, :1] + normals[:, :1] * t).astype(np.float32)
    ys = (pts[:, 1:] + normals[:, 1:] * t).astype(np.float32)
    band = cv2.remap(bgr, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lab = along_mean(cv2.cvtColor(band, cv2.COLOR_BGR2LAB).astype(np.float32), ALONG_HALF)
    gray = along_mean(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32), ALONG_HALF)
    sg = step_profile(gray)
    sb = step_profile(lab[..., 2])
    # Предел — оболочка своей печати по страницам: линейки таблиц и колонтитулов в защиту глифов не входят.
    hits = cv2.remap(content_hulls(content, book).astype(np.uint8), xs, ys, cv2.INTER_NEAREST, borderValue=0) > 0
    n = len(pts)
    start = np.where(hits.any(axis=1), hits.argmax(axis=1), RAY_PX).astype(np.int64)
    strong = (sg >= STEP_GRAY) | (sb >= STEP_B)
    raw = np.full(n, np.nan, np.float32)
    for i in range(n):
        lim = max(0, start[i] - STEP_MIN_FROM_PRINT)
        where = np.nonzero(strong[i, :lim])[0]
        if len(where):
            # Самая внутренняя полоса сильного скачка; кромка — максимум силы в ней (на большом скачке порог
            # выполняется на ±(окно + зазор) вокруг кромки, и крайняя точка полосы смещена внутрь на ~7 px).
            run_end = where[-1]
            run_start = run_end
            while run_start - 1 >= 0 and strong[i, run_start - 1]:
                run_start -= 1
            seg = np.maximum(sg[i, run_start : run_end + 1] / STEP_GRAY, sb[i, run_start : run_end + 1] / STEP_B)
            raw[i] = run_start + int(np.argmax(seg))
    # Кромка по цвету: от печати наружу — первое место, где оттенок b уходит от поля у печати больше чем на
    # COLOUR_DEV на COLOUR_RUN px подряд (белый торец холоднее жёлтого поля, грязная крышка — желтее; скачком не
    # видно, переход размыт на 10–15 px). Берётся более внутренняя из двух кромок.
    b_along = lab[..., 2]
    for i in range(n):
        s0 = start[i] - STEP_MIN_FROM_PRINT
        if s0 <= COLOUR_REF_PX:
            continue
        ref = np.median(b_along[i, s0 - COLOUR_REF_PX : s0])
        off = np.abs(b_along[i, :s0] - ref) > COLOUR_DEV
        runs = np.convolve(off.astype(float), np.ones(COLOUR_RUN), "valid") >= COLOUR_RUN
        where = np.nonzero(runs)[0]
        if len(where):
            cand = float(where[-1] + COLOUR_RUN)
            raw[i] = cand if not np.isfinite(raw[i]) else max(raw[i], cand)
    # Лучи без скачка (кромки не видно) берут глубину по соседям: медиана только по найденным.
    k = EDGE_MEDIAN // 2
    padded = np.r_[raw[-k:], raw, raw[:k]]
    edge = np.array([np.nanmedian(padded[i : i + EDGE_MEDIAN]) if np.isfinite(padded[i : i + EDGE_MEDIAN]).any() else 0.0
                     for i in range(n)], np.float32)  # fmt: skip
    raw = np.nan_to_num(raw)
    poly = (pts + normals * edge[:, None]).round().astype(np.int32)
    top = np.zeros((h, w), np.uint8)
    cv2.fillPoly(top, [poly], 1)
    top = ((top > 0) & book) | (drop_edge_lines(content) & book)
    from research.bound_volumes.layers import largest_filled

    return StripeCut(book, pts, normals, start, raw, edge, sg, sb, band, largest_filled(top))


# Линия кромки по скачкам: потолок силы скачка (в долях порога), бонус за глубину на всю длину луча (в тех же долях),
# предел изменения глубины между соседними точками контура (шаг STRIDE), px.
SEAM_CLIP = 3.0
SEAM_BONUS = 1.0
SEAM_STEP = 1








def seam_cut(bgr: np.ndarray, book: np.ndarray, content: np.ndarray) -> StripeCut:
    """Кромка верхних страниц: непрерывная линия по скачкам яркости/желтизны между краем книги и оболочкой печати.

    Сила скачка — ``max(скачок серого / STEP_GRAY, скачок b / STEP_B)``; учитываются точки не слабее порога. Путь по
    (точка контура × глубина) с наибольшей суммой силы плюс бонус за глубину (из параллельных кромок стопки — самая
    внутренняя), соседние глубины отличаются не больше :data:`SEAM_STEP`; за оболочкой печати путь не проходит.

    Args:
        bgr: Кадр BGR в масштабе поиска.
        book: Маска книги (bool).
        content: Своя печать с зазором (bool).

    Returns:
        Итог движка.
    """
    h, w = book.shape
    pts, normals = contour_rays(book)
    t = np.arange(RAY_PX, dtype=np.float32)
    xs = (pts[:, :1] + normals[:, :1] * t).astype(np.float32)
    ys = (pts[:, 1:] + normals[:, 1:] * t).astype(np.float32)
    band = cv2.remap(bgr, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lab = along_mean(cv2.cvtColor(band, cv2.COLOR_BGR2LAB).astype(np.float32), ALONG_HALF)
    gray = along_mean(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32), ALONG_HALF)
    sg, sb = step_profile(gray), step_profile(lab[..., 2])
    hulls = content_hulls(content, book)
    hits = cv2.remap(hulls.astype(np.uint8), xs, ys, cv2.INTER_NEAREST, borderValue=0) > 0
    n, length = len(pts), RAY_PX
    start = np.where(hits.any(axis=1), hits.argmax(axis=1), length).astype(np.int64)
    score = np.maximum(sg / STEP_GRAY, sb / STEP_B)
    strong = score >= 1.0
    gain = np.where(strong, np.minimum(score, SEAM_CLIP) + SEAM_BONUS * t / length, 0.0)
    allowed = t[None, :] <= (start[:, None] - STEP_MIN_FROM_PRINT)
    gain = np.where(allowed, gain, -1e3).astype(np.float32)
    # Динамическое программирование по замкнутому контуру (разрыв в точке 0 — допустим).
    acc = gain[0].copy()
    back = np.zeros((n, length), np.int8)
    offsets = np.arange(-SEAM_STEP, SEAM_STEP + 1)
    for i in range(1, n):
        cand = np.full((len(offsets), length), -np.inf, np.float32)
        for k, o in enumerate(offsets):
            lo, hi = max(0, -o), min(length, length - o)
            cand[k, lo:hi] = acc[lo + o : hi + o]
        best = cand.argmax(axis=0)
        back[i] = offsets[best]
        acc = cand[best, np.arange(length)] + gain[i]
    path = np.zeros(n, np.int64)
    path[-1] = int(acc.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = path[i] + back[i, path[i]]
    edge = path.astype(np.float32)
    poly = (pts + normals * edge[:, None]).round().astype(np.int32)
    top = np.zeros((h, w), np.uint8)
    cv2.fillPoly(top, [poly], 1)
    top = ((top > 0) & book) | (content & book)
    from research.bound_volumes.layers import largest_filled

    return StripeCut(book, pts, normals, start, edge, edge, sg, sb, band, largest_filled(top))
