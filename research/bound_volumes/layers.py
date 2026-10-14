"""Маска верхних страниц по ступенькам глубины MoGe-2: рост от глубины книги до барьера «ступенька слоя»."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from scipy import ndimage

from research.bound_volumes.geometry import Geometry

# Порог оценки SAM2 для маски книги; ниже — берётся лучшая по оценке.
SAM_MIN_SCORE = 0.5
# Тёмный пиксель (коврик) — серый ниже этого; кандидат SAM с долей тёмного выше MAX_DARK_SHARE — не книга.
DARK_GRAY = 45
MAX_DARK_SHARE = 0.10
# Затравка роста: точки книги не ближе этого расстояния к её краю, px рабочего масштаба (~90 dpi, 60 px ≈ 17 мм).
SEED_DEPTH_PX = 60
# Окно медианы глубины перед градиентом, px: гасит шум сетки токенов MoGe.
DEPTH_MEDIAN_PX = 5
# Порог ступеньки: относительный перепад глубины на пиксель (|∇D| / D). Подбирается по распределению на наборе.
STEP_REL = 0.004
# Порог излома нормалей между соседями, градусы на пиксель.
NORMAL_BEND_DEG = 6.0
# Сглаживание нормалей и шаг сравнения для карты излома, px рабочего масштаба.
NORMAL_SIGMA_PX = 1.0
BEND_STEP_PX = 2
# Смыкание барьера, px: замыкает разрывы ступеньки, через которые рост утекает на торец.
BARRIER_CLOSE_PX = 3
# Открытие итоговой маски, px: срезает тонкие языки роста.
OPEN_PX = 5


@dataclass(frozen=True)
class Layers:
    """Итог движка по слоям глубины.

    Поля:
        book: Маска книги SAM2 (весь силуэт, с торцами и крышкой).
        step: Карта ступеньки глубины (|∇D| / D), float32.
        bend: Карта излома нормалей, градусы.
        barrier: Барьер роста (ступенька или излом).
        seed: Затравка роста.
        top: Маска верхних страниц.
    """

    book: np.ndarray
    step: np.ndarray
    bend: np.ndarray
    barrier: np.ndarray
    seed: np.ndarray
    top: np.ndarray


def book_mask(geo: Geometry, gray: np.ndarray | None = None, min_score: float = SAM_MIN_SCORE) -> np.ndarray:
    """Маска книги: самый крупный кандидат SAM2 с оценкой не ниже порога и без коврика, крупнейшая компонента.

    Кандидат, в котором доля тёмных пикселей (коврик) больше :data:`MAX_DARK_SHARE`, отбрасывается: на
    02/IMG_0896 крупнейшая уверенная маска захватила весь коврик вокруг книги.

    Args:
        geo: Геометрия кадра.
        gray: Кадр рабочего масштаба в сером (для доли тёмного); None — без этой проверки.
        min_score: Порог оценки.

    Returns:
        Булева маска.
    """
    masks, scores = geo.sam_masks, geo.sam_scores
    # Без уверенных кандидатов (с отрицательными точками на коврике SAM2 иногда ставит всем оценку 0,
    # 06/IMG_1296) выбор идёт по всем кандидатам.
    good = [i for i in range(len(masks)) if scores[i] >= min_score] or list(range(len(masks)))
    if gray is not None:
        dark = gray < DARK_GRAY
        shares = {i: float(dark[masks[i]].mean()) for i in good}
        # Если коврик захвачен у всех кандидатов (02/IMG_0896: 0.27 и 0.46) — берётся наименее тёмный.
        good = [i for i in good if shares[i] <= MAX_DARK_SHARE] or [min(shares, key=shares.get)]
    areas = masks.reshape(len(masks), -1).mean(axis=1)
    idx = max(good, key=lambda i: areas[i])
    return largest_filled(masks[idx])


def largest_filled(mask: np.ndarray) -> np.ndarray:
    """Крупнейшая компонента маски с залитыми дырами.

    Args:
        mask: Булева маска.

    Returns:
        Булева маска (пустая, если на входе пусто).
    """
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask.copy()
    sizes = ndimage.sum(mask, labels, index=range(1, count + 1))
    return ndimage.binary_fill_holes(labels == 1 + int(np.argmax(sizes)))


def step_map(depth: np.ndarray, median_px: int = DEPTH_MEDIAN_PX) -> np.ndarray:
    """Относительная ступенька глубины: модуль градиента сглаженной глубины, делённый на глубину.

    Args:
        depth: Глубина (NaN вне валидной области).
        median_px: Окно медианы.

    Returns:
        Карта float32 (0 там, где глубины нет).
    """
    valid = np.isfinite(depth)
    fill = np.where(valid, depth, np.nanmedian(depth)).astype(np.float32)
    smooth = cv2.medianBlur(fill, median_px)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8
    return np.where(valid, np.hypot(gx, gy) / np.maximum(smooth, 1e-6), 0).astype(np.float32)


def bend_map(normal: np.ndarray, sigma: float = NORMAL_SIGMA_PX, step: int = BEND_STEP_PX) -> np.ndarray:
    """Излом нормалей: наибольший угол между сглаженными нормалями на расстоянии ``step`` по осям, градусы.

    Шаг в несколько px и сглаживание поднимают излом на кромке торца (2–4° на соседних пикселях) над шумом
    сетки MoGe внутри страницы.

    Args:
        normal: Нормали ``(H, W, 3)``.
        sigma: Гауссово сглаживание компонент нормали, px.
        step: Расстояние между сравниваемыми пикселями, px.

    Returns:
        Карта float32 (угол приписан пикселю посередине пары).
    """
    n = np.stack([cv2.GaussianBlur(normal[..., c], (0, 0), sigma) for c in range(3)], axis=-1)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    half = step // 2
    out = np.zeros(n.shape[:2], np.float32)
    right = np.degrees(np.arccos(np.clip((n[:, :-step] * n[:, step:]).sum(-1), -1, 1)))
    down = np.degrees(np.arccos(np.clip((n[:-step] * n[step:]).sum(-1), -1, 1)))
    out[:, half : half + right.shape[1]] = np.maximum(out[:, half : half + right.shape[1]], right)
    out[half : half + down.shape[0]] = np.maximum(out[half : half + down.shape[0]], down)
    return out


def grow_top(
    geo: Geometry, step_rel: float = STEP_REL, bend_deg: float = NORMAL_BEND_DEG, seed_px: int = SEED_DEPTH_PX
) -> Layers:
    """Верхние страницы: связная область книги без барьера, содержащая глубину книги.

    Торец стопки и край крышки лежат ниже верхней страницы: на их границе глубина делает ступеньку, а нормаль —
    излом. Эти пиксели — барьер; рост из середины книги до барьера даёт верхний слой.

    Args:
        geo: Геометрия кадра.
        step_rel: Порог ступеньки глубины.
        bend_deg: Порог излома нормалей.
        seed_px: Глубина затравки от края книги, px.

    Returns:
        Карты и маски движка.
    """
    book = book_mask(geo)
    step = step_map(geo.depth)
    bend = bend_map(geo.normal)
    barrier = (step > step_rel) | (bend > bend_deg)
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BARRIER_CLOSE_PX + 1,) * 2)
    barrier = cv2.morphologyEx(barrier.astype(np.uint8), cv2.MORPH_CLOSE, close) > 0
    dist = cv2.distanceTransform(book.astype(np.uint8), cv2.DIST_L2, 5)
    seed = (dist > seed_px) & ~barrier
    labels, _ = ndimage.label(book & ~barrier)
    keep = np.unique(labels[seed])
    top = np.isin(labels, keep[keep > 0])
    # Барьер внутри страницы (корешок, складка) не должен дырявить маску; тонкие языки срезаются открытием.
    # Корешок делит разворот на две страницы — берутся все компоненты с затравкой, не только крупнейшая.
    opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * OPEN_PX + 1,) * 2)
    top = cv2.morphologyEx(top.astype(np.uint8), cv2.MORPH_OPEN, opening) > 0
    # Щель барьера по корешку закрывается: страницы снова одна область, как и просил пользователь.
    top = cv2.morphologyEx(top.astype(np.uint8), cv2.MORPH_CLOSE, close) > 0
    labels, _ = ndimage.label(top)
    keep = np.unique(labels[seed & top])
    top = ndimage.binary_fill_holes(np.isin(labels, keep[keep > 0]))
    return Layers(book, step, bend, barrier, seed, top)


# Лучи внутрь книги: длина, px рабочего масштаба (~3 px/мм: торцы стопки и крышка уже 30 мм), шаг точек контура,
# порог пика излома, градусы, окно медианы глубины среза вдоль контура, точек.
RAY_MAX_PX = 100
RAY_STRIDE = 2
PEAK_DEG = 2.0
CUT_MEDIAN = 9
# Линия кромки (seam_depth): бонус за глубину, градусы на длину луча; шаг глубины между соседними точками, px;
# потолок излома, градусы (одна резкая кромка не перевешивает длинную линию).
SEAM_INNER_BONUS = 1.5
SEAM_MAX_STEP = 1
SEAM_CLIP_DEG = 6.0
# Перевод перепада яркости (уровней серого на px) в шкалу излома: 8 уровней/px ≈ 2° — порог кромки.
IMAGE_DEG = 0.25


class EdgeSource(str, Enum):
    """Чем мерить кромку на луче."""

    # Излом нормалей MoGe-2.
    NORMALS = "normals"
    # Перепад яркости кадра.
    IMAGE = "image"


@dataclass(frozen=True)
class RayCut:
    """Итог движка лучей.

    Поля:
        book: Маска книги SAM2.
        bend: Карта излома нормалей.
        points: Точки контура книги ``(N, 2)`` (x, y).
        normals: Внутренние нормали в точках ``(N, 2)``.
        raw_depth: Глубина самого внутреннего пика по лучу, px (0 — пика нет).
        depth: Глубина среза после медианы вдоль контура, px.
        top: Маска верхних страниц.
    """

    book: np.ndarray
    bend: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    raw_depth: np.ndarray
    depth: np.ndarray
    top: np.ndarray


def innermost_peak(profile: np.ndarray, threshold: float) -> int:
    """Глубина самого внутреннего локального максимума профиля выше порога.

    Args:
        profile: Излом вдоль луча от края книги внутрь.
        threshold: Порог пика.

    Returns:
        Индекс пика или 0, если пиков нет.
    """
    p = profile
    peaks = np.nonzero((p[1:-1] >= threshold) & (p[1:-1] >= p[:-2]) & (p[1:-1] >= p[2:]))[0] + 1
    return int(peaks[-1]) if len(peaks) else 0


def seam_depth(profiles: np.ndarray, inner_bonus: float = SEAM_INNER_BONUS, max_step: int = SEAM_MAX_STEP) -> np.ndarray:
    """Непрерывная линия кромки по профилям излома: путь по (точка контура × глубина) с наибольшей суммой излома.

    Динамическое программирование: глубина соседних точек контура отличается не больше чем на ``max_step``,
    к излому прибавляется бонус за глубину (``inner_bonus`` градусов на всю длину луча) — из параллельных кромок
    стопки выбирается самая внутренняя, то есть край верхней страницы.

    Args:
        profiles: Излом вдоль лучей ``(N, L)``, градусы (за печатью — 0).
        inner_bonus: Бонус за глубину на всю длину луча, градусы.
        max_step: Предел изменения глубины между соседними точками, px.

    Returns:
        Глубина кромки по точкам ``(N,)`` float32.
    """
    n, length = profiles.shape
    # Бонус за глубину — только на кромках (излом не ниже PEAK_DEG): на гладкой бумаге шум излома ~0.2°, и
    # бонус там уводил линию в текст (06/IMG_1296).
    edge = profiles >= PEAK_DEG
    gain = np.clip(profiles, 0, SEAM_CLIP_DEG) * edge + inner_bonus * np.arange(length) / length * edge
    score = gain[0].copy()
    back = np.zeros((n, length), np.int16)
    offsets = np.arange(-max_step, max_step + 1)
    for i in range(1, n):
        cand = np.full((len(offsets), length), -np.inf, np.float32)
        for k, o in enumerate(offsets):
            lo, hi = max(0, -o), min(length, length - o)
            cand[k, lo:hi] = score[lo + o : hi + o]
        best = cand.argmax(axis=0)
        back[i] = offsets[best]
        score = cand[best, np.arange(length)] + gain[i]
    path = np.zeros(n, np.int64)
    path[-1] = int(score.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = path[i] + back[i, path[i]]
    return path.astype(np.float32)


def ray_cut(
    geo: Geometry, gray: np.ndarray | None = None, content: np.ndarray | None = None, edge_source: EdgeSource = EdgeSource.NORMALS, ray_px: int = RAY_MAX_PX, peak_deg: float = PEAK_DEG) -> RayCut:
    """Верхние страницы: от каждой точки края книги срез до самой внутренней кромки (пика излома нормалей).

    Торцы стопки, край крышки и сама кромка верхней страницы дают на луче внутрь несколько пиков излома нормалей;
    кромка верхней страницы — самый внутренний из них. Глубина среза сглаживается медианой вдоль контура и
    разносится на пиксели книги по ближайшей точке контура.

    Args:
        geo: Геометрия кадра.
        gray: Кадр рабочего масштаба в сером (выбор маски книги) или None.
        content: Своя печать (блоки текста с зазором, bool) рабочего масштаба или None. Луч ищет кромку только до
            первого пикселя печати, а итог обязательно включает печать: срез не заходит на текст.
        edge_source: Чем мерить кромку на луче.
        ray_px: Длина луча, px.
        peak_deg: Порог пика излома, градусы.

    Returns:
        Карты, лучи и маска верхних страниц.
    """
    book = book_mask(geo, gray)
    bend = bend_map(geo.normal)
    h, w = book.shape
    contours, _ = cv2.findContours(book.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)[:, 0, :][::RAY_STRIDE].astype(np.float32)
    # Внутренняя нормаль — градиент сглаженного расстояния до края книги.
    dist = cv2.GaussianBlur(cv2.distanceTransform(book.astype(np.uint8), cv2.DIST_L2, 5), (0, 0), 3)
    gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=5)
    xi = np.clip(contour[:, 0].astype(int), 0, w - 1)
    yi = np.clip(contour[:, 1].astype(int), 0, h - 1)
    normals = np.stack([gx[yi, xi], gy[yi, xi]], axis=1)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)
    t = np.arange(ray_px, dtype=np.float32)
    xs = contour[:, :1] + normals[:, :1] * t
    ys = contour[:, 1:] + normals[:, 1:] * t
    if edge_source == EdgeSource.IMAGE and gray is not None:
        # Перепад яркости вдоль луча (модуль), в «градусах» шкалы излома: 1 уровень серого на px = IMAGE_DEG.
        g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
        along = cv2.remap(g, xs, ys, cv2.INTER_LINEAR, borderValue=0)
        profiles = np.abs(np.gradient(along, axis=1)) * IMAGE_DEG
    else:
        profiles = cv2.remap(bend, xs, ys, cv2.INTER_LINEAR, borderValue=0)
    if content is not None:
        # За первым пикселем печати на луче кромку не ищем: дальше — сама страница.
        hits = cv2.remap(content.astype(np.float32), xs, ys, cv2.INTER_NEAREST, borderValue=0) > 0.5
        first = np.where(hits.any(axis=1), hits.argmax(axis=1), ray_px)
        profiles = np.where(np.arange(ray_px)[None, :] < first[:, None], profiles, 0)
    raw = np.array([innermost_peak(p, peak_deg) for p in profiles], np.float32)
    seam = seam_depth(profiles)
    # Медиана по замкнутому контуру.
    k = CUT_MEDIAN // 2
    padded = np.r_[seam[-k:], seam, seam[:k]]
    depth = np.array([np.median(padded[i : i + CUT_MEDIAN]) for i in range(len(raw))], np.float32)
    # Глубина среза на пиксели книги: по ближайшей точке контура.
    seeds = np.full((h, w), 255, np.uint8)
    seeds[yi, xi] = 0
    _, labels = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    label_depth = np.zeros(labels.max() + 1, np.float32)
    label_depth[labels[yi, xi]] = depth
    cut = label_depth[labels]
    top = book & (cv2.distanceTransform(book.astype(np.uint8), cv2.DIST_L2, 5) > cut)
    if content is not None:
        top |= content & book
    top = largest_filled(top)
    return RayCut(book, bend, contour, normals, raw, depth, top)
