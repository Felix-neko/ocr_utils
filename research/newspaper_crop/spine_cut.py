"""Отрезка за линией сгиба: складка у стороны выпрямленной полосы (детектор ``spread_defects.fold_line``), всё за ней — не контент."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from research.newspaper_crop.fill import quad_homography
from research.newspaper_crop.ink_guard import text_lines
from research.newspaper_crop.zone import Zone, zone_of
from research.spread_defects.fold_line import (
    COLUMN_STEP,
    Assembly,
    Candidate,
    SpineSide,
    Variant,
    apply_boxes,
    best_path,
    column_levels,
    column_points,
    fit_ransac,
    points_from_spine_top,
    response_map,
    smooth_path,
    to_spine_top,
)

# Вариант детектора складки: минимум уровня бумаги (ловит и тонкую складку, и широкую бурую полосу торца и завала
# нашего листа, к которому вплотную прилегает соседняя полоса — 1969 DSCF0033/0131/0158, где тонкой линии нет),
# путь минимальной стоимости (держит волну валика у корешка, RANSAC-парабола на нём теряет опору); строки печати
# и цветная печать — запрет. Выбран на разметке ``sets/spine_gt.csv``: A+3 — 11/13 чужих полос, 0 ложных из 202;
# BC+3 — 7/13; C+2 (как ЭГ 1985) — 6/13.
FOLD_VARIANT = Variant(Candidate.MINIMUM, topmost=False, paper_below=False, clips=False, assembly=Assembly.PATH)
# Складка «рядом с краем»: глубже от стороны полосы, px превью (~4 см бумаги газеты при ~70 px/см) — это уже не
# корешок, а линейка или рамка внутри полосы.
MAX_DEPTH = 300
# Складка не на самой кромке: мельче — это край листа или наши же буквы, уходящие в корешок (2_1540: «складка»
# в 11 px от края по концам строк), px превью.
MIN_DEPTH = 20
# Доля столбцов стороны, где точки-кандидаты согласны с линией (≤ RANSAC_TOL), — складка идёт вдоль всей стороны.
MIN_SUPPORT = 0.45
RANSAC_TOL = 6.0
# Столбец-полоса «опирает» путь, если нормированный отклик на нём не ниже порога.
MIN_POINT_RESPONSE = 0.3
# Наибольший прогиб линии на длину стороны, px: складка почти прямая.
MAX_SAG = 60.0
# Сдвиг цвета линии от бумаги рядом (a*, b* Lab, OpenCV 8 бит): тень складки — того же тона, что бумага; красная,
# синяя, зелёная линейка рамки печати — другого.
MAX_AB_SHIFT = 14.0
# Пиксель «цветной»: хрома не меньше порога и тон отходит от тона бумаги больше чем на угол, градусы; клетка
# (строка × столбец-полоса) — если таких больше доли.
COLOR_CHROMA, COLOR_ANGLE, COLOR_SHARE = 12.0, 40.0, 0.25
# Нейтральное (чёрная печать) не запрещается: глубокая тень в складке тоже теряет тон бумаги (DSCF0105, DSCF0131).
# Полоса бумаги внутри от линии (от, до), px, и зазор за линией, px.
INSIDE_BAND, BEYOND_GAP = (6, 40), 4
# За линией должна быть соседняя полоса: её бумага обращена к камере и освещена не хуже нашей (уровень за линией
# к уровню внутри не ниже порога), и на ней есть краска. Завал нашего края в тень (внешний край, низ) темнее и
# без печати — его не отрезаем. Пороги — по разметке 57 кадров ``sets/spine_gt.csv``.
MIN_BEYOND_LEVEL, MIN_BEYOND_INK = 1.0, 0.04
# Глубокая линия (не ближе, px превью) годится, только если печати за ней не меньше доли: по всем 936 кадрам настоящие
# глубокие полосы — от 0.107, ложные (линейка внутри нашей полосы) — 0.042–0.049.
DEEP_DEPTH, DEEP_MIN_INK = 150.0, 0.08
# Стороны, где бывает корешок: съёмка постраничная, полоса стоит вертикально — корешок слева или справа. Верх и низ
# не режутся: глубокие «складки» там — наше же пустое поле над шапкой (2_0640, 2_2160, DSCF0132 по всей папке).
SPINE_SIDES = (SpineSide.LEFT, SpineSide.RIGHT)
# Широкая соседняя полоса в тени: линия не мельче, px превью, краски за ней не меньше доли, уровень не ниже доли нашего.
# На выборке: чужие полосы в тени — 109 и 161 px, наш завал с печатью — не глубже 67 px.
WIDE_DEPTH, WIDE_MIN_INK, WIDE_MIN_LEVEL = 100.0, 0.10, 0.85
# Наименьший средний нормированный отклик хребта вдоль линии.
MIN_RIDGE = 0.25
# Полоса строк печати для запрета, px превью: рамки строк расширяются на столько, чтобы складка не прошла вплотную.
LINE_PAD = 4


@dataclass(frozen=True)
class SideFold:
    """Складка у одной стороны выпрямленной полосы.

    Attributes:
        side: Сторона полосы.
        accepted: Складка принята: за ней — не контент, отрезать.
        reason: Почему не принята (пусто, если принята).
        depth_min: Наименьшая глубина линии от стороны, px выпрямленной полосы.
        depth_max: Наибольшая глубина линии от стороны.
        support: Доля столбцов, согласных с линией.
        ridge: Средний нормированный отклик хребта вдоль линии.
        sat: Медиана насыщенности под линией.
        ab_shift: Сдвиг цвета линии от бумаги рядом внутри (расстояние по a*, b* Lab): цветная линейка — большой.
        beyond_level: Уровень бумаги за линией к уровню внутри (80-й перцентиль серого): загиб в тени — меньше 1.
        beyond_ink: Доля краски (адаптивный порог) за линией.
        points: (N, 2) точки линии в координатах выпрямленной полосы (x, y).
    """

    side: SpineSide
    accepted: bool
    reason: str
    depth_min: float
    depth_max: float
    support: float
    ridge: float
    sat: float
    ab_shift: float
    beyond_level: float
    beyond_ink: float
    points: np.ndarray


def rectify(bgr: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Выпрямленная полоса (гомография четырёхугольника) и матрица.

    Args:
        bgr: Кадр BGR.
        quad: Четырёхугольник в пикселях кадра.

    Returns:
        (выпрямленная полоса BGR, гомография кадр → полоса).
    """
    m, w, h = quad_homography(quad)
    return cv2.warpPerspective(bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), m


def line_boxes(rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Рамки строк печати (обе ориентации набора) в пикселях картинки — запрет пересечения для складки.

    Args:
        rgb: Картинка RGB (выпрямленная полоса).

    Returns:
        Рамки ``(x0, y0, x1, y1)``.
    """
    mask = text_lines(rgb, rgb.shape[:2])
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for x, y, w, h, _ in stats[1:]:
        out.append((x - LINE_PAD, y - LINE_PAD, x + w + LINE_PAD, y + h + LINE_PAD))
    return out


def colored_cells(canon: np.ndarray, levels) -> np.ndarray:
    """Клетки (строка × столбец-полоса ``COLUMN_STEP``) верхней зоны, где заметная доля пикселей другого цвета, чем бумага.

    Цветная печать (линейки и рамки, плашки) по тону отличается от бумаги, тень складки — нет.

    Args:
        canon: Полоса RGB, сторона сверху.
        levels: ``column_levels`` той же полосы.

    Returns:
        Булев массив формы ``levels.level``.
    """
    rows, cols = levels.level.shape
    top = canon[:rows, : cols * COLUMN_STEP]
    lab = cv2.cvtColor(top, cv2.COLOR_RGB2LAB).astype(np.float32)
    light = lab[..., 0] >= np.percentile(lab[..., 0], 60)
    # a*, b* относительно нейтрали (в OpenCV 8 бит нейтраль — 128): вектор тона.
    ab = lab[..., 1:] - 128.0
    paper = np.median(ab[light], axis=0)
    chroma = np.linalg.norm(ab, axis=2)
    # Угол между тоном пикселя и тоном бумаги: тень складки темнее, но того же тона (жёлто-бурая), красная, синяя,
    # зелёная печать — другого; серая и чёрная краска почти без хромы — не цветная.
    paper_chroma = float(np.linalg.norm(paper))
    cos = (ab @ paper) / (chroma * paper_chroma + 1e-6)
    colored = (chroma > COLOR_CHROMA) & (cos < np.cos(np.radians(COLOR_ANGLE)))
    share = colored.reshape(rows, cols, COLUMN_STEP).mean(axis=2)
    return share > COLOR_SHARE


def fold_rows(
    canon: np.ndarray, levels, boxes: list[tuple[float, float, float, float]], variant: Variant
) -> tuple[np.ndarray, float, float]:
    """Линия складки у верхней стороны (как ``fold_line.find_line``), без цветной печати и строк, и её опора.

    Args:
        canon: Полоса RGB, сторона сверху.
        levels: ``column_levels``.
        boxes: Рамки строк печати в канонической системе (запрет пересечения).
        variant: Отклик и сборка (RANSAC-парабола или путь минимальной стоимости, который держит волну валика).

    Returns:
        (строка линии на каждый столбец полосы — NaN, если собрать не из чего; опора — доля столбцов-полос,
        где отклик на линии не ниже :data:`MIN_POINT_RESPONSE` (для RANSAC — доля точек в допуске); средний
        нормированный отклик вдоль линии).
    """
    response = response_map(levels, variant.candidate, variant.paper_below)
    response = apply_boxes(response, levels.xs, boxes)
    response[colored_cells(canon, levels)] = 0.0
    width = canon.shape[1]
    grid = np.arange(width)
    cols = np.arange(len(levels.xs))
    if variant.assembly in (Assembly.PATH, Assembly.PATH_SMOOTH):
        path = best_path(response).astype(float)
        if variant.assembly is Assembly.PATH_SMOOTH:
            path = smooth_path(levels.xs, path)
        rows = np.interp(grid, levels.xs, path)
        at = response[np.clip(np.round(path).astype(int), 0, response.shape[0] - 1), cols]
        return rows, float(np.mean(at >= MIN_POINT_RESPONSE)), float(np.mean(np.maximum(at, 0)))
    px, py = column_points(response, levels, variant.topmost, variant.clips)
    coef = fit_ransac(px, py)
    if coef is None:
        return np.full(width, np.nan), 0.0, 0.0
    rows = np.polyval(coef, grid)
    near = np.abs(np.polyval(coef, px) - py) <= RANSAC_TOL
    ys = np.clip(np.round(np.interp(levels.xs, grid, rows)).astype(int), 0, response.shape[0] - 1)
    return rows, float(near.sum() / max(len(cols), 1)), float(np.mean(np.maximum(response[ys, cols], 0)))


def neighbor_beyond(level: float, ink: float, depth_min: float) -> bool:
    """За линией соседняя полоса: освещённая не хуже нашей и с печатью — или широкая затенённая полоса с густой печатью.

    Соседняя полоса, лежащая в тени (2_3190, 1_0960), темнее нашей, но широкая (линия глубже :data:`WIDE_DEPTH`) и
    густо напечатана; наш собственный завал края в тень — узкий (≤ 90 px на выборке) и прижат к кромке.

    Args:
        level: Уровень бумаги за линией к уровню внутри.
        ink: Доля краски за линией.
        depth_min: Наименьшая глубина линии от стороны, px.

    Returns:
        True — отрезать.
    """
    if depth_min >= DEEP_DEPTH and ink < DEEP_MIN_INK:
        # Глубокая линия с редкой печатью за ней — внутренняя линейка или граница блока нашей полосы (2_0850, DSCF0005).
        return False
    lit = level >= MIN_BEYOND_LEVEL and ink >= MIN_BEYOND_INK
    wide_shaded = depth_min >= WIDE_DEPTH and ink >= WIDE_MIN_INK and level >= WIDE_MIN_LEVEL
    return lit or wide_shaded


def find_side_fold(
    rgb: np.ndarray, side: SpineSide, boxes: list[tuple[float, float, float, float]], variant: Variant | None = None
) -> SideFold:
    """Складка у стороны ``side`` выпрямленной полосы и решение «отрезать или нет».

    Args:
        rgb: Выпрямленная полоса RGB.
        side: Сторона.
        boxes: Рамки строк печати в координатах полосы.
        variant: Вариант детектора складки; None — :data:`FOLD_VARIANT`.

    Returns:
        ``SideFold``.
    """
    from research.spread_defects.fold_line import boxes_to_spine_top

    canon = to_spine_top(rgb, side)
    canon_boxes = boxes_to_spine_top(boxes, side, rgb.shape[:2])
    levels = column_levels(canon)
    variant = variant or FOLD_VARIANT
    rows, support, ridge = fold_rows(canon, levels, canon_boxes, variant)
    width = canon.shape[1]
    empty = np.empty((0, 2))
    if not np.isfinite(rows).all():
        return SideFold(side, False, "нет линии", 0, 0, 0, 0, 0, 0, 0, 0, empty)
    # Насыщенность под линией.
    hsv = cv2.cvtColor(canon, cv2.COLOR_RGB2HSV)
    yy = np.clip(np.round(rows).astype(int), 0, canon.shape[0] - 1)
    sat = float(np.median(hsv[yy, np.arange(width), 1]))
    ab_shift, beyond_level, beyond_ink = _across_line(canon, rows)
    depth_min, depth_max = float(rows.min()), float(rows.max())
    sag = float(np.ptp(rows - np.polyval(np.polyfit(np.arange(width), rows, 1), np.arange(width))))
    xs = np.arange(width, dtype=float)
    ux, uy = points_from_spine_top(xs, rows, side, rgb.shape[:2])
    points = np.stack([ux, uy], axis=1)
    reason = ""
    if depth_max > MAX_DEPTH:
        reason = f"глубоко ({depth_max:.0f} px)"
    elif float(np.median(rows)) < MIN_DEPTH:
        # Медиана, а не минимум: путь у концов стороны может прижиматься к кромке.
        reason = f"на кромке (медиана {float(np.median(rows)):.0f} px)"
    elif support < MIN_SUPPORT:
        reason = f"мало опоры ({support:.2f})"
    elif variant.assembly is Assembly.RANSAC and sag > MAX_SAG:
        reason = f"кривая ({sag:.0f} px)"
    elif ab_shift > MAX_AB_SHIFT:
        reason = f"цветная линейка (Δab {ab_shift:.0f})"
    elif ridge < MIN_RIDGE:
        reason = f"слабый хребет ({ridge:.2f})"
    elif not neighbor_beyond(beyond_level, beyond_ink, float(rows.min())):
        reason = f"за линией нет чужой полосы (уровень {beyond_level:.2f}, краска {beyond_ink:.3f})"
    return SideFold(
        side, not reason, reason, depth_min, depth_max, support, ridge, sat, ab_shift, beyond_level, beyond_ink, points
    )


def _across_line(canon: np.ndarray, rows: np.ndarray) -> tuple[float, float, float]:
    """Признаки поперёк линии в канонической системе (сторона сверху): цвет линии, уровень и краска за ней.

    Args:
        canon: Полоса RGB, сторона сверху.
        rows: Строка линии для каждого столбца.

    Returns:
        (сдвиг a*b* линии от бумаги внутри, уровень бумаги за линией к уровню внутри, доля краски за линией).
    """
    h, w = canon.shape[:2]
    lab = cv2.cvtColor(canon, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(canon, cv2.COLOR_RGB2GRAY)
    ink = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 18) > 0
    yy, xx = np.mgrid[0:h, 0:w]
    r = rows[None, :]
    on_line = np.abs(yy - r) <= 1
    inside = (yy > r + INSIDE_BAND[0]) & (yy <= r + INSIDE_BAND[1])
    beyond = (yy < r - BEYOND_GAP) & (yy >= 0)
    # Цвет — по тёмным пикселям линии (сама линия), бумага — по светлым внутри (без букв).
    line_ab = np.median(lab[on_line][:, 1:], axis=0)
    inside_l = lab[..., 0][inside]
    paper = lab[inside][inside_l >= np.percentile(inside_l, 50)]
    paper_ab = np.median(paper[:, 1:], axis=0)
    ab_shift = float(np.linalg.norm(line_ab - paper_ab))
    level_in = float(np.percentile(gray[inside], 80))
    if beyond.sum() < 50:
        return ab_shift, 1.0, 0.0
    level_out = float(np.percentile(gray[beyond], 80))
    return ab_shift, level_out / max(level_in, 1.0), float(ink[beyond].mean())


def side_folds(rgb: np.ndarray, variant: Variant | None = None) -> list[SideFold]:
    """Складки у боковых сторон выпрямленной полосы (:data:`SPINE_SIDES`).

    Args:
        rgb: Выпрямленная полоса RGB.
        variant: Вариант детектора складки; None — :data:`FOLD_VARIANT`.

    Returns:
        По ``SideFold`` на сторону.
    """
    boxes = line_boxes(rgb)
    return [find_side_fold(rgb, side, boxes, variant) for side in SPINE_SIDES]


def beyond_mask(shape: tuple[int, int], fold: SideFold) -> np.ndarray:
    """Маска «за складкой» (между линией и стороной полосы) в координатах выпрямленной полосы.

    Args:
        shape: (H, W) полосы.
        fold: Принятая складка.

    Returns:
        Булева маска.
    """
    h, w = shape
    pts = fold.points
    corners = {
        SpineSide.TOP: [[w - 1, 0], [0, 0]],
        SpineSide.BOTTOM: [[w - 1, h - 1], [0, h - 1]],
        SpineSide.LEFT: [[0, h - 1], [0, 0]],
        SpineSide.RIGHT: [[w - 1, h - 1], [w - 1, 0]],
    }[fold.side]
    # Точки линии идут по стороне; замыкаем многоугольник через углы этой стороны в нужном порядке.
    first, last = pts[0], pts[-1]
    c0, c1 = np.array(corners[0], float), np.array(corners[1], float)
    if np.linalg.norm(last - c0) > np.linalg.norm(last - c1):
        c0, c1 = c1, c0
    poly = np.vstack([pts, [c0], [c1]])
    if np.linalg.norm(c1 - first) > np.linalg.norm(c0 - first):
        poly = np.vstack([pts, [c1], [c0]])
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def straight_cut(fold: SideFold, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Новая сторона четырёхугольника по складке: проходит по самой ВНЕШНЕЙ точке линии, параллельно её общему ходу.

    Наш лист целиком остаётся внутри (прямая по самой внутренней точке на выгнутой складке срезала концы наших строк:
    2_2060 «ЦК проф» без «ф»); узкий клин чужой полосы между прямой и изогнутой складкой убран из маски листа
    (:func:`beyond_mask`) и дорисовывается.

    Args:
        fold: Принятая складка.
        shape: (H, W) полосы.

    Returns:
        Две точки прямой (x, y) в координатах полосы, на концах стороны.
    """
    pts = fold.points
    horizontal = fold.side in (SpineSide.TOP, SpineSide.BOTTOM)
    t, d = (pts[:, 0], pts[:, 1]) if horizontal else (pts[:, 1], pts[:, 0])
    k, b = np.polyfit(t, d, 1)
    residual = d - (k * t + b)
    # Сдвиг наружу (к стороне) до самой внешней точки складки.
    inward = {SpineSide.TOP: 1, SpineSide.LEFT: 1, SpineSide.BOTTOM: -1, SpineSide.RIGHT: -1}[fold.side]
    b += float(residual.min() if inward > 0 else residual.max())
    t0, t1 = 0.0, float(shape[1] - 1 if horizontal else shape[0] - 1)
    p0, p1 = (t0, k * t0 + b), (t1, k * t1 + b)
    if not horizontal:
        p0, p1 = (p0[1], p0[0]), (p1[1], p1[0])
    return np.array(p0), np.array(p1)


@dataclass(frozen=True)
class SpineCut:
    """Зона после отрезки за складками: новая зона и складки по сторонам (принятые — с линией в координатах маски).

    Attributes:
        zone: Зона на рабочем масштабе с перенесёнными сторонами и маской листа без полосок за складками.
        folds: Складки всех сторон выпрямленной полосы.
        lines: Принятые складки — ломаные (N, 2) в пикселях рабочей маски.
    """

    zone: Zone
    folds: list[SideFold]
    lines: list[np.ndarray]


def cut_zone(bgr: np.ndarray, zone: Zone, variant: Variant | None = None) -> SpineCut:
    """Отрезает от зоны всё, что за принятыми складками у сторон: сторона четырёхугольника — на прямую по самой
    внутренней точке складки, лист — без полоски за складкой (она уходит в дорисовку).

    Args:
        bgr: Кадр BGR (превью; масштаб любой — зона пересчитывается).
        zone: Зона на рабочем масштабе (с четырёхугольником).
        variant: Вариант детектора складки; None — :data:`FOLD_VARIANT`.

    Returns:
        ``SpineCut`` (если складок нет — зона прежняя).
    """
    if zone.quad is None:
        return SpineCut(zone, [], [])
    h_work, w_work = zone.paper.shape
    scale = bgr.shape[0] / h_work
    quad_px = zone.quad * scale
    page, m = rectify(bgr, quad_px)
    rgb = cv2.cvtColor(page, cv2.COLOR_BGR2RGB)
    folds = side_folds(rgb, variant)
    accepted = [f for f in folds if f.accepted]
    if not accepted:
        return SpineCut(zone, folds, [])
    ph, pw = page.shape[:2]
    # Прямоугольник полосы: углы TL, TR, BR, BL; принятые стороны переносятся на прямые отрезки.
    corners = {"tl": np.array([0.0, 0.0]), "tr": np.array([pw - 1.0, 0.0]),
               "br": np.array([pw - 1.0, ph - 1.0]), "bl": np.array([0.0, ph - 1.0])}  # fmt: skip
    beyond = np.zeros((ph, pw), bool)
    for fold in accepted:
        p0, p1 = straight_cut(fold, (ph, pw))
        corners = _move_side(corners, fold.side, p0, p1)
        beyond |= beyond_mask((ph, pw), fold)
    rect = np.array([corners["tl"], corners["tr"], corners["br"], corners["bl"]], np.float32)
    # Обратно в кадр: прямоугольник — гомографией, полоски за складками — тоже (на рабочий масштаб).
    inv = np.linalg.inv(m)
    new_quad = cv2.perspectiveTransform(rect.reshape(-1, 1, 2), inv).reshape(-1, 2) / scale
    to_work = np.diag([1 / scale, 1 / scale, 1.0]) @ inv
    beyond_work = cv2.warpPerspective(beyond.astype(np.uint8), to_work, (w_work, h_work), flags=cv2.INTER_NEAREST) > 0
    lines = [
        cv2.perspectiveTransform(f.points.reshape(-1, 1, 2).astype(np.float32), to_work).reshape(-1, 2)
        for f in accepted
    ]
    return SpineCut(zone_of(zone.paper & ~beyond_work, new_quad.astype(np.float64)), folds, lines)


def _move_side(corners: dict, side: SpineSide, p0: np.ndarray, p1: np.ndarray) -> dict:
    """Переносит сторону прямоугольника полосы на прямую ``p0``–``p1`` (концы прямой — на концах этой стороны).

    Args:
        corners: Углы ``tl, tr, br, bl`` (могли уже сдвинуться другими сторонами).
        side: Сторона.
        p0: Точка прямой у начала стороны (x=0 для верха/низа, y=0 для боков).
        p1: Точка у конца стороны.

    Returns:
        Новые углы (вход не меняется).
    """
    out = dict(corners)
    pair = {SpineSide.TOP: ("tl", "tr"), SpineSide.BOTTOM: ("bl", "br"),
            SpineSide.LEFT: ("tl", "bl"), SpineSide.RIGHT: ("tr", "br")}[side]  # fmt: skip
    out[pair[0]], out[pair[1]] = p0.astype(float), p1.astype(float)
    return out
