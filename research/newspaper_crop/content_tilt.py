"""Наклон сторон четырёхугольника по содержимому полосы: строки, края колонок и линейки → точки схода → направление каждой стороны."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from research.newspaper_crop.zone import quad_mask

# Окно оценки фона (закрытие убирает буквы), px рабочего масштаба: ~1.5 шага строки ЭГ (шаг ~13 px).
BACKGROUND_PX = 21
# Порог краски на нормированном сером (доля фона): буквы ЭГ на нормированном сером 0.3–0.6, тень загиба после
# нормировки ~0.9.
INK_LEVEL = 0.72
# Отступ от края листа, внутри которого признаки не берутся, px рабочего масштаба: тень и торец края.
EDGE_PX = 8
# Смыкание букв в строку вдоль набора, px рабочего масштаба: межсловный пробел ЭГ ~3–5 px, межстрочный просвет ~6 px.
LINE_CLOSE_PX = 11
# Строка набора: длина не меньше, толщина не больше, px рабочего масштаба.
MIN_LINE_LEN, MAX_LINE_THICK = 60, 14
# Край колонки: столько концов строк на одной прямой (допуск :data:`EDGE_TOL`), не короче :data:`MIN_EDGE_SPAN`.
MIN_EDGE_POINTS, EDGE_TOL, MIN_EDGE_SPAN = 8, 1.5, 120.0
# Линейка (LSD): длина не меньше, px рабочего масштаба.
MIN_RULE_LEN = 150.0
# Отбор в семейство: угол сегмента к направлению семейства не больше, градусы.
FAMILY_TOL = 15.0
# RANSAC точки схода: допуск угла сегмента к направлению «на точку», градусы.
VP_TOL_DEG = 0.35
# Бутстрап разброса угла стороны.
BOOTSTRAP = 30


class Family(Enum):
    """Семейство направлений: вдоль строк набора (горизонтали полосы) или поперёк (края колонок, вертикальные линейки)."""

    ALONG = "along"
    ACROSS = "across"


class SegmentKind(Enum):
    """Откуда сегмент: строка набора, край колонки (концы строк на прямой), линейка (LSD)."""

    TEXT_LINE = "text_line"
    COLUMN_EDGE = "column_edge"
    RULE = "rule"


@dataclass(frozen=True)
class Segment:
    """Прямолинейный признак содержимого: середина, единичное направление, длина (вес), семейство и происхождение."""

    centre: np.ndarray
    direction: np.ndarray
    length: float
    family: Family
    kind: SegmentKind

    @property
    def line(self) -> np.ndarray:
        """Прямая ``(a, b, c)`` с единичной нормалью."""
        n = np.array([-self.direction[1], self.direction[0]])
        return np.array([n[0], n[1], -float(n @ self.centre)])


@dataclass
class ContentModel:
    """Сегменты содержимого кадра и точки схода семейств (однородные координаты; None — семейство не набралось)."""

    segments: list[Segment]
    along_axis: np.ndarray
    vp: dict[Family, np.ndarray | None] = field(default_factory=dict)
    inliers: dict[Family, int] = field(default_factory=dict)

    def direction_at(self, family: Family, point: np.ndarray) -> np.ndarray | None:
        """Направление семейства в точке: на точку схода (или её направление, если она на бесконечности).

        Args:
            family: Семейство.
            point: Точка (2,) в пикселях рабочего масштаба.

        Returns:
            Единичный вектор или None (точки схода нет).
        """
        return _direction_to(self.vp.get(family), point)


def _direction_to(vp: np.ndarray | None, point: np.ndarray) -> np.ndarray | None:
    """Единичное направление из точки на точку схода ``vp`` (однородные координаты).

    Args:
        vp: Точка схода (3,) или None.
        point: Точка (2,).

    Returns:
        Единичный вектор или None.
    """
    if vp is None:
        return None
    d = vp[:2] - point * vp[2]
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-12 else None


def normalized_gray(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Серое, делённое на локальный фон (закрытие + размытие): тень загиба и неровный свет снимаются.

    Args:
        rgb: Кадр RGB.
        shape: Рабочий (H, W).

    Returns:
        float32 (H, W): ~1 на бумаге, меньше на краске.
    """
    gray = cv2.cvtColor(cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BACKGROUND_PX, BACKGROUND_PX))
    bg = cv2.GaussianBlur(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k), (0, 0), BACKGROUND_PX / 3)
    return gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)


def _line_blobs(ink: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    """Строки набора, идущие вдоль оси x: смыкание букв по x, вытянутые тонкие пятна, ось — главная компонента пикселей.

    Args:
        ink: Булева маска краски.

    Returns:
        Список (центр (2,), направление (2,), длина, толщина) в координатах маски.
    """
    closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, LINE_CLOSE_PX), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = []
    for i in range(1, count):
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # Быстрый отбор по габариту: длинная и низкая (наклон до ~6° добавляет к высоте ≤ 0.1 длины).
        if w < MIN_LINE_LEN or h > MAX_LINE_THICK + 0.1 * w:
            continue
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        ys, xs = np.nonzero(labels[y0 : y0 + h, x0 : x0 + w] == i)
        pts = np.column_stack([xs + x0, ys + y0]).astype(np.float64)
        centre = pts.mean(axis=0)
        cov = np.cov((pts - centre).T)
        vals, vecs = np.linalg.eigh(cov)
        direction = vecs[:, 1]
        along = (pts - centre) @ direction
        length = float(along.max() - along.min())
        thick = float(4 * np.sqrt(max(vals[0], 0.0)))
        if length < MIN_LINE_LEN or thick > MAX_LINE_THICK or abs(direction[0]) < 0.9:
            continue
        out.append((centre, direction if direction[0] > 0 else -direction, length, thick))
    return out


def _column_edges(ends: np.ndarray, across: np.ndarray, rng: np.random.Generator) -> list[Segment]:
    """Края колонок: концы строк, лежащие на одной прямой поперёк набора (последовательный RANSAC).

    Args:
        ends: Концы строк (N, 2).
        across: Единичное направление «поперёк набора» (ось y для набора вдоль x).
        rng: Генератор.

    Returns:
        Сегменты семейства ``ACROSS``.
    """
    out = []
    pts = ends.copy()
    for _ in range(40):
        if len(pts) < MIN_EDGE_POINTS:
            break
        best, best_in = None, None
        for _ in range(400):
            i, j = rng.choice(len(pts), 2, replace=False)
            d = pts[j] - pts[i]
            norm = np.linalg.norm(d)
            if norm < 40:
                continue
            d = d / norm
            # Только почти поперёк набора.
            if abs(d @ across) < np.cos(np.radians(FAMILY_TOL)):
                continue
            n = np.array([-d[1], d[0]])
            inl = np.abs((pts - pts[i]) @ n) < EDGE_TOL
            if best_in is None or inl.sum() > best_in.sum():
                best, best_in = (pts[i], d), inl
        if best_in is None or best_in.sum() < MIN_EDGE_POINTS:
            break
        sel = pts[best_in]
        centre = sel.mean(axis=0)
        vx, vy, _, _ = cv2.fitLine(sel.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        d = np.array([vx, vy], float)
        span = float(np.ptp((sel - centre) @ d))
        pts = pts[~best_in]
        if span < MIN_EDGE_SPAN:
            continue
        out.append(Segment(centre, d if d @ across > 0 else -d, span, Family.ACROSS, SegmentKind.COLUMN_EDGE))
    return out


def content_segments(rgb: np.ndarray, paper: np.ndarray, quad: np.ndarray, seed: int = 0) -> ContentModel:
    """Сегменты содержимого полосы: строки набора, края колонок по концам строк, линейки LSD.

    Ориентация набора выбирается по суммарной длине строк: набор вдоль x (обычная полоса) или вдоль y (полоса
    повёрнута в кадре). Признаки берутся внутри листа и четырёхугольника, глубже :data:`EDGE_PX` от края листа.

    Args:
        rgb: Кадр RGB.
        paper: Булева маска листа на рабочем масштабе.
        quad: Четырёхугольник зоны (пиксели маски).
        seed: Зерно RANSAC.

    Returns:
        :class:`ContentModel` без точек схода (их ставит :func:`fit_vanishing_points`).
    """
    h, w = paper.shape
    norm = normalized_gray(rgb, (h, w))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EDGE_PX + 1, 2 * EDGE_PX + 1))
    inside = (cv2.erode(paper.astype(np.uint8), k) > 0) & quad_mask(quad, (h, w))
    ink = (norm < INK_LEVEL) & inside
    rng = np.random.default_rng(seed)
    # Набор вдоль x и вдоль y (транспонированная маска); берём ориентацию с большей суммарной длиной строк.
    horiz = _line_blobs(ink)
    vert = [(c[::-1], d[::-1], length, t) for c, d, length, t in _line_blobs(ink.T)]
    use_vert = sum(b[2] for b in vert) > sum(b[2] for b in horiz)
    blobs, other = (vert, horiz) if use_vert else (horiz, vert)
    along_axis = np.array([0.0, 1.0]) if use_vert else np.array([1.0, 0.0])
    across_axis = np.array([1.0, 0.0]) if use_vert else np.array([0.0, 1.0])
    segments = [Segment(c, d, length, Family.ALONG, SegmentKind.TEXT_LINE) for c, d, length, _ in blobs]
    # Строки повёрнутых блоков (набор поперёк) — тоже «поперёк» полосы.
    segments += [Segment(c, d, length, Family.ACROSS, SegmentKind.TEXT_LINE) for c, d, length, _ in other]
    # Концы строк — по обе стороны; края колонок ищутся отдельно по левым и правым концам.
    for sign in (-1.0, 1.0):
        ends = np.array([c + sign * d * length / 2 for c, d, length, _ in blobs]).reshape(-1, 2)
        segments += _column_edges(ends, across_axis, rng)
    # Линейки: LSD по нормированному серому внутри листа.
    gray8 = np.clip(norm * 200, 0, 255).astype(np.uint8)
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(gray8)[0]
    for x1, y1, x2, y2 in ([] if lines is None else lines.reshape(-1, 4)):
        a, b = np.array([x1, y1], float), np.array([x2, y2], float)
        length = float(np.linalg.norm(b - a))
        if length < MIN_RULE_LEN:
            continue
        mid = (a + b) / 2
        if not inside[int(mid[1]), int(mid[0])]:
            continue
        d = (b - a) / length
        if abs(d @ along_axis) >= np.cos(np.radians(FAMILY_TOL)):
            segments.append(Segment(mid, d if d @ along_axis > 0 else -d, length, Family.ALONG, SegmentKind.RULE))
        elif abs(d @ across_axis) >= np.cos(np.radians(FAMILY_TOL)):
            segments.append(Segment(mid, d if d @ across_axis > 0 else -d, length, Family.ACROSS, SegmentKind.RULE))
    return ContentModel(segments, along_axis)


def _angle_error(segs: list[Segment], vp: np.ndarray) -> np.ndarray:
    """Угол между направлением сегмента и направлением из его середины на точку схода, градусы.

    Args:
        segs: Сегменты.
        vp: Точка схода (3,).

    Returns:
        Углы (N,).
    """
    out = np.empty(len(segs))
    for i, s in enumerate(segs):
        d = _direction_to(vp, s.centre)
        out[i] = 90.0 if d is None else np.degrees(np.arccos(np.clip(abs(d @ s.direction), 0, 1)))
    return out


def _vp_lsq(segs: list[Segment]) -> np.ndarray:
    """Точка схода по МНК: наименьший собственный вектор суммы ``w·l·lᵀ`` (прямые с единичной нормалью, вес — длина).

    Args:
        segs: Сегменты.

    Returns:
        Точка (3,) с единичной нормой.
    """
    # Обусловленность: координаты делятся на 1000 (прямая (a, b, c) → (a, b, c/1000)), точка обратно — (x, y, w/1000).
    m = np.zeros((3, 3))
    for s in segs:
        line = s.line * np.array([1.0, 1.0, 1e-3])
        m += s.length * np.outer(line, line)
    _, vecs = np.linalg.eigh(m)
    vp = vecs[:, 0] * np.array([1.0, 1.0, 1e-3])
    return vp / np.linalg.norm(vp)


def fit_vanishing_point(segs: list[Segment], rng: np.random.Generator) -> tuple[np.ndarray | None, np.ndarray]:
    """Точка схода семейства: RANSAC по парам сегментов (допуск :data:`VP_TOL_DEG`), уточнение МНК по опорным.

    Args:
        segs: Сегменты одного семейства.
        rng: Генератор.

    Returns:
        (точка схода (3,) или None, булева маска опорных сегментов).
    """
    if len(segs) < 3:
        return None, np.zeros(len(segs), bool)
    lengths = np.array([s.length for s in segs])
    p = lengths / lengths.sum()
    best, best_score = None, -1.0
    for _ in range(300):
        i, j = rng.choice(len(segs), 2, replace=False, p=p)
        vp = np.cross(segs[i].line, segs[j].line)
        if np.linalg.norm(vp) < 1e-12:
            continue
        inl = _angle_error(segs, vp) < VP_TOL_DEG
        score = float(lengths[inl].sum())
        if score > best_score:
            best, best_score = inl, score
    if best is None or best.sum() < 3:
        return None, np.zeros(len(segs), bool)
    vp = _vp_lsq([s for s, k in zip(segs, best) if k])
    inl = _angle_error(segs, vp) < VP_TOL_DEG
    return vp, inl


def fit_vanishing_points(model: ContentModel, seed: int = 0) -> ContentModel:
    """Ставит точки схода обоих семейств в модель.

    Args:
        model: Модель с сегментами.
        seed: Зерно RANSAC.

    Returns:
        Та же модель с заполненными ``vp`` и ``inliers``.
    """
    rng = np.random.default_rng(seed)
    for family in Family:
        segs = [s for s in model.segments if s.family is family]
        vp, inl = fit_vanishing_point(segs, rng)
        model.vp[family] = vp
        model.inliers[family] = int(inl.sum())
    return model


def side_family(quad: np.ndarray, i: int, along_axis: np.ndarray) -> Family:
    """Семейство стороны ``i`` четырёхугольника: сторона вдоль набора — ``ALONG`` (верх/низ обычной полосы).

    Args:
        quad: Вершины (4, 2).
        i: Номер стороны (от вершины i к i+1).
        along_axis: Ось набора кадра.

    Returns:
        Семейство.
    """
    d = quad[(i + 1) % 4] - quad[i]
    d = d / (np.linalg.norm(d) + 1e-12)
    return Family.ALONG if abs(d @ along_axis) > np.cos(np.radians(45)) else Family.ACROSS


def signed_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Угол от прямой ``a`` до прямой ``b`` (направления без знака), градусы в (−90, 90].

    Args:
        a: Направление (2,).
        b: Направление (2,).

    Returns:
        Угол.
    """
    ang = np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))
    return float((ang + 90) % 180 - 90)


@dataclass(frozen=True)
class SideTilt:
    """Сторона четырёхугольника против содержимого: угол расхождения, его разброс, опора семейства."""

    index: int
    family: Family
    mask_dir: np.ndarray
    content_dir: np.ndarray | None
    delta_deg: float
    sigma_deg: float
    support: int


def side_tilts(model: ContentModel, quad: np.ndarray, seed: int = 0) -> list[SideTilt]:
    """Направление содержимого у середины каждой стороны и расхождение со стороной (с бутстрап-разбросом).

    Args:
        model: Модель с точками схода.
        quad: Четырёхугольник (пиксели маски).
        seed: Зерно бутстрапа.

    Returns:
        Четыре :class:`SideTilt`; без точки схода семейства — ``content_dir`` None, ``delta_deg`` NaN.
    """
    rng = np.random.default_rng(seed)
    out = []
    by_family = {f: [s for s in model.segments if s.family is f] for f in Family}
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        mask_dir = (b - a) / (np.linalg.norm(b - a) + 1e-12)
        mid = (a + b) / 2
        family = side_family(quad, i, model.along_axis)
        content = model.direction_at(family, mid)
        if content is None:
            out.append(SideTilt(i, family, mask_dir, None, float("nan"), float("nan"), 0))
            continue
        delta = signed_angle(mask_dir, content)
        # Бутстрап: пересчёт точки схода по выборке опорных с возвращением.
        segs = by_family[family]
        inl = _angle_error(segs, model.vp[family]) < VP_TOL_DEG
        core = [s for s, k in zip(segs, inl) if k]
        boots = []
        for _ in range(BOOTSTRAP):
            pick = rng.choice(len(core), len(core), replace=True)
            d = _direction_to(_vp_lsq([core[j] for j in pick]), mid)
            if d is not None:
                boots.append(signed_angle(mask_dir, d))
        sigma = float(np.std(boots)) if boots else float("nan")
        out.append(SideTilt(i, family, mask_dir, content, delta, sigma, len(core)))
    return out


# Полоса у стороны, из которой берутся признаки локальной оценки, px рабочего масштаба (~5 см бумаги).
LOCAL_BAND = 250.0
# Сколько ближайших к стороне сегментов семейства берётся в локальную оценку (по длине — вес).
LOCAL_COUNT = 12


def local_direction(model: ContentModel, quad: np.ndarray, i: int) -> tuple[np.ndarray | None, float, int]:
    """Локальная оценка направления стороны: взвешенное по длине среднее направление ближайших к ней сегментов
    её семейства (без глобальной модели — годится и для изогнутого у корешка листа).

    Для боковых сторон берутся края колонок и линейки (строки повёрнутых блоков тоже), для верха и низа — строки
    и линейки. Сегмент должен лежать в полосе :data:`LOCAL_BAND` от стороны.

    Args:
        model: Модель с сегментами.
        quad: Четырёхугольник (пиксели маски).
        i: Номер стороны.

    Returns:
        (единичное направление или None, разброс углов сегментов в градусах (взвешенное СКО), число сегментов).
    """
    a, b = quad[i], quad[(i + 1) % 4]
    side_dir = (b - a) / (np.linalg.norm(b - a) + 1e-12)
    normal = np.array([-side_dir[1], side_dir[0]])
    family = side_family(quad, i, model.along_axis)
    cands = []
    for s in model.segments:
        if s.family is not family:
            continue
        dist = abs(float((s.centre - a) @ normal))
        if dist > LOCAL_BAND:
            continue
        cands.append((dist, s))
    if len(cands) < 2:
        return None, float("nan"), len(cands)
    cands = sorted(cands, key=lambda t: t[0])[:LOCAL_COUNT]
    angles = np.array([signed_angle(side_dir, s.direction) for _, s in cands])
    weights = np.array([s.length for _, s in cands])
    # Медиана по весу устойчивее среднего к одиночному косому сегменту (заголовок, изгиб).
    order = np.argsort(angles)
    cum = np.cumsum(weights[order]) / weights.sum()
    med = float(angles[order][np.searchsorted(cum, 0.5)])
    spread = float(np.sqrt(np.average((angles - med) ** 2, weights=weights)))
    rad = np.radians(med)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return rot @ side_dir, spread, len(cands)


def border_share(quad: np.ndarray, i: int, shape: tuple[int, int], tol: float = 3.0) -> float:
    """Доля стороны на рамке кадра или за ней: сторона легла на обрез кадра, края листа там не видно.

    Args:
        quad: Четырёхугольник (пиксели маски).
        i: Номер стороны.
        shape: (H, W) маски.
        tol: Допуск до рамки, px.

    Returns:
        Доля точек стороны (200 отсчётов) не дальше ``tol`` от рамки или вне кадра.
    """
    h, w = shape
    t = np.linspace(0, 1, 200)[:, None]
    pts = quad[i] + t * (quad[(i + 1) % 4] - quad[i])
    near = (pts[:, 0] <= tol) | (pts[:, 1] <= tol) | (pts[:, 0] >= w - 1 - tol) | (pts[:, 1] >= h - 1 - tol)
    return float(near.mean())
