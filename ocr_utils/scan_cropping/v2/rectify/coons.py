"""Карта выпрямления разворота по подгоночным областям страниц: поверхность Кунса по четырём сторонам каждой страницы (верх, низ — сплайны, внешняя сторона — сплайн, у корешка — линия корешка) → прямоугольник; за пределами области — линейное продолжение сторон касательной."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.page_frames import PageFrame

# Дорисовка за краем итога v2: σ размытия = доля удаления за край, потолок σ в мм бумаги, ниже какого σ не размывать.
FADE_SLOPE = 0.5
FADE_MAX_MM = 15.0
FADE_MIN_SIGMA_PX = 0.5
# Отрезок у конца стороны, по которому берётся касательная для продолжения за угол, мм бумаги: короче — направление
# дрожит от ступенек полилинии, длиннее — в него входит изгиб у угла.
TANGENT_MM = 5.0


@dataclass(frozen=True)
class ArcCurve:
    """Ломаная, параметризованная долей длины дуги ``t ∈ [0, 1]``; вне отрезка — продолжение касательной у концов.

    Поля:
        points: Вершины (N, 2) px без повторов подряд.
        arc: Накопленная длина дуги по вершинам (N,), ``arc[0] = 0``.
        head: Единичное направление у начала (вперёд по кривой).
        tail: Единичное направление у конца (вперёд по кривой).
    """

    points: np.ndarray
    arc: np.ndarray
    head: np.ndarray
    tail: np.ndarray

    @classmethod
    def from_points(cls, points: np.ndarray, tangent_px: float) -> ArcCurve:
        """Кривая по вершинам.

        Args:
            points: Вершины (N, 2), N ≥ 2, по порядку обхода.
            tangent_px: Длина отрезка у конца для касательной, px.

        Returns:
            Кривая.

        Raises:
            ValueError: Меньше двух различных вершин.
        """
        pts = np.asarray(points, np.float64).reshape(-1, 2)
        # Повторы подряд дают нулевые шаги дуги — ``np.interp`` с ними неоднозначен.
        step = np.r_[1.0, np.hypot(*np.diff(pts, axis=0).T)]
        pts = pts[step > 1e-9]
        if len(pts) < 2:
            raise ValueError("сторона области вырождена: меньше двух различных вершин")
        arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        length = arc[-1]
        span = min(tangent_px, length)
        # Касательная у начала — к точке кривой на расстоянии ``span`` по дуге, у конца — от такой точки.
        head = interp_arc(pts, arc, np.array([span]))[0] - pts[0]
        tail = pts[-1] - interp_arc(pts, arc, np.array([length - span]))[0]
        return cls(pts, arc, head / np.linalg.norm(head), tail / np.linalg.norm(tail))

    @property
    def length(self) -> float:
        """Длина кривой, px."""
        return float(self.arc[-1])

    def at(self, t: np.ndarray) -> np.ndarray:
        """Точки кривой при долях длины ``t`` (любые, вне ``[0, 1]`` — на касательных у концов).

        Args:
            t: Доли длины (M,).

        Returns:
            Точки (M, 2) px.
        """
        d = np.asarray(t, np.float64) * self.length
        out = interp_arc(self.points, self.arc, d)
        # До начала и после конца ``np.interp`` держит крайнюю точку — заменяем продолжением касательной.
        before, after = d < 0, d > self.length
        out[before] = self.points[0] + d[before, None] * self.head
        out[after] = self.points[-1] + (d[after, None] - self.length) * self.tail
        return out


def interp_arc(points: np.ndarray, arc: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Точки ломаной на расстояниях ``d`` по дуге (внутри ``[0, L]``; снаружи — крайние вершины).

    Args:
        points: Вершины (N, 2).
        arc: Накопленная длина дуги (N,).
        d: Расстояния по дуге (M,).

    Returns:
        Точки (M, 2).
    """
    return np.stack([np.interp(d, arc, points[:, 0]), np.interp(d, arc, points[:, 1])], axis=1)


@dataclass(frozen=True)
class Patch:
    """Четырёхсторонняя область страницы в координатах итога: ``u`` — слева направо, ``v`` — сверху вниз.

    Поля:
        top: Верх слева направо.
        bottom: Низ слева направо.
        left: Левая сторона сверху вниз.
        right: Правая сторона сверху вниз.
    """

    top: ArcCurve
    bottom: ArcCurve
    left: ArcCurve
    right: ArcCurve

    def corners(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Углы по верху и низу: левый верхний, правый верхний, левый нижний, правый нижний."""
        return self.top.at(np.array([0.0]))[0], self.top.at(np.array([1.0]))[0], \
            self.bottom.at(np.array([0.0]))[0], self.bottom.at(np.array([1.0]))[0]  # fmt: skip

    def sides_at(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Точки левой и правой сторон при долях ``v``, подогнанные концами к углам верха и низа.

        Стороны и верх (низ) строятся отдельно, и их концы могут разойтись на доли пикселя; Кунс точно сажает
        стороны на стороны прямоугольника, только если углы общие. Расхождение снимается линейной поправкой по ``v``.

        Args:
            v: Доли высоты (M,).

        Returns:
            Левая и правая сторона, по (M, 2).
        """
        c00, c10, c01, c11 = self.corners()
        v = np.asarray(v, np.float64)[:, None]
        out = []
        for side, top_end, bottom_end in ((self.left, c00, c01), (self.right, c10, c11)):
            ends = side.at(np.array([0.0, 1.0]))
            out.append(side.at(v[:, 0]) + (1 - v) * (top_end - ends[0]) + v * (bottom_end - ends[1]))
        return out[0], out[1]

    def points(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Точки поверхности Кунса для пар ``(u[i], v[i])``.

        Args:
            u: Доли ширины (M,).
            v: Доли высоты (M,).

        Returns:
            Точки (M, 2) px итога.
        """
        u = np.asarray(u, np.float64)[:, None]
        v = np.asarray(v, np.float64)[:, None]
        c00, c10, c01, c11 = self.corners()
        left, right = self.sides_at(v[:, 0])
        # Сумма линейных по ``v`` (верх-низ) и по ``u`` (стороны) минус билинейная по углам.
        ruled_v = (1 - v) * self.top.at(u[:, 0]) + v * self.bottom.at(u[:, 0])
        ruled_u = (1 - u) * left + u * right
        bilinear = (1 - u) * (1 - v) * c00 + u * (1 - v) * c10 + (1 - u) * v * c01 + u * v * c11
        return ruled_v + ruled_u - bilinear

    def grid(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Поверхность Кунса на сетке ``v × u`` (по строкам и столбцам холста) — раздельно X и Y, float32.

        Сумма раскладывается на одномерные множители по ``u`` и ``v``, поэтому вся сетка — одна свёртка broadcast без
        циклов (на кадр 4600×3600 — около 0.3 с).

        Args:
            u: Доли ширины по столбцам (W,).
            v: Доли высоты по строкам (H,).

        Returns:
            ``(X, Y)`` — по массиву (H, W) float32.
        """
        u = np.asarray(u, np.float64)
        v = np.asarray(v, np.float64)
        c00, c10, c01, c11 = self.corners()
        top, bottom = self.top.at(u), self.bottom.at(u)
        left, right = self.sides_at(v)
        out = []
        for c in (0, 1):
            # Столбцовые множители (1, W) и строчные (H, 1); произведения дают (H, W).
            uu = u[None, :].astype(np.float32)
            vv = v[:, None].astype(np.float32)
            t_row = (top[:, c] - ((1 - u) * c00[c] + u * c10[c]))[None, :].astype(np.float32)
            b_row = (bottom[:, c] - ((1 - u) * c01[c] + u * c11[c]))[None, :].astype(np.float32)
            l_col = left[:, c][:, None].astype(np.float32)
            r_col = right[:, c][:, None].astype(np.float32)
            # ``(1-v)·(T - угловая) + v·(B - угловая) + (1-u)·L + u·R`` — то же, что :meth:`points`.
            plane = (1 - vv) * t_row + vv * b_row
            plane += (1 - uu) * l_col
            plane += uu * r_col
            out.append(plane)
        return out[0], out[1]


def page_patch(frame: PageFrame, left_page: bool, to_output, tangent_px: float) -> Patch:
    """Область страницы → :class:`Patch` в координатах итога.

    Args:
        frame: Подгоночная область (``page_frames.PageFrame``; верх и низ — от внешнего угла к корешку).
        left_page: Левая страница разворота (внешняя сторона — слева, корешок — справа).
        to_output: Перевод точек (N, 2) из координат области в px итога.
        tangent_px: Длина отрезка у конца стороны для касательной, px итога.

    Returns:
        Область с верхом и низом слева направо, сторонами сверху вниз.
    """
    top, bottom = to_output(frame.top), to_output(frame.bottom)
    far, spine = to_output(frame.far), to_output(frame.spine)
    if not left_page:
        # У правой страницы верх и низ идут от внешнего угла (справа) к корешку (слева).
        top, bottom = top[::-1], bottom[::-1]
    left, right = (far, spine) if left_page else (spine, far)
    curves = [ArcCurve.from_points(p, tangent_px) for p in (top, bottom, left, right)]
    return Patch(*curves)


@dataclass(frozen=True)
class PageRect:
    """Прямоугольник страницы на холсте выпрямленного кадра, px (края — по сторонам области)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """Ширина, px."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Высота, px."""
        return self.y1 - self.y0


@dataclass(frozen=True)
class Layout:
    """Расклад выпрямленного кадра.

    Поля:
        width: Ширина холста, px.
        height: Высота холста, px.
        rects: Прямоугольники левой и правой страницы.
        spine_x: X линии корешка на холсте (левее — карта левой страницы, правее — правой).
    """

    width: int
    height: int
    rects: tuple[PageRect, PageRect]
    spine_x: float


def plan_layout(patches: tuple[Patch, Patch], shape: tuple[int, int]) -> Layout:
    """Прямоугольники страниц и холст.

    Ширина прямоугольника — средняя длина верха и низа, высота — длина стороны у корешка: тогда точки линии корешка
    с двух сторон ложатся на одну высоту и шва на корешке нет (у правой и левой страницы высоты разные — это
    допускается). Прямоугольники встык по линии корешка, верх каждого — на высоте его угла у корешка. Поля холста —
    расстояния от области до краёв итога (слева — от внешней стороны левой страницы, сверху — от высшей точки верха
    и т. д.).

    Args:
        patches: Области левой и правой страницы в px итога.
        shape: ``(H, W)`` итога.

    Returns:
        Расклад холста.
    """
    h, w = shape
    left, right = patches
    # Корешок — средний X концов стороны у корешка обеих страниц.
    spine_pts = np.vstack([left.right.points[[0, -1]], right.left.points[[0, -1]]])
    xs = float(spine_pts[:, 0].mean())
    rects = []
    for patch, is_left in ((left, True), (right, False)):
        width = (patch.top.length + patch.bottom.length) / 2
        spine_side = patch.right if is_left else patch.left
        y0 = float(spine_side.points[0, 1])
        x0, x1 = (xs - width, xs) if is_left else (xs, xs + width)
        rects.append(PageRect(x0, y0, x1, y0 + spine_side.length))
    margin_left = float(left.left.points[:, 0].min())
    margin_right = float(w - right.right.points[:, 0].max())
    # Сторона области может выйти за край итога на пиксели (продление к углу) — поле тогда нулевое, не отрицательное.
    top_edge = min(r.y0 - max(float(p.top.points[:, 1].min()), 0.0) for p, r in zip(patches, rects))
    bottom_edge = max(r.y1 + max(float(h - p.bottom.points[:, 1].max()), 0.0) for p, r in zip(patches, rects))
    x_min = rects[0].x0 - max(margin_left, 0.0)
    x_max = rects[1].x1 + max(margin_right, 0.0)
    # Холст — от (x_min, top_edge); сдвигаем прямоугольники в его координаты.
    shifted = tuple(PageRect(r.x0 - x_min, r.y0 - top_edge, r.x1 - x_min, r.y1 - top_edge) for r in rects)
    return Layout(int(np.ceil(x_max - x_min)), int(np.ceil(bottom_edge - top_edge)), shifted, xs - x_min)


def rect_fractions(rect: PageRect, cols: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Доли ``u`` по столбцам и ``v`` по строкам холста относительно прямоугольника страницы."""
    return (cols - rect.x0) / rect.width, (rows - rect.y0) / rect.height


def rectify_maps(patches: tuple[Patch, Patch], layout: Layout) -> tuple[np.ndarray, np.ndarray]:
    """Карты ``cv2.remap``: для каждого пикселя холста — точка итога, откуда он берётся.

    Args:
        patches: Области левой и правой страницы.
        layout: Расклад холста.

    Returns:
        ``(map_x, map_y)`` — по (H, W) float32 размера холста.
    """
    rows = np.arange(layout.height, dtype=np.float64)
    cols = np.arange(layout.width, dtype=np.float64)
    map_x = np.empty((layout.height, layout.width), np.float32)
    map_y = np.empty_like(map_x)
    # Левее линии корешка холста — карта левой страницы, правее — правой.
    split = int(np.ceil(layout.spine_x))
    for patch, rect, part in zip(patches, layout.rects, (slice(0, split), slice(split, layout.width))):
        u, v = rect_fractions(rect, cols[part], rows)
        map_x[:, part], map_y[:, part] = patch.grid(u, v)
    return map_x, map_y


def rectify_image(image: np.ndarray, maps: tuple[np.ndarray, np.ndarray], dpi: float) -> np.ndarray:
    """Выпрямленный кадр: бикубическая выборка итога по картам; за краем итога — дорисовка с размытием по удалению.

    Где карта выходит за край итога v2 (поля холста за областью копирования), берётся ближний краевой пиксель
    (продление края), а картинка там размывается тем сильнее, чем дальше точка за краем: σ = ``FADE_SLOPE`` · удаление,
    не больше ``FADE_MAX_MM``. Прежнее отражение от края зеркалило текст у края на поле (ХиП 1991/01 IMG_1049: строка
    шапки вверх ногами над страницей).

    Args:
        image: Итог v2 (BGR или серый).
        maps: ``(map_x, map_y)`` из :func:`rectify_maps`.
        dpi: Разрешение (потолок размытия — в мм).

    Returns:
        Картинка размера карт.
    """
    out = cv2.remap(image, maps[0], maps[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    h, w = image.shape[:2]
    over_x = np.maximum(np.maximum(-maps[0], maps[0] - (w - 1)), 0)
    over_y = np.maximum(np.maximum(-maps[1], maps[1] - (h - 1)), 0)
    sigma = np.minimum(FADE_SLOPE * np.hypot(over_x, over_y), FADE_MAX_MM * dpi / 25.4)
    outside = sigma > FADE_MIN_SIGMA_PX
    if not outside.any():
        return out
    # Пирамида размытий σ = 2^k px; в каждом пикселе за краем — смесь двух соседних уровней по log2 σ.
    top = int(np.ceil(np.log2(max(float(sigma.max()), 2.0))))
    levels = [out] + [blurred(out, 2.0**k) for k in range(1, top + 1)]
    pos = np.clip(np.log2(np.maximum(sigma[outside], 1.0)), 0, top)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, top)
    frac = (pos - lo)[:, None] if out.ndim == 3 else pos - lo
    stack = np.stack([lv[outside] for lv in levels]).astype(np.float32)  # (уровни, N[, каналы])
    idx = np.arange(len(lo))
    mixed = stack[lo, idx] * (1 - frac) + stack[hi, idx] * frac
    out[outside] = np.clip(np.round(mixed), 0, 255).astype(out.dtype)
    return out


def blurred(image: np.ndarray, sigma: float) -> np.ndarray:
    """Гауссово размытие с большим σ через уменьшенную копию (на полном кадре ядро в сотни пикселей — медленно).

    Args:
        image: Картинка.
        sigma: σ, px.

    Returns:
        Размытая картинка того же размера.
    """
    f = max(1.0, sigma / 2.0)
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(int(w / f), 1), max(int(h / f), 1)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def sagitta(curve: ArcCurve) -> float:
    """Наибольшее отклонение кривой от хорды между её концами, px (мера изгиба стороны)."""
    a, b = curve.points[0], curve.points[-1]
    d = b - a
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    return float(np.abs((curve.points - a) @ n).max())
