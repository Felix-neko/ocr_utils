"""Поле смещений «без коррекции → с коррекцией»: что FineReader сделал со страницей.

Страница B (без коррекции) режется на тайлы, каждый ищется на странице A (с коррекцией)
нормированной кросс-корреляцией. Смещения тайлов — поле; по нему робастно подгоняется
аффинное преобразование (поворот, сдвиг осей, масштаб — это «перекос» и «трапеция»
FineReader), а остаток — локальная нежёсткая деформация, то есть распрямление строк.
Остаток внутри текста — ожидаемое лекарство; остаток внутри чертежа или блок-схемы —
порча: там нет строк, которые надо было выпрямлять (1966/01 с.78, 1967/01 с.80).

ДВА ПРОХОДА. Поворот в 1.5° и масштаб в 3 % дают на краях страницы смещения за 40 px при
150 dpi, а широкое окно поиска дорого и ловит соседнюю строку. Поэтому сначала редкая сетка
с широким окном → аффинная подгонка, потом плотная сетка с узким окном вокруг предсказания.

ОТБРАКОВКА ТАЙЛОВ. Тайл с одной вертикальной линейкой или отточиями «плывёт» вдоль себя:
пик корреляции размазан по полосе, и выбранная точка случайна (замер на 1966/01 с.95 —
стрелки вдоль рамки по 40 px в разные стороны). Такие тайлы отсекаются требованием
ОТЧЁТЛИВОГО пика: он должен быть заметно выше максимума отклика вне своей окрестности.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from ocr_utils.geometry_regression import WORK_DPI, mm_to_px, px_to_mm

# Сетка тайлов: 27 мм (160 px при 150 dpi) — 5-6 строк корпуса, хватает для острого пика.
TILE_MM = 27.0
STEP_MM = 17.0
# Окна поиска: широкое на редкой сетке (поворот 1.5° и масштаб 3 % дают до 8 мм на краях),
# узкое вокруг аффинного предсказания.
COARSE_SEARCH_MM = 11.0
FINE_SEARCH_MM = 3.4
# Порог пика нормированной корреляции и его отчётливости (пик / максимум вне окрестности пика).
MIN_PEAK = 0.45
MIN_DISTINCT = 1.2
PEAK_RADIUS_MM = 0.7
# Тайл, не нашедший пары в масштабе 1:1, ищется ещё и с шаблоном, растянутым/сжатым на эти
# доли: убирая трапецию, FineReader растягивает верх страницы на 6–10 % (1975/05 с.61: шаг
# строк 42 → 46 px при 300 dpi), и 27-мм тайл с текстом при таком масштабе не совпадает
# (пик 0.2 против 0.6 при масштабе 1.08). Без этого верхняя половина страницы оставалась без
# тайлов, аффинная часть поля туда экстраполировалась мимо на 5–8 мм, и пробы линеек падали
# на текст. Погнутая схема (1967/01 с.80) масштабом не совпадает — её тайлы остаются без пары.
SCALE_TRIES = (0.96, 1.04, 0.92, 1.08)
# Пустой тайл: меньше такой доли краски — искать нечего.
MIN_INK_FRAC = 0.012
# Меньше стольких тайлов — аффинная подгонка не имеет смысла.
MIN_TILES = 8
# Доля тайлов без пары внутри line art считается только по рисунку хотя бы из стольких тайлов
# (≈ 40×40 мм): виньетка «100 лет» на 1970/04 с.25 — 2 тайла, один без пары — «порча» 0.5.
MIN_LINEART_TILES = 6
# Робастная подгонка: итерации IRLS и константа Тьюки в единицах MAD-масштаба; нижняя
# граница масштаба остатков — полпикселя (точность субпиксельного пика).
IRLS_ITERS = 10
TUKEY_C = 4.685
MIN_SCALE_PX = 0.5


@dataclass
class Field:
    """Поле смещений страницы B → A на копии 150 dpi.

    ``tiles`` — строки ``(cx, cy, ux, uy, peak)``: центр тайла в B и полное смещение
    (точка B ``p`` оказалась в A в ``p + u``). ``affine`` — робастно подогнанная матрица
    2×3 (B → A), ``resid`` — остаток смещения после неё, ``weight`` — вес IRLS
    (0 у выбросов). ``failed`` — строки ``(cx, cy, kind)`` тайлов с краской, которым пары
    не нашлось: ``kind`` 0 — слабый пик (содержимое тайла деформировано — так выглядит
    погнутая блок-схема), 1 — пик неотчётлив (периодика: отточия, штриховка, одна линейка).
    """

    width: int
    height: int
    dpi: float
    tiles: np.ndarray
    affine: np.ndarray
    resid: np.ndarray
    weight: np.ndarray
    failed: np.ndarray

    @property
    def rot_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.affine[1, 0], self.affine[0, 0])))

    @property
    def shear_deg(self) -> float:
        """Отклонение осей от прямого угла: наклон образа вертикали сверх поворота (трапеция/сдвиг FineReader)."""
        return float(np.degrees(np.arctan2(self.affine[0, 1], self.affine[1, 1]))) + self.rot_deg

    @property
    def scale_x(self) -> float:
        return float(np.hypot(self.affine[0, 0], self.affine[1, 0]))

    @property
    def scale_y(self) -> float:
        return float(np.hypot(self.affine[0, 1], self.affine[1, 1]))

    @property
    def resid_norm(self) -> np.ndarray:
        return np.hypot(self.resid[:, 0], self.resid[:, 1])

    def transform(self, points: np.ndarray, k: int = 4) -> np.ndarray:
        """Точки B → A: аффинная часть плюс остаток, взятый с ближайших тайлов (обратные расстояния)."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        out = pts @ self.affine[:, :2].T + self.affine[:, 2]
        good = self.weight > 0
        if good.sum() >= 1:
            centres = self.tiles[good, :2]
            resid = self.resid[good]
            tree = cKDTree(centres)
            dist, idx = tree.query(pts, k=min(k, len(centres)))
            dist = np.atleast_2d(dist)
            idx = np.atleast_2d(idx)
            w = 1.0 / np.maximum(dist, 1.0)
            out += (resid[idx] * w[..., None]).sum(axis=1) / w.sum(axis=1, keepdims=True)
        return out

    def resid_inside(self, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
        """Нормы остатков тайлов, чьи центры лежат в одном из прямоугольников (x0, y0, x1, y1)."""
        if not boxes or len(self.tiles) == 0:
            return np.zeros(0)
        inside = _inside(self.tiles[:, :2], boxes)
        return self.resid_norm[inside & (self.weight > 0)]

    def weak_frac_inside(self, boxes: list[tuple[int, int, int, int]]) -> tuple[float, int]:
        """Доля тайлов со слабым пиком среди всех тайлов с краской внутри рамок, и их общее число."""
        if not boxes:
            return 0.0, 0
        matched = int(_inside(self.tiles[:, :2], boxes).sum()) if len(self.tiles) else 0
        failed = self.failed[_inside(self.failed[:, :2], boxes)] if len(self.failed) else np.zeros((0, 3))
        total = matched + len(failed)
        weak = int((failed[:, 2] == 0).sum()) if len(failed) else 0
        return (weak / total if total else 0.0), total


def _inside(points: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        inside |= (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
    return inside


def _pad_to(a: np.ndarray, height: int, width: int) -> np.ndarray:
    out = np.full((height, width), 255, np.uint8)
    out[: a.shape[0], : a.shape[1]] = a
    return out


def _global_shift(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    fb = cv2.GaussianBlur(255 - before, (0, 0), 3).astype(np.float32)
    fa = cv2.GaussianBlur(255 - after, (0, 0), 3).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(fb, fa)
    return float(dx), float(dy)


def _refine(response: np.ndarray, x: int, y: int) -> tuple[float, float]:
    """Субпиксельное уточнение пика параболой по каждой оси."""

    def axis(m: float, c: float, p: float) -> float:
        denom = m - 2 * c + p
        return 0.0 if abs(denom) < 1e-9 else float(np.clip(0.5 * (m - p) / denom, -0.5, 0.5))

    h, w = response.shape
    dx = axis(response[y, x - 1], response[y, x], response[y, x + 1]) if 0 < x < w - 1 else 0.0
    dy = axis(response[y - 1, x], response[y, x], response[y + 1, x]) if 0 < y < h - 1 else 0.0
    return dx, dy


def _match_once(template: np.ndarray, image: np.ndarray, x_pred: float, y_pred: float, search: int, radius: int):
    """Смещение шаблона ``(x, y, peak)`` в A, либо код отказа: 0 — слабый пик, 1 — неотчётливый, None — окно за краем."""
    tile_h, tile_w = template.shape
    h, w = image.shape
    x0 = int(np.clip(round(x_pred) - search, 0, w))
    y0 = int(np.clip(round(y_pred) - search, 0, h))
    x1 = int(np.clip(round(x_pred) + tile_w + search, 0, w))
    y1 = int(np.clip(round(y_pred) + tile_h + search, 0, h))
    window = image[y0:y1, x0:x1]
    if window.shape[0] < tile_h + 2 * radius + 1 or window.shape[1] < tile_w + 2 * radius + 1:
        return None
    response = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (px, py) = cv2.minMaxLoc(response)
    if peak < MIN_PEAK:
        return 0
    masked = response.copy()
    masked[max(0, py - radius) : py + radius + 1, max(0, px - radius) : px + radius + 1] = -1.0
    second = float(masked.max())
    if second > 0 and peak / second < MIN_DISTINCT:
        return 1
    sx, sy = _refine(response, px, py)
    return x0 + px + sx, y0 + py + sy, float(peak)


def _match(
    template: np.ndarray,
    image: np.ndarray,
    x_pred: float,
    y_pred: float,
    search: int,
    radius: int,
    rescale: bool = True,
):
    """Смещение тайла в A: сначала 1:1, при отказе и ``rescale`` — с шаблоном в масштабах ``SCALE_TRIES``.

    Возвращает ``(x, y, peak)`` — положение левого верхнего угла НЕмасштабированного тайла,
    как если бы он лёг центром туда же, куда лёг масштабированный; либо код отказа последней
    попытки 1:1 (0 — слабый пик, 1 — неотчётливый) или None — окно за краем.
    """
    found = _match_once(template, image, x_pred, y_pred, search, radius)
    if not isinstance(found, int) or not rescale:
        return found
    tile_h, tile_w = template.shape
    for scale in SCALE_TRIES:
        scaled = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # Масштабированный шаблон центрируется там же, где стоял бы исходный.
        shift_x, shift_y = (scaled.shape[1] - tile_w) / 2.0, (scaled.shape[0] - tile_h) / 2.0
        retry = _match_once(scaled, image, x_pred - shift_x, y_pred - shift_y, search, radius)
        if retry is not None and not isinstance(retry, int):
            fx, fy, peak = retry
            return fx + shift_x, fy + shift_y, peak
    return found


def robust_affine(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Аффинная матрица 2×3 ``src → dst`` методом IRLS с весами Тьюки.

    Returns:
        ``(affine, resid, weight)``: остатки ``dst − affine(src)`` и веса точек (0 — выброс).
    """
    X = np.c_[src, np.ones(len(src))]
    weight = np.ones(len(src))
    affine = np.zeros((2, 3))
    resid = np.zeros_like(dst)
    for _ in range(IRLS_ITERS):
        sw = np.sqrt(weight)[:, None]
        solution, _, _, _ = np.linalg.lstsq(X * sw, dst * sw, rcond=None)
        affine = solution.T
        resid = dst - X @ solution
        norm = np.hypot(resid[:, 0], resid[:, 1])
        scale = max(MIN_SCALE_PX, 1.4826 * float(np.median(norm)))
        ratio = norm / (TUKEY_C * scale)
        weight = np.where(ratio < 1.0, (1.0 - ratio**2) ** 2, 0.0)
        if weight.sum() < 3:
            weight = np.ones(len(src))
            break
    return affine, resid, weight


def _tile_matches(
    before: np.ndarray, after: np.ndarray, predict, tile: int, step: int, search: int, radius: int, lineart_boxes: list
) -> tuple[np.ndarray, np.ndarray]:
    """Сопоставление тайлов сетки с шагом ``step``; ``predict(x, y)`` — где искать в A.

    Тайлы с центром внутри ``lineart_boxes`` ищутся только 1:1: рисунок, который совпадает лишь
    после масштабирования, — деформированный рисунок (перекошенное фото 1971/04 с.44), и его
    тайлы должны остаться без пары; для текста же масштаб — законная правка трапеции.

    Returns:
        Найденные ``(cx, cy, ux, uy, peak)`` и ненайденные ``(cx, cy, kind)`` тайлы с краской.
    """
    ink_b = (255 - before).astype(np.float32)
    ink_a = (255 - after).astype(np.float32)
    h, w = before.shape
    rows, failed = [], []
    for y in range(0, h - tile + 1, step):
        for x in range(0, w - tile + 1, step):
            template = ink_b[y : y + tile, x : x + tile]
            if template.mean() / 255.0 < MIN_INK_FRAC:
                continue
            x_pred, y_pred = predict(x, y)
            centre = np.array([[x + tile / 2.0, y + tile / 2.0]])
            in_lineart = bool(_inside(centre, lineart_boxes)[0]) if lineart_boxes else False
            found = _match(template, ink_a, x_pred, y_pred, search, radius, rescale=not in_lineart)
            if found is None:
                continue
            if isinstance(found, int):
                failed.append((x + tile / 2.0, y + tile / 2.0, found))
                continue
            fx, fy, peak = found
            rows.append((x + tile / 2.0, y + tile / 2.0, fx - x, fy - y, peak))
    return np.array(rows, dtype=np.float64).reshape(-1, 5), np.array(failed, dtype=np.float64).reshape(-1, 3)


def estimate_field(
    before: np.ndarray, after: np.ndarray, dpi: float = WORK_DPI, lineart_boxes: list | None = None
) -> Field | None:
    """Поле смещений B → A на серых копиях одного dpi (размеры могут отличаться).

    ``lineart_boxes`` — рамки рисунков на B в пикселях ``dpi``: внутри них тайлы не
    подбираются по масштабу (см. ``_tile_matches``).
    """
    boxes = list(lineart_boxes or [])
    h = max(before.shape[0], after.shape[0])
    w = max(before.shape[1], after.shape[1])
    b = _pad_to(before, h, w)
    a = _pad_to(after, h, w)
    dx, dy = _global_shift(b, a)
    tile, step = mm_to_px(TILE_MM, dpi), mm_to_px(STEP_MM, dpi)
    radius = mm_to_px(PEAK_RADIUS_MM, dpi)

    coarse, _ = _tile_matches(
        b, a, lambda x, y: (x + dx, y + dy), tile, step * 2, mm_to_px(COARSE_SEARCH_MM, dpi), radius, boxes
    )
    if len(coarse) < MIN_TILES:
        return None
    affine, _, _ = robust_affine(coarse[:, :2], coarse[:, :2] + coarse[:, 2:4])

    def predict(x: float, y: float) -> tuple[float, float]:
        p = affine @ np.array([x, y, 1.0])
        return float(p[0]), float(p[1])

    tiles, failed = _tile_matches(b, a, predict, tile, step, mm_to_px(FINE_SEARCH_MM, dpi), radius, boxes)
    if len(tiles) < MIN_TILES:
        return None
    affine, resid, weight = robust_affine(tiles[:, :2], tiles[:, :2] + tiles[:, 2:4])
    return Field(before.shape[1], before.shape[0], dpi, tiles, affine, resid, weight, failed)


def field_metrics(
    field: Field | None, lineart_boxes: list, text_boxes: list, raster_boxes: list | None = None
) -> dict[str, float]:
    """Метрики поля: аффинная часть, остатки (в мм бумаги) по всей странице и по типам областей.

    Args:
        field: Поле смещений или ``None`` (тогда только ``field_tiles = 0``).
        lineart_boxes: Рамки рисунков на B: доля их тайлов без пары — ``field_lineart_weak_frac``.
        text_boxes: Рамки строк текста на B.
        raster_boxes: Рамки растра (фотографий) на B: доля их тайлов без пары —
            ``field_raster_weak_frac``, отдельно от рисунков и только для сводок: фотография —
            растровая сетка, и часть её тайлов не находит пару и без всякой порчи (1970/12
            с.76, 1975/04 с.2 — по 50 % при целом снимке), у перекошенного — 55–75 % (1971/04
            с.44, 1970/10 с.71, 1971/07 с.43); порог тут не ставится, порчу снимка ловят
            кромки (``raster.py``).

    Returns:
        Плоский словарь метрик ``field_*``.
    """
    if field is None:
        return {"field_tiles": 0.0}
    raster_boxes = list(raster_boxes or [])
    weak_raster, raster_total = field.weak_frac_inside(raster_boxes)
    mm = px_to_mm(1.0, field.dpi)
    norm = field.resid_norm[field.weight > 0] * mm
    u = field.tiles[:, 2:4]
    change = np.hypot(*(u - np.median(u, axis=0)).T) * mm
    lineart = field.resid_inside(lineart_boxes) * mm
    text = field.resid_inside(text_boxes) * mm
    weak_lineart, lineart_total = field.weak_frac_inside(lineart_boxes)
    weak_text, text_total = field.weak_frac_inside(text_boxes)

    def p90(values: np.ndarray) -> float:
        return float(np.percentile(values, 90)) if values.size else 0.0

    return {
        "field_tiles": float(len(field.tiles)),
        "field_inliers": float(int((field.weight > 0).sum())),
        "field_rot_deg": field.rot_deg,
        "field_shear_deg": field.shear_deg,
        "field_scale_x": field.scale_x,
        "field_scale_y": field.scale_y,
        "field_change_p90_mm": p90(change),
        "field_resid_p90_mm": p90(norm),
        "field_resid_max_mm": float(norm.max()) if norm.size else 0.0,
        "field_weak_frac": (
            float((field.failed[:, 2] == 0).sum() / (len(field.tiles) + len(field.failed)))
            if len(field.failed)
            else 0.0
        ),
        "field_lineart_tiles": float(lineart_total),
        "field_resid_lineart_p90_mm": p90(lineart),
        "field_resid_lineart_max_mm": float(lineart.max()) if lineart.size else 0.0,
        "field_lineart_weak_frac": weak_lineart if lineart_total >= MIN_LINEART_TILES else 0.0,
        "field_raster_tiles": float(raster_total),
        "field_raster_weak_frac": weak_raster if raster_total >= MIN_LINEART_TILES else 0.0,
        "field_text_tiles": float(text_total),
        "field_resid_text_p90_mm": p90(text),
        "field_text_weak_frac": weak_text,
    }
