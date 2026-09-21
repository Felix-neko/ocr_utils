"""Сдвиг из якобиана поля смещений: блок текста стал параллелограммом.

Разбор 28 страниц «равнение текста по краю перекосило» (2026-09-22): строки в A горизонтальны,
а левая и правая кромки блока ушли на 1.5–5 мм на колонке 220 мм — FineReader сдвинул
страницу вдоль x пропорционально y. По тайлам поля смещений это видно как компонента du_x/dy
локального якобиана (наклон образа вертикали); на 28 страницах её p90 по тайлам — медиана
0.96°, на 117 случайных ok-страницах — медиана 0.01°, p90 0.67°. Считается как в DIC
(digital image correlation): вокруг каждого тайла по соседям в радиусе ``NEIGHBOUR_MM``
подгоняется локальный аффин смещения, его градиент даёт сдвиг и поворот. Глобальный сдвиг
аффинной части (``Field.shear_deg``) — частный случай; здесь берётся локальный, потому что
FineReader перекашивает и половину страницы (1973/09 с.76 — одна колонка).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ocr_utils.geometry_regression import mm_to_px, px_to_mm
from ocr_utils.geometry_regression.field import Field

Box = tuple[int, int, int, int]

# Радиус соседей для локального аффина: 45 мм — 5–7 тайлов при шаге 17 мм, хватает для
# устойчивого градиента и ещё не размывает перекос одной колонки (ширина колонки ~80 мм).
NEIGHBOUR_MM = 45.0
# Меньше стольких соседей — градиент не считается.
MIN_NEIGHBOURS = 6
# Тайлов текста меньше — сводка по странице не считается (две-три строки подписи — не блок).
MIN_TEXT_TILES = 6
# Виновник на оверлее: тайлы со сдвигом не меньше этой доли от максимума по странице.
CULPRIT_FRAC = 0.7


@dataclass(frozen=True)
class ShearMap:
    """Локальный сдвиг и поворот по тайлам поля (градусы), в порядке ``centres``."""

    centres: np.ndarray  # N × 2, пиксели поля
    shear_deg: np.ndarray  # сдвиг сверх поворота: положительный — низ уехал вправо относительно строк
    rot_deg: np.ndarray  # локальный поворот


def shear_map(field: Field, dpi: float) -> ShearMap | None:
    """Карта сдвига по тайлам поля с весом > 0.

    Args:
        field: Поле смещений B → A.
        dpi: Разрешение поля (``field.dpi``), чтобы радиус соседей задать в мм.

    Returns:
        :class:`ShearMap` или ``None``, если тайлов с соседями меньше ``MIN_NEIGHBOURS``.
    """
    good = field.weight > 0
    if good.sum() < MIN_NEIGHBOURS:
        return None
    pts = field.tiles[good, :2]
    u = field.tiles[good, 2:4]
    tree = cKDTree(pts)
    radius = mm_to_px(NEIGHBOUR_MM, dpi)
    centres, shears, rots = [], [], []
    for i, p in enumerate(pts):
        idx = tree.query_ball_point(p, radius)
        if len(idx) < MIN_NEIGHBOURS:
            continue
        design = np.c_[pts[idx] - p, np.ones(len(idx))]
        # Градиент смещения по МНК: u_x ≈ gx·[dx, dy, 1], u_y ≈ gy·[dx, dy, 1].
        gx = np.linalg.lstsq(design, u[idx, 0], rcond=None)[0]
        gy = np.linalg.lstsq(design, u[idx, 1], rcond=None)[0]
        centres.append(p)
        # Симметричная часть градиента: du_x/dy + du_y/dx — сдвиг (у чистого поворота du_x/dy =
        # −du_y/dx и сумма 0; у сдвига вдоль x — сам коэффициент); антисимметричная — поворот.
        shears.append(np.degrees(np.arctan(gx[1] + gy[0])))
        rots.append(np.degrees(np.arctan((gy[0] - gx[1]) / 2.0)))
    if not centres:
        return None
    return ShearMap(np.array(centres), np.array(shears), np.array(rots))


def _inside(points: np.ndarray, boxes: list[Box]) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        inside |= (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
    return inside


def shear_metrics(
    field: Field | None, text_boxes: list[Box], exclude_boxes: list[Box], dpi: float
) -> tuple[dict[str, float], dict]:
    """Сводка сдвига по тайлам текста страницы и рамка виновника.

    Args:
        field: Поле смещений B → A (``None`` — метрики нулевые).
        text_boxes: Рамки строк текста на B (пиксели ``dpi``): тайл текстовый, если его центр в одной из них.
        exclude_boxes: Рамки растра, line art и таблиц: их тайлы в сводку по тексту не идут.
        dpi: Разрешение поля.

    Returns:
        Метрики: ``field_shear_p90_deg`` — p90 |сдвига| по тайлам текста; ``field_shear_max_deg``
        — максимум; ``field_shear_tiles`` — сколько тайлов текста; ``field_rot_local_p90_deg`` —
        p90 |локального поворота| (контекст). Виновник — рамка тайлов с наибольшим сдвигом.
    """
    metrics = {
        "field_shear_p90_deg": 0.0,
        "field_shear_max_deg": 0.0,
        "field_shear_tiles": 0.0,
        "field_rot_local_p90_deg": 0.0,
    }
    culprits: dict = {}
    if field is None:
        return metrics, culprits
    shear = shear_map(field, dpi)
    if shear is None:
        return metrics, culprits
    text = _inside(shear.centres, text_boxes) & ~_inside(shear.centres, exclude_boxes)
    metrics["field_shear_tiles"] = float(text.sum())
    if text.sum() < MIN_TEXT_TILES:
        return metrics, culprits
    values = np.abs(shear.shear_deg[text])
    metrics["field_shear_p90_deg"] = float(np.percentile(values, 90))
    metrics["field_shear_max_deg"] = float(values.max())
    metrics["field_rot_local_p90_deg"] = float(np.percentile(np.abs(shear.rot_deg[text]), 90))
    # Виновник: тайлы с сильным сдвигом одним прямоугольником, в обеих версиях по полю.
    strong = shear.centres[text][values >= CULPRIT_FRAC * values.max()]
    half = mm_to_px(13.5, dpi)
    box_b = (
        int(strong[:, 0].min() - half),
        int(strong[:, 1].min() - half),
        int(strong[:, 0].max() + half),
        int(strong[:, 1].max() + half),
    )
    moved = field.transform(strong)
    box_a = (
        int(moved[:, 0].min() - half),
        int(moved[:, 1].min() - half),
        int(moved[:, 0].max() + half),
        int(moved[:, 1].max() + half),
    )
    culprits["field_shear_p90_deg"] = {"b": box_b, "a": box_a}
    return metrics, culprits


def shear_inside(shear: ShearMap | None, box: Box) -> float:
    """Медианный сдвиг (градусы, со знаком) по тайлам внутри рамки; 0, если тайлов нет."""
    if shear is None:
        return 0.0
    inside = _inside(shear.centres, [box])
    return float(np.median(shear.shear_deg[inside])) if inside.any() else 0.0


def shear_to_mm(shear_deg: float, height_px: float, dpi: float) -> float:
    """Уход кромки блока в мм: сдвиг × высота блока."""
    return px_to_mm(height_px, dpi) * abs(float(np.sin(np.radians(shear_deg))))


__all__ = ["ShearMap", "shear_map", "shear_metrics", "shear_inside", "shear_to_mm"]
