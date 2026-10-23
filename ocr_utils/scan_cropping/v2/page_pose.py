"""Страницы кадра: охватывающая трапеция каждой и класс «лежит плоско / приподнята / опущена» по сходу сторон трапеции."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.glue import outside_distance, split_pages
from ocr_utils.scan_cropping.v2.keystone import vertical_sides
from ocr_utils.scan_cropping.v2.trapezoid import fit_enclosing_quad, keystone_deg, quad_mask

# Плотность масштаба сегментации, px/мм (масштаб сегментации всегда приведён к калибровочным кадрам МТО 1990).
SEG_PX_PER_MM = 6.9
# Куски бумаги меньше этой доли всей бумаги — не страницы (обрывки у края, как в ``research/cut_lines``).
MIN_PAGE_SHARE = 0.05
# Полоса высоты страницы, по строкам которой меряется ширина (доли от верха до низа страницы): середина, где нет
# ни выгиба углов, ни пальцев у краёв.
WIDTH_ROWS = (0.3, 0.7)
# Припуск «касания» зоны закраса пальца к странице, px сегментации.
FINGER_TOUCH_PX = 10
# Перцентиль отклонения контура страницы от стороны трапеции внутрь (мера выгиба края), %.
CURL_PERCENTILE = 95

# Пороги правила «приподнята» (см. :func:`classify_poses`). Приподнятая страница наклонена к камере: её трапеция
# сильно сходится (верх и низ не параллельны), у плоской — нет. Признаки всех страниц трёх папок пака-6 (МТС 1990/03
# и 1991/05, РИСК 1991/05; отсмотр 2026-10-12): плоские страницы обычных разворотов — сход 1.6–6.6°; приподнятая
# рукой страница «фото по страницам» — 13–35°, слегка приподнятая (строки у внешнего края уходят вниз, IMG_1018) —
# 9–13°. Ширина и свет к соседней странице не разделяют (приподнятая ближе к камере и в проекции не уже).
RAISED_MIN_KEYSTONE = 8.0
# Сход приподнятой страницы больше, чем у соседней, хотя бы на столько градусов (общая перспектива кадра — у обеих).
RAISED_MIN_KEYSTONE_GAP = 4.0

# Знак схода. Угол схода (``keystone_deg``) без знака: страница, чей внешний край свисает с опоры и уходит от камеры
# (опущена, МТС 1990/01 IMG_0875), сходится так же сильно, как приподнятая, только в другую сторону. Знак — отношение
# длины внешней стороны трапеции (от сгиба дальше) к стороне у сгиба (``PageFeatures.outer_ratio``): у приподнятой
# внешний край ближе к камере и длиннее. Пак-6 (кэш pack6_v6, 4 917 страниц двухстраничных кадров; отсмотр 2026-10-21):
# плоские — 0.91–1.10 (медиана 0.95: у сгиба книга выше стола); приподнятые по сходу — 807 страниц 1.03–1.45 (поднята
# рукой) и 50 страниц 0.905–0.93 (опущена: внешний край свисает, отсмотрены IMG_0875, 0270, 0562, 0715, 1324);
# между 0.95 и 1.03 — только две одиночные обложки РИСК, ошибочно разрезанные на две «страницы». Опущенной страница
# считается, если её отношение не больше этого и меньше, чем у соседней (иначе остаётся «приподнята»).
LOWERED_MAX_OUTER_RATIO = 0.95


class PagePose(Enum):
    """Положение страницы в кадре."""

    FLAT = "flat"
    RAISED = "raised"
    LOWERED = "lowered"

    @property
    def on_book(self) -> bool:
        """Страница лежит на блоке (плоско или свисает внешним краем) — её контент защищается, как у обычной."""
        return self is not PagePose.RAISED


@dataclass(frozen=True)
class PageFeatures:
    """Признаки страницы для классификации положения.

    Поля:
        fill: Площадь страницы / площадь её трапеции (выгнутая страница заполняет трапецию хуже).
        keystone: Наибольший угол между противоположными сторонами трапеции, ° (перспектива).
        width_share: Ширина страницы / ширина соседней по средней полосе строк; без соседа — 1.
        light_share: Медианная яркость страницы / яркость соседней; без соседа — 1.
        finger: Доля зоны закраса пальцев, касающейся страницы.
        curl: Отклонение края страницы внутрь от сторон трапеции (перцентиль ``CURL_PERCENTILE``), мм.
        outer_ratio: Длина внешней стороны трапеции (дальней от сгиба) к стороне у сгиба — знак схода: больше
            соседней — внешний край ближе к камере (приподнята), меньше — дальше (опущена); 1 — без трапеции или
            страница не крайняя.
    """

    fill: float
    keystone: float
    width_share: float
    light_share: float
    finger: float
    curl: float
    outer_ratio: float = 1.0


@dataclass(frozen=True)
class PageQuad:
    """Страница кадра: маска, трапеция, признаки и класс положения.

    Поля:
        mask: Маска страницы (масштаб сегментации).
        quad: Вершины трапеции (4, 2) px сегментации по часовой стрелке или None (вырожденная).
        features: Признаки.
        pose: Класс положения.
    """

    mask: np.ndarray
    quad: np.ndarray | None
    features: PageFeatures
    pose: PagePose


def frame_pages(paper: np.ndarray, book: np.ndarray, hulls: np.ndarray, inst: np.ndarray) -> list[np.ndarray]:
    """Маски страниц кадра слева направо, без мелких кусков.

    Args:
        paper: Итоговая маска бумаги (масштаб сегментации).
        book: Грубая маска книги.
        hulls: Оболочки своей печати.
        inst: Карта экземпляров-страниц SAM 3.

    Returns:
        Маски страниц, отсортированные по среднему X.
    """
    total = max(int(paper.sum()), 1)
    pages = [p for p in split_pages(paper, book, hulls, inst) if p.sum() >= MIN_PAGE_SHARE * total]
    return sorted(pages, key=lambda p: float(np.nonzero(p.any(axis=0))[0].mean()))


def mid_width(page: np.ndarray) -> float:
    """Ширина страницы по средней полосе строк: медиана числа пикселей страницы в строке.

    Args:
        page: Маска страницы.

    Returns:
        Ширина, px (0 для пустой маски).
    """
    rows = np.nonzero(page.any(axis=1))[0]
    if not len(rows):
        return 0.0
    lo = int(rows[0] + WIDTH_ROWS[0] * (rows[-1] - rows[0]))
    hi = int(rows[0] + WIDTH_ROWS[1] * (rows[-1] - rows[0])) + 1
    return float(np.median(page[lo:hi].sum(axis=1)))


def edge_curl(page: np.ndarray, quad: np.ndarray) -> float:
    """Отклонение контура страницы внутрь от сторон трапеции (без точек на рамке кадра), мм.

    Args:
        page: Маска страницы.
        quad: Трапеция (4, 2).

    Returns:
        Перцентиль ``CURL_PERCENTILE`` расстояния от точки контура до ближайшей стороны, мм.
    """
    contours, _ = cv2.findContours(page.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    pts = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    h, w = page.shape
    pts = pts[(pts[:, 0] > 1) & (pts[:, 1] > 1) & (pts[:, 0] < w - 2) & (pts[:, 1] < h - 2)]
    if not len(pts):
        return 0.0
    # Расстояние до ближайшей из четырёх сторон-прямых.
    dists = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-12)
        dists.append(np.abs((pts - a) @ n))
    return float(np.percentile(np.min(dists, axis=0), CURL_PERCENTILE)) / SEG_PX_PER_MM


def outer_ratio(quad: np.ndarray | None, outer_left: bool | None) -> float:
    """Знак схода: длина внешней «вертикальной» стороны трапеции к стороне у сгиба.

    Args:
        quad: Трапеция страницы или None.
        outer_left: True — внешний край слева (левая страница), False — справа (правая), None — страница не крайняя
            или одна в кадре.

    Returns:
        Отношение; 1 без трапеции или без внешнего края.
    """
    if quad is None or outer_left is None:
        return 1.0
    left, right = vertical_sides(quad)
    lengths = (float(np.linalg.norm(left[1] - left[0])), float(np.linalg.norm(right[1] - right[0])))
    outer, fold = lengths if outer_left else lengths[::-1]
    return outer / max(fold, 1e-6)


def page_features(
    page: np.ndarray,
    quad: np.ndarray | None,
    gray: np.ndarray,
    paint: np.ndarray,
    neighbor: np.ndarray | None,
    outer_left: bool | None = None,
) -> PageFeatures:
    """Признаки одной страницы.

    Args:
        page: Маска страницы.
        quad: Трапеция страницы или None.
        gray: Кадр без пальцев в сером (масштаб сегментации).
        paint: Зона закраса пальцев (bool, масштаб сегментации).
        neighbor: Маска соседней страницы (самой большой из остальных) или None.
        outer_left: Где внешний край страницы (``outer_ratio``): True — слева, False — справа, None — нет.

    Returns:
        Признаки.
    """
    if quad is not None:
        area = max(int(quad_mask(quad, page.shape).sum()), 1)
        fill, keystone, curl = float(page.sum() / area), keystone_deg(quad), edge_curl(page, quad)
    else:
        fill, keystone, curl = 0.0, 0.0, 0.0
    width_share, light_share = 1.0, 1.0
    if neighbor is not None and neighbor.any():
        width_share = mid_width(page) / max(mid_width(neighbor), 1.0)
        light_share = float(np.median(gray[page])) / max(float(np.median(gray[neighbor])), 1.0)
    finger = 0.0
    if paint.any():
        k = 2 * FINGER_TOUCH_PX + 1
        near = cv2.dilate(page.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        finger = float((paint & near).sum() / paint.sum())
    return PageFeatures(fill, keystone, width_share, light_share, finger, curl, outer_ratio(quad, outer_left))


def raised_by_rule(f: PageFeatures, neighbor: PageFeatures) -> bool:
    """Правило «страница приподнята»: трапеция сильно сходится, и сильнее, чем у соседней страницы.

    Args:
        f: Признаки страницы.
        neighbor: Признаки соседней страницы.

    Returns:
        True — приподнята.
    """
    return f.keystone >= RAISED_MIN_KEYSTONE and f.keystone - neighbor.keystone >= RAISED_MIN_KEYSTONE_GAP


def lowered_by_sign(f: PageFeatures, neighbor: PageFeatures) -> bool:
    """Знак схода наклонённой страницы: внешний край короче, чем у плоской, — свисает с опоры, а не поднят.

    Args:
        f: Признаки страницы, наклонённой по ``raised_by_rule``.
        neighbor: Признаки соседней страницы.

    Returns:
        True — опущена.
    """
    return f.outer_ratio <= LOWERED_MAX_OUTER_RATIO and f.outer_ratio < neighbor.outer_ratio


def classify_poses(features: list[PageFeatures]) -> list[PagePose]:
    """Классы положения страниц кадра.

    Одна страница в кадре — лежит плоско. Наклонённой (по сходу) может быть только одна страница: если правило
    отметило несколько, все считаются плоскими (правило сбилось — защита контента остаётся на всех). Соседняя
    страница для правила — с наименьшим сходом среди остальных. Наклонённая страница — опущена, если по знаку схода
    её внешний край уходит от камеры (``lowered_by_sign``), иначе приподнята.

    Args:
        features: Признаки страниц кадра.

    Returns:
        Классы в том же порядке.
    """
    if len(features) < 2:
        return [PagePose.FLAT] * len(features)
    tilted, neighbors = [], []
    for k, f in enumerate(features):
        others = [g for j, g in enumerate(features) if j != k]
        neighbors.append(min(others, key=lambda g: g.keystone))
        tilted.append(raised_by_rule(f, neighbors[-1]))
    if sum(tilted) != 1:
        return [PagePose.FLAT] * len(features)
    return [
        (PagePose.LOWERED if lowered_by_sign(f, nb) else PagePose.RAISED) if t else PagePose.FLAT
        for f, nb, t in zip(features, neighbors, tilted)
    ]


def page_quads(
    paper: np.ndarray, book: np.ndarray, hulls: np.ndarray, inst: np.ndarray, gray: np.ndarray, paint: np.ndarray
) -> list[PageQuad]:
    """Страницы кадра с трапециями, признаками и классом положения.

    Args:
        paper: Итоговая маска бумаги (масштаб сегментации).
        book: Грубая маска книги.
        hulls: Оболочки своей печати.
        inst: Карта экземпляров-страниц SAM 3.
        gray: Кадр без пальцев в сером (масштаб сегментации).
        paint: Зона закраса пальцев (bool, масштаб сегментации).

    Returns:
        Страницы слева направо.
    """
    pages = frame_pages(paper, book, hulls, inst)
    quads = [fit_enclosing_quad(p) for p in pages]
    features = []
    for k, page in enumerate(pages):
        others = [p for j, p in enumerate(pages) if j != k]
        neighbor = max(others, key=lambda p: int(p.sum())) if others else None
        # Внешний край — у крайних страниц разворота: левая — слева, правая — справа.
        outer_left = None if len(pages) < 2 else True if k == 0 else False if k == len(pages) - 1 else None
        features.append(page_features(page, quads[k], gray, paint, neighbor, outer_left))
    poses = classify_poses(features)
    return [PageQuad(p, q, f, c) for p, q, f, c in zip(pages, quads, features, poses)]


def page_owner(pages: list[PageQuad], shape: tuple[int, int]) -> np.ndarray:
    """Карта «чья страница ближе» на весь кадр: индекс ближайшей страницы для каждого пикселя.

    Args:
        pages: Страницы кадра (непустой список).
        shape: ``(H, W)`` масштаба сегментации.

    Returns:
        Карта индексов int (H, W).
    """
    return np.stack([outside_distance(p.mask) for p in pages]).argmin(axis=0)
