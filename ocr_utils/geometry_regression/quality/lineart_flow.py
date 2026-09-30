"""Порча line art по плотному полю смещений B → A внутри рамки рисунка: AAD-подобный разброс по строкам и столбцам и искажение по якобиану.

ЗАЧЕМ. Семейство line art стенда v16 (``ocr_utils/geometry_regression/v16/lineart.py``) меряет линии рисунка по
отрезкам LSD и исключает рамку, внутри которой четыре и больше строк текста («врезка»). Блок-схемы с подписями
в блоках (1967/10 с.74, 1974/12 с.34, эталон bad) поэтому не мерились вовсе (``lineart_strokes`` = 0), и
перекошенный столбец блоков оставался невидим. Здесь рамка мерится целиком, текст в ней не мешает.

КАКОЙ РИСУНОК С КАКИМ. Пар «объект B ↔ объект A» не ищется: рамка берётся в B, а в A — то же место после
выравнивания A в кадр B (режим ``Align``). Трапеция FineReader — проективное искажение, аффинной матрицей на
всю страницу не снимается, и у дальнего края полосы остаётся сдвиг в миллиметры (замечание пользователя
2026-09-29), поэтому выравнивание сравнивается в трёх режимах: аффинная часть поля, гомография по тайлам поля
(RANSAC), гомография плюс местный сдвиг рамки по остатку тайлов в ней. Что в A на этом месте тот же рисунок,
проверяет корреляция размытой краски после выравнивания (``MIN_NCC``); ниже — рамка «не найдена», AAD по ней
не считается.

МЕТОД.

1. Рендеры B и A 150 dpi; A переводится в кадр B (``Align``).
2. Плотное поле B → A' внутри рамки (``cv2.DISOpticalFlow``) по размытой краске; из него снимается аффинная
   часть, подогнанная по пикселям краски этой рамки (остаток трапеции в пределах одного рисунка почти аффинный):
   остаток — неригидная деформация рисунка.
3. **AAD** (Wang и др., «Axis-Aligned Document Dewarping», AAAI 2026, arXiv 2507.15000): в каждой строке рамки
   вертикальное смещение пикселей горизонтальных черт (вес — |Sobel_y| B) должно быть одним и тем же, в каждом
   столбце — горизонтальное смещение вертикальных черт. Мера — взвешенное среднее |отклонения| (мм) и его p95.
4. **Якобиан** (Hormann, Polthier, Sheffer, «Mesh parameterization: theory and practice»): поле сглаживается
   гауссом, ``J = I + ∇u``, сингулярные числа σ1 ≥ σ2; конформное искажение ``σ1/σ2 − 1`` и угол перекоса на
   пикселях краски, p95 (для отчёта: на логотипах рубрик шумит).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from ocr_utils.geometry_regression.field import Field
from ocr_utils.page_layout import px_to_mm

# Рабочее разрешение рендеров и поля.
WORK_DPI = 150.0
# Поле по размытой краске: гаусс, пиксели 150 dpi (штрих 1–2 px превращается в гладкий «холм»).
INK_BLUR_PX = 1.5
# Сглаживание поля перед якобианом, мм: деформация бумаги и перекос блоков — масштаб сантиметров, шум — пикселей.
JACOBIAN_SIGMA_MM = 3.0
# Рамка меньше — не меряется, мм: короткая сторона и длинная. Рисунки — от 20 мм по обеим; формулы (строка
# формулы 5–10 мм высотой) — короткая от 3 мм, длинная от 15 мм (``FORMULA_MIN_SIDE_MM``, ``FORMULA_MIN_LONG_MM``).
MIN_BOX_MM = 20.0
FORMULA_MIN_SIDE_MM = 3.0
FORMULA_MIN_LONG_MM = 15.0
# Поля вокруг рамки при вырезке, мм.
PAD_MM = 3.0
# Порог краски (яркость ниже — краска) и порог силы градиента в долях максимума для весов AAD.
INK_LEVEL = 160
GRAD_FRAC = 0.2
# Корреляция размытой краски B и выровненной A ниже — в A на этом месте другой рисунок (или он потерян).
MIN_NCC = 0.3
# Гомография по тайлам: порог RANSAC, пиксели 150 dpi, и минимум тайлов.
RANSAC_PX = 3.0
MIN_TILES = 8
# Местный сдвиг рамки — медиана остатка тайлов после гомографии в рамке, если тайлов там не меньше.
LOCAL_MIN_TILES = 3
# Рамки, перекрытые больше этой доли меньшей из двух, сливаются в одну (v16 и v6 дают одну и ту же схему).
MERGE_OVERLAP = 0.5
# Варианты мер «только по линиям» (сравнение на эталоне 2026-09-29):
# длинные горизонтальные и вертикальные черты — открытие ядром такой длины, мм (буквы короче);
LINE_OPEN_MM = 3.0
# отрезки LSD любой ориентации (наклонные — лучи, стрелки, скосы соединителей) — от такой длины, мм;
SEGMENT_MIN_MM = 5.0
# соседние параллельные отрезки для относительного сдвига: разница направлений, градусы, расстояние между
# серединами, мм, и минимум расстояния поперёк, мм (ближе — две кромки одного штриха, их не сравнивать).
PARALLEL_TOL_DEG = 5.0
NEIGHBOUR_MM = 40.0
SAME_STROKE_MM = 1.0
# Черты (дробные черты формул, длинные горизонтали рисунков): отрезки LSD от BAR_MIN_MM, отклонённые от горизонтали
# не больше BAR_MAX_DEG; мера — размах поперечного смещения вдоль черты (изгиб, перелом относительно рамки).
BAR_MIN_MM = 8.0
BAR_MAX_DEG = 5.0
# Форма рисунка (v18, ``shape_measures``): модели поля рамки подгоняются по пикселям краски, не больше SHAPE_MAX_POINTS
# (случайная выборка с фиксированным зерном); точки с остатком больше SHAPE_TRIM_PX (промахи сопоставления)
# выбрасываются, модель подгоняется заново — до SHAPE_TRIM_ROUNDS раз. Уход моделей сравнивается на контуре
# охвата краски (перцентили SHAPE_HULL_PCT и 100 − SHAPE_HULL_PCT по x и y), SHAPE_BORDER_STEPS точек на сторону.
SHAPE_MAX_POINTS = 30000
SHAPE_TRIM_PX = 2.0
SHAPE_TRIM_ROUNDS = 3
# Половина рисунка для поворота частей — от стольких точек.
SHAPE_PART_MIN = 12
SHAPE_HULL_PCT = 2.0
SHAPE_BORDER_STEPS = 12
# Абсолютный наклон рамки по отрезкам LSD (v18): отрезки от TILT_SEGMENT_MM, отклонённые от ближайшей оси не больше
# TILT_AXIS_DEG, суммарно не короче TILT_TOTAL_MM; иначе — по профилю краски (``ink_angle``).
TILT_SEGMENT_MM = 5.0
TILT_AXIS_DEG = 10.0
TILT_TOTAL_MM = 30.0
# Перенос оси строки в A (v18, ``transfer_axis``): полоса вокруг оси — TRANSFER_BAND высот строки вверх и вниз и
# TRANSFER_PAD_MM по краям; поле усредняется по краске гауссом TRANSFER_SIGMA высот строки (не меньше 2 px).
TRANSFER_BAND = 1.5
TRANSFER_PAD_MM = 3.0
# Полоса не ниже TRANSFER_MIN_BAND_MM вверх и вниз от оси, а вырезка — не меньше DIS_MIN_PX по каждой стороне: на
# полосе 14 px (ось высотой 4 px, 1967/04 с.97) ``DISOpticalFlow.calc`` падает с segfault и валит воркер пула.
TRANSFER_MIN_BAND_MM = 3.0
DIS_MIN_PX = 32
TRANSFER_SIGMA = 0.5
# Опора формы (v18, ``text_reference``): тайлы поля текста в кольце REF_RING_MM вокруг рамки; если их меньше
# REF_MIN_TILES — тайлы и внутри рамки (рамка «неясно» на всю полосу бланка, 1974/02 с.95). Модель опоры —
# квадратичная от REF_QUAD_TILES тайлов, аффинная от REF_MIN_TILES, иначе — медианный сдвиг (от одного тайла).
REF_RING_MM = 40.0
REF_MIN_TILES = 6
REF_QUAD_TILES = 12
# Прямота длинных линий рисунка (v18, ``straightness``): открытие краски ядром STRAIGHT_OPEN_MM оставляет длинные
# горизонтальные и вертикальные черты; черта от STRAIGHT_MIN_MM мерится размахом средней линии вокруг прямой в B и
# в A'. Если длиновзвешенная медиана изменения размаха ниже −STRAIGHT_GAIN_MM, линии рисунка стали прямее: порча
# формы не засчитывается (1973/05 с.91, 1975/04 с.95 — FineReader выпрямил кривой росчерк логотипа, эталон good).
STRAIGHT_OPEN_MM = 8.0
STRAIGHT_MIN_MM = 15.0
STRAIGHT_GAIN_MM = 0.15
# Соответствия для формы (v18, ``patch_matches``): участки PATCH_MM с шагом PATCH_STEP_MM, с долей краски в
# [PATCH_INK_MIN, PATCH_INK_MAX] (пустые и заливки не сопоставляются), поиск в A' в пределах ±PATCH_SEARCH_MM;
# берутся участки с пиком корреляции не ниже PATCH_MIN_PEAK, у которых второй пик за пределами PATCH_SECOND_MM ниже
# первого хотя бы на PATCH_PEAK_GAP (периодика — штриховка, решётка окон — даёт несколько равных пиков).
PATCH_MM = 10.0
PATCH_STEP_MM = 5.0
PATCH_INK_MIN = 0.04
PATCH_INK_MAX = 0.6
PATCH_SEARCH_MM = 2.5
PATCH_MIN_PEAK = 0.6
PATCH_SECOND_MM = 1.0
PATCH_PEAK_GAP = 0.08
# Участков на рамку после отсева промахов не меньше: у квадратичной модели 12 параметров, и на 13 точках она почти
# интерполирует, отсев ничего не отсеивает (портрет 1970/04 с.28 — перекос «12°»).
PATCH_MIN_COUNT = 30
# Грубый сдвиг рамки целиком перед участками: поиск по корреляции краски в пределах ±FRAME_SEARCH_MM от опоры
# (портрет 1970/04 с.28 в A сдвинут относительно соседнего текста на 20 px — дальше окна участков).
FRAME_SEARCH_MM = 10.0
# Участки после отсева промахов не должны лежать в одну полосу: меньшее из их стандартных отклонений по осям —
# не меньше этой доли охвата (иначе перекос и изгиб модели не определены: портрет 1970/04 с.28, перекос «12°»).
SHAPE_MIN_SPREAD = 0.15
# Рамка больше этой доли полосы меряется по форме, только если опора — тайлы текста вокруг рамки (кольцо): у бланка
# на всю полосу (1974/02 с.95) кольца нет, а тайлы внутри — сама выправленная полоса; оргсхема 1967/10 с.74 (0.68
# полосы) окружена текстом.
SHAPE_MAX_PAGE_SHARE = 0.4
# Меры формы в метриках страницы (приставка — ``lineart`` или ``formula``): групповая мера (после проверки
# прямоты), её части, сырая групповая мера, изменение прямоты и число длинных черт.
SHAPE_NAMES = (
    "shape_mm", "nonsim_mm", "curve_mm", "shape_shear_deg", "shape_aniso", "shape_raw_mm", "straight_delta_mm",
    "straight_lines", "patches", "part_turn_deg", "part_turn_mm",
)  # fmt: skip


class Align(str, Enum):
    """Как A переводится в кадр B перед плотным полем рамки."""

    AFFINE = "affine"  # аффинная часть поля v16
    HOMOGRAPHY = "homography"  # гомография по тайлам поля (снимает трапецию целиком)
    LOCAL = "local"  # гомография плюс местный сдвиг рамки по остатку тайлов в ней


DEFAULT_ALIGN = Align.LOCAL


@dataclass(frozen=True)
class FlowMeasure:
    """Меры одной рамки line art.

    Attributes:
        box: Рамка в B, пиксели 150 dpi.
        ncc: Корреляция размытой краски B и выровненной A в рамке.
        found: Рисунок в A на этом месте найден (``ncc ≥ MIN_NCC``); иначе меры AAD и якобиана — нули.
        aad_mean_mm: Взвешенное среднее отклонение AAD, мм.
        aad_p95_mm: p95 отклонения AAD по пикселям черт, мм.
        aniso_p95: p95 конформного искажения ``σ1/σ2 − 1`` на пикселях краски.
        shear_p95_deg: p95 угла перекоса (отклонение угла между образами осей от 90°), градусы.
    """

    box: tuple[float, float, float, float]
    ncc: float
    found: bool
    aad_mean_mm: float = 0.0
    aad_p95_mm: float = 0.0
    aniso_p95: float = 0.0
    shear_p95_deg: float = 0.0
    # Варианты «только по линиям»: AAD по длинным осевым чертам, изгиб отрезков LSD любой ориентации (смещение
    # поперёк отрезка вдоль него, p90, среднее по длине), относительный поперечный сдвиг соседних параллельных
    # отрезков (p95) — мм.
    aad_lines_mm: float = 0.0
    seg_bend_mm: float = 0.0
    seg_rel_mm: float = 0.0
    # Худшая черта рамки: размах поперечного смещения вдоль длинного почти горизонтального отрезка, мм.
    bar_ptp_mm: float = 0.0
    # Собственная аффинная часть поля рамки после выравнивания страницы: поворот горизонталей (наклон рамки целиком
    # относительно страницы, градусы) и уход её конца по ширине рамки, мм.
    tilt_deg: float = 0.0
    tilt_mm: float = 0.0
    # Абсолютный наклон горизонталей рамки к оси изображения: в B — по профилю краски, в A — наклон B плюс поворот
    # полного отображения B → A в центре рамки (страница + собственная аффинная часть рамки), градусы; ухудшение
    # наклона — уход конца рамки, мм: ширина × (|tg θ_A| − |tg θ_B|) (минус — рамку выпрямило).
    abs_tilt_b_deg: float = 0.0
    abs_tilt_a_deg: float = 0.0
    skew_mm: float = 0.0
    # Для оверлея: отрезки LSD ``(n, 5)`` — концы в пикселях вырезки и сдвиг относительно соседей (px, NaN — без соседей).
    segments: np.ndarray | None = None
    # Для оверлея (только при ``keep_maps``): вырезки B и A в кадре B, карта отклонения AAD (px) и местный сдвиг.
    crop_b: np.ndarray | None = None
    crop_a: np.ndarray | None = None
    heat: np.ndarray | None = None
    shift: tuple[int, int] = (0, 0)


def _ink(gray: np.ndarray) -> np.ndarray:
    """Краска как «высота»: 255 − яркость, размытая гауссом (float32)."""
    return cv2.GaussianBlur((255 - gray).astype(np.float32), (0, 0), INK_BLUR_PX)


def homography(field: Field) -> tuple[np.ndarray | None, np.ndarray]:
    """Гомография B → A по тайлам поля (RANSAC) и остатки тайлов после неё.

    Args:
        field: Поле v16: центры тайлов в B и их смещения.

    Returns:
        ``(H 3×3 или None, остатки (n, 2))``: остаток — где тайл на самом деле минус где его ставит гомография;
        ``None``, если тайлов с весом меньше ``MIN_TILES`` или RANSAC не сошёлся.
    """
    good = field.weight > 0
    src = field.tiles[good, :2].astype(np.float64)
    dst = src + field.tiles[good, 2:4]
    if len(src) < MIN_TILES:
        return None, np.zeros((0, 2))
    matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if matrix is None:
        return None, np.zeros((0, 2))
    projected = cv2.perspectiveTransform(field.tiles[:, :2].reshape(-1, 1, 2).astype(np.float64), matrix).reshape(-1, 2)
    return matrix, field.tiles[:, :2] + field.tiles[:, 2:4] - projected


def align_page(
    gray_b: np.ndarray, gray_a: np.ndarray, field: Field | None, mode: Align
) -> tuple[np.ndarray, np.ndarray]:
    """A в кадре B и остатки тайлов поля после выравнивания (для местного сдвига).

    Args:
        gray_b: Рендер B.
        gray_a: Рендер A.
        field: Поле v16 B → A; ``None`` — A кладётся как есть.
        mode: Режим выравнивания.

    Returns:
        ``(A в кадре B, остатки тайлов (n, 2) в порядке field.tiles)``; без поля — пустые остатки.
    """
    h, w = gray_b.shape
    if field is None:
        aligned = np.full_like(gray_b, 255)
        hh, ww = min(h, gray_a.shape[0]), min(w, gray_a.shape[1])
        aligned[:hh, :ww] = gray_a[:hh, :ww]
        return aligned, np.zeros((0, 2))
    if mode is not Align.AFFINE:
        matrix, resid = homography(field)
        if matrix is not None:
            # Пиксель B p берётся из A в H·p (флаг обратного отображения).
            aligned = cv2.warpPerspective(
                gray_a, matrix, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=255
            )
            return aligned, resid
    aligned = cv2.warpAffine(
        gray_a, field.affine.astype(np.float64), (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=255
    )
    centres = field.tiles[:, :2]
    return aligned, field.tiles[:, :2] + field.tiles[:, 2:4] - (centres @ field.affine[:, :2].T + field.affine[:, 2])


def page_matrix(field: Field | None, mode: Align) -> np.ndarray:
    """Матрица 3×3 отображения страницы B → A, которой :func:`align_page` переводит A в кадр B.

    Args:
        field: Поле v16; ``None`` — тождественное отображение.
        mode: Режим выравнивания: гомография по тайлам (если сошлась) или аффинная часть поля.

    Returns:
        Матрица 3×3 (однородные координаты, пиксели 150 dpi).
    """
    if field is None:
        return np.eye(3)
    if mode is not Align.AFFINE:
        matrix, _ = homography(field)
        if matrix is not None:
            return matrix
    return np.vstack([field.affine.astype(np.float64), [0.0, 0.0, 1.0]])


def ink_angle(ink_mask: np.ndarray, limit_deg: float = 6.0, step_deg: float = 0.1) -> float:
    """Наклон строк краски к горизонтали по профилю: угол, при котором проекция краски на вертикаль самая резкая.

    Args:
        ink_mask: Маска краски вырезки.
        limit_deg: Перебор углов в пределах ±limit, градусы.
        step_deg: Шаг перебора, градусы.

    Returns:
        Угол, градусы (плюс — строка поднимается вправо вниз по кадру, как ``atan2(dy, dx)`` в пикселях).
    """
    ys, xs = np.nonzero(ink_mask)
    if len(xs) < 20:
        return 0.0
    xs = xs - xs.mean()
    best, best_score = 0.0, -1.0
    for angle in np.arange(-limit_deg, limit_deg + 1e-9, step_deg):
        # Сдвиг строки по наклону: точка (x, y) ложится в строку y − x·tg θ.
        rows = np.round(ys - xs * np.tan(np.radians(angle))).astype(np.int64)
        counts = np.bincount(rows - rows.min())
        score = float((counts.astype(np.float64) ** 2).sum())
        if score > best_score:
            best, best_score = float(angle), score
    return best


def lsd_angle(gray: np.ndarray, dpi: float = WORK_DPI) -> float | None:
    """Наклон рисунка к осям кадра по его прямым: длиновзвешенная медиана отклонения отрезков LSD от ближайшей оси.

    Профиль краски (:func:`ink_angle`) на перспективном рисунке ловит не горизонталь, а сходящиеся линии (1972/10
    с.79: +1.4° у ровного здания), и поворот рисунка шёл в выигрыш. Прямые рисунка — вертикали колонн, кромки
    блоков схемы — дают наклон надёжнее: и горизонтали, и вертикали отклоняются от своей оси на один и тот же угол.

    Args:
        gray: Серая вырезка рисунка.
        dpi: Разрешение вырезки.

    Returns:
        Угол, градусы (знак как у :func:`ink_angle`: ``atan2(dy, dx)`` в пикселях); ``None``, если околоосевых
        отрезков от ``TILT_SEGMENT_MM`` набирается меньше ``TILT_TOTAL_MM`` суммарно.
    """
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found = detector.detect(cv2.GaussianBlur(gray, (0, 0), 1.0))[0]
    if found is None:
        return None
    segments = found.reshape(-1, 4).astype(np.float64)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    angles = np.degrees(np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0]))
    # Отклонение от ближайшей оси: угол по модулю 90° в [−45°, 45°) — горизонтали и вертикали в одной шкале
    # (вертикаль, наклонённая как повёрнутая на θ горизонталь, даёт то же θ).
    deviation = (angles + 45.0) % 90.0 - 45.0
    keep = (lengths >= TILT_SEGMENT_MM * dpi / 25.4) & (np.abs(deviation) <= TILT_AXIS_DEG)
    if lengths[keep].sum() < TILT_TOTAL_MM * dpi / 25.4:
        return None
    # Взвешенная медиана: сортировка по углу, первая точка, где накопленный вес доходит до половины.
    order = np.argsort(deviation[keep])
    values, weights = deviation[keep][order], lengths[keep][order]
    return float(values[np.searchsorted(np.cumsum(weights), 0.5 * weights.sum())])


def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Подобие (поворот, масштаб, сдвиг) ``src → dst`` наименьшими квадратами; матрица 2×3."""
    # В комплексных числах подобие — w = a·z + b; a и b — линейная регрессия по центрированным точкам.
    z = src[:, 0] + 1j * src[:, 1]
    w = dst[:, 0] + 1j * dst[:, 1]
    zc, wc = z - z.mean(), w - w.mean()
    a = complex((np.conj(zc) @ wc) / max(float(np.real(np.conj(zc) @ zc)), 1e-9))
    b = w.mean() - a * z.mean()
    return np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag]])


def _robust_turn(src: np.ndarray, dst: np.ndarray) -> float:
    """Поворот подобия ``src → dst`` (градусы) с отсевом точек, чей остаток больше ``SHAPE_TRIM_PX``."""
    for _ in range(SHAPE_TRIM_ROUNDS):
        matrix = _similarity(src, dst)
        keep = np.hypot(*(src @ matrix[:, :2].T + matrix[:, 2] - dst).T) <= SHAPE_TRIM_PX
        if keep.all() or keep.sum() < SHAPE_PART_MIN:
            break
        src, dst = src[keep], dst[keep]
    matrix = _similarity(src, dst)
    return float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))


def _quad_design(points: np.ndarray, centre: np.ndarray, scale: float) -> np.ndarray:
    """Матрица квадратичной модели ``1, x, y, x², xy, y²`` по нормированным координатам (центр и масштаб охвата)."""
    x, y = ((points - centre) / scale).T
    return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])


def shape_measures(src: np.ndarray, dst: np.ndarray, dpi: float = WORK_DPI) -> dict[str, float]:
    """Форма рисунка по соответствиям точек краски B → A': насколько отображение рамки отличается от подобия.

    ЗАЧЕМ (1972/10 с.79, 2026-09-30). FineReader перекосил рисунок иначе, чем соседний текст: относительно страницы
    горизонтали рисунка повернулись на −1°, вертикали — нет (перекос осей 1.06°), рисунок вытянулся по вертикали на
    3.3 % против горизонтали, низ (галерея) перекошен, а верх (башня) нет — кривые линии стали кривее. AAD этого не
    видит: он снимает аффинную часть рамки целиком. Поворот и масштаб рисунка целиком (подобие) — не порча
    (решение 2026-09-22: «поворот рисунка целиком прощается, как деcкью»); всё, что сверх подобия, — порча формы.

    МЕРЫ. Три модели поля рамки по одним и тем же точкам — подобие, аффинная, квадратичная (двумерный многочлен
    второй степени: изгиб и трапеция). Уход одной модели от другой берётся по контуру охвата краски, максимум:

    * ``nonsim_mm`` — аффинная от подобия: перекос осей и изменение пропорций;
    * ``curve_mm`` — квадратичная от аффинной: изгиб и трапеция (дуга с прогибом s у краёв даёт около s/3: лучшая
      аффинная модель забирает её линейную часть);
    * ``shape_mm`` — квадратичная от подобия: всё сразу (групповая мера формы);
    * ``shape_shear_deg`` — угол между образами осей аффинной модели минус 90°, по модулю;
    * ``shape_aniso`` — |ln(|образ оси y| / |образ оси x|)| аффинной модели: изменение пропорций;
    * ``part_turn_deg``, ``part_turn_mm`` — наибольшая разность поворотов половин рисунка (левой и правой, верхней
      и нижней; подобие по каждой, от ``SHAPE_PART_MIN`` точек) и уход конца половины рамки от неё, мм;
    * ``inliers``, ``spread`` — точек после отсева и меньшее из их стандартных отклонений по осям в долях охвата
      (для решения, мерима ли форма).

    Args:
        src: Точки краски в B, ``(n, 2)``, пиксели.
        dst: Те же точки в A' (кадр B после выравнивания страницы), ``(n, 2)``.
        dpi: Разрешение пикселей (для мм).

    Returns:
        Словарь мер; нули, если точек меньше десяти.
    """
    out = {"shape_shear_deg": 0.0, "shape_aniso": 0.0, "nonsim_mm": 0.0, "curve_mm": 0.0, "shape_mm": 0.0,
           "part_turn_deg": 0.0, "part_turn_mm": 0.0, "inliers": 0.0, "spread": 0.0}  # fmt: skip
    if len(src) < 10:
        return out
    if len(src) > SHAPE_MAX_POINTS:
        pick = np.random.default_rng(0).choice(len(src), SHAPE_MAX_POINTS, replace=False)
        src, dst = src[pick], dst[pick]
    lo = np.percentile(src, SHAPE_HULL_PCT, axis=0)
    hi = np.percentile(src, 100.0 - SHAPE_HULL_PCT, axis=0)
    centre, scale = (lo + hi) / 2.0, max(float((hi - lo).max()) / 2.0, 1.0)
    all_src, all_dst = src, dst
    design = _quad_design(src, centre, scale)
    # Квадратичная модель с отсевом промахов: подгонка, выброс точек с остатком больше SHAPE_TRIM_PX, снова —
    # до SHAPE_TRIM_ROUNDS раз, пока точек не меньше десяти.
    coef, *_ = np.linalg.lstsq(design, dst, rcond=None)
    for _ in range(SHAPE_TRIM_ROUNDS):
        keep = np.hypot(*(design @ coef - dst).T) <= SHAPE_TRIM_PX
        if keep.all() or keep.sum() < 10:
            break
        src, dst, design = src[keep], dst[keep], design[keep]
        coef, *_ = np.linalg.lstsq(design, dst, rcond=None)
    # Сколько точек осталось и как они разбросаны: меньшее из стандартных отклонений по x и y в долях размера
    # охвата (точки вдоль одной полосы не задают ни перекос, ни изгиб).
    out["inliers"] = float(len(src))
    out["spread"] = float(src.std(axis=0).min() / max(float((hi - lo).max()), 1.0))
    affine, *_ = np.linalg.lstsq(np.column_stack([src, np.ones(len(src))]), dst, rcond=None)
    similar = _similarity(src, dst)
    # Контур охвата краски: SHAPE_BORDER_STEPS точек на каждой стороне прямоугольника [lo, hi].
    t = np.linspace(0.0, 1.0, SHAPE_BORDER_STEPS)
    border = np.vstack(
        [
            np.column_stack([lo[0] + t * (hi[0] - lo[0]), np.full_like(t, lo[1])]),
            np.column_stack([lo[0] + t * (hi[0] - lo[0]), np.full_like(t, hi[1])]),
            np.column_stack([np.full_like(t, lo[0]), lo[1] + t * (hi[1] - lo[1])]),
            np.column_stack([np.full_like(t, hi[0]), lo[1] + t * (hi[1] - lo[1])]),
        ]
    )
    at_quad = _quad_design(border, centre, scale) @ coef
    at_affine = np.column_stack([border, np.ones(len(border))]) @ affine
    at_similar = border @ similar[:, :2].T + similar[:, 2]
    worst = lambda one, other: px_to_mm(float(np.hypot(*(one - other).T).max()), dpi)  # noqa: E731
    out["nonsim_mm"] = worst(at_affine, at_similar)
    out["curve_mm"] = worst(at_quad, at_affine)
    out["shape_mm"] = worst(at_quad, at_similar)
    # Поворот частей рисунка друг относительно друга: подобие отдельно по левой и правой, верхней и нижней половинам
    # точек; мера — наибольшая разность поворотов (1972/10 с.79: правое крыло галереи довёрнуто на 1.4–2.9° против
    # левого). В мм — уход конца половины рамки: tg(разности) × половина охвата поперёк деления.
    # Считается по всем точкам до отсева: довёрнутая половина для квадратичной модели — «промахи», и отсев её выкинул
    # бы; промахи отсеиваются внутри каждой половины по её подобию.
    for axis in (0, 1):
        middle = float(np.median(all_src[:, axis]))
        halves = [all_src[:, axis] < middle, all_src[:, axis] >= middle]
        if min(int(half.sum()) for half in halves) < SHAPE_PART_MIN:
            continue
        turns = [_robust_turn(all_src[h], all_dst[h]) for h in halves]
        turn = abs(turns[0] - turns[1])
        span = float(hi[axis] - lo[axis]) / 2.0
        if turn > out["part_turn_deg"]:
            out["part_turn_deg"] = turn
            out["part_turn_mm"] = px_to_mm(abs(np.tan(np.radians(turn))) * span, dpi)
    # Линейная часть аффинной модели: образы осей x и y (столбцы affine[:2].T).
    linear = affine[:2].T
    ex, ey = linear[:, 0], linear[:, 1]
    cos = float(ex @ ey / max(np.linalg.norm(ex) * np.linalg.norm(ey), 1e-9))
    out["shape_shear_deg"] = abs(float(np.degrees(np.arcsin(np.clip(cos, -1.0, 1.0)))))
    out["shape_aniso"] = abs(float(np.log(max(np.linalg.norm(ey), 1e-9) / max(np.linalg.norm(ex), 1e-9))))
    return out


def text_reference(field: Field | None, page_map: np.ndarray, box) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Смещения текста вокруг рамки в кадре A': как FineReader сдвинул соседний текст (опора формы рисунка).

    FineReader выпрямляет изогнутую страницу целиком, и рисунок законно деформируется вместе с текстом вокруг
    (1974/02 с.95: бланк на всю полосу). Порча рисунка — то, что сверх деформации соседнего текста. Тайл поля v16 —
    центр в B ``c`` и его место в A ``c + u``; в кадре A' (A, переведённая в кадр B матрицей ``H``) это
    ``H⁻¹(c + u)``, смещение — ``H⁻¹(c + u) − c``.

    Args:
        field: Поле v16 (тайлы текста с весом больше нуля).
        page_map: Матрица страницы B → A (:func:`page_matrix`).
        box: Рамка в B, пиксели 150 dpi.

    Returns:
        ``(центры тайлов в B (n, 2), смещения в кадре A' (n, 2), опора только по кольцу вокруг рамки)``;
        ``None`` — тайлов с парой рядом нет.
    """
    if field is None or not len(field.tiles):
        return None
    good = field.weight > 0
    centres, moves = field.tiles[good, :2].astype(np.float64), field.tiles[good, 2:4].astype(np.float64)
    if not len(centres):
        return None
    ring = REF_RING_MM * WORK_DPI / 25.4
    x0, y0, x1, y1 = box
    near = (centres[:, 0] >= x0 - ring) & (centres[:, 0] <= x1 + ring) & (centres[:, 1] >= y0 - ring)
    near &= centres[:, 1] <= y1 + ring
    inside = (centres[:, 0] >= x0) & (centres[:, 0] <= x1) & (centres[:, 1] >= y0) & (centres[:, 1] <= y1)
    # Кольцо без самой рамки: тайлы внутри рисунка двигаются вместе с ним и прячут его порчу; внутренние — только
    # если в кольце тайлов мало (рамка на всю полосу).
    pick = near & ~inside
    ring_only = bool(pick.sum() >= REF_MIN_TILES)
    if not ring_only:
        pick = near
    if not pick.any():
        return None
    inverse = np.linalg.inv(page_map)
    landed = cv2.perspectiveTransform((centres[pick] + moves[pick]).reshape(-1, 1, 2), inverse).reshape(-1, 2)
    return centres[pick], landed - centres[pick], ring_only


def _reference_model(points: np.ndarray, moves: np.ndarray, centre: np.ndarray, scale: float):
    """Модель смещений опоры: коэффициенты и функция «точки → смещения» (квадратичная, аффинная или сдвиг)."""
    if len(points) >= REF_QUAD_TILES:
        coef, *_ = np.linalg.lstsq(_quad_design(points, centre, scale), moves, rcond=None)
        return lambda q: _quad_design(q, centre, scale) @ coef
    if len(points) >= REF_MIN_TILES:
        coef, *_ = np.linalg.lstsq(np.column_stack([points, np.ones(len(points))]), moves, rcond=None)
        return lambda q: np.column_stack([q, np.ones(len(q))]) @ coef
    shift = np.median(moves, axis=0)
    return lambda q: np.tile(shift, (len(q), 1))


def straightness(crop_b: np.ndarray, displacement: np.ndarray, dpi: float = WORK_DPI) -> tuple[float, int]:
    """Изменение прямоты длинных черт рисунка B → A': направление деформации (выпрямил или погнул).

    Черты — длинные горизонтали и вертикали краски B (открытие ядром ``STRAIGHT_OPEN_MM``), от ``STRAIGHT_MIN_MM``.
    Средняя линия черты в B — средняя координата краски черты по каждому столбцу (строке у вертикали); в A' — та
    же линия, сдвинутая полем. Мера черты — размах отклонения средней линии от подогнанной прямой; в A' против B.
    Гомография страницы переводит прямую в прямую, поэтому прямота в A' — та же, что в A.

    Args:
        crop_b: Вырезка B (серая).
        displacement: Полное поле B → A' в пикселях вырезки ``(H, W, 2)``.
        dpi: Разрешение.

    Returns:
        ``(длиновзвешенная медиана изменения размаха A' − B, мм; число черт)``; ``(0.0, 0)`` без черт.
    """
    ink = (crop_b < INK_LEVEL).astype(np.uint8)
    length = max(3, int(round(STRAIGHT_OPEN_MM * dpi / 25.4)))
    minimum = STRAIGHT_MIN_MM * dpi / 25.4
    deltas, weights = [], []
    for kernel, along in ((np.ones((1, length), np.uint8), 0), (np.ones((length, 1), np.uint8), 1)):
        mask = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, count):
            size = stats[label, cv2.CC_STAT_WIDTH if along == 0 else cv2.CC_STAT_HEIGHT]
            if size < minimum:
                continue
            ys, xs = np.nonzero(labels == label)
            # Средняя линия: по каждой координате вдоль черты — средняя поперечная координата краски.
            main, cross = (xs, ys) if along == 0 else (ys, xs)
            keys, inverse = np.unique(main, return_inverse=True)
            middle = np.bincount(inverse, weights=cross) / np.bincount(inverse)
            line_b = np.column_stack([keys, middle]) if along == 0 else np.column_stack([middle, keys])
            line_a = line_b + _sample(displacement, line_b)
            spreads = []
            for line in (line_b, line_a):
                u, v = (line[:, 0], line[:, 1]) if along == 0 else (line[:, 1], line[:, 0])
                fit = np.polyval(np.polyfit(u, v, 1), u)
                spreads.append(float(np.ptp(v - fit)))
            deltas.append(px_to_mm(spreads[1] - spreads[0], dpi))
            weights.append(float(size))
    if not deltas:
        return 0.0, 0
    order = np.argsort(deltas)
    values, weights_sorted = np.array(deltas)[order], np.array(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights_sorted), 0.5 * weights_sorted.sum())]), len(deltas)


def _subpixel(values: np.ndarray, index: int) -> float:
    """Поправка к целому положению пика по параболе через три соседние точки (в пределах ±0.5)."""
    if index <= 0 or index >= len(values) - 1:
        return 0.0
    left, mid, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2.0 * mid + right
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5)) if abs(denominator) > 1e-9 else 0.0


def patch_matches(
    ink_b: np.ndarray, ink_a: np.ndarray, mask_b: np.ndarray, dpi: float = WORK_DPI
) -> tuple[np.ndarray, np.ndarray]:
    """Соответствия B → A' по участкам рисунка: корреляция размытой краски с отсевом слабых и неоднозначных пиков.

    Попиксельный поток DIS на заливках и штриховке «плывёт» (портрет 1970/04 с.32: перекос осей 10° на ровном месте),
    и модели формы, подогнанные по нему, дают миллиметры порчи на страницах эталона good. Участок же сопоставляется
    целиком и отбрасывается, если пик слаб или не единственен.

    Args:
        ink_b: Размытая краска вырезки B (:func:`_ink`).
        ink_a: Размытая краска вырезки A' того же размера (уже сдвинутой на опору).
        mask_b: Маска краски B.
        dpi: Разрешение.

    Returns:
        ``(центры участков в B (n, 2), смещения в A' (n, 2))``, пиксели вырезки.
    """
    size = int(round(PATCH_MM * dpi / 25.4))
    step = int(round(PATCH_STEP_MM * dpi / 25.4))
    search = int(round(PATCH_SEARCH_MM * dpi / 25.4))
    second = PATCH_SECOND_MM * dpi / 25.4
    h, w = ink_b.shape
    centres, moves = [], []
    template_b = ink_b.astype(np.float32)
    image_a = cv2.copyMakeBorder(ink_a.astype(np.float32), search, search, search, search, cv2.BORDER_CONSTANT, value=0)
    for top in range(0, max(1, h - size + 1), step):
        for left in range(0, max(1, w - size + 1), step):
            share = float(mask_b[top : top + size, left : left + size].mean())
            if not PATCH_INK_MIN <= share <= PATCH_INK_MAX:
                continue
            patch = template_b[top : top + size, left : left + size]
            if patch.shape != (size, size) or float(patch.std()) < 1e-3:
                continue
            window = image_a[top : top + size + 2 * search, left : left + size + 2 * search]
            score = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
            iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
            peak = float(score[iy, ix])
            if peak < PATCH_MIN_PEAK:
                continue
            # Второй пик — вне круга PATCH_SECOND_MM вокруг первого.
            yy, xx = np.mgrid[0 : score.shape[0], 0 : score.shape[1]]
            far = np.hypot(yy - iy, xx - ix) > second
            if far.any() and peak - float(score[far].max()) < PATCH_PEAK_GAP:
                continue
            # Пик на краю окна — истинное место дальше окна поиска (или его нет): участок не берётся.
            if min(ix, iy) == 0 or ix == score.shape[1] - 1 or iy == score.shape[0] - 1:
                continue
            fx = ix + _subpixel(score[iy, :], ix) - search
            fy = iy + _subpixel(score[:, ix], iy) - search
            centres.append((left + size / 2.0, top + size / 2.0))
            moves.append((fx, fy))
    return np.array(centres, dtype=np.float64).reshape(-1, 2), np.array(moves, dtype=np.float64).reshape(-1, 2)


def _frame_shift(crop_b: np.ndarray, aligned_a: np.ndarray, ax0: int, ay0: int, dx: int, dy: int) -> tuple[int, int]:
    """Сдвиг рамки целиком: максимум корреляции размытой краски B в окне A' ±``FRAME_SEARCH_MM`` вокруг ``(ax0, ay0)``.

    Returns:
        Уточнённый ``(dx, dy)``; если окно выходит за страницу целиком или пик не найден — прежний.
    """
    search = int(round(FRAME_SEARCH_MM * WORK_DPI / 25.4))
    h, w = crop_b.shape
    window = np.full((h + 2 * search, w + 2 * search), 255, dtype=aligned_a.dtype)
    wx0, wy0 = ax0 - search, ay0 - search
    sx0, sy0 = max(0, wx0), max(0, wy0)
    sx1, sy1 = min(aligned_a.shape[1], wx0 + window.shape[1]), min(aligned_a.shape[0], wy0 + window.shape[0])
    if sx1 <= sx0 or sy1 <= sy0:
        return dx, dy
    window[sy0 - wy0 : sy1 - wy0, sx0 - wx0 : sx1 - wx0] = aligned_a[sy0:sy1, sx0:sx1]
    score = cv2.matchTemplate(_ink(window), _ink(crop_b), cv2.TM_CCOEFF_NORMED)
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    return dx + int(ix) - search, dy + int(iy) - search


def frame_shape(
    gray_b: np.ndarray, aligned_a: np.ndarray, box, field: Field | None, page_map: np.ndarray
) -> dict[str, float] | None:
    """Форма рисунка сверх деформации соседнего текста, с проверкой направления по прямоте его черт.

    1. Опора — смещения тайлов текста вокруг рамки (:func:`text_reference`), модель по ним.
    2. Вырезка A' сдвигается на смещение опоры в центре рамки (у поля страницы на краю полосы уход бывает
       10–20 px, 1970/04 с.32), дальше поиск идёт в малом окне.
    3. Соответствия — участки рисунка (:func:`patch_matches`), место в A' за вычетом смещения опоры →
       :func:`shape_measures` (форма сверх опоры); участков меньше ``PATCH_MIN_COUNT`` — форма не меряется.
    4. :func:`straightness` по полному полю DIS: если черты рисунка стали прямее (медиана ниже −``STRAIGHT_GAIN_MM``),
       FineReader выправил рисунок, и ``shape_mm`` обнуляется (сырая мера — ``shape_raw_mm``).

    Args:
        gray_b: Рендер B, 150 dpi.
        aligned_a: Рендер A в кадре B.
        box: Рамка в B.
        field: Поле v16.
        page_map: Матрица страницы B → A.

    Returns:
        Меры ``SHAPE_NAMES`` (без приставки); ``None`` — рамка мала, краски нет или в A на этом месте другой
        рисунок (``ncc < MIN_NCC``).
    """
    pad = int(round(PAD_MM * WORK_DPI / 25.4))
    h, w = gray_b.shape
    x0, y0 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
    x1, y1 = min(w, int(box[2]) + pad), min(h, int(box[3]) + pad)
    if px_to_mm(min(x1 - x0, y1 - y0), WORK_DPI) < MIN_BOX_MM:
        return None
    crop_b = gray_b[y0:y1, x0:x1]
    ink_mask = crop_b < INK_LEVEL
    if ink_mask.sum() < 50:
        return None
    origin = np.array([x0, y0], dtype=np.float64)
    reference = text_reference(field, page_map, box)
    # Большая рамка — только с опорой по кольцу вокруг неё.
    large = (x1 - x0) * (y1 - y0) > SHAPE_MAX_PAGE_SHARE * h * w
    if large and (reference is None or not reference[2]):
        return None
    centre = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    scale = max(x1 - x0, y1 - y0) / 2.0
    model = _reference_model(reference[0], reference[1], centre, scale) if reference is not None else None
    # Сдвиг вырезки A': смещение опоры в центре рамки, уточнённое поиском рамки целиком по корреляции краски.
    dx, dy = (int(round(v)) for v in (model(centre[None])[0] if model is not None else (0.0, 0.0)))
    dx, dy = _frame_shift(crop_b, aligned_a, x0 + dx, y0 + dy, dx, dy)
    crop_a = np.full_like(crop_b, 255)
    ax0, ay0 = x0 + dx, y0 + dy
    sx0, sy0 = max(0, ax0), max(0, ay0)
    sx1, sy1 = min(aligned_a.shape[1], ax0 + crop_b.shape[1]), min(aligned_a.shape[0], ay0 + crop_b.shape[0])
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    crop_a[sy0 - ay0 : sy1 - ay0, sx0 - ax0 : sx1 - ax0] = aligned_a[sy0:sy1, sx0:sx1]
    ink_b, ink_a = _ink(crop_b), _ink(crop_a)
    if float(cv2.matchTemplate(ink_a, ink_b, cv2.TM_CCOEFF_NORMED)[0, 0]) < MIN_NCC:
        return None
    to_u8 = lambda image: np.clip(image, 0, 255).astype(np.uint8)  # noqa: E731
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(to_u8(ink_b), to_u8(ink_a), None).astype(np.float64)
    displacement = flow + np.array([dx, dy], dtype=np.float64)  # полное поле вырезки B → A'
    points, moves = patch_matches(ink_b, ink_a, ink_mask)
    if len(points) < PATCH_MIN_COUNT:
        return None
    moved = moves + np.array([dx, dy], dtype=np.float64)
    if model is not None:
        moved = moved - model(points + origin)
    out = shape_measures(points, points + moved)
    out["patches"] = float(len(points))
    # Форма мерима, если после отсева промахов осталось не меньше PATCH_MIN_COUNT участков и они не лежат в одну полосу.
    if out["inliers"] < PATCH_MIN_COUNT or out["spread"] < SHAPE_MIN_SPREAD:
        return None
    delta, lines = straightness(crop_b, displacement)
    out["shape_raw_mm"] = out["shape_mm"]
    out["straight_delta_mm"] = delta
    out["straight_lines"] = float(lines)
    if lines and delta < -STRAIGHT_GAIN_MM:
        out["shape_mm"] = 0.0
    return out


def transfer_axis(
    gray_b: np.ndarray, aligned_a: np.ndarray, page_map: np.ndarray, points: np.ndarray, height: float
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Ось строки B, перенесённая в A плотным полем: для строки, чьей оси в разборе A нет.

    Разбор A теряет строки (1972/10 с.79: подпись «VII.INSYMA 1971» под рисунком есть в B и не найдена в A), и такие
    строки не мерились вовсе, хотя FineReader наклонил подпись на 2.5°. Здесь ось A строится без разбора A: полоса
    вокруг оси B, DIS-поток B → A' по размытой краске, поле усредняется по краске строки гауссом и берётся в точках
    оси; точка оси B ``p`` в кадре B встаёт в ``p + поле(p)``, а в кадре A — в ``H·(p + поле(p))`` (``H`` — матрица
    страницы, которой :func:`align_page` переводит A в кадр B).

    Args:
        gray_b: Рендер B, 150 dpi.
        aligned_a: Рендер A в кадре B (:func:`align_page`).
        page_map: Матрица страницы B → A (:func:`page_matrix`).
        points: Точки оси B ``(n, 2)``, пиксели 150 dpi.
        height: Высота строки B, пиксели.

    Returns:
        ``(точки оси в A (n, 2), корреляция полосы B ↔ A', уверенность точек (n,) в долях медианы по оси)``;
        ``None``, если полоса пустая или вне страницы.
    """
    h, w = gray_b.shape
    pad = TRANSFER_PAD_MM * WORK_DPI / 25.4
    band = max(TRANSFER_BAND * height, TRANSFER_MIN_BAND_MM * WORK_DPI / 25.4)
    x0, x1 = int(max(0, points[:, 0].min() - pad)), int(min(w, points[:, 0].max() + pad))
    y0, y1 = int(max(0, points[:, 1].min() - band)), int(min(h, points[:, 1].max() + band))
    if x1 - x0 < DIS_MIN_PX or y1 - y0 < DIS_MIN_PX:
        return None
    crop_b, crop_a = gray_b[y0:y1, x0:x1], aligned_a[y0:y1, x0:x1]
    ink_b, ink_a = _ink(crop_b), _ink(crop_a)
    mask = (crop_b < INK_LEVEL).astype(np.float32)
    if mask.sum() < 20:
        return None
    ncc = float(cv2.matchTemplate(ink_a, ink_b, cv2.TM_CCOEFF_NORMED)[0, 0])
    to_u8 = lambda image: np.clip(image, 0, 255).astype(np.uint8)  # noqa: E731
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(to_u8(ink_b), to_u8(ink_a), None).astype(np.float32)
    # Поле по краске строки: гаусс по произведению поля на вес, делённый на гаусс веса (фон, где поток DIS не
    # определён, в среднее не входит). Вес составляющей — сила кромки поперёк неё: вертикальный сдвиг надёжен на
    # горизонтальных кромках букв (|∂I/∂y|), горизонтальный — на вертикальных (|∂I/∂x|). У вертикальной линейки,
    # до которой дотянулась ось (1971/06 с.3, «ИЮНЬ»), вертикальный сдвиг не определён, и без этого веса размах
    # перенесённой оси вырастал до 3 мм на неизменной строке.
    sigma = max(2.0, TRANSFER_SIGMA * height)
    edge_y = np.abs(cv2.Sobel(ink_b.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))
    edge_x = np.abs(cv2.Sobel(ink_b.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    smooth = np.zeros_like(flow)
    support = np.zeros(flow.shape[:2], dtype=np.float32)
    for k, edge in ((0, edge_x), (1, edge_y)):
        weight = mask * edge / max(float(edge.max()), 1e-6)
        blurred = cv2.GaussianBlur(weight, (0, 0), sigma)
        smooth[..., k] = cv2.GaussianBlur(flow[..., k] * weight, (0, 0), sigma) / (blurred + 1e-6)
        if k == 1:
            support = blurred
    local = points - np.array([x0, y0], dtype=np.float64)
    moved = points + _sample(smooth, local).astype(np.float64)
    in_a = cv2.perspectiveTransform(moved.reshape(-1, 1, 2), page_map).reshape(-1, 2)
    # Уверенность точки — опора вертикального сдвига (размытый вес горизонтальных кромок) в долях медианы по оси:
    # у конца оси, упёршегося в вертикальную линейку или в пустоту, она почти нулевая, и сдвиг там — шум деления.
    confidence = _sample(np.dstack([support, support]), local)[:, 0].astype(np.float64)
    confidence = confidence / max(float(np.median(confidence)), 1e-9)
    return in_a, ncc, confidence


def page_alignment(
    gray_b: np.ndarray, gray_a: np.ndarray, field: Field | None, mode: Align = DEFAULT_ALIGN
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Выравнивание страницы один раз на все меры: ``(A в кадре B, остатки тайлов, матрица страницы B → A)``."""
    aligned, resid = align_page(gray_b, gray_a, field, mode)
    return aligned, resid, page_matrix(field, mode)


def merge_boxes(boxes: list) -> list[tuple[float, float, float, float]]:
    """Слить рамки, перекрытые больше ``MERGE_OVERLAP`` меньшей из двух (объединением), пока сливается."""
    out = [tuple(float(v) for v in box) for box in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                iw = min(a[2], b[2]) - max(a[0], b[0])
                ih = min(a[3], b[3]) - max(a[1], b[1])
                if iw <= 0 or ih <= 0:
                    continue
                smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                if iw * ih >= MERGE_OVERLAP * smaller:
                    out[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del out[j]
                    merged = True
                    break
            if merged:
                break
    return out


def measure_box(
    gray_b: np.ndarray,
    aligned_a: np.ndarray,
    box,
    field: Field | None = None,
    resid: np.ndarray | None = None,
    local: bool = False,
    keep_maps: bool = False,
    min_side_mm: float = MIN_BOX_MM,
    min_long_mm: float = MIN_BOX_MM,
    page_map: np.ndarray | None = None,
) -> FlowMeasure | None:
    """Меры порчи одной рамки line art.

    Args:
        gray_b: Рендер B 150 dpi.
        aligned_a: Рендер A 150 dpi, уже в кадре B (:func:`align_page`).
        box: Рамка в B ``(x0, y0, x1, y1)``, пиксели 150 dpi.
        field: Поле v16 (центры тайлов — для местного сдвига).
        resid: Остатки тайлов после выравнивания (:func:`align_page`).
        local: Сдвигать ли вырезку A на медианный остаток тайлов в рамке.
        keep_maps: Вернуть вырезки и карту отклонения AAD — для оверлея.
        min_side_mm: Рамка с короткой стороной меньше — не меряется.
        min_long_mm: Рамка с длинной стороной меньше — не меряется.
        page_map: Матрица страницы B → A (:func:`page_matrix`) — для абсолютного наклона рамки в A; ``None`` —
            абсолютный наклон не считается.

    Returns:
        :class:`FlowMeasure` или ``None``, если рамка мала или краски в ней нет.
    """
    pad = int(round(PAD_MM * WORK_DPI / 25.4))
    h, w = gray_b.shape
    x0, y0 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
    x1, y1 = min(w, int(box[2]) + pad), min(h, int(box[3]) + pad)
    if (
        px_to_mm(min(x1 - x0, y1 - y0), WORK_DPI) < min_side_mm
        or px_to_mm(max(x1 - x0, y1 - y0), WORK_DPI) < min_long_mm
    ):
        return None
    crop_b = gray_b[y0:y1, x0:x1]
    # Местный сдвиг: медианный остаток тайлов, чьи центры в рамке (FineReader двинул этот рисунок сильнее страницы).
    dx = dy = 0
    if local and field is not None and resid is not None and len(resid) == len(field.tiles):
        centres = field.tiles[:, :2]
        inside = (centres[:, 0] >= x0) & (centres[:, 0] <= x1) & (centres[:, 1] >= y0) & (centres[:, 1] <= y1)
        inside &= field.weight > 0
        if inside.sum() >= LOCAL_MIN_TILES:
            dx, dy = (int(round(v)) for v in np.median(resid[inside], axis=0))
    shifted = np.full_like(crop_b, 255)
    ax0, ay0, ax1, ay1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
    sx0, sy0 = max(0, ax0), max(0, ay0)
    sx1, sy1 = min(aligned_a.shape[1], ax1), min(aligned_a.shape[0], ay1)
    if sx1 > sx0 and sy1 > sy0:
        shifted[sy0 - ay0 : sy1 - ay0, sx0 - ax0 : sx1 - ax0] = aligned_a[sy0:sy1, sx0:sx1]
    crop_a = shifted
    ink_b, ink_a = _ink(crop_b), _ink(crop_a)
    ink_mask = crop_b < INK_LEVEL
    if ink_mask.sum() < 50:
        return None
    # Тот же ли рисунок: корреляция размытой краски вырезок одного размера.
    ncc = float(cv2.matchTemplate(ink_a, ink_b, cv2.TM_CCOEFF_NORMED)[0, 0])
    out_box = tuple(float(v) for v in box)
    if ncc < MIN_NCC:
        maps = {"crop_b": crop_b, "crop_a": crop_a, "shift": (dx, dy)} if keep_maps else {}
        return FlowMeasure(out_box, ncc, False, **maps)
    to_u8 = lambda image: np.clip(image, 0, 255).astype(np.uint8)  # noqa: E731
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(to_u8(ink_b), to_u8(ink_a), None)  # B → A': точка B p оказалась в A' в p + flow(p)
    ys, xs = np.nonzero(ink_mask)
    points = np.column_stack([xs, ys]).astype(np.float64)
    # Аффинная часть поля рамки по пикселям краски — снимается со всего поля.
    grid_y, grid_x = np.mgrid[0 : crop_b.shape[0], 0 : crop_b.shape[1]]
    all_points = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(
        np.column_stack([points, np.ones(len(points))]), flow[ys, xs].astype(np.float64), rcond=None
    )
    affine = (np.column_stack([all_points, np.ones(len(all_points))]) @ coef).reshape(flow.shape)
    rest = flow.astype(np.float64) - affine
    # Поворот горизонталей рамки: образ оси x под аффинной частью (du_x/dx = coef[0,0], du_y/dx = coef[0,1]).
    tilt = float(np.degrees(np.arctan2(coef[0, 1], 1.0 + coef[0, 0])))
    tilt_mm = px_to_mm(abs(np.tan(np.radians(tilt))) * crop_b.shape[1], WORK_DPI)
    # Абсолютный наклон: в B — по прямым рисунка (LSD), без них — по профилю краски; в A — плюс поворот горизонталей
    # страницы в центре рамки и собственный поворот рамки.
    by_lines = lsd_angle(crop_b)
    abs_b = by_lines if by_lines is not None else ink_angle(ink_mask)
    abs_a = abs_b
    if page_map is not None:
        centre = np.array([[[(x0 + x1) / 2.0, (y0 + y1) / 2.0]], [[(x0 + x1) / 2.0 + 10.0, (y0 + y1) / 2.0]]])
        mapped = cv2.perspectiveTransform(centre, page_map).reshape(2, 2)
        page_turn = float(np.degrees(np.arctan2(mapped[1, 1] - mapped[0, 1], mapped[1, 0] - mapped[0, 0])))
        abs_a = abs_b + page_turn + tilt
    width_px = float(crop_b.shape[1])
    skew_mm = px_to_mm((abs(np.tan(np.radians(abs_a))) - abs(np.tan(np.radians(abs_b)))) * width_px, WORK_DPI)
    # AAD: веса — нормированная сила градиента B по осям.
    gy = np.abs(cv2.Sobel(ink_b, cv2.CV_32F, 0, 1, ksize=3))
    gx = np.abs(cv2.Sobel(ink_b, cv2.CV_32F, 1, 0, ksize=3))
    gy, gx = gy / max(float(gy.max()), 1e-6), gx / max(float(gx.max()), 1e-6)
    gy[gy < GRAD_FRAC] = 0.0
    gx[gx < GRAD_FRAC] = 0.0
    vy, vx = rest[..., 1], rest[..., 0]
    row_mean = (vy * gy).sum(axis=1, keepdims=True) / (gy.sum(axis=1, keepdims=True) + 1e-6)
    col_mean = (vx * gx).sum(axis=0, keepdims=True) / (gx.sum(axis=0, keepdims=True) + 1e-6)
    d = np.hypot(gy * np.abs(vy - row_mean), gx * np.abs(vx - col_mean))
    weight = np.hypot(gy, gx)
    feature = weight > 0
    if not feature.any():
        return None
    aad_mean = float(d.sum() / weight.sum())
    variants, segments = line_variants(crop_b, rest)
    # p95 по пикселям черт: отклонение без веса (сам сдвиг черты, px).
    raw = np.hypot(np.where(gy > 0, np.abs(vy - row_mean), 0.0), np.where(gx > 0, np.abs(vx - col_mean), 0.0))
    aad_p95 = float(np.percentile(raw[feature], 95))
    # Якобиан сглаженного остатка на пикселях краски.
    sigma = JACOBIAN_SIGMA_MM * WORK_DPI / 25.4
    ux = cv2.GaussianBlur(rest[..., 0].astype(np.float32), (0, 0), sigma)
    uy = cv2.GaussianBlur(rest[..., 1].astype(np.float32), (0, 0), sigma)
    dux_dy, dux_dx = np.gradient(ux)
    duy_dy, duy_dx = np.gradient(uy)
    jac = np.stack(
        [np.stack([1 + dux_dx[ys, xs], dux_dy[ys, xs]], -1), np.stack([duy_dx[ys, xs], 1 + duy_dy[ys, xs]], -1)], -2
    )
    singular = np.linalg.svd(jac, compute_uv=False)
    aniso = singular[:, 0] / np.maximum(singular[:, 1], 1e-6) - 1.0
    # Перекос: угол между образами осей x и y минус 90°.
    ex, ey = jac[:, :, 0], jac[:, :, 1]
    cosang = np.abs((ex * ey).sum(-1)) / (np.linalg.norm(ex, axis=-1) * np.linalg.norm(ey, axis=-1) + 1e-9)
    shear = np.degrees(np.arcsin(np.clip(cosang, 0.0, 1.0)))
    return FlowMeasure(
        box=out_box,
        ncc=ncc,
        found=True,
        aad_mean_mm=px_to_mm(aad_mean, WORK_DPI),
        aad_p95_mm=px_to_mm(aad_p95, WORK_DPI),
        **variants,
        tilt_deg=tilt,
        tilt_mm=tilt_mm,
        abs_tilt_b_deg=abs_b,
        abs_tilt_a_deg=abs_a,
        skew_mm=skew_mm,
        aniso_p95=float(np.percentile(aniso, 95)),
        shear_p95_deg=float(np.percentile(shear, 95)),
        **(
            {"crop_b": crop_b, "crop_a": crop_a, "heat": raw, "shift": (dx, dy), "segments": segments}
            if keep_maps
            else {}
        ),
    )


def _sample(field_xy: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Значения поля ``(H, W, 2)`` в точках ``(n, 2)`` (x, y) — билинейно."""
    maps = points.astype(np.float32).reshape(-1, 1, 2)
    return np.stack(
        [
            cv2.remap(field_xy[..., k].astype(np.float32), maps[..., 0], maps[..., 1], cv2.INTER_LINEAR).ravel()
            for k in (0, 1)
        ],
        axis=-1,
    )


def line_variants(crop_b: np.ndarray, rest: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Меры «только по линиям» по остатку поля рамки (после снятия её аффинной части).

    * ``aad_lines_mm`` — AAD, где весами служат только пиксели длинных горизонтальных (для строк) и вертикальных
      (для столбцов) черт: открытие бинарной краски ядром ``LINE_OPEN_MM`` отсекает буквы.
    * ``seg_bend_mm`` — отрезки LSD любой ориентации от ``SEGMENT_MIN_MM``: вдоль отрезка смещение поперёк него
      (проекция остатка на нормаль) должно быть одним; мера отрезка — p90 |отклонения от медианы|, по странице —
      среднее, взвешенное длиной.
    * ``seg_rel_mm`` — соседние параллельные отрезки (не кромки одного штриха): разность их медианных поперечных
      смещений, p95 по парам — сдвиг одной части схемы относительно соседней (перекошенный столбец блоков).

    Args:
        crop_b: Вырезка B (серая).
        rest: Остаток поля B → A' ``(H, W, 2)``.

    Returns:
        ``(меры, отрезки)``: меры в мм (нули, если линий нет); отрезки ``(n, 5)`` — ``x0, y0, x1, y1`` в пикселях
        вырезки и поперечный сдвиг отрезка относительно соседних параллельных (p95 по соседям, пиксели; NaN — у
        отрезка нет соседей) — для оверлея.
    """
    out = {"aad_lines_mm": 0.0, "seg_bend_mm": 0.0, "seg_rel_mm": 0.0, "bar_ptp_mm": 0.0}
    empty = np.zeros((0, 5))
    ink = (crop_b < INK_LEVEL).astype(np.uint8)
    length = max(3, int(round(LINE_OPEN_MM * WORK_DPI / 25.4)))
    mask_h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8)).astype(bool)
    mask_v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8)).astype(bool)
    vy, vx = rest[..., 1], rest[..., 0]
    if mask_h.any() or mask_v.any():
        wy, wx = mask_h.astype(np.float64), mask_v.astype(np.float64)
        row_mean = (vy * wy).sum(axis=1, keepdims=True) / (wy.sum(axis=1, keepdims=True) + 1e-6)
        col_mean = (vx * wx).sum(axis=0, keepdims=True) / (wx.sum(axis=0, keepdims=True) + 1e-6)
        d = np.hypot(wy * np.abs(vy - row_mean), wx * np.abs(vx - col_mean))
        out["aad_lines_mm"] = px_to_mm(float(d.sum() / max(1.0, np.hypot(wy, wx).sum())), WORK_DPI)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found = detector.detect(cv2.GaussianBlur(crop_b, (0, 0), 1.0))[0]
    if found is None:
        return out, empty
    segments = found.reshape(-1, 4).astype(np.float64)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    keep = lengths >= SEGMENT_MIN_MM * WORK_DPI / 25.4
    segments, lengths = segments[keep], lengths[keep]
    if not len(segments):
        return out, empty
    bends, medians, normals, middles, angles = [], [], [], [], []
    for (x0, y0, x1, y1), size in zip(segments, lengths):
        steps = max(2, int(size))
        t = np.linspace(0.0, 1.0, steps)
        points = np.column_stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)])
        direction = np.array([x1 - x0, y1 - y0]) / size
        normal = np.array([-direction[1], direction[0]])
        across = _sample(rest, points) @ normal
        median = float(np.median(across))
        # Черта: длинный почти горизонтальный отрезок — размах поперечного смещения вдоль неё.
        if (
            size >= BAR_MIN_MM * WORK_DPI / 25.4
            and abs(np.degrees(np.arctan2(direction[1], abs(direction[0])))) <= BAR_MAX_DEG
        ):
            out["bar_ptp_mm"] = max(out["bar_ptp_mm"], px_to_mm(float(np.ptp(across)), WORK_DPI))
        bends.append(float(np.percentile(np.abs(across - median), 90)))
        medians.append(median)
        normals.append(normal)
        middles.append(points[len(points) // 2])
        angles.append(float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0))
    out["seg_bend_mm"] = px_to_mm(float(np.average(bends, weights=lengths)), WORK_DPI)
    middles, angles, medians = np.array(middles), np.array(angles), np.array(medians)
    near = NEIGHBOUR_MM * WORK_DPI / 25.4
    same = SAME_STROKE_MM * WORK_DPI / 25.4
    diffs = []
    own = np.full(len(segments), np.nan)
    for i in range(len(segments)):
        delta = np.abs(angles - angles[i])
        parallel = np.minimum(delta, 180.0 - delta) <= PARALLEL_TOL_DEG
        gap = middles - middles[i]
        distance = np.hypot(gap[:, 0], gap[:, 1])
        across = np.abs(gap @ normals[i])
        # Знак нормали у LSD зависит от направления отрезка: поперечные смещения сравниваются в нормали i.
        signs = np.sign(np.array([n @ normals[i] for n in normals]))
        pick = parallel & (distance <= near) & (across >= same)
        pick[i] = False
        if pick.any():
            shifts = np.abs(medians[pick] * signs[pick] - medians[i])
            diffs.extend(shifts.tolist())
            own[i] = float(np.percentile(shifts, 95))
    if diffs:
        out["seg_rel_mm"] = px_to_mm(float(np.percentile(diffs, 95)), WORK_DPI)
    return out, np.column_stack([segments, own])


def lineart_flow_metrics(
    gray_b,
    gray_a,
    field: Field | None,
    boxes: list,
    mode: Align = DEFAULT_ALIGN,
    prefix: str = "lineart",
    min_side_mm: float = MIN_BOX_MM,
    min_long_mm: float = MIN_BOX_MM,
    alignment: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Метрики страницы: худшая рамка по каждой мере.

    Один и тот же расчёт служит рамкам line art (``prefix="lineart"``) и рамкам формул (``prefix="formula"``).

    Args:
        gray_b: Рендер B 150 dpi.
        gray_a: Рендер A 150 dpi.
        field: Поле v16.
        boxes: Рамки в B, 150 dpi (дубли сливаются, :func:`merge_boxes`).
        mode: Режим выравнивания A в кадр B.
        prefix: Приставка имён метрик.
        min_side_mm: Рамка с короткой стороной меньше — не меряется (у формул — ``FORMULA_MIN_SIDE_MM``).
        min_long_mm: Рамка с длинной стороной меньше — не меряется.
        alignment: Готовое выравнивание ``(A в кадре B, остатки тайлов, матрица страницы)`` (:func:`page_alignment`)
            — чтобы не считать его заново для линий и формул; ``None`` — посчитать здесь.

    Returns:
        ``(метрики, виновники)``: ``<prefix>_aad_mm``, ``_aad_p95_mm``, ``_aniso_p95``, ``_shear_deg``,
        ``_aad_lines_mm``, ``_seg_bend_mm``, ``_seg_rel_mm``, ``_tilt_mm`` (уход конца рамки от наклона целиком),
        ``_tilt_deg`` — максимум по найденным рамкам; ``_skew_mm`` — ухудшение абсолютного наклона худшей рамки,
        ``_skew_gain_mm`` — выпрямление лучшей; ``_flow_lost`` — рамок, не найденных в A; ``_flow_ncc_min`` —
        худшая корреляция; виновники — рамки с худшим AAD и с худшим наклоном. Форма (v18, :func:`frame_shape`):
        меры ``SHAPE_NAMES`` с приставкой — у худшей рамки по ``_shape_mm`` (все берутся у неё, чтобы части
        складывались в одну картину), виновник ``_shape_mm``.
    """
    aligned, resid, matrix = alignment if alignment is not None else page_alignment(gray_b, gray_a, field, mode)
    measured = [
        measure_box(gray_b, aligned, box, field, resid, mode is Align.LOCAL, min_side_mm=min_side_mm,
                    min_long_mm=min_long_mm, page_map=matrix)
        for box in merge_boxes(boxes)
    ]  # fmt: skip
    measured = [m for m in measured if m is not None]
    found = [m for m in measured if m.found]
    names = ("aad_mm", "aad_p95_mm", "aniso_p95", "shear_deg", "aad_lines_mm", "seg_bend_mm", "seg_rel_mm", "bar_ptp_mm", "tilt_mm",
             "tilt_deg", "skew_mm", "skew_gain_mm", *SHAPE_NAMES)  # fmt: skip
    metrics = {
        f"{prefix}_flow_boxes": float(len(found)),
        f"{prefix}_flow_lost": float(len(measured) - len(found)),
        f"{prefix}_flow_ncc_min": min((m.ncc for m in measured), default=1.0),
        **{f"{prefix}_{name}": 0.0 for name in names},
    }
    culprits: dict[str, dict] = {}
    if not found:
        return metrics, culprits
    worst = max(found, key=lambda m: m.aad_mean_mm)
    tilted = max(found, key=lambda m: m.tilt_mm)
    metrics[f"{prefix}_aad_mm"] = worst.aad_mean_mm
    metrics[f"{prefix}_aad_p95_mm"] = max(m.aad_p95_mm for m in found)
    metrics[f"{prefix}_aniso_p95"] = max(m.aniso_p95 for m in found)
    metrics[f"{prefix}_shear_deg"] = max(m.shear_p95_deg for m in found)
    metrics[f"{prefix}_aad_lines_mm"] = max(m.aad_lines_mm for m in found)
    metrics[f"{prefix}_seg_bend_mm"] = max(m.seg_bend_mm for m in found)
    metrics[f"{prefix}_seg_rel_mm"] = max(m.seg_rel_mm for m in found)
    metrics[f"{prefix}_bar_ptp_mm"] = max(m.bar_ptp_mm for m in found)
    metrics[f"{prefix}_tilt_mm"] = tilted.tilt_mm
    metrics[f"{prefix}_tilt_deg"] = abs(tilted.tilt_deg)
    # Ухудшение абсолютного наклона: худшая рамка; выпрямление — лучшая (выигрыш, положительным числом).
    skewed = max(found, key=lambda m: m.skew_mm)
    metrics[f"{prefix}_skew_mm"] = max(0.0, skewed.skew_mm)
    metrics[f"{prefix}_skew_gain_mm"] = max(0.0, -min(m.skew_mm for m in found))
    culprits[f"{prefix}_skew_mm"] = {"b": list(skewed.box)}
    # Форма: все меры — у рамки с худшей групповой мерой (сырая мера — при равенстве, чтобы в кэше была видна
    # рамка, прощённая по прямоте).
    shapes = [(m.box, frame_shape(gray_b, aligned, m.box, field, matrix)) for m in found]
    shapes = [(box, shape) for box, shape in shapes if shape is not None]
    if shapes:
        box, shape = max(shapes, key=lambda item: (item[1]["shape_mm"], item[1]["shape_raw_mm"]))
        for name in SHAPE_NAMES:
            metrics[f"{prefix}_{name}"] = float(shape[name])
        culprits[f"{prefix}_shape_mm"] = {"b": list(box)}
    culprits[f"{prefix}_aad_mm"] = {"b": list(worst.box)}
    culprits[f"{prefix}_tilt_mm"] = {"b": list(tilted.box)}
    return metrics, culprits


__all__ = [
    "Align",
    "DEFAULT_ALIGN",
    "FlowMeasure",
    "align_page",
    "homography",
    "lineart_flow_metrics",
    "measure_box",
    "ink_angle",
    "merge_boxes",
    "page_alignment",
    "page_matrix",
    "shape_measures",
    "frame_shape",
    "straightness",
    "text_reference",
    "lsd_angle",
    "transfer_axis",
    "SHAPE_NAMES",
]
