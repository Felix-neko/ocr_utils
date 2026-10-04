"""Расплыв текста у корешка: горизонтальное сжатие глифов по шагу штрихов в полосах у сгиба.

Что видно на примерах (1979/07 IMG_0063/64, Экономист 1991/01 IMG_0046). Страница у корешка круто
уходит вниз, и последние 1–3 см строки снимаются под углом: буквы сжаты по горизонтали (по вертикали
нет) и вдобавок чуть размыты. Мера — горизонтальный шаг штрихов: первый пик автокорреляции строк
изображения в узкой вертикальной полосе. Вдали от корешка он постоянен (кегль и гарнитура одни),
у корешка падает пропорционально сжатию. Расфокус период почти не сдвигает, поэтому мера не путает
сжатие с общей мягкостью кадра.

Сгиб ищется по тени: межколонник — светлая бумага, корешок — тёмная впадина. Готовый
``gutter_loss_detection.geometry.fit_fold`` ищет минимум краски и на двухколонных полосах 1979 года
принимает межколонник за сгиб.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.gutter_loss_detection.geometry import build_masks

# Рабочий масштаб для поиска сгиба — тот же, под который настроена маска краски ``build_masks``.
WORK_WIDTH = 2000
# Полоса поиска сгиба, доли ширины кадра (книга бывает смещена от центра).
FOLD_LO, FOLD_HI = 0.30, 0.70
# Сглаживание профиля краски по столбцам, px рабочего масштаба.
FOLD_SMOOTH = 9
# Столбец свободен от краски, если её меньше этой доли от типичного (p75) уровня полосы поиска.
FOLD_FREE = 0.15
# Полуширина окна проверки «сгиб режет текст», px рабочего масштаба.
FOLD_CHECK = 3
# Строки тела кадра для поиска сгиба (без верхнего и нижнего обреза), доли высоты.
FOLD_TOP, FOLD_BOTTOM = 0.15, 0.85
# Уточнение сгиба по полосам высоты: окно вокруг грубой оценки, px рабочего масштаба; отбраковка точек.
FOLD_REFINE, FOLD_OUTLIER = 30, 10

# Полосы замера шага на полном разрешении (450 dpi, 17.7 px/мм): ширина, шаг, дальность от сгиба.
STRIP, STRIP_STEP, REACH = 64, 24, 1800
# Тело полосы по высоте для замера (без колонтитулов и нижнего поля), доли высоты кадра.
BODY_TOP, BODY_BOTTOM = 0.18, 0.86
# Столько горизонтальных зон по высоте замеряется отдельно: сжатие у корешка зависит от изгиба листа.
ZONES = 4
# Строка изображения «с краской»: доля тёмных пикселей детали не меньше этой.
ROW_INK = 0.02
# Глубина краски в детали изображения (серый минус фон), уровни 0..255.
INK_DEPTH = 25
# Минимум строк с краской в полосе зоны, чтобы шаг считался.
MIN_ROWS = 25
# Доля строк с краской, выше которой полоса — не текст (тень сгиба, край листа, линейка).
MAX_INKED_SHARE = 0.9
# Начало замера от сгиба, px: сама тень сгиба шириной ~10 px в замер не идёт.
FOLD_SKIP = 12
# Максимальный шаг, px: дальше — шаг слов, а не штрихов.
MAX_PITCH = 40
# Опорный шаг страницы — медиана по полосам не ближе этого к сгибу, px.
BASE_FROM = 500

LEFT, RIGHT = "L", "R"


def find_fold(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Прямая сгиба разворота: самый тёмный коридор без краски между массами текста.

    Межколонник и корешок оба свободны от краски, но межколонник — светлая бумага, а корешок в тени.
    Книга в кадре часто смещена (корешок на 53–60 % ширины у 1979/07), поэтому поиск идёт в широкой
    полосе ``FOLD_LO``–``FOLD_HI``, а не у центра.

    Args:
        gray: Полутоновый кадр (любой масштаб).

    Returns:
        Пара (коэффициенты прямой ``x = k*y + b`` в px переданного кадра; доля строк тела, у которых в
        окне ±``FOLD_CHECK`` px рабочего масштаба вокруг сгиба есть краска — мера «сгиб режет текст»,
        у верного сгиба близка к нулю).
    """
    height, width = gray.shape
    scale = WORK_WIDTH / width
    work = cv2.resize(gray, (WORK_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA)
    ink, _, _ = build_masks(work)
    wh = work.shape[0]
    y0, y1 = int(wh * FOLD_TOP), int(wh * FOLD_BOTTOM)
    # Профиль краски по столбцам (сглаженный) и медианная яркость столбца по телу кадра.
    ink_profile = cv2.blur(ink[y0:y1].mean(axis=0).astype(np.float32).reshape(1, -1), (FOLD_SMOOTH, 1)).ravel()
    light = np.median(cv2.GaussianBlur(work, (0, 0), 3)[y0:y1], axis=0)
    lo, hi = int(WORK_WIDTH * FOLD_LO), int(WORK_WIDTH * FOLD_HI)
    # Коридоры: столбцы, где краски меньше FOLD_FREE от типичной для текста.
    typical = float(np.percentile(ink_profile[lo:hi], 75))
    free = ink_profile < FOLD_FREE * max(typical, 1e-6)
    candidates = np.nonzero(free[lo:hi])[0] + lo
    if len(candidates) == 0:
        coarse = lo + int(np.argmin(light[lo:hi]))
    else:
        coarse = int(candidates[np.argmin(light[candidates])])
    # Уточнение по горизонтальным полосам высоты: минимум яркости в окне вокруг грубой оценки.
    blurred = cv2.GaussianBlur(work, (0, 0), 2)
    step = max(8, wh // 60)
    ys, xs = [], []
    left, right = max(0, coarse - FOLD_REFINE), min(WORK_WIDTH, coarse + FOLD_REFINE)
    for y in range(y0, y1, step):
        # Самый тёмный столбец среди свободных от краски: у плоско раскрытой книги тени у корешка
        # нет, и без этого условия «сгибом» становится край строк (1979/07 IMG_0024).
        column_light = blurred[y : y + step, left:right].mean(axis=0)
        inked = cv2.dilate(
            ink[y : y + step, left:right].any(axis=0).astype(np.uint8).reshape(1, -1), np.ones((1, 5), np.uint8)
        )
        column_light = np.where(inked.ravel() > 0, np.inf, column_light)
        if not np.isfinite(column_light).any():
            continue
        ys.append(y + step / 2)
        xs.append(left + int(np.argmin(column_light)))
    ys, xs = np.array(ys), np.array(xs, float)
    keep = np.abs(xs - np.median(xs)) < FOLD_OUTLIER if len(xs) else np.zeros(0, bool)
    line = np.polyfit(ys[keep], xs[keep], 1) if keep.sum() >= 6 else np.array([0.0, float(coarse)])
    # Проверка: режет ли найденный сгиб строки текста.
    rows = np.arange(y0, y1)
    fold_x = np.polyval(line, rows).astype(int)
    hits = [ink[r, max(0, x - FOLD_CHECK) : x + FOLD_CHECK + 1].any() for r, x in zip(rows, fold_x)]
    # Обратно в масштаб переданного кадра: x/scale = k*(y/scale) + b/scale.
    return np.array([line[0], line[1] / scale]), float(np.mean(hits))


def strip_pitch(block: np.ndarray) -> tuple[float, int]:
    """Горизонтальный шаг штрихов в полосе: первый пик автокорреляции строк с краской.

    Args:
        block: Полоса полутонового кадра float32 (строки × ширина ``STRIP``).

    Returns:
        Пара (шаг в px или NaN, число строк с краской).
    """
    detail = block - cv2.GaussianBlur(block, (0, 0), 10)
    inked = (detail < -INK_DEPTH).mean(axis=1) > ROW_INK
    rows = detail[inked]
    # Текст — строки с межстрочными промежутками; краска почти в каждой строке — это тень сгиба,
    # край листа или вертикальная линейка, а не набор.
    if len(rows) < MIN_ROWS or inked.mean() > MAX_INKED_SHARE:
        return float("nan"), 0
    rows = rows - rows.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(rows * np.hanning(rows.shape[1]), n=256, axis=1)
    acf = np.fft.irfft((np.abs(spectrum) ** 2).mean(axis=0), n=256)
    acf = acf / max(acf[0], 1e-9)
    # Первый локальный максимум после первого спада.
    lag = 2
    while lag < MAX_PITCH and acf[lag] <= acf[lag - 1]:
        lag += 1
    while lag < MAX_PITCH and acf[lag] >= acf[lag - 1]:
        lag += 1
    if lag >= MAX_PITCH:
        return float("nan"), len(rows)
    # Параболическое уточнение вершины.
    a, b, c = acf[lag - 2], acf[lag - 1], acf[lag]
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
    return float(lag - 1 + shift), len(rows)


@dataclass(frozen=True)
class SideProfile:
    """Профиль шага штрихов одной полосы разворота по зонам высоты.

    Attributes:
        side: ``L`` или ``R``.
        offsets: float[n] — расстояние центра полосы замера от сгиба, px полного кадра.
        pitch: float[ZONES, n] — шаг штрихов в полосе, px (NaN — нет текста).
        rows: int[ZONES, n] — строк изображения с краской в полосе (мера «здесь есть текст»).
        base: float[ZONES] — опорный шаг зоны (медиана вдали от сгиба).
    """

    side: str
    offsets: np.ndarray
    pitch: np.ndarray
    rows: np.ndarray
    base: np.ndarray

    @property
    def ratio(self) -> np.ndarray:
        """Сжатие: шаг полосы к опорному шагу зоны (1 — как на середине страницы)."""
        return self.pitch / self.base[:, None]


def side_profile(gray: np.ndarray, fold: np.ndarray, side: str) -> SideProfile:
    """Профиль шага штрихов полосы разворота от сгиба наружу.

    Args:
        gray: Полутоновый кадр float32 полного разрешения.
        fold: Прямая сгиба ``x = k*y + b`` в px этого кадра.
        side: ``L`` (левая полоса, идём от сгиба влево) или ``R``.

    Returns:
        ``SideProfile`` по ``ZONES`` зонам высоты.
    """
    height, width = gray.shape
    y_lo, y_hi = int(height * BODY_TOP), int(height * BODY_BOTTOM)
    edges = np.linspace(y_lo, y_hi, ZONES + 1).astype(int)
    offsets = np.arange(FOLD_SKIP, REACH, STRIP_STEP)
    pitch = np.full((ZONES, len(offsets)), np.nan)
    rows = np.zeros((ZONES, len(offsets)), int)
    for zone in range(ZONES):
        y0, y1 = edges[zone], edges[zone + 1]
        # Сгиб наклонный: в каждой зоне своя точка отсчёта.
        x_fold = float(np.polyval(fold, (y0 + y1) / 2))
        for n, offset in enumerate(offsets):
            x0 = int(x_fold + offset) if side == RIGHT else int(x_fold - offset - STRIP)
            if x0 < 0 or x0 + STRIP > width:
                break
            pitch[zone, n], rows[zone, n] = strip_pitch(gray[y0:y1, x0 : x0 + STRIP])
    far = offsets >= BASE_FROM
    with np.errstate(all="ignore"):
        base = np.nanmedian(np.where(far[None, :], pitch, np.nan), axis=1)
    return SideProfile(side=side, offsets=offsets + STRIP / 2, pitch=pitch, rows=rows, base=base)


# Край текста у сгиба: первая полоса, где строк с краской не меньше этой доли от медианы дальних полос.
EDGE_ROWS_SHARE = 0.6
# Сколько полос от края текста считается «краем» для меры сжатия (24 px шаг → ~3 мм на полосу).
EDGE_STRIPS = 3


@dataclass(frozen=True)
class EdgeCompression:
    """Сжатие шага штрихов у края текста, обращённого к сгибу, по зонам высоты.

    Attributes:
        margin: float[ZONES] — расстояние от сгиба до края текста, px (NaN — текста нет).
        edge_ratio: float[ZONES] — минимум отношения шага к опорному по ``EDGE_STRIPS`` полосам у края.
    """

    margin: np.ndarray
    edge_ratio: np.ndarray


def edge_compression(profile: SideProfile) -> EdgeCompression:
    """Сжатие у края текста по профилю полосы.

    Args:
        profile: Профиль шага штрихов полосы разворота.

    Returns:
        ``EdgeCompression`` по зонам высоты.
    """
    margin = np.full(ZONES, np.nan)
    edge_ratio = np.full(ZONES, np.nan)
    far = profile.offsets >= BASE_FROM
    for zone in range(ZONES):
        typical = np.median(profile.rows[zone, far]) if far.any() else 0
        if typical < MIN_ROWS:
            continue
        hits = np.nonzero(profile.rows[zone] >= EDGE_ROWS_SHARE * typical)[0]
        if len(hits) == 0:
            continue
        first = int(hits[0])
        margin[zone] = profile.offsets[first] - STRIP / 2
        window = profile.ratio[zone, first : first + EDGE_STRIPS]
        if np.isfinite(window).any():
            edge_ratio[zone] = float(np.nanmin(window))
    return EdgeCompression(margin=margin, edge_ratio=edge_ratio)


# --- Мера по глифам (вторая версия) ---------------------------------------------------------------------------
# Профиль шага штрихов по полосам (выше) путает сжатие у корешка со всем, что даёт вертикальную периодичность:
# стопкой обрезов листов у корешка, отточиями оглавлений, линейками таблиц. Низ его рейтинга по паку — сплошь
# такие артефакты (отсмотр 2026-10-02). Мера по глифам берёт только связные компоненты высотой с букву и
# сравнивает их форму (ширина/высота) у края текста с формой вдали от сгиба.

# Буква: высота в долях медианной высоты компонент тела кадра; ширина — не шире стольких высот.
GLYPH_H_LO, GLYPH_H_HI, GLYPH_W_HI = 0.65, 1.5, 2.5
# Компоненты для оценки медианной высоты буквы, px полного кадра (450 dpi: корпус ~23–26 px).
GLYPH_SEED_H = (10, 80)
# Полосы от края текста у сгиба, px: «у края» (~4 буквы) и «следом» (контроль, что сжатие — именно у края).
NEAR_WIDTH, NEXT_WIDTH = 100, 150
# Опорная форма — у букв дальше этого от сгиба, px.
GLYPH_BASE_FROM = 500
# Край текста у сгиба — этот процентиль расстояний букв до сгиба (устойчив к одиночному сору).
EDGE_PERCENTILE = 1.0
# Минимум букв на стороне, чтобы мерить.
MIN_GLYPHS = 200


@dataclass(frozen=True)
class GlyphEdge:
    """Сжатие букв у края текста, обращённого к сгибу, на одной полосе разворота.

    Attributes:
        near_ratio: Медиана ширины/высоты букв в полосе ``NEAR_WIDTH`` у края к опорной (1 — как вдали).
        next_ratio: То же для следующей полосы ``NEXT_WIDTH`` (контроль).
        zone_ratio: float[ZONES] — ``near_ratio`` по зонам высоты (NaN — мало букв).
        edge: Расстояние края текста от сгиба, px.
        glyphs: Букв на стороне.
        near_glyphs: Букв в полосе у края.
        glyph_h: Медианная высота буквы кадра, px.
    """

    near_ratio: float
    next_ratio: float
    zone_ratio: np.ndarray
    edge: float
    glyphs: int
    near_glyphs: int
    glyph_h: float


def glyph_components(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Буквы кадра: связные компоненты краски высотой с букву.

    Args:
        gray: Полутоновый кадр float32 полного разрешения.

    Returns:
        (x центров, y центров, ширины, высоты, медианная высота буквы) — по буквам тела кадра.
    """
    height = gray.shape[0]
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 10)
    ink = (detail < -INK_DEPTH).astype(np.uint8)
    _, _, stats, centers = cv2.connectedComponentsWithStats(ink, connectivity=8)
    w, h = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
    cx, cy = centers[1:, 0], centers[1:, 1]
    body = (cy > BODY_TOP * height) & (cy < BODY_BOTTOM * height)
    seed = body & (h > GLYPH_SEED_H[0]) & (h < GLYPH_SEED_H[1])
    glyph_h = float(np.median(h[seed])) if seed.any() else float("nan")
    keep = body & (h > GLYPH_H_LO * glyph_h) & (h < GLYPH_H_HI * glyph_h) & (w >= 3) & (w < GLYPH_W_HI * glyph_h)
    return cx[keep], cy[keep], w[keep].astype(float), h[keep].astype(float), glyph_h


def glyph_edge(
    cx: np.ndarray,
    cy: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    glyph_h: float,
    fold: np.ndarray,
    side: str,
    height: int,
) -> GlyphEdge | None:
    """Сжатие букв у края текста одной полосы разворота.

    Args:
        cx, cy, w, h: Буквы кадра (``glyph_components``).
        glyph_h: Медианная высота буквы.
        fold: Прямая сгиба ``x = k*y + b``.
        side: ``L`` или ``R``.
        height: Высота кадра, px (для зон).

    Returns:
        ``GlyphEdge`` или None, если букв на стороне меньше ``MIN_GLYPHS`` или нет опоры вдали от сгиба.
    """
    signed = cx - np.polyval(fold, cy)
    mine = signed < 0 if side == LEFT else signed > 0
    if mine.sum() < MIN_GLYPHS:
        return None
    dist, aspect, ys = np.abs(signed[mine]), (w / h)[mine], cy[mine]
    far = dist > GLYPH_BASE_FROM
    if far.sum() < MIN_GLYPHS // 2:
        return None
    base = float(np.median(aspect[far]))
    edge = float(np.percentile(dist, EDGE_PERCENTILE))
    near = (dist >= edge) & (dist < edge + NEAR_WIDTH)
    following = (dist >= edge + NEAR_WIDTH) & (dist < edge + NEAR_WIDTH + NEXT_WIDTH)
    edges = np.linspace(BODY_TOP * height, BODY_BOTTOM * height, ZONES + 1)
    zone_ratio = np.full(ZONES, np.nan)
    for zone in range(ZONES):
        inside = near & (ys >= edges[zone]) & (ys < edges[zone + 1])
        if inside.sum() >= 15:
            zone_ratio[zone] = float(np.median(aspect[inside])) / base
    return GlyphEdge(
        near_ratio=float(np.median(aspect[near])) / base if near.sum() >= 30 else float("nan"),
        next_ratio=float(np.median(aspect[following])) / base if following.sum() >= 30 else float("nan"),
        zone_ratio=zone_ratio,
        edge=edge,
        glyphs=int(mine.sum()),
        near_glyphs=int(near.sum()),
        glyph_h=glyph_h,
    )
