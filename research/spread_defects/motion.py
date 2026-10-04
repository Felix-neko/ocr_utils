"""Зональный смаз (motion blur) по относительному кепстру плиток: двоение штрихов как пик кепстра на векторе сдвига.

Идея. Смаз при съёмке с рук на этом паке — не гладкое пятно, а «двойная экспозиция»: штрих и его
ослабленная копия со сдвигом d (на 1985/02 IMG_0017 — около 5 px по горизонтали). Двойное изображение
умножает спектр на косинусную рябь с периодом 1/d, и в кепстре (обратное преобразование лог-спектра)
появляется пик ровно на векторе сдвига (dx, dy) — классика оценки смаза (Fabian & Malah 1991,
Park et al. 2011). Сам текст тоже даёт структуру в кепстре (ширина штриха, шаг букв и строк), поэтому
лог-спектр плитки сравнивается с медианным лог-спектром текстовых плиток того же кадра: спектр
набора сокращается, остаётся «передаточная функция» плитки относительно типичной. Гладкий расфокус
пика не даёт — он отдельно виден как потеря высоких частот ``hf_loss``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Сторона плитки на полном разрешении (450 dpi): ~4 строки корпуса, хватает на сдвиги до 25 px.
TILE = 256

# Кольцо поиска пика кепстра, px: ниже — скат главного лепестка, выше — шаг букв и строк. Нижняя граница 4:
# при 3 рубленый шрифт служебных полос 1989–90 давал ложный диагональный «сдвиг» (3,−2) ≈ 3.6 px по всему
# кадру, а настоящее двоение на паке — 4–5 px (отсмотр топ-32 ранга, 2026-10-02).
RING_MIN, RING_MAX = 4.0, 25.0

# Сигма вычитания освещения в плитке, px: больше штриха, меньше плитки.
ILLUM_SIGMA = 15.0

# Полоса частот (цикл/px), где сравниваются спектры; ниже — освещение и строки, выше — шум JPEG.
BAND_LO, BAND_HI = 0.02, 0.35

# Высокие частоты для меры потери резкости, цикл/px.
HF_LO, HF_HI = 0.12, 0.35

# Текстовая плитка: светлая малонасыщенная бумага с заметной, но не сплошной краской.
PAPER_MIN_V = 110  # медиана яркости плитки, 0..255
PAPER_MAX_S = 90  # медиана насыщенности плитки, 0..255 (бумага кремовая, обложки синие, кожа розовая)
INK_DROP = 60  # краска — темнее светлого уровня плитки (p90) на столько
INK_MIN, INK_MAX = 0.02, 0.35  # доля краски в плитке
PAPER_SHARE = 0.45
LINE_LENGTH = 61  # прямая длиннее буквы и межбуквенного пробела: линейка, рамка, штрихкод, край листа  # доля пикселей, близких к светлому уровню (бумага, а не тень/обложка)
MIN_DETAIL_STD = 8.0  # минимальная СКО после вычитания освещения — иначе плитка пустая


@dataclass(frozen=True)
class MotionTiles:
    """Сетка признаков смаза по плиткам одного кадра.

    Attributes:
        text: bool[gy, gx] — плитка текстовая (по ней считаются признаки и опора кадра).
        ink: float32[gy, gx] — доля краски в текстовой плитке (0 у нетекстовых): отличает плотный
            основной набор от заголовков и линеек.
        lines: float32[gy, gx] — доля краски плитки, лежащей на длинных прямых (линейки, рамки схем,
            штрихкод, край листа): у основного набора около нуля.
        z: float32[gy, gx] — высота пика относительного кепстра в робастных СКО фона кольца
            (0 у нетекстовых).
        dx, dy: int16[gy, gx] — вектор пика кепстра, px (знак условен: кепстр симметричен).
        hf_loss: float32[gy, gx] — потеря высоких частот относительно опоры, натуральный лог мощности.
        ref: float16[TILE, TILE] — медианный лог-спектр текстовых плиток кадра (опора; нужен для
            сравнения кадра с выпуском).
    """

    text: np.ndarray
    ink: np.ndarray
    lines: np.ndarray
    z: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    hf_loss: np.ndarray
    ref: np.ndarray


def _hann(size: int) -> np.ndarray:
    """Двумерное окно Ханна ``size``×``size`` (float32) против утечки спектра от краёв плитки."""
    window = np.hanning(size).astype(np.float32)
    return np.outer(window, window)


_WINDOW = _hann(TILE)
_FY = np.fft.fftfreq(TILE)[:, None]
_FX = np.fft.fftfreq(TILE)[None, :]
_FR = np.hypot(_FY, _FX)
_BAND = ((_FR > BAND_LO) & (_FR < BAND_HI)).astype(np.float32)
_HF = (_FR > HF_LO) & (_FR < HF_HI)
_LAG_Y, _LAG_X = np.mgrid[-TILE // 2 : TILE // 2, -TILE // 2 : TILE // 2]
_RING = (np.hypot(_LAG_Y, _LAG_X) >= RING_MIN) & (np.hypot(_LAG_Y, _LAG_X) <= RING_MAX)


def line_share(value: np.ndarray, light: float) -> float:
    """Доля краски плитки, лежащей на длинных горизонтальных или вертикальных прямых.

    Args:
        value: Канал V плитки (0..255).
        light: Светлый уровень плитки (p90 V).

    Returns:
        Отношение пикселей прямых длиной от ``LINE_LENGTH`` к пикселям краски (0, если краски нет).
    """
    ink = (value < light - INK_DROP).astype(np.uint8)
    total = int(ink.sum())
    if total == 0:
        return 0.0
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, LINE_LENGTH), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((LINE_LENGTH, 1), np.uint8))
    return float((horizontal | vertical).sum()) / total


def tile_ink(hsv: np.ndarray) -> float | None:
    """Доля краски в плитке, если плитка похожа на печатный текст на бумаге.

    Args:
        hsv: Плитка в HSV (OpenCV, V и S в 0..255).

    Returns:
        Доля пикселей краски (темнее светлого уровня плитки на ``INK_DROP``) или None, если плитка
        тёмная, насыщенная (обложка, кожа), без краски, сплошь в краске или бумаги в ней мало.
    """
    value = hsv[..., 2]
    if np.median(value) < PAPER_MIN_V or np.median(hsv[..., 1]) > PAPER_MAX_S:
        return None
    light = float(np.percentile(value, 90))
    ink = float((value < light - INK_DROP).mean())
    paper = float((value > light - 25).mean())
    return ink if INK_MIN <= ink <= INK_MAX and paper >= PAPER_SHARE else None


def tile_log_spectrum(gray_tile: np.ndarray) -> np.ndarray | None:
    """Лог-спектр мощности плитки после вычитания освещения и нормировки контраста.

    Args:
        gray_tile: Плитка TILE×TILE float32.

    Returns:
        Лог-спектр TILE×TILE float32 или None, если плитка почти пустая.
    """
    detail = gray_tile - cv2.GaussianBlur(gray_tile, (0, 0), ILLUM_SIGMA)
    std = float(detail.std())
    if std < MIN_DETAIL_STD:
        return None
    spectrum = np.fft.fft2((detail / std) * _WINDOW)
    return np.log(np.abs(spectrum) ** 2 + 1e-2).astype(np.float32)


def relative_peak(log_spectrum: np.ndarray, ref: np.ndarray) -> tuple[float, int, int, float]:
    """Пик кепстра разности лог-спектров плитки и опоры.

    Args:
        log_spectrum: Лог-спектр плитки.
        ref: Опорный лог-спектр (медиана текстовых плиток кадра).

    Returns:
        Кортеж (z, dx, dy, hf_loss): высота пика в робастных СКО кольца, вектор пика в px,
        средняя потеря лог-мощности на высоких частотах (плюс — плитка мягче опоры).
    """
    diff = (log_spectrum - ref) * _BAND
    hf_loss = -float(diff[_HF].mean())
    cepstrum = np.fft.fftshift(np.real(np.fft.ifft2(diff)))
    values = cepstrum[_RING]
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median))) + 1e-9
    # Только настоящие локальные максимумы (выше всех 8 соседей, включая точки внутри кольца):
    # иначе «пиком» становится скат главного лепестка на внутренней границе кольца — так шапки
    # рубрик с линейками давали ложный сдвиг (3, 0) на резких кадрах 1985/02.
    local_max = cepstrum >= cv2.dilate(cepstrum, np.ones((3, 3), np.uint8))
    masked = np.where(_RING & local_max, cepstrum, -np.inf)
    if not np.isfinite(masked).any():
        return 0.0, 0, 0, hf_loss
    py, px = np.unravel_index(int(np.argmax(masked)), cepstrum.shape)
    z = (float(cepstrum[py, px]) - median) / mad
    return z, int(px - TILE // 2), int(py - TILE // 2), hf_loss


def motion_tiles(rgb: np.ndarray) -> MotionTiles:
    """Признаки смаза по сетке плиток кадра полного разрешения.

    Args:
        rgb: Кадр H×W×3 uint8, полное разрешение (сдвиги в px полного кадра).

    Returns:
        ``MotionTiles`` с сеткой ``H // TILE`` × ``W // TILE``.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gy, gx = gray.shape[0] // TILE, gray.shape[1] // TILE
    text = np.zeros((gy, gx), bool)
    ink = np.zeros((gy, gx), np.float32)
    lines = np.zeros((gy, gx), np.float32)
    spectra: dict[tuple[int, int], np.ndarray] = {}
    # Первый проход: текстовые плитки и их лог-спектры.
    for row in range(gy):
        for col in range(gx):
            ys, xs = slice(row * TILE, (row + 1) * TILE), slice(col * TILE, (col + 1) * TILE)
            ink_share = tile_ink(hsv[ys, xs])
            if ink_share is None:
                continue
            spectrum = tile_log_spectrum(gray[ys, xs])
            if spectrum is None:
                continue
            text[row, col] = True
            ink[row, col] = ink_share
            value = hsv[ys, xs][..., 2]
            lines[row, col] = line_share(value, float(np.percentile(value, 90)))
            spectra[(row, col)] = spectrum
    z = np.zeros((gy, gx), np.float32)
    dx = np.zeros((gy, gx), np.int16)
    dy = np.zeros((gy, gx), np.int16)
    hf_loss = np.zeros((gy, gx), np.float32)
    if len(spectra) < 4:
        return MotionTiles(text, ink, lines, z, dx, dy, hf_loss, np.zeros((TILE, TILE), np.float16))
    # Опора — поэлементная медиана: устойчива, пока смазано меньше половины текста кадра.
    ref = np.median(np.stack(list(spectra.values())), axis=0)
    # Второй проход: пик относительного кепстра каждой текстовой плитки.
    for (row, col), spectrum in spectra.items():
        z[row, col], dx[row, col], dy[row, col], hf_loss[row, col] = relative_peak(spectrum, ref)
    return MotionTiles(text, ink, lines, z, dx, dy, hf_loss, ref.astype(np.float16))


# Агрегат кадра. Минимум соседей-внутренних плиток в окне 3×3, чтобы медиана была осмысленной.
MIN_WINDOW = 5
# Векторы сдвига соседей считаются согласными, если отличаются не больше чем на столько px.
SHIFT_AGREE = 1.6
# Плитка основного набора: доля краски не ниже этой. На 1985/02 корпус даёт 0.12–0.20, шапки рубрик,
# заголовки и линейки — 0.03–0.09; заголовки (особенно контурный шрифт с тенью) дают ложное «двоение».
BODY_INK = 0.11
# Плитка основного набора: доля краски на длинных прямых не выше этой (схемы, штрихкоды обложек, линейки
# таблиц и край листа у корешка давали ложное «двоение» в топе ранга). Первое приближение, калибруется.
BODY_LINES = 0.05
# Порог «смазанной» плитки по сглаженному z — для площади зоны (первое приближение, калибруется по паку).
ZONE_Z = 8.0


def interior_mask(text: np.ndarray) -> np.ndarray:
    """Текстовые плитки, у которых все четыре соседа тоже текстовые.

    Краевые плитки текстовой области (колонтитул, обрез книжного блока, край страницы) несут прямые
    края и полосы, которые дают ложный пик кепстра, — в оценку кадра они не идут.

    Args:
        text: bool-сетка текстовых плиток.

    Returns:
        bool-сетка внутренних плиток.
    """
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    return cv2.erode(text.astype(np.uint8), cross, borderValue=0).astype(bool)


def canonical_shift(dx: int, dy: int) -> tuple[int, int]:
    """Вектор сдвига с точностью до знака (кепстр симметричен): приводим к dx > 0 или dx == 0, dy > 0.

    Args:
        dx, dy: Вектор пика кепстра.

    Returns:
        Тот же вектор или противоположный.
    """
    return (dx, dy) if (dx > 0 or (dx == 0 and dy > 0)) else (-dx, -dy)


@dataclass(frozen=True)
class MotionScore:
    """Оценка смаза кадра по плиточной сетке.

    Attributes:
        peak: Максимум по внутренним плиткам медианы z в окне 3×3 (сила самой смазанной зоны).
        agree_peak: То же, но медиана берётся только у плиток, чей вектор сдвига согласен с центром
            (смаз — одно движение камеры, вектор в зоне общий; случайные пики текста — вразнобой).
        zone_share: Доля внутренних плиток со сглаженным z не меньше ``ZONE_Z``.
        shift_dx, shift_dy: Вектор сдвига в точке ``agree_peak`` (канонический знак), px.
        row, col: Плитка ``agree_peak`` в сетке.
        interior: Число внутренних плиток.
        median_z: Медиана z по внутренним плиткам (фон кадра).
    """

    peak: float
    agree_peak: float
    zone_share: float
    shift_dx: int
    shift_dy: int
    row: int
    col: int
    interior: int
    median_z: float


def smoothed_maps(tiles: MotionTiles) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Сглаженные карты z по внутренним плиткам.

    Args:
        tiles: Плиточная сетка кадра.

    Returns:
        Тройка (inner, plain, agree): маска внутренних плиток основного набора (``BODY_INK``); медиана z в окне 3×3 по внутренним
        соседям; медиана z в окне только по соседям с согласным вектором сдвига (с весом доли согласных).
        Где окно набрало меньше ``MIN_WINDOW`` плиток, карты равны 0.
    """
    inner = interior_mask(tiles.text) & (tiles.ink >= BODY_INK) & (tiles.lines <= BODY_LINES)
    gy, gx = inner.shape
    plain = np.zeros((gy, gx), np.float32)
    agree = np.zeros((gy, gx), np.float32)
    shifts = {(r, c): canonical_shift(int(tiles.dx[r, c]), int(tiles.dy[r, c])) for r, c in zip(*np.nonzero(inner))}
    for (row, col), center in shifts.items():
        window = [
            (r, c)
            for r in range(max(0, row - 1), min(gy, row + 2))
            for c in range(max(0, col - 1), min(gx, col + 2))
            if inner[r, c]
        ]
        if len(window) < MIN_WINDOW:
            continue
        plain[row, col] = float(np.median([tiles.z[r, c] for r, c in window]))
        agreeing = [
            (r, c)
            for r, c in window
            if np.hypot(shifts[(r, c)][0] - center[0], shifts[(r, c)][1] - center[1]) <= SHIFT_AGREE
        ]
        # Медиана по согласным соседям, умноженная на долю согласных: одиночный пик не набирает веса.
        agree[row, col] = float(np.median([tiles.z[r, c] for r, c in agreeing])) * len(agreeing) / len(window)
    return inner, plain, agree


def motion_score(tiles: MotionTiles) -> MotionScore:
    """Оценка смаза кадра из плиточной сетки.

    Args:
        tiles: Плиточная сетка кадра (``motion_tiles`` или npz прогона).

    Returns:
        ``MotionScore``; у кадра без внутренних плиток все величины нулевые.
    """
    inner, plain, agree = smoothed_maps(tiles)
    if not inner.any():
        return MotionScore(0.0, 0.0, 0.0, 0, 0, -1, -1, 0, 0.0)
    row, col = np.unravel_index(int(np.argmax(agree)), agree.shape)
    dx, dy = canonical_shift(int(tiles.dx[row, col]), int(tiles.dy[row, col]))
    return MotionScore(
        peak=float(plain.max()),
        agree_peak=float(agree[row, col]),
        zone_share=float((plain[inner] >= ZONE_Z).mean()),
        shift_dx=dx,
        shift_dy=dy,
        row=int(row),
        col=int(col),
        interior=int(inner.sum()),
        median_z=float(np.median(tiles.z[inner])),
    )


def frame_vs_issue(ref: np.ndarray, issue_ref: np.ndarray) -> tuple[float, int, int, float]:
    """Смаз всего кадра: пик кепстра разности опорного спектра кадра и медианы опор выпуска.

    Зональная оценка сравнивает плитки с опорой того же кадра и равномерный смаз всего кадра не видит
    (опора смазана так же). Здесь опора кадра сравнивается с медианой опор кадров выпуска — набор тот же,
    а смаз у соседних кадров разный.

    Args:
        ref: Опорный лог-спектр кадра (``MotionTiles.ref``).
        issue_ref: Медиана опорных лог-спектров кадров выпуска.

    Returns:
        То же, что ``relative_peak``: (z, dx, dy, hf_loss).
    """
    return relative_peak(ref.astype(np.float32), issue_ref.astype(np.float32))


# --- Полутоновый ореол ---------------------------------------------------------------------------------------
# Смаз (двоение) и расфокус окружают штрих полутоном: между ядром краски и бумагой много пикселей промежуточной
# светлоты. У резкого полужирного набора и заголовков, которые дают ложный пик кепстра, краска однородно тёмная.
# На разметке топ-64 рейтинга смаза доля полутонов у смаза 0.23–0.70 (медиана 0.48), у ложных 0.14–0.29
# (медиана 0.21) — отсмотр 2026-10-02, ``sets/motion.csv``.

# Краска — темнее бумаги (p90) на эту долю размаха «бумага − ядро»; полутон — светлее ядра (p3) на эту долю.
MID_INK, MID_CORE = 0.2, 0.45


def midtone_share(gray_tile: np.ndarray) -> float:
    """Доля полутонов среди краски плитки.

    Args:
        gray_tile: Полутоновая плитка (любой тип).

    Returns:
        Отношение числа пикселей краски промежуточной светлоты к числу пикселей краски.
    """
    paper, core = np.percentile(gray_tile, (90, 3))
    span = max(float(paper - core), 1.0)
    dark = gray_tile < paper - MID_INK * span
    middle = dark & (gray_tile > core + MID_CORE * span)
    return float(middle.sum()) / max(int(dark.sum()), 1)


@dataclass(frozen=True)
class MidtoneTiles:
    """Сетка доли полутонов по плиткам кадра.

    Attributes:
        text: bool[gy, gx] — текстовая плитка (``tile_ink``).
        ink: float32[gy, gx] — доля краски текстовой плитки.
        lines: float32[gy, gx] — доля краски на длинных прямых.
        mid: float32[gy, gx] — доля полутонов среди краски (0 у нетекстовых).
    """

    text: np.ndarray
    ink: np.ndarray
    lines: np.ndarray
    mid: np.ndarray


def midtone_tiles(rgb: np.ndarray) -> MidtoneTiles:
    """Доля полутонов по сетке плиток ``TILE`` кадра полного разрешения.

    Args:
        rgb: Кадр H×W×3 uint8.

    Returns:
        ``MidtoneTiles`` с сеткой ``H // TILE`` × ``W // TILE`` (та же, что у ``motion_tiles``).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gy, gx = gray.shape[0] // TILE, gray.shape[1] // TILE
    text = np.zeros((gy, gx), bool)
    ink, lines, mid = (np.zeros((gy, gx), np.float32) for _ in range(3))
    for row in range(gy):
        for col in range(gx):
            ys, xs = slice(row * TILE, (row + 1) * TILE), slice(col * TILE, (col + 1) * TILE)
            share = tile_ink(hsv[ys, xs])
            if share is None:
                continue
            value = hsv[ys, xs][..., 2]
            text[row, col], ink[row, col] = True, share
            lines[row, col] = line_share(value, float(np.percentile(value, 90)))
            mid[row, col] = midtone_share(gray[ys, xs])
    return MidtoneTiles(text, ink, lines, mid)


@dataclass(frozen=True)
class MidtoneScore:
    """Полутоновый ореол кадра: у пика кепстра и самая полутоновая зона.

    Attributes:
        at_peak: Медиана доли полутонов в окне 3×3 вокруг плитки пика кепстра (NaN — нет плиток).
        at_peak_rel: То же к медиане по плиткам основного набора кадра.
        zone_rel: Максимум по окнам 3×3 медианы доли полутонов к медиане кадра (размытие без пика кепстра).
        zone_row, zone_col: Центр этого окна.
        frame_median: Медиана доли полутонов по плиткам основного набора.
    """

    at_peak: float
    at_peak_rel: float
    zone_rel: float
    zone_row: int
    zone_col: int
    frame_median: float


def window_median(values: np.ndarray, mask: np.ndarray, row: int, col: int) -> float:
    """Медиана сетки по плиткам маски в окне 3×3 вокруг (row, col).

    Args:
        values: Сетка значений.
        mask: bool-сетка плиток, которые идут в медиану.
        row, col: Центр окна.

    Returns:
        Медиана или NaN, если плиток маски в окне меньше ``MIN_WINDOW``.
    """
    rows, cols = slice(max(0, row - 1), row + 2), slice(max(0, col - 1), col + 2)
    chosen = values[rows, cols][mask[rows, cols]]
    return float(np.median(chosen)) if len(chosen) >= MIN_WINDOW else np.nan


def midtone_score(mid: MidtoneTiles, row: int, col: int) -> MidtoneScore:
    """Полутоновый ореол кадра.

    Args:
        mid: Сетка полутонов кадра.
        row, col: Плитка пика кепстра (``MotionScore.row/col``; -1 — пика нет).

    Returns:
        ``MidtoneScore``; без плиток основного набора — NaN и -1.
    """
    body = interior_mask(mid.text) & (mid.ink >= BODY_INK) & (mid.lines <= BODY_LINES)
    if body.sum() < MIN_WINDOW:
        return MidtoneScore(np.nan, np.nan, np.nan, -1, -1, np.nan)
    frame_median = float(np.median(mid.mid[body]))

    at_peak = window_median(mid.mid, body, row, col) if row >= 0 else np.nan
    windows = {(r, c): window_median(mid.mid, body, r, c) for r, c in zip(*np.nonzero(body))}
    windows = {k: v for k, v in windows.items() if np.isfinite(v)}
    (zr, zc), zone = max(windows.items(), key=lambda kv: kv[1]) if windows else ((-1, -1), np.nan)
    return MidtoneScore(
        at_peak=at_peak,
        at_peak_rel=at_peak / frame_median if frame_median > 0 else np.nan,
        zone_rel=zone / frame_median if frame_median > 0 else np.nan,
        zone_row=int(zr),
        zone_col=int(zc),
        frame_median=frame_median,
    )
