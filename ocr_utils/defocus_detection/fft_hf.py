"""Независимый метод HF/MID — доля высокочастотной энергии (для кросс-проверки, этап E).

ИДЕЯ. Оптический расфокус физически убивает высокие пространственные частоты, оставляя
средние. Поэтому для каждого тайла считаем долю ВЧ-энергии:

    HF_ratio = E(высокие частоты) / E(средние частоты)

Величина почти не зависит от контента: и чёткий, и размытый тело-текст имеют схожую
средне-частотную энергию (шаг строк/букв), но при расфокусе ВЧ-составляющая (тонкие штрихи)
обрушивается. На карте HF_ratio зона расфокуса видна как двумерный «провал».

Это ВТОРОЙ, независимый от муара детектор. Он чувствительнее к тонкому зональному
расфокусу (на эталоне 1960/0650 перепад ratio ~10× против ~2× у муара), поэтому в pipeline
используется как кросс-проверка: совпадение обоих методов = высокая уверенность.

Ядро метода (``hf_ratio_map`` + ``detect_array``) повторяет валидированный одноимённый
скрипт ``detect_zonal_defocus.py`` в корне репозитория (там же — параллельный CLI для
больших папок и обоснование/валидация в focus_detection_report.md).
"""

import numpy as np

# Дефолтные параметры метода (откалиброваны в detect_zonal_defocus.py / focus_detection_report.md).
DEF_GRID_X, DEF_GRID_Y = 12, 8  # сетка тайлов по превью 4416×2944 (~370×370 px)
DEF_HF_ABS = 0.22  # абсолютный порог «обвала» доли ВЧ-энергии
DEF_HF_REL = 0.30  # порог относительно медианы резкости самой полосы
DEF_MIN_SEVERE = 4  # минимум «тяжёлых» тайлов в зоне
DEF_MIN_ROWS, DEF_MIN_COLS = 2, 3  # зона должна быть 2D: >=2 строк и >=3 столбцов


def hf_ratio_map(gray: np.ndarray, grid_x: int, grid_y: int) -> tuple[np.ndarray, np.ndarray]:
    """Карта доли ВЧ-энергии по тайлам и карта средне-частотной энергии.

    Args:
        gray: Полутоновое изображение.
        grid_x: Число тайлов по горизонтали.
        grid_y: Число тайлов по вертикали.

    Returns:
        (R, MID), где R[j, i] = HF_ratio тайла, а MID[j, i] — средне-частотная энергия
        (используется как «есть ли тут контент» для маски).
    """
    H, W = gray.shape
    R = np.zeros((grid_y, grid_x))
    MID = np.zeros((grid_y, grid_x))
    for j in range(grid_y):
        for i in range(grid_x):
            # границы текущего тайла (целочисленное деление — крайние тайлы могут чуть отличаться)
            y0, y1 = j * H // grid_y, (j + 1) * H // grid_y
            x0, x1 = i * W // grid_x, (i + 1) * W // grid_x
            t = gray[y0:y1, x0:x1].astype(np.float64)
            t -= t.mean()  # убираем постоянную составляющую (яркость фона)
            h, w = t.shape
            # окно Хэннинга гасит краевые разрывы тайла (иначе ложная ВЧ-энергия на швах)
            t = t * np.hanning(h)[:, None] * np.hanning(w)[None, :]
            # спектр мощности тайла
            F = np.abs(np.fft.rfft2(t)) ** 2
            fy = np.fft.fftfreq(h)[:, None]
            fx = np.fft.rfftfreq(w)[None, :]
            rad = np.sqrt(fy**2 + fx**2) / 0.5  # радиус в долях Найквиста (1.0 = Найквист)
            # СЧ-кольцо: общий «рисунок» текста (шаг строк/букв) — переживает расфокус
            mid = F[(rad > 0.10) & (rad <= 0.35)].sum()
            # ВЧ-кольцо: тонкие штрихи — первыми гибнут при расфокусе
            hi = F[(rad > 0.35) & (rad <= 0.85)].sum()
            MID[j, i] = mid
            R[j, i] = hi / (mid + 1e-9)
    return R, MID


def detect_array(
    gray: np.ndarray,
    grid_x: int = DEF_GRID_X,
    grid_y: int = DEF_GRID_Y,
    hf_abs: float = DEF_HF_ABS,
    hf_rel: float = DEF_HF_REL,
    min_severe: int = DEF_MIN_SEVERE,
    min_rows: int = DEF_MIN_ROWS,
    min_cols: int = DEF_MIN_COLS,
) -> tuple[bool, dict]:
    """Прогоняет метод HF/MID по уже загруженному grayscale-изображению.

    Args:
        gray: Полутоновое изображение.
        grid_x, grid_y: Размер сетки тайлов.
        hf_abs: Абсолютный порог обвала доли ВЧ.
        hf_rel: Порог относительно медианы резкости полосы.
        min_severe: Минимум «тяжёлых» тайлов в зоне.
        min_rows, min_cols: Зона должна быть двумерной (>= строк/столбцов).

    Returns:
        (is_defocus, info_dict) с полями ref, severe, rows, cols, bbox.
    """
    R, MID = hf_ratio_map(gray, grid_x, grid_y)

    # маска контента: тело-текст/детали (высокая СЧ-энергия), без полей, сгиба и пустот
    content = MID > np.percentile(MID, 50)
    ref = float(np.median(R[content]))  # «здоровая» резкость именно этой полосы

    # «тяжёлые» тайлы: одновременно абсолютный обвал ВЧ и обвал относительно своей же полосы
    severe = content & (R < hf_abs) & (R < hf_rel * ref)
    ys, xs = np.where(severe)
    cnt = len(ys)

    rows = cols = 0
    bbox = None
    if cnt >= min_severe:
        rows = int(ys.max() - ys.min() + 1)
        cols = int(xs.max() - xs.min() + 1)
        bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))

    # решение: это должна быть двумерная зона, а не одномерная полоса (фото/заголовок)
    is_defocus = cnt >= min_severe and rows >= min_rows and cols >= min_cols
    return is_defocus, dict(ref=ref, severe=cnt, rows=rows, cols=cols, bbox=bbox)
