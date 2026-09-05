"""Расчёт по кадру: защитная маска контента → размытие фона → композиция.

Здесь только чистые функции над массивами: ничего не читается с диска и никуда
не пишется, обход пачки и сохранение — в ``pipeline``.

ПОЧЕМУ РАЗМЫТИЕ НОРМИРОВАННОЕ. Наивное ``blur(I_source)`` на сканах текста не
годится: в окно размытия попадают чернила соседних строк, и фон вне защитной
маски проседает. Замер на IMG_0130_1L (1966/03, 600 dpi, радиус размытия 60 px,
радиус дилатации 15 px, уровень бумаги 253):

    blur(I_source)                       фон вне маски 168-190, скачок на шве 18.3 (p95 63.2)
    нормированное, W = ~M_dilated        фон вне маски 241-253, скачок на шве  0.7 (p95  2.7)

Наивный вариант даёт тёмные полосы между строками и чёрную обводку по контуру
маски — ровно тот «перец», ради которого всё и затевалось. Нормированное
размытие исключает пиксели маски ИЗ САМОГО УСРЕДНЕНИЯ (а не зануляет их — от
зануления фон просел бы ещё сильнее) и делит на фактически набранный в окне вес::

    I_blurred = blur(I * W) / blur(W),    W = 1 вне M_dilated, 0 внутри

Деления на ноль в значимых пикселях быть не может: пиксель вне ``M_dilated``
всегда сам себе даёт вес в собственном окне, поэтому там знаменатель строго
положителен. Ноль возможен только внутри ``M_dilated``, где результат всё равно
отбрасывается композицией.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.scan_cropping.morphology import dilate_disk

# Методы построения первичной маски (значения --method).
METHOD_OTSU = "otsu"
METHOD_SAUVOLA = "sauvola"
MASK_METHODS = (METHOD_OTSU, METHOD_SAUVOLA)

# Сдвиг глобального порога от Оцу в сторону бумаги, доля расстояния до неё.
# Маску выгодно делать щедрой: лучше не размыть часть фона, чем размыть контент.
# Замер по 1966/03 (Оцу 146-152, бумага 252-253): при 0.5 порог ≈200, маска растёт
# с 12.6% до 15.2% кадра, и при этом НЕ затягивает просвет с оборота — 99.9%
# пикселей призрака лежат выше 226.
DEFAULT_THRESHOLD_BIAS = 0.5

# Минимальный разрыв между медианами тёмного и светлого классов Оцу, при котором
# считаем, что на странице вообще есть контент. Оцу всегда делит гистограмму надвое,
# даже если делить нечего: на чистом листе он режет собственное зерно бумаги и
# помечает контентом половину кадра. На реальном тексте разрыв больше 200 уровней
# (чернила ~40, бумага ~252), на пустом листе — единицы.
MIN_CONTENT_CONTRAST = 40.0

# Детектор растровых полос (обложки, полутоновые вкладки), которые этот подпакет
# трогать не должен: их фактура — содержимое, а не шум фона. Ищем КРУПНЫЕ СПЛОШНЫЕ
# области средних тонов: у текста таких нет (там либо чернила, либо бумага, а серое
# сидит тонкой каймой по краям букв), у растровой печати — есть.
HALFTONE_DOWNSCALE = 4  # во сколько раз уменьшать кадр перед поиском
HALFTONE_LO, HALFTONE_HI = 100, 225  # границы «средних тонов»
HALFTONE_OPEN_PX = 15  # сторона ядра размыкания: убирает тонкие каймы букв
# Доля кадра, с которой считаем, что растр есть. Замер по 1966/03 (98 файлов):
# обложки 4.95% и 10.12%, у всех 96 текстовых страниц — ровно 0.0, так что порог
# лежит посреди пустого промежутка.
HALFTONE_MIN_FRAC = 0.01

# Параметр k формулы Саволы. Меньше k — порог ближе к локальному среднему, то есть
# маска щедрее; классические 0.2 для нашей задачи излишне строги.
DEFAULT_SAUVOLA_K = 0.10

# Динамический диапазон СКО в формуле Саволы (канонические 128 для 8 бит).
SAUVOLA_R = 128.0

# Окно Саволы как доля длинной стороны кадра: ~101 px при 6100 px (600 dpi),
# порядка двух-трёх высот строки. Доля, а не константа — чтобы масштабировалось с DPI.
SAUVOLA_WINDOW_FRAC = 0.0165

# Радиус дилатации защитной маски как доля длинной стороны кадра: ~15 px при
# 6100 px (600 dpi), то есть ядро ≈30x30 — половинка средней буквы. Припуск нужен,
# чтобы под защиту попали полутона на границах букв, бледные перемычки и тонкие
# переходы: FineReader использует их при распознавании.
PROTECT_DILATE_FRAC = 0.00246

# Во сколько раз радиус размытия больше радиуса дилатации — ЗАПАСНОЙ способ
# задать размытие, когда его радиус не назван явно (см. :func:`blur_radius`).
DEFAULT_BLUR_MULT = 4.0

# Режимы построения размытого фона (значения --blur-mode). Живут здесь, рядом с
# самим размытием, а не в ``pipeline``: тот их реэкспортирует ради обратной
# совместимости импортов.
BLUR_MODE_MASKED = "masked"
BLUR_MODE_PLAIN = "plain"
BLUR_MODES = (BLUR_MODE_MASKED, BLUR_MODE_PLAIN)

# Число проходов box-фильтра, приближающих гауссиану (три — классический компромисс).
BLUR_PASSES = 3

# Порог знаменателя, ниже которого считаем, что опоры в окне нет вовсе.
DEN_EPS = 1e-6


def odd(value: int) -> int:
    """Ближайшее нечётное не меньше 1 — размер окна box-фильтра должен быть нечётным."""
    value = max(1, int(value))
    return value if value % 2 else value + 1


def analysis_samples(gray: np.ndarray, roi: "np.ndarray | None") -> np.ndarray:
    """Пиксели, по которым считаются пороги: весь кадр либо только область ``roi``.

    ``roi`` (uint8 0/255 того же размера) отсекает участки, чья гистограмма не
    относится к делу, — при ``--use-surya-layout`` это блоки-иллюстрации: средние
    тона фотографии тянут порог Оцу вверх, и часть бумаги вокруг текста уезжает
    под маску. Выборка приводится к столбцу (N, 1): ``cv2.threshold`` ждёт
    двумерный массив, а порядок пикселей для гистограммных методов не важен.
    """
    if roi is None:
        return gray
    return gray[roi > 0].reshape(-1, 1)


def _otsu(samples: np.ndarray) -> float:
    """Порог Оцу по выборке пикселей."""
    t_otsu, _ = cv2.threshold(samples, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t_otsu)


def global_threshold(gray: np.ndarray, bias: float = DEFAULT_THRESHOLD_BIAS, roi: "np.ndarray | None" = None) -> float:
    """Порог Оцу, сдвинутый в сторону бумаги на долю ``bias`` расстояния до неё.

    Уровень бумаги берётся как медиана пикселей светлее Оцу — то есть порог целиком
    вычисляется из поданной картинки, без привязки к конкретному паку. ``bias = 0``
    даёт чистый Оцу, ``bias = 1`` — уровень бумаги (маска станет почти всем кадром).

    ``roi`` ограничивает выборку (см. :func:`analysis_samples`); сам порог потом
    применяется ко всему кадру.
    """
    samples = analysis_samples(gray, roi)
    if samples.size == 0:  # анализировать нечего (весь кадр — иллюстрация)
        return 0.0
    t_otsu = _otsu(samples)
    lighter = samples[samples > t_otsu]
    if lighter.size == 0:  # одноцветный кадр — сдвигать не от чего
        return t_otsu
    paper = float(np.median(lighter))
    return t_otsu + bias * (paper - t_otsu)


def has_content(gray: np.ndarray, roi: "np.ndarray | None" = None) -> bool:
    """Есть ли на кадре что-то темнее бумаги, или это чистый лист.

    Оцу делит гистограмму надвое всегда, даже когда делить нечего, поэтому на пустой
    странице маска контента получилась бы размером в половину кадра, а результат —
    пятнистым (половина размыта, половина нет). Сравниваем медианы двух классов:
    настоящий текст даёт разрыв в сотни уровней, зерно чистой бумаги — единицы.

    ``roi`` ограничивает выборку (см. :func:`analysis_samples`). Пустая область —
    ``False``: если весь кадр занят иллюстрациями, сглаживать вне них нечего.
    """
    samples = analysis_samples(gray, roi)
    if samples.size == 0:
        return False
    t_otsu = _otsu(samples)
    darker, lighter = samples[samples <= t_otsu], samples[samples > t_otsu]
    if darker.size == 0 or lighter.size == 0:
        return False
    return float(np.median(lighter)) - float(np.median(darker)) >= MIN_CONTENT_CONTRAST


def has_halftone(gray: np.ndarray, min_frac: float = HALFTONE_MIN_FRAC, roi: "np.ndarray | None" = None) -> bool:
    """Есть ли на кадре крупная растровая (полутоновая) область — обложка или вкладка.

    Сглаживать такие кадры нельзя: их зерно — это само изображение, а не фактура фона,
    и размытие вне защитной маски выедает его островами. Признак — КРУПНЫЕ СПЛОШНЫЕ
    пятна средних тонов: у текста серое сидит тонкой каймой по краям букв и размыкание
    ядром ``HALFTONE_OPEN_PX`` её убирает, у растровой печати пятна остаются.

    Считается на копии, уменьшенной в ``HALFTONE_DOWNSCALE`` раз: признак крупный,
    и гонять морфологию по кадру в 21 Мп ради него незачем.

    ``roi`` исключает область из поиска И из знаменателя доли. При
    ``--use-surya-layout`` туда попадают уже найденные иллюстрации: искать растр на
    фотографии бессмысленно — он там заведомо есть, и вопрос ровно в том, есть ли
    он ЕЩЁ и в остальной части страницы.
    """
    h, w = gray.shape[:2]
    size = (max(1, w // HALFTONE_DOWNSCALE), max(1, h // HALFTONE_DOWNSCALE))
    # Интерполяция ИМЕНОВАННЫМ аргументом: третий позиционный у cv2.resize — это dst,
    # и cv2.INTER_AREA там молча проглатывался, а уменьшение шло INTER_LINEAR.
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    mid = (small > HALFTONE_LO) & (small < HALFTONE_HI)

    area = float(mid.size)
    if roi is not None:
        roi_small = cv2.resize(roi, size, interpolation=cv2.INTER_NEAREST) > 0
        mid &= roi_small
        area = float(np.count_nonzero(roi_small))
        if area == 0.0:
            return False

    kernel = np.ones((HALFTONE_OPEN_PX, HALFTONE_OPEN_PX), np.uint8)
    opened = cv2.morphologyEx(mid.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return float(np.count_nonzero(opened)) / area > min_frac


def _local_mean_std(gray: np.ndarray, window: int) -> "tuple[np.ndarray, np.ndarray]":
    """Локальные среднее и СКО в окне ``window`` — через box-фильтр в float32.

    ``skimage.filters.threshold_sauvola`` считает то же самое, но в float64: на кадре
    в 21 Мп это ~170 МБ на каждый промежуточный массив и несколько таких массивов
    сразу. ``cv2.boxFilter`` работает по бегущим суммам, то есть стоит O(1) на пиксель
    независимо от размера окна, и здесь его хватает с запасом.
    """
    gf = gray.astype(np.float32)
    ksize = (window, window)
    mean = cv2.boxFilter(gf, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(gf * gf, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return mean, np.sqrt(var)


def sauvola_window(shape: "tuple[int, ...]", window: "int | None" = None) -> int:
    """Размер окна Саволы: из длинной стороны кадра, если не задан явно."""
    if window is not None:
        return odd(window)
    return odd(int(round(SAUVOLA_WINDOW_FRAC * max(shape[0], shape[1]))))


def dilate_radius(
    shape: "tuple[int, ...]", dilate_px: "float | None" = None, dilate_frac: float = PROTECT_DILATE_FRAC
) -> float:
    """Радиус дилатации защитной маски: из длинной стороны кадра, если не задан явно."""
    if dilate_px is not None:
        return max(0.0, float(dilate_px))
    return max(1.0, dilate_frac * max(shape[0], shape[1]))


def blur_radius(
    shape: "tuple[int, ...]",
    blur_px: "float | None" = None,
    blur_frac: "float | None" = None,
    *,
    dilate_px: float,
    blur_mult: float = DEFAULT_BLUR_MULT,
) -> float:
    """Радиус размытия фона. Приоритет: явные пиксели → доля стороны → ``dilate_px * blur_mult``.

    ПОЧЕМУ ЭТО ОТДЕЛЬНАЯ ВЕЛИЧИНА. Раньше радиус размытия был жёстко производным
    от радиуса дилатации, и подобрать их независимо было нельзя: расширяешь
    защитный поясок — вместе с ним, молча, вчетверо сильнее раздувается размытие.
    При сравнении вариантов глазами это делает результат нечитаемым — непонятно,
    что именно изменилось.

    Запасная ветка (обе явные опции не заданы) воспроизводит прежнее поведение
    ЧИСЛО В ЧИСЛО, поэтому у существующих прогонов ничего не сдвинулось.
    """
    if blur_px is not None:
        return max(0.0, float(blur_px))
    if blur_frac is not None:
        return max(1.0, blur_frac * max(shape[0], shape[1]))
    return dilate_px * blur_mult


def primary_mask(
    gray: np.ndarray,
    method: str = METHOD_OTSU,
    bias: float = DEFAULT_THRESHOLD_BIAS,
    sauvola_k: float = DEFAULT_SAUVOLA_K,
    window: "int | None" = None,
    roi: "np.ndarray | None" = None,
) -> np.ndarray:
    """Первичная маска контента ``M_primary`` (uint8 0/255): напечатанное и подозрение на него.

    Сравнение нестрогое (``<=``) — как в самом OpenCV, где Оцу относит к фону то, что
    строго СВЕТЛЕЕ порога. На строго бимодальном кадре порог садится ровно на уровень
    чернил, и строгое ``<`` потеряло бы их целиком.

    Ветка ``sauvola`` ОБЪЕДИНЯЕТСЯ с глобальной маской, а не пересекается с ней.
    Смысл локального порога — добирать контент, который глобальный порог пропустил,
    то есть пиксели СВЕТЛЕЕ ``t_global``; пересечение вырезало бы ровно их и вдобавок
    выгрызало бы дыры в толстых штрихах (внутри крупного тёмного пятна локальное СКО
    около нуля, порог Саволы сваливается к ``m * (1 - k)`` и уходит ниже самих чернил).

    На ровной бумаге то же вырождение даёт порог примерно на ``k`` ниже уровня
    бумаги — при k = 0.1 это ~25 уровней, много больше зерна, так что ложных
    срабатываний на чистом поле нет. Но чем меньше ``sauvola_k``, тем ближе порог к
    бумаге, и на каком-то значении под защиту начнёт попадать просвет с оборота: он
    будет не размыт, а лишь не тронут. Это осознанный размен в пользу контента.

    ``roi`` ограничивает область, ПО КОТОРОЙ считается глобальный порог (см.
    :func:`analysis_samples`); сама маска строится по всему кадру. Внутри
    исключённых блоков-иллюстраций её значение роли не играет: такие блоки
    защищаются целиком и отдельно (см. ``pipeline.process_frame``).
    """
    if method not in MASK_METHODS:
        raise ValueError(f"неизвестный метод бинаризации: {method!r} (доступны {MASK_METHODS})")

    if not has_content(gray, roi):  # чистый лист — защищать нечего, сглаживается весь кадр
        return np.zeros(gray.shape, np.uint8)

    mask = gray <= global_threshold(gray, bias, roi)

    if method == METHOD_SAUVOLA:
        mean, std = _local_mean_std(gray, sauvola_window(gray.shape, window))
        t_local = mean * (1.0 + sauvola_k * (std / SAUVOLA_R - 1.0))
        mask |= gray <= t_local

    return mask.astype(np.uint8) * 255


def normalized_blur(
    img: np.ndarray, weight: "np.ndarray | None", radius_px: float, passes: int = BLUR_PASSES
) -> np.ndarray:
    """Сильное размытие изображения; ``weight`` (0/1) исключает пиксели из усреднения.

    ``weight=None`` — обычное размытие всего кадра (режим ``plain``). Иначе считается
    нормированная свёртка ``blur(I * W) / blur(W)``: см. мотивировку в докстринге модуля.
    Там, где опоры в окне не нашлось вовсе, пишется 0 — такие пиксели лежат строго
    внутри защитной маски и в результат не попадают, это лишь предохранитель от NaN.

    Размытие — ``passes`` проходов box-фильтра, а не ``cv2.GaussianBlur``: box
    разделим и стоит O(1) на пиксель независимо от радиуса, три прохода дают
    практически гауссиану. Гауссиана с sigma в десятки пикселей на кадре в 21 Мп
    обошлась бы в секунды.
    """
    ksize = (odd(int(round(radius_px * 2 + 1))),) * 2

    def box(a: np.ndarray) -> np.ndarray:
        for _ in range(max(1, passes)):
            a = cv2.boxFilter(a, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
        return a

    src = img.astype(np.float32)
    if weight is None:
        return np.clip(box(src), 0, 255).astype(np.uint8)

    # Знаменатель общий для всех каналов — маска одна на кадр, считается один раз.
    w = (weight > 0).astype(np.float32)
    den = box(w)
    support = den > DEN_EPS
    safe_den = np.where(support, den, 1.0)

    channels = [src] if src.ndim == 2 else [src[:, :, c] for c in range(src.shape[2])]
    out = [np.where(support, box(ch * w) / safe_den, 0.0) for ch in channels]
    stacked = out[0] if src.ndim == 2 else np.dstack(out)
    return np.clip(stacked, 0, 255).astype(np.uint8)


def compose(src: np.ndarray, blurred: np.ndarray, m_dilated: np.ndarray) -> np.ndarray:
    """Собирает результат: под защитной маской — исходник побитово, вне её — размытый фон."""
    mask = m_dilated > 0
    if src.ndim == 3:
        mask = mask[:, :, None]
    return np.where(mask, src, blurred)


@dataclass
class SmoothResult:
    """Результат расчёта по кадру: итог, обе маски и разрешённые радиусы.

    ``skip_reason`` непуст, когда кадр трогать нельзя, — тогда ``image`` это сам
    исходник, а маски пустые. Отдельным полем, а не исключением: вызывающий обязан
    такой кадр всё равно записать (пропустить файл совсем значило бы оставить дыру
    в паке), а причину — показать человеку.
    """

    image: np.ndarray
    m_primary: np.ndarray
    m_dilated: np.ndarray
    dilate_px: float
    blur_px: float
    skip_reason: str = ""


def smooth_frame(
    src: np.ndarray,
    gray: np.ndarray,
    *,
    protect_mask: "np.ndarray | None" = None,
    roi: "np.ndarray | None" = None,
    method: str = METHOD_OTSU,
    bias: float = DEFAULT_THRESHOLD_BIAS,
    sauvola_k: float = DEFAULT_SAUVOLA_K,
    sauvola_window: "int | None" = None,
    dilate_px: "float | None" = None,
    dilate_frac: float = PROTECT_DILATE_FRAC,
    blur_px: "float | None" = None,
    blur_frac: "float | None" = None,
    blur_mult: float = DEFAULT_BLUR_MULT,
    blur_mode: str = BLUR_MODE_MASKED,
    check_content: bool = True,
    check_halftone: bool = True,
) -> SmoothResult:
    """Полный расчёт по кадру: маска контента → размытие фона → композиция.

    Аргументы:
        src: что размывать и что вернуть — BGR или серый uint8;
        gray: серая версия ТОГО ЖЕ кадра, по ней считаются пороги;
        protect_mask: области, защищаемые ЦЕЛИКОМ (иллюстрации), uint8 0/255 или
            ``None``. Приходит либо от Surya, либо из базы разметки — этому модулю
            всё равно, откуда;
        roi: где считать пороги (обычно кадр минус иллюстрации), см.
            :func:`analysis_samples`;
        dilate_px / dilate_frac: радиус защитного припуска, см. :func:`dilate_radius`;
        blur_px / blur_frac / blur_mult: радиус размытия, см. :func:`blur_radius`;
        check_content, check_halftone: предохранители «кадр трогать нельзя».

    Возвращает :class:`SmoothResult`. Под защитной маской результат совпадает с
    ``src`` побитово.

    ``protect_mask`` дилатируется ОТДЕЛЬНО и присоединяется уже после: припуск у
    иллюстраций такой же, как у текста, но объединять маски до дилатации нельзя —
    на оверлее фотография залилась бы красным как «найденный контент».
    """
    zeros = np.zeros(gray.shape, np.uint8)
    radius = dilate_radius(gray.shape, dilate_px, dilate_frac)
    blur_r = blur_radius(gray.shape, blur_px, blur_frac, dilate_px=radius, blur_mult=blur_mult)

    # Два случая, когда кадр трогать нельзя, и оба кончаются возвратом исходника:
    #   * контент не выделяется — Оцу режет собственное зерно, маске верить нельзя;
    #   * есть крупная растровая область — обложка или полутоновая вкладка, там зерно
    #     и есть содержимое, а размытие выело бы его островами.
    if check_content and not has_content(gray, roi):
        return SmoothResult(src, zeros, zeros, radius, blur_r, "контент не выделяется (чистый лист?)")
    if check_halftone and has_halftone(gray, roi=roi):
        return SmoothResult(src, zeros, zeros, radius, blur_r, "крупная растровая область (обложка или вкладка?)")

    m_primary = primary_mask(gray, method=method, bias=bias, sauvola_k=sauvola_k, window=sauvola_window, roi=roi)

    m_dilated = dilate_disk(m_primary, radius)
    if protect_mask is not None and np.any(protect_mask):
        m_dilated = cv2.bitwise_or(m_dilated, dilate_disk(protect_mask, radius))

    weight = None if blur_mode == BLUR_MODE_PLAIN else (m_dilated == 0).astype(np.uint8)
    blurred = normalized_blur(src, weight, blur_r)
    return SmoothResult(compose(src, blurred, m_dilated), m_primary, m_dilated, radius, blur_r)
