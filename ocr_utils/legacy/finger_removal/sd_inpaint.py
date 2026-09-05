"""Инпейнтинг пальца через Stable Diffusion + диспетчер методов (помойка).

Вынесено из ``finger_removal/finger_inpaint.py``: рабочий пайплайн
(``ocr_utils.scan_cropping``) закрашивает палец только LaMa, а SD-ветку
использовала лишь самостоятельная CLI ``legacy.finger_removal.detect_fingers``.
SD мощнее LaMa на текстурах, но на структурной кромке переплёта галлюцинирует —
выдумывает наклейки, текст и торчащие «язычки» бумаги; ``_suppress_protrusions``
как раз про борьбу с последними.

Механика закраса (ROI, растушёвка, разбиение маски, прогон сетей) целиком
берётся из рабочего кода: цикл — ``inpainting.apply.inpaint_by_groups``, сами сети
— ``GpuModels``. Своего загрузчика SD здесь больше нет: пайплайн грузится там же,
где живут остальные модели, иначе видеопамять делили бы два независимых кэша.
"""

import numpy as np

from ocr_utils.inpainting.apply import inpaint_by_groups
from ocr_utils.inpainting.backends import DEFAULT_SD_MODEL as _DEFAULT_SD_MODEL, SdParams
from ocr_utils.inpainting.roi import DEFAULT_ROI_SCALE


# Реэкспорт: имя модели по умолчанию задано в общем ``inpainting.backends``, а
# здесь оставлено ради обратной совместимости импорта в ``detect_fingers``.
DEFAULT_SD_MODEL = _DEFAULT_SD_MODEL
# Под пальцем — кромка обложки и страниц книги, лежащей на чёрном столе. Прямо
# подсказываем это SD, чтобы он достраивал именно фон/кромку, а не выдумывал текст.
DEFAULT_SD_PROMPT = (
    "a clean straight edge of an old book lying flat on a plain black table; "
    "beyond the straight book edge there is only flat empty solid black background; "
    "smooth uniform surfaces, sharp neat straight edge, nothing protruding past the cover, "
    "no text, no letters, no fingers, no hand, no objects"
)
DEFAULT_SD_NEGATIVE = (
    "leaf, leaves, petal, plant, foliage, paper scrap, paper flap, torn paper, frayed edge, "
    "protruding paper, sticking out, bump, fold, curl, flap, tab, "
    "finger, fingers, hand, nail, skin, text, letters, numbers, words, logo, label, sticker, "
    "watermark, drawing, pattern, ornament, artifacts, blurry, distortion, extra objects, duplicate, noise"
)


def _book_left_edge(nd: np.ndarray, run: int, frac: float) -> np.ndarray:
    """Для каждой строки — x левой кромки книги (первый устойчивый не-тёмный участок).

    ``nd`` — бинарная карта «не тёмный» (1/0). Возвращает массив длины h: x, начиная
    с которого идёт книга; если книги в строке нет — ширина кадра (вся строка — стол).
    """
    h, w = nd.shape
    edge = np.full(h, w, dtype=np.float32)
    cs = np.cumsum(nd, axis=1)
    # Бегущее среднее по окну run: mean[x] = (cs[x+run-1]-cs[x-1]) / run
    csum = np.concatenate([np.zeros((h, 1), nd.dtype), cs], axis=1)
    run_mean = (csum[:, run:] - csum[:, :-run]) / run  # (h, w-run+1)
    for y in range(h):
        idx = np.argmax(run_mean[y] > frac)
        if run_mean[y, idx] > frac:
            edge[y] = idx
    return edge


def _suppress_protrusions(
    roi: np.ndarray, mroi: np.ndarray, res: np.ndarray, dark_thr: int = 60, run: int = 8, frac: float = 0.6
) -> np.ndarray:
    """Убирает торчащие за кромку «язычки», заливая фоновую часть пальца цветом стола.

    Палец входит сбоку: часть его маски лежит над ЧЁРНЫМ столом (слева от кромки
    книги), часть — над книгой (справа). И SD, и классический inpaint тянут кремовую
    кромку влево, в область, которая должна быть чёрной → торчащий язычок.

    Решаем геометрией кромки. По строкам ВНЕ маски (выше/ниже пальца) находим x левой
    кромки книги, линейно интерполируем её на строки пальца — получаем (почти прямую)
    линию настоящего края книги. Всё, что в маске левее этой линии, заливаем цветом
    стола; правее — оставляем заполнение SD (там действительно книга).
    """
    import cv2

    h, w = mroi.shape[:2]
    m = mroi > 0
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    nd = (gray >= dark_thr).astype(np.float32)  # «не тёмный» = книга/кромка

    edge = _book_left_edge(nd, run, frac)
    rows_masked = m.any(axis=1)
    good = np.where(~rows_masked)[0]
    if len(good) < 3:
        return res  # нет надёжных строк для оценки кромки

    # Интерполируем кромку на все строки по «чистым» строкам (вне маски)
    edge_interp = np.interp(np.arange(h), good, edge[good])

    xs = np.arange(w)[None, :]
    table_region = m & (xs < (edge_interp[:, None] - 1))
    if int(table_region.sum()) < 10:
        return res

    table_bg = (gray < dark_thr) & ~m
    if int(table_bg.sum()) > 30:
        bg_color = np.median(roi[table_bg], axis=0).astype(np.uint8)
    else:
        bg_color = np.array([20, 20, 20], np.uint8)
    out = res.copy()
    out[table_region] = bg_color
    return out


def sd_inpaint(
    rgb: np.ndarray,
    mask: np.ndarray,
    models,
    padding: int = 64,
    feather: int = 9,
    sd_prompt: str = DEFAULT_SD_PROMPT,
    sd_negative: str = DEFAULT_SD_NEGATIVE,
    sd_steps: int = 30,
    sd_guidance: float = 3.0,
    sd_size: int = 512,
    roi_scale: float = DEFAULT_ROI_SCALE,
):
    """SD-инпейнтинг ПОКОМПОНЕНТНО: для каждой связной области свой ROI.

    ``models`` — ``GpuModels``, созданный с ``sd_model=...``. Промпт здесь один на
    весь кадр: под пальцем всегда одно и то же — кромка книги на тёмном столе.
    (В закрасе разметки из CVAT промпт зависит от места на полосе, там он приходит
    колбэком, см. ``inpainting.backends.SdFiller``.)

    Сверх общего цикла остаётся одна пальце-специфичная поправка: SD любит
    достроить за кромку книги торчащий «язычок» бумаги, и ``_suppress_protrusions``
    его срезает.
    """
    params = SdParams(steps=sd_steps, guidance=sd_guidance, size=sd_size)

    def fill(roi, roi_mask, _bounds):
        filled = models.sd_fill_roi(roi, roi_mask, sd_prompt, sd_negative, params)
        return _suppress_protrusions(roi, roi_mask, filled)

    result, _rois = inpaint_by_groups(rgb, mask, fill, padding=padding, feather=feather, roi_scale=roi_scale)
    return result


def inpaint_fingers(
    rgb: np.ndarray,
    mask: np.ndarray,
    method: str,
    models,
    padding: int = 64,
    feather: int = 9,
    sd_prompt: str = DEFAULT_SD_PROMPT,
    sd_negative: str = DEFAULT_SD_NEGATIVE,
    sd_steps: int = 30,
    sd_guidance: float = 3.0,
) -> np.ndarray:
    """Убирает пальцы под маской выбранным методом (``lama`` или ``sd``).

    ``rgb`` — RGB uint8, ``mask`` — uint8 0/255 (область пальца), ``models`` —
    ``scan_cropping.gpu_models.GpuModels`` (для ``sd`` — созданный с ``sd_model=...``).
    Возвращает RGB uint8. Пустая маска возвращает исходник без изменений.
    """
    if int(np.count_nonzero(mask)) == 0:
        return rgb.copy()
    if method == "lama":
        return models.inpaint(rgb, mask, padding=padding, feather=feather)
    if method == "sd":
        return sd_inpaint(
            rgb,
            mask,
            models,
            padding=padding,
            feather=feather,
            sd_prompt=sd_prompt,
            sd_negative=sd_negative,
            sd_steps=sd_steps,
            sd_guidance=sd_guidance,
        )
    raise ValueError(f"Неизвестный метод инпейнтинга: {method}")
