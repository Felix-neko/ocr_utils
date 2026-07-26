"""Инпейнтинг пальца через Stable Diffusion + диспетчер методов (помойка).

Вынесено из ``finger_removal/finger_inpaint.py``: рабочий пайплайн
(``ocr_utils.scan_cropping``) закрашивает палец только LaMa, а SD-ветку
использовала лишь самостоятельная CLI ``legacy.finger_removal.detect_fingers``.
SD мощнее LaMa на текстурах, но на структурной кромке переплёта галлюцинирует —
выдумывает наклейки, текст и торчащие «язычки» бумаги; ``_suppress_protrusions``
как раз про борьбу с последними.

CPU-хелперы (ROI, растушёвка, разбиение маски на компоненты) берутся из рабочего
``scan_cropping.finger_removal.inpaint_roi``, сама LaMa — из ``GpuModels.inpaint``.
"""

import numpy as np
import torch

from ocr_utils.scan_cropping.finger_removal.inpaint_roi import DEFAULT_ROI_SCALE, blend_roi, mask_components, roi_bounds


DEFAULT_SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
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

_CACHE: dict = {}


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


def _load_sd(model_id: str, device: str):
    """Ленивая загрузка StableDiffusionInpaintPipeline с кэшем."""
    key = f"sd:{model_id}:{device}"
    if key not in _CACHE:
        from diffusers import StableDiffusionInpaintPipeline

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype, safety_checker=None)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        _CACHE[key] = pipe
    return _CACHE[key]


def sd_inpaint(
    rgb: np.ndarray,
    mask: np.ndarray,
    device: str = "cuda",
    padding: int = 64,
    feather: int = 9,
    sd_prompt: str = DEFAULT_SD_PROMPT,
    sd_negative: str = DEFAULT_SD_NEGATIVE,
    sd_steps: int = 30,
    sd_guidance: float = 3.0,
    sd_model: str = DEFAULT_SD_MODEL,
    sd_size: int = 512,
    roi_scale: float = DEFAULT_ROI_SCALE,
):
    """SD-инпейнтинг ПОКОМПОНЕНТНО: для каждой связной области свой ROI (вход — ``sd_size``)."""
    from PIL import Image

    comps = mask_components(mask)
    if not comps:
        return rgb.copy()

    pipe = _load_sd(sd_model, device)
    generator = torch.Generator(device=device).manual_seed(0)
    result = rgb.copy()
    for comp in comps:
        bounds = roi_bounds(comp, padding, roi_scale, rgb.shape[:2])
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        roi = result[y1:y2, x1:x2]
        mroi = comp[y1:y2, x1:x2]
        h0, w0 = roi.shape[:2]

        # ВАЖНО: сохраняем пропорции ROI. Если жёстко ужать в квадрат 512×512,
        # прямая вертикальная кромка книги смазывается в диагональ/блоб и после
        # обратного масштаба превращается в торчащую «бумажку». Поэтому масштабируем
        # по длинной стороне до sd_size, обе стороны кратны 8 (требование SD).
        scale = sd_size / max(h0, w0)
        nw = max(8, int(round(w0 * scale / 8)) * 8)
        nh = max(8, int(round(h0 * scale / 8)) * 8)

        img_pil = Image.fromarray(roi).resize((nw, nh), Image.BICUBIC)
        msk_pil = Image.fromarray((mroi > 0).astype(np.uint8) * 255).resize((nw, nh), Image.NEAREST)
        out = pipe(
            prompt=sd_prompt,
            negative_prompt=sd_negative,
            image=img_pil,
            mask_image=msk_pil,
            height=nh,
            width=nw,
            num_inference_steps=sd_steps,
            guidance_scale=sd_guidance,
            generator=generator,
        ).images[0]

        res = np.array(out.resize((w0, h0), Image.BICUBIC))
        # Убираем торчащие за кромку «язычки» (фоновую часть пальца заливаем цветом стола)
        res = _suppress_protrusions(roi, mroi, res)
        # Вклеиваем только маскированную область — изменения SD вне маски отбрасываем
        result[y1:y2, x1:x2] = blend_roi(roi, res, mroi, feather)
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
    sd_model: str = DEFAULT_SD_MODEL,
) -> np.ndarray:
    """Убирает пальцы под маской выбранным методом (``lama`` или ``sd``).

    ``rgb`` — RGB uint8, ``mask`` — uint8 0/255 (область пальца), ``models`` —
    ``scan_cropping.gpu_models.GpuModels``. Возвращает RGB uint8. Пустая маска
    возвращает исходник без изменений.
    """
    if int(np.count_nonzero(mask)) == 0:
        return rgb.copy()
    if method == "lama":
        return models.inpaint(rgb, mask, padding=padding, feather=feather)
    if method == "sd":
        return sd_inpaint(
            rgb,
            mask,
            device=models.device,
            padding=padding,
            feather=feather,
            sd_prompt=sd_prompt,
            sd_negative=sd_negative,
            sd_steps=sd_steps,
            sd_guidance=sd_guidance,
            sd_model=sd_model,
        )
    raise ValueError(f"Неизвестный метод инпейнтинга: {method}")
