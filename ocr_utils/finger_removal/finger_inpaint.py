"""Инпейнтинг области пальца: LaMa или Stable Diffusion (реализация с нуля).

Почему отдельный модуль и почему «по ROI». Палец всегда входит с края книги, где
в кадре доминирует ЧЁРНЫЙ фон. Если гнать нейросеть по всему снимку 5696×4272,
LaMa «затягивает» дыру доминирующим чёрным фоном, а Stable Diffusion начинает
галлюцинировать текст/детали. Поэтому здесь сеть работает по ТЕСНОМУ ROI вокруг
маски (с контекстным полем ``padding``): сеть видит локальный контекст — кромку
переплёта, поле страницы — и достраивает именно его. Результат вклеивается обратно
ТОЛЬКО внутри маски с растушёвкой краёв (``feather``), поэтому остальная часть
кадра не трогается, а шов незаметен.

Веса LaMa (torchscript big-lama) лежат/качаются в ``finger_models/``. SD-пайплайн
тянется из HuggingFace по ``sd_model``.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "finger_models"
LAMA_WEIGHTS = MODELS_DIR / "big-lama.pt"
# Тот же torchscript-вес, что использует simple-lama-inpainting
LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"

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

# ROI вокруг маски увеличиваем в 1.5 раза — сети нужен контекст кромки/фона, иначе
# дыра «заливается» доминирующим цветом (LaMa) либо выдумываются детали (SD).
DEFAULT_ROI_SCALE = 1.5

# Кэш загруженных моделей (чтобы не грузить заново на каждом кадре)
_CACHE: dict = {}


# ============================================================
# ROI вокруг маски + растушёванное вклеивание
# ============================================================


def _mask_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Возвращает (x1, y1, x2, y2) — bbox ненулевых пикселей маски, либо None."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _roi_bounds(
    mask: np.ndarray, padding: int, roi_scale: float, shape: tuple[int, int]
) -> Optional[tuple[int, int, int, int]]:
    """ROI вокруг маски: bbox + ``padding``, затем масштаб ``roi_scale`` от центра.

    Итоговый прямоугольник обрезается границами кадра ``shape`` (h, w). Возвращает
    (x1, y1, x2, y2) или None, если маска пустая.
    """
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    h, w = shape
    x1, y1, x2, y2 = bbox
    # Поле контекста
    x1, y1, x2, y2 = x1 - padding, y1 - padding, x2 + padding, y2 + padding
    # Масштаб вокруг центра
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * roi_scale, (y2 - y1) * roi_scale
    x1, x2 = int(round(cx - bw / 2)), int(round(cx + bw / 2))
    y1, y2 = int(round(cy - bh / 2)), int(round(cy + bh / 2))
    # Обрезаем по кадру
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    """Разбивает маску на отдельные связные компоненты (список масок uint8 0/255)."""
    num, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return [((labels == i).astype(np.uint8) * 255) for i in range(1, num)]


def roi_bounds_list(
    mask: np.ndarray, padding: int = 64, roi_scale: float = DEFAULT_ROI_SCALE
) -> list[tuple[int, int, int, int]]:
    """ROI каждой связной компоненты маски (список (x1, y1, x2, y2)) — для отладки.

    Покомпонентно, чтобы несколько разнесённых пальцев не сливались в один
    гигантский ROI на всю полосу кадра.
    """
    rois = []
    for comp in _mask_components(mask):
        b = _roi_bounds(comp, padding, roi_scale, mask.shape[:2])
        if b is not None:
            rois.append(b)
    return rois


def _blend(orig: np.ndarray, filled: np.ndarray, mask: np.ndarray, feather: int) -> np.ndarray:
    """Вклеивает ``filled`` в ``orig`` по маске с мягким спадом краёв (alpha=1 внутри)."""
    m = (mask > 0).astype(np.float32)
    if feather > 0:
        k = 2 * feather + 1
        a = cv2.GaussianBlur(m, (k, k), 0)
        a = np.maximum(a, m)  # внутри маски заполнение всегда полное
    else:
        a = m
    a = a[..., None]
    return (a * filled.astype(np.float32) + (1.0 - a) * orig.astype(np.float32)).astype(np.uint8)


# ============================================================
# LaMa (torchscript big-lama)
# ============================================================


def _ensure_lama_weights() -> None:
    """Качает big-lama.pt в finger_models/, если его ещё нет."""
    if LAMA_WEIGHTS.exists():
        return
    import urllib.request

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Качаю веса LaMa: %s", LAMA_URL)
    urllib.request.urlretrieve(LAMA_URL, str(LAMA_WEIGHTS))


def _load_lama(device: str):
    """Ленивая загрузка torchscript big-lama с кэшем."""
    key = f"lama:{device}"
    if key not in _CACHE:
        _ensure_lama_weights()
        model = torch.jit.load(str(LAMA_WEIGHTS), map_location=device)
        model.eval().to(device)
        _CACHE[key] = model
    return _CACHE[key]


def _pad_to_modulo(arr: np.ndarray, mod: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    """Симметрично дополняет CHW-массив до кратности ``mod``; возвращает (arr, (h, w))."""
    _, h, w = arr.shape
    ph = (mod - h % mod) % mod
    pw = (mod - w % mod) % mod
    padded = np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode="symmetric")
    return padded, (h, w)


def _lama_fill_roi(model, roi: np.ndarray, mroi: np.ndarray, device: str) -> np.ndarray:
    """Прогон LaMa по одному ROI; возвращает заполненный ROI (RGB uint8, тот же размер)."""
    img = (roi.astype(np.float32) / 255.0).transpose(2, 0, 1)  # CHW
    msk = ((mroi > 0).astype(np.float32))[None, ...]  # 1HW

    img_p, (oh, ow) = _pad_to_modulo(img)
    msk_p, _ = _pad_to_modulo(msk)

    it = torch.from_numpy(img_p).unsqueeze(0).to(device)
    mt = torch.from_numpy(msk_p).unsqueeze(0).to(device)
    mt = (mt > 0).float()

    with torch.inference_mode():
        out = model(it, mt)
    res = out[0].permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(res * 255.0, 0, 255).astype(np.uint8)[:oh, :ow]


def lama_inpaint(
    rgb: np.ndarray,
    mask: np.ndarray,
    device: str = "cuda",
    padding: int = 64,
    feather: int = 9,
    roi_scale: float = DEFAULT_ROI_SCALE,
):
    """LaMa-инпейнтинг ПОКОМПОНЕНТНО: для каждой связной области свой ROI. RGB uint8."""
    comps = _mask_components(mask)
    if not comps:
        return rgb.copy()

    model = _load_lama(device)
    result = rgb.copy()
    for comp in comps:
        bounds = _roi_bounds(comp, padding, roi_scale, rgb.shape[:2])
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        roi = result[y1:y2, x1:x2]
        mroi = comp[y1:y2, x1:x2]
        filled = _lama_fill_roi(model, roi, mroi, device)
        result[y1:y2, x1:x2] = _blend(roi, filled, mroi, feather)
    return result


# ============================================================
# Stable Diffusion inpainting
# ============================================================


def _book_left_edge(nd: np.ndarray, run: int, frac: float) -> np.ndarray:
    """Для каждой строки —x левой кромки книги (первый устойчивый не-тёмный участок).

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
    bg_color = np.median(roi[table_bg], axis=0).astype(np.uint8) if int(table_bg.sum()) > 30 else np.array(
        [20, 20, 20], np.uint8
    )
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

    comps = _mask_components(mask)
    if not comps:
        return rgb.copy()

    pipe = _load_sd(sd_model, device)
    generator = torch.Generator(device=device).manual_seed(0)
    result = rgb.copy()
    for comp in comps:
        bounds = _roi_bounds(comp, padding, roi_scale, rgb.shape[:2])
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
        result[y1:y2, x1:x2] = _blend(roi, res, mroi, feather)
    return result


# ============================================================
# Диспетчер
# ============================================================


def inpaint_fingers(
    rgb: np.ndarray,
    mask: np.ndarray,
    method: str,
    device: str = "cuda",
    padding: int = 64,
    feather: int = 9,
    sd_prompt: str = DEFAULT_SD_PROMPT,
    sd_negative: str = DEFAULT_SD_NEGATIVE,
    sd_steps: int = 30,
    sd_guidance: float = 3.0,
    sd_model: str = DEFAULT_SD_MODEL,
) -> np.ndarray:
    """Убирает пальцы под маской выбранным методом (``lama`` или ``sd``).

    rgb — RGB uint8, mask — uint8 0/255 (область пальца). Возвращает RGB uint8.
    Пустая маска возвращает исходник без изменений.
    """
    if int(np.count_nonzero(mask)) == 0:
        return rgb.copy()
    if method == "lama":
        return lama_inpaint(rgb, mask, device=device, padding=padding, feather=feather)
    if method == "sd":
        return sd_inpaint(
            rgb,
            mask,
            device=device,
            padding=padding,
            feather=feather,
            sd_prompt=sd_prompt,
            sd_negative=sd_negative,
            sd_steps=sd_steps,
            sd_guidance=sd_guidance,
            sd_model=sd_model,
        )
    raise ValueError(f"Неизвестный метод инпейнтинга: {method}")
