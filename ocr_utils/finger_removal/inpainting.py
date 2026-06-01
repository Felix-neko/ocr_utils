"""Инпейнтинг области под пальцем.

Переключаемые бэкенды (опция --inpaint-method):
  - ``edge`` (по умолчанию) — продолжение полос кромки книги. Палец всегда входит
    с края, где содержимое (чёрный фон, мраморный переплёт, поле) постоянно вдоль
    края, поэтому маска заполняется интерполяцией соседних линий — «как было», без
    выдумок и чёрной заливки. Ось (вертикаль/горизонталь) выбирается ПОКОМПОНЕНТНО
    по тому, какой границы кадра касается конкретный палец (корректно для двух
    пальцев с разных краёв). См. edge_inpaint.
  - ``vertical`` / ``horizontal`` — то же продолжение, но ось задана принудительно.
  - ``lama`` — big-LaMa (torchscript). Хорош на текстурах, но рядом с большой
    чёрной зоной склонен заливать всё чёрным.
  - ``sd``   — Stable Diffusion inpainting. Мощнее, но на структурной кромке
    галлюцинирует (выдумывает наклейки/текст).

edge/vertical/horizontal работают покомпонентно по всему кадру (дёшево); lama/sd —
по одному ROI (bbox маски + контекст), чтобы не гонять нейросеть по всему кадру
3000×4300. Результат вклеивается с растушёвкой краёв маски (альфа=1 внутри маски,
мягкий спад только наружу).
"""

import logging

import cv2
import numpy as np
from PIL import Image

from .masking import MODELS_DIR

logger = logging.getLogger(__name__)

# Torchscript big-lama (тот же вес, что использует simple-lama-inpainting)
LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"

# Нейтральный промпт для SD: ровная чистая бумага/обложка без текста и без фактуры.
# Книга обычно лежит на ровном чёрном фоне — подсказываем это модели.
DEFAULT_SD_PROMPT = (
    "clean smooth uniform blank paper, plain flat book cover, solid even color, no texture, no text, "
    "book lying on a plain solid black background"
)
DEFAULT_SD_NEGATIVE = (
    "spots, stains, smudges, scratches, scuffs, marks, blemishes, dirt, wear, fingerprints, "
    "wrinkles, creases, noise, grain, texture, shadows, text, letters, words, hand, finger, "
    "blurry, artifacts, watermark, signature"
)

_PIPE_CACHE: dict = {}


# ============================================================
# ROI: bbox маски + контекст
# ============================================================


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Возвращает (x1, y1, x2, y2) ненулевой области маски или None, если пусто."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int, w: int, h: int) -> tuple[int, int, int, int]:
    """Расширяет bbox на pad пикселей с каждой стороны, обрезая по границам кадра."""
    x1, y1, x2, y2 = bbox
    return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)


# ============================================================
# Бэкенд LaMa
# ============================================================


def _load_lama(device: str):
    """Ленивая загрузка torchscript big-lama с кэшем; вес в MODELS_DIR."""
    key = f"lama:{device}"
    if key not in _PIPE_CACHE:
        import torch

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "big-lama.pt"
        if not model_path.exists():
            logger.info("Скачиваем big-lama...")
            torch.hub.download_url_to_file(LAMA_URL, str(model_path), progress=True)
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval().to(device)
        _PIPE_CACHE[key] = model
    return _PIPE_CACHE[key]


def _lama_inpaint(roi_rgb: np.ndarray, roi_mask: np.ndarray, device: str) -> np.ndarray:
    """Инпейнтит ROI через big-lama. roi_mask: uint8 0/255 (255 = заменять)."""
    import torch
    import torch.nn.functional as F

    model = _load_lama(device)
    h, w = roi_rgb.shape[:2]

    image = torch.from_numpy(roi_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    mask = torch.from_numpy(roi_mask).float().unsqueeze(0).unsqueeze(0) / 255.0
    mask = (mask > 0).float()

    # LaMa требует размеры кратные 8 — паддим отражением, потом обрезаем
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.inference_mode():
        out = model(image.to(device), mask.to(device))
    out = out[0].permute(1, 2, 0).detach().cpu().numpy()
    out = np.clip(out, 0, 255).astype(np.uint8)[:h, :w]
    return out


# ============================================================
# Бэкенд Stable Diffusion inpainting
# ============================================================


def _load_sd(device: str, model_id: str):
    """Ленивая загрузка StableDiffusionInpaintPipeline с кэшем."""
    key = f"sd:{device}:{model_id}"
    if key not in _PIPE_CACHE:
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype, safety_checker=None)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        _PIPE_CACHE[key] = pipe
    return _PIPE_CACHE[key]


def _round8(x: int) -> int:
    """Округляет вверх до кратного 8 (требование диффузионных моделей)."""
    return max(8, ((x + 7) // 8) * 8)


def _sd_inpaint(
    roi_rgb: np.ndarray,
    roi_mask: np.ndarray,
    device: str,
    prompt: str,
    negative: str,
    steps: int,
    model_res: int,
    model_id: str,
    guidance: float,
    smooth: float,
) -> np.ndarray:
    """Инпейнтит ROI через SD. ROI ресайзится так, чтобы длинная сторона = model_res.

    ``smooth`` (0..1) — степень пост-сглаживания результата с сохранением границ
    (edge-preserving фильтр). Убирает «придуманные» SD пятна/потёртости, оставляя
    бумагу ровной, но не размывая стыки материалов (край обложки/бумаги).
    """
    import torch

    pipe = _load_sd(device, model_id)
    h, w = roi_rgb.shape[:2]
    scale = model_res / max(h, w)
    tw, th = _round8(int(w * scale)), _round8(int(h * scale))

    img_pil = Image.fromarray(roi_rgb).resize((tw, th), Image.LANCZOS)
    mask_pil = Image.fromarray(roi_mask).convert("L").resize((tw, th), Image.NEAREST)

    generator = torch.Generator(device=device).manual_seed(0)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=img_pil,
        mask_image=mask_pil,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=tw,
        height=th,
        generator=generator,
    ).images[0]

    out = np.array(result.convert("RGB"))
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)

    if smooth > 0:
        # Edge-preserving сглаживание: выравнивает бумагу, сохраняя сильные границы
        flat = cv2.edgePreservingFilter(out, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.15)
        s = float(np.clip(smooth, 0.0, 1.0))
        out = np.clip((1.0 - s) * out + s * flat, 0, 255).astype(np.uint8)

    return out


# ============================================================
# Бэкенд "edge" — продолжение полос кромки книги (vertical/horizontal)
# ============================================================
#
# Идея: палец всегда входит в кадр С КРАЯ, а вдоль кромки книги содержимое
# (чёрный фон, мраморный переплёт, поле бумаги) ПОСТОЯННО вдоль этого края.
# Поэтому маску можно заполнить продолжением соседних линий — «как было», без
# выдумок (SD) и без чёрной заливки (LaMa).
#
# Направление продолжения зависит от того, с какого края вошёл палец:
#   - палец слева/справа  → полосы вертикальные  → заполняем вдоль КОЛОНОК (vertical);
#   - палец снизу/сверху  → полосы горизонтальные → заполняем вдоль СТРОК   (horizontal).
# Определяем это по тому, какой границы кадра маска касается сильнее
# (см. dominant_fill_axis / border_contact).


def border_contact(mask: np.ndarray) -> tuple[int, int]:
    """Сколько пикселей маски лежит на границах кадра.

    Возвращает (vertical_contact, horizontal_contact):
      - vertical_contact   — пиксели на ЛЕВОЙ + ПРАВОЙ кромках (столбцы x=0 и x=w-1);
      - horizontal_contact — пиксели на ВЕРХНЕЙ + НИЖНЕЙ кромках (строки y=0 и y=h-1).
    """
    m = mask > 0
    vertical = int(m[:, 0].sum() + m[:, -1].sum())
    horizontal = int(m[0, :].sum() + m[-1, :].sum())
    return vertical, horizontal


def dominant_fill_axis(mask: np.ndarray) -> str:
    """Выбирает ось заполнения по преобладающему касанию границы кадра.

    Если компонента сильнее касается горизонтальной границы (вошла снизу/сверху) —
    возвращает "horizontal", иначе "vertical" (вошла слева/справа).
    """
    vertical, horizontal = border_contact(mask)
    return "horizontal" if horizontal > vertical else "vertical"


def _fill_along_columns(roi_rgb: np.ndarray, fill_mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Продолжает каждую КОЛОНКУ: заполняет ``fill_mask`` интерполяцией по ``valid_mask``.

    Раздельные fill/valid важны при нескольких пальцах: заполняем пиксели ОДНОГО
    пальца, но опорными берём только реально валидные пиксели (не накрытые НИКАКОЙ
    маской), чтобы не интерполировать по коже другого пальца.

    Для каждого столбца берём валидные пиксели этого столбца как опорные и линейно
    интерполируем значения заполняемых. np.interp на краях зажимает к крайнему
    опорному значению (копия края).
    """
    h, w = fill_mask.shape[:2]
    out = roi_rgb.astype(np.float32).copy()
    ys = np.arange(h)
    for x in np.where(fill_mask.any(axis=0))[0]:  # только столбцы, где есть что заполнять
        fcol = fill_mask[:, x]
        vcol = valid_mask[:, x]
        if vcol.sum() < 2:  # нечем интерполировать — пропускаем
            continue
        yv = ys[vcol]
        for c in range(3):
            out[fcol, x, c] = np.interp(ys[fcol], yv, roi_rgb[vcol, x, c])
    return out


def _fill_component(roi_rgb: np.ndarray, fill_mask: np.ndarray, valid_mask: np.ndarray, axis: str) -> np.ndarray:
    """Заполнение одной компоненты вдоль оси (horizontal = vertical на транспонировании)."""
    if axis == "horizontal":
        out = _fill_along_columns(roi_rgb.transpose(1, 0, 2), fill_mask.T, valid_mask.T)
        return out.transpose(1, 0, 2)
    return _fill_along_columns(roi_rgb, fill_mask, valid_mask)


def edge_inpaint(rgb: np.ndarray, mask: np.ndarray, padding: int, force_axis: str | None = None) -> np.ndarray:
    """Заполняет маску продолжением кромки ПОКОМПОНЕНТНО.

    Каждый палец (связная компонента маски) обрабатывается отдельно: ось выбирается
    по тому, какой границы кадра касается ИМЕННО эта компонента (force_axis=None),
    либо принудительно задаётся. Это корректно, когда разные пальцы входят с разных
    краёв (например, один сверху, другой сбоку). Опорой служат только пиксели вне
    ЛЮБОЙ маски. Возвращает полноразмерное изображение с заполненными пикселями.
    """
    h, w = mask.shape[:2]
    out = rgb.astype(np.float32).copy()
    invalid_full = mask > 0  # ни один из этих пикселей не годится в опору
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    for i in range(1, num):
        comp_full = labels == i
        axis = force_axis or dominant_fill_axis(comp_full.astype(np.uint8))
        # ROI вокруг компоненты + контекст
        cx, cy = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        x1, y1 = max(0, cx - padding), max(0, cy - padding)
        x2, y2 = min(w, cx + cw + padding), min(h, cy + ch + padding)
        roi_rgb = rgb[y1:y2, x1:x2]
        fill = comp_full[y1:y2, x1:x2]
        valid = ~invalid_full[y1:y2, x1:x2]
        filled = _fill_component(roi_rgb, fill, valid, axis)
        out[y1:y2, x1:x2][fill] = filled[fill]
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================
# Публичная точка входа
# ============================================================


def inpaint_image(
    rgb: np.ndarray,
    mask: np.ndarray,
    method: str = "lama",
    device: str = "cuda",
    padding: int = 96,
    feather: int = 9,
    sd_prompt: str = DEFAULT_SD_PROMPT,
    sd_negative: str = DEFAULT_SD_NEGATIVE,
    sd_steps: int = 40,
    sd_model_res: int = 512,
    sd_model_id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting",
    sd_guidance: float = 5.0,
    sd_smooth: float = 0.6,
) -> np.ndarray:
    """Убирает палец: строит заполнение области маски и вклеивает его с растушёвкой.

    Методы edge/vertical/horizontal работают покомпонентно по всему кадру; lama/sd —
    по одному ROI вокруг всей маски (нейросети дороги, гоняем только нужный кусок).
    Если маска пустая — возвращает исходное изображение без изменений.
    """
    if int(np.count_nonzero(mask)) == 0:
        return rgb.copy()

    h, w = rgb.shape[:2]

    # Готовим полноразмерное «заполненное» изображение inpainted_full —
    # отличается от rgb только внутри маски; ниже оно бесшовно подмешивается.
    if method in ("edge", "vertical", "horizontal"):
        force_axis = None if method == "edge" else method
        inpainted_full = edge_inpaint(rgb, mask, padding=padding, force_axis=force_axis)
    elif method in ("lama", "sd"):
        # ROI вокруг всей маски + контекст
        bbox = _mask_bbox(mask)
        x1, y1, x2, y2 = _expand_bbox(bbox, padding, w, h)
        roi_rgb = rgb[y1:y2, x1:x2].copy()
        roi_mask = mask[y1:y2, x1:x2].copy()
        if method == "lama":
            roi_out = _lama_inpaint(roi_rgb, roi_mask, device)
        else:
            roi_out = _sd_inpaint(
                roi_rgb,
                roi_mask,
                device,
                sd_prompt,
                sd_negative,
                sd_steps,
                sd_model_res,
                sd_model_id,
                sd_guidance,
                sd_smooth,
            )
        inpainted_full = rgb.copy()
        inpainted_full[y1:y2, x1:x2] = roi_out
    else:
        raise ValueError(f"Неизвестный метод инпейнтинга: {method}")

    # Растушёванная альфа по маске для бесшовной вклейки.
    # Внутри маски альфа=1 (полный инпейнт), мягкий спад — ТОЛЬКО наружу,
    # иначе у внутренней кромки подмешивается исходный пиксель (кожа → розовое гало).
    mask_norm = (mask > 0).astype(np.float32)
    alpha = mask_norm
    if feather > 0:
        k = 2 * feather + 1
        alpha = np.maximum(mask_norm, cv2.GaussianBlur(mask_norm, (k, k), 0))
    alpha = alpha[:, :, np.newaxis]

    out = alpha * inpainted_full.astype(np.float32) + (1.0 - alpha) * rgb.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
