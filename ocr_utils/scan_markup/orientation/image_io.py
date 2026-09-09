"""Чтение полосы в рабочие копии.

Одно разжатие на полосу, из него — всё остальное. Полоса пака-1 это 600 dpi, 3420x6071,
и разжимать её дважды (отдельно под tesseract, отдельно под сеть) значило бы удвоить
самую дорогую часть прогона.

Масштабов ДВА, и это не перестраховка: на разведке по четырём известным боковым полосам
tesseract OSD при 300 dpi узнал три из них, при 150 dpi — три ДРУГИЕ из тех же четырёх,
а объединение масштабов дало все четыре. Один масштаб терял бы полосу на ровном месте.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ocr_utils.scan_markup.orientation.detectors.base import (  # noqa: F401 (rotate_cw — реэкспорт)
    ROTATIONS,
    Frame,
    rotate_cw,
)

# Сканы заведомо свои, бомб не ждём, а 21 Мп PIL считает подозрительным размером.
Image.MAX_IMAGE_PIXELS = None

# Рабочие разрешения. 300 dpi — то, на что рассчитан tesseract; 150 dpi берётся из него же
# уменьшением, отдельного разжатия не стоит.
WORK_DPI_FINE = 300
WORK_DPI_COARSE = 150

# Ниже этого разрешение в теге считается мусором (бывает 1, бывает 72 от конвертера).
MIN_PLAUSIBLE_DPI = 72

# Качество JPEG, которым кадр едет из воркера в родителя под GPU-детекторы. Картинка уже
# уменьшена до ``gpu_side``, а 85 — та граница, за которой артефакты кодека начинают
# съедать тонкие штрихи текста.
GPU_JPEG_QUALITY = 85


def resolve_dpi(image: Image.Image, default_dpi: int) -> int:
    """Разрешение файла по тегу, с откатом на заданное умолчание."""
    dpi = image.info.get("dpi")
    if not dpi:
        return default_dpi
    try:
        value = int(round(float(dpi[0])))
    except (TypeError, ValueError):
        return default_dpi
    return value if value >= MIN_PLAUSIBLE_DPI else default_dpi


def _fit(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Приводит к точному размеру. INTER_AREA: уменьшение усреднением, без муара."""
    if image.shape[0] == target_h and image.shape[1] == target_w:
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def read_frame(
    path: Path, rel_path: str, default_dpi: int = 600, gpu_side: int = 0, allowed: tuple[int, ...] = ROTATIONS
) -> tuple[Frame, bytes | None]:
    """Кадр под CPU-детекторы плюс, если нужен, JPEG под GPU-детекторы.

    GPU-детекторы живут в родителе, а разжимает воркер — значит, картинку надо как-то
    передать между процессами. Передаётся именно JPEG, а не массив: уменьшенный до 1536
    кадр в сыром виде это 4 МБ на полосу, и очередь пула такими кусками забивает память
    быстрее, чем родитель успевает их разбирать. Тот же кадр в JPEG — около 150 КБ,
    и разжимается в родителе за десяток миллисекунд.
    """
    with Image.open(path) as image:
        full_w, full_h = image.size
        dpi = resolve_dpi(image, default_dpi)
        scale = WORK_DPI_FINE / dpi
        if scale < 1.0:
            # У JPEG draft уменьшает прямо при разжатии (в DCT-области) и почти бесплатен,
            # но кратности только степени двойки; у TIFF он не делает ничего.
            image.draft("RGB", (max(1, round(full_w * scale)), max(1, round(full_h * scale))))
        rgb = np.asarray(image.convert("RGB"))

    fine_h = max(1, round(full_h * WORK_DPI_FINE / dpi)) if scale < 1.0 else full_h
    fine_w = max(1, round(full_w * WORK_DPI_FINE / dpi)) if scale < 1.0 else full_w
    rgb_fine = _fit(rgb, fine_h, fine_w)

    gray300 = cv2.cvtColor(rgb_fine, cv2.COLOR_RGB2GRAY)
    gray150 = _fit(gray300, max(1, fine_h // 2), max(1, fine_w // 2))

    frame = Frame(
        rel_path=rel_path,
        path=path,
        width=full_w,
        height=full_h,
        dpi=dpi,
        gray150=np.ascontiguousarray(gray150),
        gray300=np.ascontiguousarray(gray300),
        allowed=allowed,
    )
    return frame, _encode_for_gpu(rgb_fine, gpu_side) if gpu_side else None


def _encode_for_gpu(rgb: np.ndarray, side: int) -> bytes:
    """Уменьшает до ``side`` по длинной стороне и кодирует в JPEG."""
    h, w = rgb.shape[:2]
    factor = side / max(h, w)
    if factor < 1.0:
        rgb = cv2.resize(rgb, (max(1, round(w * factor)), max(1, round(h * factor))), interpolation=cv2.INTER_AREA)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=GPU_JPEG_QUALITY)
    return buffer.getvalue()


def decode_gpu_jpeg(payload: bytes) -> Image.Image:
    """Обратная операция к :func:`_encode_for_gpu` — уже в родителе."""
    return Image.open(io.BytesIO(payload)).convert("RGB")
