"""Движок page-dewarp (Matt Zucker / lmmx) — классический алгоритм без нейросети.

Обнаруживает текстовые строки, оптимизирует кубическую модель поверхности страницы и
ремаппит изображение. Работает на CPU, веса/репозиторий не нужны (pip-пакет
``page-dewarp``). Выдаёт grayscale (особенность алгоритма) — нормально для downstream OCR.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ocr_utils.legacy.dewarp.engines.base import DewarpEngine

logger = logging.getLogger(__name__)

# Поле детекции = 0: углы страницы берутся от полного кадра, поэтому ремап покрывает
# ВСЮ ширину (page-dewarp ничего не обрезает по бокам).
PAGE_MARGIN_X = 0
PAGE_MARGIN_Y = 0
# Размер копии для детекции: крупнее дефолта (1280×700) — точнее детекция строк.
SCREEN_MAX_W = 1500
SCREEN_MAX_H = 2100


def _remap_color(img: np.ndarray, page_dims, params, config) -> np.ndarray:
    """Цветной ремап по параметрам page-dewarp (та же геометрия, но не grayscale).

    Повторяет вычисление карты из ``page_dewarp.dewarp.RemappedImage``, но применяет
    ``cv2.remap`` к ЦВЕТНОМУ ``img`` (библиотека ремапит только grayscale).
    """
    from page_dewarp.normalisation import norm2pix
    from page_dewarp.projection import project_xy

    # Ширину выхода фиксируем РОВНО по входу (ничего не обрезаем и не растягиваем по X);
    # высоту берём пропорционально форме страницы (это и есть вертикальное выпрямление).
    out_w = int(img.shape[1])
    out_h = max(1, int(round(out_w * page_dims[1] / page_dims[0])))

    width_small = max(2, out_w // config.REMAP_DECIMATE)
    height_small = max(2, out_h // config.REMAP_DECIMATE)
    page_x_range = np.linspace(0, page_dims[0], width_small)
    page_y_range = np.linspace(0, page_dims[1], height_small)
    page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)
    page_xy = np.hstack((page_x_coords.flatten().reshape(-1, 1), page_y_coords.flatten().reshape(-1, 1))).astype(
        np.float32
    )

    image_points = norm2pix(img.shape, project_xy(page_xy, params), False)
    ix = image_points[:, 0, 0].reshape(page_x_coords.shape)
    iy = image_points[:, 0, 1].reshape(page_y_coords.shape)
    ix = cv2.resize(ix, (out_w, out_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    iy = cv2.resize(iy, (out_w, out_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return cv2.remap(img, ix, iy, cv2.INTER_CUBIC, None, cv2.BORDER_REPLICATE)


def pagedewarp_one(src: str, dst: str, no_binary: bool, extended: bool = False) -> tuple[str, bool, str]:
    """Обрабатывает один файл (для пула процессов, верхнеуровневая → picklable).

    ``extended=False`` (vanilla) — чистый page-dewarp с дефолтами библиотеки
    (поля 50/20, библиотечный размер с обрезкой, grayscale-выход).
    ``extended=True`` — наши правки: поля 0 (не режем по бокам), ширина ровно как у
    входа и ЦВЕТНОЙ ремап (при ``no_binary``).
    """
    try:
        from page_dewarp.image import WarpedImage
        from page_dewarp.options import Config

        img = cv2.imread(src)
        if img is None:
            return (src, False, "не прочитан")
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.png"
            cv2.imwrite(str(inp), img)

            cfg_kwargs = dict(OUTPUT_DIR=td, OUTPUT_FORMAT="png", NO_BINARY=int(no_binary), DEBUG_LEVEL=0)
            if extended:
                cfg_kwargs.update(
                    PAGE_MARGIN_X=PAGE_MARGIN_X,
                    PAGE_MARGIN_Y=PAGE_MARGIN_Y,
                    SCREEN_MAX_W=SCREEN_MAX_W,
                    SCREEN_MAX_H=SCREEN_MAX_H,
                )
            cfg = Config(**cfg_kwargs)

            wimg = WarpedImage(str(inp), config=cfg)
            if not wimg.written:
                return (src, False, "нет текстовых spans")

            if extended and no_binary:
                out = _remap_color(wimg.cv2_img, wimg.page_dims, wimg.params, cfg)
            else:
                out = cv2.imread(str(wimg.outfile))  # библиотечный выход (grayscale / binary)
            cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return (src, True, "")
    except Exception as e:  # noqa: BLE001 — ошибку отдаём наверх, пул не валим
        return (src, False, str(e))


class PageDewarpEngine(DewarpEngine):
    name = "pagedewarp"

    def load(self, device: str) -> None:
        # Нейросеть не нужна; держим флаг бинаризации (выкл — оставляем grayscale)
        self.no_binary = True

    def dewarp(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        from page_dewarp.image import WarpedImage  # noqa: PLC0415
        from page_dewarp.options import Config  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.jpg"
            cv2.imwrite(str(inp), img_bgr)
            cfg = Config(
                OUTPUT_DIR=td,
                OUTPUT_FORMAT="jpg",
                NO_BINARY=int(self.no_binary),  # 1 = без порога, сырой grayscale
                DEBUG_LEVEL=0,
            )
            wimg = WarpedImage(str(inp), config=cfg)
            if not wimg.written:
                # < 1 span — алгоритм не нашёл текстовых строк
                logger.warning("page-dewarp: не найдено текстовых spans, кадр пропущен")
                return None
            out = cv2.imread(str(wimg.outfile))
            return out
