"""Доля высокочастотной энергии HF/MID — спектральная мера резкости.

ИДЕЯ. Расфокус физически убивает высокие пространственные частоты, оставляя средние.
Поэтому для тайла считаем

    HF_ratio = E(высокие частоты) / E(средние частоты)

Средние частоты — это «рисунок» полосы (шаг строк, ширина букв), он переживает лёгкий
расфокус почти без потерь и служит естественной нормировкой: величина не зависит ни от
количества краски, ни от контраста. Высокие — тонкие штрихи и типографский растр, они
гибнут первыми.

Метод уже валидирован в этом репозитории на зональных расфокусах
(``ocr_utils/legacy/defocus_detection/fft_hf.py``, ``detect_zonal_defocus.py``); здесь он
переиспользуется как одна из метрик общего ранжирования кадров.
"""

import numpy as np

from ocr_utils.defocus_detection.metrics.base import Algorithm
from ocr_utils.defocus_detection.tiles import Grid

# Границы колец в долях частоты Найквиста. Подобраны в focus_detection_report.md
# на превью 4416×2944.
MID_LO, MID_HI = 0.10, 0.35
HI_LO, HI_HI = 0.35, 0.85


def _tile_sharpness(gray: np.ndarray, grid: Grid) -> np.ndarray:
    """Карта HF/MID по тайлам (больше = резче).

    Args:
        gray: Полутоновый кадр.
        grid: Сетка тайлов.

    Returns:
        Массив (ny, nx) с отношением ВЧ- к СЧ-энергии.
    """
    out = np.full((grid.ny, grid.nx), np.nan)
    for iy in range(grid.ny):
        for ix in range(grid.nx):
            y1, y2, x1, x2 = grid.bounds(iy, ix)
            tile = gray[y1:y2, x1:x2].astype(np.float64)
            h, w = tile.shape
            if h < 8 or w < 8:
                continue
            tile = tile - tile.mean()  # убираем постоянную составляющую (яркость фона)
            # Окно Хэннинга гасит разрыв на границах тайла — иначе шов даёт ложную ВЧ-энергию.
            tile = tile * np.hanning(h)[:, None] * np.hanning(w)[None, :]
            power = np.abs(np.fft.rfft2(tile)) ** 2
            fy = np.fft.fftfreq(h)[:, None]
            fx = np.fft.rfftfreq(w)[None, :]
            radius = np.sqrt(fy**2 + fx**2) / 0.5  # 1.0 = частота Найквиста
            mid = power[(radius > MID_LO) & (radius <= MID_HI)].sum()
            high = power[(radius > HI_LO) & (radius <= HI_HI)].sum()
            out[iy, ix] = high / (mid + 1e-9)
    return out


ALGORITHM = Algorithm(
    name="hf_mid",
    summary="доля ВЧ-энергии спектра (HF/MID): нормирована на «рисунок» полосы, валидирована на зональных расфокусах",
    tile_sharpness=_tile_sharpness,
    unit="HF/MID",
)
