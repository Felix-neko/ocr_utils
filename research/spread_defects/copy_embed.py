"""Эмбеддинги DINOv2 по вырезкам текста каждой полосы — второй судья «ксерокопия или страница книги».

Ручные признаки (``photocopy``) держатся на цвете бумаги, а ксерокопия бывает и на пожелтевшей бумаге
(1964/12-copy IMG_0026 справа, IMG_0027 слева). Отличие таких копий — в фактуре: тонер, крап, рваный край
штриха. Поэтому вырезки берутся в родном разрешении (без уменьшения, фактура иначе теряется) из текстовых
плиток сетки ``copies`` по каждой стороне сгиба. Модель одна на прогон и считается на GPU в главном
процессе, воркеры пула только читают кадры и режут вырезки.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from research.spread_defects.frames import Frame, read_rgb
from research.spread_defects.motion import TILE

# Сторона вырезки в родном разрешении кадра, px (вход DINOv2 — 224).
CROP = 224
# Сколько вырезок на полосу (равномерно по текстовым плиткам полосы).
CROPS_PER_SIDE = 8
# Ширина зоны у сгиба, где вырезки не берутся (тень и изгиб листа), px полного кадра.
FOLD_MARGIN = 200
# Модель Hugging Face.
MODEL_NAME = "facebook/dinov2-small"


def side_crops(frame: Frame, grid_path: Path, fold_x: float) -> dict[str, np.ndarray]:
    """Вырезки текста каждой полосы кадра (воркер пула).

    Args:
        frame: Кадр.
        grid_path: npz сетки ``copies`` (нужна маска ``text`` текстовых плиток).
        fold_x: x сгиба посередине высоты, px полного кадра.

    Returns:
        Словарь сторона → uint8[n, CROP, CROP, 3] (n ≤ ``CROPS_PER_SIDE``); сторона без текстовых плиток
        не попадает в словарь.
    """
    text = np.load(grid_path)["text"]
    rgb = read_rgb(frame.path)
    centers_x = (np.arange(text.shape[1]) + 0.5) * TILE
    out = {}
    for side, cols in (("L", centers_x < fold_x - FOLD_MARGIN), ("R", centers_x > fold_x + FOLD_MARGIN)):
        rows, columns = np.nonzero(text & cols[None, :])
        if len(rows) == 0:
            continue
        # Равномерная выборка плиток по порядку обхода (сверху вниз), без случайности — повторяемо.
        pick = np.linspace(0, len(rows) - 1, min(CROPS_PER_SIDE, len(rows))).round().astype(int)
        crops = []
        for k in pick:
            # Вырезка CROP×CROP в центре плитки.
            y = rows[k] * TILE + (TILE - CROP) // 2
            x = columns[k] * TILE + (TILE - CROP) // 2
            crops.append(rgb[y : y + CROP, x : x + CROP])
        out[side] = np.stack(crops)
    return out


class Embedder:
    """DINOv2 на GPU: средний эмбеддинг (CLS + среднее по патчам) по набору вырезок полосы."""

    def __init__(self, device: str = "cuda") -> None:
        """Грузит модель один раз на прогон.

        Args:
            device: Устройство torch.
        """
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval().half()
        # Нормировка ImageNet, как в препроцессоре DINOv2.
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        """Эмбеддинг полосы.

        Args:
            crops: uint8[n, CROP, CROP, 3] — вырезки одной полосы.

        Returns:
            float32[768] — среднее по вырезкам от конкатенации CLS-токена и среднего патч-токена.
        """
        torch = self.torch
        with torch.inference_mode():
            batch = torch.from_numpy(crops).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            batch = ((batch - self.mean) / self.std).half()
            hidden = self.model(pixel_values=batch).last_hidden_state
            features = torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=1)
        return features.float().mean(dim=0).cpu().numpy()
