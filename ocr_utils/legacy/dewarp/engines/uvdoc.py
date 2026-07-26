"""Движок UVDoc (SIGGRAPH Asia 2023) — neural grid-based unwarping.

Репозиторий tanguymagne/UVDoc. Сеть UVDocnet предсказывает 2D-сетку точек, по которой
``bilinear_unwarping`` распрямляет изображение. Веса ``model/best_model.pkl`` лежат
прямо в репозитории (скачиваются вместе с клоном).
"""

import logging
from typing import Optional

import cv2
import numpy as np
import torch

from ocr_utils.legacy.dewarp.engines.base import DewarpEngine
from ocr_utils.legacy.dewarp.engines.download import add_to_path, ensure_repo, forget_modules

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/tanguymagne/UVDoc"
REPO_NAME = "UVDoc"
REPO_MODULES = ["utils", "model"]


class UVDocEngine(DewarpEngine):
    name = "uvdoc"

    def load(self, device: str) -> None:
        self.device = device
        repo = ensure_repo(REPO_URL, REPO_NAME)
        add_to_path(repo)

        ckpt = repo / "model" / "best_model.pkl"
        if not ckpt.exists():
            raise FileNotFoundError(f"Нет весов UVDoc: {ckpt}")

        from utils import IMG_SIZE, bilinear_unwarping, load_model  # noqa: PLC0415

        self.img_size = tuple(IMG_SIZE)  # (W, H) для входа сети
        self._unwarp = bilinear_unwarping
        model = load_model(str(ckpt)).to(device)
        model.eval()
        self.model = model

        forget_modules(REPO_MODULES)

    def dewarp(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = torch.from_numpy(cv2.resize(rgb, self.img_size).transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            point_positions2D, _ = self.model(inp)
            size = rgb.shape[:2][::-1]  # (W, H) оригинала
            unwarped = self._unwarp(
                warped_img=torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device),
                point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
                img_size=tuple(size),
            )
        res = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
