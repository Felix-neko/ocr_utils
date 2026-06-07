"""Движок DocTr++ (TMM 2023) — «нерестриктед» ректификация документов.

Репозиторий fh2019ustc/DocTr-Plus. Только геометрия (GeoTr, без маски документа):
вход 288×288 → backward map → ``F.grid_sample``. Хорош для частичных/двухстраничных
разворотов. Веса ``DocTrP.pth`` НЕ входят в репозиторий — положите их вручную в
``dewarp_models/doctr_plus/DocTrP.pth`` (ссылка в README репозитория DocTr-Plus).
"""

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ocr_utils.dewarp.engines.base import DewarpEngine
from ocr_utils.dewarp.engines.download import MODELS_DIR, add_to_path, ensure_repo, forget_modules

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/fh2019ustc/DocTr-Plus"
REPO_NAME = "DocTr-Plus"
REPO_MODULES = ["GeoTr", "extractor", "update", "position_encoding"]


class DocTrPlusEngine(DewarpEngine):
    name = "doctr_plus"

    def load(self, device: str) -> None:
        self.device = device
        repo = ensure_repo(REPO_URL, REPO_NAME)
        add_to_path(repo)

        weights = MODELS_DIR / "doctr_plus" / "DocTrP.pth"
        if not weights.exists():
            raise FileNotFoundError(
                f"Нет весов DocTr++: {weights}. Скачайте DocTrP.pth (см. README "
                "fh2019ustc/DocTr-Plus) и положите в этот путь."
            )

        from GeoTr import GeoTr  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415

        class GeoTrP(nn.Module):
            def __init__(self):
                super().__init__()
                self.GeoTr = GeoTr()

            def forward(self, x):
                bm = self.GeoTr(x)
                bm = (2 * (bm / 286.8) - 1) * 0.99
                return bm

        model = GeoTrP().to(device)
        sd = torch.load(weights, map_location=device)
        md = model.GeoTr.state_dict()
        filtered = {k[7:]: v for k, v in sd.items() if k[7:] in md}  # снять префикс 'module.'
        md.update(filtered)
        model.GeoTr.load_state_dict(md)
        model.eval()
        self.model = model

        forget_modules(REPO_MODULES)

    def dewarp(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        im_ori = rgb.astype(np.float32) / 255.0

        im = cv2.resize(im_ori, (288, 288)).transpose(2, 0, 1)
        im_t = torch.from_numpy(im).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            bm = self.model(im_t).cpu()

        bm0 = cv2.blur(cv2.resize(bm[0, 0].numpy(), (w, h)), (3, 3))
        bm1 = cv2.blur(cv2.resize(bm[0, 1].numpy(), (w, h)), (3, 3))
        lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0).float()
        img_t = torch.from_numpy(im_ori.transpose(2, 0, 1)).unsqueeze(0).float()

        out = F.grid_sample(img_t, lbl, align_corners=True)
        res = (out[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
