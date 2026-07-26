"""Движок DocScanner (IJCV 2025) — основной для выпрямления страниц.

DocScanner: Robust Document Image Rectification with Progressive Learning
(репозиторий fh2019ustc/DocScanner). Архитектура: U2NETP-сегментация документа +
прогрессивная (RAFT-подобная) сеть, предсказывающая обратную сетку (backward map).
Логика инференса повторяет ``inference.py`` из репозитория: вход 288×288 → bm в
[-1,1] → ``F.grid_sample`` на оригинале.

Веса (DocScanner-L.pth + seg.pth) тянутся из Google Drive в ``dewarp_models/docscanner``.
"""

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ocr_utils.legacy.dewarp.engines.base import DewarpEngine
from ocr_utils.legacy.dewarp.engines.download import MODELS_DIR, add_to_path, ensure_repo, forget_modules, gdrive_folder

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/fh2019ustc/DocScanner"
REPO_NAME = "DocScanner"
WEIGHTS_GDRIVE_FOLDER = "1W1_DJU8dfEh6FqDYqFQ7ypR38Z8c5r4D"
# Локальные модули репозитория, которые надо забыть после загрузки (общие имена)
REPO_MODULES = ["model", "seg", "extractor", "update"]


class DocScannerEngine(DewarpEngine):
    name = "docscanner"

    def load(self, device: str) -> None:
        self.device = device
        repo = ensure_repo(REPO_URL, REPO_NAME)
        add_to_path(repo)

        weights_dir = MODELS_DIR / "docscanner"
        seg_path = weights_dir / "seg.pth"
        rec_path = weights_dir / "DocScanner-L.pth"
        if not (seg_path.exists() and rec_path.exists()):
            gdrive_folder(WEIGHTS_GDRIVE_FOLDER, weights_dir)

        from model import DocScanner  # noqa: PLC0415 (импорт из клона репо)
        from seg import U2NETP  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415

        class Net(nn.Module):
            """Обёртка: U2NETP-маска документа → DocScanner-ректификация (как в inference.py)."""

            def __init__(self):
                super().__init__()
                self.msk = U2NETP(3, 1)
                self.bm = DocScanner()

            def forward(self, x):
                msk, *_ = self.msk(x)
                msk = (msk > 0.5).float()
                x = msk * x
                bm = self.bm(x, iters=12, test_mode=True)
                bm = (2 * (bm / 286.8) - 1) * 0.99  # нормировка в [-1,1] (как в оригинале)
                return bm

        net = Net().to(device)
        # seg.pth: ключи с префиксом 6 символов; DocScanner-L.pth: без префикса (как в inference.py)
        self._reload(net.msk, seg_path, strip=6, device=device)
        self._reload(net.bm, rec_path, strip=0, device=device)
        net.eval()
        self.net = net

        forget_modules(REPO_MODULES)

    @staticmethod
    def _reload(model: torch.nn.Module, path, strip: int, device: str) -> None:
        """Грузит веса со снятием префикса ``strip`` символов у ключей."""
        sd = torch.load(path, map_location=device)
        md = model.state_dict()
        filtered = {k[strip:]: v for k, v in sd.items() if k[strip:] in md}
        md.update(filtered)
        model.load_state_dict(md)

    def dewarp(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        im_ori = rgb.astype(np.float32) / 255.0

        im = cv2.resize(im_ori, (288, 288)).transpose(2, 0, 1)
        im_t = torch.from_numpy(im).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            bm = self.net(im_t).cpu()

        bm0 = cv2.blur(cv2.resize(bm[0, 0].numpy(), (w, h)), (3, 3))
        bm1 = cv2.blur(cv2.resize(bm[0, 1].numpy(), (w, h)), (3, 3))
        lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0).float()
        img_t = torch.from_numpy(im_ori.transpose(2, 0, 1)).unsqueeze(0).float()

        out = F.grid_sample(img_t, lbl, align_corners=True)
        res = (out[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
