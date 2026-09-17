"""Движок DocTr / GeoTr (ACM MM 2021) — перенос из корневого dewarp.py.

U2NETP-сегментация документа + трансформер GeoTr (backward flow). Вход 288×288 →
bm в [-1,1] → ``F.grid_sample``. Код — клон fh2019ustc/DocTr в third_party/DocTrFH,
веса — dewarp_models/doctr/{seg.pth,geotr.pth}.
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

REPO_URL = "https://github.com/fh2019ustc/DocTr"
REPO_NAME = "DocTrFH"
# Общие имена модулей разных репо — забываем после загрузки
REPO_MODULES = ["GeoTr", "seg", "extractor", "update", "position_encoding"]


class DocTrEngine(DewarpEngine):
    name = "doctr"

    def load(self, device: str) -> None:
        self.device = device
        repo = ensure_repo(REPO_URL, REPO_NAME)
        add_to_path(repo)

        weights_dir = MODELS_DIR / "doctr"
        seg_path = weights_dir / "seg.pth"
        geotr_path = weights_dir / "geotr.pth"
        if not (seg_path.exists() and geotr_path.exists()):
            raise FileNotFoundError(
                f"Нет весов DocTr в {weights_dir} (нужны seg.pth и geotr.pth). "
                "Скачайте из HuggingFace Space HaoFeng2019/DocTr (model_pretrained/)."
            )

        from GeoTr import GeoTr  # noqa: PLC0415
        from seg import U2NETP  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415

        class GeoTr_Seg(nn.Module):
            """Сначала маска документа (U2Net), затем GeoTr (backward flow)."""

            def __init__(self):
                super().__init__()
                self.msk = U2NETP(3, 1)
                self.GeoTr = GeoTr(num_attn_layers=6)

            def forward(self, x):
                msk, *_ = self.msk(x)
                msk = (msk > 0.5).float()
                x = msk * x
                bm = self.GeoTr(x)
                bm = (2 * (bm / 286.8) - 1) * 0.99
                return bm

        model = GeoTr_Seg().to(device)
        self._reload(model.msk, seg_path, "net.", device)
        self._reload(model.GeoTr, geotr_path, "module.", device)
        model.eval()
        self.model = model

        forget_modules(REPO_MODULES)

    @staticmethod
    def _reload(model: torch.nn.Module, path, prefix: str, device: str) -> None:
        sd = torch.load(path, map_location=device)
        md = model.state_dict()
        n = len(prefix)
        filtered = {k[n:]: v for k, v in sd.items() if k[n:] in md}
        md.update(filtered)
        model.load_state_dict(md)

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
