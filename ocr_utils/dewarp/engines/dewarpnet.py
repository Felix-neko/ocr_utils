"""Движок DewarpNet (ICCV 2019) — перенос из корневого dewarp.py.

Две сети: WC-Net (3D-координаты поверхности) + BM-Net (backward mapping). Вход 256×256,
BM-Net работает на 128×128, итог — ``F.grid_sample`` на оригинале. Код — клон
cvlab-stonybrook/DewarpNet в third_party/DewarpNet, веса — dewarp_models/dewarpnet/*.pkl.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ocr_utils.dewarp.engines.base import DewarpEngine
from ocr_utils.dewarp.engines.download import MODELS_DIR, add_to_path, ensure_repo, forget_modules

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/cvlab-stonybrook/DewarpNet"
REPO_NAME = "DewarpNet"
REPO_MODULES = ["models", "utils"]

WC_NAME = "unetnc_doc3d_final.pkl"
BM_NAME = "dnetccnl_doc3d_final.pkl"


class DewarpNetEngine(DewarpEngine):
    name = "dewarpnet"

    def load(self, device: str) -> None:
        self.device = torch.device(device)
        repo = ensure_repo(REPO_URL, REPO_NAME)
        add_to_path(repo)

        weights_dir = MODELS_DIR / "dewarpnet"
        wc_path = weights_dir / WC_NAME
        bm_path = weights_dir / BM_NAME
        if not (wc_path.exists() and bm_path.exists()):
            raise FileNotFoundError(
                f"Нет весов DewarpNet в {weights_dir} (нужны {WC_NAME} и {BM_NAME})."
            )

        from models import get_model  # noqa: PLC0415
        from utils import convert_state_dict  # noqa: PLC0415

        def _load(model_path: Path, model_name: str, n_classes: int) -> torch.nn.Module:
            model = get_model(model_name, n_classes, in_channels=3)
            state = convert_state_dict(torch.load(model_path, map_location=self.device)["model_state"])
            model.load_state_dict(state)
            model.eval()
            return model.to(self.device)

        self.wc_model = _load(wc_path, wc_path.stem.split("_")[0], n_classes=3)
        self.bm_model = _load(bm_path, bm_path.stem.split("_")[0], n_classes=2)

        forget_modules(REPO_MODULES)

    def dewarp(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        img_256 = cv2.resize(rgb, (256, 256))
        img_256_bgr = img_256[:, :, ::-1].astype(np.float32) / 255.0  # RGB→BGR + норм
        tensor_wc = torch.from_numpy(img_256_bgr.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        htan = torch.nn.Hardtanh(0, 1.0)
        with torch.no_grad():
            wc_out = htan(self.wc_model(tensor_wc))
            bm_input = F.interpolate(wc_out, (128, 128))
            bm_out = self.bm_model(bm_input).detach().cpu()

        bm0 = cv2.blur(cv2.resize(bm_out[0, 0].numpy(), (w, h)), (3, 3))
        bm1 = cv2.blur(cv2.resize(bm_out[0, 1].numpy(), (w, h)), (3, 3))
        lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0).double()
        img_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).double() / 255.0

        out = F.grid_sample(input=img_tensor, grid=lbl, align_corners=True)
        res = out[0].numpy().transpose(1, 2, 0)
        return cv2.cvtColor((res * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
