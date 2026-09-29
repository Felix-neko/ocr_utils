"""Воркер карт CRAFT для защиты сторон блоков (питон окружения craft): карта «регион» (центры символов) по списку полос, PNG 8 бит в пикселях полосы.

Аргументы: ``--jobs`` (JSON ``[{"key", "png"}]``), ``--out-dir`` (карты ``<key>.png``; готовые не пересчитываются),
``--fp16``. Модель грузится один раз на прогон; в конце — строка «карт N за S с».
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import craft_worker as cw  # noqa: E402

# Предел длинной стороны входа сети: полоса 300 dpi (до ~3500 px) идёт без уменьшения.
CANVAS_SIZE = 3840


def region_map(net, gray: np.ndarray, device, fp16: bool) -> np.ndarray:
    """Карта «регион» CRAFT (0…1) в пикселях поданной полосы.

    Args:
        net: Модель CRAFT.
        gray: Полоса в оттенках серого.
        device: Устройство.
        fp16: Половинная точность.

    Returns:
        Карта ``gray.shape``.
    """
    image, ratio = cw.prepare_input(gray, CANVAS_SIZE, 1.0)
    score = cw.run_network(net, image, device, fp16)
    # Карта сети — в половинном разрешении входа; обратно к размеру полосы.
    full = cv2.resize(score[:, :, 0], None, fx=2.0 / ratio, fy=2.0 / ratio, interpolation=cv2.INTER_LINEAR)
    out = np.zeros(gray.shape, dtype=np.float32)
    h, w = min(out.shape[0], full.shape[0]), min(out.shape[1], full.shape[1])
    out[:h, :w] = full[:h, :w]
    return np.clip(out, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [job for job in jobs if not (args.out_dir / f"{job['key']}.png").exists()]
    if not todo:
        print("карт 0 за 0.0 с (все в кэше)", flush=True)
        return
    device = torch.device("cuda")
    root = Path(cw.DEFAULT_ROOT)
    net = cw.load_model(str(root / "src/craft"), str(root / "models/craft/craft_mlt_25k.pth"), device)
    started = time.monotonic()
    for job in todo:
        gray = cv2.imread(job["png"], cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(args.out_dir / f"{job['key']}.png"), (region_map(net, gray, device, args.fp16) * 255).astype(np.uint8))
    print(f"карт {len(todo)} за {time.monotonic() - started:.1f} с", flush=True)


if __name__ == "__main__":
    main()
