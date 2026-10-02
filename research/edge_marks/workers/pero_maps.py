"""Воркер карт pero ParseNet по списку полос (питон окружения pero, пакет ocr_utils не импортирует): вероятность базовой линии в пикселях полосы, PNG 8 бит.

Аргументы: ``--jobs`` (JSON ``[{"key", "png"}]``), ``--out-dir`` (карты ``<key>.png``; готовые не пересчитываются),
``--model-dir``. Модель грузится один раз на прогон. В конце — строка журнала «карт N за S с».
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pero_ocr.layout_engines.torch_parsenet import TorchParseNet

MODEL_DIR = Path("/mnt/hotstore/scan_processing/mts/curved_layout_engines/pero_model/pero_eu_cz_print_newspapers_2022-09-26")
# Канал карт ParseNet с вероятностью базовой линии.
BASELINE_CHANNEL = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Готовые карты не пересчитываются.
    todo = [job for job in jobs if not (args.out_dir / f"{job['key']}.png").exists()]
    if not todo:
        print("карт 0 за 0.0 с (все в кэше)", flush=True)
        return
    started = time.monotonic()
    parsenet = TorchParseNet(str(args.model_dir / "ParseNet_296000.pt"), torch.device("cuda"), downsample=5, max_mp=5,
                             detection_threshold=0.2)  # fmt: skip
    print(f"модель за {time.monotonic() - started:.1f} с", flush=True)
    tick = time.monotonic()
    for job in todo:
        page = cv2.imread(job["png"], cv2.IMREAD_COLOR)
        maps, ds = parsenet.get_maps_with_optimal_resolution(page)
        # Карта в пониженном разрешении ds — обратно к размеру полосы.
        baseline = np.clip(maps[:, :, BASELINE_CHANNEL], 0, 1)
        full = cv2.resize(baseline, None, fx=ds, fy=ds, interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros(page.shape[:2], dtype=np.float32)
        h, w = min(canvas.shape[0], full.shape[0]), min(canvas.shape[1], full.shape[1])
        canvas[:h, :w] = full[:h, :w]
        cv2.imwrite(str(args.out_dir / f"{job['key']}.png"), (canvas * 255).astype(np.uint8))
    print(f"карт {len(todo)} за {time.monotonic() - tick:.1f} с", flush=True)


if __name__ == "__main__":
    main()
