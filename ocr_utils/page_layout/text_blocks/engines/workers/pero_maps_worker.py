"""Воркер карт pero ParseNet для защиты сторон блоков (питон окружения pero): вероятность базовой линии по списку полос, PNG 8 бит в пикселях полосы.

Аргументы: ``--jobs`` (JSON ``[{"key", "png"}]``), ``--out-dir`` (карты ``<key>.png``; готовые не пересчитываются),
``--model-dir`` (папка модели pero с ``ParseNet_296000.pt``). Модель грузится один раз на прогон; в конце — строка
«карт N за S с».
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pero_ocr.layout_engines.torch_parsenet import TorchParseNet

# Канал карт ParseNet с вероятностью базовой линии.
BASELINE_CHANNEL = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [job for job in jobs if not (args.out_dir / f"{job['key']}.png").exists()]
    if not todo:
        print("карт 0 за 0.0 с (все в кэше)", flush=True)
        return
    parsenet = TorchParseNet(str(args.model_dir / "ParseNet_296000.pt"), torch.device("cuda"), downsample=5, max_mp=5,
                             detection_threshold=0.2)  # fmt: skip
    started = time.monotonic()
    for job in todo:
        page = cv2.imread(job["png"], cv2.IMREAD_COLOR)
        maps, ds = parsenet.get_maps_with_optimal_resolution(page)
        # Карта в пониженном разрешении ds — обратно к размеру полосы.
        full = cv2.resize(np.clip(maps[:, :, BASELINE_CHANNEL], 0, 1), None, fx=ds, fy=ds, interpolation=cv2.INTER_LINEAR)
        out = np.zeros(page.shape[:2], dtype=np.float32)
        h, w = min(out.shape[0], full.shape[0]), min(out.shape[1], full.shape[1])
        out[:h, :w] = full[:h, :w]
        cv2.imwrite(str(args.out_dir / f"{job['key']}.png"), (out * 255).astype(np.uint8))
    print(f"карт {len(todo)} за {time.monotonic() - started:.1f} с", flush=True)


if __name__ == "__main__":
    main()
