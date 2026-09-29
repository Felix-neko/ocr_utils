"""Воркер карт CRAFT по списку полос (питон окружения craft, пакет ocr_utils не импортирует): карта «регион» в пикселях полосы, PNG 8 бит.

Аргументы: ``--jobs`` (JSON ``[{"key", "png"}]``), ``--out-dir`` (карты ``<key>.png``; готовые не пересчитываются),
``--fp16``. Модель грузится один раз на прогон. В конце — строка журнала «карт N за S с».
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
from craft_judge import cw, region_map  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Готовые карты не пересчитываются.
    todo = [job for job in jobs if not (args.out_dir / f"{job['key']}.png").exists()]
    if not todo:
        print("карт 0 за 0.0 с (все в кэше)", flush=True)
        return
    started = time.monotonic()
    device = torch.device("cuda")
    net = cw.load_model(
        str(Path(cw.DEFAULT_ROOT) / "src/craft"), str(Path(cw.DEFAULT_ROOT) / "models/craft/craft_mlt_25k.pth"), device
    )
    print(f"модель за {time.monotonic() - started:.1f} с", flush=True)
    tick = time.monotonic()
    for job in todo:
        gray = cv2.imread(job["png"], cv2.IMREAD_GRAYSCALE)
        region = region_map(net, gray, device, args.fp16)
        cv2.imwrite(str(args.out_dir / f"{job['key']}.png"), (region * 255).astype(np.uint8))
    print(f"карт {len(todo)} за {time.monotonic() - tick:.1f} с", flush=True)


if __name__ == "__main__":
    main()
