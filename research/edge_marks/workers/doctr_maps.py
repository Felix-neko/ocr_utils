"""Воркер карт docTR DBNet по списку полос (питон окружения doctr, пакет ocr_utils не импортирует): вероятность текста в пикселях полосы, PNG 8 бит.

Аргументы: ``--jobs`` (JSON ``[{"key", "png"}]``), ``--out-dir`` (карты ``<key>.png``; готовые не пересчитываются).
Модель ``db_resnet50`` грузится один раз на прогон. В конце — строка журнала «карт N за S с».
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from doctr.models import detection_predictor


def seg_map(det, page: np.ndarray) -> np.ndarray:
    """Карта вероятности текста DBNet в пикселях полосы (0…1).

    Сеть считает на входе с сохранением пропорций и дополнением до квадрата по центру; дополнение снимается.

    Args:
        det: Детектор docTR.
        page: Полоса RGB.

    Returns:
        Карта ``(H, W)``.
    """
    h, w = page.shape[:2]
    with torch.inference_mode():
        _, maps = det([page], return_maps=True)
    seg = np.asarray(maps[0])
    seg = seg[..., 0] if seg.ndim == 3 else seg
    scale = max(h, w) / max(seg.shape)
    full = cv2.resize(seg, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    top = (full.shape[0] - h) // 2 if full.shape[0] > h else 0
    left = (full.shape[1] - w) // 2 if full.shape[1] > w else 0
    return np.clip(full[top : top + h, left : left + w], 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [job for job in jobs if not (args.out_dir / f"{job['key']}.png").exists()]
    if not todo:
        print("карт 0 за 0.0 с (все в кэше)", flush=True)
        return
    started = time.monotonic()
    det = detection_predictor("db_resnet50", pretrained=True, assume_straight_pages=True).cuda().eval()
    print(f"модель за {time.monotonic() - started:.1f} с", flush=True)
    tick = time.monotonic()
    for job in todo:
        page = cv2.cvtColor(cv2.imread(job["png"]), cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(args.out_dir / f"{job['key']}.png"), (seg_map(det, page) * 255).astype(np.uint8))
    print(f"карт {len(todo)} за {time.monotonic() - tick:.1f} с", flush=True)


if __name__ == "__main__":
    main()
