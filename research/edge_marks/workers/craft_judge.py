"""Воркер-судья CRAFT (питон окружения craft, пакет ocr_utils не импортирует): карта центров символов («регион») в рамке кандидата.

Масштабы: полоса целиком (c) и вырезка кандидата (a). Оценка «это символ» — максимум карты региона в рамке
кандидата (0…1): у буквы и знака препинания центр символа даёт пик, у сора и штриха пометки — нет или слабый.
Карты полос сохраняются (``<out>_maps/<полоса>.png``, 8 бит) для оверлеев.

Аргументы: ``--cand-dir``, ``--out`` (JSONL), ``--fp16``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

WORKERS = Path("/home/felix/Projects/ocr_utils/ocr_utils/page_layout/text_blocks/engines/workers")
sys.path.insert(0, str(WORKERS))
import craft_worker as cw  # noqa: E402

CROP_A_XH, UPSCALE, MARGIN_PX = 1.5, 2, 16


def region_map(net, gray, device, fp16, canvas_size=3840, mag_ratio=1.0):
    """Карта региона CRAFT в пикселях поданного изображения (0…1)."""
    image, ratio = cw.prepare_input(gray, canvas_size, mag_ratio)
    score = cw.run_network(net, image, device, fp16)
    region = score[:, :, 0]
    # Карта — в половинном разрешении входа сети; обратно к исходнику.
    full = cv2.resize(region, None, fx=2.0 / ratio, fy=2.0 / ratio, interpolation=cv2.INTER_LINEAR)
    return np.clip(full[: gray.shape[0], : gray.shape[1]], 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    device = torch.device("cuda")
    net = cw.load_model(
        str(Path(cw.DEFAULT_ROOT) / "src/craft"), str(Path(cw.DEFAULT_ROOT) / "models/craft/craft_mlt_25k.pth"), device
    )
    maps_dir = args.out.parent / f"{args.out.stem}_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    candidates = json.loads((args.cand_dir / "candidates.json").read_text())
    handle = args.out.open("w", encoding="utf-8")
    handle.write(
        json.dumps({"meta": {"load_seconds": time.monotonic() - started, "device": "cuda", "fp16": args.fp16}}) + "\n"
    )
    by_page = {}
    for c in candidates:
        by_page.setdefault(c["key"], []).append(c)
    for key, items in by_page.items():
        tick = time.monotonic()
        gray = cv2.imread(str(args.cand_dir / "pages" / f"{key}.png"), cv2.IMREAD_GRAYSCALE)
        region = region_map(net, gray, device, args.fp16)
        page_seconds = time.monotonic() - tick
        cv2.imwrite(str(maps_dir / f"{key}.png"), (region * 255).astype(np.uint8))
        for c in items:
            x0, y0, x1, y1 = c["box"]
            score = float(region[y0:y1, x0:x1].max()) if y1 > y0 and x1 > x0 else 0.0
            handle.write(
                json.dumps(
                    {"id": c["id"], "scale": "c", "kind": "score", "score": score, "seconds": page_seconds / len(items)}
                )
                + "\n"
            )
            # Вырезка (a): тот же вопрос на увеличенной вырезке с полем.
            tick = time.monotonic()
            crop = cv2.imread(str(args.cand_dir / "a" / f"{c['id']}.png"), cv2.IMREAD_GRAYSCALE)
            crop_region = region_map(net, crop, device, args.fp16, mag_ratio=1.0)
            pad = int(CROP_A_XH * c["x_h"])
            cx0, cy0 = pad * UPSCALE + MARGIN_PX, pad * UPSCALE + MARGIN_PX
            cx1, cy1 = cx0 + (x1 - x0) * UPSCALE, cy0 + (y1 - y0) * UPSCALE
            part = crop_region[cy0:cy1, cx0:cx1]
            score = float(part.max()) if part.size else 0.0
            handle.write(
                json.dumps(
                    {"id": c["id"], "scale": "a", "kind": "score", "score": score, "seconds": time.monotonic() - tick}
                )
                + "\n"
            )
        handle.flush()
    handle.close()


if __name__ == "__main__":
    main()
