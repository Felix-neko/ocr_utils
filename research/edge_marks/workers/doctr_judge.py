"""Воркер-судья docTR (питон окружения doctr, пакет ocr_utils не импортирует).

* Полоса (c): детектор слов DBNet (``db_resnet50``) — карта вероятности текста в рамке кандидата (``kind=score``)
  и рамки слов (``kind=boxes``).
* Строка (b): распознаватель PARSeq многоязычный (с кириллицей, ``Felix92/doctr-torch-parseq-multilingual-v1``) —
  текст строки с кандидатом и без (``kind=diff``).

Аргументы: ``--cand-dir``, ``--out`` (JSONL).
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from doctr.models import detection_predictor, from_hub, recognition_predictor

RECO_HUB = "Felix92/doctr-torch-parseq-multilingual-v1"
# Распознаватель слов: строку в 60 мм он сплющивает в 128 пикселей, поэтому подаётся хвост строки у кандидата —
# столько мм внутрь текста от кандидата и столько наружу. Кайма вырезок строки (``candidates.MARGIN_PX``).
TAIL_IN_MM, TAIL_OUT_MM, MARGIN_PX = 12.0, 2.0, 16
PX_PER_MM = 300 / 25.4


def tail(image, c):
    """Хвост вырезки строки у кандидата (последнее слово с кандидатом)."""
    lb0 = c["line_box"][0]
    x0, x1 = c["box"][0], c["box"][2]
    if c["side"] == "right":
        a, b = x0 - TAIL_IN_MM * PX_PER_MM, x1 + TAIL_OUT_MM * PX_PER_MM
    else:
        a, b = x0 - TAIL_OUT_MM * PX_PER_MM, x1 + TAIL_IN_MM * PX_PER_MM
    a, b = int(max(0, a - lb0 + MARGIN_PX)), int(min(image.shape[1], b - lb0 + MARGIN_PX))
    return image[:, a:b]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    det = detection_predictor("db_resnet50", pretrained=True, assume_straight_pages=True).cuda().eval()
    reco = recognition_predictor(from_hub(RECO_HUB)).cuda().eval()
    maps_dir = args.out.parent / f"{args.out.stem}_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    candidates = json.loads((args.cand_dir / "candidates.json").read_text())
    handle = args.out.open("w", encoding="utf-8")
    handle.write(json.dumps({"meta": {"load_seconds": time.monotonic() - started, "device": "cuda"}}) + "\n")
    by_page = {}
    for c in candidates:
        by_page.setdefault(c["key"], []).append(c)
    for key, items in by_page.items():
        tick = time.monotonic()
        page = cv2.cvtColor(cv2.imread(str(args.cand_dir / "pages" / f"{key}.png")), cv2.COLOR_BGR2RGB)
        h, w = page.shape[:2]
        with torch.inference_mode():
            preds, seg_maps = det([page], return_maps=True)
        # Карта вероятности текста — в размере входа сети с сохранением пропорций и дополнением: снимаем дополнение.
        seg = np.asarray(seg_maps[0])[..., 0] if np.asarray(seg_maps[0]).ndim == 3 else np.asarray(seg_maps[0])
        scale = max(h, w) / max(seg.shape)
        seg_full = cv2.resize(seg, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        top = (seg_full.shape[0] - h) // 2 if seg_full.shape[0] > h else 0
        left = (seg_full.shape[1] - w) // 2 if seg_full.shape[1] > w else 0
        seg_full = seg_full[top : top + h, left : left + w]
        cv2.imwrite(str(maps_dir / f"{key}.png"), (np.clip(seg_full, 0, 1) * 255).astype(np.uint8))
        words = preds[0]["words"] if isinstance(preds[0], dict) else preds[0]
        boxes = [
            [float(b[0] * w), float(b[1] * h), float(b[2] * w), float(b[3] * h), float(b[4])] for b in np.asarray(words)
        ]
        page_seconds = time.monotonic() - tick
        for c in items:
            x0, y0, x1, y1 = c["box"]
            part = seg_full[y0:y1, x0:x1]
            score = float(part.max()) if part.size else 0.0
            share = page_seconds / len(items)
            handle.write(
                json.dumps({"id": c["id"], "scale": "c_map", "kind": "score", "score": score, "seconds": share}) + "\n"
            )
            handle.write(
                json.dumps({"id": c["id"], "scale": "c_words", "kind": "boxes", "boxes": boxes, "seconds": share})
                + "\n"
            )
            tick = time.monotonic()
            crops = [
                cv2.cvtColor(cv2.imread(str(args.cand_dir / f / f"{c['id']}.png")), cv2.COLOR_BGR2RGB)
                for f in ("b", "b_erased")
            ]
            with torch.inference_mode():
                out = reco(crops)
            texts = [item[0] for item in out]
            handle.write(json.dumps({"id": c["id"], "scale": "b", "kind": "diff", "with": texts[0], "erased": texts[1],
                                     "seconds": time.monotonic() - tick}, ensure_ascii=False) + "\n")  # fmt: skip
        handle.flush()
    handle.close()


if __name__ == "__main__":
    main()
