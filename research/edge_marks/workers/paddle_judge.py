"""Воркер-судья PaddleOCR 3 (питон окружения paddle, пакет ocr_utils не импортирует).

* Полоса (c): детектор PP-OCRv5_server_det — рамки строк/слов (``kind=boxes``).
* Строка (b): распознаватель восточнославянский ``eslav_PP-OCRv5_mobile_rec`` — текст строки с кандидатом и
  без (``kind=diff``).

Аргументы: ``--cand-dir``, ``--out`` (JSONL), ``--device`` (``gpu`` | ``cpu``).
"""

import argparse
import json
import os
import time
from pathlib import Path

# Видеопамять — по мере надобности: по умолчанию paddle сразу занимает почти всю свободную, и замер врёт.
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", "/mnt/hotstore/scan_processing/mts_markup/line_axis_engines/models/paddle")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from paddleocr import TextDetection, TextRecognition  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    args = parser.parse_args()
    started = time.monotonic()
    det = TextDetection(model_name="PP-OCRv5_server_det", device=args.device, limit_side_len=4000, limit_type="max")
    rec = TextRecognition(model_name="eslav_PP-OCRv5_mobile_rec", device=args.device)
    candidates = json.loads((args.cand_dir / "candidates.json").read_text())
    handle = args.out.open("w", encoding="utf-8")
    handle.write(json.dumps({"meta": {"load_seconds": time.monotonic() - started, "device": args.device}}) + "\n")
    by_page = {}
    for c in candidates:
        by_page.setdefault(c["key"], []).append(c)
    for key, items in by_page.items():
        tick = time.monotonic()
        page = cv2.imread(str(args.cand_dir / "pages" / f"{key}.png"))
        result = det.predict(page)[0]
        boxes = []
        for poly in result["dt_polys"]:
            poly = np.asarray(poly, dtype=np.float64)
            boxes.append(
                [float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())]
            )
        page_seconds = time.monotonic() - tick
        for c in items:
            handle.write(
                json.dumps(
                    {"id": c["id"], "scale": "c", "kind": "boxes", "boxes": boxes, "seconds": page_seconds / len(items)}
                )
                + "\n"
            )
            tick = time.monotonic()
            texts = [
                rec.predict(cv2.imread(str(args.cand_dir / f / f"{c['id']}.png")))[0]["rec_text"]
                for f in ("b", "b_erased")
            ]
            handle.write(json.dumps({"id": c["id"], "scale": "b", "kind": "diff", "with": texts[0], "erased": texts[1],
                                     "seconds": time.monotonic() - tick}, ensure_ascii=False) + "\n")  # fmt: skip
        handle.flush()
    handle.close()


if __name__ == "__main__":
    main()
