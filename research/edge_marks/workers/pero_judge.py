"""Воркер-судья pero (запускается питоном окружения pero, пакет ocr_utils не импортирует).

Два судьи в одном прогоне:

* **ParseNet (сеть вёрстки pero), полоса (c)** — карты страницы ``get_maps_with_optimal_resolution``:
  вероятность базовой линии (канал 2) в столбцах кандидата у его строки. Настоящий знак в конце строки
  продолжает базовую линию, сор за её концом — нет. Выход ``kind=score``: оценка «это символ» 0…1.
* **OCR pero (VGG-LSTM CTC, с кириллицей), строка (b)** — текст строки с кандидатом и с закрашенным
  кандидатом (``kind=diff``) и вероятность не-пробела CTC в столбцах кандидата (``kind=score``).

Аргументы: ``--cand-dir`` (``candidates.json``, ``pages/``, ``b/``, ``b_erased/``), ``--out`` (JSONL), ``--model-dir``.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pero_ocr.layout_engines.torch_parsenet import TorchParseNet
from pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR

MODEL_DIR = Path("/mnt/hotstore/scan_processing/mts/curved_layout_engines/pero_model/pero_eu_cz_print_newspapers_2022-09-26")
# Базовая линия ниже середины строки на столько x-высот; окно поиска по высоте ± столько x-высот.
BASELINE_BELOW_XH, BASELINE_WINDOW_XH = 0.5, 0.6


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    started = time.monotonic()
    device = torch.device("cuda")
    parsenet = TorchParseNet(
        str(args.model_dir / "ParseNet_296000.pt"), device, downsample=5, max_mp=5, detection_threshold=0.2
    )
    ocr = PytorchEngineLineOCR(str(args.model_dir / "ocr_engine.json"), device)
    candidates = json.loads((args.cand_dir / "candidates.json").read_text())
    handle = args.out.open("w", encoding="utf-8")
    handle.write(json.dumps({"meta": {"load_seconds": time.monotonic() - started, "device": "cuda"}}) + "\n")
    by_page = {}
    for c in candidates:
        by_page.setdefault(c["key"], []).append(c)
    for key, items in by_page.items():
        tick = time.monotonic()
        page = cv2.imread(str(args.cand_dir / "pages" / f"{key}.png"), cv2.IMREAD_COLOR)
        maps, ds = parsenet.get_maps_with_optimal_resolution(page)
        page_seconds = time.monotonic() - tick
        baseline = maps[:, :, 2]
        for c in items:
            x0, y0, x1, y1 = c["box"]
            base_y = c["row_y"] + BASELINE_BELOW_XH * c["x_h"]
            ry0 = int((base_y - BASELINE_WINDOW_XH * c["x_h"]) / ds)
            ry1 = int((base_y + BASELINE_WINDOW_XH * c["x_h"]) / ds) + 1
            rx0, rx1 = int(x0 / ds), int(np.ceil(x1 / ds)) + 1
            window = baseline[max(0, ry0) : ry1, max(0, rx0) : rx1]
            score = float(np.clip(window.max(), 0, 1)) if window.size else 0.0
            handle.write(json.dumps({"id": c["id"], "scale": "c_parsenet", "kind": "score", "score": score,
                                     "seconds": page_seconds / len(items)}) + "\n")  # fmt: skip
            # OCR строки: высота вырезки → высота строки сети.
            tick = time.monotonic()
            lines, scales = [], []
            for folder in ("b", "b_erased"):
                image = cv2.imread(str(args.cand_dir / folder / f"{c['id']}.png"), cv2.IMREAD_COLOR)
                scale = ocr.line_px_height / image.shape[0]
                lines.append(cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
                scales.append(scale)
            texts, logits, coords = ocr.process_lines(lines, sparse_logits=False)
            seconds = time.monotonic() - tick
            handle.write(json.dumps({"id": c["id"], "scale": "b_pero_ocr", "kind": "diff", "with": texts[0],
                                     "erased": texts[1], "seconds": seconds}, ensure_ascii=False) + "\n")  # fmt: skip
            # Вероятность знака в столбцах кандидата: 1 − P(пусто) по кадрам CTC (шаг сети — 4 пикселя строки).
            probs = softmax(np.asarray(logits[0]))
            start = coords[0][0] if coords[0][0] is not None else 0
            bx0 = (x0 - c["line_box"][0]) * scales[0] / ocr.net_subsampling + start
            bx1 = (x1 - c["line_box"][0]) * scales[0] / ocr.net_subsampling + start
            frames = probs[max(0, int(bx0)) : int(np.ceil(bx1)) + 1]
            # Пробел (символ 1 словаря) знаком не считается.
            char = float((1.0 - frames[:, -1] - frames[:, 1]).max()) if len(frames) else 0.0
            handle.write(json.dumps({"id": c["id"], "scale": "b_pero_ctc", "kind": "score", "score": char,
                                     "seconds": seconds}) + "\n")  # fmt: skip
        handle.flush()
    handle.close()


if __name__ == "__main__":
    main()
