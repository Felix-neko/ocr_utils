"""DeepSeek-OCR-2 на transformers (без vLLM) по списку вырезок: сырой вывод с grounding-рамками в JSONL.

Запуск в окружении ``research/external_ocr_models/local/deepseek_ocr2``:
``uv run --project research/external_ocr_models/local/deepseek_ocr2 python research/bound_volumes/workers/deepseek_hf.py
--jobs jobs.jsonl --out out.jsonl``. Нужен, когда vLLM не стартует (NVML не инициализируется после обновления драйвера).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path

MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"
PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "


def main() -> int:
    """Прогон по заданиям ``{"id", "crop"}``; дописывает ``{"id", "raw", "size", "seconds"}`` в выходной JSONL.

    Returns:
        Код возврата.
    """
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 (transformers) по вырезкам")
    parser.add_argument("--jobs", type=Path, required=True, help='JSONL: {"id", "crop" — путь к картинке}')
    parser.add_argument("--out", type=Path, required=True, help="Выходной JSONL (дописывается, готовые id пропускаются)")
    parser.add_argument("--crop-mode", action="store_true", help="Плитки (медленнее); по умолчанию — одна картинка")
    args = parser.parse_args()
    done = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in args.out.read_text().splitlines() if line.strip()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, _attn_implementation="eager", trust_remote_code=True,
                                      use_safetensors=True, torch_dtype=torch.bfloat16).eval().cuda()  # fmt: skip
    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    with args.out.open("a") as out:
        for job in jobs:
            if job["id"] in done:
                continue
            started = time.time()
            buf = io.StringIO()
            # Сырой вывод с тегами <|ref|>/<|det|> модель печатает в stdout; сохранённый файл их уже не содержит.
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(buf):
                model.infer(tokenizer, prompt=PROMPT, image_file=job["crop"], output_path=tmp, base_size=1024,
                            image_size=768, crop_mode=args.crop_mode, save_results=False, test_compress=False)  # fmt: skip
            size = list(Image.open(job["crop"]).size)
            rec = dict(id=job["id"], raw=buf.getvalue(), size=size, seconds=round(time.time() - started, 2))
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{job['id']}: {rec['seconds']} с", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
