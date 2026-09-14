"""DeepSeek-OCR-2 по папке полос: «Convert the document to markdown» → markdown.

Выход на полосу — ``<имя>.md`` (markdown модели с вырезанными grounding-метками) и
``<имя>.meta.json``. Запуск: ``uv run --project research/external_ocr_models/local/deepseek_ocr2 python worker.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import tempfile
import time
from pathlib import Path

MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"
PROMPT_MARKDOWN = "<image>\n<|grounding|>Convert the document to markdown. "
PROMPT_FREE = "<image>\nFree OCR. "

# Метки привязки к координатам: <|ref|>text<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>.
_GROUNDING = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>.*?<\|/det\|>", re.DOTALL)
_TAGS = re.compile(r"<\|[^|]*\|>")


def clean(markdown: str) -> str:
    text = _GROUNDING.sub(r"\1", markdown)
    text = _TAGS.sub("", text)
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 по папке полос")
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    # sdpa remote code этой модели не поддерживает, flash-attn под Blackwell не собран — eager;
    # на плитках 768 px он укладывается в ~13 ГБ.
    parser.add_argument("--attn", default="eager", help="flash_attention_2 | eager")
    parser.add_argument("--free-ocr", action="store_true", help="Режим «Free OCR» без разметки")
    parser.add_argument(
        "--extra-prompt", default="", help="Дописать к штатному промпту (эксперимент: модель не инструктивная)"
    )
    # Разрешение: base_size — глобальный вид, image_size — плитки, crop_mode — резать на плитки
    # («Gundam»: до 6 плиток 768 + вид 1024). Для полосы журнала это и есть штатный режим.
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--no-crop", action="store_true")
    arguments = parser.parse_args()
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Веса грузим сразу в bf16: загрузка в fp32 с последующим .to(bfloat16) держит пик
    # 12 + 6 ГБ и на 16 ГБ карте падает по OOM ещё до первой полосы.
    model = AutoModel.from_pretrained(
        MODEL_ID,
        _attn_implementation=arguments.attn,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    prompt = (PROMPT_FREE if arguments.free_ocr else PROMPT_MARKDOWN) + arguments.extra_prompt

    for path in sorted(arguments.in_dir.glob("*.jpg")):
        started = time.time()
        meta: dict = {
            "engine": "DeepSeek-OCR-2",
            "attn": arguments.attn,
            "base_size": arguments.base_size,
            "image_size": arguments.image_size,
            "crop": not arguments.no_crop,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                # infer() печатает результат в stdout и пишет result.mmd в output_path; берём файл,
                # а печать глушим, чтобы лог не разбухал.
                with contextlib.redirect_stdout(io.StringIO()):
                    result = model.infer(
                        tokenizer,
                        prompt=prompt,
                        image_file=str(path),
                        output_path=tmp,
                        base_size=arguments.base_size,
                        image_size=arguments.image_size,
                        crop_mode=not arguments.no_crop,
                        save_results=True,
                        test_compress=False,
                    )
                candidates = list(Path(tmp).glob("*.mmd")) + list(Path(tmp).glob("*.md"))
                raw = (
                    candidates[0].read_text(encoding="utf-8")
                    if candidates
                    else (result if isinstance(result, str) else "")
                )
            meta["vram_mb"] = int(torch.cuda.max_memory_allocated() / 2**20)
            (arguments.out_dir / (path.stem + ".raw.txt")).write_text(raw, encoding="utf-8")
            (arguments.out_dir / (path.stem + ".md")).write_text(clean(raw), encoding="utf-8")
        except Exception as error:  # одна битая полоса не должна ронять пакет
            meta["error"] = f"{type(error).__name__}: {str(error)[:300]}"
        meta["seconds"] = round(time.time() - started, 2)
        (arguments.out_dir / (path.stem + ".meta.json")).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{path.name}: {meta.get('error') or 'ok'}, {meta['seconds']} с", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
