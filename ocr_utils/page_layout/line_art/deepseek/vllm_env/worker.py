"""Батч-воркер DeepSeek-OCR-2 через vLLM: задания «id, вырезка» → ``<out-dir>/<промпт>.jsonl`` (дописывается, готовое пропускается).

Запускается в своём окружении (соседний ``pyproject.toml``: vLLM ≥ 0.11 знает
``DeepseekOCR2ForCausalLM``, колёса под CUDA 13 знают Blackwell) и пакет ``ocr_utils`` целиком
не импортирует — только соседний ``parse.py`` по пути:

    uv run --project ocr_utils/page_layout/line_art/deepseek/vllm_env \\
        python ocr_utils/page_layout/line_art/deepseek/vllm_env/worker.py --jobs jobs.jsonl --out-dir out/

Тяжёлый запуск — под сторожем памяти ``scripts/memory_watchdog.py``: ``systemd-run --user
MemoryMax`` на этой машине не ограничивает память, а сборка ядер FlashInfer без ``MAX_JOBS``
съедала всю память и вешала рабочий стол (2026-09-27).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# До импорта vLLM (процесс движка наследует окружение):
# - модель уже в кэше HF, а список файлов vLLM всё равно спрашивает у хаба — работаем без сети;
# - ядро внимания vLLM 0.30 читает глобальную константу LOG2E, а Triton 3.7 это запрещает;
# - сэмплер FlashInfer собирается nvcc на лету; генерация жадная — он не нужен.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parse import MODEL_ID, PROMPTS, parse  # noqa: E402

# Запрет повтора n-грамм — как у HF ``infer(eval_mode=True)``: ``no_repeat_ngram_size=35``.
NGRAM_SIZE = 35
# Токены ``<td>`` и ``</td>``, которым повторяться можно (список из README модели).
NGRAM_WHITELIST = {128821, 128822}
# Китайские иероглифы: токены с ними запрещаются (``--ban-cjk``). На нечитаемых заголовках модель
# выдумывает китайский текст с LaTeX; запрет убрал выдумки (3 → 1 на 356 размеченных) без потерь.
CJK = re.compile(r"[一-鿿]")
# Концевой токен, который vLLM оставляет в тексте при ``skip_special_tokens=False``.
END = "<｜end▁of▁sentence｜>"


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 батчем через vLLM по заданиям")
    parser.add_argument("--jobs", type=Path, required=True, help='JSONL: {"id", "crop" — путь к PNG}')
    parser.add_argument("--out-dir", type=Path, required=True, help="Куда дописывать <промпт>.jsonl")
    parser.add_argument("--prompts", default="markdown,ocr", help="Промпты через запятую (ключи PROMPTS)")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    # Эксперты декодера (MoE): Triton — без компиляции; «auto» на sm_120 собирает FlashInfer CUTLASS nvcc.
    parser.add_argument("--moe-backend", default="triton")
    parser.add_argument("--ban-cjk", action=argparse.BooleanOptionalAction, default=True)
    arguments = parser.parse_args()

    jobs = [json.loads(line) for line in arguments.jobs.read_text().splitlines() if line.strip()]
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = arguments.prompts.split(",")
    todo: dict[str, list[dict]] = {}
    for name in prompts:
        path = arguments.out_dir / f"{name}.jsonl"
        done = {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()} if path.is_file() else set()
        todo[name] = [job for job in jobs if job["id"] not in done]
        print(f"{name}: заданий {len(jobs)}, готово {len(done)}, к обработке {len(todo[name])}", flush=True)
    if not any(todo.values()):
        return 0

    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

    llm = LLM(
        model=MODEL_ID,
        max_model_len=8192,
        max_num_seqs=arguments.max_num_seqs,
        gpu_memory_utilization=arguments.gpu_memory_utilization,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
        moe_backend=arguments.moe_backend,
    )
    bias = None
    if arguments.ban_cjk:
        tokenizer = llm.get_tokenizer()
        bias = {t: -100.0 for t in range(len(tokenizer)) if CJK.search(tokenizer.decode([t]))}
        print(f"Запрещено токенов с иероглифами: {len(bias)}", flush=True)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=arguments.max_new_tokens,
        skip_special_tokens=False,
        logit_bias=bias,
        extra_args=dict(
            ngram_size=NGRAM_SIZE, window_size=arguments.max_new_tokens, whitelist_token_ids=NGRAM_WHITELIST
        ),
    )
    # Пачками: вывод пишется после каждой, и прерванный прогон теряет не больше пачки.
    chunk = 2000
    for name in prompts:
        pending = todo[name]
        started = time.time()
        with (arguments.out_dir / f"{name}.jsonl").open("a", encoding="utf-8") as out:
            for begin in range(0, len(pending), chunk):
                part = pending[begin : begin + chunk]
                images = [Image.open(job["crop"]).convert("RGB") for job in part]
                requests = [{"prompt": PROMPTS[name], "multi_modal_data": {"image": image}} for image in images]
                for job, image, output in zip(part, images, llm.generate(requests, sampling)):
                    raw = output.outputs[0].text.removesuffix(END).strip()
                    record = {
                        "id": job["id"],
                        "size": list(image.size),
                        "raw": raw,
                        "elements": parse(raw, *image.size),
                        "finish_reason": output.outputs[0].finish_reason,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
        elapsed = time.time() - started
        print(f"{name}: {len(pending)} вырезок за {elapsed:.1f} с", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
