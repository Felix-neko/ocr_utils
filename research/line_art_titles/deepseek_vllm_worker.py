"""DeepSeek-OCR-2 батчем через upstream vLLM по вырезкам стенда → ``deepseek_vllm_<промпт>.jsonl`` той же схемы, что у HF-воркера.

Запускается в своём окружении (``research/line_art_titles/vllm_env``: vLLM ≥ 0.11.1 знает
``DeepseekOCR2ForCausalLM``, колёса под CUDA 13 знают Blackwell), пакет ``ocr_utils`` не
импортирует. Промпты и разбор вывода — из соседнего ``deepseek_worker.py`` (тоже без
зависимостей от проекта), чтобы выходы двух воркеров сравнивались один в один:

    uv run --project research/line_art_titles/vllm_env \\
        python research/line_art_titles/deepseek_vllm_worker.py --out-dir <выход стенда>

Все вырезки промпта уходят одним ``llm.generate`` — пакет собирает сам vLLM (continuous batching).
Размеры картинки у vLLM те же, что у HF remote code: глобальный вид 1024, плитки 768, до 6 плиток.
"""

from __future__ import annotations

import argparse
import json
import re
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# До импорта vLLM (процесс движка наследует окружение):
# - модель уже в кэше HF, а список файлов vLLM всё равно спрашивает у хаба и при медленной сети
#   падает по таймауту — работаем без сети;
# - ядро внимания vLLM 0.30 читает глобальную константу ``LOG2E``, а Triton 3.7 это запрещает
#   («Cannot access global variable LOG2E from within @jit'ed function»); обход — из текста ошибки.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
# - FlashInfer собирает ядра на лету системным nvcc, а он здесь 12.0 и не знает sm_120 (нужна CUDA
#   12.8+): сэмплер top-k/top-p FlashInfer падает уже на прогреве. Генерация жадная — он не нужен.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepseek_worker import MODEL_ID, PROMPTS, parse  # noqa: E402

# Запрет повтора n-грамм — как у HF ``infer(eval_mode=True)``: ``no_repeat_ngram_size=35`` по всей
# генерации (окно = вся длина ответа). Без него модель на таблицах зацикливается.
NGRAM_SIZE = 35

# Китайские иероглифы (для запрета токенов, ``--ban-cjk``).
CJK = re.compile(r"[\u4e00-\u9fff]")

# Токены ``<td>`` и ``</td>``, которым повторяться можно (список из README модели).
NGRAM_WHITELIST = {128821, 128822}


def peak_vram_mib(stop: threading.Event, samples: list[int], period_s: float = 0.5) -> None:
    """Опрашивать ``nvidia-smi`` до ``stop`` и складывать занятую видеопамять, МиБ, в ``samples``.

    vLLM держит модель в отдельном процессе движка, поэтому ``torch.cuda.max_memory_allocated``
    здесь ничего не покажет; меряется занятость всей карты (с чужими процессами).

    Args:
        stop: Событие остановки опроса.
        samples: Куда дописывать замеры (единственный способ отдать их из потока).
        period_s: Период опроса, с.
    """
    while not stop.is_set():
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True
        )
        if done.returncode == 0 and done.stdout.strip():
            samples.append(int(done.stdout.split()[0]))
        stop.wait(period_s)


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 батчем через vLLM по вырезкам стенда line_art_titles")
    parser.add_argument("--out-dir", type=Path, required=True, help="Выход стенда: regions.jsonl и crops/")
    parser.add_argument("--prompts", default="markdown,ocr", help="Промпты через запятую (ключи PROMPTS)")
    parser.add_argument("--ids", default=None, help="Файл со списком id областей; по умолчанию все")
    parser.add_argument("--prefix", default="deepseek_vllm", help="Начало имени выходных файлов")
    # Сколько последовательностей движок держит одновременно; упирается в KV-кэш на 16 ГБ.
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-new-tokens", type=int, default=1536, help="Потолок ответа — как у HF-воркера")
    # Слои экспертов (декодер — DeepSeekV2 MoE): «auto» выбирает на sm_120 ядро FlashInfer CUTLASS,
    # которое FlashInfer собирает на лету nvcc и падает («No supported CUDA architectures found for
    # major versions [12]»). Ядро Triton компилируется без nvcc.
    parser.add_argument("--moe-backend", default="triton")
    # Запрет токенов с китайскими иероглифами (logit_bias −100): на нечитаемых заголовках модель
    # выдумывает китайский текст с LaTeX («已知 \\( f(x) … \\)»), на паке-1 китайского текста нет.
    # По умолчанию включено (сравнение трёх вариантов, reports/line_art_titles.md): выдумок 3 → 1,
    # метрики не хуже, цена — генерация в 1.5 раза медленнее. ``--no-ban-cjk`` — прежнее поведение.
    parser.add_argument("--ban-cjk", action=argparse.BooleanOptionalAction, default=True)
    # Дописка к промпту (эксперимент: модель не инструктивная, см. external_ocr_models/local/deepseek_ocr2/README.md).
    parser.add_argument("--prompt-suffix", default="")
    arguments = parser.parse_args()

    rows = [json.loads(line) for line in (arguments.out_dir / "regions.jsonl").read_text().splitlines() if line]
    if arguments.ids:
        wanted = {line.strip() for line in Path(arguments.ids).read_text().splitlines() if line.strip()}
        rows = [row for row in rows if row["id"] in wanted]

    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

    images = [Image.open(arguments.out_dir / "crops" / row["crop"]).convert("RGB") for row in rows]
    samples: list[int] = []
    stop = threading.Event()
    watcher = threading.Thread(target=peak_vram_mib, args=(stop, samples), daemon=True)
    watcher.start()

    started = time.time()
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
    load_s = time.time() - started
    bias = None
    if arguments.ban_cjk:
        tokenizer = llm.get_tokenizer()
        # Токен с хотя бы одним иероглифом CJK (декодируется по одному: байтовые куски иероглифов
        # отдельно не видны, но полные иероглифы в словаре DeepSeek — целые токены).
        bias = {token_id: -100.0 for token_id in range(len(tokenizer)) if CJK.search(tokenizer.decode([token_id]))}
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

    meta: dict = {
        "model": MODEL_ID,
        "images": len(rows),
        "load_s": round(load_s, 1),
        "max_num_seqs": arguments.max_num_seqs,
        "max_new_tokens": arguments.max_new_tokens,
        "moe_backend": arguments.moe_backend,
        "ban_cjk": arguments.ban_cjk,
        "prompt_suffix": arguments.prompt_suffix,
        "prompts": {},
    }
    for name in arguments.prompts.split(","):
        # Хвостовой пробел промпта из README («…markdown. ») HF remote code срезает: format_messages
        # делает content.strip() и get_prompt().strip(). В vLLM промпт идёт как есть, и лишний токен
        # пробела (223) после «.» уводит модель в мусор («1. 1. 1. …», описание картинки) — на 356
        # размеченных вырезках это 23 мусорных ответа и 17 потерянных надписей против 1 и 0 у HF.
        prompt = (PROMPTS[name].strip() + (" " + arguments.prompt_suffix if arguments.prompt_suffix else "")).strip()
        requests = [{"prompt": prompt, "multi_modal_data": {"image": image}} for image in images]
        started = time.time()
        outputs = llm.generate(requests, sampling)
        elapsed = time.time() - started
        truncated = 0
        with (arguments.out_dir / f"{arguments.prefix}_{name}.jsonl").open("w", encoding="utf-8") as out:
            for row, image, output in zip(rows, images, outputs):
                raw = output.outputs[0].text
                # Хвост «конец предложения» HF-воркер срезает; здесь его нет при skip_special_tokens=False
                # только если модель остановилась сама — срезаем так же.
                raw = raw.removesuffix("<｜end▁of▁sentence｜>").strip()
                truncated += output.outputs[0].finish_reason == "length"
                record = {
                    "id": row["id"],
                    "prompt": name,
                    "size": list(image.size),
                    "raw": raw,
                    "elements": parse(raw, *image.size),
                    "finish_reason": output.outputs[0].finish_reason,
                    "output_tokens": len(output.outputs[0].token_ids),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta["prompts"][name] = {
            "seconds": round(elapsed, 1),
            "per_image_s": round(elapsed / max(1, len(rows)), 3),
            "truncated": truncated,
            "output_tokens": sum(len(o.outputs[0].token_ids) for o in outputs),
        }
        print(
            f"{name}: {len(rows)} вырезок за {elapsed:.1f} с ({elapsed / max(1, len(rows)):.2f} с/вырезку)", flush=True
        )

    stop.set()
    watcher.join()
    meta["peak_vram_mib"] = max(samples) if samples else None
    (arguments.out_dir / f"{arguments.prefix}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
