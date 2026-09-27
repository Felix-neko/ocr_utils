"""DeepSeek-OCR-2 в режиме grounding по вырезкам областей line art: блоки с рамками и текстом → ``deepseek_<промпт>.jsonl``.

Запускается в отдельном окружении модели (remote code пришит к transformers 4.46.3, см.
``research/external_ocr_models/local/deepseek_ocr2/README.md``), поэтому пакет ``ocr_utils``
не импортирует — только стандартная библиотека и сама модель:

    uv run --project research/external_ocr_models/local/deepseek_ocr2 \\
        python research/line_art_titles/deepseek_worker.py --out-dir <выход стенда> --prompt markdown

Вывод модели в режиме ``<|grounding|>`` — теги ``<|ref|>метка<|/ref|><|det|>[[x1, y1, x2, y2], …]<|/det|>``,
за которыми идёт текст элемента. Координаты нормированы на 0–999 по каждой оси исходной картинки
(так их переводит и штатная отрисовка модели: ``x / 999 * ширина``). Прогон дописывает файл и
пропускает уже обработанные области — прерванный прогон перезапускается.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path

MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"

# Промпты режима grounding. ``markdown`` — штатный промпт документа из карточки модели (рамки
# блоков вёрстки с метками title/text/image …); ``ocr`` — промпт «прочих картинок» из первой
# DeepSeek-OCR (у неё он давал рамки по строкам текста; у второй версии в карточке не значится).
PROMPTS = {
    "markdown": "<image>\n<|grounding|>Convert the document to markdown. ",
    "ocr": "<image>\n<|grounding|>OCR this image. ",
}

# Один элемент вывода: метка, список рамок и текст до следующего тега.
ELEMENT = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>(.*?)(?=<\|ref\|>|\Z)", re.DOTALL)

# Служебные теги модели, которые в тексте элемента не нужны.
TAGS = re.compile(r"<\|[^|]*\|>")

# Шкала нормированных координат модели.
SCALE = 999


def parse(raw: str, width: int, height: int) -> list[dict]:
    """Разобрать вывод модели в элементы с рамками в пикселях картинки.

    Args:
        raw: Сырой вывод ``infer(..., eval_mode=True)`` с тегами grounding.
        width: Ширина картинки, px.
        height: Высота картинки, px.

    Returns:
        По словарю на рамку: ``x0, y0, x1, y1`` (px, ``x1``/``y1`` — за последним пикселем),
        ``label`` — содержимое ``<|ref|>`` (тип блока или сам текст, смотря по промпту),
        ``text`` — текст после тега до следующего элемента (без служебных тегов).
        Элемент с несколькими рамками даёт несколько словарей с одним текстом.
    """
    elements = []
    for label, boxes, text in ELEMENT.findall(raw):
        try:
            coordinates = ast.literal_eval(boxes.strip())
        except (ValueError, SyntaxError):
            continue
        clean = TAGS.sub("", text).strip()
        for box in coordinates:
            if len(box) != 4:
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            elements.append(
                {
                    "x0": int(round(min(x0, x1) / SCALE * width)),
                    "y0": int(round(min(y0, y1) / SCALE * height)),
                    "x1": int(round(max(x0, x1) / SCALE * width)),
                    "y1": int(round(max(y0, y1) / SCALE * height)),
                    "label": label.strip(),
                    "text": clean,
                }
            )
    return elements


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 grounding по вырезкам стенда line_art_titles")
    parser.add_argument("--out-dir", type=Path, required=True, help="Выход стенда: regions.jsonl и crops/")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="markdown")
    parser.add_argument("--ids", default=None, help="Файл со списком id областей (по строке); по умолчанию все")
    parser.add_argument("--output", default=None, help="Имя выходного JSONL; по умолчанию deepseek_<промпт>.jsonl")
    # sdpa remote code не поддерживает, flash-attn под Blackwell не собран — eager (как у внешнего OCR).
    parser.add_argument("--attn", default="eager")
    # Разрешение: глобальный вид 1024, плитки 768 («Gundam») — штатный режим карточки модели.
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--no-crop", action="store_true", help="Не резать крупную вырезку на плитки")
    # ``infer()`` жёстко просит до 8192 токенов: на таблице это 4 минуты генерации HTML, который
    # нам не нужен. Потолок ставится подменой ``generate`` у экземпляра модели.
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    arguments = parser.parse_args()

    rows = [json.loads(line) for line in (arguments.out_dir / "regions.jsonl").read_text().splitlines() if line]
    if arguments.ids:
        wanted = {line.strip() for line in Path(arguments.ids).read_text().splitlines() if line.strip()}
        rows = [row for row in rows if row["id"] in wanted]
    output = arguments.out_dir / (arguments.output or f"deepseek_{arguments.prompt}.jsonl")
    done = set()
    if output.is_file():
        done = {json.loads(line)["id"] for line in output.read_text().splitlines() if line.strip()}
    rows = [row for row in rows if row["id"] not in done]
    print(f"Вырезок к обработке: {len(rows)} (уже есть {len(done)}), промпт {arguments.prompt}", flush=True)
    if not rows:
        return 0

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Веса сразу в bf16: загрузка в fp32 с последующим приведением даёт пик 18 ГБ.
    model = AutoModel.from_pretrained(
        MODEL_ID,
        _attn_implementation=arguments.attn,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    original_generate = model.generate

    def limited_generate(*args, **kwargs):
        # Функция внутри функции нужна: подмена метода экземпляра должна замкнуть исходный
        # ``generate`` и потолок из аргументов командной строки.
        kwargs["max_new_tokens"] = min(kwargs.get("max_new_tokens", arguments.max_new_tokens), arguments.max_new_tokens)
        return original_generate(*args, **kwargs)

    model.generate = limited_generate

    started_all = time.time()
    with output.open("a", encoding="utf-8") as out:
        for number, row in enumerate(rows, 1):
            path = arguments.out_dir / "crops" / row["crop"]
            width, height = Image.open(path).size
            started = time.time()
            record: dict = {"id": row["id"], "prompt": arguments.prompt, "size": [width, height]}
            try:
                raw = model.infer(
                    tokenizer,
                    prompt=PROMPTS[arguments.prompt],
                    image_file=str(path),
                    output_path=str(arguments.out_dir),
                    base_size=arguments.base_size,
                    image_size=arguments.image_size,
                    crop_mode=not arguments.no_crop,
                    save_results=False,
                    eval_mode=True,
                )
                record["raw"] = raw
                record["elements"] = parse(raw, width, height)
            except Exception as error:  # одна битая вырезка не должна ронять прогон
                record["error"] = f"{type(error).__name__}: {str(error)[:300]}"
            record["seconds"] = round(time.time() - started, 2)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if number % 25 == 0 or number == len(rows):
                elapsed = time.time() - started_all
                print(
                    f"{number}/{len(rows)}, {elapsed / number:.1f} с на вырезку, "
                    f"пик VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} ГБ",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
