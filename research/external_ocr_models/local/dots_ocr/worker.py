"""dots.ocr по папке полос: layout + текст одной моделью → markdown и поля страницы.

Вход — папка с JPEG, выход — на каждую полосу ``<имя>.json`` (поля PageResult: тело в
markdown, номер страницы и колонтитулы из элементов Page-header/Page-footer) и
``<имя>.meta.json`` (секунды, число элементов, ошибка). Запускается в своём окружении:
``uv run --project research/external_ocr_models/local/dots_ocr python worker.py ...``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ID = "rednote-hilab/dots.ocr"
# Папка с весами не должна содержать точку: remote code импортируется как python-модуль
# по имени папки, и «dots.ocr» ломает импорт (известная особенность модели).
WEIGHTS_DIR = Path(__file__).parent / "weights" / "DotsOCR"

PROMPT_LAYOUT_ALL = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
3. Text Extraction & Formatting Rules:
   - Picture: Omit text field
   - Formula: Format as LaTeX
   - Table: Format as HTML
   - Others: Format as Markdown
4. Constraints: Original text only, sorted by reading order
5. Final Output: Single JSON object"""

_PAGE_NUMBER = re.compile(r"(?<!\d)(\d{1,3})(?!\d)")


def ensure_weights() -> Path:
    if not (WEIGHTS_DIR / "config.json").is_file():
        from huggingface_hub import snapshot_download

        WEIGHTS_DIR.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(REPO_ID, local_dir=str(WEIGHTS_DIR))
    return WEIGHTS_DIR


def load_model(attn: str):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    path = str(ensure_weights())
    # Башня зрения берёт свой attn_implementation из config.json (flash_attention_2), а без
    # flash-attn молча откатывается на eager с O(n²) памятью — на полосе 2200 px это OOM
    # на 16 ГБ. Поэтому прописываем в конфиг зрения тот же режим, что и у языковой части.
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config.vision_config.attn_implementation = attn
    model = AutoModelForCausalLM.from_pretrained(
        path,
        config=config,
        attn_implementation=attn,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    try:
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    except TypeError:
        # Remote code модели не передаёт video_processor, обязательный в transformers ≥ 4.54
        # (dots.ocr issue #46): собираем тот же Qwen2.5-VL-процессор руками.
        from transformers import AutoTokenizer, Qwen2_5_VLProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor

        chat_template = json.loads((Path(path) / "chat_template.json").read_text(encoding="utf-8"))["chat_template"]
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        processor = Qwen2_5_VLProcessor(
            Qwen2VLImageProcessor.from_pretrained(path), tokenizer, Qwen2VLVideoProcessor(), chat_template=chat_template
        )
        processor.image_token = getattr(tokenizer, "image_token", "<|imgpad|>")
        processor.image_token_id = getattr(tokenizer, "image_token_id", 151665)
    return model, processor


def parse_elements(text: str) -> list[dict]:
    """Ответ модели — JSON-список элементов; иногда в ```-ограждении."""
    text = text.strip()
    fence = re.match(r"^```[a-z]*\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("в ответе нет JSON-списка")
    return json.loads(text[start : end + 1])


def _page_number(margins: list[str]) -> str | None:
    """Номер страницы из колонтитулов: сначала «голое» число, потом число с краю строки.

    Число после «№» — номер выпуска в шапке журнала, не страница.
    """
    for text in margins:
        if text.strip().isdigit():
            return text.strip()
    for text in margins:
        for match in _PAGE_NUMBER.finditer(text):
            before = text[: match.start()].rstrip()
            at_edge = match.start() == 0 or match.end() >= len(text.rstrip())
            if at_edge and not before.endswith("№"):
                return match.group(1)
    return None


def to_page(elements: list[dict]) -> dict:
    """Элементы в порядке чтения → markdown-тело и поля страницы."""
    body: list[str] = []
    headers: list[str] = []
    footers: list[str] = []
    for element in elements:
        category = element.get("category") or "Text"
        text = (element.get("text") or "").strip()
        if category == "Picture":
            body.append("> [картинка]")
            continue
        if not text:
            continue
        if category == "Page-header":
            headers.append(text)
        elif category == "Page-footer":
            footers.append(text)
        elif category == "Title":
            body.append("# " + text.lstrip("# "))
        elif category == "Section-header":
            body.append("## " + text.lstrip("# "))
        elif category == "List-item":
            body.append(text if text.startswith(("-", "*", "•")) else "- " + text)
        elif category == "Caption":
            body.append("*" + text + "*")
        elif category == "Footnote":
            body.append(text)
        elif category == "Formula":
            body.append("$$" + text + "$$")
        else:  # Text, Table (HTML как есть)
            body.append(text)
    page_number = _page_number(headers + footers)
    markdown = "\n\n".join(body)
    return {
        "content_markdown": markdown,
        "page_number": page_number,
        "running_header": " | ".join(headers) or None,
        "running_footer": " | ".join(footers) or None,
        "is_toc": "СОДЕРЖАНИЕ" in markdown[:400].upper(),
        "notes": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="dots.ocr по папке полос")
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--attn", default="sdpa", help="flash_attention_2 при наличии flash-attn, иначе sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=16000)
    arguments = parser.parse_args()
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    model, processor = load_model(arguments.attn)
    for path in sorted(arguments.in_dir.glob("*.jpg")):
        started = time.time()
        meta: dict = {"engine": "dots.ocr", "attn": arguments.attn}
        try:
            image = Image.open(path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT_LAYOUT_ALL}],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=arguments.max_new_tokens, do_sample=False)
            trimmed = generated[0][inputs.input_ids.shape[1] :]
            raw = processor.batch_decode([trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            meta["prompt_tokens"] = int(inputs.input_ids.shape[1])
            meta["completion_tokens"] = int(trimmed.shape[0])
            meta["vram_mb"] = int(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None
            elements = parse_elements(raw)
            meta["elements"] = len(elements)
            page = to_page(elements)
            (arguments.out_dir / (path.stem + ".json")).write_text(
                json.dumps(page, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except Exception as error:  # одна битая полоса не должна ронять пакет
            meta["error"] = f"{type(error).__name__}: {str(error)[:300]}"
            if "raw" in locals():
                (arguments.out_dir / (path.stem + ".raw.txt")).write_text(raw, encoding="utf-8")
        meta["seconds"] = round(time.time() - started, 2)
        (arguments.out_dir / (path.stem + ".meta.json")).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{path.name}: {meta.get('error') or 'ok'}, {meta['seconds']} с", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
