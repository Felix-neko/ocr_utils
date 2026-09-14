# dots.ocr (rednote-hilab, 3B, MIT)

Layout и текст одной моделью: список элементов с категориями (Title, Section-header, Text,
Table в HTML, Page-header/footer, Picture…), из которого `worker.py` собирает markdown и
поля страницы (номер — из колонтитулов).

```bash
uv sync --project research/external_ocr_models/local/dots_ocr
uv run --project research/external_ocr_models/local/dots_ocr python research/external_ocr_models/local/dots_ocr/worker.py --in-dir … --out-dir …
```

Особенности, без которых не заводится (все учтены в `worker.py`/`pyproject.toml`):
* transformers < 4.54 — remote code модели не передаёт `video_processor`; на 4.54+ воркер
  собирает процессор Qwen2.5-VL руками, но надёжнее пин;
* веса в папке без точки в имени (`weights/DotsOCR`) — иначе не импортируется remote code;
* башне зрения принудительно `sdpa`: её `attn_implementation` берётся из config.json
  (flash_attention_2), без flash-attn она откатывается на eager и падает по OOM.

Замер на RTX 5060 Ti, полоса 2200 px: 27 с, 6,7 ГБ VRAM.
