# DeepSeek-OCR-2 (3B, Apache 2.0)

Промпт «Convert the document to markdown» → markdown с grounding-метками, которые
`worker.py` вырезает. Модель не инструктивная: дописывать к промпту правила нельзя,
результат от этого только портится (проверено).

```bash
uv sync --project research/external_ocr_models/local/deepseek_ocr2
uv run --project research/external_ocr_models/local/deepseek_ocr2 python research/external_ocr_models/local/deepseek_ocr2/worker.py --in-dir … --out-dir …
```

Особенности: карточка требует torch 2.6 + flash-attn 2.7.3 — torch 2.6 не знает Blackwell,
поэтому свежий torch и `attn=eager` (sdpa remote code не поддерживает); веса грузить сразу в
bf16 (`torch_dtype`), иначе пик загрузки 18 ГБ. Замер: 20 с на полосу, 7,2 ГБ VRAM.
Слабости: теряет тело таблиц с боковыми шапками и выдумывает шапки, нестабилен между
прогонами одной и той же полосы.
