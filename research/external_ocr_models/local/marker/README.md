# Marker 1.x (datalab, поверх surya 0.17)

Markdown с уровнями заголовков и таблицами. Пин `marker-pdf<2`: Marker 2.x / Surya 2 считают
через vllm в docker (нужен nvidia-container-runtime) или бинарник llama-server — на машине
нет ни того, ни другого.

```bash
uv sync --project research/external_ocr_models/local/marker
uv run --project research/external_ocr_models/local/marker python research/external_ocr_models/local/marker/worker.py --in-dir … --out-dir …
```

Замер: 29 с на полосу, 5,5 ГБ VRAM. Повёрнутые шапки читает, но строки таблиц путает,
отточия превращает в «20-20-20».
