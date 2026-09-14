# PaddleOCR-VL 1.6 (0.9B, Apache 2.0)

Полный документный пайплайн `PaddleOCRVL` (PP-DocLayout + VLM) → markdown с таблицами в
HTML. Требует рантайм paddlepaddle; GPU-колёса лежат только на CDN Baidu
(paddle-whl.cdn.bcebos.com), который с этой машины не отвечает, поэтому в `pyproject.toml`
CPU-сборка с PyPI (в комментарии — как переключиться на GPU/cu130).

```bash
uv sync --project research/external_ocr_models/local/paddleocr_vl
uv run --project research/external_ocr_models/local/paddleocr_vl python research/external_ocr_models/local/paddleocr_vl/worker.py --in-dir … --out-dir …
```

## Как достать GPU-колесо

Индекс `https://www.paddlepaddle.org.cn/packages/stable/cu130/paddlepaddle-gpu/` перечисляет
колёса, но ссылки ведут на CDN `paddle-whl.cdn.bcebos.com`, который с этой машины не
отвечает (таймаут/обрыв TLS). Источник отвечает — качать с него в `wheels/` (папка в
`.gitignore`, 2 ГБ), `uv sync` возьмёт колесо оттуда через `find-links`:

```bash
W=paddlepaddle_gpu-3.3.1-cp311-cp311-linux_x86_64.whl
curl -L -C - --retry 20 --retry-all-errors -o research/external_ocr_models/local/paddleocr_vl/wheels/$W \
    https://paddle-whl.bj.bcebos.com/stable/cu130/paddlepaddle-gpu/$W
uv sync --project research/external_ocr_models/local/paddleocr_vl
```

Torch в это окружение не ставить: `paddlepaddle-gpu 3.3.1` пришпилен к `nvidia-cudnn-cu13 9.13`,
torch ≥ 2.12 — к 9.20, uv их не совмещает; VLM считает сам paddle. Замер на RTX 5060 Ti:
19-29 с на полосу, 1,3 ГБ VRAM.
