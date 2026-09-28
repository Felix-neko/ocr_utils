"""Запуск боевого воркера DeepSeek-OCR-2 (vLLM) с дополнительным промптом ``free`` — простой OCR без рамок.

Боевой воркер ``ocr_utils/page_layout/line_art/deepseek/vllm_env/worker.py`` знает только промпты
``PROMPTS`` из ``parse.py`` (``markdown`` и ``ocr``, оба в режиме grounding). Для разностного судьи
(«текст вырезки с пятном» против «текст без пятна») нужен и штатный промпт простого распознавания из
карточки модели. Он добавляется в словарь ``PROMPTS`` здесь, до импорта воркера: воркер берёт тот же
объект словаря. Код пакета не меняется.

Запускается в окружении воркера (``ocr_utils`` не импортирует)::

    uv run --project ocr_utils/page_layout/line_art/deepseek/vllm_env \\
        python research/text_block_specks/deepseek_run.py --jobs jobs.jsonl --out-dir out/ --prompts free,ocr
"""

from __future__ import annotations

import sys
from pathlib import Path

# Каталоги боевого кода DeepSeek: ``parse.py`` лежит в ``deepseek/``, воркер — в ``deepseek/vllm_env/``.
DEEPSEEK_DIR = Path(__file__).resolve().parents[2] / "ocr_utils" / "page_layout" / "line_art" / "deepseek"
sys.path.insert(0, str(DEEPSEEK_DIR))
sys.path.insert(0, str(DEEPSEEK_DIR / "vllm_env"))

import parse  # noqa: E402

# Штатный промпт простого распознавания из карточки DeepSeek-OCR-2 (без хвостового пробела, см. ``parse.py``).
parse.PROMPTS["free"] = "<image>\nFree OCR."

import worker  # noqa: E402

if __name__ == "__main__":
    sys.exit(worker.main())
