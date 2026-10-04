"""Запуск боевого воркера DeepSeek-OCR-2 (vLLM) с промптом ``free`` («Free OCR.») — в окружении воркера, без ``ocr_utils``."""

from __future__ import annotations

import sys
from pathlib import Path

DEEPSEEK_DIR = Path(__file__).resolve().parents[2] / "ocr_utils" / "page_layout" / "line_art" / "deepseek"
sys.path.insert(0, str(DEEPSEEK_DIR))
sys.path.insert(0, str(DEEPSEEK_DIR / "vllm_env"))

import parse  # noqa: E402

# Штатный промпт простого распознавания из карточки модели — единственный, что не дописывает обрубки сам.
parse.PROMPTS["free"] = "<image>\nFree OCR."

import worker  # noqa: E402

if __name__ == "__main__":
    sys.exit(worker.main())
