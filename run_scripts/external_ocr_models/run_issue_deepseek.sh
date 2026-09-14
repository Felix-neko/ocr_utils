#!/usr/bin/env bash
#
# Весь выпуск через DeepSeek V4.1 Flash (2 куска) с текущим промптом — в отдельную папку,
# чтобы сравнивать версии промпта между собой.
#
# Аргументы: имя папки выхода (по умолчанию deepseek-v41-flash-s2-v<PROMPT_VERSION>),
# выпуск (по умолчанию 1966/03); остальное уходит в run.
# ВЫХОД — EXTERNAL_OCR_ROOT/<имя>/<выпуск>/…; ~4 минуты и ~9 ¢ на 97 полос.
set -euo pipefail
source "$(dirname "$0")/common.sh"

set -m
trap 'trap - EXIT INT TERM; kill -- -$$ 2>/dev/null' EXIT INT TERM

VERSION=$(uv run python -c "from research.external_ocr_models import PROMPT_VERSION; print(PROMPT_VERSION)" 2>/dev/null)
NAME="${1:-deepseek-v41-flash-s2-v$VERSION}"; shift || true
ISSUE="${1:-1966/03}"; shift || true

uv run python -m research.external_ocr_models run \
    --in-dir "$SHARPENED_DIR/$ISSUE" \
    --out-dir "$EXTERNAL_OCR_ROOT/$NAME/$ISSUE" \
    --model deepseek-v41-flash \
    --strips 2 \
    --jobs "$EXTERNAL_OCR_JOBS" \
    --skip-done \
    "$@"
uv run python -m research.external_ocr_models balance
