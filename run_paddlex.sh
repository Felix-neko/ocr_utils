#!/bin/bash
# Wrapper для запуска PaddleX с GPU поддержкой

# Добавляем CUDA 12 библиотеки в LD_LIBRARY_PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_LIB_PATH="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
CUDNN_LIB_PATH="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
CUBLAS_LIB_PATH="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/nvidia/cublas/lib"

export LD_LIBRARY_PATH="$CUDA_LIB_PATH:$CUDNN_LIB_PATH:$CUBLAS_LIB_PATH:$LD_LIBRARY_PATH"

# Запускаем скрипт
uv run python "$SCRIPT_DIR/ocr_utils/enhance_scantailor_bounds_paddlex.py" "$@"
