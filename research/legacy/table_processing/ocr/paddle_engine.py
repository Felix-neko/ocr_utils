"""Распознавание ячейки через PaddleOCR в ОТДЕЛЬНОМ окружении.

ПОЧЕМУ ОТДЕЛЬНОЕ ОКРУЖЕНИЕ. ``paddlepaddle`` тянет свой рантайм и свою сборку protobuf и
конфликтует с тем, что уже стоит в окружении проекта; в ``pyproject.toml`` проекта индекс
для ``paddlepaddle-gpu`` объявлен, а сама зависимость намеренно закомментирована. Ради
одного сравнительного замера ломать основное окружение нельзя, поэтому Paddle живёт в
``research/legacy/table_processing/paddle_env`` и вызывается подпроцессом — ровно так же, как в
проекте вызывается tesseract.

ПОЧЕМУ НЕ RAPIDOCR, где те же модели PP-OCRv5 в ONNX. Он есть в зависимостях (3.8.1), но
падает с «onnxruntime is not installed»: рантайм в проекте не заперт. Добавлять его в
основное окружение ради сравнения — то же самое, от чего мы уходим, а кириллические веса
PP-OCRv5 rapidocr всё равно качает с modelscope при первом вызове.

ОБМЕН ЧЕРЕЗ ПАПКУ. Вырезки пишутся PNG во временный каталог, ответ читается JSON. Пачкой,
а не по одной: запуск интерпретатора с paddle стоит несколько секунд, и на ячейку такое
не окупается.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from research.legacy.table_processing.ocr.base import Engine, OcrResult, prepare

ENV_DIR = Path(__file__).resolve().parents[1] / "paddle_env"
WORKER = ENV_DIR / "worker.py"

# Запуск интерпретатора с paddle и загрузка весов — десятки секунд; на пачку из полусотни
# ячеек этого хватает с большим запасом.
TIMEOUT_S = 900


def paddle_available() -> bool:
    """Окружение считается готовым, если оно синхронизировано: есть свой интерпретатор."""
    return WORKER.is_file() and (ENV_DIR / ".venv" / "bin" / "python").is_file() and shutil.which("uv") is not None


class PaddleRecognizer:
    def __call__(self, images: Sequence[np.ndarray]) -> list[OcrResult]:
        if not images:
            return []
        started = time.time()
        with tempfile.TemporaryDirectory(prefix="paddlecells_") as work:
            work_dir = Path(work)
            in_dir = work_dir / "in"
            in_dir.mkdir()
            names = []
            for index, gray in enumerate(images):
                name = f"{index:04d}.png"
                cv2.imwrite(str(in_dir / name), prepare(gray))
                names.append(name)
            out_path = work_dir / "out.json"
            try:
                subprocess.run(
                    [
                        "uv",
                        "run",
                        "--project",
                        str(ENV_DIR),
                        "python",
                        str(WORKER),
                        "--in-dir",
                        str(in_dir),
                        "--out",
                        str(out_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_S,
                    check=False,
                )
                payload = json.loads(out_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                return [OcrResult(note=f"PaddleOCR: {error}") for _ in images]

        elapsed = (time.time() - started) / max(1, len(images))
        results: list[OcrResult] = []
        for name in names:
            item = payload.get(name, {})
            results.append(
                OcrResult(
                    lines=[str(line) for line in item.get("lines", []) if str(line).strip()],
                    confidence=float(item.get("confidence", 0.0)),
                    seconds=float(item.get("seconds", elapsed)),
                    note=str(item.get("note", "")),
                )
            )
        return results


ALGORITHM = Engine(
    name="paddle",
    summary="PaddleOCR PP-OCRv5, lang=ru, в отдельном окружении (подпроцессом)",
    stage="cpu",
    make=PaddleRecognizer,
    available=paddle_available,
)
