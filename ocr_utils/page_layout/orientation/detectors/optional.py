"""Сторонний ONNX-классификатор ориентации, подключаемый файлом-описанием.

ЗАЧЕМ ТАК, А НЕ ЗАШИТЫМИ МОДЕЛЯМИ. В обзоре нашлись два готовых классификатора, которые
хотелось бы иметь в сравнении:

* PP-LCNet_x1_0_doc_ori из PaddleOCR — 99.06% top-1, 7 МБ, обучен на документах. На HF
  выложены только веса в формате paddle inference; ONNX в официальном репозитории нет,
  а ``paddlepaddle-gpu`` в pyproject намеренно закомментирован.
* DuarteBarbosa/deep-image-orientation-detection — EfficientNetV2, 98.82%, MIT. Обучен на
  фотографиях, а не на документах, и тем интересен: судит по картинке, а не по тексту.
  Предобработка описана в его репозитории, а не в карточке модели.

Общее у них одно: чтобы подключить любую, нужно знать размер входа, нормировку и порядок
классов. Угадывать это нельзя — детектор с чужой нормировкой не падает, он тихо выдаёт
правдоподобную чушь, и заметить это можно очень нескоро. Поэтому здесь не зашита ни одна
конкретная модель, а есть запуск ONNX по описанию, которое кладёт рядом тот, кто модель
достал и проверил.

Формат описания (JSON рядом с ``model.onnx``, путь задаётся ``--onnx-model``):

    {
      "size": [224, 224],            // ширина, высота входа
      "layout": "NCHW",              // или "NHWC"
      "mean": [0.485, 0.456, 0.406], // на диапазон 0..1
      "std":  [0.229, 0.224, 0.225],
      "classes": [0, 90, 180, 270],  // угол каждого выхода, ПО ЧАСОВОЙ
      "meaning": "correction"        // "correction" — угол уже наша валюта,
    }                                // "state" — угол, на который картинка повёрнута

Правильность описания подтверждается командой ``validate``: она крутит заведомо прямые
полосы и показывает матрицу ошибок. Косая нормировка или перепутанный порядок классов
вылезают там сразу.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from ocr_utils.page_layout.orientation.detectors.base import Detector, Verdict, unknown

# Переменная окружения с путём к модели: детектор в реестре один, а какую модель он поднимет,
# решается снаружи — так в сравнении можно держать несколько чужих сетей по очереди.
MODEL_ENV = "ORIENTATION_ONNX_MODEL"

DEFAULT_BATCH = 16


def model_path() -> Path | None:
    raw = os.environ.get(MODEL_ENV, "").strip()
    return Path(raw) if raw else None


def onnx_available() -> bool:
    path = model_path()
    if path is None or not path.is_file() or not path.with_suffix(".json").is_file():
        return False
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


class OnnxOrientation:
    """Четырёхклассовый классификатор из ONNX-файла по описанию рядом с ним."""

    def __init__(self, batch_size: int = DEFAULT_BATCH) -> None:
        self._batch_size = batch_size
        self._session = None
        self._config: dict = {}

    def _load(self):
        if self._session is None:
            import onnxruntime

            path = model_path()
            if path is None:
                raise RuntimeError(f"не задан {MODEL_ENV}")
            self._config = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = onnxruntime.InferenceSession(str(path), providers=providers)
        return self._session

    def _prepare(self, images: Sequence["object"]) -> np.ndarray:
        import cv2

        width, height = self._config["size"]
        mean = np.array(self._config["mean"], dtype=np.float32)
        std = np.array(self._config["std"], dtype=np.float32)
        batch = []
        for image in images:
            array = cv2.resize(np.asarray(image), (width, height), interpolation=cv2.INTER_AREA)
            array = (array.astype(np.float32) / 255.0 - mean) / std
            batch.append(array)
        stacked = np.stack(batch)
        return stacked.transpose(0, 3, 1, 2) if self._config.get("layout", "NCHW") == "NCHW" else stacked

    def __call__(self, images: Sequence["object"]) -> list[Verdict]:
        if not images:
            return []
        try:
            session = self._load()
        except Exception as error:
            return [unknown(f"onnx: {error}") for _ in images]
        classes = self._config["classes"]
        # "state" — модель называет угол, НА КОТОРЫЙ картинка повёрнута; чтобы выпрямить,
        # надо повернуть на столько же в обратную сторону.
        invert = self._config.get("meaning", "correction") == "state"
        verdicts: list[Verdict] = []
        for start in range(0, len(images), self._batch_size):
            chunk = images[start : start + self._batch_size]
            try:
                logits = session.run(None, {session.get_inputs()[0].name: self._prepare(chunk)})[0]
            except Exception as error:
                verdicts.extend(unknown(f"onnx: {error}") for _ in chunk)
                continue
            for row in np.asarray(logits, dtype=np.float64):
                shifted = np.exp(row - row.max())
                probabilities = shifted / shifted.sum()
                index = int(probabilities.argmax())
                angle = int(classes[index]) % 360
                rotation = (-angle) % 360 if invert else angle
                verdicts.append(
                    Verdict(rotation, float(probabilities[index]), metrics={"prob": float(probabilities[index])})
                )
        return verdicts


ALGORITHM = Detector(
    name="onnx",
    summary=f"сторонний ONNX-классификатор из ${MODEL_ENV} (PP-LCNet, EfficientNetV2 и т.п.)",
    stage="gpu",
    make_batch=OnnxOrientation,
    available=onnx_available,
)
