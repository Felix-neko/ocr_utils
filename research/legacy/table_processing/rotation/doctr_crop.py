"""Классификатор ориентации ВЫРЕЗКИ из docTR (``mobilenet_v3_small_crop_orientation``).

Готовая свёрточная сеть Mindee на четыре класса, обученная как раз на вырезках строк, а
не на страницах. Из семейства готовых классификаторов взята она по той же причине, что и
в пакете ориентации полос: docTR уже в зависимостях проекта, веса ставятся сами, а
PP-LCNet выложен только в формате paddle inference.

ЧТО ОНА ЗДЕСЬ ДЕЛАЕТ, чего не делают дешёвые меры: называет СТОРОНУ, не читая текста.
Ось надёжно даёт форма буквы, сторону — арбитр на tesseract, но арбитр стоит 0.3 с на
ячейку, а сеть на GPU считает пачку ячеек разом.

Модель грузится лениво и ровно одна, в РОДИТЕЛЬСКОМ процессе: видеопамять одна на всех
(см. CLAUDE.md).
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from research.legacy.table_processing.rotation.base import BatchCellDetector, CellCrop, CellDetector, Verdict, unknown

# Классы модели: [0, -90, 180, 90] — углы ПРОТИВ часовой, на которые надо повернуть вырезку,
# чтобы стало прямо. В валюту пакета переводятся сменой знака.
CCW_TO_CW = -1

# Вход сети: 256x256. Ячейку кладём в квадрат целиком, а не режем — форма ячейки сама по
# себе намекает на ориентацию, и терять её незачем.
INPUT_SIDE = 256

DEFAULT_BATCH = 32


def doctr_available() -> bool:
    try:
        import doctr.models  # noqa: F401
    except Exception:
        return False
    return True


def _square(gray: np.ndarray) -> np.ndarray:
    """Вырезка, вписанная в квадрат стороной ``INPUT_SIDE``, поля — цветом бумаги."""
    height, width = gray.shape[:2]
    scale = INPUT_SIDE / max(height, width, 1)
    resized = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))), cv2.INTER_AREA)
    canvas = np.full((INPUT_SIDE, INPUT_SIDE), 255, np.uint8)
    top = (INPUT_SIDE - resized.shape[0]) // 2
    left = (INPUT_SIDE - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


class DoctrCellOrientation(BatchCellDetector):
    def __init__(self, batch_size: int = DEFAULT_BATCH) -> None:
        self._batch_size = batch_size
        self._predictor = None

    def _load(self):
        if self._predictor is None:
            from doctr.models import crop_orientation_predictor

            self._predictor = crop_orientation_predictor(pretrained=True)
        return self._predictor

    def __call__(self, crops: Sequence[CellCrop]) -> list[Verdict]:
        if not crops:
            return []
        predictor = self._load()
        verdicts: list[Verdict] = []
        for start in range(0, len(crops), self._batch_size):
            chunk = [_square(crop.gray) for crop in crops[start : start + self._batch_size]]
            try:
                _, angles, probabilities = predictor(chunk)
            except Exception as error:  # сеть не должна ронять прогон целиком
                verdicts.extend(unknown(f"docTR: {error}") for _ in chunk)
                continue
            for angle, probability in zip(angles, probabilities):
                rotation = (int(angle) * CCW_TO_CW) % 360
                if rotation % 90:
                    verdicts.append(unknown(f"docTR вернул угол {angle}"))
                    continue
                if rotation == 180:
                    # Перевёрнутая ячейка в таблице не встречается; такой ответ означает,
                    # что сеть не разобралась, а не что ячейка вверх ногами.
                    verdicts.append(unknown("docTR сказал 180"))
                    continue
                verdicts.append(Verdict(rotation, float(probability), metrics={"prob": float(probability)}))
        return verdicts


ALGORITHM = CellDetector(
    name="doctr",
    summary="docTR mobilenet_v3_small_crop_orientation, четыре класса (GPU)",
    stage="gpu",
    gives_sign=True,
    make_batch=DoctrCellOrientation,
    available=doctr_available,
)
