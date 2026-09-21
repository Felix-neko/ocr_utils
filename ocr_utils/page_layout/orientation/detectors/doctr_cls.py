"""Классификатор ориентации страницы из docTR (``mobilenet_v3_small_page_orientation``).

Лёгкая свёрточная сеть на четыре класса, обученная Mindee на документах. Из готовых
классификаторов этого семейства (второй — PaddleOCR PP-LCNet_x1_0_doc_ori) взята именно
она: docTR уже в зависимостях проекта, веса ставятся сами, а PP-LCNet выложен только
в формате paddle inference, и ради него пришлось бы тянуть либо paddle, либо конверсию.

Сеть даёт сразу все четыре класса и стоит копейки (0.02 с на полосу), но она не видела
полос советского отраслевого журнала: на разведке по четырём известным боковым полосам
угадала три, а на четырёх обычных один раз сказала «180» с уверенностью 0.53. То есть
как единственный судья не годится, а как голос в компании — вполне.

Модель грузится лениво и ровно одна, в РОДИТЕЛЬСКОМ процессе: видеопамять одна на всех,
и держать копию сети в каждом из шестнадцати воркеров нельзя (см. CLAUDE.md).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ocr_utils.page_layout.orientation.detectors.base import Detector, Verdict, unknown

# Классы модели: [0, -90, 180, 90], углы против часовой стрелки. Возвращаемый угол — это
# поворот, КОТОРЫМ страницу надо развернуть, чтобы стало прямо (docTR так и выпрямляет
# страницу перед детекцией). В нашу валюту он переводится сменой знака: по часовой это
# минус против часовой. Соглашение проверяется командой validate, а не берётся на веру.
CCW_TO_CW = -1

DEFAULT_BATCH = 32


def doctr_available() -> bool:
    try:
        import doctr.models  # noqa: F401
    except Exception:
        return False
    return True


class DoctrOrientation:
    """Пачка картинок на вход, вердикты на выход. Веса поднимаются при первом вызове."""

    def __init__(self, batch_size: int = DEFAULT_BATCH) -> None:
        self._batch_size = batch_size
        self._predictor = None

    def _load(self):
        if self._predictor is None:
            from doctr.models import page_orientation_predictor

            self._predictor = page_orientation_predictor(pretrained=True)
        return self._predictor

    def __call__(self, images: Sequence["object"]) -> list[Verdict]:
        if not images:
            return []
        predictor = self._load()
        verdicts: list[Verdict] = []
        for start in range(0, len(images), self._batch_size):
            chunk = [np.asarray(image) for image in images[start : start + self._batch_size]]
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
                verdicts.append(Verdict(rotation, float(probability), metrics={"prob": float(probability)}))
        return verdicts


ALGORITHM = Detector(
    name="doctr",
    summary="docTR mobilenet_v3_small_page_orientation, четыре класса (GPU)",
    stage="gpu",
    make_batch=DoctrOrientation,
    available=doctr_available,
)
