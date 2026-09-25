"""Воркер surya: детектор строк ``DetectionPredictor`` в JSON. Наш пакет не импортирует.

Запускается питоном основного окружения проекта (там стоит ``surya-ocr``), но отдельным процессом:
так видеопамять модели освобождается сразу после страницы, как у остальных чужих движков.

Аргументы: ``<png> <out.json>``. Выход в пикселях поданного изображения:
``{"lines": [{"boundary": [[x, y], …], "confidence": c}], "regions": [], "meta": {…}}``.
Surya отдаёт на строку четырёхугольник, поэтому кривизны у её строк нет: ось — середина
четырёхугольника, прямая. Движок включён в сравнение как контроль «без кривизны».
"""

import json
import sys

import torch
from PIL import Image

from surya.detection import DetectionPredictor


def main() -> None:
    png, out_path = sys.argv[1], sys.argv[2]
    image = Image.open(png).convert("RGB")
    predictor = DetectionPredictor()
    # Surya сама режет высокую страницу на полосы и возвращает координаты исходной картинки.
    result = predictor([image])[0]
    lines = []
    for box in result.bboxes:
        lines.append(
            {
                "baseline": [],
                "boundary": [[float(x), float(y)] for x, y in box.polygon],
                "centre": [],
                "height": 0.0,
                "confidence": float(box.confidence) if box.confidence is not None else None,
            }
        )
    meta = {"model": "surya DetectionPredictor", "device": "cuda" if torch.cuda.is_available() else "cpu"}
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"lines": lines, "regions": [], "meta": meta}, handle)


if __name__ == "__main__":
    main()
