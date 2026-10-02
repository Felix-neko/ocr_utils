"""Воркер Doc-UFCN (Teklia): строки текста полигонами в JSON. Запускается ЧУЖИМ питоном, наш пакет не импортирует.

Аргументы: ``<png> <out.json> [--model generic-historical-line] [--input-size 768] [--min-cc 50] [--cpu]``.
Выход: ``{"lines": [{"baseline": [], "boundary": [[x, y], …], "centre": [], "height": 0}], "regions": [],
"meta": {...}}`` в пикселях поданного изображения. Модель — семантическая сегментация «фон / строка»,
полигон строки — внешний контур связной компоненты маски; осевой линии и базовой линии модель не даёт.
"""

import argparse
import json
import os
import sys
import time

# Кэш весов HuggingFace и torch держим в каталоге моделей стенда, а не в ~/.cache.
MODELS_DIR = "/mnt/hotstore/scan_processing/mts_markup/line_axis_engines/models/docufcn"
os.environ.setdefault("XDG_CACHE_HOME", MODELS_DIR)
os.environ.setdefault("HF_HOME", os.path.join(MODELS_DIR, "hf"))
# Чекпойнт сохранён старым torch целиком (не только тензоры): torch>=2.6 по умолчанию грузит weights_only.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from doc_ufcn import models  # noqa: E402
from doc_ufcn.main import DocUFCN  # noqa: E402

# Сторона, на которой модель обучалась (большая сторона картинки, пикселей).
TRAIN_INPUT_SIZE = 768


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    :param argv: аргументы без имени скрипта.
    :return: пространство имён: ``png`` — путь к странице, ``out`` — куда писать JSON, ``model`` — имя модели
        Teklia без префикса ``doc-ufcn-``, ``input_size`` — большая сторона входа сети в пикселях (``0`` —
        родная из ``parameters.yml``), ``min_cc`` — порог площади компоненты в пикселях входа сети при родной
        стороне (масштабируется вместе со стороной), ``cpu`` — считать на процессоре.
    """
    parser = argparse.ArgumentParser(description="Doc-UFCN: полигоны строк страницы в JSON")
    parser.add_argument("png")
    parser.add_argument("out")
    parser.add_argument("--model", default="generic-historical-line")
    parser.add_argument("--input-size", type=int, default=0)
    parser.add_argument("--min-cc", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def run_model(model: DocUFCN, image: np.ndarray, min_cc: int) -> list[dict]:
    """Прогнать сеть по странице, при нехватке видеопамяти подождать и повторить.

    :param model: загруженная модель Doc-UFCN.
    :param image: страница RGB, ``uint8``, ``H×W×3``.
    :param min_cc: порог площади связной компоненты в пикселях входа сети.
    :return: список словарей Doc-UFCN ``{"confidence", "polygon"}`` класса «строка» в пикселях страницы.
    """
    for attempt in range(5):
        try:
            polygons, _, _, _ = model.predict(image, min_cc=min_cc)
            return polygons[1]
        except torch.OutOfMemoryError:
            # Видеокарту делят с другими процессами: освобождаем кэш и ждём.
            torch.cuda.empty_cache()
            time.sleep(10 * (attempt + 1))
    polygons, _, _, _ = model.predict(image, min_cc=min_cc)
    return polygons[1]


def main() -> None:
    """Точка входа: загрузить модель, сегментировать страницу, записать полигоны строк в JSON."""
    args = parse_args(sys.argv[1:])
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    # Веса и параметры нормировки — из репозитория Teklia на HuggingFace (кэш в MODELS_DIR).
    model_path, parameters = models.download_model(args.model)
    native_size = int(parameters["input_size"])
    input_size = args.input_size or native_size
    # Порог площади задан в пикселях входа сети при родной стороне: при большем входе растёт квадратично.
    min_cc_native = args.min_cc or int(parameters.get("min_cc", 50))
    min_cc = max(1, int(round(min_cc_native * (input_size / native_size) ** 2)))

    model = DocUFCN(len(parameters["classes"]), input_size, device)
    model.load(model_path, parameters["mean"], parameters["std"])

    # Страница серая: сеть ждёт RGB, поэтому дублируем канал.
    gray = cv2.imread(args.png, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    started = time.time()
    found = run_model(model, image, min_cc)
    seconds = time.time() - started

    # Полигоны Doc-UFCN уже пересчитаны в пиксели страницы (пары (x, y)); базовой и осевой линий нет.
    lines = []
    for item in found:
        boundary = [[float(x), float(y)] for x, y in item["polygon"]]
        if len(boundary) < 3:
            continue
        lines.append(
            {"baseline": [], "boundary": boundary, "centre": [], "height": 0.0, "confidence": float(item["confidence"])}
        )

    meta = {
        "model": f"doc-ufcn-{args.model}",
        "device": device,
        "params": {"input_size": input_size, "native_input_size": native_size, "min_cc": min_cc},
        "seconds": round(seconds, 2),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"lines": lines, "regions": [], "meta": meta}, handle)


if __name__ == "__main__":
    main()
