"""Воркер PaddleOCR: детектор текстовых строк PP-OCRv5_server_det (DB) в JSON. Запускается ЧУЖИМ питоном, наш пакет не импортирует.

Только детекция, без распознавания. Постобработка DB переключена на ``box_type="poly"``: каждая найденная
область отдаётся многоугольником по контуру карты вероятности (кривые строки — кривыми многоугольниками),
а не повёрнутым четырёхугольником.

Аргументы: ``<png> <out.json> [--model ИМЯ] [--device gpu:0|cpu] [--limit-side-len N] [--limit-type max|min]``.

* ``png`` — страница (оттенки серого, 300 dpi); переводится в три канала, как ждёт модель.
* ``out.json`` — куда записать результат.
* ``--model`` — имя модели PaddleX (по умолчанию ``PP-OCRv5_server_det``; есть и ``PP-OCRv6_*_det``).
* ``--device`` — ``gpu:0`` (по умолчанию) или ``cpu``; при нехватке видеопамяти воркер сам уходит на CPU.
* ``--limit-side-len`` / ``--limit-type`` — правило масштабирования входа. По умолчанию ``4000`` / ``max``:
  страница до 4000 px по длинной стороне подаётся в полном разрешении (штатно модель ужимает до 960).

Выход:
``{"lines": [{"baseline": [], "boundary": [[x, y], …], "centre": [], "height": 0, "confidence": c}],
"regions": [], "meta": {"model": …, "device": "cuda|cpu", "params": {…}, "seconds": t}}``
в пикселях поданного изображения. Базовых линий и высот модель не даёт — поля пустые.
Кэш весов — ``$PADDLE_PDX_CACHE_HOME`` (по умолчанию ``…/mts_markup/line_axis_engines/models/paddle``).
"""

import argparse
import json
import os
import sys
import time

# Кэш весов и проверку источника моделей надо задать до импорта paddleocr: paddlex читает их при импорте.
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", "/home/felix/Projects/mts_markup/line_axis_engines/models/paddle")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Разобрать командную строку.

    :return: пространство имён с полями ``png``, ``out``, ``model``, ``device``, ``limit_side_len``, ``limit_type``.
    """
    parser = argparse.ArgumentParser(description="PaddleOCR: детекция строк страницы в JSON")
    parser.add_argument("png", help="страница, PNG")
    parser.add_argument("out", help="выходной JSON")
    parser.add_argument("--model", default="PP-OCRv5_server_det", help="имя модели детекции PaddleX")
    parser.add_argument("--device", default="gpu:0", help="gpu:0 или cpu")
    parser.add_argument("--limit-side-len", type=int, default=4000, help="предел стороны при масштабировании")
    parser.add_argument("--limit-type", default="max", choices=["max", "min"], help="к какой стороне предел")
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: str):
    """Собрать детектор и переключить постобработку DB на многоугольники.

    :param args: разобранные аргументы (имя модели и правило масштабирования).
    :param device: устройство paddle — ``gpu:0`` или ``cpu``.
    :return: объект ``paddleocr.TextDetection``, готовый к ``predict``.
    """
    from paddleocr import TextDetection

    model = TextDetection(
        model_name=args.model, device=device, limit_side_len=args.limit_side_len, limit_type=args.limit_type
    )
    # Штатный API PaddleOCR 3.x не даёт задать box_type для TextDetection; он берётся из конфига модели
    # (там его нет → «quad»). Переключаем прямо у постпроцессора внутреннего предиктора PaddleX.
    predictor = model.paddlex_predictor
    post_op = getattr(predictor, "post_op", None)
    if post_op is None:
        post_op = getattr(getattr(predictor, "_predictor", None), "post_op", None)
    if post_op is None:
        raise RuntimeError("не найден постпроцессор DB у предиктора — не могу включить box_type=poly")
    post_op.box_type = "poly"
    return model


def run(args: argparse.Namespace, image: np.ndarray, device: str) -> tuple[list, list, float]:
    """Прогнать детектор по странице.

    :param args: разобранные аргументы.
    :param image: страница, BGR uint8 (H, W, 3).
    :param device: устройство paddle.
    :return: кортеж (многоугольники, оценки уверенности, секунды на predict без загрузки модели).
    """
    model = build_model(args, device)
    start = time.perf_counter()
    result = list(model.predict(image, batch_size=1))[0]
    seconds = time.perf_counter() - start
    return list(result["dt_polys"]), list(result["dt_scores"]), seconds


def main() -> None:
    """Точка входа: прочитать страницу, найти строки, записать JSON."""
    args = parse_args()
    # Серую страницу размножаем в три канала (BGR — порядок, которого ждёт конфиг модели).
    gray = np.asarray(Image.open(args.png).convert("L"))
    image = np.ascontiguousarray(np.stack([gray] * 3, axis=-1))

    device = args.device
    try:
        polys, scores, seconds = run(args, image, device)
    except (MemoryError, RuntimeError) as error:
        # Видеопамять делим с другими процессами: при нехватке — повтор на CPU.
        if device == "cpu" or "memory" not in str(error).lower() and "alloc" not in str(error).lower():
            raise
        print(f"нехватка видеопамяти ({error}); перехожу на CPU", file=sys.stderr)
        device = "cpu"
        polys, scores, seconds = run(args, image, device)

    lines = []
    for poly, score in zip(polys, scores):
        boundary = [[float(x), float(y)] for x, y in np.asarray(poly).reshape(-1, 2)]
        lines.append({"baseline": [], "boundary": boundary, "centre": [], "height": 0, "confidence": float(score)})
    meta = {
        "model": args.model,
        "device": "cpu" if device == "cpu" else "cuda",
        "params": {"box_type": "poly", "limit_side_len": args.limit_side_len, "limit_type": args.limit_type},
        "seconds": round(seconds, 3),
        "image_size": [int(gray.shape[1]), int(gray.shape[0])],
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"lines": lines, "regions": [], "meta": meta}, handle)


if __name__ == "__main__":
    main()
