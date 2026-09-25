"""Воркер Orli (плагин kraken 7): базовые линии строк в порядке чтения в JSON. Запускается ЧУЖИМ питоном.

Наш пакет не импортирует; окружение — ``mts_markup/line_axis_engines/orli`` (kraken 7 + orli + torch cu12x/cu13x).

Аргументы: ``<png> <out.json> [--model PATH] [--device cuda|cpu] [--polygonize] [--max-lines N]``.

* ``png`` — страница (любой режим PIL; переводится в RGB, как требует нормировка ImageNet в модели).
* ``out.json`` — куда писать результат.
* ``--model`` — веса ``orli_base.safetensors`` (Zenodo 10.5281/zenodo.20558179).
* ``--device`` — ``cuda`` (по умолчанию, если есть) или ``cpu``.
* ``--polygonize`` — дополнительно построить контуры строк штатной функцией kraken
  ``calculate_polygonal_environment`` (опция самой модели; по умолчанию выключено — родной выход Orli
  только базовые линии).
* ``--max-lines`` — предел числа строк авторегрессионного декодера (штатно 768).
* ``--precision`` — точность Fabric, по умолчанию ``bf16-true`` (см. ниже).

Модель внутри растягивает страницу до фиксированного размера (``image_size`` из метаданных весов,
у базовой модели 1920×1440, без сохранения пропорций) и отдаёт кривые в долях [0, 1], которые сама
умножает на ширину и высоту исходного изображения — пересчитывать координаты не нужно.
Работает только в bfloat16: в fp32 генерация «убегает» (упирается в ``--max-lines``, строки
кучкуются вверху страницы). ``bf16-mixed`` через этот API фактически считает в fp32 (Fabric лишь
объявляет autocast, но не оборачивает им ``predict``), поэтому по умолчанию ``bf16-true``.
Нужен релиз ``orli==0.0.2``: HEAD репозитория (после «fp32 curve predictor») с базовыми весами
тоже убегает.

Выход: ``{"lines": [{"baseline": [[x, y], …], "boundary": [[x, y], …] или [], "centre": [],
"height": 0}], "regions": [], "meta": {"model": …, "device": …, "seconds": …}}`` — в пикселях поданного
изображения, строки в порядке чтения, который выдала модель. ``regions`` пуст: Orli областей не даёт.
"""

import argparse
import json
import time

import torch
from PIL import Image

from kraken.models import load_models
from kraken.tasks.segmentation import SegmentationTaskModel
from orli.configs import OrliSegmentationInferenceConfig

DEFAULT_MODEL = "/home/felix/Projects/mts_markup/line_axis_engines/models/orli/orli_base.safetensors"


def parse_args() -> argparse.Namespace:
    """Разобрать командную строку.

    Возвращает пространство имён с полями ``image``, ``out``, ``model``, ``device``, ``polygonize``,
    ``max_lines``, ``precision`` (смысл — в докстринге модуля).
    """
    parser = argparse.ArgumentParser(description="Orli: базовые линии строк страницы в JSON")
    parser.add_argument("image")
    parser.add_argument("out")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--polygonize", action="store_true")
    parser.add_argument("--max-lines", type=int, default=768)
    parser.add_argument("--precision", default="bf16-true")
    return parser.parse_args()


def points(polyline) -> list[list[float]]:
    """Перевести ломаную kraken (кортежи или None) в список ``[[x, y], …]`` из float.

    Аргумент ``polyline`` — последовательность пар координат (кортежи или numpy-массив) или None. Возвращает список пар
    (пустой, если ломаной нет).
    """
    # Контур из calculate_polygonal_environment приходит numpy-массивом: у него нет однозначной
    # истинности, поэтому проверяем на None явно, а не через «or».
    if polyline is None:
        return []
    return [[float(x), float(y)] for x, y in polyline]


def main() -> None:
    """Загрузить модель, сегментировать страницу и записать JSON (формат — в докстринге модуля)."""
    args = parse_args()
    # Модель нормирует три канала по ImageNet, поэтому серый скан переводим в RGB.
    image = Image.open(args.image).convert("RGB")
    # Точность bf16 обязательна для базовой модели; устройство задаём явно, без «auto».
    config = OrliSegmentationInferenceConfig(
        precision=args.precision,
        accelerator="gpu" if args.device == "cuda" else "cpu",
        device=1,
        max_predicted_lines=args.max_lines,
        polygonize=args.polygonize,
    )
    started = time.perf_counter()
    task = SegmentationTaskModel(load_models(args.model, tasks=["segmentation"]))
    loaded = time.perf_counter()
    segmentation = task.predict(image, config)
    finished = time.perf_counter()

    # Родной выход Orli — базовая линия; контур есть только при --polygonize.
    lines = []
    for line in segmentation.lines:
        baseline = points(line.baseline)
        if baseline:
            lines.append({"baseline": baseline, "boundary": points(line.boundary), "centre": [], "height": 0})
    meta = {
        "model": f"orli ({args.model.rsplit('/', 1)[-1]})",
        "device": args.device,
        "polygonize": args.polygonize,
        "precision": args.precision,
        "hit_max_lines": len(segmentation.lines) >= args.max_lines,
        "load_seconds": round(loaded - started, 2),
        "seconds": round(finished - loaded, 2),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"lines": lines, "regions": [], "meta": meta}, handle)


if __name__ == "__main__":
    main()
