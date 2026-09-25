"""Воркер Chronicling Germany (Digital History Bonn): разметка полосы и базовые линии строк в JSON.

Запускается питоном из venv модели, наш пакет не импортирует. Код модели — клон
``Chronicling-Germany-Code`` (пакет ``cgprocess``, ставится в тот же venv через ``pip install -e --no-deps``).

Два шага, как в их ``script/pipeline.sh``:

1. сегментация вёрстки ``cgprocess.layout_segmentation.predict`` (dhSegment) — отдельным процессом,
   пишет PAGE-XML с текстовыми областями, таблицами и линейками;
2. базовые линии ``cgprocess.baseline_detection.predict.BaselineEngine`` (U-Net в духе PERO) — в этом
   процессе; разбор карт идёт отдельно в каждой текстовой области, таблицы закрашиваются.

Аргументы: ``<png> <out.json> [--layout-model P] [--baseline-model P] [--layout-scale S]
[--layout-threshold T] [--bbox-threshold N] [--cpu] [--keep-tmp DIR]``.

Выход: ``{"lines": [{"baseline", "boundary", "centre": [], "height"}], "regions": [...], "meta": {...}}``,
координаты — в пикселях поданного изображения. Высота строки — сумма медианных высоты над и под базовой
линией из карт модели (то, что сама модель кладёт в многоугольник строки).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from threading import Thread
from typing import List, Tuple

# Корень клона и каталог весов: по умолчанию — рядом с venv модели.
ENGINES_ROOT = Path("/home/felix/Projects/mts_markup/line_axis_engines")
REPO_ROOT = ENGINES_ROOT / "src" / "chronicling_germany"
MODELS_DIR = ENGINES_ROOT / "models" / "chronicling" / "Chronicling-Germany-Dataset-main-models" / "models"
DEFAULT_LAYOUT_MODEL = MODELS_DIR / "layout_2025-05-14.pt"
DEFAULT_BASELINE_MODEL = MODELS_DIR / "baseline_2025-05-19.pt"

# Сколько раз повторять шаг при нехватке видеопамяти (GPU делят несколько процессов).
OOM_RETRIES = 5
OOM_SLEEP_SECONDS = 20


def join_threads(threads: List[Thread]) -> None:
    """Дожидается всех потоков списка (копия функции из их модуля OCR, см. ``install_kraken_stub``).

    Аргументы:
        threads: запущенные потоки.
    """
    for thread in threads:
        thread.join()


def install_kraken_stub() -> None:
    """Подменяет модуль ``cgprocess.OCR.LSTM.predict`` заглушкой до импорта предсказателя базовых линий.

    Их ``baseline_detection.predict`` берёт оттуда только ``join_threads``, а сам модуль тянет kraken,
    который в этот venv не ставился (конфликт pin-ов shapely). Заглушка даёт ту же функцию.
    """
    stub = types.ModuleType("cgprocess.OCR.LSTM.predict")
    stub.join_threads = join_threads
    sys.modules["cgprocess.OCR.LSTM.predict"] = stub


# Заглушка ставится до импорта их модулей; клон подключается и через sys.path (на случай запуска без -e).
sys.path.insert(0, str(REPO_ROOT / "src"))
install_kraken_stub()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from shapely.geometry import LineString, Polygon  # noqa: E402

from cgprocess.baseline_detection.predict import BaselineEngine, apply_polygon_mask, extract_layout  # noqa: E402
from cgprocess.baseline_detection.utils import load_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Разбирает командную строку воркера.

    Возвращает:
        Namespace с полями ``image`` (входная полоса), ``out`` (выходной JSON), путями к весам
        вёрстки и строк, масштабом и порогами сегментации вёрстки, флагом ``cpu`` и ``keep_tmp``
        (куда сохранить промежуточный PAGE-XML вёрстки; пусто — не сохранять).
    """
    parser = argparse.ArgumentParser(description="Chronicling Germany: вёрстка + базовые линии")
    parser.add_argument("image", help="входное изображение полосы (png/jpg)")
    parser.add_argument("out", help="куда записать JSON")
    parser.add_argument("--layout-model", default=str(DEFAULT_LAYOUT_MODEL), help="веса сегментации вёрстки")
    parser.add_argument("--baseline-model", default=str(DEFAULT_BASELINE_MODEL), help="веса базовых линий")
    # Значения по умолчанию — как в их pipeline.sh: -th 0.6 -s 0.5 -bt 100.
    parser.add_argument("--layout-scale", type=float, default=0.5, help="масштаб картинки на входе вёрстки")
    parser.add_argument("--layout-threshold", type=float, default=0.6, help="порог уверенности пикселя вёрстки")
    parser.add_argument("--bbox-threshold", type=int, default=100, help="минимальный размер рамки области")
    parser.add_argument("--cpu", action="store_true", help="считать на CPU")
    parser.add_argument("--keep-tmp", default="", help="каталог, куда скопировать PAGE-XML вёрстки")
    return parser.parse_args()


def run_layout(args: argparse.Namespace, work_dir: Path) -> Path:
    """Запускает их сегментацию вёрстки отдельным процессом и возвращает путь к PAGE-XML.

    Аргументы:
        args: разобранные аргументы воркера (веса, масштаб, пороги, флаг CPU).
        work_dir: временный каталог с единственной входной картинкой; XML ляжет в ``work_dir/page``.

    Возвращает:
        Путь к PAGE-XML с областями вёрстки (координаты уже в пикселях исходной картинки).
    """
    command = [
        sys.executable,
        "-m",
        "cgprocess.layout_segmentation.predict",
        "-d",
        f"{work_dir}/",
        "-m",
        args.layout_model,
        "-th",
        str(args.layout_threshold),
        "-a",
        "dh_segment",
        "-s",
        str(args.layout_scale),
        "-e",
        "-bt",
        str(args.bbox_threshold),
    ]
    env = dict(os.environ)
    if args.cpu:
        # Их MPPredictor выбирает устройство по числу видимых GPU: прячем их.
        env["CUDA_VISIBLE_DEVICES"] = ""
    xml_path = work_dir / "page" / f"{Path(args.image).stem}.xml"
    for attempt in range(OOM_RETRIES):
        # Запуск из корня клона: их код местами читает относительные пути.
        result = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        if result.returncode == 0 and xml_path.exists():
            return xml_path
        log = result.stdout[-3000:] + result.stderr[-3000:]
        if "out of memory" in log.lower() and attempt + 1 < OOM_RETRIES:
            time.sleep(OOM_SLEEP_SECONDS)
            continue
        raise RuntimeError(f"сегментация вёрстки упала (код {result.returncode}):\n{log}")
    raise RuntimeError("сегментация вёрстки: исчерпаны повторы после нехватки видеопамяти")


class KeepHeightsEngine(BaselineEngine):
    """Движок базовых линий, который кладёт в выход ещё и высоты строк, отброшенные в их ``predict``.

    ``predict`` отдаёт пару (не используется, список строк по областям), где строка —
    ``(baseline Nx2, boundary Mx2, (высота над, высота под))``.
    """

    def postprocess_per_region(self, baseline_lst, prediction, roi, textline_lst) -> None:
        """Разбор карт в одной области; повторяет их метод, только добавляет высоты.

        Аргументы:
            baseline_lst: общий список (по областям) — сюда кладётся список кортежей строк
                (так устроен их потоковый разбор, вернуть значение из потока нельзя).
            prediction: карты модели (высота над, под, базовая линия, концы) в размере картинки.
            roi: многоугольник текстовой области, вне которого карты обнуляются.
            textline_lst: не используется (сигнатура родителя).
        """
        mask_map, offset = apply_polygon_mask(prediction.clone(), roi)
        b_list, h_list, t_list = self.parse(mask_map.permute(1, 2, 0).numpy())
        region_lines = []
        for baseline, heights, textline in zip(b_list, h_list, t_list):
            # То же упрощение с допуском 1 пиксель, что и в их коде перед записью в XML.
            line = LineString(baseline + offset).simplify(tolerance=1)
            poly = Polygon(textline + offset).buffer(0).simplify(tolerance=1)
            boundary = np.asarray(poly.exterior.coords) if poly.geom_type == "Polygon" else np.zeros((0, 2))
            region_lines.append((np.asarray(line.coords), boundary, (float(heights[0]), float(heights[1]))))
        baseline_lst.append(region_lines)


def run_baselines(args: argparse.Namespace, xml_path: Path) -> Tuple[list, list, str]:
    """Считает базовые линии по картинке и вёрстке.

    Аргументы:
        args: разобранные аргументы воркера (путь к картинке, веса, флаг CPU).
        xml_path: PAGE-XML вёрстки от ``run_layout``.

    Возвращает:
        Кортеж: список строк по областям (см. ``KeepHeightsEngine``), список многоугольников текстовых
        областей (Nx2) и имя устройства (``cuda`` / ``cpu``).
    """
    _, text_regions = extract_layout(str(xml_path))
    for attempt in range(OOM_RETRIES):
        try:
            engine = KeepHeightsEngine(model_name=args.baseline_model, cuda=-1 if args.cpu else 0, thread_count=4)
            image = load_image(args.image)
            with torch.no_grad():
                _, lines_by_region = engine.predict(image, str(xml_path))
            return lines_by_region, text_regions, engine.device.type
        except torch.cuda.OutOfMemoryError:
            engine = None
            torch.cuda.empty_cache()
            if attempt + 1 == OOM_RETRIES:
                raise
            time.sleep(OOM_SLEEP_SECONDS)
    raise RuntimeError("недостижимо")


def main() -> None:
    """Вёрстка → базовые линии → JSON в формате воркеров ``text_blocks.engines``."""
    args = parse_args()
    work_dir = Path(tempfile.mkdtemp(prefix="chronicling_"))
    try:
        # Во временном каталоге только наша картинка: их предсказатель берёт все png/jpg каталога.
        shutil.copy(args.image, work_dir / Path(args.image).name)
        xml_path = run_layout(args, work_dir)
        if args.keep_tmp:
            os.makedirs(args.keep_tmp, exist_ok=True)
            shutil.copy(xml_path, Path(args.keep_tmp) / xml_path.name)
        lines_by_region, text_regions, device = run_baselines(args, xml_path)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    # Перекладываем родной выход в общий JSON: области по порядку их разбора, строки внутри области.
    lines = []
    for region_lines in lines_by_region:
        for baseline, boundary, (ascender, descender) in region_lines:
            lines.append(
                {
                    "baseline": [[float(x), float(y)] for x, y in baseline],
                    "boundary": [[float(x), float(y)] for x, y in boundary],
                    "centre": [],
                    "height": ascender + descender,
                }
            )
    regions = [[[float(x), float(y)] for x, y in region] for region in text_regions if len(region) >= 3]
    meta = {
        "model": f"chronicling-germany {Path(args.layout_model).name} + {Path(args.baseline_model).name}",
        "device": device,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"lines": lines, "regions": regions, "meta": meta}, handle)


if __name__ == "__main__":
    main()
