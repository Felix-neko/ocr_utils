"""Воркер Laypa (Loghi, KNAW HuC): базовые линии строк через docker-образы Laypa и loghi-tooling, разбор PAGE-XML.

Запускается любым питоном 3.9+ (только стандартная библиотека), наш пакет не импортирует. Цепочка — та же, что в
боевом конвейере Loghi (``scripts/pipeline.sh`` Laypa): образ ``loghi/docker.laypa`` строит попиксельную маску
базовых линий (``inference.py``), образ ``loghi/docker.loghi-tooling`` превращает её в PAGE-XML с базовыми линиями
и контурами строк (``MinionExtractBaselines``). Laypa внутри уменьшает страницу (``SCALING_TEST`` конфига), но маску
возвращает в размере входа, так что координаты PAGE-XML уже в пикселях поданного изображения.

Аргументы: ``<png> <out.json> [каталог модели] [--gpu]``.

* ``<png>`` — страница (градации серого или цвет), лучше 300 dpi: этот dpi Laypa принимает по умолчанию;
* ``<out.json>`` — куда записать результат;
* ``[каталог модели]`` — каталог с ``config.yaml`` и ``model_best_mIoU.pth`` (по умолчанию — общая модель
  ``baseline2`` в ``/mnt/hotstore/scan_processing/mts_markup/line_axis_engines/models/laypa/baseline2``);
* ``--gpu`` — передать докеру ``--gpus all``; без флага модель считается на CPU (у Rancher Desktop проброса
  видеокарты нет, а на CPU страница идёт за несколько секунд).

Рабочие файлы кладутся во временный каталог под ``LAYPA_WORK_ROOT`` (по умолчанию
``/mnt/hotstore/scan_processing/mts_markup/line_axis_engines/laypa/tmp``): его должна видеть виртуальная машина докера.

Выход: ``{"lines": [{"baseline": [[x, y], …], "boundary": [[x, y], …], "centre": [], "height": 0,
"x_height": h}], "regions": [[[x, y], …]], "meta": {"model": …, "device": "cpu|cuda", "seconds": …}}``.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# Каталоги по умолчанию: модель и корень для временных каталогов (оба — на SSD, видимом докеру).
ENGINES_ROOT = Path("/mnt/hotstore/scan_processing/mts_markup/line_axis_engines")
DEFAULT_MODEL_DIR = ENGINES_ROOT / "models" / "laypa" / "baseline2"
DEFAULT_WORK_ROOT = ENGINES_ROOT / "laypa" / "tmp"

# Образы Loghi: сеть Laypa и Java-утилиты извлечения линий из маски.
LAYPA_IMAGE = "loghi/docker.laypa:latest"
TOOLING_IMAGE = "loghi/docker.loghi-tooling:latest"
MINION = "/src/loghi-tooling/minions/target/appassembler/bin/MinionExtractBaselines"

# Точка «x,y» в атрибуте points PAGE-XML (тег берём по суффиксу, издание схемы не важно).
POINTS = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")


def points_of(node) -> list[list[float]]:
    """Разобрать атрибут ``points`` узла ``Coords``/``Baseline``.

    Аргументы:
        node: элемент XML или ``None``.

    Возвращает:
        список точек ``[[x, y], …]``; пустой, если узла нет.
    """
    if node is None:
        return []
    return [[float(x), float(y)] for x, y in POINTS.findall(node.get("points", ""))]


def child(node, tag: str):
    """Первый потомок узла с именем тега ``tag`` без учёта пространства имён.

    Аргументы:
        node: родительский элемент XML;
        tag: локальное имя тега (``Coords``, ``Baseline``, ``TextStyle``).

    Возвращает:
        найденный элемент или ``None``.
    """
    for item in node:
        if item.tag.rsplit("}", 1)[-1] == tag:
            return item
    return None


def docker_binary() -> str:
    """Путь к клиенту docker: из PATH, иначе — клиент Rancher Desktop в ``~/.rd/bin``.

    Возвращает:
        путь к исполняемому файлу docker.
    """
    found = shutil.which("docker")
    if found:
        return found
    return str(Path.home() / ".rd" / "bin" / "docker")


def run_step(command: list[str], name: str) -> None:
    """Выполнить шаг конвейера в докере и упасть с хвостом лога, если он не удался.

    Аргументы:
        command: полная команда ``docker run …``;
        name: имя шага для сообщения об ошибке.
    """
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-10:]
        raise SystemExit(f"Laypa: шаг «{name}» упал (код {result.returncode}): " + " | ".join(tail))


def parse_page(xml_path: Path) -> tuple[list[dict], list[list[list[float]]]]:
    """Разобрать PAGE-XML Loghi в строки и регионы.

    Аргументы:
        xml_path: путь к PAGE-XML после ``MinionExtractBaselines``.

    Возвращает:
        кортеж ``(lines, regions)``: строки со своими ``baseline``/``boundary``/``x_height`` и контуры регионов.
    """
    root = ET.parse(xml_path).getroot()
    lines, regions = [], []
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "TextRegion":
            coords = points_of(child(node, "Coords"))
            if len(coords) >= 3:
                regions.append(coords)
        elif tag == "TextLine":
            baseline = points_of(child(node, "Baseline"))
            if len(baseline) < 2:
                continue
            boundary = points_of(child(node, "Coords"))
            # Высоту строки Loghi не даёт, только оценку высоты строчных букв в TextStyle.
            style = child(node, "TextStyle")
            x_height = float(style.get("xHeight", 0)) if style is not None else 0.0
            lines.append(
                {
                    "baseline": baseline,
                    "boundary": boundary if len(boundary) >= 3 else [],
                    "centre": [],
                    "height": 0,
                    "x_height": x_height,
                }
            )
    return lines, regions


def main() -> None:
    """Прогнать одну страницу через Laypa + MinionExtractBaselines и записать JSON."""
    # Разбор аргументов: позиционные — png, out.json, необязательный каталог модели; флаг --gpu.
    args = [item for item in sys.argv[1:] if item != "--gpu"]
    use_gpu = "--gpu" in sys.argv[1:]
    png, out_path = Path(args[0]).resolve(), Path(args[1])
    model_dir = Path(args[2]).resolve() if len(args) > 2 else DEFAULT_MODEL_DIR
    work_root = Path(os.environ.get("LAYPA_WORK_ROOT", DEFAULT_WORK_ROOT))
    work_root.mkdir(parents=True, exist_ok=True)
    docker = docker_binary()
    started = time.time()

    with tempfile.TemporaryDirectory(prefix="laypa_", dir=work_root) as folder:
        work = Path(folder)
        in_dir, out_dir = work / "in", work / "out"
        in_dir.mkdir()
        # Laypa читает каталог целиком, поэтому кладём туда копию одной страницы.
        shutil.copy2(png, in_dir / png.name)
        page_dir = out_dir / "page"
        mount = ["-v", f"{work}:{work}", "-v", f"{model_dir}:/model:ro"]

        # Шаг 1: маска базовых линий. Веса — через TEST.WEIGHTS, как в pipeline.sh; устройство — CPU или CUDA.
        device = "cuda" if use_gpu else "cpu"
        gpu_flags = ["--gpus", "all"] if use_gpu else []
        run_step(
            [docker, "run", "--rm", "--shm-size", "4G", *gpu_flags, *mount, LAYPA_IMAGE]
            + ["python", "inference.py", "-c", "/model/config.yaml", "-i", str(in_dir), "-o", str(out_dir)]
            + ["--opts", "MODEL.WEIGHTS", "", "TEST.WEIGHTS", "/model/model_best_mIoU.pth", "MODEL.DEVICE", device],
            "inference",
        )

        # Шаг 2: маска → базовые линии и контуры строк в PAGE-XML (Java из loghi-tooling).
        # Без -input_path_image Minion ищет исходник по несуществующему пути и падает.
        run_step(
            [docker, "run", "--rm", *mount, TOOLING_IMAGE, MINION]
            + ["-input_path_png", f"{page_dir}/", "-input_path_page", f"{page_dir}/"]
            + ["-output_path_page", f"{page_dir}/", "-input_path_image", f"{in_dir}/"]
            + ["-as_single_region", "-recalculate_textline_contours_from_baselines"]
            + ["-laypaconfig", "/model/config.yaml"],
            "MinionExtractBaselines",
        )

        xml_path = page_dir / (png.stem + ".xml")
        if not xml_path.exists():
            raise SystemExit(f"Laypa не отдал PAGE-XML: {xml_path}")
        lines, regions = parse_page(xml_path)

    meta = {"model": f"laypa:{model_dir.name}", "device": device, "seconds": round(time.time() - started, 2)}
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"lines": lines, "regions": regions, "meta": meta}, handle)


if __name__ == "__main__":
    main()
