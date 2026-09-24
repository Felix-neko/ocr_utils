"""Воркер eynollah: сегментация страницы в JSON через его CLI и разбор PAGE-XML.

Аргументы: ``<png> <out.json> <каталог моделей>``. Выход:
``{"lines": [{"baseline": …, "boundary": …}], "regions": [[[x, y], …]]}`` в пикселях изображения.
"""

import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Пространство имён PAGE-XML: у eynollah это издание 2019 года, но тег берём по суффиксу.
POINTS = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")


def points_of(node) -> list[list[float]]:
    """Разобрать атрибут ``points`` узла ``Coords``/``Baseline`` в список точек."""
    if node is None:
        return []
    return [[float(x), float(y)] for x, y in POINTS.findall(node.get("points", ""))]


def child(node, tag: str):
    """Первый потомок с указанным именем тега без учёта пространства имён."""
    for item in node:
        if item.tag.rsplit("}", 1)[-1] == tag:
            return item
    return None


def main() -> None:
    png, out_path, models = sys.argv[1], sys.argv[2], sys.argv[3]
    with tempfile.TemporaryDirectory(prefix="eynollah_") as folder:
        # -cl: точные контуры строк с выпрямлением каждого региона (нам важны именно кривые строки);
        # -fl: полная вёрстка, чтобы заголовки шли отдельными регионами.
        command = [
            str(Path(sys.executable).parent / "eynollah"),
            # Каталог моделей — опция верхнего уровня, до имени подкоманды.
            "-m",
            models,
            "layout",
            "-i",
            png,
            "-o",
            folder,
            "-cl",
            "-fl",
            "-O",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        xmls = sorted(Path(folder).glob("*.xml"))
        if not xmls:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-8:]
            raise SystemExit("eynollah не отдал PAGE-XML: " + " | ".join(tail))
        root = ET.parse(xmls[0]).getroot()

        lines, regions = [], []
        for node in root.iter():
            tag = node.tag.rsplit("}", 1)[-1]
            if tag == "TextRegion":
                coords = points_of(child(node, "Coords"))
                if len(coords) >= 3:
                    regions.append(coords)
            elif tag == "TextLine":
                boundary = points_of(child(node, "Coords"))
                baseline = points_of(child(node, "Baseline"))
                if len(baseline) < 2 and len(boundary) < 3:
                    continue
                lines.append({"baseline": baseline, "boundary": boundary})
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"lines": lines, "regions": regions}, handle)


if __name__ == "__main__":
    main()
