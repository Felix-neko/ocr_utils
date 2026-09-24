"""Воркер kraken: сегментация страницы в JSON. Запускается ЧУЖИМ питоном, наш пакет не импортирует.

Аргументы: ``<png> <out.json>``. Выход: ``{"lines": [{"baseline": [[x, y], …], "boundary": […]}]}``
в пикселях поданного изображения.
"""

import json
import sys

from PIL import Image

from kraken import blla


def main() -> None:
    image = Image.open(sys.argv[1]).convert("L")
    segmentation = blla.segment(image)
    lines = []
    for line in segmentation.lines:
        baseline = [[float(x), float(y)] for x, y in (line.baseline or [])]
        boundary = [[float(x), float(y)] for x, y in (line.boundary or [])]
        if baseline:
            lines.append({"baseline": baseline, "boundary": boundary})
    regions = []
    for group in (segmentation.regions or {}).values():
        for region in group:
            regions.append([[float(x), float(y)] for x, y in (region.boundary or [])])
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        json.dump({"lines": lines, "regions": regions}, handle)


if __name__ == "__main__":
    main()
