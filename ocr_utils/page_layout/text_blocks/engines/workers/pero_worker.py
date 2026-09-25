"""Воркер pero: сегментация страницы в JSON. Запускается ЧУЖИМ питоном, наш пакет не импортирует.

Аргументы: ``<png> <out.json> <config.ini>``. Выход:
``{"lines": [{"baseline": [[x, y], …], "boundary": […], "height": h}], "regions": [[[x, y], …]]}``
в пикселях поданного изображения.
"""

import configparser
import json
import os
import sys

import cv2
import numpy as np

from pero_ocr.core.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser


def main() -> None:
    png, out_path, config_path = sys.argv[1], sys.argv[2], sys.argv[3]
    # Из конфига модели оставляем только разбор вёрстки: распознавание нам не нужно и оно долгое.
    config = configparser.ConfigParser()
    config.read(config_path)
    config["PAGE_PARSER"]["RUN_LINE_CROPPER"] = "no"
    config["PAGE_PARSER"]["RUN_OCR"] = "no"
    config["PAGE_PARSER"]["RUN_DECODER"] = "no"

    image = cv2.imread(png, cv2.IMREAD_COLOR)
    layout = PageLayout(id="page", page_size=(image.shape[0], image.shape[1]))
    parser = PageParser(config, config_path=os.path.dirname(config_path))
    layout = parser.process_page(image, layout)

    lines, regions = [], []
    for region in layout.regions:
        if region.polygon is not None and len(region.polygon) >= 3:
            regions.append([[float(x), float(y)] for x, y in np.asarray(region.polygon)])
        for line in region.lines:
            if line.baseline is None or len(line.baseline) < 2:
                continue
            baseline = [[float(x), float(y)] for x, y in np.asarray(line.baseline)]
            polygon = []
            if line.polygon is not None and len(line.polygon) >= 3:
                polygon = [[float(x), float(y)] for x, y in np.asarray(line.polygon)]
            # heights = (над базовой линией, под ней): полная высота строки — их сумма.
            height = float(np.sum(line.heights)) if line.heights is not None else 0.0
            lines.append({"baseline": baseline, "boundary": polygon, "height": height})
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"lines": lines, "regions": regions}, handle)


if __name__ == "__main__":
    main()
