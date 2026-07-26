#!/usr/bin/env python3
"""
Тестирование surya-ocr для определения границ полезной области на страницах.
Обрабатывает все TIF-файлы из входной директории, находит bounding box полезной области
и сохраняет изображения с нарисованными границами.
"""

from pathlib import Path
from PIL import Image, ImageDraw
from surya.layout import LayoutPredictor
from surya.foundation import FoundationPredictor
from surya.settings import settings
import torch


def find_useful_area_bbox(layout_result):
    """
    Находит общий bounding box полезной области страницы.
    Объединяет все найденные блоки (текст, изображения, линии, виньетки) в один bbox.

    Args:
        layout_result: результат layout detection от surya

    Returns:
        tuple: (x1, y1, x2, y2) или None если блоки не найдены
    """
    if not layout_result.bboxes:
        return None

    x1 = min(bbox.bbox[0] for bbox in layout_result.bboxes)
    y1 = min(bbox.bbox[1] for bbox in layout_result.bboxes)
    x2 = max(bbox.bbox[2] for bbox in layout_result.bboxes)
    y2 = max(bbox.bbox[3] for bbox in layout_result.bboxes)

    return (x1, y1, x2, y2)


def process_images(input_dir: Path, output_dir: Path):
    """
    Обрабатывает все TIF и JPG файлы из input_dir (нерекурсивно), находит границы полезной области
    и сохраняет результаты с нарисованными bounding boxes в output_dir.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in (".tif", ".jpg", ".jpeg")]
    )

    if not image_files:
        print(f"Не найдено TIF/JPG файлов в {input_dir}")
        return

    print(f"Найдено {len(image_files)} изображений")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")
    print(f"Загрузка моделей surya-ocr...")

    layout_predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))

    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Обработка: {img_path.name}")

        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        print(f"  Размер изображения: {img.size}")
        print(f"  Запуск layout detection...")

        layout_results = layout_predictor([img])
        layout_result = layout_results[0]

        print(f"  Найдено блоков: {len(layout_result.bboxes)}")

        block_types = {}
        for block_bbox in layout_result.bboxes:
            label = block_bbox.label
            block_types[label] = block_types.get(label, 0) + 1

        if block_types:
            print(f"  Типы блоков: {dict(block_types)}")

        bbox = find_useful_area_bbox(layout_result)

        if bbox:
            x1, y1, x2, y2 = bbox
            print(f"  Границы полезной области: ({x1}, {y1}) -> ({x2}, {y2})")
            print(f"  Размер области: {x2-x1} x {y2-y1}")

            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

            color_map = {
                "Text": "blue",
                "Title": "green",
                "Figure": "purple",
                "Table": "orange",
                "Caption": "cyan",
                "Footnote": "magenta",
                "Page-header": "yellow",
                "Page-footer": "pink",
                "Section-header": "lime",
            }

            for block_bbox in layout_result.bboxes:
                bx1, by1, bx2, by2 = block_bbox.bbox
                label = block_bbox.label
                color = color_map.get(label, "blue")
                draw.rectangle([bx1, by1, bx2, by2], outline=color, width=2)
        else:
            print(f"  ⚠ Полезная область не найдена")

        output_path = output_dir / img_path.name
        img.save(output_path, compression="tiff_deflate")
        print(f"  ✓ Сохранено: {output_path}")


if __name__ == "__main__":
    input_directory = Path("/mnt/dump3/DOWN/1975-12")
    output_directory = Path("/mnt/dump3/DOWN/1975-12_surya")

    print("=" * 80)
    print("Тестирование surya-ocr для определения границ полезной области")
    print("=" * 80)
    print(f"Входная директория: {input_directory}")
    print(f"Выходная директория: {output_directory}")
    print()

    process_images(input_directory, output_directory)

    print("\n" + "=" * 80)
    print("✓ Обработка завершена")
    print("=" * 80)
