#!/usr/bin/env python3
"""
Тестирование doctr для определения границ полезной области на страницах.
Обрабатывает все TIF-файлы из входной директории, находит bounding box полезной области
и сохраняет изображения с нарисованными границами.
"""

from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from doctr.models import ocr_predictor


def find_useful_area_bbox(result):
    """
    Находит общий bounding box полезной области страницы из doctr OCR result.
    Объединяет все найденные блоки (текст, таблицы и т.д.) в один bbox.

    Args:
        result: результат от doctr ocr_predictor (Document object)

    Returns:
        tuple: (x1, y1, x2, y2) и список блоков, или (None, []) если блоки не найдены
    """
    if not result or not result.pages:
        return None, []

    bboxes = []
    
    for page in result.pages:
        for block in page.blocks:
            block_geom = block.geometry
            x1, y1 = block_geom[0]
            x2, y2 = block_geom[1]
            
            bboxes.append({
                'bbox': [x1, y1, x2, y2],
                'label': 'block',
                'score': 1.0
            })
            
            for line in block.lines:
                line_geom = line.geometry
                lx1, ly1 = line_geom[0]
                lx2, ly2 = line_geom[1]
                
                bboxes.append({
                    'bbox': [lx1, ly1, lx2, ly2],
                    'label': 'line',
                    'score': 1.0
                })
    
    if not bboxes:
        return None, []

    x1 = min(b['bbox'][0] for b in bboxes)
    y1 = min(b['bbox'][1] for b in bboxes)
    x2 = max(b['bbox'][2] for b in bboxes)
    y2 = max(b['bbox'][3] for b in bboxes)

    return (x1, y1, x2, y2), bboxes


def process_images(input_dir: Path, output_dir: Path):
    """
    Обрабатывает все TIF-файлы из input_dir, находит границы полезной области
    и сохраняет результаты с нарисованными bounding boxes в output_dir.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(input_dir.glob("*.tif"))

    if not tif_files:
        print(f"Не найдено TIF-файлов в {input_dir}")
        return

    print(f"Найдено {len(tif_files)} TIF-файлов")

    print(f"Загрузка модели doctr OCR...")
    predictor = ocr_predictor(pretrained=True)
    print(f"  ✓ Модель загружена")

    for idx, tif_path in enumerate(tif_files, 1):
        print(f"\n[{idx}/{len(tif_files)}] Обработка: {tif_path.name}")

        img = Image.open(tif_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        print(f"  Размер изображения: {img.size}")
        print(f"  Запуск OCR и layout detection...")

        img_array = np.array(img)
        result = predictor([img_array])
        
        bbox, bboxes = find_useful_area_bbox(result)

        print(f"  Найдено блоков: {len(bboxes)}")

        if bboxes:
            block_types = {}
            for b in bboxes:
                label = b["label"]
                block_types[label] = block_types.get(label, 0) + 1

            if block_types:
                print(f"  Типы блоков: {dict(block_types)}")

        if bbox:
            x1, y1, x2, y2 = bbox
            
            width, height = img.size
            x1_px = int(x1 * width)
            y1_px = int(y1 * height)
            x2_px = int(x2 * width)
            y2_px = int(y2 * height)
            
            print(f"  Границы полезной области: ({x1_px}, {y1_px}) -> ({x2_px}, {y2_px})")
            print(f"  Размер области: {x2_px-x1_px} x {y2_px-y1_px}")

            draw = ImageDraw.Draw(img)
            draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline="red", width=5)

            color_map = {
                "block": "green",
                "line": "blue",
            }

            for b in bboxes:
                bx1, by1, bx2, by2 = b["bbox"]
                bx1_px = int(bx1 * width)
                by1_px = int(by1 * height)
                bx2_px = int(bx2 * width)
                by2_px = int(by2 * height)
                label = b["label"]
                color = color_map.get(label, "blue")
                width_px = 3 if label == "block" else 1
                draw.rectangle([bx1_px, by1_px, bx2_px, by2_px], outline=color, width=width_px)
        else:
            print(f"  ⚠ Полезная область не найдена")

        output_path = output_dir / tif_path.name
        img.save(output_path, compression="tiff_deflate")
        print(f"  ✓ Сохранено: {output_path}")


if __name__ == "__main__":
    input_directory = Path("/mnt/dump3/DOWN/1975-12/out")
    output_directory = Path("/mnt/dump3/DOWN/1975-12/out_doctr")

    print("=" * 80)
    print("Тестирование doctr для определения границ полезной области")
    print("=" * 80)
    print(f"Входная директория: {input_directory}")
    print(f"Выходная директория: {output_directory}")
    print()

    process_images(input_directory, output_directory)

    print("\n" + "=" * 80)
    print("✓ Обработка завершена")
    print("=" * 80)
