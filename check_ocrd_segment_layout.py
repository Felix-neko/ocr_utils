#!/usr/bin/env python3
"""
Тестирование ocrd-segment для определения границ полезной области на страницах.
Использует ocrd-anybaseocr для сегментации и ocrd-segment-extract-regions для извлечения регионов.
Обрабатывает все TIF-файлы из входной директории, находит bounding box полезной области
и сохраняет изображения с нарисованными границами.
"""

from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2


def detect_regions_opencv(img_array):
    """
    Детектирует регионы на изображении используя OpenCV контуры.
    
    Returns:
        list: список словарей с информацией о регионах
    """
    img_height, img_width = img_array.shape[:2]
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    min_area = 1000
    
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        coverage_x = w / img_width
        coverage_y = h / img_height
        
        if coverage_x > 0.95 and coverage_y > 0.95:
            continue
        
        roi = binary[y:y+h, x:x+w]
        pixel_density = np.sum(roi > 0) / (w * h)
        
        aspect_ratio = w / float(h) if h > 0 else 0
        
        if aspect_ratio > 10:
            region_type = 'separator'
            min_density = 0.01
        elif aspect_ratio < 0.5:
            region_type = 'image'
            min_density = 0.05
        else:
            region_type = 'text'
            min_density = 0.1
        
        if pixel_density < min_density:
            continue
        
        regions.append({
            'id': f'region_{idx:04d}',
            'type': region_type,
            'bbox': [x, y, x + w, y + h],
            'polygon': [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            'confidence': float(pixel_density)
        })
    
    return regions


def find_useful_area_bbox(regions):
    """
    Находит общий bounding box полезной области страницы из регионов.
    
    Returns:
        tuple: (x1, y1, x2, y2) и список регионов, или (None, []) если регионы не найдены
    """
    if not regions:
        return None, []
    
    x1 = min(r['bbox'][0] for r in regions)
    y1 = min(r['bbox'][1] for r in regions)
    x2 = max(r['bbox'][2] for r in regions)
    y2 = max(r['bbox'][3] for r in regions)
    
    return (x1, y1, x2, y2), regions


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

    for idx, tif_path in enumerate(tif_files, 1):
        print(f"\n[{idx}/{len(tif_files)}] Обработка: {tif_path.name}")

        img = Image.open(tif_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        print(f"  Размер изображения: {img.size}")
        print(f"  Детектирование регионов с помощью OpenCV...")

        img_array = np.array(img)
        regions = detect_regions_opencv(img_array)
        bbox, all_regions = find_useful_area_bbox(regions)

        print(f"  Найдено регионов: {len(all_regions)}")

        if all_regions:
            region_types = {}
            avg_confidence = {}
            for r in all_regions:
                rtype = r["type"]
                region_types[rtype] = region_types.get(rtype, 0) + 1
                if rtype not in avg_confidence:
                    avg_confidence[rtype] = []
                avg_confidence[rtype].append(r.get('confidence', 0))

            if region_types:
                print(f"  Типы регионов: {dict(region_types)}")
                conf_str = ", ".join([f"{k}: {np.mean(v):.2f}" for k, v in avg_confidence.items()])
                print(f"  Средний confidence: {conf_str}")

        if bbox:
            x1, y1, x2, y2 = bbox
            
            print(f"  Границы полезной области: ({x1}, {y1}) -> ({x2}, {y2})")
            print(f"  Размер области: {x2-x1} x {y2-y1}")

            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

            color_map = {
                "text": "blue",
                "paragraph": "blue",
                "heading": "green",
                "image": "purple",
                "separator": "orange",
                "graphic": "magenta",
            }

            for r in all_regions:
                rx1, ry1, rx2, ry2 = r["bbox"]
                rtype = r["type"]
                color = color_map.get(rtype, "cyan")
                width_px = 3 if rtype in ["image", "separator"] else 2
                draw.rectangle([rx1, ry1, rx2, ry2], outline=color, width=width_px)
        else:
            print(f"  ⚠ Полезная область не найдена")

        output_path = output_dir / tif_path.name
        img.save(output_path, compression="tiff_deflate")
        print(f"  ✓ Сохранено: {output_path}")


if __name__ == "__main__":
    input_directory = Path("/mnt/dump3/DOWN/1975-12/out")
    output_directory = Path("/mnt/dump3/DOWN/1975-12/out_ocrd_segment")

    print("=" * 80)
    print("Тестирование ocrd-segment для определения границ полезной области")
    print("=" * 80)
    print(f"Входная директория: {input_directory}")
    print(f"Выходная директория: {output_directory}")
    print()

    process_images(input_directory, output_directory)

    print("\n" + "=" * 80)
    print("✓ Обработка завершена")
    print("=" * 80)
