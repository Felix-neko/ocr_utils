#!/usr/bin/env python3
"""
Тестирование PaddleX для определения границ полезной области на страницах.
Обрабатывает все TIF-файлы из входной директории, находит bounding box полезной области
и сохраняет изображения с нарисованными границами.
"""

import os
from pathlib import Path

os.environ['FLAGS_use_mkldnn'] = 'False'
os.environ['FLAGS_use_mkldnn_int8'] = 'False'
os.environ['FLAGS_use_mkldnn_bfloat16'] = 'False'
os.environ['ONEDNN_VERBOSE'] = '0'

from PIL import Image, ImageDraw
from paddlex import create_model


def find_useful_area_bbox(layout_result):
    """
    Находит общий bounding box полезной области страницы.
    Объединяет все найденные блоки текста/изображений в один bbox.
    
    Returns:
        tuple: (x1, y1, x2, y2) или None если блоки не найдены
    """
    if 'boxes' not in layout_result or not layout_result['boxes']:
        return None
    
    boxes_source = layout_result['boxes']
    
    layout_boxes = []
    for box in boxes_source:
        label = box.get('label', 'unknown')
        coordinate = box.get('coordinate', [])
        
        if label.lower() in ['header', 'footer', 'page_header', 'page_footer']:
            continue
        
        if len(coordinate) >= 4:
            layout_boxes.append({
                'bbox': [float(coordinate[0]), float(coordinate[1]), float(coordinate[2]), float(coordinate[3])],
                'label': label,
                'score': box.get('score', 0.0)
            })
    
    if not layout_boxes:
        return None
    
    x1 = min(box['bbox'][0] for box in layout_boxes)
    y1 = min(box['bbox'][1] for box in layout_boxes)
    x2 = max(box['bbox'][2] for box in layout_boxes)
    y2 = max(box['bbox'][3] for box in layout_boxes)
    
    return (x1, y1, x2, y2), layout_boxes


def process_images(input_dir: Path, output_dir: Path):
    """
    Обрабатывает все PNG-файлы из input_dir, находит границы полезной области
    и сохраняет результаты с нарисованными bounding boxes в output_dir.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    png_files = sorted(input_dir.glob("*.png"))
    
    if not png_files:
        print(f"Не найдено PNG-файлов в {input_dir}")
        return
    
    print(f"Найдено {len(png_files)} PNG-файлов")
    print(f"Загрузка модели PaddleX Layout Detection...")
    
    model = create_model('RT-DETR-H_layout_17cls')
    
    for idx, png_path in enumerate(png_files, 1):
        print(f"\n[{idx}/{len(png_files)}] Обработка: {png_path.name}")
        
        img = Image.open(png_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        print(f"  Размер изображения: {img.size}")
        print(f"  Запуск layout detection...")
        
        result_gen = model.predict(str(png_path))
        result = next(result_gen)
        
        bbox_result = find_useful_area_bbox(result)
        
        if bbox_result:
            bbox, layout_boxes = bbox_result
            x1, y1, x2, y2 = bbox
            
            print(f"  Найдено блоков: {len(layout_boxes)}")
            
            element_types = {}
            for box in layout_boxes:
                label = box['label']
                element_types[label] = element_types.get(label, 0) + 1
            
            elements_info = ", ".join([f"{count} {label}" for label, count in sorted(element_types.items())])
            print(f"  Типы элементов: {elements_info}")
            print(f"  Границы полезной области: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
            print(f"  Размер области: {x2-x1:.1f} x {y2-y1:.1f}")
            
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
            
            for block_bbox in layout_boxes:
                bx1, by1, bx2, by2 = block_bbox['bbox']
                draw.rectangle([bx1, by1, bx2, by2], outline="blue", width=2)
        else:
            print(f"  ⚠ Полезная область не найдена")
        
        output_path = output_dir / png_path.name
        img.save(output_path)
        print(f"  ✓ Сохранено: {output_path}")


if __name__ == "__main__":
    input_directory = Path("/mnt/dump3/DOWN/1975-12/out")
    output_directory = Path("/mnt/dump3/DOWN/1975-12/out4")
    
    print("=" * 80)
    print("Тестирование PaddleX для определения границ полезной области")
    print("=" * 80)
    print(f"Входная директория: {input_directory}")
    print(f"Выходная директория: {output_directory}")
    print()
    
    process_images(input_directory, output_directory)
    
    print("\n" + "=" * 80)
    print("✓ Обработка завершена")
    print("=" * 80)
