#!/usr/bin/env python3
"""
Тестирование eynollah для определения границ полезной области на страницах.
Обрабатывает все TIF-файлы из входной директории, находит bounding box полезной области
и сохраняет изображения с нарисованными границами.
"""

from pathlib import Path
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import tempfile
import shutil


def parse_pagexml_regions(xml_path):
    """
    Парсит PAGE-XML файл и извлекает координаты всех текстовых регионов.
    
    Args:
        xml_path: путь к PAGE-XML файлу
    
    Returns:
        list: список словарей с информацией о регионах [{type, coords}, ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    
    regions = []
    
    for text_region in root.findall('.//pc:TextRegion', ns):
        region_type = text_region.get('type', 'unknown')
        coords_elem = text_region.find('.//pc:Coords', ns)
        
        if coords_elem is not None:
            points_str = coords_elem.get('points', '')
            if points_str:
                points = []
                for point_str in points_str.split():
                    x, y = map(int, point_str.split(','))
                    points.append((x, y))
                
                if points:
                    regions.append({
                        'type': region_type,
                        'coords': points
                    })
    
    for image_region in root.findall('.//pc:ImageRegion', ns):
        coords_elem = image_region.find('.//pc:Coords', ns)
        
        if coords_elem is not None:
            points_str = coords_elem.get('points', '')
            if points_str:
                points = []
                for point_str in points_str.split():
                    x, y = map(int, point_str.split(','))
                    points.append((x, y))
                
                if points:
                    regions.append({
                        'type': 'image',
                        'coords': points
                    })
    
    return regions


def find_useful_area_bbox(regions):
    """
    Находит общий bounding box полезной области страницы из результатов eynollah.
    
    Args:
        regions: список регионов с координатами
    
    Returns:
        tuple: (x1, y1, x2, y2) или None если регионы не найдены
    """
    if not regions:
        return None
    
    all_x = []
    all_y = []
    
    for region in regions:
        coords = region['coords']
        for x, y in coords:
            all_x.append(x)
            all_y.append(y)
    
    if not all_x:
        return None
    
    return (min(all_x), min(all_y), max(all_x), max(all_y))


def process_images(input_dir: Path, output_dir: Path, model_dir: Path):
    """
    Обрабатывает все TIF-файлы из input_dir, находит границы полезной области
    и сохраняет результаты с нарисованными bounding boxes в output_dir.
    """
    import subprocess
    import os
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tif_files = sorted(input_dir.glob("*.tif"))
    
    if not tif_files:
        print(f"Не найдено TIF-файлов в {input_dir}")
        return
    
    print(f"Найдено {len(tif_files)} TIF-файлов")
    
    if not model_dir.exists():
        print(f"⚠ Директория с моделями не найдена: {model_dir}")
        print(f"Скачайте модели с https://qurator-data.de/eynollah/")
        return
    
    print(f"Директория с моделями: {model_dir}")
    
    temp_xml_dir = tempfile.mkdtemp()
    
    try:
        for idx, tif_path in enumerate(tif_files, 1):
            print(f"\n[{idx}/{len(tif_files)}] Обработка: {tif_path.name}")
            
            img = Image.open(tif_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            print(f"  Размер изображения: {img.size}")
            print(f"  Запуск layout detection...")
            
            eynollah = Eynollah(
                image_filename=str(tif_path),
                dir_out=temp_xml_dir,
                dir_models=str(model_dir),
                full_layout=True,
            )
            
            eynollah.run()
            
            xml_filename = tif_path.stem + ".xml"
            xml_path = Path(temp_xml_dir) / xml_filename
            
            if not xml_path.exists():
                print(f"  ⚠ XML файл не создан: {xml_path}")
                continue
            
            regions = parse_pagexml_regions(xml_path)
            print(f"  Найдено регионов: {len(regions)}")
            
            region_types = {}
            for region in regions:
                rtype = region['type']
                region_types[rtype] = region_types.get(rtype, 0) + 1
            
            if region_types:
                print(f"  Типы регионов: {dict(region_types)}")
            
            bbox = find_useful_area_bbox(regions)
            
            if bbox:
                x1, y1, x2, y2 = bbox
                print(f"  Границы полезной области: ({x1}, {y1}) -> ({x2}, {y2})")
                print(f"  Размер области: {x2-x1} x {y2-y1}")
                
                draw = ImageDraw.Draw(img)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
                
                color_map = {
                    'paragraph': 'blue',
                    'heading': 'green',
                    'header': 'yellow',
                    'footer': 'pink',
                    'page-number': 'orange',
                    'drop-capital': 'purple',
                    'marginalia': 'cyan',
                    'image': 'magenta',
                }
                
                for region in regions:
                    coords = region['coords']
                    rtype = region['type']
                    color = color_map.get(rtype, 'blue')
                    
                    if len(coords) >= 2:
                        draw.polygon(coords, outline=color, width=2)
            else:
                print(f"  ⚠ Полезная область не найдена")
            
            output_path = output_dir / tif_path.name
            img.save(output_path, compression="tiff_deflate")
            print(f"  ✓ Сохранено: {output_path}")
    
    finally:
        shutil.rmtree(temp_xml_dir, ignore_errors=True)


if __name__ == "__main__":
    input_directory = Path("/mnt/dump3/DOWN/1975-12/out")
    output_directory = Path("/mnt/dump3/DOWN/1975-12/out_eynollah")
    models_directory = Path.home() / ".local" / "share" / "eynollah" / "models_eynollah"
    
    print("=" * 80)
    print("Тестирование eynollah для определения границ полезной области")
    print("=" * 80)
    print(f"Входная директория: {input_directory}")
    print(f"Выходная директория: {output_directory}")
    print(f"Директория с моделями: {models_directory}")
    print()
    
    process_images(input_directory, output_directory, models_directory)
    
    print("\n" + "=" * 80)
    print("✓ Обработка завершена")
    print("=" * 80)
