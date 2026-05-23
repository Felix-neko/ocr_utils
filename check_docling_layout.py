#!/usr/bin/env python3
"""
Тестирование docling для определения границ полезной области на страницах.
Обрабатывает все TIF-файлы из входной директории, находит bounding box полезной области
и сохраняет изображения с нарисованными границами.
"""

from pathlib import Path
from PIL import Image, ImageDraw
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
from docling.utils.model_downloader import download_models
import torch


def find_useful_area_bbox(predictions):
    """
    Находит общий bounding box полезной области страницы из layout predictions.
    Объединяет все найденные блоки (текст, изображения, линии, виньетки) в один bbox.

    Args:
        predictions: список predictions (dict) от LayoutPredictor

    Returns:
        tuple: (x1, y1, x2, y2) и список блоков, или (None, []) если блоки не найдены
    """
    if not predictions:
        return None, []

    bboxes = []
    for pred in predictions:
        bboxes.append({
            'bbox': [pred['l'], pred['t'], pred['r'], pred['b']],
            'label': pred.get('label', 'unknown'),
            'score': pred.get('confidence', 0.0)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")
    print(f"Загрузка моделей docling...")
    print(f"  (модели будут загружены автоматически при первом запуске)")
    
    download_models()
    
    cache_dir = Path.home() / ".cache" / "docling" / "models"
    layout_model_path = cache_dir / "docling-project--docling-layout-heron"
    
    if not layout_model_path.exists():
        raise FileNotFoundError(f"Layout model не найдена по пути: {layout_model_path}")
    
    print(f"  Используется модель: docling-layout-heron")
    layout_predictor = LayoutPredictor(
        artifact_path=str(layout_model_path),
        device=device
    )

    for idx, tif_path in enumerate(tif_files, 1):
        print(f"\n[{idx}/{len(tif_files)}] Обработка: {tif_path.name}")

        img = Image.open(tif_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        print(f"  Размер изображения: {img.size}")
        print(f"  Запуск layout detection...")

        layout_result_gen = layout_predictor.predict(img)
        predictions = list(layout_result_gen)
        
        bbox, bboxes = find_useful_area_bbox(predictions)

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
            print(f"  Границы полезной области: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
            print(f"  Размер области: {x2-x1:.1f} x {y2-y1:.1f}")

            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

            color_map = {
                "text": "blue",
                "title": "green",
                "figure": "purple",
                "picture": "purple",
                "table": "orange",
                "caption": "cyan",
                "footnote": "magenta",
                "page-header": "yellow",
                "page-footer": "pink",
                "section-header": "lime",
            }

            for b in bboxes:
                bx1, by1, bx2, by2 = b["bbox"]
                label = b["label"]
                color = color_map.get(label.lower(), "blue")
                draw.rectangle([bx1, by1, bx2, by2], outline=color, width=2)
        else:
            print(f"  ⚠ Полезная область не найдена")

        output_path = output_dir / tif_path.name
        img.save(output_path, compression="tiff_deflate")
        print(f"  ✓ Сохранено: {output_path}")


if __name__ == "__main__":
    input_directory = Path("/mnt/dump3/DOWN/1975-12/out")
    output_directory = Path("/mnt/dump3/DOWN/1975-12/out5")

    print("=" * 80)
    print("Тестирование docling для определения границ полезной области")
    print("=" * 80)
    print(f"Входная директория: {input_directory}")
    print(f"Выходная директория: {output_directory}")
    print()

    process_images(input_directory, output_directory)

    print("\n" + "=" * 80)
    print("✓ Обработка завершена")
    print("=" * 80)
