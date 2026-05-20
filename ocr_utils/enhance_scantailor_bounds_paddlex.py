#!/usr/bin/env python3
"""
Скрипт для улучшения content bounding boxes в ScanTailor-проектах с помощью PaddleX Layout Detection.
"""

import os
import sys
from pathlib import Path

# Добавляем CUDA 12 библиотеки в LD_LIBRARY_PATH для GPU-версии PaddlePaddle
venv_path = Path(__file__).parent.parent / '.venv'
cuda_paths = [
    venv_path / 'lib/python3.12/site-packages/nvidia/cuda_runtime/lib',
    venv_path / 'lib/python3.12/site-packages/nvidia/cudnn/lib',
    venv_path / 'lib/python3.12/site-packages/nvidia/cublas/lib',
]
existing_paths = [str(p) for p in cuda_paths if p.exists()]
if existing_paths:
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = ':'.join(existing_paths + [current_ld_path])

# Отключаем OneDNN ДО импорта любых paddle модулей
os.environ['FLAGS_use_mkldnn'] = 'False'
os.environ['FLAGS_use_mkldnn_int8'] = 'False'
os.environ['FLAGS_use_mkldnn_bfloat16'] = 'False'
os.environ['ONEDNN_VERBOSE'] = '0'

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Optional
import logging

from PIL import Image

# Импортируем paddle и принудительно отключаем OneDNN
import paddle
import paddle.base as base

# Отключаем OneDNN через paddle API
try:
    base.core.disable_mkldnn()
except:
    pass

from paddlex import create_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_paddlex_pipeline():
    """Загружает PaddleX pipeline для анализа layout."""
    logger.info("Загрузка модели PaddleX Layout Detection (это займёт ~30 сек)...")
    pipeline = create_pipeline(pipeline="layout_parsing")
    return pipeline


def detect_content_bbox_with_paddlex(image_path: Path, pipeline, sub_page: Optional[str] = None, page_rect_x: float = 0, page_rect_width: Optional[float] = None) -> Optional[Tuple[Tuple[float, float, float, float], dict]]:
    """
    Определяет bounding box контента на изображении с помощью PaddleX.
    
    Args:
        image_path: Путь к изображению
        pipeline: PaddleX pipeline
        sub_page: 'left', 'right' или None (для фильтрации bboxes)
        page_rect_x: X-координата начала pageRect (смещение страницы в исходном изображении)
        page_rect_width: Ширина pageRect (ширина страницы)
    
    Returns:
        Tuple ((x, y, width, height), stats_dict) или None если контент не найден
    """
    try:
        # Запускаем детекцию layout (все файлы должны быть уже в PNG)
        result_gen = pipeline.predict(str(image_path))
        
        # PaddleX возвращает генератор, берём первый результат
        result = next(result_gen)
        
        # Для layout_parsing результат содержит layout_det_res
        if not result or 'layout_det_res' not in result:
            logger.warning(f"Контент не найден на {image_path.name}")
            return None
        
        layout_det_res = result['layout_det_res']
        if 'boxes' not in layout_det_res or not layout_det_res['boxes']:
            logger.warning(f"Layout boxes не найдены на {image_path.name}")
            return None
        
        # Получаем все layout boxes
        layout_boxes = []
        for box in layout_det_res['boxes']:
            label = box.get('label', 'unknown')
            score = box.get('score', 0.0)
            coordinate = box.get('coordinate', [])
            
            # Фильтруем служебные элементы (header, footer)
            if label.lower() in ['header', 'footer', 'page_header', 'page_footer']:
                continue
            
            # coordinate формат: [x1, y1, x2, y2]
            if len(coordinate) >= 4:
                layout_boxes.append({
                    'bbox': [float(coordinate[0]), float(coordinate[1]), float(coordinate[2]), float(coordinate[3])],
                    'label': label,
                    'score': score
                })
        
        if not layout_boxes:
            logger.warning(f"Контент не найден на {image_path.name}")
            return None
        
        # Фильтруем boxes по области страницы, используя pageRect
        if page_rect_width is not None:
            # Границы страницы в исходном изображении
            page_left = page_rect_x
            page_right = page_rect_x + page_rect_width
            
            filtered_boxes = []
            
            for box in layout_boxes:
                bbox = box['bbox']
                # bbox формат: [x1, y1, x2, y2]
                box_center_x = (bbox[0] + bbox[2]) / 2
                
                # Берём только те boxes, центр которых находится в пределах страницы
                if page_left <= box_center_x < page_right:
                    filtered_boxes.append(box)
            
            if not filtered_boxes:
                logger.warning(f"Контент не найден в области страницы {image_path.name}")
                return None
            
            layout_boxes = filtered_boxes
        
        # Находим общий охватывающий прямоугольник для всех элементов
        min_x = min(box['bbox'][0] for box in layout_boxes)
        min_y = min(box['bbox'][1] for box in layout_boxes)
        max_x = max(box['bbox'][2] for box in layout_boxes)
        max_y = max(box['bbox'][3] for box in layout_boxes)
        
        # PaddleX использует координаты в исходном изображении
        # Вычитаем смещение page_rect_x, чтобы получить координаты относительно начала страницы
        min_x -= page_rect_x
        max_x -= page_rect_x
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Подсчитываем типы элементов для статистики
        element_types = {}
        for box in layout_boxes:
            label = box['label']
            element_types[label] = element_types.get(label, 0) + 1
        
        stats = {
            'total_elements': len(layout_boxes),
            'element_types': element_types,
            'bbox': (min_x, min_y, width, height)
        }
        
        return ((min_x, min_y, width, height), stats)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке {image_path.name}: {e}")
        return None


def process_scantailor_project(project_path: Path, output_suffix: str = "_paddlex", input_images_dir: str = "out", output_images_dir: str = "out2"):
    """
    Обрабатывает ScanTailor-проект, улучшая content bounding boxes с помощью PaddleX.
    
    Args:
        project_path: Путь к .ScanTailor файлу
        output_suffix: Суффикс для нового файла проекта
        input_images_dir: Папка с обработанными изображениями (после deskew)
        output_images_dir: Папка для выходных изображений в новом проекте
    """
    logger.info(f"Обработка проекта: {project_path}")
    
    # Загружаем PaddleX pipeline
    pipeline = load_paddlex_pipeline()
    
    # Парсим XML
    tree = ET.parse(project_path)
    root = tree.getroot()
    
    # Обновляем outputDirectory на новую папку
    root.set('outputDirectory', output_images_dir)
    
    # Получаем базовую директорию проекта
    base_dir = project_path.parent
    input_dir = base_dir / input_images_dir
    
    # Создаём словарь file_id -> filename
    file_mapping = {}
    for file_elem in root.findall('.//file'):
        file_id = file_elem.get('id')
        filename = file_elem.get('name')
        file_mapping[file_id] = filename
    
    # Создаём словарь image_id -> file_id
    image_to_file = {}
    for image_elem in root.findall('.//images/image'):
        image_id = image_elem.get('id')
        file_id = image_elem.get('fileId')
        image_to_file[image_id] = file_id
    
    # Создаём словарь page_id -> (image_id, subPage)
    page_to_image = {}
    for page_elem in root.findall('.//pages/page'):
        page_id = page_elem.get('id')
        image_id = page_elem.get('imageId')
        sub_page = page_elem.get('subPage')
        page_to_image[page_id] = (image_id, sub_page)
    
    # Обрабатываем каждую страницу в select-content
    select_content = root.find('.//select-content')
    if select_content is None:
        logger.error("Секция select-content не найдена в проекте")
        return
    
    pages_processed = 0
    pages_updated = 0
    
    for page in select_content.findall('.//page'):
        page_id = page.get('id')
        
        # Находим соответствующий файл изображения через page_id -> image_id -> file_id
        page_info = page_to_image.get(page_id)
        if not page_info:
            logger.warning(f"Не найден image_id для page_id={page_id}")
            continue
        
        image_id, sub_page = page_info
        
        file_id = image_to_file.get(image_id)
        if not file_id:
            logger.warning(f"Не найден file_id для image_id={image_id}")
            continue
            
        filename = file_mapping.get(file_id)
        if not filename:
            logger.warning(f"Не найден filename для file_id={file_id}")
            continue
        
        # Строим имя файла из папки out: базовое_имя_без_расширения_subPageIndex_subPage.tif
        base_filename = Path(filename).stem
        
        # Определяем индекс подстраницы (1 или 2) на основе image_id
        pages_for_image = [p for p in root.findall('.//pages/page') if p.get('imageId') == image_id]
        pages_for_image_sorted = sorted(pages_for_image, key=lambda p: p.get('id'))
        subpage_index = next((i+1 for i, p in enumerate(pages_for_image_sorted) if p.get('id') == page_id), 1)
        
        subpage_letter = sub_page[0].upper() if sub_page else 'R'
        processed_filename_base = f"{base_filename}_{subpage_index}{subpage_letter}"
        
        # Пробуем найти файл с разными расширениями (приоритет PNG для PaddleX)
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif']:
            candidate = input_dir / f"{processed_filename_base}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            logger.warning(f"Файл не найден: {input_dir / processed_filename_base}.*")
            continue
        
        # Для обработанных изображений из out не нужна фильтрация по page_rect
        # Каждое изображение уже содержит только одну страницу
        logger.info(f"Обработка {image_path.name} (страница {page_id})...")
        
        result = detect_content_bbox_with_paddlex(image_path, pipeline, sub_page, page_rect_x=0, page_rect_width=None)
        
        if result is None:
            logger.warning(f"  ❌ Контент не найден, страница пропущена")
            pages_processed += 1
            continue
        
        bbox, stats = result
        
        # Выводим информацию о найденных элементах
        elements_info = ", ".join([f"{count} {label}" for label, count in sorted(stats['element_types'].items())])
        logger.info(f"  Найдено: {stats['total_elements']} элементов ({elements_info})")
        
        # Обновляем content-rect в XML
        content_rect = page.find('.//content-rect')
        if content_rect is not None:
            x, y, width, height = bbox
            content_rect.set('x', str(x))
            content_rect.set('y', str(y))
            content_rect.set('width', str(width))
            content_rect.set('height', str(height))
            pages_updated += 1
            logger.info(f"  ✓ Обновлён bbox: ({x:.1f}, {y:.1f}, {width:.1f}×{height:.1f})")
        else:
            logger.warning(f"  ⚠ content-rect не найден для страницы {page_id}")
        
        # Устанавливаем contentDetectionMode в manual (pageDetectionMode не меняем)
        params = page.find('.//params')
        if params is not None:
            params.set('contentDetectionMode', 'manual')
        
        pages_processed += 1
    
    # Сохраняем новый XML
    output_path = project_path.parent / f"{project_path.stem}{output_suffix}{project_path.suffix}"
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    logger.info("=" * 70)
    logger.info(f"✓ Обработка завершена!")
    logger.info(f"  Всего страниц: {total_pages}")
    logger.info(f"  Обработано: {pages_processed}")
    logger.info(f"  Обновлено: {pages_updated}")
    logger.info(f"  Пропущено: {pages_processed - pages_updated}")
    logger.info(f"  Результат: {output_path}")
    logger.info("=" * 70)


def main():
    """Точка входа скрипта."""
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python enhance_scantailor_bounds_paddlex.py <путь_к_проекту.ScanTailor>")
        sys.exit(1)
    
    project_path = Path(sys.argv[1])
    
    if not project_path.exists():
        print(f"Ошибка: файл не найден: {project_path}")
        sys.exit(1)
    
    if not project_path.suffix == '.ScanTailor':
        print(f"Предупреждение: файл не имеет расширения .ScanTailor")
    
    process_scantailor_project(project_path)


if __name__ == "__main__":
    main()
