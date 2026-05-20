#!/usr/bin/env python3
"""
Скрипт для улучшения content bounding boxes в ScanTailor-проектах с помощью Surya OCR.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Optional
import logging

from PIL import Image
from surya.layout import LayoutPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_surya_predictor():
    """Загружает предиктор Surya для анализа layout (текст + иллюстрации)."""
    logger.info("Загрузка модели Surya Layout...")
    predictor = LayoutPredictor()
    return predictor


def detect_content_bbox_with_surya(image_path: Path, predictor, sub_page: Optional[str] = None, page_rect_x: float = 0, page_rect_width: Optional[float] = None) -> Optional[Tuple[float, float, float, float]]:
    """
    Определяет bounding box контента на изображении с помощью Surya.
    
    Args:
        image_path: Путь к изображению
        predictor: Предиктор Surya
        sub_page: 'left', 'right' или None (для фильтрации bboxes)
        page_rect_x: X-координата начала pageRect (смещение страницы в исходном изображении)
        page_rect_width: Ширина pageRect (ширина страницы)
    
    Returns:
        Tuple (x, y, width, height) или None если контент не найден
    """
    try:
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        # Запускаем детекцию текста
        predictions = predictor([image])
        
        if not predictions or not predictions[0].bboxes:
            logger.warning(f"Контент не найден на {image_path.name}")
            return None
        
        # Получаем все layout boxes (текст, иллюстрации, таблицы и т.д.)
        layout_boxes = predictions[0].bboxes
        
        # Фильтруем boxes по области страницы, используя pageRect
        if page_rect_width is not None:
            # Границы страницы в исходном изображении
            page_left = page_rect_x
            page_right = page_rect_x + page_rect_width
            
            filtered_boxes = []
            
            for box in layout_boxes:
                box_center_x = (box.bbox[0] + box.bbox[2]) / 2
                
                # Берём только те boxes, центр которых находится в пределах страницы
                if page_left <= box_center_x < page_right:
                    filtered_boxes.append(box)
            
            if not filtered_boxes:
                logger.warning(f"Контент не найден в области страницы {image_path.name}")
                return None
            
            layout_boxes = filtered_boxes
        
        # Находим общий охватывающий прямоугольник для всех элементов
        min_x = min(box.bbox[0] for box in layout_boxes)
        min_y = min(box.bbox[1] for box in layout_boxes)
        max_x = max(box.bbox[2] for box in layout_boxes)
        max_y = max(box.bbox[3] for box in layout_boxes)
        
        # ScanTailor использует координаты относительно начала pageRect
        # Вычитаем смещение page_rect_x, чтобы получить координаты относительно начала страницы
        min_x -= page_rect_x
        max_x -= page_rect_x
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Подсчитываем типы элементов для логирования
        element_types = {}
        for box in layout_boxes:
            element_types[box.label] = element_types.get(box.label, 0) + 1
        
        elements_info = ", ".join([f"{count} {label}" for label, count in sorted(element_types.items())])
        sub_page_info = f" ({sub_page})" if sub_page else ""
        logger.info(f"{image_path.name}{sub_page_info}: найдено {len(layout_boxes)} элементов ({elements_info}), bbox=({min_x:.1f}, {min_y:.1f}, {width:.1f}, {height:.1f})")
        
        return (min_x, min_y, width, height)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке {image_path.name}: {e}")
        return None


def process_scantailor_project(project_path: Path, output_suffix: str = "_surya", input_images_dir: str = "out", output_images_dir: str = "out2"):
    """
    Обрабатывает ScanTailor-проект, улучшая content bounding boxes с помощью Surya.
    
    Args:
        project_path: Путь к .ScanTailor файлу
        output_suffix: Суффикс для нового файла проекта
        input_images_dir: Папка с обработанными изображениями (после deskew)
        output_images_dir: Папка для выходных изображений в новом проекте
    """
    logger.info(f"Обработка проекта: {project_path}")
    
    # Загружаем предиктор Surya
    predictor = load_surya_predictor()
    
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
        # Например: IMG_2026_05_14_17_15_12S_2R.tif
        base_filename = Path(filename).stem
        
        # Определяем индекс подстраницы (1 или 2) на основе image_id
        # Для одного image могут быть две страницы (left и right)
        # Нужно найти порядковый номер этой страницы среди всех страниц с тем же image_id
        pages_for_image = [p for p in root.findall('.//pages/page') if p.get('imageId') == image_id]
        pages_for_image_sorted = sorted(pages_for_image, key=lambda p: p.get('id'))
        subpage_index = next((i+1 for i, p in enumerate(pages_for_image_sorted) if p.get('id') == page_id), 1)
        
        subpage_letter = sub_page[0].upper() if sub_page else 'R'
        processed_filename_base = f"{base_filename}_{subpage_index}{subpage_letter}"
        
        # Пробуем найти файл с разными расширениями
        image_path = None
        for ext in ['.tif', '.jpg', '.jpeg', '.png']:
            candidate = input_dir / f"{processed_filename_base}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            logger.warning(f"Файл не найден: {input_dir / processed_filename_base}.*")
            continue
        
        # Для обработанных изображений из out не нужна фильтрация по page_rect
        # Каждое изображение уже содержит только одну страницу
        bbox = detect_content_bbox_with_surya(image_path, predictor, sub_page, page_rect_x=0, page_rect_width=None)
        
        if bbox is None:
            logger.warning(f"Пропускаем страницу {page_id} ({filename})")
            pages_processed += 1
            continue
        
        # Обновляем content-rect в XML
        content_rect = page.find('.//content-rect')
        if content_rect is not None:
            x, y, width, height = bbox
            content_rect.set('x', str(x))
            content_rect.set('y', str(y))
            content_rect.set('width', str(width))
            content_rect.set('height', str(height))
            pages_updated += 1
            logger.info(f"Обновлён bbox для страницы {page_id}")
        else:
            logger.warning(f"content-rect не найден для страницы {page_id}")
        
        # Устанавливаем contentDetectionMode в manual (pageDetectionMode не меняем)
        params = page.find('.//params')
        if params is not None:
            params.set('contentDetectionMode', 'manual')
        
        pages_processed += 1
    
    # Сохраняем новый XML
    output_path = project_path.parent / f"{project_path.stem}{output_suffix}{project_path.suffix}"
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    logger.info(f"Обработано страниц: {pages_processed}, обновлено: {pages_updated}")
    logger.info(f"Результат сохранён в: {output_path}")


def main():
    """Точка входа скрипта."""
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python enhance_scantailor_bounds.py <путь_к_проекту.ScanTailor>")
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
