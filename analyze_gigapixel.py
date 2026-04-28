#!/usr/bin/env python3
"""
Скрипт для анализа состояния обработки изображений через Topaz Gigapixel.

Анализирует папки с изображениями журналов и определяет:
- Наличие готовых djvu-файлов
- Полноту обработки Topaz Gigapixel (все/частично/нет)
- Используемые суффиксы в обработанных файлах
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from datetime import datetime


def get_base_filename(filename: str) -> str:
    """
    Извлекает базовое имя файла без префикса "_" и суффиксов Gigapixel.
    
    Примеры:
    - "_page_0001-gigapixel-text-shapes-4x.jpeg" -> "page_0001.jpeg"
    - "page_0001.jpg" -> "page_0001.jpg"
    """
    if not filename.startswith('_'):
        return filename
    
    # Убираем префикс "_"
    filename = filename[1:]
    
    # Убираем суффиксы вида "-gigapixel-...-4x" или другие варианты
    # Паттерн: все что между последним "-gigapixel" и расширением файла
    pattern = r'-gigapixel-[^.]+(\.[^.]+)$'
    match = re.search(pattern, filename)
    if match:
        # Заменяем суффикс на расширение
        extension = match.group(1)
        filename = re.sub(pattern, extension, filename)
    
    return filename


def extract_gigapixel_suffix(filename: str) -> str:
    """
    Извлекает суффикс Gigapixel из имени файла.
    
    Примеры:
    - "_page_0001-gigapixel-text-shapes-4x.jpeg" -> "-gigapixel-text-shapes-4x"
    - "page_0001.jpg" -> ""
    """
    if not filename.startswith('_'):
        return ""
    
    pattern = r'(-gigapixel-[^.]+)\.[^.]+$'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    
    return ""


def analyze_directory(dir_path: Path) -> Dict:
    """
    Анализирует одну директорию с изображениями журнала.
    
    Возвращает словарь с информацией:
    - has_djvu: bool - есть ли djvu-файл
    - original_files: Set[str] - набор исходных файлов (без "_")
    - processed_files: Dict[str, List[str]] - обработанные файлы: {base_name: [suffixes]}
    - gigapixel_suffixes: Set[str] - все найденные суффиксы Gigapixel
    """
    result = {
        'has_djvu': False,
        'original_files': set(),
        'processed_files': defaultdict(list),
        'gigapixel_suffixes': set(),
    }
    
    if not dir_path.exists() or not dir_path.is_dir():
        return result
    
    for item in dir_path.iterdir():
        if item.is_file():
            filename = item.name
            
            # Проверяем наличие djvu-файла
            if filename.endswith('.djvu'):
                result['has_djvu'] = True
                continue
            
            # Обрабатываем изображения
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                if filename.startswith('_'):
                    # Это обработанный файл
                    base_name = get_base_filename(filename)
                    suffix = extract_gigapixel_suffix(filename)
                    result['processed_files'][base_name].append(suffix)
                    if suffix:
                        result['gigapixel_suffixes'].add(suffix)
                else:
                    # Это исходный файл
                    result['original_files'].add(filename)
    
    return result


def get_processing_status(original_files: Set[str], processed_files: Dict[str, List[str]]) -> str:
    """
    Определяет статус обработки Gigapixel.
    
    Возвращает:
    - "complete" - все исходные файлы обработаны
    - "partial" - часть файлов обработана
    - "none" - нет обработанных файлов
    """
    if not original_files:
        return "none"
    
    processed_count = sum(1 for orig in original_files if orig in processed_files)
    
    if processed_count == 0:
        return "none"
    elif processed_count == len(original_files):
        return "complete"
    else:
        return "partial"


def analyze_root_directory(root_path: str) -> List[Dict]:
    """
    Анализирует всю структуру папок с журналами.
    
    Возвращает список словарей с информацией о каждой папке журнала.
    """
    root = Path(root_path)
    results = []
    
    # Ищем все директории напрямую в корне (без промежуточного уровня по годам)
    for journal_dir in sorted(root.iterdir()):
        if not journal_dir.is_dir():
            continue
        
        # Пропускаем служебные директории
        if journal_dir.name.startswith('.'):
            continue
        
        analysis = analyze_directory(journal_dir)
        status = get_processing_status(
            analysis['original_files'],
            analysis['processed_files']
        )
        
        results.append({
            'path': str(journal_dir.relative_to(root)),
            'full_path': str(journal_dir),
            'has_djvu': analysis['has_djvu'],
            'status': status,
            'original_count': len(analysis['original_files']),
            'processed_count': len(analysis['processed_files']),
            'gigapixel_suffixes': sorted(analysis['gigapixel_suffixes']),
        })
    
    return results


def generate_markdown_report(results: List[Dict], output_path: str):
    """
    Генерирует Markdown-отчёт о состоянии обработки.
    """
    # Собираем статистику
    total = len(results)
    with_djvu = sum(1 for r in results if r['has_djvu'])
    complete = sum(1 for r in results if r['status'] == 'complete')
    partial = sum(1 for r in results if r['status'] == 'partial')
    none = sum(1 for r in results if r['status'] == 'none')
    
    # Собираем все уникальные суффиксы
    all_suffixes = set()
    for r in results:
        all_suffixes.update(r['gigapixel_suffixes'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Анализ обработки Topaz Gigapixel\n\n")
        f.write(f"**Дата анализа:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Приоритетные таблицы: необработанные и частично обработанные
        priority_categories = [
            ("Не обработано Gigapixel (без DJVU)", lambda r: r['status'] == 'none' and not r['has_djvu']),
            ("Частично обработано Gigapixel", lambda r: r['status'] == 'partial'),
        ]
        
        for category_name, filter_func in priority_categories:
            filtered = [r for r in results if filter_func(r)]
            f.write(f"## {category_name}\n\n")
            f.write(f"**Количество:** {len(filtered)}\n\n")
            
            if filtered:
                f.write("| Папка | Исходных файлов | Обработано файлов | Покрытие | Суффиксы |\n")
                f.write("|-------|-----------------|-------------------|----------|----------|\n")
                for r in filtered:
                    coverage = f"{r['processed_count'] * 100 // r['original_count'] if r['original_count'] else 0}%" if r['status'] == 'partial' else "—"
                    suffixes = ', '.join(f'`{s}`' for s in r['gigapixel_suffixes']) if r['gigapixel_suffixes'] else "—"
                    f.write(f"| {r['path']} | {r['original_count']} | {r['processed_count']} | {coverage} | {suffixes} |\n")
                f.write("\n")
            else:
                f.write("*Нет папок в этой категории*\n\n")
        
        # Общая статистика
        f.write("## Общая статистика\n\n")
        f.write(f"- **Всего папок с журналами:** {total}\n")
        f.write(f"- **С готовым DJVU:** {with_djvu} ({with_djvu*100//total if total else 0}%)\n")
        f.write(f"- **Полностью обработано Gigapixel:** {complete} ({complete*100//total if total else 0}%)\n")
        f.write(f"- **Частично обработано Gigapixel:** {partial} ({partial*100//total if total else 0}%)\n")
        f.write(f"- **Не обработано Gigapixel:** {none} ({none*100//total if total else 0}%)\n\n")
        
        # Найденные суффиксы
        f.write("## Найденные суффиксы Gigapixel\n\n")
        if all_suffixes:
            for suffix in sorted(all_suffixes):
                count = sum(1 for r in results if suffix in r['gigapixel_suffixes'])
                f.write(f"- `{suffix}` — используется в {count} папках\n")
        else:
            f.write("*Суффиксы не найдены*\n")
        f.write("\n")
        
        # Остальные категории
        other_categories = [
            ("С готовым DJVU", lambda r: r['has_djvu']),
            ("Полностью обработано Gigapixel (без DJVU)", lambda r: r['status'] == 'complete' and not r['has_djvu']),
        ]
        
        for category_name, filter_func in other_categories:
            filtered = [r for r in results if filter_func(r)]
            f.write(f"## {category_name}\n\n")
            f.write(f"**Количество:** {len(filtered)}\n\n")
            
            if filtered:
                for r in filtered:
                    f.write(f"### {r['path']}\n\n")
                    f.write(f"- **Путь:** `{r['full_path']}`\n")
                    f.write(f"- **Исходных файлов:** {r['original_count']}\n")
                    f.write(f"- **Обработано файлов:** {r['processed_count']}\n")
                    
                    if r['gigapixel_suffixes']:
                        f.write(f"- **Суффиксы:** {', '.join(f'`{s}`' for s in r['gigapixel_suffixes'])}\n")
                    
                    f.write("\n")
            else:
                f.write("*Нет папок в этой категории*\n\n")


def main():
    root_path = "/mnt/dump3/DOWN/Плановое хозяйство (1931-1989) [pics_only]"
    output_path = "/home/felix/Projects/ocr_utils/gigapixel.md"
    
    print(f"Анализирую директорию: {root_path}")
    results = analyze_root_directory(root_path)
    
    print(f"Найдено папок с журналами: {len(results)}")
    print(f"Генерирую отчёт: {output_path}")
    generate_markdown_report(results, output_path)
    
    print("Готово!")


if __name__ == "__main__":
    main()
