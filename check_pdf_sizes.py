#!/usr/bin/env python3
"""Скрипт для сравнения размеров PDF-файлов и исходных изображений."""

from pathlib import Path


def format_size(size_bytes: int) -> str:
    """Форматирует размер в человекочитаемый вид."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def analyze_pdf_sizes(root_dir: Path):
    """Анализирует размеры PDF и исходных изображений."""
    pdf_files = sorted(root_dir.rglob("*.pdf"))
    
    print(f"Найдено PDF-файлов: {len(pdf_files)}\n")
    
    total_pdf_size = 0
    total_images_size = 0
    
    for pdf_path in pdf_files:
        pdf_size = pdf_path.stat().st_size
        total_pdf_size += pdf_size
        
        pics_dir = pdf_path.parent / (pdf_path.stem + ".page_pics")
        
        if not pics_dir.exists():
            print(f"⚠ Папка не найдена: {pics_dir.name}")
            continue
        
        image_files = list(pics_dir.glob("_*"))
        images_size = sum(f.stat().st_size for f in image_files)
        total_images_size += images_size
        
        ratio = (pdf_size / images_size * 100) if images_size > 0 else 0
        
        print(f"📄 {pdf_path.name}")
        print(f"   PDF: {format_size(pdf_size)}")
        print(f"   Исходные изображения: {format_size(images_size)} ({len(image_files)} файлов)")
        print(f"   Соотношение: {ratio:.1f}%")
        print()
    
    print("=" * 60)
    print(f"ИТОГО:")
    print(f"  Все PDF: {format_size(total_pdf_size)}")
    print(f"  Все исходные изображения: {format_size(total_images_size)}")
    if total_images_size > 0:
        total_ratio = total_pdf_size / total_images_size * 100
        print(f"  Общее соотношение: {total_ratio:.1f}%")


if __name__ == "__main__":
    root = Path("/mnt/dump3/DOWN/Плановое хозяйство (1931-1989) [pics_only]")
    analyze_pdf_sizes(root)
