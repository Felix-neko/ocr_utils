#!/usr/bin/env python3
"""Проверка, что картинки сохранены без перекодировки."""

import io
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

# Исходный PDF
src_pdf = Path("/mnt/dump3/DOWN/Плановое хозяйство 1965-1975/1966/Плановое хозяйство № 1-1966.pdf")
# Директория с сохранёнными картинками
pics_dir = Path("/mnt/dump3/DOWN/Плановое хозяйство (1931-1989) [распознанное] [pics_only]/1966/Плановое хозяйство № 1-1966.page_pics")

print(f"Исходный PDF: {src_pdf}")
print(f"Директория с картинками: {pics_dir}")
print()

# Открываем PDF
doc = fitz.open(str(src_pdf))
print(f"Количество страниц в PDF: {len(doc)}")

# Проверяем первые 5 страниц
for page_idx in range(min(5, len(doc))):
    page = doc[page_idx]
    
    # Извлекаем изображение из PDF
    images = page.get_images(full=True)
    if not images:
        print(f"\nСтраница {page_idx}: нет изображений, пропускаем")
        continue
    
    xref = images[0][0]
    img_dict = doc.extract_image(xref)
    img_data_pdf = img_dict["image"]
    img_ext = img_dict["ext"]
    
    print(f"\nСтраница {page_idx}:")
    print(f"  Формат в PDF: {img_ext}")
    print(f"  Размер данных в PDF: {len(img_data_pdf)} байт")
    
    # Загружаем изображение из PDF как numpy array
    img_pdf = Image.open(io.BytesIO(img_data_pdf))
    arr_pdf = np.array(img_pdf)
    
    # Ищем соответствующую сохранённую картинку
    saved_pics = sorted(pics_dir.glob(f"page_{page_idx:04d}.*"))
    if not saved_pics:
        print(f"  ❌ Сохранённая картинка не найдена!")
        continue
    
    saved_pic = saved_pics[0]
    print(f"  Сохранённая картинка: {saved_pic.name}")
    print(f"  Размер файла: {saved_pic.stat().st_size} байт")
    
    # Загружаем сохранённую картинку
    saved_data = saved_pic.read_bytes()
    img_saved = Image.open(saved_pic)
    arr_saved = np.array(img_saved)
    
    # Сравниваем размеры
    print(f"  Размер массива из PDF: {arr_pdf.shape}")
    print(f"  Размер массива из файла: {arr_saved.shape}")
    
    if arr_pdf.shape != arr_saved.shape:
        print(f"  ❌ Размеры не совпадают!")
        continue
    
    # Сравниваем побайтово исходные данные
    if img_data_pdf == saved_data:
        print(f"  ✅ Данные ПОЛНОСТЬЮ идентичны (побайтовое сравнение)")
    else:
        # Сравниваем пиксели
        diff = arr_pdf != arr_saved
        diff_pixels = np.sum(diff)
        total_pixels = arr_pdf.size
        diff_percent = (diff_pixels / total_pixels) * 100
        
        print(f"  Побайтовое сравнение: данные отличаются")
        print(f"  Несовпадающих пикселей: {diff_pixels} из {total_pixels} ({diff_percent:.6f}%)")
        
        if diff_percent == 0:
            print(f"  ✅ Пиксели ПОЛНОСТЬЮ идентичны")
        else:
            print(f"  ❌ Есть различия в пикселях!")
            
            # Показываем максимальное отличие
            if arr_pdf.dtype == arr_saved.dtype:
                max_diff = np.max(np.abs(arr_pdf.astype(int) - arr_saved.astype(int)))
                print(f"  Максимальное отличие значения пикселя: {max_diff}")

doc.close()
print("\n" + "="*60)
print("Проверка завершена")
