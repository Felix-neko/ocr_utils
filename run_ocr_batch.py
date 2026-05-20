#!/usr/bin/env python3
"""Скрипт для массовой оцифровки PDF-файлов из директории."""

from pathlib import Path

from ocr_utils import process_directory

# Параметры
SRC_DIR = Path("/mnt/dump3/DOWN/Плановое хозяйство (1931-1989)")
DST_DIR = Path("/mnt/dump3/DOWN/Плановое хозяйство (1931-1989) [распознанное]")

# Параметры OCR (аналогично pipeline.py)
LANGUAGE = "rus"
OVERSAMPLE_DPI = 600
DESKEW = True
CLEAN = True
ROTATE_PAGES = True


def main():
    """Запуск массовой оцифровки."""
    print("=== Массовая оцифровка PDF ===")
    print(f"Исходная директория: {SRC_DIR}")
    print(f"Выходная директория: {DST_DIR}")
    print("Параметры OCR:")
    print(f"  - Язык: {LANGUAGE}")
    print(f"  - Oversample DPI: {OVERSAMPLE_DPI}")
    print(f"  - Deskew: {'включён' if DESKEW else 'выключен'}")
    print(f"  - Clean: {'включён' if CLEAN else 'выключен'}")
    print(f"  - Rotate: {'включён' if ROTATE_PAGES else 'выключен'}")
    print()

    # Проверка существования исходной директории
    if not SRC_DIR.exists():
        print(f"Ошибка: исходная директория не найдена: {SRC_DIR}")
        return 1

    # Подсчёт количества PDF-файлов
    pdf_files = list(SRC_DIR.rglob("*.pdf"))
    print(f"Найдено PDF-файлов: {len(pdf_files)}")
    print()

    # Запуск обработки
    print("Запуск обработки...")
    print()

    results = process_directory(
        src_dir=SRC_DIR,
        dst_dir=DST_DIR,
        language=LANGUAGE,
        oversample_dpi=OVERSAMPLE_DPI,
        deskew=DESKEW,
        clean=CLEAN,
        rotate_pages=ROTATE_PAGES,
    )

    # Подсчёт результатов
    success_count = sum(1 for error in results.values() if error is None)
    error_count = sum(1 for error in results.values() if error is not None)

    print()
    print("=== Обработка завершена ===")
    print(f"Успешно обработано: {success_count}")
    print(f"Ошибок: {error_count}")

    if error_count > 0:
        print("\nФайлы с ошибками:")
        for filename, error in results.items():
            if error is not None:
                print(f"  - {filename}: {error}")
        return 1

    print(f"\nРезультаты сохранены в: {DST_DIR}")
    return 0


if __name__ == "__main__":
    exit(main())
