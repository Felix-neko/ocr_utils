#!/usr/bin/env python3
"""
Скрипт для учёта рабочего времени сотрудников при фотографировании газетных подшивок.

Анализирует RAF-файлы в директории, определяет время начала и окончания работы,
общее время работы и все перерывы >= 15 минут.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import rawpy
from PIL import Image
from PIL.ExifTags import TAGS
from pillow_heif import register_heif_opener
from tqdm import tqdm

register_heif_opener()


def extract_datetime_from_image(file_path: Path) -> datetime | None:
    """
    Извлекает дату и время съёмки из файла изображения (RAF, HEIC, JPG/JPEG).

    Args:
        file_path: Путь к файлу изображения

    Returns:
        datetime объект или None, если не удалось извлечь
    """
    file_ext = file_path.suffix.lower()

    if file_ext in [".raf"]:
        return extract_datetime_from_raf(file_path)
    elif file_ext in [".jpg", ".jpeg", ".heic"]:
        return extract_datetime_from_exif(file_path)
    else:
        return None


def extract_datetime_from_raf(file_path: Path) -> datetime | None:
    """
    Извлекает дату и время съёмки из RAF-файла.

    Args:
        file_path: Путь к RAF-файлу

    Returns:
        datetime объект или None, если не удалось извлечь
    """
    try:
        with rawpy.imread(str(file_path)) as raw:
            timestamp = raw.other.timestamp
            if timestamp:
                return timestamp

    except Exception as e:
        print(f"⚠ Ошибка при чтении {file_path}: {e}", file=sys.stderr)

    return None


def extract_datetime_from_exif(file_path: Path) -> datetime | None:
    """
    Извлекает дату и время съёмки из EXIF-данных (JPG/JPEG/HEIC).

    Args:
        file_path: Путь к файлу изображения

    Returns:
        datetime объект или None, если не удалось извлечь
    """
    try:
        with Image.open(file_path) as img:
            exif_data = img.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "DateTimeOriginal" or tag == "DateTime":
                        try:
                            return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                        except ValueError:
                            pass

    except Exception as e:
        print(f"⚠ Ошибка при чтении {file_path}: {e}", file=sys.stderr)

    return None


def find_image_files(directory: Path) -> List[Path]:
    """
    Рекурсивно находит все поддерживаемые файлы изображений в директории.

    Args:
        directory: Директория для поиска

    Returns:
        Список путей к файлам изображений
    """
    extensions = ["*.RAF", "*.raf", "*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.HEIC", "*.heic"]
    files = []
    for ext in extensions:
        files.extend(directory.rglob(ext))
    return sorted(files)


def analyze_work_time(directory: Path, min_break_minutes: int = 15) -> None:
    """
    Анализирует рабочее время на основе файлов изображений в директории.

    Args:
        directory: Директория с файлами изображений
        min_break_minutes: Минимальная длительность перерыва в минутах (по умолчанию 15)
    """
    print(f"🔍 Поиск файлов изображений в: {directory}")
    image_files = find_image_files(directory)

    if not image_files:
        print("❌ Файлы изображений не найдены")
        return

    print(f"✓ Найдено файлов изображений: {len(image_files)}")
    print()

    print("📸 Извлечение времени съёмки...")
    timestamps: List[Tuple[datetime, Path]] = []

    for image_file in tqdm(image_files, desc="Обработка файлов", unit="файл"):
        dt = extract_datetime_from_image(image_file)
        if dt:
            timestamps.append((dt, image_file))

    if not timestamps:
        print("❌ Не удалось извлечь время съёмки ни из одного файла")
        return

    timestamps.sort(key=lambda x: x[0])

    print(f"✓ Успешно обработано файлов: {len(timestamps)} из {len(image_files)}")
    print()

    start_time = timestamps[0][0]
    end_time = timestamps[-1][0]
    total_duration = end_time - start_time

    print("=" * 80)
    print("📊 СТАТИСТИКА РАБОЧЕГО ВРЕМЕНИ")
    print("=" * 80)
    print()
    print(f"🕐 Начало работы:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🕐 Окончание работы: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱  Общее время:      {format_duration(total_duration)}")
    print()

    breaks = find_breaks(timestamps, min_break_minutes)

    if breaks:
        print(f"☕ ПЕРЕРЫВЫ (>= {min_break_minutes} минут):")
        print("-" * 80)
        total_break_time = timedelta()

        for i, (break_start, break_end, duration) in enumerate(breaks, 1):
            print(f"{i}. {break_start.strftime('%H:%M:%S')} → {break_end.strftime('%H:%M:%S')} " f"({format_duration(duration)})")
            total_break_time += duration

        print("-" * 80)
        print(f"Всего времени на перерывы: {format_duration(total_break_time)}")
        print()

        net_work_time = total_duration - total_break_time
        print(f"⏱  Чистое рабочее время: {format_duration(net_work_time)}")
    else:
        print(f"✓ Перерывов >= {min_break_minutes} минут не обнаружено")
        net_work_time = total_duration

    print()

    net_work_hours = net_work_time.total_seconds() / 3600
    if net_work_hours > 0:
        photos_per_hour = len(timestamps) / net_work_hours
        print(f"📈 Производительность: {photos_per_hour:.1f} фотографий/час")
        print()

    print("=" * 80)


def find_breaks(timestamps: List[Tuple[datetime, Path]], min_break_minutes: int) -> List[Tuple[datetime, datetime, timedelta]]:
    """
    Находит все перерывы >= заданной длительности.

    Args:
        timestamps: Список кортежей (время, путь к файлу)
        min_break_minutes: Минимальная длительность перерыва в минутах

    Returns:
        Список кортежей (начало перерыва, конец перерыва, длительность)
    """
    breaks = []
    min_break_duration = timedelta(minutes=min_break_minutes)

    for i in range(len(timestamps) - 1):
        current_time = timestamps[i][0]
        next_time = timestamps[i + 1][0]
        gap = next_time - current_time

        if gap >= min_break_duration:
            breaks.append((current_time, next_time, gap))

    return breaks


def format_duration(duration: timedelta) -> str:
    """
    Форматирует длительность в читаемый вид.

    Args:
        duration: Длительность

    Returns:
        Строка вида "2ч 30м 15с"
    """
    total_seconds = int(duration.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours}ч")
    if minutes > 0:
        parts.append(f"{minutes}м")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}с")

    return " ".join(parts)


def main():
    """Главная функция скрипта."""
    parser = argparse.ArgumentParser(
        description="Учёт рабочего времени при фотографировании газетных подшивок",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("directory", type=str, help="Входная директория с файлами изображений (RAF, JPG/JPEG, HEIC)")

    parser.add_argument(
        "--min-break",
        type=int,
        default=15,
        metavar="МИНУТЫ",
        help="Минимальная длительность перерыва в минутах (по умолчанию: 15)",
    )

    args = parser.parse_args()

    directory = Path(args.directory)

    if not directory.exists():
        print(f"❌ Директория не существует: {directory}", file=sys.stderr)
        sys.exit(1)

    if not directory.is_dir():
        print(f"❌ Указанный путь не является директорией: {directory}", file=sys.stderr)
        sys.exit(1)

    analyze_work_time(directory, args.min_break)


if __name__ == "__main__":
    main()
