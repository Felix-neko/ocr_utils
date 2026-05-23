#!/usr/bin/env python3
"""
Тест eynollah на одном файле для быстрой проверки.
"""

from pathlib import Path
import sys

from eynollah.eynollah import Eynollah
from eynollah.model_zoo import EynollahModelZoo

if __name__ == '__main__':
    input_file = Path("/mnt/dump3/DOWN/1975-12/out").glob("*.tif").__next__()
    output_dir = Path("/tmp/eynollah_test")
    model_dir = Path.home() / ".local" / "share" / "eynollah" / "models_eynollah"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Входной файл: {input_file}")
    print(f"Выходная директория: {output_dir}")
    print(f"Директория с моделями: {model_dir}")

    if not model_dir.exists():
        print(f"⚠ Модели не найдены в {model_dir}")
        sys.exit(1)

    print("\nЗагрузка моделей...")
    model_zoo = EynollahModelZoo(str(model_dir))

    print("Создание Eynollah...")
    eynollah = Eynollah(
        model_zoo=model_zoo,
        full_layout=True,
        num_jobs=1,
    )

    print("Запуск обработки...")
    eynollah.run_single(
        img_filename=str(input_file),
        dir_out=str(output_dir),
    )

    print(f"\n✓ Готово! Результаты в {output_dir}")
