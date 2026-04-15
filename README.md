# ocr_utils

Утилита для постобработки PDF-сканов журналов: разбивка разворотов на отдельные страницы и добавление OCR-слоя максимального качества.

## Установка

### Системные зависимости

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-rus libjpeg-turbo-progs

# Arch
sudo pacman -S tesseract tesseract-data-rus libjpeg-turbo
```

### Python-пакет

```bash
uv sync
```

## Алгоритм работы

### Шаг 1: Разбивка разворотов на страницы

1. Для каждой страницы исходного PDF определяем соотношение сторон (aspect ratio = ширина / высота).
2. Если aspect ratio ≥ **1.152** (`288/250`) — считаем страницу разворотом и разбиваем пополам по вертикали.
3. Если aspect ratio < 1.152 — это обычная страница, оставляем как есть.
4. **Для JPEG-изображений** используется `jpegtran` для **lossless crop** — обрезка без перекодирования, чтобы не вносить дополнительные артефакты сжатия. На краях, не кратных размеру MCU-блока, используется padding.
5. Для изображений в других форматах — обрезка через PyMuPDF.
6. Результат — промежуточный PDF #1 с разбитыми страницами.

### Шаг 2: OCR через ocrmypdf

1. Промежуточный PDF #1 передаётся в `ocrmypdf` с параметрами:
   - **oversample** = 900 DPI (высокое разрешение для лучшего распознавания)
   - **deskew** — выравнивание наклонённых страниц
   - **clean** — очистка шума
   - **rotate-pages** — автоповорот по ориентации текста
   - **language** = rus
   - **optimize** = 0 (не пережимать изображения)
   - **skip-text** = true (не перераспознавать страницы с текстом)
2. Результат — финальный PDF с исходными изображениями + текстовый OCR-слой.

## API

### `process_single_pdf` — обработка одного PDF

```python
from ocr_utils import process_single_pdf

process_single_pdf(
    src_pdf="input.pdf",          # Путь к исходному PDF
    dst_pdf="output.pdf",         # Путь к выходному PDF
    pages=None,                   # None=все, [0,2,5]=список, slice(1,10)=диапазон
    tmp_dir=None,                 # None=авто (tempfile), или путь к директории
    language="rus",               # Язык OCR
    oversample_dpi=900,           # DPI оверсемплинга
    deskew=True,                  # Выравнивание
    clean=True,                   # Очистка шума
    rotate_pages=True,            # Автоповорот
)
```

### `process_directory` — рекурсивная обработка директории

```python
from ocr_utils import process_directory

results = process_directory(
    src_dir="scans/",             # Исходная директория
    dst_dir="output/",            # Выходная директория (создаётся автоматически)
    workers=None,                 # None = 3/4 ядер, или конкретное число
    language="rus",
    oversample_dpi=900,
    deskew=True,
    clean=True,
    rotate_pages=True,
)
# results: dict {относительный_путь: None (успех) или строка ошибки}
```

Структура директорий сохраняется: если файл был в `scans/1931/issue1.pdf`, результат будет в `output/1931/issue1.pdf`.

## CLI

```bash
# Один файл
ocr-utils single input.pdf output.pdf
ocr-utils single input.pdf output.pdf --pages "0,2,5"
ocr-utils single input.pdf output.pdf --pages "1:10"
ocr-utils single input.pdf output.pdf --language rus --dpi 900

# Директория
ocr-utils dir scans/ output/
ocr-utils dir scans/ output/ --workers 4
ocr-utils dir scans/ output/ --language rus --dpi 600

# Опции
ocr-utils single -v input.pdf output.pdf   # подробный вывод
ocr-utils single -q input.pdf output.pdf   # только ошибки
ocr-utils single input.pdf output.pdf --no-deskew --no-clean --no-rotate
```

## Тестирование

```bash
uv run pytest
uv run pytest -v              # подробный вывод
uv run pytest -k splitting    # только тесты разбивки
```

## Форматирование

```bash
black -l 120 -C .
```

## Структура проекта

```
ocr_utils/
├── __init__.py      # Экспорт публичного API
├── __main__.py      # CLI-точка входа
├── config.py        # Константы и конфигурация
├── splitting.py     # Разбивка разворотов на страницы, lossless JPEG crop
├── ocr.py           # OCR-слой через ocrmypdf
└── pipeline.py      # Оркестрация: один PDF, директория, параллелизация
tests/
├── conftest.py      # Фикстуры (тестовые PDF)
├── test_cli.py      # Тесты CLI
├── test_config.py   # Тесты конфигурации
├── test_pipeline.py # Тесты пайплайна
└── test_splitting.py # Тесты разбивки
```
