#!/bin/bash

# Скрипт для массовой оцифровки PDF-файлов
# Использует ocr_utils для обработки всей директории с журналами

set -euo pipefail

# Параметры
SRC_DIR="/mnt/dump3/DOWN/Плановое хозяйство (1931-1989)"
DST_DIR="/mnt/dump3/DOWN/Плановое хозяйство (1931-1989) [распознанное]"

# Параметры OCR (аналогично pipeline.py)
LANGUAGE="rus"
UPSCALE_RATIO=2.0
WORKERS=""  # Пусто = автоматически (3/4 ядер)

# Флаги OCR
DESKEW="--deskew"
CLEAN="--clean"
ROTATE="--rotate"

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Массовая оцифровка PDF ===${NC}"
echo "Исходная директория: ${SRC_DIR}"
echo "Выходная директория: ${DST_DIR}"
echo "Параметры OCR:"
echo "  - Язык: ${LANGUAGE}"
echo "  - Upscale Ratio: ${UPSCALE_RATIO}"
echo "  - Deskew: включён"
echo "  - Clean: включён"
echo "  - Rotate: включён"
echo "  - Workers: автоматически (3/4 ядер)"
echo ""

# Проверка существования исходной директории
if [ ! -d "${SRC_DIR}" ]; then
    echo -e "${RED}Ошибка: исходная директория не найдена: ${SRC_DIR}${NC}"
    exit 1
fi

# Создание выходной директории
mkdir -p "${DST_DIR}"

# Подсчёт количества PDF-файлов
PDF_COUNT=$(find "${SRC_DIR}" -type f -name "*.pdf" | wc -l)
echo -e "${YELLOW}Найдено PDF-файлов: ${PDF_COUNT}${NC}"
echo ""

# Запуск обработки
echo -e "${GREEN}Запуск обработки...${NC}"
echo ""

# Формируем команду
CMD="uv run python -m ocr_utils -v dir \"${SRC_DIR}\" \"${DST_DIR}\" \
    --language ${LANGUAGE} \
    --upscale-ratio ${UPSCALE_RATIO} \
    ${DESKEW} \
    ${CLEAN} \
    ${ROTATE}"

# Добавляем workers если указано
if [ -n "${WORKERS}" ]; then
    CMD="${CMD} --workers ${WORKERS}"
fi

# Выполняем команду
eval ${CMD}

# Проверка результата
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== Обработка завершена успешно ===${NC}"
    echo "Результаты сохранены в: ${DST_DIR}"
    
    # Подсчёт обработанных файлов
    PROCESSED_COUNT=$(find "${DST_DIR}" -type f -name "*.pdf" | wc -l)
    echo -e "${GREEN}Обработано файлов: ${PROCESSED_COUNT}${NC}"
else
    echo ""
    echo -e "${RED}=== Обработка завершилась с ошибками ===${NC}"
    exit 1
fi
