"""Пути пака-1 и умолчания прогона.

Собраны в одном месте по той же причине, что и в ``run_scripts/scan_markup/pack1/common.sh``:
шаги связаны друг с другом через имена файлов, и разъехавшийся путь в одном из них заводит
второй набор результатов вместо продолжения первого, а заметить это можно очень нескоро.

Регистр в ``/mnt/system`` ЗНАЧИМ: с 2026-09-20 том смонтирован как /mnt/system строчными; прежний /mnt/SYSTEM заглавными больше не существует.
"""

from __future__ import annotations

from pathlib import Path

PACK_NAME = "пак-1"

# Выгрузка FineReader «форматированный текст»: по файлу на выпуск, 98 из 123 выпусков.
DOCX_DIR = Path("/mnt/system/raw/mts/pack1_pdf/docx_форматированный_текст")

# Промежуточные PDF, которые скармливались FineReader: одна страница — один скан.
INTERMEDIATE_PDF_DIR = Path("/mnt/system/raw/mts/pack1_pdf/full_intermediate_pdfs")

# Распознанные PDF с текстовым слоем. Число страниц совпадает с промежуточными, а текст
# читается через fitz — по нему таблица DOCX и привязывается к странице (см. mining/page_match).
RECOGNIZED_PDF_DIR = Path("/mnt/system/raw/mts/pack1_pdf/full_pdfs_binary_no_bg_brightening")

# Те же страницы, распознанные БЕЗ коррекции геометрии FineReader: 112 выпусков из 123.
# Нужны, чтобы отличить «таблица уже была кривой» от «её искорёжила коррекция».
UNCORRECTED_PDF_DIR = Path("/mnt/system/raw/mts/pack1_pdf/full_pdfs_binary_no_bg_brightening_no_geometry_correction")

# Заострённые копии сканов, 600 dpi JPEG: {год}/{выпуск}/IMG_xxxx_{1L|2R}.jpg.
SHARPENED_DIR = Path("/mnt/system/raw/mts/pack1_background_blurred_v2/sharpened")

# База разметки — ТОЛЬКО НА ЧТЕНИЕ. В ней ручная разметка, которой больше нигде нет.
MARKUP_DB = Path("/home/felix/Projects/mts_markup/pack1_reviewed.sqlite")

# Куда кладутся результаты исследования. Не в корень /mnt/dump3: его синхронит Яндекс.Диск,
# он переименовывает новые файлы поверх исходных.
DEFAULT_OUT_DIR = Path("/mnt/system/raw/mts/pack1_table_research")

# Кэш измерений — в репозитории (каталог уже в .gitignore), рядом с кодом, который его пишет.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "table_processing"

# Ручная разметка коммитится: без неё сравнение алгоритмов не воспроизвести.
LABELS_DIR = Path(__file__).resolve().parent / "labels"

# Разрешение исходных сканов. Все пиксельные константы пакета даны для него, и все
# они пересчитываются, если поданы сканы другого разрешения (см. detection/ruling.py).
SOURCE_DPI = 600

# Воркеров по умолчанию. Не 16: каждый воркер разжимает JPEG 600 dpi на 23 мегапикселя,
# и на таком прогоне упираются в память и в диск раньше, чем в счёт.
DEFAULT_JOBS = 14
