# Данные и диски

Где что лежит, на каком носителе и что там нельзя делать. Пути в run-скриптах берутся из
`run_scripts/scan_markup/pack1/common.sh` — при расхождении прав он, а не этот файл.

## Носители

| Путь | Что это | Ограничения |
|---|---|---|
| `/mnt/dump3` | 11 ТБ NTFS-3G (fuseblk) на шпинделе, `/dev/sda2` | Медленный. Чтение пака ~0,5 ТБ. Любой рекурсивный обход (`find /`, индексаторы) устраивает seek-шторм и роняет запись пайплайна в 8 раз. Поиск по файлам ограничивать каталогом, при тормозах смотреть `iostat -x -d sda`, а не профилировать код |
| `/mnt/dump3/yandex_disk_linux_baby_zergling` | Корень **живой синхронизации** демона `yandex-disk` | **Ничего туда не писать.** Новые файлы демон переименовывает поверх исходных (2026-09-01 так потерялись два исходника скана). Результаты писать вне корня, переносить руками при `yandex-disk stop`. Демон запущен не всегда: перед любой записью `yandex-disk status` |
| `/mnt/SYSTEM` | Том с данными, SSD | **Регистр значим.** `/mnt` — ext4, `/mnt/system` там отдельный пустой каталог: путь строчными не падает, а тихо пишет на системный диск |
| `~/Projects/mts_markup` | Рабочие файлы разметки пака-1, SSD | Базы, превью для CVAT, отладочные оверлеи. Базы невоспроизводимы (ручная разметка неделями) — `common.sh` бэкапит их перед каждым шагом |
| `/` (корень, `/var/lib/docker`) | Системный диск | Держится у 90%. CVAT падает по health-check при заполнении выше `CVAT_HEALTH_DISK_USAGE_MAX` (поднят до 95). Чистить только `docker builder prune -af` и `docker image prune -f`; `image prune -a` и `volume prune` — нет. Пожиратели вне докера (`~/vmware`, `~/Downloads`, `~/software`) — без спроса не трогать |
| `~/vmware/win10` | VM Windows 10 с FineReader | См. раздел «FineReader» |

## Пак-1

Журнал «Материально-техническое снабжение», 1966–1976: 11 годовых комплектов, 123 выпуска,
12 136 полос, TIFF RGB 600 dpi по ~40 МБ. Раскладка `{год}/{выпуск}/{полоса}`; все производные
каталоги её повторяют. Замер набора при 600 dpi: штрих 7 px, шаг строк 89 px.

| Переменная (common.sh) | Путь | Что там | Диск |
|---|---|---|---|
| `PACK_DIR` | `/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/Готовое/пак-1` | Оригиналы TIFF | dump3, Я.Диск |
| `MARKUP_ROOT` | `~/Projects/mts_markup` | Корень рабочих файлов разметки | SSD |
| `DB` | `$MARKUP_ROOT/pack1.sqlite` | База разметки: полосы, хеши, divisor, регионы, признаки (`is_toc`, `is_year_index`, ориентация) | SSD |
| `DB_REVIEWED` | `$MARKUP_ROOT/pack1_reviewed.sqlite` | Та же схема после ручной правки в CVAT (`from-cvat`). Копия лежит и в `run_scripts/scan_markup/pack1/` | SSD |
| `DEBUG_DIR` | `$MARKUP_ROOT/debug` | Оверлеи детекторов | SSD |
| `SHARE_ROOT` | `$MARKUP_ROOT/cvat_share` | Уменьшенные копии для CVAT (75 dpi, делитель `round(dpi/75)`, для 600 dpi — 8). Обязан лежать внутри `IMAGES_DIR` из `docker/.env` | SSD |
| `CASES_DIR`, `VALIDATE_DIR` | `$MARKUP_ROOT/некоторые проблемные картинки`, `.../validate` | Валидационные примеры дефектов, надёрганные глазами | SSD |
| `COMPARE_DIR` | `$MARKUP_ROOT/compare` | Сравнения параметров, живут до выбора | SSD |
| `TOC_DIR` | `$MARKUP_ROOT/toc` | Оглавления: `lists/{год}/{выпуск}/toc_pages.txt` для внешнего OCR, `sheets/`, `found/` | SSD |
| `LAYOUT_CACHE_DIR` | `/mnt/SYSTEM/raw/mts/pack1_table_research/layout_surya_готовое` | **Кэш surya layout**: pickle на полосу по сырым TIFF, полон на все полосы. `detect --layout-cache` читает и пополняет; секунда GPU на полосу = 3,5 ч на пак | SYSTEM |
| — | `/mnt/SYSTEM/raw/mts/pack1_table_research/layout_surya` | Тот же кэш по заострённым JPEG 150 dpi (стенд детектора таблиц) | SYSTEM |
| `CLEAN_ROOT` | `/mnt/SYSTEM/raw/mts/pack1_background_blurred_v2` | Очистка пака: `blurred/` (закрас разметки + размытие фона), `debug/` | SYSTEM |
| `SHARPENED_DIR` | `$CLEAN_ROOT/sharpened` | Заострённые копии из Capture One, собраны `run_collect_sharpened` в раскладку пака. Вход для toc, orientation, curved_lines, внешнего OCR | SYSTEM |
| `PDF_ROOT` | `/mnt/SYSTEM/raw/mts/pack1_pdf` | Промежуточные и распознанные PDF, по паку (не по годам — одно задание FineReader вместо одиннадцати) | SYSTEM |
| `FULL_PDF_DIR` | `$PDF_ROOT/full_intermediate_pdfs` | Полные промежуточные PDF под FineReader | SYSTEM |
| `PICS_ONLY_PDF_DIR` | `$PDF_ROOT/intermediate_pdfs_pages_with_pics_only` | Только полосы с растром | SYSTEM |
| `GEO_PDF_DIR` | `$PDF_ROOT/full_pdfs_binary_no_bg_brightening` | FineReader по полным промежуточным PDF: бинаризация **и** коррекция геометрии | SYSTEM |
| `NOGEO_PDF_DIR` | `$PDF_ROOT/full_pdfs_binary_no_bg_brightening_no_geometry_correction` | То же **без** коррекции геометрии: образ страницы = полоса + поля пиксель в пиксель, сюда возвращаются иллюстрации | SYSTEM |
| `FINAL_PDF_DIR` | `$PDF_ROOT/final_pdfs` | Финальные PDF, `{год}_{выпуск}.pdf`, все в одной папке | SYSTEM |
| `FINAL_WORK_DIR` | `/mnt/SYSTEM/raw/mts/pack1_final_pdfs_work` | Рабочий каталог сборщика: `pages/<pdf>/pNNNN.json` (анализ страницы: источник, зоны, вердикты, чтения), `analysis.csv`, `pages.csv`, `summary.csv`, `preview/` | SYSTEM |
| — | `/mnt/SYSTEM/raw/mts/pack1_geometry_regression/pack1_v12` | Прогон детектора порчи геометрии: `cache/<pdf>/pNNN.json`, `metrics.csv`; сборщик читает кэш и дописывает промахи | SYSTEM |
| — | `/mnt/SYSTEM/raw/mts/pack1_text_layer_fix/pack1_v1` | Исследование текстового слоя по выборке 1050 стр.: кэш, CSV, оверлеи, исправленные копии | SYSTEM |
| `FINEREADER_PDF_DIR` | `$PDF_ROOT/full_pdfs_binary_brightened_bg` | Текстовый слой FineReader как прокси-эталон для внешнего OCR (страница i = i-я полоса по сортировке имён) | SYSTEM |
| `RECOGNIZED_PDF_DIR` | `$PDF_ROOT/full_pdfs_binary_no_bg_brightening` | Распознанные PDF для привязки DOCX → скан | SYSTEM |
| `DOCX_DIR` | `$PDF_ROOT/docx_форматированный_текст` | Выгрузка FineReader в DOCX. **Номера страниц DOCX ≠ страницам PDF**: привязка только по редким токенам через текстовый слой PDF | SYSTEM |
| `OUT_DIR` (table_processing) | `/mnt/SYSTEM/raw/mts/pack1_table_research` | Исследование таблиц, `JOBS=12` (память и диск, не счёт) | SYSTEM |
| `ROTATED_INFO_DIR` | `/mnt/SYSTEM/raw/mts/pack1_rotated_tables` | Таблицы с повёрнутым текстом: `sheets/`, `pairs/`, `after/`, `info/`, `summary.csv` | SYSTEM |
| `EXTERNAL_OCR_ROOT` | `/mnt/SYSTEM/raw/mts/pack1_external_ocr` | Выходы внешних OCR-моделей: подпапка на модель, `{год}/{выпуск}/полоса.{json,md,meta.json}`. `EXTERNAL_OCR_JOBS=4` — лимиты провайдеров | SYSTEM |
| `EXTERNAL_OCR_SERVICES_ROOT` | `/mnt/SYSTEM/raw/mts/pack1_external_ocr_services` | Боевой внешний OCR: `out/{год}/{выпуск}/полоса.{md,json,meta.json}` + `toc.json`/`toc.md` на выпуск, рядом `out/{год}/{выпуск}.md` (весь выпуск одним файлом) и `{выпуск}.pages.json` (привязка полос), `debug/` с сырыми ответами, промптами и тайлами, `cache/` — кэш запросов (папка на каждый запрос к модели по всем проходам; не удалять — это «продолжить с места»), `probe/` для проб. `missed_toc.txt` в `out/` — оглавления, найденные моделью вне базы | SYSTEM |
| `EXTERNAL_OCR_PROBE_ROOT` | `/mnt/SYSTEM/raw/mts/pack1_external_ocr_probe` | Пробник: 13 полос на всех моделях, отдельно, чтобы полные прогоны не затирали | SYSTEM |

Ключ OpenRouter — только из `$OPENROUTER_API_KEY`, в скрипты и логи не попадает.

## Другие паки и данные

| Путь | Что там |
|---|---|
| `/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе` | Оригиналы других паков МТС (тоже под Я.Диском) |
| `/mnt/dump3/DOWN/{год}-{месяц}` | Съёмка газет, Fujifilm RAF; `defocus_detection` и `select_best_raws` работают по встроенным JPEG-превью |
| `tests/ПХ_1931_02_03_кадры_с_просвечивающей_бумагой` | Тестовые кадры для show-through в репо |
| `finger_models/`, `dewarp_models/`, `third_party/` | Веса и внешний код, вне git; качаются автоматически, кроме DocShadow |
| `ai_slop/` | Черновой код от ИИ, вне git |

## CVAT

- Инстанс этого проекта: `docker/`, compose-проект **`cvat_mts`**, порт **8081** (8080 оставлен под
  airflow; dev-CVAT пользователя в `~/Projects/cvat` тоже хочет 8080). `./up.sh` / `./down.sh`
  (`--wipe` — со сбросом). Учётки `admin/admin`, `user/user` (worker). Организация `masochists`,
  проект «Материально-техническое снабжение».
- Картинки берутся **только** из смонтированного `IMAGES_DIR` (`docker/.env`), по сети не
  передаются. После смены `IMAGES_DIR` — `cd docker && SKIP_BOOTSTRAP=1 ./up.sh`.
- Разметка приходит в координатах уменьшенных копий — умножать на `divisor` из базы, не на константу.
- Метки: растр = rectangle, печать/надпись = mask; теги полос «Оглавление», «Годовой указатель».
- Здоровье: `curl -s -H 'Accept: application/json' 'http://localhost:8081/api/server/health/?format=json'`
  (`/api/server/about` отдаёт 200 даже когда UI пишет «services are not healthy»).

## FineReader (win10 VM)

Подробно — `docs/win10_vm.md`. Коротко:

- `~/vmware/win10/win10.vmx`, гость 192.168.116.128, ABBYY FineReader PDF 16. Диск 400 GiB.
- **CLI не работает**: ключи командной строки лицензируются, `FineReader.exe <файл> /out ...`
  открывает документ и висит. Пакетный путь — только **Hot Folder** (задача настраивается в GUI).
  `HotFolderCoreCount` может резать параллелизм сильнее 16 ядер хоста.
- Доступ с хоста: `vmrun -T ws -gu admin -gp <пароль>` guest ops (`copyFileToGuest`,
  `runProgramInGuest`; всю команду cmd одним аргументом, вывод в CP866). Пароль у пользователя.
  Прав администратора guest ops не дают — всё, что требует UAC, отдаётся пользователю .bat-файлом.
- Общие папки в гостя: `/mnt/dump3` → `DUMP`, `/mnt/SYSTEM` → `system`.
- Файлы из идущего экспорта (FineReader, Capture One, ScanTailor) открывать только при `mtime`
  старше 10 минут: недописанный TIFF/JPEG неотличим от готового.
