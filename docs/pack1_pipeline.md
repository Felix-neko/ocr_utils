# Конвейер пака-1

Пак-1 — журнал «Материально-техническое снабжение», 1966–1976: 11 годовых комплектов, 123 выпуска, 12 135 полос TIFF RGB 600 dpi; раскладка каталогов и смысл переменных `common.sh` (`PACK_DIR`, `MARKUP_ROOT`, `DB`, `DB_REVIEWED`, `SHARE_ROOT`, `CLEAN_ROOT`, `SHARPENED_DIR`, `PDF_ROOT`, `LAYOUT_CACHE_DIR`) — в [docs/data_layout.md](data_layout.md).

## Основная цепочка (шаги 0–9)

Все скрипты в `run_scripts/scan_markup/pack1/`, каждый делает `source common.sh`: тот переходит в корень репозитория и кладёт `$DB.bak` перед любым шагом.

| Шаг | Скрипт | Команда `python -m …` | Читает | Пишет | Замечания |
|---|---|---|---|---|---|
| 0 | `run_0_validate.sh` | `ocr_utils.scan_markup validate` | `CASES_DIR`, `PACK_DIR`, `DB` | `VALIDATE_DIR` | Калибровка порогов на ~100 файлах (полминуты) вместо полного прогона; нужен GPU; `--control 40` ловит обратную ошибку. Гнать ПЕРЕД шагом 1. |
| 1 | `run_1_detect.sh` | `scan_markup detect`, затем `scan_markup toc` | `PACK_DIR` (~0.5 ТБ), `LAYOUT_CACHE_DIR` | `DB`, `DEBUG_DIR`, кэш layout | Первичный детектор — surya (GPU), пиксельные проверки в 16 воркерах; ~2.2 ч. Идемпотентен через `--skip-detected`. Ориентация и таблицы считаются этим же проходом, чтобы не читать пак дважды. `toc` идёт после `detect` (берёт дерево выпусков из базы) и с `SHARPENED_DIR`, если тот есть. |
| 2 | `run_2_to_cvat.sh` | `scan_markup to-cvat` | `DB`, `PACK_DIR` | `SHARE_ROOT`, задачи CVAT | Требует поднятого CVAT и `SHARE_ROOT` внутри `IMAGES_DIR` из `docker/.env`. Последнее чтение оригиналов. `--annotator user` обязателен. `--force-annotations` затирает ручную правку; `--recreate-stale` — только после шага 3. `--append-kinds`/`--append-tags` дозаливают PATCH-ем, не трогая ручное. |
| — | **ручной шаг** | — | CVAT | CVAT | Разметчик правит рамки растра, рисует кистью печати и надписи, ставит теги оглавления. Недели работы, на диске больше нигде нет. |
| 3 | `run_3_from_cvat.sh` | `scan_markup from-cvat` | `DB`, CVAT | `DB_REVIEWED` | Отдельный файл, автодетекция в `DB` остаётся нетронутой. Оригиналы не читаются. Полезно прогонять «просто так» как снимок разметки. |
| 4 | `run_4_update_pages.sh` | `detect`, `to-cvat --recreate-stale`, `count-shapes` | `PACK_DIR`, `DB`, CVAT | `DB`, CVAT, `MARKUP_ROOT/backup_update_*` | Точечно, по именам обновившихся полос. Сам делает именной бэкап sqlite и `pg_dump` CVAT, сверяет число шейпов до/после. Задача-года пересоздаётся целиком (CVAT не даёт подменить кадр). «Завершён» с выпуска не снимает — решает человек. |
| 5 | `run_5_compare_inpaint.sh` | `ocr_utils.scan_cleanup compare-inpaint` | `DB_REVIEWED`, `PACK_DIR` | `COMPARE_DIR/inpaint` | До шага 7: выбирает бэкенд закраса. Итог уже получен — LaMa, `--lama-hole-max-px 300`; SD выдумывает псевдотекст. Перегонять только при смене материала. |
| 6 | `run_6_compare_masks.sh` | `scan_cleanup compare-masks` | `DB_REVIEWED`, `PACK_DIR` | `COMPARE_DIR/masks` | До шага 7: выбирает `--method`, `--dilate-px`, `--blur-px`. Смотреть кропы 1:1 на бледные перемычки букв. |
| 7 | `run_7_cleanup.sh` | `scan_cleanup run` | `DB_REVIEWED`, `PACK_DIR` | `BLURRED_DIR`, `CLEAN_DEBUG_DIR`, `report.csv`, `pages.cleaned_rel_path` | Боевая очистка: закрас + размытие фона одним проходом. Имена содержат отпечаток sha256 исходника. Полосы без цветной разметки уходят в серый (167 цветных из 12 135). ~101 ГиБ. `--jobs 8` — упор в диск. Перед полным прогоном проба на `--only-year 1976`. Все размеры в пикселях, для другого пака перемерять. |
| 8 | `run_8_migrate_db.sh` | `scan_markup.db.migrate` | `DB`, `DB_REVIEWED`, оба `.bak` | те же файлы | Переименование колонок под сборку PDF, идемпотентно, снимает свои `.bak-до-переименования`. Гонять по всем четырём файлам, включая бэкапы. Обязателен перед `run_intermediate_pdfs_mts_pack1.sh`. |
| 9 | `run_9_copy_table_regions.sh` | `scan_markup copy-regions` | `DB` | `DB_REVIEWED` | Переносит `table`/`line_art_schema` с `source=auto`, не дожидаясь разметчика; ручной растр в целевой базе не трогается, следующий `from-cvat` заменит их уточнёнными. |

## Боковые ветки

### Оглавления (toc) — после шага 1

| Скрипт | Команда | Читает | Пишет | Замечания |
|---|---|---|---|---|
| `run_toc.sh` | `scan_markup toc` | `PACK_DIR`/`SHARPENED_DIR`, `LAYOUT_CACHE_DIR`, `TOC_LABELS` | `DB`, `TOC_DIR/{toc_pack1.csv,sheets,found}` | Отдельно от шага 1 ради калибровки: меряется только окно выпуска (~2 200 полос), минуты. Пороги через `--thr`. Выпуск без «Содержания» в выводе — красный флаг. |
| `run_toc_pages.sh` | `scan_markup toc-pages` | `DB` | `TOC_LISTS_DIR/<год>/<выпуск>/toc_pages.txt` | Списки уходят во внешний OCR как `--skip-pages` и как `--pages`. |

### Ориентация полос — на `SHARPENED_DIR`, `DB_REVIEWED` только на чтение

| Скрипт | Команда | Читает | Пишет | Замечания |
|---|---|---|---|---|
| `run_orientation.sh` | `scan_markup.orientation run` | `SHARPENED_DIR`, `DB_REVIEWED` | симлинки `orientation_pack1_rotate`, CSV, md, контактный лист | Ищет полосы с боком напечатанной иллюстрацией, на которых падает FineReader с «исправлять ориентацию». Углы 0,90 — против часовой в паке не встретилось. ~1.5 ч, `--jobs 12` (родитель занят постобработкой surya). |
| `run_orientation_validate.sh` | `scan_markup.orientation validate` | `SHARPENED_DIR` | md-отчёт | Синтетические повороты заведомо прямых полос: матрица ошибок ловит перепутанный знак угла у чужих движков. Не показывает поведение на настоящей боковой полосе. |

### Кривые строки и dewarp — вход `SHARPENED_DIR`, симлинки целят в `PACK_DIR`

| Скрипт | Команда | Читает | Пишет | Замечания |
|---|---|---|---|---|
| `run_curved_lines_validate.sh` | `scan_markup.curved_lines run --labelled-only` | 14 полос из `curved_lines_labels.csv` | md с таблицей разделения, оверлеи | Калибровка порогов по эталону пользователя (8 кривых + 6 прямых), пара минут. |
| `run_curved_lines.sh` | `scan_markup.curved_lines run` | `SHARPENED_DIR`, `DB_REVIEWED`, кэш | симлинки `…_links` (в т.ч. `combo/`), CSV, md, лист, оверлеи | Полный прогон ~1.5 ч, узкое место surya. Идёт в «щедром» режиме порогов (строгие −40%): пропуск кривой полосы опаснее ложной коррекции. |
| `run_dewarp_found.sh` | `ocr_utils.dewarp run --from-links` | `curved_lines_pack1_links/combo` | `pack1_dewarp/found/<движок>`, 300 dpi | Требует прогона `run_curved_lines.sh` — падает, если каталога нет. Часы, в основном pagedewarp. |
| `run_dewarp_validate.sh` | `ocr_utils.dewarp run --only …` | 8 эталонных полос из `SHARPENED_DIR` | `pack1_dewarp/validate` + `compare/`, `quality.{csv,md}` | Все движки на одном материале, сравнение парами «было \| стало», ~10 минут. |

### Таблицы (`run_scripts/table_processing/`) — после FineReader; своя `common.sh`, `JOBS=12`

| Скрипт | Команда (`research.legacy.table_processing`) | Читает | Пишет | Замечания |
|---|---|---|---|---|
| `run_layout_pack_source.sh` | `layout-pack --ext tif` | исходные TIFF пака | `LAYOUT_CACHE_DIR` | Именно этот кэш потребляет `detect` на шаге 1. ~3.5 ч, `--readers 3` (диск). Прерванный прогон перезапускается. |
| `run_layout_pack.sh` | `layout-pack` | `SHARPENED_DIR` | `OUT_DIR/layout_surya` | То же по заострённым копиям. GPU в одном процессе. |
| `run_mine_pack1.sh` | `mine` + `sheet` | DOCX/PDF FineReader, `SHARPENED_DIR`, `DB` | вырезки, контактные листы | Шаг 1 ветки: таблицы, которые FineReader превратил в мешанину (порог 0.20), ~230 вырезок. |
| `run_pipeline_pack1.sh` | `run` + `pairs` | вырезки | `pairs/`, `sheets/` | Шаг 2: сетка ячеек, боковой текст, tesseract (лучший по CER). |
| `run_compare_pack1.sh` | `compare-rotation`, `compare-ocr` | ручная разметка ячеек | `reports/table_processing_*.md` | Шаг 3. GPU-движки идут последовательно. PaddleOCR — в своём окружении. |
| `run_compare_detector.sh` | `compare-detector` | 190 размеченных полос | оверлеи по диагнозам | Быстрый цикл после каждой правки порога. |
| `run_check_detector_{pack1,v3,v4,v42_source}.sh` | `check-detector` | DOCX, PDF, сканы, `DB` | `reports/<своя папка>/` | Приёмка версии по всему паку (~6–15 мин). Каждая версия пишет в отдельную папку: в старых лежит ручная раскладка по диагнозам. |
| `run_audit_detector_pack1.sh` | `audit-detector` | `SHARPENED_DIR` | `detector_audit/` | Разбор принятых/отклонённых находок с оверлеями. |
| `run_geometry_pack1.sh` | `find-broken`, `diagnose-geometry`, `fix-geometry`, `geometry-pairs` | PDF с коррекцией и без | листы «было-стало» | Отличает «криво уже на скане» от «искорёжил FineReader». |

### Внешний OCR (`run_scripts/external_ocr_models/`) — вход `SHARPENED_DIR`, ключ из `$OPENROUTER_API_KEY`

| Скрипт | Команда (`research.external_ocr_models`) | Читает | Пишет | Замечания |
|---|---|---|---|---|
| `run_probe_1966_03.sh` | `run` × 9 моделей + DeepSeek в 2 и 3 куска | 13 полос `probe_pages_1966_03.txt` | `EXTERNAL_OCR_PROBE_ROOT/<модель>/` | $1–2, 20–40 мин. `EXTERNAL_OCR_JOBS=4` — лимиты провайдеров, не CPU. |
| `run_local_probe_1966_03.sh` | `run` × 4 локальных движка | те же 13 полос | `EXTERNAL_OCR_PROBE_ROOT/local-*/` | Движки строго по очереди: видеопамять одна. 20–35 с на полосу. |
| `run_issue_1966_03.sh` | `run` × 4 модели + `balance` | `SHARPENED_DIR/1966/03` | `EXTERNAL_OCR_ROOT/<модель>/` | ~80 ¢ на выпуск. DeepSeek только `--strips 2`: целой полосой упирается в 1024 токена на картинку. |
| `run_issue_deepseek.sh` | `run` + `balance` | выпуск + `TOC_LISTS_DIR/<выпуск>/toc_pages.txt` | папка на версию промпта | Зависит от `run_toc_pages.sh`: оглавление исключается из основного прогона и запрашивается отдельно. |
| `run_reports_1966_03.sh` | `report` | готовые выходы, PDF FineReader, `ROTATED_INFO_DIR` | `reports/external_ocr_models_*.md` | Ничего не запрашивает. Эталон букв — текстовый слой FineReader, фразы боковых шапок — от `rotated_text`. |
| `run_split_damaged.sh`, `run_damaged_deepseek.sh` | `split-spreads`, `run --damage` + `report` | мини-набор повреждённых сканов | `damaged/нарезанное…`, `damaged/выход/` | Линии сгиба заданы руками: автоподгонка промахивается на двухколонных страницах. |

### Внешний OCR, боевой (`run_scripts/external_ocr_services/`) — вход `SHARPENED_DIR`, `DB_REVIEWED` только на чтение

| Скрипт | Команда (`ocr_utils.external_ocr_services`) | Читает | Пишет | Замечания |
|---|---|---|---|---|
| `run_issue_sharpened.sh [год/выпуск]` | `run --only-year --only-issue` | `SHARPENED_DIR/<выпуск>`, `DB_REVIEWED` | `EXTERNAL_OCR_SERVICES_ROOT/{out,debug}/<выпуск>` | Два этапа: полосы с тегами «Оглавление»/«Годовой указатель» → `toc.json` → остальные полосы со списком статей в промпте. Интерактивный `--on-missed-toc ask`. ~10 ¢ на выпуск. |
| `run_pack_sharpened.sh` | `run --skip-done --on-missed-toc skip` | весь `SHARPENED_DIR`, `DB_REVIEWED` | те же `out/`, `debug/`, `out/missed_toc.txt` | Идемпотентен, в фон через `setsid … < /dev/null`. Оглавления вне базы копятся в `missed_toc.txt`: проставить теги в CVAT (в т. ч. вето «Не оглавление»), `run_3_from_cvat.sh`, перегнать выпуск `run_issue_sharpened.sh`. ~$12 на пак. |
| `run_probe_tiles.sh` | `run --pages` + `run` по 1975/12 | разворот `1967/10/IMG_0041`, выпуск 1975/12 | `EXTERNAL_OCR_SERVICES_ROOT/probe` | Проверка сетки тайлов (2×2 у разворота) и слияния двухполосного «Содержания» с указателем за год. |

### PDF и FineReader (`run_scripts/pdf_utils/`) — после шага 7

| Скрипт | Команда (`ocr_utils.pdf_utils`) | Читает | Пишет | Замечания |
|---|---|---|---|---|
| `run_collect_sharpened_mts_pack1.sh` | `collect_sharpened` | `BLURRED_DIR/*/*/sharpened` | `SHARPENED_DIR`, `DB_REVIEWED` | Перенос, не копия (~144 ГиБ). Можно запускать, не дожидаясь конца выгрузки Capture One: молодые файлы и неполные выпуски пропускаются. |
| `run_verify_sharpened_mts_pack1.sh` | `verify_sharpened` | `BLURRED_DIR`, `SHARPENED_DIR`, `DB_REVIEWED` | `verify_sharpened.csv` | Обязателен ПЕРЕД сборкой промежуточных PDF: ловит подмену одноимённых полос по содержимому. Пороги выверены по паку. |
| `run_intermediate_pdfs_mts_pack1.sh` | `intermediate_pdfs` | `SHARPENED_DIR`, `BLURRED_DIR`, `DB_REVIEWED` | `FULL_PDF_DIR`, `PICS_ONLY_PDF_DIR`, номера страниц в базе | Сам проверяет, что шаг 8 пройден, и отказывается работать на немигрированной базе. Поля 12/6 мм только у полной PDF. ~144 ГиБ, до 20 ГБ ОЗУ при `--jobs 8`. |
| `run_final_pdfs_mts_pack1.sh` | `final_pdfs` | `FULL_RECOGNIZED_DIR`, `PICS_ONLY_RECOGNIZED_DIR`, `BLURRED_DIR` | `FINAL_PDF_DIR` | Только после FineReader. Сначала сверяет число страниц с записанным сборщиком: выпуск с расхождением не собирается вовсе. |
| `run_finereader_compare.sh` (в `pack1/`) | `scan_markup.curved_lines.finereader_compare` | `curved_lines_pack1_links/combo`, распознанные PDF | пары «было \| стало» | Зависит от ветки кривых строк. Порядок страниц PDF = порядок файлов в папке выпуска. |
| `run_extract_images_planhoz_pack_1*.sh` | `ocr_utils.pdf_utils` | PDF «Планового хозяйства» | картинки | К паку-1 МТС не относится — другой пак, обратная задача (PDF → картинки). |

### Повёрнутый текст (`run_scripts/rotated_text/run_tables_pack1.sh`) — после шага 9

`ocr_utils.rotated_text.tables run` + `sheets`: берёт `rect_regions.kind='table'` из `DB_REVIEWED` (903 штуки) и `SHARPENED_DIR`, пишет `pack1_rotated_tables/{after,pairs,info,sheets}`. Углы все четыре (перевёрнутый текст ломает FineReader так же, как боковой), `--lang rus`, `--work-dpi 300`, `--jobs 16` (чистый CPU). `info/` потребляет отчёт внешнего OCR. ~7 минут, с `--surya` ещё ~8.

## Ручные шаги вне репо

- **CVAT** — локальный инстанс в `docker/`: `./up.sh` (идемпотентен, первый запуск долгий), `./down.sh`, `./down.sh --wipe`. Разметка ведётся на http://localhost:8081 под `user/user` в организации «Клуб мазохистов», вкладка **Jobs**: 1 задача = год, 1 джоб = выпуск. Руками: прямоугольники для растровых изображений, кисть для библиотечных печатей и рукописных надписей, теги оглавления. Роль `worker` видит только назначенное лично — списки Projects/Tasks у неё пусты; чтобы ходить по годам, нужен `maintainer`, и задаётся он только через `ANN_ROLE` в `.env` (правка через API откатывается на следующем `up.sh`). Наружу можно раздать туннелем, но пароли в `.env` дефолтные — менять перед раздачей.
- **Capture One** — заострение полос: берёт `BLURRED_DIR` (выход шага 7), кладёт в `blurred/{год}/{выпуск}/sharpened`, откуда `run_collect_sharpened` переносит в `SHARPENED_DIR`. Обрабатывает выпуски параллельно и уже однажды перепутал одноимённые полосы — отсюда `run_verify_sharpened`.
- **FineReader на win10 VM** — пакетное распознание через Hot Folder по папке (поэтому промежуточные PDF собираются по паку, а не по годам). Два задания: полные PDF — бинаризация **и** распрямление строк; `pages_with_pics_only` — бинаризация **без** распрямления, иначе иллюстрации некуда возвращать. «Исправлять ориентацию страницы» валит прогон на боковых иллюстрациях (см. ветку orientation). Результат кладётся в `FULL_RECOGNIZED_DIR` и `PICS_ONLY_RECOGNIZED_DIR` под теми же именами.
- **Глазами** — контактные листы и оверлеи почти каждой ветки: `VALIDATE_DIR`, `DEBUG_DIR`, `TOC_DIR/sheets`, `COMPARE_DIR/{inpaint,masks}`, листы находок таблиц и dewarp, «спорные» в ориентации. Валидационная выборка `CASES_DIR` набирается руками из `DEBUG_DIR`.
- **`uv sync --project research/legacy/table_processing/paddle_env`** — иначе PaddleOCR просто выпадет из сравнения.

## Подводные камни

- Шаги 0 → 1 и 5,6 → 7: калибровочные прогоны обязаны идти перед боевыми, иначе часы чтения уходят впустую.
- Шаг 8 обязателен перед сборкой промежуточных PDF; мигрировать надо и `.bak`-файлы — незамигрированный бэкап не откроется ровно тогда, когда понадобится.
- `to-cvat --force-annotations` затирает ручную разметку задачи целиком. `--recreate-stale` пересоздаёт задачу-года (CVAT не удаляет отдельные джобы) — перед ним прогнать шаг 3.
- Без `--annotator user` разметчик видит пустой список и никакой ошибки.
- `SHARE_ROOT` должен лежать внутри `IMAGES_DIR` из `docker/.env`; переименовывать `IMAGES_DIR` (или любого родителя) при живых контейнерах нельзя — bind-mount останется на старом inode, `to-cvat` падает с `FileNotFoundError`. Лечится `down.sh` + `up.sh`; заодно удалить задачу с нулём кадров от упавшего прогона.
- Координаты из CVAT — в масштабе уменьшенных копий, умножать на `divisor` из базы (`round(dpi/75)`, для 600 dpi — 8). Полосам, уже залитым в CVAT, делитель не меняется.
- Регистр `/mnt/SYSTEM` значим: `/mnt/system` — другой, пустой каталог, вывод тихо уедет на системный диск.
- В `/mnt/dump3` (корень Яндекс.Диска) ничего не писать: синхронизация переименовывает новые файлы поверх исходных.
- Перепрогон шага 7 после смены схемы имён пересчитает всё (`--skip-if-exists` ищет имя с отпечатком); старую папку `blurred` снести заранее. 1976 год уже лежит в выводе и будет пропущен — нужен `--no-skip-if-exists`.
- `JOBS` меньше 16: шаг 7 и `--jobs 8` — упор в диск (чтение и запись по ~300 ГиБ на разные тома); `table_processing` `JOBS=12` — память и диск на разжатии 20-мегапиксельных JPEG; orientation/curved_lines `--jobs 12` и validate `--jobs 14` — родитель занимает три-четыре ядра постобработкой surya и GPU; `layout_pack_source` `--readers 3` — 38 МБ на полосу с NTFS-3G; внешний OCR `--jobs 4` — лимиты провайдеров (429 у Fireworks).
- GPU-движки нельзя заворачивать в пул: видеопамять одна, они идут последовательно в родителе.
- Фоновые прогоны запускать с `set -m` и `trap 'kill -- -$$'`: пул на forkserver переживает смерть хозяина, осиротевшие воркеры жили почти три часа и вчетверо замедляли соседний прогон. Ждать по сохранённому PID, а не по имени процесса.
- `intermediate_pdfs` не перекодирует полосы (jpegtran по DCT-блокам) и честно падает, если не справился, вместо тихой пересжатой подмены.
- `final_pdfs` отказывается собирать выпуск, у которого FineReader изменил число страниц: номера страниц в базе к такому выводу уже не относятся.
- После сборки промежуточных PDF `SHARPENED_DIR` можно снести — байты лежат внутри PDF, это те же 144 ГиБ.
