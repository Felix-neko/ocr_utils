# ocr_utils — правила проекта

Обработка сканов книг, журналов и газет: кадрирование, детекторы дефектов, разметка в CVAT,
очистка, сборка PDF под FineReader, внешний OCR. Один разработчик, только Claude Code.

## Справочники (открывать до grep по репо)

- `docs/modules.md` — карта всех модулей и классов (генерируется, не править руками).
- `docs/data_layout.md` — где лежат данные, на каких дисках, что там нельзя делать.
- `docs/pack1_pipeline.md` — сквозной конвейер пака-1: шаги, скрипты, что читают и пишут.
- `docs/status.md` — состояние направлений, **что пробовали и отвергли**, действующие пороги.
  Обновлять при завершении направления.
- `docs/win10_vm.md` — Windows-VM с FineReader: Hot Folder, доступ через vmrun, ограничения.
- README подпакетов: `ocr_utils/<пакет>/README.md` (есть у 11 из 18).

Навыки: `/run-pack1-step`, `/new-detector`, `/write-report`, `/cvat-roundtrip`, `/draw-overlay` (палитра,
легенда и полупрозрачность отладочных оверлеев; легенда и шапка — только в поле картинки, не поверх
страницы: `page_layout/overlay_frame.framed`), `/recall` (поиск по переписке прошлых сессий),
`/web-search` (веб-поиск через Chrome пользователя).
Любой поиск в интернете — по умолчанию через навык `web-search` (браузер), `WebSearch`/`WebFetch` — только
если Chrome не подключён или сбоит.
Правила по путям (`.claude/rules/`) подгружаются сами при работе с `scan_markup`, `run_scripts`, `tests`.

## Карта пакета

| Подпакет | Что делает | Точка входа |
|---|---|---|
| `scan_cropping` | Главный пайплайн кадра: YOLO-World+SAM → разворот, палец (LaMa), поворот, crop-зона | `python -m ocr_utils.scan_cropping` |
| `db` | Схема SQLite-базы разметки (пак → год → выпуск → полоса → области/маски/точки), открытие с дописыванием колонок, идемпотентная запись, миграции | `python -m ocr_utils.db.migrate` |
| `page_layout` | Разбор структуры страницы одним пакетом: `PageImage` (варианты картинки: скан / заострённая / PDF FineReader geo и no-geo), единая модель surya и JSON-кэш по варианту, детекторы растра, таблиц, line art (схемы + surya + пятна, вне растра и таблиц), повёрнутого текста, ориентации; защита выровненных сторон блоков от сора (выступы → CRAFT + pero → недостоверные участки, `text_blocks/edge_guard.py`); версии семейств; те же детекторы у `detect`, `geometry_regression`, `text_layer_fix`; внутри — `text_blocks` | `python -m ocr_utils.page_layout analyze\|prefill-surya`, `python -m ocr_utils.page_layout.orientation` |
| `page_layout.text_blocks` | **Детектор текстовых блоков**: локальные межколонники и зоны вёрстки, осевые кривые строк, гладкие огибающие текстовых блоков, выключка блока (лево/право/формат); на входе принимает подсказки `page_layout` и ячейки таблиц. Оси строк и стороны блоков (сайдкар `store.py`) — вход детектора геометрии v18 (`geometry_regression.quality`); README подпакета с примерами входа-выхода и `example.py` | `python -m ocr_utils.page_layout.text_blocks analyze` |
| `scan_markup` | Разметка пака: растр, таблицы, line art, повёрнутый текст, печати (разбор `page_layout`) → CVAT → SQLite; внутри `toc`, `curved_lines` | `python -m ocr_utils.scan_markup <команда>` |
| `scan_cleanup` | Закрас разметки из CVAT (LaMa) и размытие фона по паку | `python -m ocr_utils.scan_cleanup` |
| `pdf_utils` | Промежуточные PDF под FineReader, сбор и сверка заострённых копий, JPEG-примитивы | `python -m ocr_utils.pdf_utils.intermediate_pdfs` |
| `final_pdfs` | Финальные PDF выпуска из двух прогонов FineReader: источник страницы (растр в базе, детектор геометрии), правка слоя, иллюстрации JPEG верхним слоем, снятие образов-фигур FineReader; стадии анализ → surya → сборка со сверкой | `python -m ocr_utils.final_pdfs run` |
| `geometry_regression` | Детектор «FineReader ухудшил геометрию». Боевой — v18: `v16/` (поле смещений, штрихи, line art, фото) + `quality/` (строки и края блоков по разбору `page_layout` обоих PDF, строки без пары в A — перенос оси полем, меры по плотному полю, вердикт по группам); корень — ядро v14 (кирпичи v16) | `python -m ocr_utils.geometry_regression.quality run`; библиотека (`quality.verdict.verdict_for_page`) |
| `text_layer_fix` | Правка текстового слоя FineReader (ядро + стенд исследования): разбор потока до слов и глифов, зоны повёрнутого и пропущенного прямого текста, tesseract/surya, удаление россыпи и вставка невидимого текста, сверка | библиотека (постранично из `final_pdfs`) |
| `defocus_detection` | Расфокус по папке RAF-превью: ранжирование, зональный | `python -m ocr_utils.defocus_detection` |
| `show_through_detection` | Просвечивающая бумага | `python -m ocr_utils.show_through_detection` |
| `line_art_detection` | Стенд: крупный штрих по бинаризованным PDF пака (CSV, экспорт по порогу покрытия); признаки — в `page_layout.line_art.features` | `python -m ocr_utils.line_art_detection` |
| `gutter_loss_detection` / `_restoration` | Текст, ушедший под корешок: детектор / восстановление (исследование) | `python -m ocr_utils.gutter_loss_*` |
| `dewarp` | Выпрямление кривых строк, несколько движков; годен только `textline` | `python -m ocr_utils.dewarp` |
| `rotated_text` | Таблицы с боковым текстом: прочитать, набрать прямо | `python -m ocr_utils.rotated_text.tables` |
| `background_smoothing` | Сглаживание фона под бинаризацию FineReader | `python -m ocr_utils.background_smoothing` |
| `zonal_deblur` | Зональный смаз: PSF по спектру, Винер | `python -m ocr_utils.zonal_deblur` |
| `inpainting` | Общие примитивы закраса (ROI, LaMa) для пальцев и разметки | библиотека |
| `external_ocr_services` | Боевой внешний OCR (DeepSeek V4.1 Flash): оглавление → список статей → остальные полосы, тайлы по сетке, теги повреждений; кэш запросов по всем проходам (`--cache-dir`), сборка выпуска в один md со склейкой переносов через границу полос | `python -m ocr_utils.external_ocr_services run` / `assemble` |
| `docx_md` | DOCX → Markdown, нарезка под LLM | библиотека |
| `experimental` | Новый функционал внешнего OCR до переноса в основные пакеты: склейка переносов по словарю, полосы 1×N, подсказки из детектора корешка, короткий промпт; стенд `scripts/replay_page.py` | библиотека |
| `legacy` | Помойка: не поддерживается, без тестов | — |
| `research/external_ocr_models` | Полоса → размеченный markdown через VLM (OpenRouter), промпты v1–v13 | `python -m research.external_ocr_models` |
| `research/geometry_regression` | Стенд детектора порчи геометрии (ядро — в `ocr_utils/geometry_regression`): прогон по паку, сводки и пороги, картинки «было \| стало», регрессия на эталоне, пробник VLM | `python -m research.geometry_regression run\|report\|regress` |
| `research/geometry_quality` | Стенд детектора порчи геометрии v17–v18 (ядро — `ocr_utils/geometry_regression/quality`): мера пака, отчёт с эталоном и сменами против v14/v16, оверлеи «было \| стало», сравнение двух прогонов, выборки на разметку | `python -m research.geometry_quality measure\|report\|sheets\|diff\|review` |
| `research/table_traces` | Стенд трасс линеек (сплайны) и сетки ячеек по кривым (ядро — `page_layout/tables/traces.py`, `curved_grid.py`): самые перекошенные таблицы пака, сравнение с сеткой по осям, оверлеи | `python -m research.table_traces select\|run` |
| `research/surya_equations` | Стенд оценки формул surya `Equation`: эталон глазами по слоям «бокс surya × `$$` DeepSeek» (`labels/`), полнота, точность, качество рамки с пересчётом на пак, оверлеи по трём папкам | `python -m research.surya_equations collect\|evaluate\|overlays` |
| `research/gutter_crossing` | Стенд «строки через межколонник» (ядро — `page_layout/text_blocks/columns.py`, `GutterMode`): пересчёт текстовых блоков пака по ключам в режимах `legacy`/`segmented`/`short`, мера `metrics.gutter_crossings_of`, выбор проблемных и нормальных полос, склейки «было \| стало» | `python -m research.gutter_crossing run\|measure\|select\|compare` |
| `research/row_jumps` | Стенд «перескок оси на соседнюю строку» (ядро — `metrics.row_jumps_of`, `segment._split_two_rows`): мера по осям пака, P и N, пересчёт с резкой двухрядных сгустков и без, «было \| стало» | `python -m research.row_jumps measure\|select\|run\|compare` |
| `research/heading_merge` | Стенд «сращивание строк разного набора» (подпись + заголовок, шапка + «Год издания», дата + оглавление): мера стыков в ряду, трассировка шага-виновника, C/P/N и «было \| стало» двух прогонов (ядро — `page_layout/text_blocks/typeset.py`) | `python -m research.heading_merge run\|compare\|significant\|trace` |
| `research/loose_rules` | Стенд «ложные линейки-сироты детектора таблиц» (боевой код не тронут): признаки каждой сироты пака (доля трассы на буквах, пробег «не букв», краска с обеих сторон, ядро штриха), листы глазами, разметка (`sets/`), правила отбраковки, пересчёт текстовых блоков «было \| стало» | `python -m research.loose_rules features\|sheets\|evaluate\|compare` |
| `research/text_block_specks` | Стенд «соринки и пометки у края строки» (ядро — `page_layout/text_blocks`): скан выступов концов строк по паку (PDF nogeo), разметка глазами, шаблоны знаков в `x_h` строки с подменой `segments_of`/`text_ink`, разностный судья DeepSeek-OCR-2/tesseract, сверка чужих движков | `python -m research.text_block_specks scan\|candidates\|evaluate\|score` |
| `research/edge_marks` | Стенд «сор и пометки у края текстового блока»: выбросы на выровненной стороне, разметка (`sets/`), кандидат — крайняя компонента, судьи «символ или нет» в своих окружениях (DeepSeek-OCR-2, pero, CRAFT, docTR, PaddleOCR, surya, tesseract, правило) с замером времени, VRAM и ОЗУ, оверлеи и листы | `python -m research.edge_marks select\|candidates\|judge\|score\|overlays` |
| `research/article_problems` | Отбор статей пака-1 о проблемах предприятий (классы 1–10: запчасти, не тот ассортимент, непоставки, сроки, брак, Госснаб, хуже иностранного, отказ от своей продукции): статья целиком в DeepSeek V4.1 Flash через OpenRouter (sync для пилота, Batch API для пака), случаи с цитатами, проверка цитат, листы по классам | `python -m research.article_problems run\|batch-submit\|batch-collect\|report\|check\|compare\|export` |
| `research/legacy/table_processing` | Стенд исследования таблиц; живой код переехал в `scan_markup` | заморожено |
| `scripts/` | Разовые утилиты; `gen_module_map.py` — генератор карты, `search_sessions.py` — поиск по прошлым сессиям Claude | — |
| `run_scripts/<пакет>/` | Готовые прогоны с числами в шапке, `source common.sh` | — |
| `reports/` | Отчёты по прогонам; оверлеи в подпапках вне git | — |
| `ai_slop/` | Черновой код от ИИ, вне git | — |

Подпакеты без `__init__.py`-экспортов, все namespace; конфиг только через CLI-опции (click).

## Команды

```bash
uv sync                                        # окружение (Python 3.11)
uv run pytest tests/ocr_utils/<пакет> -q       # тесты зеркалят пакет; полный прогон долгий
uv run black -l 120 -C .                       # хук форматирует .py сам после правки
uv run python scripts/gen_module_map.py        # карта модулей (хук делает сам)
```

Железо: 16 физических ядер, RTX 5060 Ti 16 ГБ, 135 ГБ RAM. Видеопамять одна на всех.

## Куда что класть

Новый отчёт — `reports/` (навык `write-report`). Run-скрипт — `run_scripts/<пакет>/`. Черновик —
`ai_slop/`. Выход прогона — на SSD `/mnt/hotstore/scan_processing/...` (промежуточные файлы; базы и превью CVAT — в `~/Projects/mts_markup`),
не в корень репо и **никогда в `/mnt/dump3/yandex_disk_*`** (Я.Диск затирает исходники).
Хук блокирует `pgrep -f` без `[x]`-разрыва, запись в корень Я.Диска, старый путь `/mnt/SYSTEM` заглавными
и `rm` баз разметки.

## Ожидание фоновых процессов

Ждать завершения процесса **по сохранённому PID**, а не по шаблону командной строки.

```bash
setsid uv run python long_job.py > log 2>&1 < /dev/null & PID=$!
while kill -0 "$PID" 2>/dev/null; do sleep 10; done
```

**Почему.** `pgrep -f "шаблон"` матчит **сам себя**: строка поиска входит в командную
строку того самого шелла, который её выполняет, поэтому условие никогда не станет
ложным и цикл висит вечно. В этом проекте так уже терялся час машинного времени —
ожидатель `while pgrep -f "exiftool -q -fast2 -r -ext RAF"` крутился больше часа после
того, как exiftool давно завершился. Хуже того, проверка «а жив ли он?» тем же `pgrep -f`
попадает в ту же ловушку и отвечает «жив» про уже мёртвый процесс. А `pkill -f` тем же
шаблоном убивает собственный шелл (код 144).

Если PID недоступен и без поиска по имени никак — разрывать самосовпадение классом
символов: `pgrep -f '[e]xiftool'`. Проверять живость конкретного процесса — через
`ps -p "$PID"`, а не `pgrep`. Запускать в фон через `setsid`, иначе pgid ≠ pid и
`kill -- -$PID` промахивается; останавливать — всю группу, иначе forkserver и воркеры
остаются сиротами.

## Стиль

* Комментарии, докстринги, логи — на русском.
* Форматирование — `black -l 120 -C`.
* Каждый модуль и класс — с докстрингом в одну содержательную строку: из них собирается карта.
* В Python-коде пиши подробные докстринги (с обязательным объяснием, что делает каждый аргумент и что возвращается) и подробные комменты (чтобы было понятно, что делает каждый фрагмент кода).
* Имена функций, методов, классов и переменных — на английском в американском написании: `recognize`, `normalize`, `summarize`, `serialize`, `modernize`, `color` (не `recognise`, `colour`). То же — в английском тексте промптов (`modernize`, `neighboring`, `centered`); правка промпта поднимает `PROMPT_VERSION`.
* Вместо строковых ключей старайся применять enum.
* Объявления функций внутри функций и методов - избегай, функции внутри функций м методов объявляй только тогда, когда без этого код не работает.
* Если функция или метод отдают данные — желательно отдавать через возвращаемое значение, а не через изменение объектов-аргументов (аккумуляторы, «заполни мне этот объект»). Несколько результатов — кортеж или dataclass; счётчики складывать у вызывающего (`stats = stats + run_issue(...)`).



## Параллелизм

В машине **16 физических ядер**. Любую CPU-интенсивную пакетную задачу (обход тысяч
сканов, декодирование/сжатие картинок, инференс на CPU) распараллеливать по процессам,
а не гонять в один поток. Ориентир — пул на 16 воркеров (`multiprocessing.Pool`,
`concurrent.futures.ProcessPoolExecutor`), по элементу пакета на воркер.

**Коэффициент параллелизма всегда делать настраиваемым** (CLI-опция `--jobs` или
параметр функции) с разумным значением по умолчанию. Многие задачи здесь упираются не
в счёт, а в скорость дисков — на такой задаче лишние воркеры только мешают, и число
процессов надо иметь возможность сбавить, не правя код.

Оговорки:

* Данные на `/mnt/dump3` лежат на медленном NTFS-3G — типичный кандидат на то, чтобы
  упереться в диск.
* Задачи на GPU (surya, torch) в пул не заворачивать: видеопамять одна на всех.
* В инициализаторе воркера отключать hugepage и потоки BLAS (см. `.claude/rules/gpu_and_pools.md`).

## Пороги детекторов

Калибровать по распределению метрик всего пака и выборке глазами по поясам score, а не по
эталону из десятка полос: эталон из 14 полос дал 23% ложных флагов на паке. Размеры,
привязанные к бумаге, мерить заранее и класть числами в run-скрипт, не долей от кадра.
