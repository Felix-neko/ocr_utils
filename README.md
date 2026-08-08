# ocr_utils

Обработка фотоснимков раскрытых книг и журналов: найти в кадре разворот, убрать
придерживающий страницу палец, повернуть и вырезать crop-зону. На выходе — кадры,
готовые к разбивке на страницы во внешнем ScanTailor.

## Установка

```bash
uv sync
```

Веса нейромоделей качаются автоматически в `finger_models/` (кроме DocShadow —
его `.pth` кладутся туда руками, в подпапку `docshadow/`).

## Запуск

```bash
uv run python -m ocr_utils.scan_cropping \
    --input-dir  IN \
    --output-dir OUT \
    --debug-dir  DBG \
    --recursive \
    --left-margin -150 --top-margin -150 --right-margin -150 --bottom-margin -150
```

Полный список опций — `uv run python -m ocr_utils.scan_cropping --help`.
Готовые прогоны под конкретные партии сканов лежат в `run_detect_and_crop*.sh`.

Ключевые опции:

| Опция | Зачем |
|---|---|
| `--crop-mode` | `rotate` — повернуть кадр и вырезать выпрямленный прямоугольник; `pixel-exact` — скопировать зону пиксель-в-пиксель, не трогая исходные пиксели (выпрямление снаружи) |
| `--*-margin` | припуски crop-зоны по каждой стороне отдельно, пикс. (>0 шире, <0 уже) |
| `--extra-erosion-px` | доп. обрезка краёв силуэта книги — срезает тёмные фрагменты обложки в углах |
| `--remove-fingers` | детекция и закраска придерживающего страницу пальца (включено) |
| `--protect-text-layout` | прогон через Surya layout и защита найденных блоков от закраски |
| `--bg-fill-method` | чем заливать всё за краем страницы: `average` или `nearest` |
| `--compensate-levels` | контраст-стретч по перцентилям внутри маски страницы |
| `--debug-dir` | оверлей: что нашёл детектор и что будет вырезано |

## Структура

```
ocr_utils/
├── scan_cropping/          # рабочий пайплайн
│   ├── gpu_models.py       #   все нейросети в одном объекте GpuModels
│   ├── page_detection.py   #   YOLO-World + SAM → силуэт разворота (E1)
│   ├── finger_removal/     #   детекция и закраска пальца (маска, LaMa, Surya layout)
│   ├── geometry.py         #   правильный поворот, crop-зона, область копирования (E2)
│   ├── levels.py           #   компенсация уровней
│   ├── background_fill.py  #   заливка всего, где нет содержимого книги
│   ├── cropping.py         #   вырезка (rotate / pixel-exact)
│   ├── layout_filtering.py #   отсев паразитных блоков layout
│   ├── overlay.py          #   debug-оверлей
│   ├── image_io.py         #   чтение/запись файлов
│   ├── pipeline.py         #   оркестрация одного кадра + обход пачки
│   └── cli.py              #   только click
├── defocus_detection/      # поиск расфокусов в папке: см. ocr_utils/defocus_detection/README.md
├── legacy/                 # помойка: см. ocr_utils/legacy/README.md
├── docx_md/                # конвертация docx ↔ md
├── pdf_utils/              # извлечение картинок из PDF
└── timing.py               # логирование таймингов операций
```

Все нейромодели (YOLO-World для страниц и для рук, SAM, LaMa, Surya layout,
DocShadow) живут в одном объекте `scan_cropping.gpu_models.GpuModels`. Он
создаётся один раз на прогон в `cli.py` и передаётся по пайплайну вместо строки
`device`; Surya и DocShadow грузятся только если их просят соответствующие опции.

Заброшенные эксперименты (dewarp, поиск печатей, ранние детекторы расфокуса) и
старый пайплайн «PDF → OCR-слой» лежат в `ocr_utils/legacy/` — они импортируются,
но не поддерживаются и не покрыты тестами.

## Поиск расфокусов

`ocr_utils/defocus_detection/` ранжирует сканы папки по качеству фокуса и показывает
самые подозрительные — те, что стоит переснять:

```bash
uv run python -m ocr_utils.defocus_detection "/путь/к/выпуску" --worst-percent 5 --zonal-percent 5
```

Подробности, список алгоритмов и результаты валидации — в
`ocr_utils/defocus_detection/README.md` и `defocus_detection_validation_report.md`.

## Тесты и форматирование

```bash
uv run pytest
uv run black -l 120 -C .
```
