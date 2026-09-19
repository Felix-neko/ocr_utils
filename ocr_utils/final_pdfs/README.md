# final_pdfs — сборка финальных PDF из двух прогонов FineReader

Зачем и что делаем — в докстринге `__init__.py`. Запуск по паку-1 — `run_scripts/final_pdfs/run_pack1.sh`
(проба: `--only-year 1966 --only-issue 03`).

```bash
uv run python -m ocr_utils.final_pdfs run \
    --geo-dir <PDF с коррекцией> --nogeo-dir <PDF без коррекции> --out-dir <финальные> \
    --pictures-dir <blurred/> --db pack1_reviewed.sqlite --pack-name пак-1 --work-dir <рабочий> \
    --geometry-run-dir <прогон детектора геометрии> [--only-year Г --only-issue НН] \
    [--picture-dpi 300 --jpeg-quality 75 --descreen-sigma-mm 0.05 --margin-x-mm 12.192 --margin-y-mm 6.096] \
    [--no-second-opinion] [--no-text-layer] [--redo] [--preview-pages N] [--no-assemble]
```

## Стадии

| Стадия | Где | Что | Выход |
|---|---|---|---|
| проверка пар | родитель | оба PDF выпуска на месте, число страниц одинаково и равно числу полос в базе | выпуск без пары — в `summary.csv` со статусом `error` |
| A — анализ | пул (`--jobs` − `--reserve-cpu-cores`) | по странице: источник (растр в базе → nogeo; вердикт детектора геометрии `bad` → nogeo; иначе geo), разбор слоя выбранной страницы (`text_layer_fix.pipeline.process_page`, tesseract) | `work/pages/<pdf>/pNNNN.json`, `work/analysis.csv` |
| B — surya | родитель (GPU) | второе мнение по зонам, где tesseract не принят или неуверен | те же JSON |
| C — сборка | пул (`--assemble-jobs`) | страница копируется из источника → правка слоя → снятие образов-фигур FineReader под иллюстрациями → JPEG иллюстраций → сохранение → сверка по файлу | `out/{год}_{выпуск}.pdf`, `work/pages.csv`, `work/summary.csv`, `work/preview/` |

Идемпотентность: JSON текущей версии (`analysis_version` + версия `text_layer_fix`) и готовый
PDF с верным числом страниц не пересчитываются (`--skip-done`); `--redo` — всё заново.
Вердикт геометрии берётся из `cache/<pdf>/pNNN.json` прогона `--geometry-run-dir` той же
версии детектора, при промахе страница меряется (~3 с) и дописывается туда же.

## Что сверяется на каждой собранной странице

* образ страницы без коррекции = повёрнутая полоса из базы + поля (страницы с иллюстрациями;
  расхождение — выпуск не собирается: сборка со сдвигом на страницу хуже её отсутствия);
* слой: оставленные глифы на прежних местах, удалённые не извлекаются, каждая вставка
  находится `search_for` в своей рамке, основной образ не пересжат (по сырым байтам, иначе по
  пикселям), кроме страниц, где он снят намеренно;
* иллюстрации: DCT-образ нужного размера, `DeviceGray` у серых / `DeviceRGB` у цветных, на своём
  месте с допуском 1 pt; при полностраничном растре бинарного образа на странице нет.

## Модули

| Модуль | Что в нём |
|---|---|
| `plan.py` | выпуски и полосы из базы явным `select` (только чтение), `PageSource`, `decide_source`, имя финального PDF |
| `sources.py` | пары PDF, поля в px, `verify_page_geometry` |
| `pictures.py` | вырезка из очищенной полосы, `resample` (Gaussian + INTER_AREA), JPEG, `placement_rect`, `figures_under`, `remove_images`, `insert_picture` |
| `analysis.py` | стадия A: `AnalysisParams`, `analyse_chunk`, `load_analysis`, запросы для surya |
| `assemble.py` | стадия C: `assemble_issue`, сверка, `pages.csv`/`summary.csv` |
| `cli.py` | команда `run`, пулы, сводка |

Умолчания для пака-1 (`pictures.py`): 300 dpi, JPEG 75, σ = 0.05 мм; поля 12.192 / 6.096 мм
(`cli.py`). Порог «образ FineReader под иллюстрацией» — 80 % площади образа внутри рамки.
