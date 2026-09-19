# text_layer_fix — стенд исследования текстового слоя FineReader

Ядро (разбор слоя, зоны, чтение, вердикты, правка, кэш) — `ocr_utils/text_layer_fix`, здесь стенд.
Зачем и что делаем — в докстринге `ocr_utils/text_layer_fix/__init__.py`. Отчёт по прогону — `reports/text_layer_fix.md`,
оверлеи к нему — `reports/text_layer_fix/` (вне git).

## Команды

```bash
# Обзор слоя всех PDF пака (только чтение): формы спанов, повёрнутые матрицы, привязка → survey.csv
./run_scripts/text_layer_fix/run_survey.sh

# Выборка страниц: зоны, вердикты по словам, чтение зон → cache/, pages.csv, words.csv, zones.csv
./run_scripts/text_layer_fix/run_sample.sh            # ~1050 страниц, 12 воркеров, ~5 мин

# Второе мнение surya (GPU, один процесс) по зонам с ненадёжным чтением; JSON кэша обновляется
./run_scripts/text_layer_fix/run_second_opinion.sh

# Пересчитать вердикты слов по кэшу (после surya или смены порогов)
uv run python -m research.text_layer_fix reclassify --pdf-dir ... --out-dir ...

# Исправленные копии PDF (исходники не трогаются) → pdf/, fix.csv со сверкой каждой страницы
./run_scripts/text_layer_fix/run_fix.sh

# Оверлеи → overlays/<категория>/
./run_scripts/text_layer_fix/run_overlay.sh

# Точность и полнота источников line art против ручной разметки → lineart_eval.{csv,json}
./run_scripts/text_layer_fix/run_eval_lineart.sh

# Сравнение с языковой моделью (нужен OPENROUTER_API_KEY) → llm_compare.{csv,json}
./run_scripts/text_layer_fix/run_llm_compare.sh

# Сводка для отчёта
uv run python -m research.text_layer_fix report --out-dir /mnt/SYSTEM/raw/mts/pack1_text_layer_fix/pack1_v1
```

Пути — в `run_scripts/text_layer_fix/common.sh`; выход прогона — `/mnt/SYSTEM/raw/mts/pack1_text_layer_fix/<версия>/`.

## Как устроено

Модули ядра описаны в `ocr_utils/text_layer_fix/README.md`. Здесь:

| Модуль | Что в нём |
|---|---|
| `cli.py` | Команды стенда: `survey`, `run` (выборка страниц + разбор в пуле), `fix`, `overlay`, `second-opinion`, `eval-lineart`, `llm-compare`, `reclassify`, `report` |
| `pages.py` | Выборка страниц и привязка «страница PDF ↔ полоса» по базе-зонду |
| `db_models.py` | Константы видов областей базы разметки для выборки |
| `lineart_eval.py` | Точность/полнота источников line art против ручной разметки на no-geo PDF |
| `llm_sanitize.py` | Сравнение геометрического вердикта с вычёркиванием мусора LLM |
| `overlay.py`, `report.py` | Оверлеи и markdown-сводка |
