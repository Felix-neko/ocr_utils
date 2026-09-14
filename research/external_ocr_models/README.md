# external_ocr_models — полоса журнала → размеченный markdown через VLM

Зачем и где граница задачи — в докстринге `__init__.py`. Здесь — как пользоваться.

## Команды

```bash
# распознать папку картинок одной моделью; структура папок выхода повторяет входную
uv run python -m research.external_ocr_models run \
    --in-dir /mnt/SYSTEM/raw/mts/pack1_background_blurred_v2/sharpened \
    --out-dir /mnt/SYSTEM/raw/mts/pack1_external_ocr/gemini-31-flash-lite \
    --model gemini-31-flash-lite \
    --pages run_scripts/external_ocr_models/probe_pages_1966_03.txt   # или без --pages: всё под --in-dir
    [--strips 2] [--max-side 2200] [--output-mode json|markdown] [--reasoning off|low|medium]
    [--jobs 4] [--skip-done] [--limit N] [--format json|md|both]

# сводка по готовым выходам (ничего не запрашивает, можно гонять сколько угодно)
uv run python -m research.external_ocr_models report \
    --out-root /mnt/SYSTEM/raw/mts/pack1_external_ocr \
    --in-dir .../sharpened --issue 1966/03 \
    --reference-pdf /mnt/SYSTEM/raw/mts/pack1_pdf/full_pdfs_binary_brightened_bg/full_1966_03.pdf \
    --rotated-info-dir /mnt/SYSTEM/raw/mts/pack1_rotated_tables/info \
    --report reports/external_ocr_models_probe.md

uv run python -m research.external_ocr_models models    # реестр с ценами
uv run python -m research.external_ocr_models balance   # баланс ключа OpenRouter
```

Ключ OpenRouter — `$OPENROUTER_API_KEY` (или `--api-key`); в логи и метаданные не попадает.
Готовые прогоны — `run_scripts/external_ocr_models/`.

## Что лежит в выходе

На каждую полосу `имя.json` (поля ниже), `имя.md` (то же с YAML-шапкой) и `имя.meta.json`
(провайдер, токены, `cost_usd` из `usage.cost` ответа, латентность, режим JSON, ошибки).
При сбое разбора рядом остаётся `имя.raw.txt` с сырым ответом. В корне папки модели —
`summary.csv` и `run.log`. `--skip-done` пропускает полосы с `.meta.json` без ошибки, так
что прерванный прогон продолжается тем же вызовом.

Поля ответа (`schema.py`): `page_number`, `running_header`, `running_footer`, `is_toc`,
`content_markdown`, `notes`. Разметка тела: `#` заголовок статьи, `##` подзаголовок,
`###` рубрика, `**автор**`, `*должность*`, таблицы GFM или `<table>`, `> [блок-схема]`,
`> [картинка: …]`, сноски `[^1]`.

## Как устроен запрос

* Картинка: серая, длинная сторона `--max-side` (2200 px ≈ 216 dpi), JPEG q85, base64 в
  `image_url`. `--strips N` режет полосу на N перекрывающихся горизонтальных кусков и
  шлёт их **одним** запросом — для моделей с потолком токенов на картинку (DeepSeek).
* Промпты — Jinja-шаблоны в `prompts/`; при правке поднимать `PROMPT_VERSION`.
* Ответ — `response_format: json_schema` (strict) с `provider.require_parameters`; если
  провайдер отверг — `json_object`, потом без `response_format`; параметр `reasoning`,
  если отвергнут, убирается и запрос повторяется. Что было использовано — в meta.
* `--damage` — режим повреждённых сканов: достроенные буквы `<restored>…</restored>`,
  сомнительные `<fuzzy>…</fuzzy>`, невосстановимые `<unknown/>`; в `.json` — поля `damage`
  (что модель видит), `restored`, `fuzzy`, `unknown`, `edge_words` (по повреждённым
  строкам; из него теги доставляются в текст, если модель их не поставила). Работает
  только с подсказкой, что и где повреждено: `--hint "…"` на прогон, `--hints файл`
  («путь<TAB>текст») по страницам, `--damage-side auto` по суффиксу `_L`/`_R`. Без
  подсказки модели достраивают молча, с завышенной — выдумывают повреждения (отчёт,
  разделы 8 и 10).
* `split-spreads --in-dir … --out-dir … [--fold ИМЯ=X_СВЕРХУ,X_СНИЗУ]` — разрезать
  фото-развороты на страницы по прямой сгиба (контрольные картинки в `_линии`).
* Рассуждения по умолчанию выключены (`reasoning: {enabled: false}`): на OCR они не
  помогают, а стоят втрое (замер на DeepSeek V4.1 Flash).

## Локальные движки

`local/<движок>/` — своё окружение (`uv sync --project research/external_ocr_models/local/<движок>`)
и `worker.py --in-dir --out-dir`; `run --model local-<движок>` подготавливает картинки и
зовёт воркер подпроцессом. Подробности и что завелось — в `reports/external_ocr_models.md`.
