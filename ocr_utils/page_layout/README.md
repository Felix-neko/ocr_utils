# page_layout — разбор структуры страницы одним пакетом

Растр (фотографии, печати), таблицы, line art (схемы, чертежи, графики, формулы), повёрнутый
текст вне таблиц, ориентация полосы и блоки surya layout — одни детекторы, одни версии, один кэш
surya для всех потребителей: `scan_markup detect` (разметка в базу и CVAT), `geometry_regression`
(выбор страницы при сборке финальных PDF), `text_layer_fix` (правка текстового слоя),
`background_smoothing` и `scan_cropping` (защита контента).

Зачем — в докстринге `__init__.py`. История: до 2026-09-21 те же вопросы решались в пяти
пакетах пятью способами (пять конструкторов surya, два формата кэша, три детектора line art);
`detect` размечал полосу одним набором детекторов, а сборка PDF решала судьбу страницы другим.

## API

```python
from ocr_utils.page_layout.analysis import Find, LayoutOptions, PageLayout
from ocr_utils.page_layout.image import PageImage, Variant
from ocr_utils.page_layout.surya import SuryaCache, SuryaSource, SuryaSourceConfig
from ocr_utils.page_layout.surya.model import SuryaLayoutModel

image = PageImage.from_file(path, Variant.SCAN, cache_name="1966/01/IMG_0003_2R")      # тег dpi обязателен
image = PageImage.from_pdf_page(doc, index, Variant.FR_NOGEO)                          # <stem>/pNNNN
image = PageImage.from_array(gray_or_bgr, dpi=600, variant=Variant.FR_GEO, cache_name=..., source=...)

surya = SuryaSource(SuryaCache(root), SuryaLayoutModel())        # родитель: кэш + модель
surya = SuryaSourceConfig(root).open()                           # воркер: кэш только чтение, промах = SuryaMissing
layout = PageLayout(image, {Find.TABLES, Find.LINE_ART}, LayoutOptions()).process(surya)
layout.raster_pics, layout.stamp_suspects, layout.tables, layout.line_arts,
layout.rotated_text_not_in_tables_regions, layout.best_page_orientation, layout.raw_surya_content
```

Результат — `Region(box, kind: RegionKind, confidence, source, full_page, info)` в **родных пикселях**
картинки. Два этапа под пул: `prepare(cache)` в воркере (всё пиксельное; при попадании в кэш —
достраивает всё), `needs_surya` → родитель зовёт модель → `finish(blocks, gpu_orientation)`.
`known={Find.RASTER: [...], Find.TABLES: [...]}` — области семейств, которые сейчас не ищутся
(из базы): line art и повёрнутый текст ищутся вне них.

**Порядок внутри `process()`** — часть алгоритма: surya → ориентация → растр → таблицы → line art
только вне растра ∪ таблиц → повёрнутый текст вне таблиц ∪ растра.

## Модули

| Модуль | Что |
|---|---|
| `image.py` | `PageImage`: варианты (`Variant`), ленивые копии (`gray_at`, `bgr_at`, `bitonal_at` — Оцу), кадр surya (150 dpi, ≤ 2048 px), отпечаток источника, pickle |
| `surya/blocks.py` | `Block`, `LayoutBlocks` (метка, уверенность, рамка, полигон), классы меток |
| `surya/model.py` | `SuryaLayoutModel` — единственный `LayoutPredictor` в репозитории; ленивая загрузка, пачки по 8 |
| `surya/cache.py` | `SuryaCache`: `<root>/<variant>/<cache_name>.json`, попадание по отпечатку источника или дайджесту кадра, `legacy` из старых pickle (`import-legacy-cache`) |
| `surya/source.py` | `SuryaSource` / `SuryaSourceConfig`: кэш → модель → политика промаха (`FAIL` / `SKIP`) |
| `raster/` | растровый детектор (точки сетки, тон, цвет, обложка, сборка) — бывший `scan_markup.detection.*` |
| `tables/` | детектор таблиц v4.2 — бывший `scan_markup.table_detection` (README там же) |
| `line_art/features.py` | признаки пятен и скоплений линеек — бывший `line_art_detection.features` |
| `line_art/detector.py` | единый детектор: затравки (схемы детектора таблиц + surya Figure/Form/Equation/Picture + пятна и линейки) через одну пиксельную проверку, вне растра и таблиц; уверенность = доля семейств источников |
| `rotated_text/` | Docstrum (бывший `text_layer_fix.docstrum`) и зоны повёрнутого текста вне таблиц (без tesseract) |
| `orientation/` | ориентация полосы целиком — бывший `scan_markup.orientation` (`python -m ocr_utils.page_layout.orientation`) |
| `analysis.py` | фасад `PageLayout`, `LayoutOptions`, `Find` |
| `prefill.py` | набивка кэша surya пачкой: рендер в пуле, модель в родителе |
| `cli.py` | `analyze` (одна страница, оверлей, JSON), `prefill-surya`, `import-legacy-cache`, `verify-cache` |

## Версии

`RASTER_VERSION`, `TABLES_VERSION`, `LINE_ART_VERSION`, `ROTATED_TEXT_VERSION`, `ORIENTATION_VERSION`
в `__init__.py` — по семействам: правка порога у таблиц не заставляет перечитывать пак ради растра.
Потребители пишут их рядом с результатом (`Page.*_version` в базе, `page_layout` в JSON кэшей
geometry_regression и text_layer_fix) и по ним решают, пересчитывать ли страницу. Поднимать при любой
правке, меняющей результат семейства.

## Кэш surya

`LAYOUT_CACHE_DIR` в `common.sh` пака = корень; варианты `scan`, `sharpened` (перенесены из старых
pickle, `legacy`), `fr_geo`, `fr_nogeo` (страницы PDF FineReader; набивает `prefill-surya`, ~0.7 с GPU
на страницу), `blurred`, `camera`. Промах в воркере пула — ошибка (`SuryaMissing`), не тихий разбор без
surya: кэш набивается до пула (`final_pdfs` стадия 0, `research.geometry_regression run`,
`text_layer_fix run`/`eval-lineart` делают это сами).

## Что смотреть при правке детекторов

* line art: `text_layer_fix eval-lineart` (источники `page_layout`, `surya_render` против 221 ручной
  области), `reports/page_layout_migration.md`;
* геометрия: `research.geometry_regression regress` (эталон 70 страниц);
* таблицы: `run_scripts/table_processing/run_compare_detector.sh` (190 полос);
* растр: `scan_markup validate` по папкам-эталонам.
