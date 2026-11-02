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
| `tables/` | детектор таблиц (`TABLES_VERSION`) — бывший `scan_markup.table_detection` (README там же) |
| `line_art/features.py` | признаки пятен и скоплений линеек — бывший `line_art_detection.features` |
| `line_art/detector.py` | единый детектор: затравки (схемы детектора таблиц + surya Figure/Form/Equation/Picture + блоки `image` DeepSeek-OCR-2 из его кэша + пятна и линейки) через одну пиксельную проверку, вне растра и таблиц; пересекающиеся затравки разных источников сливаются в одну (`merge_seeds`) — двух комплектов одной находки нет; уверенность = доля семейств источников (знаменатель — таблицы, surya, пиксели; DeepSeek сверх него) |
| `tables/detector.py`: полосы у края (TABLES v7) | `edge_strips` снимает из штриховой краски роста схемы (`line_art_mask`) компоненты, прижатые к краю картинки и не толще `EDGE_STRIP_MM` = 1 мм поперёк края: тень сгиба у корешка вырезки страницы холста v2 сшивалась склейкой 5 мм с рисунком и тянула рамку до заголовка и на 59 мм вниз (ХиП 1991/06 IMG_0419) |
| `tables/curve.py`: изогнутые линейки (TABLES v8) | признаки находки (`detector._find`, `signs_of` и перерост) — по вырезке, выпрямленной вдоль осей длинных горизонтальных линеек (ось — середина краски по столбцам, многочлен 3-й степени; между линейками — линейная интерполяция по y), если изгиб от 0,4 мм. Таблица на изогнутом низу холста Кунса ХиП 1991/02 IMG_1161 (прогиб ~2 мм, `edge_fill` 0,42 при пороге 0,45) теперь находится; ХиП 1991 — таблиц 8 → 11, пак-1 — без смен |
| `line_art/features.py`: правила v6 | против ложных находок пака-2 (CVAT, 2026-10-18): **A** — низкое (≤ 1,5 строки) пятно под текстовым блоком surya — слипшиеся буквы заголовка; **B** — открытая ломаная из тонких линеек (≥ 80 % краски в осевых прогонах от двух строк, толщина ≤ 0,5 мм) — обломок рамки; **C** — затравка без подтверждения пикселями или детектором таблиц обязана иметь заполнение от `MIN_FILL` и штрих (двумерное пятно или две замкнутые коробки): строка бланка и рамка «Примечание» от surya `Form` не проходят |
| `line_art/features.py`: правило A′ (v7) | только без текстовых блоков surya (`rectify.page_regions` у `scan_cropping.v2` зовёт разбор с `use_surya=False`): пятно-кандидат не выше 10 мм — слипшиеся буквы, если в его строке есть буквы-соседи (`letter_boxes`: компоненты 1–8 мм высотой, до 25 мм шириной) с обеих сторон или не меньше двух с одной; сосед — не ниже 0,35 высоты пятна, верх или низ на линии пятна (`letters_in_row`, константы `LETTER_*` в мм). На ХиП 1991 снимает 24 из 31 рамки на заголовках и сцепленных строках, рисунков не теряет (стенд `research/line_art_boxes`); на путях с surya результат прежний |
| `line_art/strokes.py`: добор штрихов (v8–v9) | к рамкам вида рисунка (обе стороны от 8 мм, вытянутость до 3) `detect_line_art` добирает отдельные компоненты краски в 4 мм от рамки (до 3 кругов, сторона — не дальше 15 мм): лучи, линии движения, дым, капли, точки знака вопроса, штрихи тени — мельче ворот пятна, раньше в рамку не попадали. Не добираются буквы строк (`is_text_component`; буквы глубже 2 мм внутри рамки — надписи рисунка — не в счёт), дефисы и запятые, ряды одинаковых знаков («* * *», `mark_rows`), волоски, полосы у края, компоненты в растре и таблицах; v9 — сторона не растёт на колонку текста (`text_rows`, `column_rows`: прирост, задевающий буквы трёх и больше строк, запрещён; подписи рисунка в одну-две строки берутся). ХиП 1991, 37 размеченных рамок: краски рисунка вне рамки 1.78 % → 0.57 % (рамок с потерей > 1 % — 18 → 3); пак-1 с surya — рубрики и рисунки шире на миллиметры |
| `line_art/deepseek/page_cache.py` | кэш разбора полосы целиком DeepSeek-OCR-2 (затравки line art): кадр surya → воркер vLLM (`runner.run_worker`, под сторожем памяти) → блоки в формате кэша surya под своим корнем; `page_layout prefill-deepseek`, `scan_markup detect --deepseek-cache` |
| `line_art/deepseek/` | решение по кандидату двумя проходами DeepSeek-OCR-2 (`decide.py`): объекты, надпись, неясно или **пометка** — карандашная пометка на полях (подчёркивание, скобка, дуга через колонку) по остатку вырезки с залитыми словами (`marks.py`, `reports/margin_marks.md`); в JSON полосы разбора пака — список `marks`, не объект и не запрет для текстовых блоков; **одиночная черта** — рамка рисунка или «неясно» тоньше 5 мм (`decide.is_thin_stroke`: концевые черты статей, линейки, края колонок, куски фото и заголовков — 57 рамок пака-1, ни одного рисунка) снимается с объектов в список `strokes` (`pack_analysis.final.split_strokes`, и при пересборке блоков из прошлого разбора); версия решения — `DECISION_VERSION` (в `versions.line_art_decision`) |
| `rotated_text/` | Docstrum (бывший `text_layer_fix.docstrum`) и зоны повёрнутого текста вне таблиц (без tesseract) |
| `orientation/` | ориентация полосы целиком — бывший `scan_markup.orientation` (`python -m ocr_utils.page_layout.orientation`) |
| `analysis.py` | фасад `PageLayout`, `LayoutOptions`, `Find` |
| `prefill.py` | набивка кэша surya пачкой: рендер в пуле, модель в родителе |
| `cli.py` | `analyze` (одна страница, оверлей, JSON), `prefill-surya`, `prefill-deepseek`, `import-legacy-cache`, `verify-cache` |

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
