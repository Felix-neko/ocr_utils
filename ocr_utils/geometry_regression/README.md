# geometry_regression — ядро детектора «FineReader ухудшил геометрию страницы»

Зачем и что меряется — в докстринге `__init__.py`. Как устроены метрики, пороги и что
отвергнуто — `research/geometry_regression/README.md` (стенд с прогоном по паку, сводками,
картинками и пробником VLM) и `docs/status.md`.

## Боевой детектор — v18 (с 2026-09-30)

Сборщик финальных PDF (`final_pdfs`) берёт вердикт у `quality.verdict.verdict_for_page`. Детектор v18 — три слоя:

| Подпакет | Что в нём | Кэш прогона |
|---|---|---|
| `v16/` | движок v16: поле смещений, перекос блоков, штрихи по следу краски, дробные черты, line art по LSD, фото по кромкам; `v16/page.py` — мера страницы и её JSON | `$GEOMETRY_V16_DIR` (`run_scripts/geometry_regression/run_pack1_v15.sh`) |
| `quality/` | меры v17–v18 по разбору `page_layout` обоих PDF: строки (`lines.py`, в том числе строки без пары в A — перенос оси полем), края блоков (`edges.py`), меры по плотному полю (`lineart_flow.py`), мера страницы (`measure.py`), вердикт по группам (`scoring.py`), вердикт для сборки (`verdict.py`) | `$GEOMETRY_V18_DIR` (`run_scripts/geometry_quality/run_pack1_v18.sh`) |
| корень | ядро v14: поле смещений, штрихи, строки, кромки, рамки — кирпичи, на которых стоит v16 | `$GEOMETRY_RUN_DIR` (v14, для сравнения) |

Разбор `page_layout` обоих вариантов (`$GEOMETRY_LAYOUT_ROOT`, `run_scripts/page_layout/run_pack1_analysis_v6_fr.sh`)
строится заранее на GPU. Прогон по паку — `python -m ocr_utils.geometry_regression.quality run`
(`run_scripts/geometry_regression/run_pack1_v18.sh`): кэш v16 (промахи меряет v16), меры v18, `verdicts.csv`. При промахе кэша v18 `verdict_for_page` меряет страницу (v16 — на месте, около 6 с), но
разбор не строит. Прогон по паку, отчёты и оверлеи — стенд `research/geometry_quality`; отчёт — `reports/geometry_regression_v18.md`.

```python
import fitz
from ocr_utils.geometry_regression.quality.verdict import verdict_for_page

with fitz.open(geo_pdf) as geo, fitz.open(nogeo_pdf) as nogeo:
    result = verdict_for_page(v18_dir, v16_dir, layout_root, "full_1972_10", 79, geo_pdf, nogeo_pdf, geo, nogeo)
result.assessment.verdict  # Verdict.BAD | MIXED | OK; result.reason — правило и виновник; result.cached
```

## Ядро v14: использование из кода

```python
import fitz
from ocr_utils.geometry_regression.cache import verdict_for_page
from ocr_utils.geometry_regression.scoring import Thresholds

with fitz.open(geo_pdf) as geo, fitz.open(nogeo_pdf) as nogeo:
    result = verdict_for_page(run_dir, "full_1966_01", page=38, geo_doc=geo, nogeo_doc=nogeo, thresholds=Thresholds())
result.verdict.verdict   # "bad" | "mixed" | "ok"; result.cached — взято из кэша или измерено (~3 с)
```

`run_dir` — каталог прогона стенда (`cache/<pdf>/pNNN.json`, номер страницы с единицы);
JSON другой `VERSION` игнорируется и переписывается. `bad` — брать страницу без коррекции,
`mixed` и `ok` — с коррекцией.

| Модуль | Что в нём |
|---|---|
| `metrics.py` | `measure_pair(gray300_nogeo, gray300_geo, Params) → PageMeasure` — все метрики страницы |
| `scoring.py` | пороги порчи и выигрыша, `Thresholds.apply(metrics) → Verdict` с гистерезисом |
| `cache.py` | `cache_path`, `load_page_cache`, `save_page_cache`, `measure_page`, `verdict_for_page` |
| `render.py` | рендер страницы в серый 300 dpi, рабочая копия 150 dpi, пары PDF по именам |
| `field.py`, `strokes.py`, `bend.py`, `lines.py`, `stretch.py`, `edges.py`, `raster.py`, `regions.py` | поле смещений, штрихи, изгиб линий, строки, растяжение, кромки колонок, кромки фотографий, области |

Тесты — `tests/ocr_utils/geometry_regression` (синтетические страницы).

С v13 (2026-09-21) рамки таблиц и line art, внутри которых тайлы поля ищутся 1:1 и штрихам разрешена толщина рисунка, даёт `ocr_utils.page_layout` (`regions.layout_boxes`: детектор таблиц + surya по варианту `fr_nogeo` из кэша + связные пятна; формулы surya `Equation` исключены). Кэш surya набивается перед пулом (`research.geometry_regression run --layout-cache`, `final_pdfs` стадия 0). Таблицы и рисунки — порознь (штрихи и изгиб в обоих, поле смещений и строки только в рисунках). Регрессия на эталоне v12→v13: 64/70, регрессий 0.

С v14 разбор ищет и растр: фотография на бинарном рендере — россыпь точек, и без растра она шла в line art (её тайлы «без пары» давали ложную порчу на целых снимках: 1970/12 с.76, 1975/04 с.2, а LSD находил в ней «изгиб» — 1973/03 с.58). Штрихи и строки внутри растра не меряются; порча самого снимка ловится по его кромкам (`raster.py`: сагитта и наклон кромки A − B, `photo_bend` 0.8 мм / `photo_tilt` 1.5 мм, непрощаемые — 1970/10 с.71, 1971/04 с.44, 1971/07 с.43). Линейки таблиц — обычные штрихи без правила «росчерк без перпендикуляра» (у открытой таблицы вертикали граф ни во что не упираются и в v13 выпадали из наклона: 1970/11 с.65, 1971/05 с.47, 1973/10 с.71); строки внутри таблиц в попарные метрики не идут (1967/01 с.47: «Январь . . . 57,0» по глифам волниста сама по себе); бланки (surya `Form`) — к таблицам (1974/06 с.96). Эталон v13→v14: 64/70, регрессий 0.
