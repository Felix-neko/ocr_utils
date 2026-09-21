# geometry_regression — ядро детектора «FineReader ухудшил геометрию страницы»

Зачем и что меряется — в докстринге `__init__.py`. Как устроены метрики, пороги и что
отвергнуто — `research/geometry_regression/README.md` (стенд с прогоном по паку, сводками,
картинками и пробником VLM) и `docs/status.md`.

## Использование из кода

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
| `field.py`, `strokes.py`, `bend.py`, `lines.py`, `stretch.py`, `edges.py`, `regions.py` | поле смещений, штрихи, изгиб линий, строки, растяжение, кромки, области |

Тесты — `tests/ocr_utils/geometry_regression` (синтетические страницы).

С v13 (2026-09-21) рамки таблиц и line art, внутри которых тайлы поля ищутся 1:1 и штрихам разрешена толщина рисунка, даёт `ocr_utils.page_layout` (`regions.layout_boxes`: детектор таблиц + surya по варианту `fr_nogeo` из кэша + связные пятна; формулы surya `Equation` исключены). Кэш surya набивается перед пулом (`research.geometry_regression run --layout-cache`, `final_pdfs` стадия 0). Таблицы и рисунки — порознь (штрихи и изгиб в обоих, поле смещений и строки только в рисунках). Регрессия на эталоне v12→v13: 64/70, регрессий 0.
