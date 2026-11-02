# research/pseudo_glyphs — псевдоглифы в разборе текстовых блоков

Компоненты краски, которые `page_layout/text_blocks` принимает за буквы и затягивает в строки: тень сгиба и кромка
листа у корешка, полоса обложки или подложки у края, нитки прошивки (крючки у края), карандаш и чернила, сор.
От них ломаются стороны блоков и выключка, особенно во втором преобразовании `scan_cropping.v2`
(`rectify/text_field.analyze_page` → `analyse_gray`, без карт CRAFT). Боевой код не тронут; итоги и план правок —
`reports/pseudo_glyphs.md`. Прогон — `run_scripts/text_blocks/run_pseudo_glyphs.sh`.

## Модули

| Модуль | Что делает |
|---|---|
| `crops.py` | вырезки холста Кунса (PNG + JSON, копия `PageCrop`/`load_crop` из незакоммиченного `research/line_art_boxes`), подготовка серого как в `analyze_page`, разбор с правилом 0.3 мм и без (`edge_touch`) |
| `candidates.py` | кандидаты: крайняя компонента каждого ряда блока слева и справа (с краской до 3 мм за концом ряда) и все компоненты ближе 1 мм к краю вырезки; признаки в мм: размеры, заполнение, штрих (площадь / скелет), край картинки, тон (0 — краска страницы, 1 — бумага), высота в p95 букв блока, выход за полосу ряда, отрыв от соседней краски ряда в робастных максимумах межсловного пробела, сколько рядов задевает |
| `features.py` | проход по вырезкам (пул): кандидаты (CSV на страницу), меры сторон, серый 300 dpi для карт |
| `scores.py` | оценки CRAFT (максимум карты по пикселям компоненты) и pero (окно базовой линии ряда) — как `edge_guard.component_scores` |
| `select.py` | классы разметки (`Label`), пояса выборки (`Band`) |
| `sheets.py` | листы плиток на разметку (подпись — в поле под плиткой, рамка вокруг компоненты, серое поле за краем вырезки) |
| `rules.py` | правила (`PseudoReason`, `RuleParams`) и голос CRAFT для спорных серых кандидатов |
| `evaluate.py` | полнота по классам, вред по классам знаков (вся разметка, зона риска, случайный пояс), сетка порога CRAFT |
| `prototype.py` | прототип встраивания: первый разбор → кандидаты → решения → закрас с ореолом 2 px → повторный разбор (до двух проходов), `FilterMode` |
| `measures.py` | меры страницы: выключка, выровненные стороны разбора, стороны-опоры v2 (`aligned_sides`), p95 отклонения опор от вертикали колонки (`side_samples`), p95 остатков краёв рядов |
| `compare.py`, `compare_report.py` | «было | стало» по всем вырезкам и сводка, оверлеи страниц с закрасом |
| `packcheck.py` | регрессия на размеченных наборах пака-1 (`research/edge_marks`, `research/text_block_specks`) |

## Команды

```bash
PYTHONPATH=. uv run python -m research.pseudo_glyphs features --jobs 2       # кандидаты и меры (1 096 страниц ХиП 1991)
PYTHONPATH=. uv run python -m research.pseudo_glyphs maps                    # карты CRAFT и pero (GPU)
PYTHONPATH=. uv run python -m research.pseudo_glyphs scores --jobs 2         # оценки детекторов у кандидатов
PYTHONPATH=. uv run python -m research.pseudo_glyphs sheets --round-name r4 --bands doubt=100,fired=100
PYTHONPATH=. uv run python -m research.pseudo_glyphs labels --round-name r4  # sets/raw/r4.txt → sets/labels.csv
PYTHONPATH=. uv run python -m research.pseudo_glyphs evaluate                # полнота, вред, CRAFT
PYTHONPATH=. uv run python -m research.pseudo_glyphs compare --craft-doubt 0.4 [--keys 07/IMG_0099_R]
PYTHONPATH=. uv run python -m research.pseudo_glyphs pack1-features && ... pack1-check
```

## Разметка

`sets/labels.csv` — `key, comp, label, band, note, round, cid` (ключ — страница и номер компоненты в разметке серого
300 dpi; `cid` меняется при пересборке `candidates.csv`, ключ — нет). Исходники глазами — `sets/raw/<раунд>.txt`:
`<лист> default <класс>` и `<номер> <класс> [заметка]`. Классы: псевдоглифы `SH` (кромка листа, тень сгиба), `CV`
(полоса обложки/подложки), `TH` (нитки, крючки), `PN` (карандаш, чернила, номера от руки, штамп), `SP` (сор); знаки
`G`, `HY`, `BR`, `NU`, `PU`, `IN`; прочее `NB` (соседняя страница), `FG` (рисунок, линейка), `U`.
Листы раундов — `/mnt/hotstore/scan_processing/pseudo_glyphs/hp1991/sheets/r{1,2,3}/`.
