# Выпрямление страниц (dewarp)

Несколько движков под одним CLI, папка на движок, пары «было | стало» и безэталонная оценка
по метрикам кривизны строк. Задача — понять, какой движок годится для флэтбед-сканов
журнала с прогибом у корешка; см. `reports/dewarp_report.md`.

```bash
uv run python -m ocr_utils.dewarp run --root КОРЕНЬ --only 1966/06/IMG_0144_1L.jpg --engines all --out-dir ВЫХОД
uv run python -m ocr_utils.dewarp run --from-links curved_lines_pack1_links/combo --out-dpi 300 --out-dir ВЫХОД
```

| Движок | Что это | Откуда | Вход | Устройство |
|---|---|---|---|---|
| `textline` | свой: центр-линии строк → поле вертикальной диспаратности (thin-plate RBF) → ремап только по вертикали; размер и перекос не меняются | по образцу `dewarp` Leptonica; сегментация строк — общая с детектором `scan_markup.curved_lines` | любое разрешение, движку сообщается `dpi` | CPU, пул |
| `pagedewarp` | cubic sheet: spans строк → модель поверхности (α, β) → Powell → ремап; расширенный вариант — поля 0, ширина как у входа, цвет | mzucker / lmmx `page-dewarp` (pip) | любое | CPU, пул; 8-40 с на полосу |
| `docscanner` | U2NETP-маска + прогрессивная сеть, backward map | fh2019ustc/DocScanner (IJCV 2025), веса `dewarp_models/docscanner` | 288×288 внутри, ремап полного кадра | GPU |
| `uvdoc` | сетка точек UVDocnet, билинейное развёртывание | tanguymagne/UVDoc (SIGGRAPH Asia 2023), веса в клоне | 488×712 внутри | GPU |
| `doctr` | U2NETP-маска + трансформер GeoTr | fh2019ustc/DocTr (ACM MM 2021), веса `dewarp_models/doctr` | 288×288 | GPU |
| `doctr_plus` | GeoTr без маски, «unrestricted» | fh2019ustc/DocTr-Plus (TMM 2023); веса `DocTrP.pth` нужно положить в `dewarp_models/doctr_plus/` | 288×288 | GPU |
| `dewarpnet` | WC-Net + BM-Net | cvlab-stonybrook/DewarpNet (ICCV 2019), веса `dewarp_models/dewarpnet` | 256×256 | GPU |

Нейросетевые движки обучены на Doc3D — фотографиях мятых листов с камеры: они ждут
целый лист на тёмном фоне и предсказывают сетку целиком, включая поворот и перспективу.
Флэтбед-скан журнала для них — вырожденный случай, и вести себя они могут как угодно;
ради этого они здесь и стоят рядом с классикой.

Код апстрим-репозиториев клонируется в `third_party/` (`engines/download.py`), веса
лежат в `dewarp_models/`. Старый CLI `ocr_utils.legacy.dewarp` продолжает работать через
тонкие реэкспорты.

## Выход

```
<out-dir>/original/<имя>.jpg        вход в рабочем разрешении — «было»
<out-dir>/<движок>/<имя>.jpg        результат
<out-dir>/compare/<движок>/<имя>.jpg пара «было | стало» с реперными линиями
<out-dir>/quality.csv, quality.md    метрики кривизны до/после, время на полосу, сбои
```

Имя полосы — `год_выпуск_имя` от `--root`; при `--from-links` — имя симлинка (в нём уже
год, выпуск, номер страницы PDF и score детектора).

## Оценка

`quality.py` считает на входе и на выходе метрики детекторов кривизны
(`skew_map.max_dev_deg`, `skew_map.resid_deg`, `line_fit.sagitta_rel_p90`, `line_fit.sagitta_rel_max3`,
`line_fit.slope_spread_deg`, `line_fit.slope_resid_deg`). Это безэталонная оценка: у
выпрямленной полосы разброс углов и прогибы должны упасть. Сравнивать только «до/после»
одной полосы — абсолютные значения зависят от вёрстки.

## Пул процессов

Воркеры CPU-движков ограничивают OpenCV и BLAS одним потоком и отключают у numpy пометку
больших массивов `MADV_HUGEPAGE`: с ней четыре одновременных ремапа полного кадра ждали
уплотнения памяти по 8-10 с вместо 0.5 с (см. `cli._init_worker`). Замер: textline 1.4 с на
полосу 600 dpi при восьми воркерах.
