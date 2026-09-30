# research/geometry_quality — стенд v17–v18 детектора порчи геометрии FineReader

Зачем и что меряется — в докстринге `__init__.py`. Коротко: страница «без коррекции (B) | с коррекцией (A)»
меряется по разбору `page_layout` **обоих** вариантов PDF; строки и края блоков — новыми мерами по
`text_blocks`, остальное (штрихи, дробные черты, line art, фото, поле смещений B → A) — из кэша стенда v16
(`ocr_utils/geometry_regression/v16`). Ядро `ocr_utils/geometry_regression` (v14) не тронуто.

## Данные

| Что | Где | Кто пишет |
|---|---|---|
| Разбор обоих вариантов (JSON + сайдкар `.npz`: оси строк, стороны блоков, трассы линеек) | `$QUALITY_LAYOUT_ROOT/{geo,nogeo}/pages/<pdf>_pNNNN.{json,npz}` (номер с нуля) | `run_scripts/page_layout/run_pack1_analysis_v6_fr.sh` |
| Кэш v16 (метрики, поле) | `$QUALITY_V16_DIR/cache/<pdf>/pNNN.json` (номер с единицы) | `run_scripts/geometry_regression/run_pack1_v15.sh` |
| Кэш стенда, `report.md`, `verdicts.csv`, оверлеи | `$QUALITY_RUN_DIR` | этот стенд |

Пути — `run_scripts/geometry_quality/common.sh`.

## Команды

```bash
python -m research.geometry_quality measure --layout-root … --v16-dir … --run-dir … [--jobs 16]
python -m research.geometry_quality report  --run-dir … [--v14-csv … --v16-csv … --labels …] [пороги]
python -m research.geometry_quality sheets  --run-dir … --geo-dir … --nogeo-dir … --out-dir … \
    --select labels|changed|belts|pages [--metric line_quality_mm --per-belt 8] [пороги]
```

Сравнение двух прогонов при одних порогах (смены bad ↔ не bad, оверлеи по папкам перехода, `diff.csv`, `diff.md`):

```bash
python -m research.geometry_quality diff --old-run … --new-run … --geo-dir … --nogeo-dir … --out-dir … --layout-root … [--labels …]
```

Выгрузки по вердиктам с описью `index.csv` (`--mode lineart_formula` — все страницы с line art или формулами по
папкам good/mixed/bad и годам; `--mode damage_belts` — страницы без line art с вердиктом bad и mixed по худшей
метрике и поясам её score):

```bash
python -m research.geometry_quality export --run-dir … --geo-dir … --nogeo-dir … --out-dir … --layout-root … --mode …
```

Боевой прогон целиком (мера, отчёт, сравнение с прошлым прогоном, обе выгрузки) —
`run_scripts/geometry_quality/run_pack1_v18_full.sh`. Прогон v18 и сравнение с v17 — `run_scripts/geometry_quality/run_pack1_v18.sh`, `run_diff_v17_v18.sh`; отчёт —
`reports/geometry_regression_v18.md`.

Пороги: `--thr имя=число` (порог метрики порчи или выигрыша), `--hard имя=число` (жёсткий порог метрики
порчи в её единицах), `--total S` (порог суммы групп), `--min-gain`, `--ratio`. Всё прогоняется
`run_scripts/geometry_quality/run_pack1.sh`.

## Меры

* **`line_quality_mm`** (`lines.py`) — размах оси строки по вертикали (мм) на чистых участках (без перескоков
  `jump_spans` и без точек и запятых `mark_spans`), на общем для B и A отрезке; порча — `w·(Q_A − Q_B)`,
  `w = √(h/h_корпуса)` в [1, 2]. Наклон, дуга и волна — одной мерой. Строки в таблицах, рисунках и формулах —
  только выигрыш. Выигрыши: `line_quality_gain_mm` (одна строка), `lines_quality_gain_mm` (медиана блока).
* **`edge_quality_mm`** (`edges.py`) — размах выровненной стороны блока по горизонтали (мм) на линии с
  заплатками PCHIP поверх невыровненных и недостоверных участков (`sides.filled_side` с `unreliable_*`);
  порча — `w·(E_A − E_B)`, `w = min(1, ряды/8)`. Для сравнения — `edge_quality_raw_mm` по сырой стороне.

* **Строки без пары в A** (v18, `lines.transfer_lines`, `lineart_flow.transfer_axis`) — ось A строится переносом
  оси B плотным полем (DIS по полосе строки, вес — кромки поперёк сдвига, точки без опоры выброшены), мера та же
  `line_quality_mm`. Только строки, похожие на текст: размах в B ≤ 1.2 мм, белый просвет над или под строкой,
  корреляция полосы ≥ 0.6, вне растра v6. Пунктир на оверлее; `line_transferred`, `line_quality_transferred_mm`.
* **Форма line art** (v18, `lineart_flow.frame_shape`) — сопоставление участков рамки, опора на тайлы текста вокруг,
  проверка направления по прямоте черт: `lineart_shape_mm`, `_nonsim_mm`, `_curve_mm`, `_part_turn_deg`,
  `_straight_delta_mm`. **Информационная**, в вердикт не входит: на паке 12 ложных из 13 (полутон фото, штриховка).
* **Абсолютный наклон рамки** (v18, `lineart_flow.lsd_angle`) — по прямым рисунка (LSD), профиль краски — запасной
  ход; `lineart_skew_*` информационные.

## Вердикт (`scoring.py`)

Группы: строки, края блоков, линейки, дробные черты, line art, фото. По порядку: ok (всё ниже порогов и
Σ < S) → bad непрощаемая → bad жёсткий порог метрики → bad совокупность (Σ score групп ≥ S, без пола) →
bad без выигрыша → bad гистерезис (жёсткая порча ≥ ratio × выигрыш) → mixed.

## Оверлеи

Склейка «без коррекции | с коррекцией»: в шапке — вердикт, правило и виновник, score групп и Σ против S,
выигрыш против `min_gain` и `ratio`, таблица метрик порчи (значение | порог | жёсткий | score | группа,
непрощаемая, мягкая); на странице — строки с заметной порчей (красные) и выигрышем (зелёные) с размахом Q,
выровненные стороны с размахом E и весом, виновники метрик толстой рамкой.
