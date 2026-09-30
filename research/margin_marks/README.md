# research/margin_marks — пометки на полях, ложно принятые за рисунок

Зачем и как — в докстринге `__init__.py` и в `reports/margin_marks.md`. Боевой код не тронут.

| Команда | Что делает |
|---|---|
| `python -m research.margin_marks features --layout-root … --out-dir …` | признаки остатка залитой вырезки у кандидатов-рисунков и «неясно» обоих вариантов → `features.csv` |
| `python -m research.margin_marks contact --layout-root … --features-csv … --out-dir … [--variant geo] [--max-thickness 0.55] [--ids файл]` | мозаики «вырезка первого прохода \| залитая» для разметки глазами |
| `python -m research.margin_marks replay --layout-root … --geo-dir … --nogeo-dir … --out-dir …` | пересчёт решения line art всех кандидатов без GPU: воспроизводимость и смены от правила → `replay.csv` |

Разметка — `sets/labels.csv` (`variant,id,label,note`; `label`: `mark` или `disputed`). Правило — `rule.py`
(`MarkRule`, пороги с обоснованием). Прогон — `run_scripts/page_layout/run_margin_marks.sh`.
