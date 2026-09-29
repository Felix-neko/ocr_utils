# research/loose_rules — ложные линейки-сироты детектора таблиц

Сирота — линейка, которую `page_layout.tables.detector.detect_all` не отнёс ни к таблице, ни к схеме
(`PageLayout.loose_rules`). Сироты становятся барьерами текстовых блоков, и ложная сирота режет строки.
Стенд берёт сироты из итоговых JSON разбора пака (боевой код не трогает), мерит признаки, строит листы
глазами, проверяет правила отбраковки на разметке и пересчитывает текстовые блоки без отброшенных.
Отчёт: `reports/loose_rules_false.md`; прогон: `run_scripts/text_blocks/run_loose_rules.sh`.

## Команды

```bash
uv run python -m research.loose_rules features --pack-dir <разбор> --sharpened-dir <сканы> --out-dir <выход>
uv run python -m research.loose_rules sheets   --pack-dir ... --sharpened-dir ... --features-csv <выход>/features.csv \
    --out-dir <выход>/sheets --name v_hi --query "horizontal==0 and glyph_share_40>=0.5" --count 30
uv run python -m research.loose_rules evaluate --features-csv <выход>/features.csv --out-dir <выход>/eval
uv run python -m research.loose_rules compare  --pack-dir ... --sharpened-dir ... --verdicts <выход>/eval/verdicts.csv \
    --out-dir <выход>/compare
```

## Модули

* `features.py` — признаки сироты на рабочей копии 150 dpi: `glyph_share_*` (доля трассы на компактных,
  «буквенных» компонентах), `rule_run_mm`, `coverage`, разрывы и `interline_gaps`, `side_ink_*`,
  `core_gray` / `core_width_mm` (профиль ядра штриха), `in_formula`, прочие (`shadow`, `band_*`) — для справки.
* `evaluate.py` — `DropReason` (`glyphs`, `short_run`, `formula`, `edge`), `Thresholds`, сводка на разметке.
* `labels.py` — `RuleLabel` и чтение `sets/labels.csv` (`rule_id` = `год/выпуск/полоса#номер в loose_rules`).
* `compare.py` — два разбора текстовых блоков (все сироты / без отброшенных) и склейка «было | стало».
* `cli.py` — команды и листы.

## Пример

`1966/02/IMG_0061_1L`: пять сирот; две горизонтали по бокам заголовка — отбивки (`glyph_share_40 = 0`,
пробег 48 и 32 мм); три вертикали по стволам «Н/И» двух строк — `glyph_share_40 = 1.0`, разрыв на межстрочье,
причина `glyphs`. Без них оси заголовка из пяти обрывков собираются в две строки.
