# research/surya_coons — surya layout по холсту Кунса против холста по трапециям

Стенд мер «было | стало» для шага surya в `scan_cropping.v2`. Боевой код — `ocr_utils/scan_cropping/v2/coons_layout.py`
(`--layout-warp coons`, по умолчанию с 2026-10-24); стенд ничего не меняет, только читает два кэша.

## Что сравнивается

- **было — `--layout-warp keystone`**: кадр выпрямляется гомографиями страниц по трапециям (`keystone.py`), блоки
  surya возвращаются на кадр наклонными четырёхугольниками.
- **стало — `--layout-warp coons`**: шаг страниц строит подгоночные области двух страниц **без текста**
  (`pipeline.surya_frames` → `layout_frames.npz`), по ним — поверхность Кунса (первое преобразование выпрямления
  выхода страниц, `rectify/coons.py`), холст с полем 25 мм на масштабе входа surya (длинная сторона ≤ 2048). Блоки
  сгущаются через 2 мм (ребро через корешок — с двойной точкой на нём), переносятся на кадр точной картой Кунса
  и упрощаются `approxPolyDP` с допуском 0.1 мм — кривые многоугольники. Нет двух областей или поверхность
  вырождена (складка, площадь ×1.5, самопересечение контура) — откат на трапеции.

## Меры (`measures.py`, по кадру, масштаб сегментации)

| Колонка | Что это |
|---|---|
| `*_line_cover` | доля «своей печати» (`book.content`: глифы цепочкой с зазором) внутри бумаги, накрытой объединением блоков без припуска |
| `*_lines_out_mm2` | та же печать вне всех блоков, мм² |
| `*_text_loose` | доля площади блоков `Text` дальше 2 мм от своей печати (рыхлость) |
| `*_spine_blocks`, `*_spine_mm2` | блоки с вершинами по обе стороны линии корешка дальше 3 мм (линия — по областям surya из кэша coons, иначе середина трапеций) |
| `*_protected_mm2` | защита контента (`paper.npz` `protected`), мм² |
| `*_cut_content_mm2` | своя печать в «не фоне» вне бумаги и защиты (срезано вырезкой), мм² |
| `*_blocks`, `*_types` | число и типы блоков |
| `matched`, `lost`, `added`, `mean_iou` | сопоставление блоков двух прогонов по IoU ≥ 0.5 (венгерский алгоритм) |
| `new_warp`, `new_reason`, `frames_reason` | что применено в прогоне coons и почему был откат |
| `*_surya_s`, `*_blocks_back`, `*_back_px` | время surya из `run.log`, copy back из `summary.csv` |
| `*_pages_*` | прямизна текста выпрямленных страниц после поля (`TextMeasures`: размах строк и сторон блоков, мм) — колонка `pages_text` сводки v2 после прохода с `--output-pages` (`OUTPUT_PAGES=1`) |

## Команды

```bash
SET=ph_1974_01 run_scripts/scan_cropping/run_v2_coons_layout.sh      # два прогона v2 + measure + report + sheets
uv run python -m research.surya_coons measure --base RUN/keystone --new RUN/coons --label SET --out-dir OUT --jobs 16
uv run python -m research.surya_coons report --out-dir OUT --top 20   # summary.csv, <набор>/select.txt
uv run python -m research.surya_coons sheets --base RUN/keystone --new RUN/coons --select OUT/SET/select.txt --out-dir OUT/SET/sheets
```

Прогоны — в `$SURYA_COONS_PH_RUNS`, `$SURYA_COONS_HP_RUNS`, меры и склейки — в `$SURYA_COONS_ROOT`
(`run_scripts/scan_cropping/common.sh`). Кэш для опытов — только полная копия (`cp -r`): `cp -al` делит inode, и запись
прогона по жёсткой ссылке переписала бы исходный кэш (запись `layout.json` теперь идёт через `os.replace`, но прочие
файлы кэша пишутся на месте).

Склейки `sheets`: `<кадр>_surya.jpg` (диптих «холст сети | кадр с перенесёнными блоками»), `<кадр>_main.jpg`,
`<кадр>_07_lines.jpg` (итоговые области сплошной линией, области surya без текста — пунктиром) — keystone сверху,
coons снизу. Итоги и выводы — `reports/surya_coons.md`.
