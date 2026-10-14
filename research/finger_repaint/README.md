# research/finger_repaint — тёмные пятна LaMa на месте пальцев

Стенд проверки гипотезы «LaMa оставляет серое пятно на месте пальца, потому что в окне много чёрного коврика;
повторный закрас по «палец ∩ книга» его уберёт». Отчёт — `reports/finger_repaint.md`. Основной код `scan_cropping`
не меняется: стенд вызывает его функции (`fingers.text_free_context`, `copy_back_ink`, `GpuModels.inpaint`,
`pipeline.cpu_finish`).

| Модуль | Что |
|---|---|
| `measure.py` | мера пятна: провал L закраса пальца на бумаге (без copy back и букв) против кольца бумаги 3–10 мм |
| `repaint.py` | варианты второго прохода B–D (LaMa по «палец ∩ книга») и тоновая поправка E–G |
| `cli.py` | `select` (мера по прогону v2), `sheets` (листы отбора), `run` (варианты + штатный финал), `compare` (листы «было \| стало») |
| `sets/` | `dark_spots.txt` (28 кадров), `control.txt` (15) — пути относительно корня пака-6 |

Итог: окно и коврик ни при чём (B–D: −15–17 %), пятно — тон самого закраса LaMa; тоновая поправка G: −72 %.

## Запуск

```bash
R=/mnt/hotstore/scan_processing/finger_repaint
uv run python -m research.finger_repaint select --run /mnt/hotstore/scan_processing/scan_cropping_v2/pack6_v3 --out-dir $R/select2_pack6_v3
uv run python -m research.finger_repaint sheets --select-dir $R/select2_pack6_v3 --min-area 20
./run_scripts/scan_cropping/run_v2_finger_repaint_set.sh                  # база с clean.png
uv run python -m research.finger_repaint run --base $R/base --src-root "<корень пака-6>" \
    --set research/finger_repaint/sets/dark_spots.txt --set research/finger_repaint/sets/control.txt --out-dir $R/variants
uv run python -m research.finger_repaint select --run $R/variants/G_tone_fine --out-dir $R/measure/G_tone_fine
uv run python -m research.finger_repaint compare --base $R/base --runs $R/variants --set research/finger_repaint/sets/dark_spots.txt
# H, I, J — круговая дыра под тень (GPU), затем мера и листы
uv run python -m research.finger_repaint run --base $R/base --src-root $R/input --set research/finger_repaint/sets/dark_spots.txt \
    --set research/finger_repaint/sets/control.txt --out-dir $R/variants --variants H_sym_hole,I_sym_hole_2x,J_sym_paste_all
uv run python -m research.finger_repaint compare --base $R/base --runs $R/variants --set research/finger_repaint/sets/dark_spots.txt \
    --variants H_sym_hole,J_sym_paste_all,G_tone_fine --suffix _HJ
# оверлеи зон: старая и новая дыра, окна новой дыры
uv run python -m research.finger_repaint.zones --base $R/base --src-root $R/input --set research/finger_repaint/sets/dark_spots.txt --set research/finger_repaint/sets/control.txt --out-dir $R/zones
# карта тени и осветление итога (A или J): профиль досягаемости тени, оценщики, меры в shadow_<источник>.csv
uv run python -m research.finger_repaint reach --base $R/base --set research/finger_repaint/sets/dark_spots.txt
uv run python -m research.finger_repaint shadow --ref $R/base --source $R/base --source-name A_v2 --set research/finger_repaint/sets/dark_spots.txt \
    --set research/finger_repaint/sets/control.txt --out-dir $R/shadow2 --methods T1_telea,P_pull_push --also $R/variants/J_sym_paste_all
# база без высветления зоны пальца перед LaMa
BASE_NAME=base_nolight EXTRA_ARGS="--finger-zone-light-increment 0,0" ./run_scripts/scan_cropping/run_v2_finger_repaint_set.sh
# что уходит в LaMa: перехват входа сети в A–D, лист на кадр
uv run python -m research.finger_repaint.rois --base $R/base --src-root $R/input --set research/finger_repaint/sets/dark_spots.txt --out-dir $R/rois
```

Проверки по листам `rois` (IMG_1022, IMG_0551, IMG_0265):
- **под дырой сеть ничего не видит** — вход под дырой как есть, белым, чёрным или шумом даёт один и тот же выход;
- **пятно строится из контекста:**
  - B, C и D дают пятно той же формы на том же месте;
  - чем сильнее ужата дыра (`hole_max_px` 600 → 80), тем темнее пятно;
  - при 600 пятно чуть светлее, чем при 300.
- **Главный источник — тень пальца на бумаге у края маски.** Опыт на 78 окнах B из `dark_spots`, разовым скриптом вне стенда; в 50 из них пятно заметно, провал L ≥ 20:
  - кольцо 8 мм вокруг дыры (в книге) заменили ровным тоном бумаги — пятно ушло в 31 окне из 50 (провал < 10);
  - в 15 окнах пятно осталось — их глазами пока не смотрел;
  - весь контекст заменили тоном бумаги — пятна нет нигде.
  - Тень лежит вне раздутой маски пальца, поэтому остаётся в контексте: LaMa продолжает её в дыру.
  - Вывод «пятно — собственный тон LaMa» из первой версии отчёта неверен.

`run` пишет `timing.csv` заново на каждый запуск (время LaMa только выбранных вариантов).

**Круговая дыра под тень (H, I, J; 2026-10-14).**

LaMa красит ядро пальца, раздутое кругом; в кадр вклеивается только часть закраса.

| Вариант | Радиус круга | Куда вклеивается закраc | Краска с оригинала |
|---|---|---|---|
| H | 3× припуска (накрывает асимметричную зону целиком) | асимметричная зона, рампа 1.5 мм внутрь от её края | как в первом проходе |
| I | 2× припуска | как в H | как в первом проходе |
| J | как в H | весь круг, рампа у его края | по всему кругу (`copy_back_ink`) |

Мера на `dark_spots` (28 кадров) и `control` (15):

| Вариант | Сумма пятен, мм² | Медиана, мм² | Кадров с пятном ≥ 50 мм² | Кадров без пятна | Control, сумма, мм² |
|---|---|---|---|---|---|
| A | 9126 | 255 | 27 | 0 | 0 |
| H | 3419 | 55 | 16 | 8 | 16 |
| I | 5454 | 177 | 23 | 2 | 8 |
| J | 3391 | 53 | 16 | 8 | 23 |

Что видно на листах:
- В H тело пятна уходит, но у края асимметричной зоны остаётся тонкий тёмный серп. Это тень вне зоны вклейки; мера её почти не ловит.
- J убирает и серп: РИСК 1993/02 IMG_0136, IMG_0145, МТС 1990/03 IMG_1022.
- Тёмная полоса у края листа (РИСК 1993/02 IMG_0126) остаётся во всех вариантах.
- На контроле у H и J появляются редкие мелкие точки от LaMa (РИСК 1991/02 IMG_0790).

**Карта тени и осветление итога (`shadow.py`, схема пользователя; 2026-10-14).**

Шаги:
1. Итог A (закрас и copy back как есть).
2. Область поиска: ядро пальца + 25 мм ∩ книга.
3. Карта затемнения бумаги: опора «без тени» (R) против «бумаги как есть» (B).
   - Из оценки исключены краска, кожа (только пиксели оригинала, сдвиг от окрестной бумаги в красное) и крупная печать.
   - Плавный рост от края области даёт smoothstep-рампа 4 мм.
4. Прибавка карты к итогу.

Подробно, с таблицами — `reports/finger_repaint.md`.

Итог `shadow2`, мера остатка тени в области поиска, `dark_spots` / `control`:

| Вариант | Остаток, мм² | Control, мм² |
|---|---|---|
| A | 17 839 | 338 |
| J | 11 904 | 424 |
| A + P (pull-push), прибавка | 3 290 | 9 |
| A + T1 (Telea, вариант пользователя) | 3 333 | 9 |
| J + P | 3 318 | 9 |

Выводы:
- **Время на кадр:** P — 0.5 с, T1 — 2.6 с.
- **Глазами чище всех J + P.**
- **Коэффициент по каналам хуже прибавки:** на глубоком пятне LaMa (IMG_0558) он даёт цветной сдвиг.
- **Остальные оценщики хуже по мере остатка:**
  - T2 (LaMa как опора) — −53 %;
  - Q (робастный фон) — −56 %;
  - S (поверхность) — −26 %.

**Без старого высветления (`base_nolight`, `--finger-zone-light-increment 0,0`).**

Мера остатка тени (`dark_spots`):

| Вариант | Остаток, мм² |
|---|---|
| A0 до карты | 25 904 |
| A0 + P | 3 183 |
| A + P | 3 290 |
| K + P (J без высветления) | 3 206 |

- Высветление само оставляло светлый ореол на месте пальца (контроль IMG_0790): у A0 + P его нет.
- J/K с круговой дырой рисуют на контроле тёмные точки LaMa.
- Кандидат на перенос — A0 + P.
