#!/usr/bin/env bash
#
# Общие пути прогонов scan_cropping v2. Сам ничего не запускает — подключается через `source` из соседних run_*.sh.
# (Прогоны v1 в этой папке пока держат пути у себя.)

# Пак-6 «от Лены» (1989–1992): фото-развороты подшивок МТО и РИСК, JPG — Я.Диск (NTFS-3G, медленно; только чтение,
# ничего туда не писать).
PACK6_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-6 от Лены (1989-1992)"
# Выход v2 — SSD hotstore: итоговые прямоугольники, кэш шагов, оверлеи.
SCAN_CROPPING_V2_ROOT="${SCAN_CROPPING_V2_ROOT:-/mnt/hotstore/scan_processing/scan_cropping_v2}"

# Пак-6 целиком (МТС 1990–1991, РИСК 1991–1993), JPG 5875×4570 — Я.Диск, только чтение. Обе папки РИСК 1991/05
# («05 …» и «05 fix …», где 12 кадров заменены) обрабатываются как есть.
PACK6_450_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-6 от Лены (1989-1993) (450 DPI)"
# Прогон v2 пака-6 целиком: итоги, кэш и оверлеи — пока на SSD; в «Готовое» на Я.Диске — после отсмотра.
# (Остановленный 2026-10-12 первый прогон — в $SCAN_CROPPING_V2_ROOT/pack6_stopped_2026-10-12.)
PACK6_450_WORK="$SCAN_CROPPING_V2_ROOT/pack6_v2"
PACK6_450_READY="$PACK6_450_WORK/output"
# Пак-4 «Плановое хозяйство» (ПХ), фото-развороты на чёрном коврике, JPG 5660×4244 — Я.Диск, только чтение. Плотность
# бумаги — 13.8 px/мм, как у паков 5–6 (замер research/edge_cut_pages: ~6.9 px/мм на ½ кадра); «450 dpi» в имени — не она.
PACK4_PH_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/Плановое хозяйство/в работе/пак-4 (450 dpi)"
# Прогоны v2 по паку-4 ПХ: итоги, кэш и оверлеи — на SSD hotstore, по годам внутри output/cache/debug.
PACK4_PH_WORK="$SCAN_CROPPING_V2_ROOT/ph/pack_4"
# Набор разворотов ПХ 1975 с проклейкой бумажной полосой у корешка (стыки тетрадей): симлинки на кадры пака-4 по
# классам strip/neighbor/maybe и strip_set.csv — отбор глазами 2026-10-21; прогоны v2 по нему — в gutter_strip_check.
PH_GUTTER_STRIP_SET="$PACK4_PH_WORK/sets/gutter_strip"
PH_GUTTER_STRIP_WORK="$SCAN_CROPPING_V2_ROOT/ph/gutter_strip_check"
# Стенд research/spread_lines (прямые разворота по кэшу прогонов v2): CSV и картинки — на SSD hotstore.
SPREAD_LINES_ROOT="/mnt/hotstore/scan_processing/spread_lines"

# «Хозяйство и право» 1991/02 (пробный пак), фото-развороты на чёрном коврике, JPG 5392×4394, 434 dpi — Я.Диск, только
# чтение. Страницы сильно изогнуты (сверху, снизу и по бокам) — вход стенда research/page_cutting (группа curved).
HP_1991_02_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Пробное/Отобранное/Хозяйство и право/1991/02 оплачено"
# Прогон v2 по этому выпуску: итоги, кэш и оверлеи — на SSD hotstore.
HP_1991_02_WORK="${HP_1991_02_WORK:-$SCAN_CROPPING_V2_ROOT/hp_1991_02}"

# Боевой прогон v2 пака-6 целиком (6-я версия ключей, 2026-10-15/16, 2480 кадров) — вход стенда research/page_cutting.
PACK6_V6_WORK="$SCAN_CROPPING_V2_ROOT/pack6_v6"
# Стенд «нарезка сканов» (research/page_cutting): меры, кандидаты, листы, выпрямленные страницы, выход ScanTailor — SSD.
PAGE_CUTTING_ROOT="${PAGE_CUTTING_ROOT:-/mnt/hotstore/scan_processing/page_cutting}"
# scantailor-cli — ScanTailor Advanced без окна (ветка cli, сборка build-cli; Qt offscreen).
SCANTAILOR_CLI="${SCANTAILOR_CLI:-$HOME/Projects/scantailor-advanced/build-cli/scantailor-cli}"
