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

# Пак-5 МТС (1985 и далее + немножко пересканов), фото-развороты на чёрном коврике, JPG 5549×4160 — Я.Диск, только
# чтение. Съёмка как у пака-6: по шагу строк 1990/01–03 против пака-6 МТС 1990/01–02 плотность на 1–2 % ниже
# (53.0 / 65.9 px против 53.6 / 67.1 px; замер 2026-10-19) — по решению пользователя ключи пака-6 берутся как есть.
PACK5_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/в работе/пак-5 (1985 и далее + немножко пересканов) (450 DPI)"
# Прогон v2 по 1990/01–03: итоги, кэш и оверлеи — на SSD hotstore.
PACK5_1990_Q1_WORK="$SCAN_CROPPING_V2_ROOT/pack5_1990_01-03"

# Пак-4 «Плановое хозяйство» (ПХ), фото-развороты на чёрном коврике, JPG 5660×4244 — Я.Диск, только чтение. Плотность
# бумаги — 13.8 px/мм, как у паков 5–6 (замер research/edge_cut_pages: ~6.9 px/мм на ½ кадра); «450 dpi» в имени — не она.
PACK4_PH_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/Плановое хозяйство/в работе/пак-4 (450 dpi)"
# Прогоны v2 по паку-4 ПХ: итоги, кэш и оверлеи — на SSD hotstore, по годам внутри output/cache/debug.
PACK4_PH_WORK="$SCAN_CROPPING_V2_ROOT/ph/pack_4"
# Набор разворотов ПХ 1975 с проклейкой бумажной полосой у корешка (стыки тетрадей): симлинки на кадры пака-4 по
# классам strip/neighbor/maybe и strip_set.csv — отбор глазами 2026-10-21; прогоны v2 по нему — в gutter_strip_check.
PH_GUTTER_STRIP_SET="$PACK4_PH_WORK/sets/gutter_strip"
PH_GUTTER_STRIP_WORK="$SCAN_CROPPING_V2_ROOT/ph/gutter_strip_check"
# Пробные прогоны v2 с выходом выпрямленных страниц (--output-pages) и флагами кадров из сайдкаров: итоги разворотов,
# страницы, кэш и оверлеи — на SSD hotstore, по выпускам внутри.
PH_PAGES_WORK="$SCAN_CROPPING_V2_ROOT/ph/pages_try"
# Стенд research/spread_lines (прямые разворота по кэшу прогонов v2): CSV и картинки — на SSD hotstore.
SPREAD_LINES_ROOT="/mnt/hotstore/scan_processing/spread_lines"
# Стенд research/gutter_strip (полоса-вклейка у корешка по кэшу прогонов v2): признаки, листы, оверлеи — SSD.
GUTTER_STRIP_ROOT="${GUTTER_STRIP_ROOT:-/mnt/hotstore/scan_processing/gutter_strip}"

# «Хозяйство и право» 1991/02 (пробный пак), фото-развороты на чёрном коврике, JPG 5392×4394, 434 dpi — Я.Диск, только
# чтение. Страницы сильно изогнуты (сверху, снизу и по бокам) — вход стенда research/page_cutting (группа curved).
HP_1991_02_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Пробное/Отобранное/Хозяйство и право/1991/02 оплачено"
# Прогон v2 по этому выпуску: итоги, кэш и оверлеи — на SSD hotstore.
HP_1991_02_WORK="${HP_1991_02_WORK:-$SCAN_CROPPING_V2_ROOT/hp_1991_02}"
# Весь 1991 год того же журнала (12 выпусков «NN оплачено…», 966 кадров) — Я.Диск, только чтение; прогон v2 по
# выпускам — на SSD hotstore, отдельно от hp_1991_02.
HP_1991_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Пробное/Отобранное/Хозяйство и право/1991"
HP_1991_WORK="${HP_1991_WORK:-$SCAN_CROPPING_V2_ROOT/hp_1991}"
# Прогоны v2 с подгоночными рамками страниц сплайнами (прежний и с узлами на изгибах) на шаге 07_lines: итоги и
# оверлеи — отдельно по наборам, кэш шагов — прежних прогонов (рамки на него не влияют).
SPLINE_BENDS_WORK="${SPLINE_BENDS_WORK:-$SCAN_CROPPING_V2_ROOT/spline_bends}"
# Те же наборы с полосой-вклейкой и подгоночными областями страниц (сплайн с узлами, у корешка — линия корешка, у полосы —
# до вспомогательных линий и отрезок через полосу) на шаге 07_lines и склейкой по частям страниц у щели.
SPLINE_FRAMES_WORK="${SPLINE_FRAMES_WORK:-$SCAN_CROPPING_V2_ROOT/spline_frames}"
# Рваные внешние края страниц (двойная грубая сегментация + детектор + тонкое уточнение, research/ragged_edges): итоги,
# оверлеи и свои копии кэша (без SAM 3, бумаги, страниц, surya) по наборам — SSD; кэши прежних прогонов не трогаются.
RAGGED_EDGES_WORK="${RAGGED_EDGES_WORK:-$SCAN_CROPPING_V2_ROOT/ragged_edges}"
# Перепрогон тех же наборов после правок 2026-10-23 (палец у края в уточнении, заполнение пропусков края грубым краем,
# щель полосы-вклейки без оторванных островков, порог 8.2 мм, потоки закраса пальцев): кэш SAM 3 — из $RAGGED_EDGES_WORK.
RAGGED_EDGES_V2_WORK="${RAGGED_EDGES_V2_WORK:-$SCAN_CROPPING_V2_ROOT/ragged_edges_v2}"
# Тестовый набор рваных и ровных сторон: 210 кадров стенда уточнения (stack_edges/refine_set) + примеры пользователя.
RAGGED_EDGES_SET="${RAGGED_EDGES_SET:-$RAGGED_EDGES_WORK/set_in}"
# Итоги spline_frames, выпрямленные по подгоночным областям страниц (research/page_rectify): выборка и все кадры — SSD.
SPLINE_RECTIFIED_SAMPLE_WORK="${SPLINE_RECTIFIED_SAMPLE_WORK:-$SCAN_CROPPING_V2_ROOT/spline_frames_rectified_sample}"
SPLINE_RECTIFIED_WORK="${SPLINE_RECTIFIED_WORK:-$SCAN_CROPPING_V2_ROOT/spline_frames_rectified}"
# То же плюс выравнивание освещения и второе преобразование по тексту (research/page_rectify text-run) — SSD.
TEXT_RECTIFIED_SAMPLE_WORK="${TEXT_RECTIFIED_SAMPLE_WORK:-$SCAN_CROPPING_V2_ROOT/spline_frames_text_rectified_sample}"
TEXT_RECTIFIED_WORK="${TEXT_RECTIFIED_WORK:-$SCAN_CROPPING_V2_ROOT/spline_frames_text_rectified}"
# Прогон v4 второго преобразования (2026-10-23, по тегу dpi) — вход меры стенда research/gutter_compression — SSD.
TEXT_RECTIFIED_V4_WORK="${TEXT_RECTIFIED_V4_WORK:-$SCAN_CROPPING_V2_ROOT/spline_frames_text_rectified_v4}"
# Стенд «сжатие букв у корешка» (research/gutter_compression): окна строк, сводки, листы, выпрямление с растяжением — SSD.
GUTTER_COMPRESSION_ROOT="${GUTTER_COMPRESSION_ROOT:-/mnt/hotstore/scan_processing/gutter_compression}"

# Боевой прогон v2 пака-6 целиком (6-я версия ключей, 2026-10-15/16, 2480 кадров) — вход стенда research/page_cutting.
PACK6_V6_WORK="$SCAN_CROPPING_V2_ROOT/pack6_v6"
# Стенд «нарезка сканов» (research/page_cutting): меры, кандидаты, листы, выпрямленные страницы, выход ScanTailor — SSD.
PAGE_CUTTING_ROOT="${PAGE_CUTTING_ROOT:-/mnt/hotstore/scan_processing/page_cutting}"
# Стенд «заострение края листа перед SAM 3» (research/edge_sharpen): SAM 3 и бумага по фильтрам, меры, листы — SSD.
EDGE_SHARPEN_ROOT="${EDGE_SHARPEN_ROOT:-/mnt/hotstore/scan_processing/edge_sharpen}"
# scantailor-cli — ScanTailor Advanced без окна (ветка cli, сборка build-cli; Qt offscreen).
SCANTAILOR_CLI="${SCANTAILOR_CLI:-$HOME/Projects/scantailor-advanced/build-cli/scantailor-cli}"

# surya layout по холсту Кунса против холста по трапециям (run_v2_coons_layout.sh, стенд research/surya_coons):
# прогоны v2 по полным копиям кэша — на SSD hotstore рядом с прочими прогонами v2; меры и склейки — отдельно.
SURYA_COONS_PH_RUNS="${SURYA_COONS_PH_RUNS:-$SCAN_CROPPING_V2_ROOT/ph/coons_layout}"
SURYA_COONS_HP_RUNS="${SURYA_COONS_HP_RUNS:-$SCAN_CROPPING_V2_ROOT/hp_1991_coons}"
SURYA_COONS_ROOT="${SURYA_COONS_ROOT:-/mnt/hotstore/scan_processing/surya_coons}"
