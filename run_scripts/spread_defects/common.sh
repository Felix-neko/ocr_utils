#!/usr/bin/env bash
#
# Общие пути стенда брака разворотов (research/spread_defects). Сам ничего не запускает — подключается
# через `source` из соседних run_*.sh.

# Пак-4 «Планового хозяйства» — исходники на Я.Диске (NTFS-3G, медленно; только чтение, ничего туда не писать).
PH_PACK4_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/Плановое хозяйство/в работе/пак-4 (450 dpi)"
# Выход стенда — SSD hotstore: плиточные сетки, сводки, листы глазами.
SPREAD_DEFECTS_ROOT="${SPREAD_DEFECTS_ROOT:-/mnt/hotstore/scan_processing/ph_pack4_defects}"

# Паки 2 и 3 «Вопросов экономики» (фото-развороты 450 dpi) — исходники на Я.Диске, только чтение.
VE_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/Вопросы экономики/в работе"
VE_PACK2_ROOT="$VE_ROOT/пак-2 (450 DPI )"
VE_PACK3_ROOT="$VE_ROOT/пак-3 (450 DPI улучшили фокус)"
# Выход стенда по пакам 2–3 ВЭ: pack2/, pack3/ (номера, признаки копий, эмбеддинги), result/ (оценки, участки).
VE_DEFECTS_ROOT="${VE_DEFECTS_ROOT:-/mnt/hotstore/scan_processing/ve_pack23_defects}"

# ЭГ 1985 10-12 (Химки): постраничная съёмка газетной подшивки, корешок у верхнего края кадра — исходники RAF на
# Я.Диске (только чтение). Выход: превью, кэш surya/DeepSeek, признаки, решения и папки симлинков.
EG_1985_10_12_RAW="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/неразобранное/2026-09-24 ЭГ/284_FUJI ЭГ 1985 10-12 Химки"
EG_1985_10_12_OUT="${EG_1985_10_12_OUT:-/mnt/hotstore/scan_processing/newspapers/eg/1985_10-12_gutter}"

# Партия «ФИН» 2026-10-07: журнальные развороты, декабрьские выпуски 1960–1979 (JPG, 434 dpi) — Я.Диск, только чтение.
FIN_2026_10_07_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/неразобранное/2026-10-07-ФИН"
FIN_2026_10_07_OUT="${FIN_2026_10_07_OUT:-/mnt/hotstore/scan_processing/fin_2026_10_07_defects}"

# «Хозяйство и право» 1991 (пробный пак): журнальные развороты JPG 434 dpi, 8 выпусков, 619 кадров — Я.Диск, только чтение.
HP_1991_PACK_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Пробное/Отобранное/Хозяйство и право"
# Прогон 2026-10-21 по всем 12 выпускам (с корешком и картой смаза); прежний (2026-10-08, 8 выпусков) — hp_1991_defects.
HP_1991_OUT="${HP_1991_OUT:-/mnt/hotstore/scan_processing/hp_1991_defects_v2}"

# Партия 2026-10-16: журнальные развороты 1979 года, выпуски 01–12 (JPG 5078×4075, 982 кадра) — Я.Диск, только чтение.
J_2026_10_16_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/неразобранное/2026-10-16"
J_2026_10_16_OUT="${J_2026_10_16_OUT:-/mnt/hotstore/scan_processing/j_2026_10_16_defects}"
# Тот же пак, выпуски 1978/07–12, 1979, 1981/01–03 (JPG 6273×5020, 1 164 кадра; 1981/04–12 на 2026-10-18 пустые).
HP_1978_1981_OUT="${HP_1978_1981_OUT:-/mnt/hotstore/scan_processing/hp_1978_1981_defects}"
# Тот же пак, выпуски 1981/01–12 и 1982/01–12 (JPG ~6300–6700 × 4900–5050, 434 dpi; 1981/01–03 — в папках «01–03 оплачено»).
HP_1981_1982_OUT="${HP_1981_1982_OUT:-/mnt/hotstore/scan_processing/hp_1981_1982_defects}"
# Тот же пак, 1977/01–12 … 1980/01–12 — пропуски и смаз (карта зон); папки «NN оплачено…», ~2 900 кадров.
HP_1977_1980_OUT="${HP_1977_1980_OUT:-/mnt/hotstore/scan_processing/hp_1977_1980_defects}"
# Та же съёмка «Вопросов экономики», 1977/01–11 — копия с файлообменника в DOWN (полнее Я.Диска на 2026-10-19: там
# 1977/07–12 пустые). Диск NTFS-3G, только чтение.
VE_DOWN_ROOT="/mnt/dump3/DOWN/VE"
VE_1977_OUT="${VE_1977_OUT:-/mnt/hotstore/scan_processing/ve_1977_down_defects}"
# Пак-1 «Планового хозяйства», 1990 (выпуски 09–12): развороты JPG 3922×3299 при 300 dpi (тег EXIF врёт: 212), 268 кадров —
# Я.Диск, только чтение. Пороги стенда — в px пака-4 (450 dpi), поэтому кадры сначала пересчитываются ×1.5 в
# $PH_PACK1_1990_OUT/frames_450 (SSD); расфокус — по оригиналам.
PH_PACK1_1990_ROOT="/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/Плановое хозяйство/в работе/пак-1 (300 dpi)"
PH_PACK1_1990_OUT="${PH_PACK1_1990_OUT:-/mnt/hotstore/scan_processing/ph_pack1_1990_defects}"
