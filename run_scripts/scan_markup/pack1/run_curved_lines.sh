#!/usr/bin/env bash
#
# Поиск полос пака-1 с кривыми, волнистыми или неравномерно наклонёнными строками.
#
# ЗАЧЕМ. FineReader с включённой коррекцией геометрии (искажение строк, перекос, трапеция)
# выправляет полосы с реально кривыми строками и портит полосы с прямыми. План — два
# пакетных прогона FineReader, с коррекцией и без, и сборка итогового PDF постранично;
# для этого нужно знать, на каких полосах строки кривые.
#
# ВХОД — заострённые копии на SSD: это ровно то, что ушло в PDF, и читаются они вчетверо
# быстрее оригиналов с /mnt/dump3.
#
# ВЫХОД — каталог симлинков (подкаталог на детектор, плюс combo, кандидаты и разметка),
# CSV со всеми метриками всех полос (по нему калибруются пороги), md-отчёт, контактный
# лист находок и оверлеи по находкам. Симлинки целят на ОРИГИНАЛЬНЫЕ TIFF. Кэш измерений
# лежит на SSD: повторный прогон с другими порогами (или команда report) не трогает ни
# картинки, ни GPU.
#
# ВРЕМЯ. Замер на 14 полосах: CPU-детекторы 0.3-0.5 с на полосу в воркере, surya 0.4 с на
# полосу на GPU. Оценка по паку (12 135 полос): около полутора часов, узкое место — surya.
# Без неё (--detectors skew_map,line_fit,strip_shift) — минут пятнадцать. Запускать в
# фоне и ждать ПО СОХРАНЁННОМУ PID (CLAUDE.md):
#
#   ./run_scripts/scan_markup/pack1/run_curved_lines.sh & PID=$!
#   while kill -0 "$PID" 2>/dev/null; do sleep 30; done
set -euo pipefail
source "$(dirname "$0")/common.sh"

# Снятие всей группы процессов при выходе: пул на forkserver переживает смерть хозяина
# (см. run_orientation_validate.sh и CLAUDE.md).
set -m
trap 'trap - EXIT INT TERM; kill -- -$$ 2>/dev/null' EXIT INT TERM

BASE_NAME="curved_lines_pack1"
CACHE_DIR="$MARKUP_ROOT/curved_lines_cache"
OVERLAY_DIR="$MARKUP_ROOT/curved_lines_overlay"   # оверлеев сотни по 400 КБ — не в репозиторий

ARGS=(
    run
    --root "$SHARPENED_DIR"
    --link-root "$PACK_DIR"      # симлинки целим на ОРИГИНАЛЫ: смотреть глазами лучше их
    --db "$DB_REVIEWED"          # только чтение, ради номеров страниц промежуточного PDF
    --pack-name "$PACK_NAME"
    --jobs 12                    # 16 ядер минус те, что заняты постобработкой surya в родителе
    --gpu-side 1536              # ~143 dpi: на этом масштабе калиброваны пороги surya_lines
    --gpu-batch 16
    --cache-dir "$CACHE_DIR"
    --labels run_scripts/scan_markup/pack1/curved_lines_labels.csv
    --link-dir "${BASE_NAME}_links"
    --csv "${BASE_NAME}.csv"
    --md-report "${BASE_NAME}.md"
    --sheet "${BASE_NAME}_sheet.png"
    --overlay-dir "$OVERLAY_DIR"
    --overlay flagged            # оверлеи только по находкам (хоть один флаг)
    # --detectors skew_map,line_fit,strip_shift,surya_lines — умолчание, все четыре.
    # --thr ДЕТЕКТОР.МЕТРИКА=ЧИСЛО — перекрыть порог; --list-thresholds — показать.
    # --combo-votes 2 --combo-strong 1.5 — сводный флаг: два голоса ИЛИ один уверенный.
)

# РЕЖИМ ПОРОГОВ. Пороги в коде детекторов — строгие (580 полос, 4.8% пака, точность ~90%
# при score ≥ 2). По просьбе пользователя (сентябрь 2026) прогон идёт в ЩЕДРОМ режиме:
# пропуск кривой полосы опаснее ложной коррекции (FineReader не распознает загнутые
# кончики), допустимо ложных не больше, чем верных. Щедрые пороги — строгие минус 40%;
# на выборке 1976/12 вместе с end_curl это 14 из 20 полос с загибами при 1 ложной из 12.
# Строгий режим — закомментировать массив GENEROUS.
GENEROUS=(
    --thr skew_map.max_dev_deg=1.2 --thr skew_map.resid_deg=0.36 --thr skew_map.spread_deg=0.9
    --thr line_fit.sagitta_rel_p90=0.11 --thr line_fit.sagitta_rel_max3=0.17
    --thr line_fit.slope_spread_deg=0.54 --thr line_fit.slope_resid_deg=0.18
    --thr surya_lines.sagitta_rel_p90=0.09 --thr surya_lines.sagitta_rel_max3=0.13
    --thr surya_lines.slope_spread_deg=0.54 --thr surya_lines.slope_resid_deg=0.21
    --thr strip_shift.resid_max_rel=0.07 --thr strip_shift.angle_max_dev_deg=0.6
)
ARGS+=("${GENEROUS[@]}")

echo "Поиск полос с кривыми строками в $SHARPENED_DIR"
uv run python -m ocr_utils.scan_markup.curved_lines "${ARGS[@]}" "$@"
