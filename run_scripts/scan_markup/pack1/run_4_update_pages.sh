#!/usr/bin/env bash
#
# Шаг 4: полоса на диске обновилась — привести к ней базу, уменьшенную копию и CVAT.
#
#   ./run_4_update_pages.sh 1973/05/IMG_0093_1L.tif 1972/11/IMG_0062_2R.tif
#   ./run_4_update_pages.sh /mnt/.../пак-1/1973/05/IMG_0093_1L.tif
#
# Принимает пути полос — относительные (от корня пака) или абсолютные. Год пересоздаётся
# ОДИН раз, сколько бы полос в нём ни обновилось.
#
# Что делает и почему именно так:
#
# 1. Бэкапит обе базы — свою (sqlite) и CVAT (pg_dump). Дальше идут необратимые шаги, а в
#    базе CVAT лежит ручная разметка, которой больше нигде нет.
# 2. Гоняет detect по затронутым выпускам с --skip-detected: пересчитается ровно та полоса,
#    у которой разошёлся хеш, остальные будут пропущены по stat.
# 3. Гоняет to-cvat с --recreate-stale по затронутым годам. Подменить кадр в существующей
#    задаче CVAT не даёт — границы джобов задаются один раз при создании, — поэтому задача
#    года пересоздаётся, а разметка неизменившихся полос и состояния джобов переносятся.
#    Уменьшенная копия переделывается сама: to-cvat видит по хешу, что файл под кадром другой.
# 4. Сверяет, что число шейпов до и после совпало. Проверка не формальность: именно она
#    поймала потерю разметки при переносе полосы между выпусками.
#
# ЧЕГО СКРИПТ НЕ ДЕЛАЕТ. Он не снимает с выпуска «завершён»: судить, надо ли смотреть его
# заново, может только человек — он знает, что на полосе. Автоматика лишь сообщает, что
# выпуск разошёлся с диском. (Полоса, ДОБАВЛЕННАЯ в выпуск, — другое дело, её разметчик не
# видел вовсе, и там «завершён» снимается автоматически.)

set -euo pipefail
source "$(dirname "$0")/common.sh"

if [ $# -eq 0 ]; then
    echo "Укажите обновившиеся полосы: $(basename "$0") 1973/05/IMG_0093_1L.tif [ещё...]" >&2
    exit 1
fi

# --- разбор путей -------------------------------------------------------------------------
# Полосы приводятся к пути относительно корня пака: именно в таком виде они лежат в базе.
declare -a REL_PATHS=()
declare -a YEARS=()
declare -a ISSUES=()
for arg in "$@"; do
    rel="${arg#"$PACK_DIR"/}"
    if [ ! -f "$PACK_DIR/$rel" ]; then
        echo "Нет такой полосы: $PACK_DIR/$rel" >&2
        exit 1
    fi
    year="${rel%%/*}"
    rest="${rel#*/}"
    issue="${rest%%/*}"
    if [ "$year" = "$rel" ] || [ "$issue" = "$rest" ]; then
        echo "Путь не похож на <год>/<выпуск>/<полоса>: $rel" >&2
        exit 1
    fi
    REL_PATHS+=("$rel")
    YEARS+=("$year")
    ISSUES+=("$year/$issue")
done

# Год и выпуск могут повторяться: две обновлённые полосы одного выпуска — обычное дело.
mapfile -t UNIQ_YEARS < <(printf '%s\n' "${YEARS[@]}" | sort -u)
mapfile -t UNIQ_ISSUES < <(printf '%s\n' "${ISSUES[@]}" | sort -u)

echo "Полос:    ${#REL_PATHS[@]} (${REL_PATHS[*]})"
echo "Выпусков: ${UNIQ_ISSUES[*]}"
echo "Годов:    ${UNIQ_YEARS[*]}"

# --- 1. бэкапы ----------------------------------------------------------------------------
# common.sh уже положил рядом с базой .bak, но он один на все шаги и перезаписывается.
# Здесь бэкап именной и с меткой времени: к нему возвращаются через недели.
BACKUP_DIR="$MARKUP_ROOT/backup_update_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp "$DB" "$BACKUP_DIR/pack1.sqlite"
DB_CONTAINER=cvat_mts_db
docker exec "$DB_CONTAINER" pg_dump \
    -U "$(docker exec "$DB_CONTAINER" printenv POSTGRES_USER)" \
    -d "$(docker exec "$DB_CONTAINER" printenv POSTGRES_DB)" > "$BACKUP_DIR/cvat.sql"
echo ">> Бэкапы: $BACKUP_DIR ($(du -sh "$BACKUP_DIR" | cut -f1))"

# Сколько шейпов было в каждом затронутом году — с этим сверимся в конце.
BEFORE_JSON="$BACKUP_DIR/shapes_before.json"
uv run python -m ocr_utils.scan_markup count-shapes \
    --db "$DB" --pack-name "$PACK_NAME" --out "$BEFORE_JSON" "${UNIQ_YEARS[@]}"

# --- 2. детекция ---------------------------------------------------------------------------
for issue in "${UNIQ_ISSUES[@]}"; do
    echo ">> Детекция: ${issue}"
    uv run python -m ocr_utils.scan_markup detect \
        --pack-dir "$PACK_DIR" \
        --db "$DB" \
        --pack-name "$PACK_NAME" \
        --debug-dir "$DEBUG_DIR" \
        --skip-detected \
        --jobs 8 \
        --first-page-is-cover \
        --only-year "${issue%%/*}" \
        --only-issue "${issue##*/}"
done

# --- 3. CVAT -------------------------------------------------------------------------------
for year in "${UNIQ_YEARS[@]}"; do
    echo ">> CVAT: пересоздаю задачу ${year}"
    uv run python -m ocr_utils.scan_markup to-cvat \
        --db "$DB" \
        --pack-name "$PACK_NAME" \
        --share-root "$SHARE_ROOT" \
        --only-year "$year" \
        --recreate-stale \
        --annotator user \
        --jobs 4
done

# --- 4. сверка -----------------------------------------------------------------------------
# Убыль шейпов — это код возврата 2, и падать на нём нельзя: именно в этот момент оператору
# и нужно сообщение о том, где лежит спасённая разметка. Поэтому проверка идёт через if.
echo ">> Сверка"
LOST=0
if ! uv run python -m ocr_utils.scan_markup count-shapes \
    --db "$DB" --pack-name "$PACK_NAME" --compare "$BEFORE_JSON" "${UNIQ_YEARS[@]}"; then
    LOST=1
fi

echo
echo "================================================================"
if [ "$LOST" = 1 ]; then
    echo " ВНИМАНИЕ: шейпов стало меньше. Убыль законна, если у заменённой"
    echo " полосы была разметка — она относилась к старому файлу. Во всех"
    echo " прочих случаях смотрите список кадров выше."
else
    echo " Готово, разметка на месте вся."
fi
echo
echo " Разметка старых задач: $MARKUP_ROOT/cvat_backup/"
echo " Обе базы целиком:      $BACKUP_DIR"
echo "================================================================"
[ "$LOST" = 0 ]
