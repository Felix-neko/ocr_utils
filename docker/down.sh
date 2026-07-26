#!/usr/bin/env bash
# Останавливает локальный CVAT.
#   ./down.sh          — остановить контейнеры, данные (БД, разметка) сохраняются.
#   ./down.sh --wipe   — то же + удалить тома CVAT (полный сброс: БД, разметка, чанки).
# Оригинальные картинки НЕ трогаются (они смонтированы read-only, это не том).
set -euo pipefail

cd "$(dirname "$0")"

COMPOSE=(docker compose
  --project-directory cvat-src
  --env-file .env
  -f cvat-src/docker-compose.yml
  -f docker-compose.override.yml)

if [ ! -d cvat-src ]; then
  echo "cvat-src/ отсутствует — нечего останавливать (CVAT ещё не запускался)." >&2
  exit 0
fi

case "${1:-}" in
  --wipe|-v)
    echo ">> Останавливаю CVAT и УДАЛЯЮ тома (полный сброс) ..."
    "${COMPOSE[@]}" down -v --remove-orphans
    ;;
  "")
    echo ">> Останавливаю CVAT (данные сохраняются) ..."
    "${COMPOSE[@]}" down --remove-orphans
    ;;
  *)
    echo "Неизвестный аргумент: $1 (допустимо: --wipe)" >&2
    exit 1
    ;;
esac
echo ">> Готово."
