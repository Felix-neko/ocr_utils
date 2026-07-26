#!/usr/bin/env bash
# Поднимает локальный CVAT и настраивает пользователей/организацию/проект/задачи.
# Повторный запуск безопасен (идемпотентен): клон не переклонируется, задачи
# не дублируются.
set -euo pipefail

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

# --- чтение значений из .env (без source: значения содержат пробелы/кириллицу) ---
env_get() { grep -E "^$1=" .env | head -1 | cut -d= -f2-; }

CVAT_VERSION="$(env_get CVAT_VERSION)"
IMAGES_DIR="$(env_get IMAGES_DIR)"
CVAT_HOST="$(env_get CVAT_HOST)"
CVAT_PORT="$(env_get CVAT_PORT)"
SDK_VERSION="${CVAT_VERSION#v}"          # v2.70.0 -> 2.70.0
NETWORK="cvat_mts_cvat"                   # проект 'cvat_mts' + сеть 'cvat' (см. name: в override)
SERVER_CONTAINER="cvat_mts_server"

if [ ! -d "$IMAGES_DIR" ]; then
  echo "ОШИБКА: папка с картинками не найдена: $IMAGES_DIR" >&2
  exit 1
fi

# --- клонирование CVAT нужной версии (базовый compose монтирует файлы репо) ---
if [ ! -d cvat-src/.git ]; then
  echo ">> Клонирую CVAT $CVAT_VERSION в cvat-src/ ..."
  git clone --branch "$CVAT_VERSION" --depth 1 https://github.com/cvat-ai/cvat.git cvat-src
else
  echo ">> cvat-src/ уже на месте, пропускаю клонирование."
fi

COMPOSE=(docker compose
  --project-directory cvat-src
  --env-file .env
  -f cvat-src/docker-compose.yml
  -f docker-compose.override.yml)

echo ">> Поднимаю контейнеры (при первом запуске тянутся образы, это долго) ..."
"${COMPOSE[@]}" up -d

# --- ждём готовности API ---
# Спрашиваем сервер ИЗНУТРИ его контейнера, а не снаружи через localhost:8080.
# Снаружи проверка врёт: пока compose пересоздаёт cvat_server (например, из-за
# смены пути к картинкам), traefik какое-то время продолжает отдавать 200 со
# старого, ещё не умершего контейнера — ожидание проскакивает, и следующий шаг
# получает 'connection refused' от нового, ещё не начавшего слушать.
echo -n ">> Жду готовности cvat_server "
ready=0
for _ in $(seq 1 90); do
  code="$(docker exec "$SERVER_CONTAINER" \
    curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/api/server/about 2>/dev/null || true)"
  case "$code" in
    200|401|403) ready=1; break ;;
  esac
  echo -n "."
  sleep 5
done
echo
if [ "$ready" != "1" ]; then
  echo "ОШИБКА: сервер не поднялся за отведённое время. Смотри: ${COMPOSE[*]} logs cvat_server" >&2
  exit 1
fi

# И отдельно — что снаружи тоже открыто: этим адресом пользуется человек.
outer="$(curl -s -o /dev/null -w '%{http_code}' "http://${CVAT_HOST}:${CVAT_PORT}/api/server/about" || true)"
echo ">> Сервер отвечает (внутри сети: $code, снаружи через traefik: $outer)."

# --- пользователи / организация / членства (Django-shell внутри контейнера) ---
echo ">> Создаю пользователей и организацию ..."
docker exec -i \
  -e ADMIN_USER="$(env_get ADMIN_USER)" \
  -e ADMIN_PASS="$(env_get ADMIN_PASS)" \
  -e ANN_USER="$(env_get ANN_USER)" \
  -e ANN_PASS="$(env_get ANN_PASS)" \
  -e ORG_SLUG="$(env_get ORG_SLUG)" \
  -e ORG_NAME="$(env_get ORG_NAME)" \
  "$SERVER_CONTAINER" python3 /home/django/manage.py shell < create_users.py

# --- проект и задачи через cvat-sdk в эфемерном контейнере ---
echo ">> Создаю проект и задачи из share (генерация чанков — тоже долго) ..."
docker run --rm \
  --network "$NETWORK" \
  --env-file .env \
  -e CVAT_URL=http://cvat-server:8080 \
  -v "$IMAGES_DIR":/home/django/share:ro \
  -v "$SCRIPT_DIR/bootstrap.py":/bootstrap.py:ro \
  python:3.11-slim \
  sh -c "pip install --quiet --no-cache-dir --disable-pip-version-check cvat-sdk==${SDK_VERSION} && python /bootstrap.py"

echo
echo "================================================================"
echo " CVAT готов:  http://${CVAT_HOST}:${CVAT_PORT}"
echo "   admin / admin  — создание проектов/задач, панель /admin"
echo "   user  / user   — разметка (роль worker в орг «Клуб мазохистов»)"
echo " Разметка: войти как user -> выбрать организацию вверху ->"
echo "           Tasks -> своя задача -> Job."
echo "================================================================"
