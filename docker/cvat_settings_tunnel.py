# Настройки Django для CVAT, раздаваемого наружу через туннель (Tuna, ngrok и подобные).
#
# Подключаются монтированием этого файла в /home/django/cvat/settings/tunnel.py и переменной
# DJANGO_SETTINGS_MODULE=cvat.settings.tunnel (см. docker-compose.override.yml). Апстрим
# такого механизма не предусматривает, но модуль настроек Django выбирается переменной
# окружения, и этого достаточно.
#
# ЗАЧЕМ. Туннель терминирует TLS у себя и отдаёт наружу https, а до traefik доносит обычный
# http. Django при этом строит собственный origin как http://<хост>, браузер присылает
# Origin: https://<хост>, и проверка CSRF отвергает запрос:
#
#     CSRF Failed: Origin checking failed - https://xxx.ru.tuna.am
#     does not match any trusted origins
#
# В интерфейсе это видно как «Could not send logs to the server»: телеметрия шлётся POST'ом
# и упирается в ту же проверку.
#
# Лечится двумя независимыми способами, и здесь сделаны оба:
#
# 1. traefik доверяет заголовкам X-Forwarded-* от туннеля (в override). Тогда Django видит
#    https и сам считает origin своим — но только если туннель эти заголовки шлёт.
# 2. Список доверенных origin'ов ниже. Работает независимо от заголовков.
#
# Полностью выключать проверку CSRF не стоит и, главное, недостаточно: сообщение выше выдаёт
# не middleware, а SessionAuthentication из DRF, и она проверяет origin своими силами. Список
# доверенных origin'ов закрывает оба места разом.

from .production import *  # noqa: F401,F403  # pylint: disable=wildcard-import
import os  # noqa: E402

# Список через запятую в CVAT_CSRF_TRUSTED_ORIGINS. Django требует схему и разрешает звёздочку
# только вместо самой левой части имени: `https://*.ru.tuna.am` годится, `https://*` — нет,
# «любой хост» так задать нельзя. Поэтому в умолчании перечислены домены туннелей, а свой
# добавляется переменной окружения.
_DEFAULT_ORIGINS = (
    "http://localhost:8081,"
    "http://127.0.0.1:8081,"
    "https://*.ru.tuna.am,"
    "https://*.tuna.am,"
    "https://*.ngrok-free.app,"
    "https://*.ngrok.io"
)

CSRF_TRUSTED_ORIGINS = [
    origin.strip() for origin in os.environ.get("CVAT_CSRF_TRUSTED_ORIGINS", _DEFAULT_ORIGINS).split(",") if origin.strip()
]
