"""Подключение к локальному CVAT: чтение ``docker/.env`` и создание клиента.

Креды и адрес живут в ``docker/.env`` рядом с compose-файлом — там же, откуда их берут
``up.sh``. Дублировать их в опциях CLI значило бы завести второй
источник правды, который рано или поздно разъедется с первым, поэтому опции CLI только
ПЕРЕКРЫВАЮТ прочитанное.

Файл разбирается вручную, а не ``dotenv``: значения там без кавычек и содержат пробелы и
кириллицу (``ORG_NAME=Клуб мазохистов``) — ровно по той же причине ``up.sh`` использует
grep + cut вместо ``source``.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# .env лежит рядом с docker-compose.override.yml: <корень репозитория>/docker/.env
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[3] / "docker" / ".env"


def read_env_file(path: Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    """Пары ключ-значение из ``.env``; отсутствующий файл — пустой словарь, не ошибка."""
    values: dict[str, str] = {}
    if not path.is_file():
        logger.debug("Нет файла %s, беру настройки CVAT только из опций и окружения", path)
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


class CvatSettings:
    """Адрес, учётка и организация CVAT.

    Приоритет: явная опция CLI -> переменная окружения -> ``docker/.env`` -> дефолт.
    """

    def __init__(
        self,
        url: str | None = None,
        user: str | None = None,
        password: str | None = None,
        org: str | None = None,
        env_path: Path = DEFAULT_ENV_PATH,
    ) -> None:
        env = read_env_file(env_path)

        def pick(option: str | None, env_key: str, default: str) -> str:
            return option or os.environ.get(env_key) or env.get(env_key) or default

        host = env.get("CVAT_HOST", "localhost")
        port = env.get("CVAT_PORT", "8080")
        # Схема указывается ЯВНО: без неё cvat-sdk идёт по https и спотыкается об SSL,
        # потому что сервер слушает обычный http.
        self.url = url or os.environ.get("CVAT_URL") or f"http://{host}:{port}"
        self.user = pick(user, "ADMIN_USER", "admin")
        self.password = pick(password, "ADMIN_PASS", "admin")
        self.org = pick(org, "ORG_SLUG", "")

    def __repr__(self) -> str:  # пароль в лог не попадает
        return f"CvatSettings(url={self.url!r}, user={self.user!r}, org={self.org!r})"


def share_root_from_env(env_path: Path = DEFAULT_ENV_PATH) -> Path | None:
    """Каталог, смонтированный в CVAT как ``/home/django/share`` (``IMAGES_DIR`` из .env)."""
    value = read_env_file(env_path).get("IMAGES_DIR")
    return Path(value) if value else None


def share_prefix(share_root: Path, env_path: Path = DEFAULT_ENV_PATH) -> Path:
    """Префикс, который отделяет ``share_root`` от корня share глазами cvat_server.

    Сервер ищет файлы по путям ВНУТРИ ``/home/django/share``, куда смонтирован
    ``IMAGES_DIR``. Значит, в ``server_files`` надо отдавать путь относительно
    ``IMAGES_DIR``, а не относительно ``--share-root``: при ``IMAGES_DIR=/mnt/dump3/share``
    и ``--share-root=/mnt/dump3/share/паки`` сервер ждёт ``паки/пак-1/1974/01/a.jpg``.

    Возвращается пустой путь, когда ``share_root`` совпадает с ``IMAGES_DIR`` или когда
    ``.env`` прочитать не удалось (запуск с другой машины — не повод отказываться работать).
    """
    images_dir = share_root_from_env(env_path)
    if images_dir is None:
        return Path()
    share_root, images_dir = Path(share_root).resolve(), images_dir.resolve()
    if share_root == images_dir:
        return Path()
    try:
        return share_root.relative_to(images_dir)
    except ValueError:  # снаружи IMAGES_DIR — об этом ругается check_share_root
        return Path()


def check_share_root(share_root: Path, env_path: Path = DEFAULT_ENV_PATH) -> str | None:
    """Текст предупреждения, если ``share_root`` лежит вне того, что видит cvat_server.

    Задачи заводятся из share (``ResourceType.SHARE``), файлы по сети не передаются:
    сервер ищет их по путям внутри ``/home/django/share``. Картинки, положенные мимо этого
    каталога, он просто не найдёт — и упадёт уже на создании задачи, где связь с причиной
    неочевидна. Поэтому проверяем заранее и говорим прямо.

    Возвращается ``None``, когда всё в порядке или когда ``.env`` прочитать не удалось
    (запуск с другой машины — не повод отказываться работать).
    """
    images_dir = share_root_from_env(env_path)
    if images_dir is None:
        return None
    share_root, images_dir = Path(share_root).resolve(), images_dir.resolve()
    if share_root == images_dir or images_dir in share_root.parents:
        return None
    return (
        f"--share-root {share_root} лежит вне IMAGES_DIR={images_dir} из {env_path}. "
        "CVAT берёт картинки из смонтированного share и по сети их не получает, поэтому "
        "задачи заведутся пустыми. Поправьте IMAGES_DIR в docker/.env и перезапустите "
        "docker/up.sh либо укажите --share-root внутри IMAGES_DIR."
    )


def make_cvat_client(settings: CvatSettings):
    """Контекстный менеджер клиента CVAT с уже выставленной организацией."""
    from cvat_sdk import make_client

    logger.info("Подключаюсь к CVAT: %s", settings)
    client = make_client(host=settings.url, credentials=(settings.user, settings.password))
    if settings.org:
        client.organization_slug = settings.org
    return client
