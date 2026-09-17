"""Утилиты загрузки: клон апстрим-репозиториев и скачивание весов моделей.

Код моделей живёт в апстрим-репозиториях (клонируются в ``third_party/`` корня
проекта и добавляются в ``sys.path``). Веса — в ``dewarp_models/``.
"""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Корень проекта: .../ocr_utils/dewarp/engines/download.py → parents[3]
ROOT = Path(__file__).resolve().parents[3]
THIRD_PARTY = ROOT / "third_party"
MODELS_DIR = ROOT / "dewarp_models"


def ensure_repo(url: str, name: str) -> Path:
    """Клонирует репозиторий в ``third_party/<name>`` (если ещё нет) и возвращает путь."""
    dest = THIRD_PARTY / name
    if not dest.exists():
        THIRD_PARTY.mkdir(parents=True, exist_ok=True)
        logger.info("git clone %s → %s", url, dest)
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    return dest


def add_to_path(path: Path) -> None:
    """Добавляет путь в начало ``sys.path`` (для импорта модулей из клона репо)."""
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def forget_modules(names: list[str]) -> None:
    """Убирает из кэша ``sys.modules`` модули с указанными именами.

    У разных репозиториев бывают одноимённые модули (``seg.py``, ``model.py``).
    После загрузки одного движка чистим кэш, чтобы следующий импортировал СВОИ модули
    из своего пути, а не подхватил чужие (важно для ``--method all``).
    """
    for n in names:
        sys.modules.pop(n, None)


def gdrive_folder(folder_id: str, out_dir: Path) -> None:
    """Скачивает папку Google Drive в ``out_dir`` через gdown."""
    import gdown

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("gdown: скачиваю папку %s → %s", folder_id, out_dir)
    gdown.download_folder(id=folder_id, output=str(out_dir), quiet=False, use_cookies=False)
