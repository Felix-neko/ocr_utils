"""Обход пака фото-сканов: кадры выпуска по порядку, ключ кадра, чтение в полутон."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Пак-4 «Планового хозяйства»: год / выпуск («03 оплачено») / IMG_0001.jpg.
PACK4_ROOT = Path("/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/Плановое хозяйство/в работе/пак-4 (450 dpi)")

# Расширения кадров, которые берём в обход.
FRAME_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff"}

# Номер кадра в имени файла: IMG_0017_дефект.jpg → 17.
_NUMBER = re.compile(r"(\d+)")


@dataclass(frozen=True)
class Frame:
    """Один кадр пака: путь и положение в выпуске.

    Attributes:
        path: Абсолютный путь к файлу кадра.
        issue: Ключ выпуска — путь папки выпуска относительно корня пака (``1985/02 оплачено``).
        index: Номер кадра из имени файла (``IMG_0017`` → 17), порядок съёмки.
    """

    path: Path
    issue: str
    index: int

    @property
    def key(self) -> str:
        """Ключ кадра: ``<выпуск>/<имя файла>`` — уникален в паке и читается глазами."""
        return f"{self.issue}/{self.path.name}"


def frame_index(name: str) -> int:
    """Номер кадра из имени файла.

    Args:
        name: Имя файла (``IMG_0017_дефект.jpg``).

    Returns:
        Первое число в имени; -1, если чисел нет.
    """
    found = _NUMBER.search(name)
    return int(found.group(1)) if found else -1


def iter_issue(issue_dir: Path, root: Path) -> list[Frame]:
    """Кадры одного выпуска в порядке съёмки.

    Args:
        issue_dir: Папка выпуска.
        root: Корень пака — от него считается ключ выпуска.

    Returns:
        Кадры, отсортированные по номеру из имени файла.
    """
    issue = str(issue_dir.relative_to(root))
    frames = [
        Frame(path=p, issue=issue, index=frame_index(p.name))
        for p in issue_dir.iterdir()
        if p.is_file() and p.suffix.lower() in FRAME_SUFFIXES
    ]
    return sorted(frames, key=lambda f: (f.index, f.path.name))


def issue_dirs(root: Path) -> list[Path]:
    """Все папки выпусков пака: каталоги, в которых прямо лежат кадры.

    Args:
        root: Корень пака (или папка года, или сама папка выпуска).

    Returns:
        Отсортированный список папок, содержащих хотя бы один кадр.
    """
    found = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES:
            found.add(path.parent)
    return sorted(found)


def iter_pack(root: Path, only: tuple[str, ...] = ()) -> list[Frame]:
    """Все кадры пака по выпускам.

    Args:
        root: Корень пака.
        only: Если не пусто — брать только выпуски, ключ которых начинается с одной из этих строк
            (``1985/02``, ``1979``).

    Returns:
        Кадры всех выпусков подряд, внутри выпуска — в порядке съёмки.
    """
    frames = []
    for issue_dir in issue_dirs(root):
        issue = str(issue_dir.relative_to(root)) if issue_dir != root else issue_dir.name
        if only and not any(issue.startswith(prefix) for prefix in only):
            continue
        frames.extend(iter_issue(issue_dir, root if issue_dir != root else root.parent))
    return frames


def read_rgb(path: Path, reduce: int = 1) -> np.ndarray:
    """Кадр в RGB uint8.

    Args:
        path: Файл кадра.
        reduce: Во сколько раз уменьшить по каждой оси. Для JPEG грубое уменьшение делает сам
            декодер (``draft``, степени двойки) — кратно быстрее полного декодирования; остаток
            добирается билинейным ресайзом.

    Returns:
        Массив H×W×3 uint8.
    """
    with Image.open(path) as image:
        target = (image.width // reduce, image.height // reduce)
        if reduce > 1 and image.format == "JPEG":
            image.draft("RGB", target)
        image = image.convert("RGB")
        # draft отдаёт размер не меньше запрошенного — дожимаем до точного.
        if reduce > 1 and image.width > target[0]:
            image = image.resize(target, Image.BILINEAR)
        return np.asarray(image)
