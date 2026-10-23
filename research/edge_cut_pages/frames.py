"""Кадры пака: обход ``<журнал>/<год>/<выпуск>/*.jpg`` без скрытых папок, ключи и относительные пути."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Расширения кадров.
FRAME_SUFFIXES = (".jpg", ".jpeg")


@dataclass(frozen=True)
class Frame:
    """Кадр пака: путь, журнал, год, выпуск (имя папки как в исходнике), имя кадра без расширения."""

    path: Path
    journal: str
    year: str
    issue: str
    name: str

    @property
    def rel(self) -> Path:
        """Путь относительно корня пака: ``<журнал>/<год>/<выпуск>/<файл>``."""
        return Path(self.journal, self.year, self.issue, self.path.name)


def is_hidden(path: Path, root: Path) -> bool:
    """Лежит ли путь в скрытой папке (имя с точки) внутри ``root``.

    Args:
        path: Путь файла.
        root: Корень пака.

    Returns:
        True — в скрытой папке (``.covers``, ``.дубликаты к удалению`` и т. п.).
    """
    return any(part.startswith(".") for part in path.relative_to(root).parts[:-1])


def iter_frames(root: Path) -> list[Frame]:
    """Все кадры пака по порядку: ``<журнал>/<год>/<выпуск>/<кадр>``, скрытые папки и файлы пропускаются.

    Args:
        root: Корень пака.

    Returns:
        Список кадров, отсортированный по относительному пути.
    """
    frames = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in FRAME_SUFFIXES or path.name.startswith(".") or is_hidden(path, root):
            continue
        parts = path.relative_to(root).parts
        if len(parts) != 4:
            # Кадры вне схемы журнал/год/выпуск — не наши.
            continue
        frames.append(Frame(path=path, journal=parts[0], year=parts[1], issue=parts[2], name=path.stem))
    return frames
