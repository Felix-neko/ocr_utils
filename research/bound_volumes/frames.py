"""Кадры стенда подшивок МТО 1990: обход папок выпусков, ключи кадров, рабочие копии и превью на SSD, разметка классов."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

# Корень выхода стенда на SSD (не Я.Диск: там исходники только читаются).
OUT_ROOT = Path("/mnt/hotstore/scan_processing/bound_volumes")
# Превью для листов миниатюр и отсмотра: <OUT_ROOT>/previews/<выпуск>/<имя>.jpg.
PREVIEW_DIR = OUT_ROOT / "previews"
# Рабочие копии кадров набора в исходном разрешении: <OUT_ROOT>/frames/<выпуск>/<имя>.jpg.
FRAMES_DIR = OUT_ROOT / "frames"
# Набор по классам дефектов: <OUT_ROOT>/set/<класс>/<выпуск>__<имя>.jpg (ссылки на рабочие копии).
SET_DIR = OUT_ROOT / "set"
# Разметка стенда в git.
SETS_DIR = Path(__file__).resolve().parent / "sets"
# Файл с классами кадров набора (ключ, метки через «;», заметка).
CATEGORIES_CSV = SETS_DIR / "categories.csv"
# Длинная сторона превью, px.
PREVIEW_LONG_SIDE = 1600

# Папка года МТО: подпапки — выпуски («04 оплачено есть фотографии по страницам» и т.п.).
SOURCE_ROOT = Path("/mnt/dump3/yandex_disk_linux_baby_zergling/Пробное/Отобранное/МТО/1990")


class Category(str, Enum):
    """Класс трудного кадра в наборе стенда."""

    # Соседняя страница поднята рукой, чтобы снять тугую подшивку.
    LIFTED = "lifted"
    # Из-под верхней страницы торчат обрез стопки или край крышки подшивки.
    STACK_EDGE = "stack_edge"
    # Сильный изгиб страницы (у корешка или у края).
    STRONG_CURL = "strong_curl"
    # Часть книги вне поля зрения кадра.
    OUT_OF_FRAME = "out_of_frame"


@dataclass(frozen=True)
class Frame:
    """Кадр: короткое имя выпуска и путь к исходному JPEG."""

    issue: str
    path: Path

    @property
    def key(self) -> str:
        """Ключ ``<выпуск>/<имя без расширения>`` — имя подпапки и файлов прогона."""
        return f"{self.issue}/{self.path.stem}"

    @property
    def flat_key(self) -> str:
        """Ключ без косой черты (``<выпуск>__<имя>``) — для плоских папок оверлеев."""
        return self.key.replace("/", "__")

    @property
    def preview_path(self) -> Path:
        """Путь превью на SSD."""
        return PREVIEW_DIR / self.issue / (self.path.stem + ".jpg")

    @property
    def work_path(self) -> Path:
        """Путь рабочей копии кадра в исходном разрешении на SSD."""
        return FRAMES_DIR / self.issue / (self.path.stem + ".jpg")


def issue_name(folder: Path) -> str:
    """Короткое имя выпуска по папке: первые два символа (номер «04»).

    Args:
        folder: Папка выпуска вида «04 оплачено есть фотографии по страницам».

    Returns:
        Номер выпуска строкой, например ``"04"``.
    """
    return folder.name.split()[0]


def all_frames(root: Path = SOURCE_ROOT) -> list[Frame]:
    """Все JPEG верхнего уровня папок выпусков года.

    Args:
        root: Папка года.

    Returns:
        Кадры по выпускам и именам.
    """
    frames = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg"):
                frames.append(Frame(issue_name(folder), path))
    return frames


def work_frames() -> list[Frame]:
    """Кадры по рабочим копиям на SSD (``FRAMES_DIR/<выпуск>/<имя>.jpg``) — когда исходников на Я.Диске нет.

    Returns:
        Кадры; ``path`` указывает на рабочую копию.
    """
    return [Frame(d.name, p) for d in sorted(FRAMES_DIR.iterdir()) if d.is_dir() for p in sorted(d.glob("*.jpg"))]


def frame_by_key(key: str, frames: list[Frame] | None = None) -> Frame:
    """Кадр по ключу ``<выпуск>/<имя>``.

    Args:
        key: Ключ кадра.
        frames: Список, где искать (по умолчанию — все кадры года).

    Returns:
        Найденный кадр.

    Raises:
        KeyError: Если ключа нет.
    """
    if frames is None:
        frames = all_frames() if SOURCE_ROOT.is_dir() else work_frames()
    for frame in frames:
        if frame.key == key:
            return frame
    raise KeyError(key)


def read_bgr(path: Path) -> np.ndarray:
    """Читает JPEG в BGR (по байтам: путь с кириллицей и пробелами).

    Args:
        path: Путь к картинке.

    Returns:
        Массив ``(H, W, 3)`` uint8.
    """
    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"не читается: {path}")
    return img


def make_preview(frame: Frame, long_side: int = PREVIEW_LONG_SIDE) -> Path:
    """Пишет превью кадра на SSD, если его ещё нет.

    Args:
        frame: Кадр.
        long_side: Длинная сторона превью, px.

    Returns:
        Путь превью.
    """
    out = frame.preview_path
    if out.exists():
        return out
    img = read_bgr(frame.path)
    scale = long_side / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


@dataclass(frozen=True)
class Labeled:
    """Кадр набора с метками классов и заметкой отсмотра."""

    frame: Frame
    categories: frozenset[Category]
    note: str = ""


def read_categories(path: Path = CATEGORIES_CSV) -> list[Labeled]:
    """Читает разметку классов набора.

    Args:
        path: CSV с колонками ``key, categories, note`` (метки через «;»).

    Returns:
        Кадры набора в порядке файла.
    """
    # Исходники на Я.Диске могут пропасть (синхронизация, перенос) — тогда кадры набора берутся с рабочих копий на SSD.
    frames = {f.key: f for f in (all_frames() if SOURCE_ROOT.is_dir() else work_frames())}
    out = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cats = frozenset(Category(c.strip()) for c in row["categories"].split(";") if c.strip())
            out.append(Labeled(frames[row["key"]], cats, row.get("note", "")))
    return out
