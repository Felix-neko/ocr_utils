"""Прогоны scan_cropping v2 — вход стенда: где лежат итоги, кэш и оверлеи прогона, плотность кадра, кадры прогона и строки наборов ``sets/*.txt``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RunSpec:
    """Прогон v2, из которого стенд берёт кадры.

    Поля:
        name: Короткое имя прогона в наборах (``ph4``, ``pack6``, ``hp91``).
        root: Папка прогона (в ней ``output/``, ``cache/``, ``debug/``).
        sub: Подпапка внутри ``output/``, ``cache/`` и ``debug/`` (``1975`` у пака-4 ПХ) или пустая строка.
        px_per_mm: Плотность кадра, px/мм (``--px-per-mm`` прогона).
    """

    name: str
    root: Path
    sub: str
    px_per_mm: float

    @property
    def cache_dir(self) -> Path:
        """Корень кэша кадров прогона."""
        return self.root / "cache" / self.sub

    @property
    def output_dir(self) -> Path:
        """Корень итоговых TIFF прогона."""
        return self.root / "output" / self.sub

    @property
    def debug_dir(self) -> Path:
        """Корень отладочных оверлеев прогона."""
        return self.root / "debug" / self.sub


def parse_run(text: str) -> RunSpec:
    """Прогон из строки CLI ``имя=папка,подпапка,px_per_mm`` (подпапка может быть пустой: ``pack6=/a/b,,13.8``).

    Args:
        text: Строка опции ``--run``.

    Returns:
        Описание прогона.
    """
    name, rest = text.split("=", 1)
    root, sub, px = rest.rsplit(",", 2)
    return RunSpec(name.strip(), Path(root), sub.strip(), float(px))


def key_of(rel: Path) -> str:
    """Ключ кадра в кэше v2: относительный путь без расширения, ``/`` → ``__`` (как ``FrameJob.key``)."""
    return "__".join(rel.with_suffix("").parts)


def rel_of(key: str) -> Path:
    """Путь кадра относительно корня прогона (без расширения) по ключу кэша — обратное ``key_of``."""
    return Path(*key.split("__"))


def run_keys(run: RunSpec) -> list[str]:
    """Ключи кадров прогона, у которых в кэше есть страницы (``pages.json``) и разметка surya (``layout.json``).

    Args:
        run: Прогон.

    Returns:
        Ключи по алфавиту.
    """
    return sorted(
        p.name for p in run.cache_dir.iterdir() if (p / "pages.json").exists() and (p / "layout.json").exists()
    )


@dataclass(frozen=True)
class SetRow:
    """Строка набора: прогон, ключ кадра и пометка.

    Поля:
        run: Имя прогона.
        key: Ключ кадра в кэше прогона.
        note: Пометка после ``#`` (что на кадре), может быть пустой.
    """

    run: str
    key: str
    note: str


def read_set(path: Path) -> list[SetRow]:
    """Кадры набора: строка ``<прогон> <ключ кадра>  # пометка``; пустые строки и строки с ``#`` в начале — пропуск.

    Args:
        path: Файл набора.

    Returns:
        Строки набора по порядку.
    """
    rows = []
    for line in path.read_text().splitlines():
        body, _, note = line.partition("#")
        body = body.strip()
        if not body:
            continue
        run, key = body.split(maxsplit=1)
        rows.append(SetRow(run, key.strip(), note.strip()))
    return rows


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Картинка (TIFF, PNG, JPEG) с путём в юникоде.

    Args:
        path: Файл.
        flags: Флаги ``cv2.imdecode`` (по умолчанию BGR uint8).

    Returns:
        Массив картинки.
    """
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), flags)
    if img is None:
        raise ValueError(f"не читается: {path}")
    return img


def write_image(path: Path, img: np.ndarray, quality: int = 90) -> None:
    """Запись картинки с путём в юникоде; JPEG — с качеством ``quality``, папка создаётся.

    Args:
        path: Файл (расширение задаёт формат).
        img: Картинка.
        quality: Качество JPEG.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(path.suffix, img, params)
    if not ok:
        raise ValueError(f"не пишется: {path}")
    buf.tofile(str(path))
