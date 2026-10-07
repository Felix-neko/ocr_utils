"""Кадры стенда: выборка RAF из папок ЭГ, извлечение встроенного JPEG-превью с учётом EXIF-поворота, кэш превью на SSD."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image, ImageOps

# Корень выхода стенда на SSD (не Я.Диск).
OUT_ROOT = Path("/mnt/hotstore/scan_processing/newspaper_crop")
# Кэш превью: <OUT_ROOT>/previews/<группа>/<имя>.jpg.
PREVIEW_DIR = OUT_ROOT / "previews"
# Разметка стенда в git.
SETS_DIR = Path(__file__).resolve().parent / "sets"
# Файл выборки (60 кадров для подбора методов) и всех кадров двух папок.
SAMPLE_CSV = SETS_DIR / "sample.csv"
ALL_CSV = SETS_DIR / "all.csv"
# Обложки и форзацы двух папок (отсмотрены по миниатюрам начала и конца каждой серии): в работу не берутся.
NOT_PAGES = {
    "1970_07-12/1_0010": "крышка подшивки",
    "1969_04-06/DSCF0001": "крышка подшивки",
    "1969_04-06/DSCF0002": "форзац",
    "1969_04-06/DSCF0342": "форзац",
}
# Набор кадров, с которым работают команды (меняется опцией ``--set`` группы команд).
ACTIVE_SET = SAMPLE_CSV

# Исходные папки ЭГ и короткие имена групп (группа — подпапка кэша и префикс ключа).
SOURCE_ROOT = Path("/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/ЭГ/в работе")
SOURCE_FOLDERS = {
    "1970_07-12": SOURCE_ROOT / "1970" / "07-12 готово ХП",
    "1969_04-06": SOURCE_ROOT / "1969" / "04-06 готово ХП",
}


@dataclass(frozen=True)
class Frame:
    """Кадр выборки: группа, путь к RAF, флаг «не брать» и заметка отсмотра."""

    group: str
    path: Path
    skip: bool = False
    note: str = ""

    @property
    def key(self) -> str:
        """Ключ кадра ``<группа>/<имя без расширения>`` — имя подпапки и файлов прогона."""
        return f"{self.group}/{self.path.stem}"

    @property
    def preview_path(self) -> Path:
        """Путь кэшированного превью JPEG на SSD."""
        return PREVIEW_DIR / self.group / (self.path.stem + ".jpg")


def list_raf(folder: Path) -> list[Path]:
    """Список RAF папки без скрытых подпапок, по имени.

    Args:
        folder: Папка выпуска.

    Returns:
        Отсортированный список путей RAF (только верхний уровень: подпапки — служебные).
    """
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".raf")


def even_pick(paths: list[Path], count: int) -> list[Path]:
    """Равномерно выбирает ``count`` элементов из списка (первый и последний входят).

    Args:
        paths: Список путей.
        count: Сколько взять.

    Returns:
        Подсписок в исходном порядке; весь список, если он короче ``count``.
    """
    if len(paths) <= count:
        return list(paths)
    idx = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[i] for i in sorted(set(idx.tolist()))]


def read_sample(path: Path | None = None) -> list[Frame]:
    """Читает набор кадров из CSV (колонки ``group, path, skip, note``).

    Args:
        path: Путь CSV; None — активный набор (:data:`ACTIVE_SET`).

    Returns:
        Список кадров в порядке файла.
    """
    with (path or ACTIVE_SET).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [Frame(r["group"], Path(r["path"]), r.get("skip", "0") == "1", r.get("note", "")) for r in rows]


def write_sample(frames: list[Frame], path: Path = SAMPLE_CSV) -> None:
    """Пишет выборку в CSV (колонки ``group, path, skip, note``).

    Args:
        frames: Кадры.
        path: Путь CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "path", "skip", "note"])
        for f in frames:
            writer.writerow([f.group, str(f.path), "1" if f.skip else "0", f.note])


def extract_preview(raf: Path) -> Image.Image:
    """Достаёт встроенное JPEG-превью RAF и поворачивает его по EXIF.

    Args:
        raf: Путь к RAF.

    Returns:
        RGB-картинка Pillow (у X100VI — 2944×4416 после поворота).

    Raises:
        ValueError: в RAF нет JPEG-превью.
    """
    with rawpy.imread(str(raf)) as raw:
        thumb = raw.extract_thumb()
        if thumb.format != rawpy.ThumbFormat.JPEG:
            raise ValueError(f"в {raf} превью не JPEG")
        img = Image.open(io.BytesIO(bytes(thumb.data)))
        img.load()
    return ImageOps.exif_transpose(img).convert("RGB")


def ensure_preview(frame: Frame) -> Path:
    """Кэширует превью кадра на SSD (если его ещё нет) и возвращает путь (воркер пула).

    Args:
        frame: Кадр.

    Returns:
        Путь JPEG превью.
    """
    out = frame.preview_path
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        extract_preview(frame.path).save(out, quality=95)
    return out


def read_preview_rgb(frame: Frame) -> np.ndarray:
    """Читает кэшированное превью как массив RGB uint8 (H, W, 3).

    Args:
        frame: Кадр (превью должно быть уже в кэше).

    Returns:
        Массив RGB.
    """
    with Image.open(frame.preview_path) as img:
        return np.asarray(img.convert("RGB"))
