"""Кэш распознавания полос: один JSON на кадр.

ЗАЧЕМ КЭШ. Распознавание разворота занимает около пятнадцати секунд на GPU, а нужен
его результат дважды: сперва чтобы собрать по всему выпуску словарь (без него не
вычислить, какое слово разорвано переносом), потом чтобы восстановить конкретную полосу.
Гонять surya второй раз по тем же кадрам незачем.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ocr_utils.gutter_loss_detection.geometry import build_masks, fit_fold, read_work_gray
from ocr_utils.gutter_loss_restoration.pageocr import Line, Word, read_halves


def fold_column(path: Path) -> int:
    """Столбец сгиба кадра в координатах исходного файла.

    Args:
        path: Путь к кадру.

    Returns:
        Столбец сгиба.
    """
    gray, scale = read_work_gray(path)
    ink, _, _ = build_masks(gray)
    line, _ = fit_fold(gray, ink)
    return int(round(float(np.polyval(line, gray.shape[0] / 2)) * scale))


def cache_path(cache_dir: Path, path: Path) -> Path:
    """Куда лечь кэшу этого кадра."""
    return cache_dir / f"{path.stem}.json"


def load(cache_dir: Path, path: Path) -> dict[str, list[Line]] | None:
    """Читает кэш распознавания, если он есть.

    Args:
        cache_dir: Папка кэша.
        path: Путь к кадру.

    Returns:
        Словарь полос либо None.
    """
    target = cache_path(cache_dir, path)
    if not target.exists():
        return None
    raw = json.loads(target.read_text(encoding="utf-8"))
    out = {}
    for side, lines in raw["halves"].items():
        out[side] = [
            Line(
                text=line["text"],
                top=line["top"],
                bottom=line["bottom"],
                x0=line["x0"],
                x1=line["x1"],
                words=tuple(Word(**w) for w in line["words"]),
            )
            for line in lines
        ]
    return out


def store(cache_dir: Path, path: Path, fold: int, halves: dict[str, list[Line]]) -> None:
    """Пишет кэш распознавания.

    Args:
        cache_dir: Папка кэша.
        path: Путь к кадру.
        fold: Столбец сгиба.
        halves: Распознанные полосы.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "file": path.name,
        "fold": fold,
        "halves": {
            side: [{**asdict(line), "words": [asdict(w) for w in line.words]} for line in lines]
            for side, lines in halves.items()
        },
    }
    cache_path(cache_dir, path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, list[Line]]:
    """Читает один файл кэша по прямому пути (для сборки словаря).

    Args:
        path: Путь к JSON кэша.

    Returns:
        Словарь полос.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        side: [
            Line(
                text=line["text"],
                top=line["top"],
                bottom=line["bottom"],
                x0=line["x0"],
                x1=line["x1"],
                words=tuple(Word(**w) for w in line["words"]),
            )
            for line in lines
        ]
        for side, lines in raw["halves"].items()
    }


def ensure(cache_dir: Path, path: Path) -> tuple[int, dict[str, list[Line]]]:
    """Возвращает распознанные полосы кадра, считая их при необходимости.

    Args:
        cache_dir: Папка кэша.
        path: Путь к кадру.

    Returns:
        Пара (столбец сгиба, полосы).
    """
    fold = fold_column(path)
    cached = load(cache_dir, path)
    if cached is not None:
        return fold, cached
    halves = read_halves(path, fold)
    store(cache_dir, path, fold, halves)
    return fold, halves
