"""Пакетный прогон восстановления: словарь, библиотека, кадры, отчёт."""

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.gutter_loss_detection.geometry import read_work_gray
from ocr_utils.gutter_loss_restoration.glyphs import ink_mask
from ocr_utils.gutter_loss_restoration.library import ENOUGH, choose, harvest, load, save
from ocr_utils.gutter_loss_restoration.ocrcache import ensure
from ocr_utils.gutter_loss_restoration.restore import DONOR_FROM, DONOR_TO, INNER, OUTER, _rows, restore_spread

# Что должно быть в библиотеке, чтобы перестать её пополнять.
NEEDED = set("абвгдежзийклмнопрстуфхцчшщыьэюя-,.")


def build_shared(files, work: Path) -> dict:
    """Собирает (или читает) библиотеку литер выпуска.

    Args:
        files: Кадры папки.
        work: Рабочая папка с кэшем.

    Returns:
        Библиотека литер.
    """
    target = work / "литеры.npz"
    if target.exists():
        return load(target)
    samples: dict[str, list] = {}
    for path in files:
        if all(len(samples.get(c, ())) >= ENOUGH for c in NEEDED):
            break
        image = cv2.imread(str(path))
        if image is None:
            continue
        fold, halves = ensure(work / "ocr", path)
        mask = ink_mask(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32))
        height, width = mask.shape
        for side in ("L", "R"):
            x0, x1 = (
                (int(width * OUTER), int(fold - width * INNER))
                if side == "L"
                else (int(fold + width * INNER), int(width * (1 - OUTER)))
            )
            rows, pitch = _rows(mask, x0, x1, halves.get(side, []), 60.0)
            span = x1 - x0
            zone = (
                (int(x0 + span * DONOR_FROM), int(x0 + span * DONOR_TO))
                if side == "L"
                else (int(x1 - span * DONOR_TO), int(x1 - span * DONOR_FROM))
            )
            harvest(image, mask, [(r.top, r.bottom, r.text) for r in rows], pitch, zone[0], zone[1], samples, x0, x1)
    library = choose(samples)
    save(target, library)
    return library


def restore_many(paths, work: Path, lexicon: set[str], shared: dict, out_dir: Path, copy_source: bool = True):
    """Восстанавливает кадры и складывает результат рядом с исходниками.

    Args:
        paths: Кадры к восстановлению.
        work: Рабочая папка с кэшем распознавания.
        lexicon: Словарь выпуска.
        shared: Библиотека литер.
        out_dir: Куда складывать.
        copy_source: Класть ли рядом исходник.

    Returns:
        Пара (строки отчёта, сводка).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report_rows, stats = [], {"кадров": 0, "строк": 0, "отказы": []}
    for path in paths:
        fold, halves = ensure(work / "ocr", path)
        canvas, reports, error = restore_spread(path, fold, halves, lexicon, shared)
        stem = path.name.split(".")[0]
        for line in reports:
            report_rows.append(
                [
                    path.name,
                    line.side,
                    line.index,
                    line.visible,
                    line.word,
                    line.added,
                    "да" if line.done else "нет",
                    line.reason,
                ]
            )
            if not line.done:
                stats["отказы"].append(line.reason)
        if canvas is None:
            continue
        cv2.imwrite(str(out_dir / f"{stem}.стало.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 97])
        if copy_source:
            shutil.copy2(path, out_dir / f"{stem}.было{path.suffix}")
        stats["кадров"] += 1
        stats["строк"] += sum(1 for line in reports if line.done)
    with (out_dir / "отчёт.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["кадр", "полоса", "строка", "видно", "набрано", "дописано", "починено", "причина"])
        writer.writerows(report_rows)
    return report_rows, stats
