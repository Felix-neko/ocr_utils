"""Разрезка фото-разворотов на страницы по прямой линии сгиба.

Сырые кадры повреждённых подшивок — развороты: две страницы, между ними тёмная щель
сгиба, снятая под небольшим наклоном. Модели нужна одна страница, поэтому разворот
режется по прямой ``x = k·y + b``, подогнанной к сгибу; линия берётся из
``ocr_utils.gutter_loss_detection.geometry.fit_fold`` — тот же алгоритм, что ранжирует
уход текста под корешок. Внешние поля (фон, пальцы) не трогаем: что резать сверх сгиба —
решает человек по контрольным картинкам.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ocr_utils.gutter_loss_detection.geometry import build_masks, fit_fold, read_work_gray

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
# Контрольная картинка с линией: ширина уменьшенного разворота.
PREVIEW_WIDTH = 1600


@dataclass(frozen=True)
class Fold:
    """Прямая сгиба в координатах исходного кадра: ``x = slope * y + intercept``."""

    slope: float
    intercept: float

    def x_at(self, y: float) -> float:
        return self.slope * y + self.intercept


def find_fold(path: Path, shift: float = 0.0) -> Fold:
    """Подогнать прямую сгиба по кадру; ``shift`` — ручной сдвиг линии в пикселях исходника."""
    gray, scale = read_work_gray(path)
    ink, _, _ = build_masks(gray)
    line, _ = fit_fold(gray, ink)
    # Перевод из рабочего масштаба: x_full = scale * (k * (y_full / scale) + b) = k * y_full + scale * b.
    return Fold(slope=float(line[0]), intercept=float(line[1]) * scale + shift)


def split_spread(image: Image.Image, fold: Fold) -> tuple[Image.Image, Image.Image]:
    """Левая и правая страницы: всё по другую сторону линии закрашено чёрным.

    Линия наклонная, поэтому у сгиба остаётся узкий чёрный треугольник — это ожидаемо.
    """
    width, height = image.size
    x_top, x_bottom = fold.x_at(0), fold.x_at(height)
    left_edge = int(np.floor(min(x_top, x_bottom)))
    right_edge = int(np.ceil(max(x_top, x_bottom)))
    left_edge, right_edge = max(0, left_edge), min(width, right_edge)

    left = image.crop((0, 0, right_edge, height)).convert("RGB")
    right = image.crop((left_edge, 0, width, height)).convert("RGB")
    # Треугольники за линией.
    ImageDraw.Draw(left).polygon(
        [(x_top, 0), (right_edge, 0), (right_edge, height), (x_bottom, height)], fill=(0, 0, 0)
    )
    ImageDraw.Draw(right).polygon(
        [(0, 0), (x_top - left_edge, 0), (x_bottom - left_edge, height), (0, height)], fill=(0, 0, 0)
    )
    return left, right


def preview(image: Image.Image, fold: Fold) -> Image.Image:
    """Уменьшенный разворот с линией разреза — для проверки глазами."""
    scale = PREVIEW_WIDTH / image.width
    small = image.convert("RGB").resize((PREVIEW_WIDTH, max(1, round(image.height * scale))), Image.BILINEAR)
    draw = ImageDraw.Draw(small)
    draw.line([(fold.x_at(0) * scale, 0), (fold.x_at(image.height) * scale, small.height)], fill=(255, 0, 0), width=3)
    return small


def fold_from_points(x_top: float, x_bottom: float, height: int) -> Fold:
    """Прямая по двум точкам: x на верхнем и нижнем краю кадра."""
    return Fold(slope=(x_bottom - x_top) / max(1, height), intercept=x_top)


def split_folder(
    in_dir: Path,
    out_dir: Path,
    both_sides_for: tuple[str, ...],
    shifts: dict[str, float] | None = None,
    folds: dict[str, tuple[float, float]] | None = None,
    quality: int = 92,
) -> list[Path]:
    """Разрезать все развороты под ``in_dir``; раскладка подпапок сохраняется.

    ``both_sides_for`` — имена разделов (подпапок), где нужны обе страницы; в остальных
    сохраняется только левая. ``shifts`` — ручные сдвиги подогнанной линии по имени файла,
    ``folds`` — линия целиком (x сверху, x снизу): подгонка промахивается, когда у сгиба
    нет своего минимума краски — например, пустая колонка рядом с ним темнее не бывает.
    """
    shifts = shifts or {}
    folds = folds or {}
    written: list[Path] = []
    preview_dir = out_dir / "_линии"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        rel = path.relative_to(in_dir)
        section = rel.parts[0] if len(rel.parts) > 1 else ""
        with Image.open(path) as image:
            image.load()
            if path.name in folds:
                fold = fold_from_points(*folds[path.name], image.height)
            else:
                fold = find_fold(path, shifts.get(path.name, 0.0))
            left, right = split_spread(image, fold)
            preview(image, fold).save(preview_dir / f"{path.stem}.jpg", quality=80)
        target = out_dir / rel.parent
        target.mkdir(parents=True, exist_ok=True)
        pages = [("L", left)] + ([("R", right)] if section in both_sides_for else [])
        for side, page in pages:
            out_path = target / f"{path.stem}_{side}.jpg"
            page.save(out_path, quality=quality)
            written.append(out_path)
        logger.info("%s: сгиб x=%.0f…%.0f, страниц %d", rel, fold.x_at(0), fold.x_at(image.height), len(pages))
    return written
