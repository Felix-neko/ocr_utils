"""Холст первого преобразования (освещение + Кунс) по кэшу и итогу прогона v2 и вырезки страниц, как их видит детектор line art в ``rectify.text_field.page_regions``."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.frame_flags.flags import Side, read_flags
from ocr_utils.scan_cropping.v2.pipeline import read_bgr
from ocr_utils.scan_cropping.v2.rectify.text_field import CROP_MARGIN_MM, crop_box
from ocr_utils.scan_cropping.v2.rectify.text_run import PageControl, TextParams, text_frame
from ocr_utils.scan_cropping.v2.spread_lines import SpineMethod
from ocr_utils.units import MM_PER_INCH, mm_to_px_float
from research.page_rectify.frames import cached_frames

# Суффикс итогов v2 (``--output-format tiff``).
V2_SUFFIX = ".tiff"
# Части страниц разворота: левая и правая.
PARTS = ("L", "R")


@dataclass(frozen=True)
class RunPaths:
    """Пути одного набора прогона v2.

    Поля:
        photos: Папка исходных кадров набора (Я.Диск, только чтение).
        cache: Папка кэша прогона v2 набора.
        output: Папка итогов v2 (TIFF).
    """

    photos: Path
    cache: Path
    output: Path


@dataclass(frozen=True)
class PageCrop:
    """Вырезка страницы холста, поданная детектору.

    Поля:
        part: ``L`` или ``R``.
        box: Вырезка на холсте ``(x0, y0, x1, y1)``.
        dpi: Плотность холста (px/мм прогона × 25.4).
        plain: Страница шла в поле (не цветная, не тонированная); у не-``plain`` ``rectify`` детектор не зовёт.
        field: Флаги кадра разрешают поле по строкам.
    """

    part: str
    box: tuple[int, int, int, int]
    dpi: float
    plain: bool
    field: bool


def frame_canvas(paths: RunPaths, issue: str, frame: str, px_per_mm: float) -> tuple[np.ndarray, list[PageCrop]] | None:
    """Холст первого преобразования кадра и вырезки его страниц.

    Повторяет начало ``text_run.text_frame`` (освещение → Кунс) через сам ``text_frame`` с нулём проходов и
    ``crop_box`` с полем ``CROP_MARGIN_MM``; области страниц — пересчёт по кэшу v2 (``research.page_rectify``).

    Args:
        paths: Пути набора.
        issue: Подпапка выпуска (``06 оплачено``).
        frame: Имя кадра без расширения (``IMG_0419``).
        px_per_mm: Плотность бумаги выпуска, px/мм (как ``--px-per-mm`` прогона).

    Returns:
        ``(холст BGR, вырезки страниц)`` или None — у кадра нет двух областей страниц, нет итога или кэш не от него.
    """
    src = paths.photos / issue / f"{frame}.jpg"
    v2_path = paths.output / issue / f"{frame}{V2_SUFFIX}"
    if not v2_path.exists() or not src.exists():
        return None
    image = read_bgr(v2_path)
    dpi = px_per_mm * MM_PER_INCH
    geometry = cached_frames(src, paths.cache / issue, frame, True, SpineMethod.AUTO)
    if geometry.box is None or geometry.frames is None or len(geometry.frames) != 2:
        return None
    # Рамка вырезки по кэшу не совпала с итогом — кэш перезаписан другим прогоном.
    if (geometry.box[3] - geometry.box[1], geometry.box[2] - geometry.box[0]) != image.shape[:2]:
        return None
    flags = read_flags(src)
    sides = (Side.LEFT, Side.RIGHT)
    controls = tuple(
        PageControl(equalize=flags.equalize(s), field=not flags.ignored(s) and flags.text_rectify(s)) for s in sides
    )
    # Ноль проходов: text_frame возвращает холст после освещения и Кунса (то, что видит page_regions).
    result = text_frame(image, geometry, dpi, TextParams(passes=0, gutter_decompression=False), controls)
    layout = result.layout
    margin = mm_to_px_float(CROP_MARGIN_MM, dpi)
    crops = [
        PageCrop(
            part,
            crop_box(rect, result.first.shape[:2], margin, layout.spine_x, i == 0),
            dpi,
            bool(result.pages[i].plain),
            controls[i].field,
        )
        for i, (part, rect) in enumerate(zip(PARTS, layout.rects))
    ]
    return result.first, crops


def save_crops(canvas: np.ndarray, crops: list[PageCrop], out_dir: Path, frame: str) -> list[Path]:
    """Записать вырезки страниц холста PNG без потерь и паспорт JSON рядом.

    Args:
        canvas: Холст BGR.
        crops: Вырезки страниц.
        out_dir: Папка выпуска в кэше холстов.
        frame: Имя кадра.

    Returns:
        Пути записанных PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for crop in crops:
        x0, y0, x1, y1 = crop.box
        path = out_dir / f"{frame}_{crop.part}.png"
        # Сжатие 1: быстро, без потерь — детектор должен видеть ровно те пиксели, что в rectify.
        cv2.imwrite(str(path), canvas[y0:y1, x0:x1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        path.with_suffix(".json").write_text(json.dumps(asdict(crop), ensure_ascii=False))
        written.append(path)
    return written


def load_crop(path: Path) -> tuple[np.ndarray, PageCrop]:
    """Прочитать вырезку страницы и её паспорт.

    Args:
        path: PNG вырезки (паспорт — тот же путь с ``.json``).

    Returns:
        ``(картинка BGR, паспорт)``.
    """
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["box"] = tuple(meta["box"])
    return cv2.imread(str(path), cv2.IMREAD_COLOR), PageCrop(**meta)
