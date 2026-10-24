"""Кадры наборов стенда и раскладка выпрямленного холста: прямоугольники страниц и линия корешка по кэшу прогона v2 (тем же кодом, что у ``research/page_rectify``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ocr_utils.scan_cropping.image_io import collect_images
from ocr_utils.scan_cropping.v2.pipeline import FrameJob, visible_frames
from ocr_utils.scan_cropping.v2.rectify.coons import TANGENT_MM, Layout, page_patch, plan_layout
from ocr_utils.scan_cropping.v2.rectify.geometry import SpreadGeometry
from ocr_utils.scan_cropping.v2.spread_lines import SpineMethod
from ocr_utils.units import MM_PER_INCH, mm_to_px_float
from research.page_rectify.frames import cached_frames

# Суффикс итогов v2 и выпрямленных итогов (``--output-format tiff``).
TIFF_SUFFIX = ".tiff"


@dataclass(frozen=True)
class FrameSet:
    """Набор кадров (выпуск) с путями прогонов.

    Поля:
        label: Подпись набора (подпуть выхода стенда), например ``hp_1991/01 оплачено``.
        input_dir: Папка исходных кадров прогона v2 (ключи кэша — от неё; Я.Диск, только чтение).
        cache: Кэш прогона v2 этого набора.
        v2_output: Итоги прогона v2 (вход выпрямления).
        rectified: Итоги выпрямления в два преобразования (``page_rectify text-run``), на которых меряется сжатие.
        px_per_mm: Плотность бумаги на кадре, px/мм (у ХиП 1991 — по выпускам, 16.3–17.9).
    """

    label: str
    input_dir: Path
    cache: Path
    v2_output: Path
    rectified: Path
    px_per_mm: float

    @property
    def dpi(self) -> float:
        """Плотность бумаги в точках на дюйм (для ``ocr_utils.units``)."""
        return self.px_per_mm * MM_PER_INCH


@dataclass(frozen=True)
class FrameRef:
    """Кадр набора.

    Поля:
        frame_set: Набор кадра.
        src: Исходный кадр.
    """

    frame_set: FrameSet
    src: Path

    @property
    def job(self) -> FrameJob:
        """Кадр прогона v2 (ключ кэша — от папки набора)."""
        return FrameJob(self.src, self.src.relative_to(self.frame_set.input_dir))

    @property
    def key(self) -> str:
        """Ключ кадра в сводках стенда: ``<набор>/<кадр без расширения>``."""
        return f"{self.frame_set.label}/{self.job.key}"

    @property
    def v2_path(self) -> Path:
        """Итог v2 кадра."""
        return self.frame_set.v2_output / self.job.rel.with_suffix(TIFF_SUFFIX)

    @property
    def rectified_path(self) -> Path:
        """Выпрямленный итог кадра."""
        return self.frame_set.rectified / self.job.rel.with_suffix(TIFF_SUFFIX)


def set_frames(frame_set: FrameSet) -> list[FrameRef]:
    """Кадры набора без скрытых папок, по порядку имён.

    Args:
        frame_set: Набор.

    Returns:
        Кадры набора.
    """
    files = visible_frames(collect_images(frame_set.input_dir, True), frame_set.input_dir)
    return [FrameRef(frame_set, f) for f in sorted(files)]


def frame_geometry(ref: FrameRef) -> SpreadGeometry:
    """Области страниц и рамка вырезки кадра по кэшу v2 (вето фона и разделитель ``auto`` — как у прогонов ХиП).

    Args:
        ref: Кадр.

    Returns:
        Геометрия разворота (``frames`` None — страниц с областями не две).
    """
    return cached_frames(ref.src, ref.frame_set.cache, ref.job.key, True, SpineMethod.AUTO)


def frame_layout(geometry: SpreadGeometry, shape: tuple[int, int], dpi: float) -> Layout | None:
    """Раскладка выпрямленного холста: прямоугольники страниц и линия корешка.

    Args:
        geometry: Геометрия разворота (``frame_geometry``).
        shape: ``(H, W)`` итога v2.
        dpi: Плотность бумаги, точек на дюйм (длина касательной продолжения сторон).

    Returns:
        Раскладка или None — областей страниц нет или бумага не найдена.
    """
    if geometry.frames is None or geometry.box is None:
        return None
    tangent_px = mm_to_px_float(TANGENT_MM, dpi)
    patches = (page_patch(geometry.frames[0], True, geometry.to_output, tangent_px),
               page_patch(geometry.frames[1], False, geometry.to_output, tangent_px))  # fmt: skip
    return plan_layout(patches, shape)
