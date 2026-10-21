"""Перевод пиксельных констант в мм бумаги: при эталонном разрешении — ровно прежние пиксели.

Каждая константа, переведённая из пикселей в мм (или мм²), обязана при том разрешении, на котором
её замеряли, давать то же число пикселей, что и раньше: иначе старые паки (600 dpi) обработались бы
иначе, чем до перевода. Таблица — «константа → (эталонный dpi, прежние пиксели)».
"""

import pytest

from ocr_utils.external_ocr_services.tiling import DEFAULT_MAX_SRC_TILE_MM
from ocr_utils.page_layout.raster.regions import GROW_STEP_MM, RASTER_WORK_DPI, SURYA_LINEART_P99_MM2
from ocr_utils.paper import PAPER_BLUR_MM, PAPER_BLUR_PX, PAPER_DILATE_MM, PAPER_DILATE_PX
from ocr_utils.scan_cleanup.inpaint import FEATHER_MM, GROUP_MIN_DILATE_MM, MIN_ZONE_AREA_MM2, ROI_PADDING_MM
from ocr_utils.scan_cleanup.smoothing import (
    DEFAULT_DILATE_MM,
    MIN_GLYPH_AREA_MM2,
    SAUVOLA_WINDOW_MM,
    SURE_GLYPH_AREA_MM2,
)
from ocr_utils.scan_cropping.v1 import cli as v1_cli
from ocr_utils.scan_cropping.v2 import pipeline as v2_pipeline
from ocr_utils.scan_cropping.v2.scale import Scale
from ocr_utils.units import mm2_to_px2, mm_to_px, mm_to_px_float, odd_px, px2_to_mm2, px_to_mm

# Длины: (мм, эталонный dpi, прежние px).
LENGTHS = {
    "page_layout.raster.GROW_STEP_MM": (GROW_STEP_MM, 600, 64),
    "paper.PAPER_DILATE_MM": (PAPER_DILATE_MM, 600, PAPER_DILATE_PX),
    "paper.PAPER_BLUR_MM": (PAPER_BLUR_MM, 600, PAPER_BLUR_PX),
    "scan_cleanup.DEFAULT_DILATE_MM": (DEFAULT_DILATE_MM, 600, 25),
    "scan_cleanup.SAUVOLA_WINDOW_MM": (SAUVOLA_WINDOW_MM, 600, 101),
    "scan_cleanup.GROUP_MIN_DILATE_MM": (GROUP_MIN_DILATE_MM, 600, 16),
    "scan_cleanup.ROI_PADDING_MM": (ROI_PADDING_MM, 600, 64),
    "scan_cleanup.FEATHER_MM": (FEATHER_MM, 600, 9),
    "external_ocr.DEFAULT_MAX_SRC_TILE_MM": (DEFAULT_MAX_SRC_TILE_MM, 600, 4500),
    "scan_cropping.v1.EXTRA_EROSION_MM": (v1_cli.EXTRA_EROSION_MM, 450, 80),
    "scan_cropping.v1.FINGER_DILATE_MM": (v1_cli.FINGER_DILATE_MM, 450, 40),
    "scan_cropping.v1.LAYOUT_PAD_MM": (v1_cli.LAYOUT_PAD_MM, 450, 12),
    "scan_cropping.v1.DETECT_PAD_TB_MM": (v1_cli.DETECT_PAD_TB_MM, 450, 250),
    "scan_cropping.v1.PAGE_CLOSE_MM (300 dpi)": (v1_cli.PAGE_CLOSE_MM, 300, 30),
    "scan_cropping.v1.PAGE_CLOSE_MM (450 dpi)": (v1_cli.PAGE_CLOSE_MM, 450, 45),
    "scan_cropping.v1.BG_FILL_EROSION_MM": (v1_cli.BG_FILL_EROSION_MM, 450, 100),
    "scan_cropping.v1.FINGER_PADDING_MM": (v1_cli.FINGER_PADDING_MM, 450, 64),
    "scan_cropping.v1.FINGER_FEATHER_MM": (v1_cli.FINGER_FEATHER_MM, 450, 9),
}

# Площади: (мм², эталонный dpi, прежние px²).
AREAS = {
    "page_layout.raster.SURYA_LINEART_P99_MM2": (SURYA_LINEART_P99_MM2, 600, 4600),
    "scan_cleanup.MIN_GLYPH_AREA_MM2": (MIN_GLYPH_AREA_MM2, 600, 34),
    "scan_cleanup.SURE_GLYPH_AREA_MM2": (SURE_GLYPH_AREA_MM2, 600, 500),
    "scan_cleanup.MIN_ZONE_AREA_MM2": (MIN_ZONE_AREA_MM2, 600, 20),
}

# Сигмы размытия: (мм, эталонный dpi, прежние px) — сравниваются без округления, с допуском.
SIGMAS = {"scan_cropping.v1.CROP_FILL_BLUR_MM": (v1_cli.CROP_FILL_BLUR_MM, 450, 48.0)}


@pytest.mark.parametrize("name", sorted(LENGTHS))
def test_length_gives_legacy_pixels(name: str) -> None:
    mm, dpi, px = LENGTHS[name]
    assert mm_to_px(mm, dpi) == px, f"{name}: {mm} мм при {dpi} dpi"


@pytest.mark.parametrize("name", sorted(AREAS))
def test_area_gives_legacy_pixels(name: str) -> None:
    mm2, dpi, px2 = AREAS[name]
    assert mm2_to_px2(mm2, dpi) == px2, f"{name}: {mm2} мм² при {dpi} dpi"


@pytest.mark.parametrize("name", sorted(SIGMAS))
def test_sigma_gives_legacy_pixels(name: str) -> None:
    mm, dpi, px = SIGMAS[name]
    assert mm_to_px_float(mm, dpi) == pytest.approx(px, abs=1e-4), f"{name}: {mm} мм при {dpi} dpi"


def test_v2_defaults_give_calibration_pixels() -> None:
    """Умолчания опций v2 в мм при калибровочной плотности 13.8 px/мм — прежние px калибровочного кадра."""
    scale = Scale()
    assert scale.mm_px(v2_pipeline.FINGER_DILATE_MM) == 40
    assert (scale.mm_px(v2_pipeline.LAYOUT_PAD_MM[0]), scale.mm_px(v2_pipeline.LAYOUT_PAD_MM[1])) == (18, 36)
    assert scale.mm_px(v2_pipeline.CORE_GROW_MM, minimum=0) == 4
    assert scale.seg_px(v2_pipeline.REFINE_DEPTH_MM) == 35
    assert scale.mm_px_float(v2_pipeline.CROP_FILL_BLUR_MM) == pytest.approx(48.0, abs=1e-4)
    # Пиксели калибровочного кадра на калибровочном кадре не меняются, нечётные ядра — тоже.
    assert (scale.frame_px(64), scale.frame_odd_px(41), scale.frame_odd_px(21)) == (64, 41, 21)


def test_v2_scale_from_dpi() -> None:
    """``--dpi 450`` — 17.72 px/мм; пиксели калибровочного кадра растут в 1.28 раза, ядра остаются нечётными."""
    scale = Scale.from_dpi(450)
    assert scale.px_per_mm == pytest.approx(17.7165, abs=1e-4)
    assert scale.frame_px(40) == 51
    assert scale.frame_odd_px(41) % 2 == 1 and scale.frame_odd_px(21) % 2 == 1


def test_raster_copy_is_exact_fraction_at_600_and_450() -> None:
    """Копия растрового детектора 150 dpi — ровно 1/4 кадра 600 dpi и 1/3 кадра 450 dpi."""
    assert 600 / RASTER_WORK_DPI == 4 and 450 / RASTER_WORK_DPI == 3


def test_units_round_trip_and_helpers() -> None:
    assert px_to_mm(mm_to_px(2.0, 450), 450) == pytest.approx(2.0, abs=25.4 / 450)
    assert px2_to_mm2(mm2_to_px2(10.0, 600), 600) == pytest.approx(10.0, abs=0.01)
    assert mm_to_px(0.0, 600) == 1 and mm_to_px(0.0, 600, minimum=0) == 0
    assert mm2_to_px2(0.0, 600, minimum=0) == 0
    assert odd_px(1.7, 600) % 2 == 1 and odd_px(0.0, 600) == 3
