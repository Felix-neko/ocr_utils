"""Поворот полосы при записи (``scan_cleanup.runner``).

Крутится результат ПОСЛЕ всей обработки, и проверяется здесь именно это: разметка
применяется в координатах оригинала, а на диск ложится уже развёрнутая полоса.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from PIL import Image

from ocr_utils.scan_cleanup.naming import cleaned_rel_path, split_cleaned_stem, split_rotation_suffix
from ocr_utils.scan_cleanup.runner import process_page
from ocr_utils.scan_cleanup.source import load_markup

HASH = "a1b2c3d4" * 8


@pytest.fixture
def params(pack, tmp_path):
    """Те же параметры, что в test_runner: радиусы под маленький синтетический кадр."""
    from ocr_utils.scan_cleanup.runner import CleanupParams
    from ocr_utils.scan_cleanup.smoothing import SmoothOptions

    db_path, pack_dir, pack_name = pack
    return CleanupParams(
        db_path=db_path,
        pack_name=pack_name,
        pack_dir=pack_dir,
        out_dir=tmp_path / "out",
        smooth=SmoothOptions(dilate_px=4.0, blur_px=16.0),
    )


def page(pack, name: str, rotate_cw: int = 0):
    db_path, _pack_dir, pack_name = pack
    markup = next(p for p in load_markup(db_path, pack_name) if p.rel_path.endswith(name))
    return dataclasses.replace(markup, rotate_cw=rotate_cw)


def test_page_without_rotation_keeps_its_name_and_size(pack, params):
    """Полос без поворота одиннадцать тысяч: переименовывать их значило бы перегнать пак."""
    report = process_page(page(pack, "0010.tif"), params)
    assert report.rotate_cw == 0
    assert report.out_rel_path.endswith("0010.tif")
    assert Image.open(params.out_dir / report.out_rel_path).size == (400, 600)


@pytest.mark.parametrize("rotation,size", [(90, (600, 400)), (180, (400, 600)), (270, (600, 400))])
def test_rotated_page_is_written_turned(pack, params, rotation, size):
    report = process_page(page(pack, "0010.tif", rotation), params)
    assert report.rotate_cw == rotation
    assert Image.open(params.out_dir / report.out_rel_path).size == size


def test_angle_lands_in_the_file_name(pack, params):
    """Иначе --skip-if-exists молча признал бы готовым старый неповёрнутый файл."""
    report = process_page(page(pack, "0010.tif", 90), params)
    assert report.out_rel_path.endswith(".cw90.tif")


def test_changing_the_decision_does_not_reuse_the_old_file(pack, params):
    """Отпечаток берётся у ИСХОДНИКА и от угла не зависит — имя обязано зависеть."""
    plain = process_page(page(pack, "0010.tif"), params)
    turned = process_page(page(pack, "0010.tif", 90), dataclasses.replace(params, skip_if_exists=True))
    assert turned.status == "ok", "повёрнутую полосу нельзя пропускать по неповёрнутому файлу"
    assert turned.out_rel_path != plain.out_rel_path


def test_rotation_happens_after_the_markup_is_applied(pack, params):
    """Маски и прямоугольники живут в координатах ОРИГИНАЛА.

    Проверка косвенная, но по существу: повёрнутый результат обязан совпасть с поворотом
    неповёрнутого. Если бы поворот случался раньше разметки, координаты указывали бы не туда
    и картинки разошлись бы.
    """
    plain = process_page(page(pack, "0010.tif"), params)
    turned = process_page(page(pack, "0010.tif", 90), params)
    a = np.asarray(Image.open(params.out_dir / plain.out_rel_path))
    b = np.asarray(Image.open(params.out_dir / turned.out_rel_path))
    assert np.array_equal(np.rot90(a, k=-1), b)


def test_dpi_survives_the_rotation(pack, params, tmp_path):
    report = process_page(page(pack, "0010.tif", 90), params)
    with Image.open(params.out_dir / report.out_rel_path) as image:
        assert image.info.get("dpi") == Image.open(params.pack_dir / "1970/01/0010.tif").info.get("dpi")


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_name_round_trips_back_to_page_and_angle(rotation):
    """Выгрузка Capture One приходит без папок пака: связь с базой — только через имя."""
    rel = cleaned_rel_path("1970/01/IMG_0034_1L.tif", HASH, ".tif", rotation)
    stem = rel.rsplit("/", 1)[-1].rsplit(".tif", 1)[0]
    assert split_cleaned_stem(stem) == ("IMG_0034_1L", HASH[:8])
    assert split_rotation_suffix(stem)[1] == rotation
