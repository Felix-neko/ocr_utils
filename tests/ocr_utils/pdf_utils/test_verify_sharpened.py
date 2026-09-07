"""Сверка заострённых копий с полосами: ловится ли подмена чужой полосой."""

import numpy as np
import pytest
from PIL import Image

from ocr_utils.pdf_utils.verify_sharpened import PageCheck, check_page, structure_profile

W, H = 200, 300


def page_image(seed: int, sharpen: bool = False) -> np.ndarray:
    """«Полоса»: бумага со строками текста в своём, зависящем от seed, рисунке абзацев."""
    rng = np.random.default_rng(seed)
    array = np.full((H, W), 250, np.uint8)
    y = 20
    while y < H - 20:
        block = int(rng.integers(3, 9))  # абзац своей высоты — он и делает полосу узнаваемой
        for _ in range(block):
            if y >= H - 20:
                break
            indent = int(rng.integers(0, 30))
            array[y : y + 3, 20 + indent : W - 20] = 40
            y += 8
        y += int(rng.integers(6, 20))
    if sharpen:
        # Грубая имитация Capture One: контраст вверх, тон уезжает.
        array = np.clip((array.astype(np.int16) - 128) * 1.4 + 118, 0, 255).astype(np.uint8)
    return array


def write(path, array, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).convert("RGB").save(path, fmt, quality=92)


@pytest.fixture
def pages(tmp_path):
    """Две разные полосы: у каждой оригинал и своя заострённая копия."""
    for index, seed in enumerate((1, 2), start=1):
        write(tmp_path / "blurred" / f"p{index}.tif", page_image(seed), "TIFF")
        write(tmp_path / "sharp" / f"p{index}.jpg", page_image(seed, sharpen=True), "JPEG")
    return tmp_path


def _check(tmp_path, original: str, sharpened: str) -> "object":
    return check_page(
        PageCheck(
            rel_path=original,
            original=tmp_path / "blurred" / original,
            sharpened=tmp_path / "sharp" / sharpened,
            width=W,
            height=H,
        )
    )


def test_its_own_copy_passes(pages):
    result = _check(pages, "p1.tif", "p1.jpg")
    assert result.status == "ok"
    assert result.correlation > 0.9


def test_another_page_of_the_same_size_is_caught(pages):
    """Главный случай: подменённая полоса ТОГО ЖЕ размера — по размеру её не отличить."""
    result = _check(pages, "p1.tif", "p2.jpg")
    assert result.status == "content"
    assert result.correlation < 0.6


def test_a_photo_page_survives_a_heavy_tone_change(pages, tmp_path):
    """Обложка, которую Capture One перекрасил, — не подмена.

    Профили на сплошной фотографии почти плоские и корреляцию не держат; вытягивает
    пиксельная мера, ради которой обе и считаются. Замер по паку-1: такие полосы дают
    0.72-0.86 против 0.20-0.35 у настоящей подмены.
    """
    rng = np.random.default_rng(11)
    photo = rng.integers(60, 200, size=(H, W), dtype=np.uint16).astype(np.uint8)
    photo = np.repeat(np.repeat(photo[::8, ::8], 8, 0), 8, 1)[:H, :W]  # крупные пятна, как на фото
    write(pages / "blurred" / "cover.tif", photo, "TIFF")
    # Capture One: контраст вверх и тон уехал — байты меняются сильно, полоса та же.
    write(
        pages / "sharp" / "cover.jpg",
        np.clip((photo.astype(np.int16) - 128) * 1.8 + 90, 0, 255).astype(np.uint8),
        "JPEG",
    )

    assert _check(pages, "cover.tif", "cover.jpg").status in ("ok", "suspect")
    assert _check(pages, "cover.tif", "p1.jpg").status == "content"


def test_a_copy_of_another_size_is_caught_before_decoding(pages):
    """Разошедшийся размер — отдельный диагноз: полосу даже не надо разжимать."""
    write(pages / "sharp" / "small.jpg", page_image(1, sharpen=True)[: H // 2], "JPEG")
    result = _check(pages, "p1.tif", "small.jpg")
    assert result.status == "size"
    assert "не" in result.reason or "полоса" in result.reason


def test_missing_copy_is_reported(pages):
    result = _check(pages, "p1.tif", "нет-такого.jpg")
    assert result.status == "missing"


def test_profile_ignores_tone_and_reacts_to_layout(pages):
    """Профили нормированы: заострение тон меняет, а форму полосы — нет."""
    plain = structure_profile(pages / "blurred" / "p1.tif")
    sharp = structure_profile(pages / "sharp" / "p1.jpg")
    other = structure_profile(pages / "sharp" / "p2.jpg")
    assert np.corrcoef(plain, sharp)[0, 1] > np.corrcoef(plain, other)[0, 1]
