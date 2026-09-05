"""Раскладка сравнений: ряд, сетка и выбор врезок (``scan_cleanup.compare``)."""

import numpy as np

from ocr_utils.scan_cleanup.compare import CompareMasksParams, _grid, _row, mask_variants, pick_crops


def tile(h: int = 40, w: int = 60, value: int = 0) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


def test_row_aligns_by_height():
    out = _row([tile(40), tile(20)], gap=10)
    assert out.shape[0] == 40
    assert out.shape[1] == 60 + 10 + 60


def test_grid_falls_back_to_a_row_when_it_fits():
    images = [tile() for _ in range(3)]
    assert _grid(images, cols=4).shape == _row(images).shape


def test_grid_wraps_into_rows():
    """Девятнадцать врезок по 1200 px одним рядом дают 23 000 пикселей — их и разбиваем."""
    out = _grid([tile() for _ in range(9)], cols=4, gap=10)
    # Три ряда: 4 + 4 + 1, последний добит белым до ширины остальных.
    assert out.shape[0] == 40 * 3 + 10 * 2
    assert out.shape[1] == 60 * 4 + 10 * 3


def test_grid_zero_cols_is_a_single_row():
    out = _grid([tile() for _ in range(9)], cols=0)
    assert out.shape[0] == 40


def test_pick_crops_prefers_dense_windows_and_spreads_them():
    mask = np.zeros((600, 600), np.uint8)
    mask[10:110, 10:110] = 255  # плотный угол
    mask[400:420, 400:420] = 255  # разреженный

    windows = pick_crops(mask, side=200, count=2)

    assert len(windows) == 2
    assert windows[0] == (0, 0)  # самое плотное окно первым
    x, y = windows[1]
    assert abs(x) >= 200 or abs(y) >= 200  # второе разнесено с первым


def test_pick_crops_is_deterministic():
    rng = np.random.default_rng(0)
    mask = (rng.random((500, 500)) > 0.7).astype(np.uint8) * 255
    assert pick_crops(mask, 100, 3) == pick_crops(mask, 100, 3)


def test_mask_variants_skip_meaningless_k_for_otsu():
    """У Оцу параметра k нет — второй его вариант не должен плодить дубли."""
    params = CompareMasksParams(
        db_path=None,
        pack_name="x",
        pack_dir=None,
        out_dir=None,
        methods=("otsu", "sauvola"),
        sauvola_ks=(0.06, 0.10),
        dilate_pxs=(15.0,),
        blur_pxs=(60.0,),
    )
    names = [name for name, _ in mask_variants(params)]
    assert names == [
        "otsu_dil15_blur60_ink0.65",
        "sauvola_k0.06_dil15_blur60_ink0.65",
        "sauvola_k0.1_dil15_blur60_ink0.65",
    ]
