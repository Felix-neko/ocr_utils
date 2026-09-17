import numpy as np

from research.geometry_regression.field import estimate_field, field_metrics, robust_affine
from research.geometry_regression.render import to_work
from tests.research.geometry_regression.synthetic import binarize, rotate, text_page, wave_region


def test_robust_affine_ignores_outliers():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 1000, (60, 2))
    dst = src @ np.array([[1.0, 0.01], [-0.01, 1.0]]).T + np.array([5.0, -3.0])
    dst[:5] += 40.0  # выбросы
    affine, _, weight = robust_affine(src, dst)
    assert np.allclose(affine[:, :2], [[1.0, 0.01], [-0.01, 1.0]], atol=1e-3)
    assert np.allclose(affine[:, 2], [5.0, -3.0], atol=0.1)
    assert (weight[:5] == 0).all()


def test_field_recovers_rotation_and_shift():
    before = binarize(text_page())
    after = binarize(rotate(text_page(), 0.8, shift=(12.0, -7.0)))
    field = estimate_field(to_work(before), to_work(after))
    assert field is not None
    assert abs(field.rot_deg - 0.8) < 0.1
    assert abs(field.shear_deg) < 0.15
    metrics = field_metrics(field, [], [])
    assert metrics["field_resid_p90_mm"] < 0.25
    assert metrics["field_weak_frac"] < 0.05


def test_wave_makes_tiles_weak_inside_region():
    page = text_page()
    before = binarize(page)
    after = binarize(wave_region(page, 1400, 2000, amplitude_px=14.0, period_px=180.0))
    b, a = to_work(before), to_work(after)
    field = estimate_field(b, a)
    assert field is not None
    region = [(0, 700, b.shape[1], 1000)]  # та же полоса в пикселях 150 dpi
    elsewhere = [(0, 0, b.shape[1], 650), (0, 1050, b.shape[1], b.shape[0])]
    inside = field_metrics(field, region, elsewhere)
    assert inside["field_lineart_weak_frac"] > 0.5
    assert inside["field_text_weak_frac"] < 0.1


def test_identity_page_is_quiet():
    before = binarize(text_page())
    field = estimate_field(to_work(before), to_work(before))
    metrics = field_metrics(field, [], [])
    assert abs(metrics["field_rot_deg"]) < 0.02
    assert metrics["field_change_p90_mm"] < 0.05
