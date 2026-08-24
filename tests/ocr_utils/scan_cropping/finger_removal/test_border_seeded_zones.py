"""Тесты отсева ложных ядер маски пальца по краевому признаку (``keep_border_seeded_zones``)."""

import numpy as np

from ocr_utils.scan_cropping.finger_removal.asymmetric_dilation import dilate_finger_zones
from ocr_utils.scan_cropping.finger_removal.masking import keep_border_seeded_zones

H = W = 1000
EDGE_FRAC = 0.04  # как FINGER_EDGE_FRAC в removal.py
DILATE_PX = 60
RATIO = 2.0

# Ядро настоящего пальца — упирается в левый край кадра.
FINGER = (slice(400, 600), slice(0, 80))
# Ложное ядро (портрет) — в глубине кадра, ближе, чем 2×DILATE_PX к пальцу:
# после дилатации они сольются в один компонент, касающийся рамки.
PORTRAIT = (slice(400, 600), slice(160, 400))


def _mask(*regions) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    for r in regions:
        m[r] = 255
    return m


def _dilated(predilate: np.ndarray) -> np.ndarray:
    return dilate_finger_zones(predilate, DILATE_PX, max_ratio=RATIO)


def test_ложное_ядро_в_глубине_кадра_убирается():
    predilate = _mask(FINGER, PORTRAIT)
    mask, cleaned, dropped = keep_border_seeded_zones(_dilated(predilate), predilate, DILATE_PX, RATIO, EDGE_FRAC)
    assert dropped == 1
    assert cleaned[PORTRAIT].max() == 0  # ложное ядро ушло из сырой маски
    assert cleaned[FINGER].min() == 255  # палец на месте
    assert mask[450, 350] == 0  # дальний край портрета больше не закрашивается


def test_слипание_после_дилатации_не_легализует_ложное_ядро():
    """Тот самый случай IMG_0011: без отсева по ядрам портрет уезжает в закраску."""
    predilate = _mask(FINGER, PORTRAIT)
    naive = _dilated(predilate)
    assert (naive[PORTRAIT] > 0).all()  # слиплись: раздутая маска накрыла портрет целиком

    mask, _, _ = keep_border_seeded_zones(naive, predilate, DILATE_PX, RATIO, EDGE_FRAC)
    assert int((mask > 0).sum()) < int((naive > 0).sum())


def test_пиксели_внутри_зоны_настоящего_пальца_остаются_закрашенными():
    """Кайма пальца обоснована соседством с ним, а не ложной детекцией."""
    predilate = _mask(FINGER, PORTRAIT)
    mask, _, _ = keep_border_seeded_zones(_dilated(predilate), predilate, DILATE_PX, RATIO, EDGE_FRAC)
    assert mask[500, 100] == 255  # в 20 px от пальца — внутри его дилатации


def test_без_ложных_ядер_маска_не_меняется():
    predilate = _mask(FINGER)
    naive = _dilated(predilate)
    mask, cleaned, dropped = keep_border_seeded_zones(naive, predilate, DILATE_PX, RATIO, EDGE_FRAC)
    assert dropped == 0
    assert np.array_equal(cleaned, predilate)
    assert np.array_equal(mask, naive)


def test_все_ядра_не_у_края_пустая_маска():
    predilate = _mask(PORTRAIT)
    mask, cleaned, dropped = keep_border_seeded_zones(_dilated(predilate), predilate, DILATE_PX, RATIO, EDGE_FRAC)
    assert dropped == 1
    assert int((mask > 0).sum()) == 0
    assert int((cleaned > 0).sum()) == 0
