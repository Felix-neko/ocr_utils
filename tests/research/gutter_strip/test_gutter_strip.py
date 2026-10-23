"""Тесты стенда полосы-вклейки на синтетике: способы обнаружения по ленте (плато, складки) и разметка; ядро — ``tests/ocr_utils/scan_cropping/v2/test_gutter_strip.py``."""

import numpy as np

from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.spread_lines import Line
from research.gutter_strip.band import Band
from research.gutter_strip.detect import detect_creases, detect_plateau
from research.gutter_strip.labels import Label, read_labels

H = 400
HALF = int(25 * SEG_PX_PER_MM)


def band(light_row: np.ndarray) -> Band:
    """Лента из одного профиля яркости поперёк корешка, повторённого по строкам."""
    us = np.arange(-HALF, HALF + 1, dtype=np.float64)
    light = np.repeat(light_row[None, :].astype(np.float32), H, axis=0)
    zeros = np.zeros(light.shape, bool)
    bgr = np.repeat(light[..., None], 3, axis=2).astype(np.uint8)
    return Band(Line(500.0, 200.0, 0.0, 0.0, 400.0), np.arange(H), us, bgr, light, np.full(light.shape, 135.0, np.float32),
                np.zeros(light.shape, np.uint8), zeros, ~zeros, zeros)  # fmt: skip


def profile(strip: bool) -> np.ndarray:
    """Бумага страниц 235; с полосой — белая полоса 250 шириной 10 мм у корешка с тонкими складками по кромкам."""
    us = np.arange(-HALF, HALF + 1)
    row = np.full(len(us), 235.0)
    if strip:
        half = int(5 * SEG_PX_PER_MM)
        row[np.abs(us) <= half] = 250.0
        for edge in (-half, half):
            row[(us >= edge - 1) & (us <= edge + 1)] = 200.0
    else:
        # Тень сгиба: плавная долина.
        row -= 30 * np.exp(-((us / (3 * SEG_PX_PER_MM)) ** 2))
    return row


def test_plateau_and_creases_find_strip():
    with_strip, without = band(profile(True)), band(profile(False))
    assert detect_plateau(with_strip).score >= 10
    assert detect_plateau(without).score < 1
    creases = detect_creases(with_strip)
    assert creases.score > 0.9
    assert 9 <= creases.extra["width_mm"] <= 11
    assert detect_creases(without).score < 0.1


def test_read_labels(tmp_path):
    path = tmp_path / "set.txt"
    path.write_text("# комментарий\n01__IMG_0021 # strip — полоса\n01__IMG_0022 # none\n\n08__IMG_0061_5 # maybe\n")
    assert read_labels(path) == {"01__IMG_0021": Label.STRIP, "01__IMG_0022": Label.NONE, "08__IMG_0061_5": Label.MAYBE}
