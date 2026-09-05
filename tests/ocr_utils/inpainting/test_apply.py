"""Тесты общего цикла закраса (``inpainting.apply.inpaint_by_groups``).

Сеть заменена заглушкой: цикл отвечает за геометрию и вклейку, и проверять его
надо без GPU.
"""

import numpy as np
import pytest

from ocr_utils.inpainting.apply import inpaint_by_groups
from ocr_utils.inpainting.grouping import group_masks
from ocr_utils.inpainting.roi import mask_components


class RecordingFiller:
    """Заливает ROI ровным цветом и запоминает, с чем её звали."""

    def __init__(self, value: int = 7):
        self.value = value
        self.calls: "list[tuple[int, int, int, int]]" = []

    def __call__(self, roi, roi_mask, bounds):
        self.calls.append(bounds)
        return np.full_like(roi, self.value)


def frame(h: int = 300, w: int = 300) -> np.ndarray:
    """Кадр с неоднородным содержимым — чтобы «не тронуто» значило что-то."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def paint(shape, boxes) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 255
    return mask


def test_empty_mask_returns_source_untouched():
    rgb = frame()
    out, bounds = inpaint_by_groups(rgb, np.zeros(rgb.shape[:2], np.uint8), RecordingFiller())
    assert bounds == []
    assert out is rgb  # даже копии не делается: заливать нечего


def test_outside_mask_is_bit_identical():
    """Главное свойство: за пределами маски кадр не меняется вовсе."""
    rgb = frame()
    mask = paint(rgb.shape[:2], [(100, 100, 140, 140)])

    out, _ = inpaint_by_groups(rgb, mask, RecordingFiller(), feather=0)

    outside = mask == 0
    assert np.array_equal(out[outside], rgb[outside])
    assert (out[mask > 0] == 7).all()


def test_one_call_per_group():
    filler = RecordingFiller()
    # Две близкие области и одна далёкая: групп должно быть две.
    mask = paint((400, 400), [(10, 10, 40, 40), (55, 10, 85, 40), (300, 300, 330, 330)])

    _, bounds = inpaint_by_groups(frame(400, 400), mask, filler, groups=group_masks)

    assert len(filler.calls) == 2
    assert bounds == filler.calls


def test_components_strategy_reproduces_per_component_behaviour():
    """``groups=mask_components`` — прежнее поведение ``GpuModels.inpaint``.

    Это страховка пальцевого пайплайна: он ходит через ту же функцию, и разбиение
    у него обязано остаться покомпонентным.
    """
    filler = RecordingFiller()
    mask = paint((400, 400), [(10, 10, 40, 40), (55, 10, 85, 40), (300, 300, 330, 330)])

    inpaint_by_groups(frame(400, 400), mask, filler, groups=mask_components)

    assert len(filler.calls) == 3


def test_roi_is_scaled_around_the_group():
    """ROI = (рамка группы + padding), растянутая в ``roi_scale`` раз от центра."""
    filler = RecordingFiller()
    mask = paint((400, 400), [(150, 150, 200, 200)])

    inpaint_by_groups(frame(400, 400), mask, filler, padding=0, roi_scale=2.0)

    (x1, y1, x2, y2) = filler.calls[0]
    assert (x2 - x1, y2 - y1) == (100, 100)  # вдвое от стороны 50
    assert ((x1 + x2) // 2, (y1 + y2) // 2) == (175, 175)  # центр на месте


def test_roi_is_clipped_by_the_frame():
    filler = RecordingFiller()
    mask = paint((400, 400), [(0, 0, 40, 40)])

    inpaint_by_groups(frame(400, 400), mask, filler, padding=64, roi_scale=2.0)

    x1, y1, x2, y2 = filler.calls[0]
    assert (x1, y1) == (0, 0)
    assert x2 <= 400 and y2 <= 400


def test_later_groups_see_earlier_results():
    """Вторая группа опирается на уже закрашенное, а не на исходный объект.

    Контекстные поля соседних групп перекрываются, и если бы ROI резался из
    исходника, вторая заливка тянула бы в себя объект, которого в итоге не будет.
    """
    seen: "list[int]" = []

    def filler(roi, roi_mask, bounds):
        seen.append(int(roi.max()))
        return np.zeros_like(roi)

    rgb = np.full((200, 600, 3), 200, np.uint8)
    # Две далёкие области; ROI второй накрывает место первой.
    mask = paint(rgb.shape[:2], [(50, 50, 100, 100), (400, 50, 450, 100)])
    rgb[50:100, 50:100] = 255  # «объект» в первой зоне

    inpaint_by_groups(rgb, mask, filler, padding=400, roi_scale=1.0, feather=0)

    # Первая заливка видит объект (255), вторая — уже занулённую первую зону.
    assert seen[0] == 255
    assert seen[1] == 200
