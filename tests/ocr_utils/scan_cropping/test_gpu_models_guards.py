"""Тесты флагов загрузки ``GpuModels`` и охранников незагруженных моделей.

Проходят без GPU и без весов ровно потому, что проверяют случай «ничего не
загружено»: ``GpuModels(with_detection=False, with_lama=False)`` не трогает ни
ultralytics, ни torchscript-веса.
"""

import numpy as np
import pytest

from ocr_utils.scan_cropping.gpu_models import GpuModels


@pytest.fixture
def empty_models():
    """Объект без единой загруженной модели."""
    with GpuModels(device="cpu", with_detection=False, with_lama=False) as models:
        yield models


def test_nothing_is_loaded(empty_models):
    assert empty_models.device == "cpu"
    assert empty_models._yolo_page is None
    assert empty_models._yolo_hand is None
    assert empty_models._sam is None
    assert empty_models._lama is None


@pytest.mark.parametrize(
    "call,flag",
    [
        (lambda m: m.detect_page_boxes(np.zeros((8, 8, 3), np.uint8), 0.1), "with_detection=True"),
        (lambda m: m.detect_hand_boxes(np.zeros((8, 8, 3), np.uint8), 0.1), "with_detection=True"),
        (lambda m: m.segment_boxes(np.zeros((8, 8, 3), np.uint8), np.array([[0, 0, 4, 4]])), "with_detection=True"),
        (lambda m: m.lama_fill_roi(np.zeros((8, 8, 3), np.uint8), np.ones((8, 8), np.uint8)), "with_lama=True"),
    ],
)
def test_guard_names_the_constructor_flag(empty_models, call, flag):
    """Сообщение обязано называть флаг: иначе по трейсбеку непонятно, что чинить."""
    with pytest.raises(RuntimeError, match=flag):
        call(empty_models)


def test_sd_guard_names_the_model_option(empty_models):
    from ocr_utils.inpainting.backends import SdParams

    with pytest.raises(RuntimeError, match="sd_model="):
        empty_models.sd_fill_roi(
            np.zeros((8, 8, 3), np.uint8), np.ones((8, 8), np.uint8), "prompt", "negative", SdParams()
        )


def test_layout_and_shadow_guards_still_work(empty_models):
    with pytest.raises(RuntimeError, match="with_layout=True"):
        empty_models.layout_blocks(np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(RuntimeError, match="shadow_variant="):
        empty_models.remove_shadow(np.zeros((8, 8, 3), np.uint8))


def test_segment_boxes_returns_empty_before_touching_sam(empty_models):
    """Пустой список боксов — не повод падать: SAM в этой ветке не зовётся вовсе."""
    out = empty_models.segment_boxes(np.zeros((8, 8, 3), np.uint8), np.empty((0, 4)))
    assert out.shape == (0, 8, 8)
