"""Тесты точного стенда тени (синтетика): вход читается из папки кадра, меры по зонам, итог прежнего прогона по ключу."""

from pathlib import Path

import cv2
import numpy as np

from research.finger_repaint.edge import baseline_path, evaluate_frame, load_edge_inputs

PX_PER_MM = 8.0
PAPER = np.array([215, 230, 238], np.float32)
H, W = 1600, 1200


def write_frame(cache_dir: Path) -> None:
    """Кадр с тёмным пятном LaMa у края листа: ``before_shadow.png`` и ``shadow_masks.npz`` в папке кадра."""
    cache_dir.mkdir(parents=True)
    book = np.zeros((H, W), bool)
    book[:, :1100] = True
    core = np.zeros((H, W), np.uint8)
    cv2.ellipse(core, (1130, 800), (60, 40), 0, 0, 360, 1, -1)
    paint = (cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0) & book
    blot = np.zeros((H, W), np.uint8)
    cv2.circle(blot, (1090, 800), 45, 1, -1)
    soft = cv2.GaussianBlur(blot.astype(np.float32), (0, 0), 1.0) * paint
    img = (np.tile(PAPER, (H, W, 1)) * (1 - 0.45 * soft)[..., None]).astype(np.uint8)
    cv2.imwrite(str(cache_dir / "before_shadow.png"), img)
    np.savez_compressed(cache_dir / "shadow_masks.npz", paint=paint, core=core > 0, book=book, original=~paint)


def test_evaluate_frame(tmp_path: Path) -> None:
    """Пятно снимается: кайма «стало» ближе к нулю, чем «до»; листы пишутся; без файлов входа — None."""
    write_frame(tmp_path / "МТС__1991__02__IMG_0001")
    inputs = load_edge_inputs(tmp_path / "МТС__1991__02__IMG_0001")
    assert inputs is not None and inputs.key == "МТС__1991__02__IMG_0001"
    rows = evaluate_frame(inputs, PX_PER_MM, inputs.image, tmp_path)
    assert rows and {"before_rim_p1", "baseline_rim_p1", "current_rim_p1", "current_over_p99"} <= set(rows[0])
    worst = min(rows, key=lambda r: r["before_rim_p1"])
    assert worst["before_rim_p1"] < -15
    assert worst["current_rim_p1"] > -5
    assert list(tmp_path.glob("*_z*.jpg"))
    assert load_edge_inputs(tmp_path / "нет") is None


def test_baseline_path(tmp_path: Path) -> None:
    """Ключ ``a__b__c`` → итог ``a/b/c.tiff``; нет файла — None."""
    target = tmp_path / "МТС" / "1990" / "04 есть фотографии по страницам" / "IMG_1112.tiff"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    assert baseline_path(tmp_path, "МТС__1990__04 есть фотографии по страницам__IMG_1112") == target
    assert baseline_path(tmp_path, "МТС__1990__04__IMG_9999") is None
