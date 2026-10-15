"""Тесты фона на коврике без печати (синтетика): освещённый коврик с градиентом и белая обложка без печати — фон весь коврик, книга — обложка (МТС 1990/04 IMG_1108)."""

import numpy as np

from ocr_utils.scan_cropping.v2.background import bg_dark, bg_flood

H, W = 600, 800


def closed_book() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Чёрный коврик с градиентом освещения 18 → 42 слева направо, белая обложка без печати, печати нет.

    Returns:
        ``(картинка BGR, печать, обложка)``.
    """
    xx = np.linspace(18, 42, W, dtype=np.float32)
    img = np.repeat(np.tile(xx, (H, 1))[..., None], 3, axis=2)
    img += np.random.default_rng(0).normal(0, 2, img.shape).astype(np.float32)  # зерно коврика
    cover = np.zeros((H, W), bool)
    cover[150:450, 250:550] = True
    img[cover] = 235
    return np.clip(img, 0, 255).astype(np.uint8), np.zeros((H, W), bool), cover


def test_gap_rule_mat_is_background() -> None:
    """Оцу внутри коврика (печати нет) не принимается: весь коврик — фон, обложка — не фон."""
    img, content, cover = closed_book()
    for bg in (bg_dark(img, content), bg_flood(img, content)):
        assert (bg & ~cover).sum() / (~cover).sum() > 0.98
        assert (bg & cover).sum() < 0.001 * cover.sum()  # углы прямоугольника скругляет закрытие, как в пакете
