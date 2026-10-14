"""Тесты карты тени пальца (синтетика): тень на бумаге с буквами восстанавливается и снимается, края области плавные, без тени поправки нет."""

import cv2
import numpy as np
import pytest

from research.finger_repaint.shadow import LightenMode, ShadowInputs, ShadowMethod, estimate, lighten, shadow_score

PX_PER_MM = 8.0
PAPER = np.array([215, 230, 238], np.float32)
H, W = 1600, 1200


def page(shadow: bool) -> ShadowInputs:
    """Страница целиком бумага, справа — коврик; палец у правого края посередине, гауссова тень вокруг, строки букв."""
    img = np.tile(PAPER, (H, W, 1))
    if shadow:
        yy, xx = np.mgrid[0:H, 0:W]
        g = np.exp(-(((xx - 1000) / 160.0) ** 2 + ((yy - 800) / 220.0) ** 2))
        img *= (1 - 0.25 * g)[..., None]
    img = img.astype(np.uint8)
    for y in range(120, H - 80, 40):
        cv2.putText(img, "TEXT TEXT TEXT TEXT", (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
    book = np.zeros((H, W), bool)
    book[:, :1100] = True
    img[~book] = (20, 20, 20)
    core = np.zeros((H, W), np.uint8)
    cv2.ellipse(core, (1100, 800), (90, 50), 0, 0, 360, 1, -1)
    paint = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0
    return ShadowInputs("synthetic", img, paint, core > 0, book, ~paint)


@pytest.mark.parametrize("method", [ShadowMethod.PULL_PUSH, ShadowMethod.ROBUST, ShadowMethod.TELEA])
def test_shadow_removed(method: ShadowMethod) -> None:
    """Тень вокруг пальца после осветления почти уходит, буквы не бледнеют."""
    inp = page(shadow=True)
    smap = estimate(inp, method, PX_PER_MM)
    out = lighten(inp.img, smap, LightenMode.GAIN)
    before = shadow_score(inp.img, inp.img, smap, PX_PER_MM)
    after = shadow_score(inp.img, out, smap, PX_PER_MM)
    assert before.residual_mm2 > 300
    assert after.residual_mm2 < 0.2 * before.residual_mm2
    assert after.ink_contrast > 0.85


def test_no_shadow_no_change() -> None:
    """Без тени поправка почти нулевая."""
    inp = page(shadow=False)
    smap = estimate(inp, ShadowMethod.PULL_PUSH, PX_PER_MM)
    out = lighten(inp.img, smap, LightenMode.GAIN)
    assert np.abs(out.astype(int) - inp.img.astype(int))[inp.book].mean() < 1.0


def test_edge_starts_from_zero() -> None:
    """На краю области (снаружи бумага) поправка нулевая и растёт без ступеньки."""
    inp = page(shadow=True)
    smap = estimate(inp, ShadowMethod.PULL_PUSH, PX_PER_MM)
    gain = smap.log_gain.max(axis=2)
    paper_out = smap.book & ~smap.area
    edge = smap.area & (cv2.dilate(paper_out.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    assert gain[edge].max() < 0.01
    assert gain.max() > 0.1


def lab_to_bgr(light: float, a: float, b: float) -> tuple[int, int, int]:
    """Цвет BGR uint8 по Lab (L 0..100, a*, b* как в CIE)."""
    lab = np.array([[[light * 255 / 100, a + 128, b + 128]]], np.uint8)
    return tuple(int(v) for v in cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0])


@pytest.mark.parametrize("paper_ab", [(0, 2), (3, 18)])
def test_skin_relative_to_paper(paper_ab: tuple[float, float]) -> None:
    """Белая и пожелтевшая бумага (ПХ, b* +18) — не кожа; кожа (a* 13, b* 20) — кожа; фиолетовая печать и тень — нет."""
    from research.finger_repaint.shadow import skin_mask

    paper = lab_to_bgr(88, *paper_ab)
    img = np.full((400, 400, 3), paper, np.uint8)
    img[100:200, 100:200] = lab_to_bgr(65, 13, 20)  # кожа
    img[250:350, 40:180] = lab_to_bgr(40, 20, -40)  # фиолетовая печать (крупная заливка)
    img[250:350, 220:380] = lab_to_bgr(66, paper_ab[0] * 0.8, paper_ab[1] * 0.8)  # тень на той же бумаге
    skin = skin_mask(img, 4.0, np.ones((400, 400), bool))
    assert skin[120:180, 120:180].all()
    assert not skin[260:340, 60:160].any()
    assert not skin[260:340, 240:360].any()
    assert not skin[:80, :].any()
