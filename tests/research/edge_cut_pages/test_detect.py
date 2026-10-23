"""Тесты детектора строк, срезанных краем листа, на синтетическом развороте: срез снизу, обычное поле, обложка."""

from __future__ import annotations

import cv2
import numpy as np

from research.edge_cut_pages.detect import Side, analyze

# Размер синтетического кадра (½ кадра пака-6 по порядку величины, но меньше — тест быстрый).
FRAME_H, FRAME_W = 1100, 1500
# Лист: левый верхний угол, правый нижний; яркость бумаги и коврика.
PAGE = (200, 120, 1300, 900)
PAPER, MAT = 225, 25
# Шаг строк, px.
LINE_STEP = 30
WORDS = "srednie normy postavok materialov i oborudovaniya po zakazam"


def synthetic_page(cut_bottom: bool, colour: tuple[int, int, int] | None = None) -> np.ndarray:
    """Разворот: коврик, светлый лист со строками текста; при ``cut_bottom`` нижний край листа режет последнюю строку.

    Args:
        cut_bottom: Последняя строка уходит за нижний край листа (кромка проходит по середине её глифов).
        colour: Цвет листа BGR (обложка) вместо бумаги.

    Returns:
        Картинка BGR.
    """
    img = np.full((FRAME_H, FRAME_W, 3), MAT, np.uint8)
    x0, y0, x1, y1 = PAGE
    img[y0:y1, x0:x1] = colour if colour is not None else (PAPER - 10, PAPER, PAPER)
    # Строки текста с полем 3 см; при срезе ещё одна строка, базовая линия которой ниже края листа на 7 px
    # (видна верхняя половина букв).
    baselines = list(range(y0 + 60, y1 - 200, LINE_STEP))
    if cut_bottom:
        baselines += list(range(y1 - 200, y1 - 20, LINE_STEP)) + [y1 + 7]
    for y in baselines:
        cv2.putText(img, WORDS, (x0 + 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    # Кромка листа: всё ниже края листа — коврик (срезает строку, заходящую за край).
    img[y1:] = MAT
    return img


def test_bottom_cut_found() -> None:
    """Строка, срезанная нижним краем листа, находится и помечается как «низ»."""
    result = analyze(synthetic_page(cut_bottom=True))
    bottoms = [line for line in result.lines if line.side is Side.BOTTOM]
    assert bottoms, result.lines
    # Срезанная строка лежит у нижнего края листа.
    assert all(abs(line.box[1] + line.box[3] - PAGE[3]) < 30 for line in bottoms)
    assert result.count(Side.TOP) == 0


def test_normal_margin_not_cut() -> None:
    """Текст с обычным полем до края листа срезом не считается."""
    result = analyze(synthetic_page(cut_bottom=False))
    assert [line for line in result.lines if line.side is not Side.SPINE] == []
    assert not result.is_cover


def test_cover_detected() -> None:
    """Голубой лист (обложка) — кадр-обложка, строки на цветной бумаге не считаются срезанными."""
    result = analyze(synthetic_page(cut_bottom=True, colour=(220, 200, 60)))
    assert result.is_cover
    assert result.count(Side.BOTTOM) == 0


def test_masks_on_request() -> None:
    """Маски для оверлея отдаются только по запросу и размера кадра."""
    assert analyze(synthetic_page(cut_bottom=True)).masks is None
    masks = analyze(synthetic_page(cut_bottom=True), keep_masks=True).masks
    assert masks is not None and masks.edge.shape == (FRAME_H, FRAME_W)
    assert masks.cut.any()
