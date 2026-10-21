"""Тесты признаков ксерокопии: цвет бумаги и крап по плиткам, сводка по сторонам сгиба, сравнение с выпуском."""

import cv2
import numpy as np
import pandas as pd

from research.spread_defects.photocopy import copy_regions, copy_tiles, relative_features, side_summary, speck_mask


def _page(width: int, paper: tuple[int, int, int], specks: int, seed: int) -> np.ndarray:
    """Полоса RGB: строки латинских слов краской ~20 на бумаге цвета ``paper`` и ``specks`` точек крапа в межстрочье."""
    rng = np.random.default_rng(seed)
    page = np.empty((2048, width, 3), np.uint8)
    page[:] = paper
    for y in range(60, 2048 - 20, 62):
        x = 10
        while x < width - 60:
            word = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), rng.integers(3, 9)))
            cv2.putText(page, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3, cv2.LINE_AA)
            x += 26 * len(word) + 30
    # Крап — точки 2×2 px посередине межстрочья (y ≡ 80 по шагу строк 62 далеко от букв).
    for _ in range(specks):
        y = 62 * int(rng.integers(1, 30)) + 18
        x = int(rng.integers(20, width - 20))
        page[y : y + 2, x : x + 2] = 60
    return page


def test_speck_mask_finds_isolated_dots_only() -> None:
    """Одиночная точка вдали от буквы — крап; точка вплотную к букве — нет."""
    dark = np.zeros((200, 200), np.uint8)
    dark[50:90, 50:60] = 1  # «буква»
    dark[100:102, 150:152] = 1  # крап
    dark[92:94, 52:54] = 1  # точка у буквы
    ink = dark.copy()
    mask = speck_mask(ink, dark)
    assert mask[100, 150] and not mask[92, 52] and not mask[60, 55]


def test_copy_tiles_and_sides() -> None:
    """Левая полоса — желтоватая бумага без крапа, правая — белая холодная с крапом: b* и крап различаются."""
    book = _page(2048, (232, 222, 190), specks=0, seed=1)
    copy = _page(2048, (238, 240, 245), specks=600, seed=2)
    gap = np.full((2048, 512, 3), 120, np.uint8)
    frame = np.hstack([book, gap, copy])
    tiles = copy_tiles(frame)
    summary = side_summary(tiles, fold_x=2048 + 256)
    assert summary["L_tiles"] >= 8 and summary["R_tiles"] >= 8
    assert summary["L_paper_b"] > summary["R_paper_b"] + 10
    assert summary["R_speck"] > summary["L_speck"]


def test_relative_features_issue_z() -> None:
    """Полоса с b* ниже медианы выпуска получает отрицательный z, остальные — около нуля."""
    pages = pd.DataFrame(
        {
            "key": [f"k{i}" for i in range(10)],
            "issue": "1965/12",
            "index": range(10),
            "side": "L",
            **{name: np.linspace(0, 0.1, 10) for name in ("paper_l", "paper_a", "grain", "speck")},
            **{name: np.linspace(0, 0.1, 10) for name in ("ink_refl", "midtone", "unevenness")},
            "paper_b": [10.0, 10.2, 9.9, 10.1, 10.0, 9.8, 10.1, 10.0, 2.0, 10.2],
        }
    )
    out = relative_features(pages)
    assert out.paper_b_iz.iloc[8] < -5
    assert abs(out.paper_b_iz.iloc[0]) < 1
    assert out.paper_b_pair.isna().all()
    # Кадры 0–5 и 4–9 — края выпуска из 10 кадров (по 6 с каждой стороны).
    assert out.edge.all()


def test_copy_regions_hysteresis() -> None:
    """Участок — ядро выше строгого порога, расширенное по мягкому; провал в одну полосу участок не рвёт."""
    scores = [0.01, 0.1, 0.9, 0.02, 0.95, 0.2, 0.01, 0.01, 0.3]
    scored = pd.DataFrame(
        {
            "pack": "pack3",
            "issue": "1965/12",
            "index": [k // 2 for k in range(9)],
            "key": [f"IMG_{k // 2:04d}.jpg" for k in range(9)],
            "side": ["L", "R"] * 4 + ["L"],
            "score": scores,
            "page": range(9),
            "edge": False,
        }
    )
    regions = copy_regions(scored, threshold=0.5, extend=0.05)
    # Полосы 1–5 (0.1 … 0.2, провал 0.02 затянут); полоса 8 (0.3) без ядра — не участок.
    assert len(regions) == 1
    assert regions.pages.iloc[0] == 5 and regions.page_from.iloc[0] == 1 and regions.page_to.iloc[0] == 5
