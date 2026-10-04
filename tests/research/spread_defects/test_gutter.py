"""Тесты признака корешка: сгиб находится по тени между полосами, сжатие глифов у сгиба снижает шаг штрихов."""

import cv2
import numpy as np

from research.spread_defects.gutter import LEFT, RIGHT, edge_compression, find_fold, side_profile


def _text_block(height: int, width: int, seed: int) -> np.ndarray:
    """Полутоновый блок строк случайных латинских слов: бумага 235, краска ~20, ~450 dpi."""
    rng = np.random.default_rng(seed)
    block = np.full((height, width), 235, np.uint8)
    for y in range(60, height - 20, 62):
        x = 10
        while x < width - 60:
            word = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), rng.integers(3, 9)))
            cv2.putText(block, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 20, 3, cv2.LINE_AA)
            x += 30 * len(word) + 30
    return block


def _spread(squeeze_left: bool) -> tuple[np.ndarray, int]:
    """Разворот 3000×4400: две полосы текста, тень сгиба посередине; по желанию край левой полосы сжат вдвое.

    Returns:
        (полутоновый кадр float32, x сгиба).
    """
    height, width, fold = 3000, 4400, 2200
    frame = np.full((height, width), 235, np.uint8)
    left = _text_block(height, 1800, 1)
    right = _text_block(height, 1800, 2)
    if squeeze_left:
        # Последние 500 px строк сжимаются в 250 — как текст, уходящий в изгиб листа у корешка.
        # Слева добавляется бумага, чтобы сжатый край по-прежнему подходил к сгибу вплотную.
        squeezed = cv2.resize(left[:, -500:], (250, height), interpolation=cv2.INTER_AREA)
        left = np.hstack([np.full((height, 250), 235, np.uint8), left[:, :-500], squeezed])
    frame[:, fold - 40 - 1800 : fold - 40] = left
    frame[:, fold + 40 : fold + 40 + 1800] = right
    # Тень корешка: гауссова впадина яркости шириной ~30 px.
    xs = np.arange(width)
    shadow = 1 - 0.35 * np.exp(-((xs - fold) ** 2) / (2 * 15.0**2))
    return (frame.astype(np.float32) * shadow[None, :]), fold


def test_fold_found_in_shadow():
    """Сгиб — в тени между полосами, а не в межсловном пробеле."""
    gray, fold = _spread(False)
    line, cut = find_fold(gray)
    assert abs(np.polyval(line, gray.shape[0] / 2) - fold) < 25
    assert cut < 0.1


def test_squeezed_edge_has_low_ratio():
    """Сжатый вдвое край левой полосы даёт отношение шага заметно ниже 1, несжатая правая — около 1."""
    gray, _ = _spread(True)
    line, _ = find_fold(gray)
    left = np.nanmedian(edge_compression(side_profile(gray, line, LEFT)).edge_ratio)
    right = np.nanmedian(edge_compression(side_profile(gray, line, RIGHT)).edge_ratio)
    assert left < 0.75
    assert right > 0.85
