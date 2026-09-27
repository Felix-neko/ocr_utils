"""Признаки стенда «надпись или рисунок» на синтетике: буквы против штриха."""

import cv2
import numpy as np

from research.line_art_titles.features import is_title_like, region_features

DPI = 300


def _row(width: int, height: int) -> dict:
    """Строка regions.jsonl для вырезки без поля."""
    return {"id": "x", "page": "p", "info": {"kind": "штрих", "sources": ["ink"]}, "crop_inner": [0, 0, width, height]}


def test_надпись_из_букв_признаётся_надписью() -> None:
    gray = np.full((140, 900), 255, np.uint8)
    cv2.putText(gray, "SNABZHENIE", (10, 110), cv2.FONT_HERSHEY_DUPLEX, 3.5, 0, 9)
    words = {"psm11": [{"x0": 10, "y0": 20, "x1": 880, "y1": 115, "conf": 40.0, "text": "СНАБЖЕНИЕ", "line": "1.1.1"}]}
    features = region_features(_row(900, 140), words, gray, DPI)
    assert features["letter_cc_share"] > 0.9 and features["weak_text_ink_share"] > 0.9
    assert features["text_ink_share"] == 0.0, "уверенность 40 ниже порога уверенного слова"
    assert is_title_like(features)


def test_ломаная_без_слов_не_надпись() -> None:
    gray = np.full((900, 900), 255, np.uint8)
    points = np.random.default_rng(0).integers(0, 900, size=(40, 2)).astype(np.int32)
    cv2.polylines(gray, [points.reshape(-1, 1, 2)], False, 0, 4)
    features = region_features(_row(900, 900), {"psm11": [], "psm6": []}, gray, DPI)
    assert features["weak_text_ink_share"] == 0.0
    assert not is_title_like(features)
