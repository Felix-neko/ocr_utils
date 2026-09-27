"""Разбор grounding-вывода DeepSeek-OCR-2 и признаки по нему — на синтетике, без модели."""

import numpy as np

from research.line_art_titles.deepseek_worker import parse
from research.line_art_titles.features import deepseek_features, ink_blobs

DPI = 300


def test_parse_переводит_координаты_999_в_пиксели_и_берёт_текст_до_следующего_тега() -> None:
    raw = (
        "<|ref|>sub_title<|/ref|><|det|>[[0, 0, 999, 499]]<|/det|>\n## ОФИЦИАЛЬНЫЙ\nОТДЕЛ\n\n"
        "<|ref|>image<|/ref|><|det|>[[100, 500, 300, 999], [400, 500, 600, 999]]<|/det|>"
    )
    elements = parse(raw, 1000, 200)
    assert [e["label"] for e in elements] == ["sub_title", "image", "image"]
    assert (elements[0]["x0"], elements[0]["y0"], elements[0]["x1"], elements[0]["y1"]) == (0, 0, 1000, 100)
    assert elements[0]["text"] == "## ОФИЦИАЛЬНЫЙ\nОТДЕЛ"
    assert elements[2]["x0"] == 400 and elements[2]["y1"] == 200


def test_parse_пропускает_битые_координаты() -> None:
    assert parse("<|ref|>text<|/ref|><|det|>[[1, 2, 3<|/det|>abc", 100, 100) == []


def test_признаки_deepseek_делят_краску_на_слова_текст_и_рисунок() -> None:
    gray = np.full((200, 400), 255, np.uint8)
    gray[20:60, 20:180] = 0  # «слово» слева
    gray[120:180, 220:380] = 0  # «рисунок» справа
    blobs = ink_blobs(gray, [0, 0, 400, 200], DPI)
    words = [{"x0": 18, "y0": 18, "x1": 182, "y1": 62, "label": "OTAEA", "text": ""}]
    markdown = [
        {"x0": 10, "y0": 10, "x1": 190, "y1": 70, "label": "sub_title", "text": "## ОТДЕЛ"},
        {"x0": 210, "y0": 110, "x1": 390, "y1": 190, "label": "image", "text": ""},
    ]
    features = deepseek_features(blobs, markdown, words)
    share_word = 160 * 40 / (160 * 40 + 160 * 60)
    assert abs(features["ds_word_ink_share"] - share_word) < 0.01
    assert features["ds_letter_word_ink_share"] == features["ds_word_ink_share"]
    assert abs(features["ds_image_block_ink_share"] - (1 - share_word)) < 0.01
    assert features["ds_block_labels"] == "image,sub_title" and features["ds_text"] == "## ОТДЕЛ"
