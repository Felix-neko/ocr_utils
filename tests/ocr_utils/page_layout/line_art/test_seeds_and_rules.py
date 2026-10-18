"""Line art v6: слияние затравок разных источников, затравки DeepSeek-OCR-2, правила A и B против ложных пятен — на синтетике."""

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box, HintBox
from ocr_utils.page_layout.line_art.deepseek.page_cache import blocks_from_elements
from ocr_utils.page_layout.line_art.detector import (
    LineArtInputs,
    detect_line_art,
    hints_from_deepseek,
    merge_seeds,
    seeds_from,
)
from ocr_utils.page_layout.line_art.features import analyse_gray, params_for_dpi
from ocr_utils.page_layout.surya.blocks import from_boxes

PAGE = (6733, 4165)  # страница пака-1 при 600 dpi
DPI = 600
DRAWING = Box(600, 2000, 3400, 5600)


def blank() -> np.ndarray:
    """Чистая белая полоса."""
    return np.full(PAGE, 255, np.uint8)


def with_drawing(page: np.ndarray) -> np.ndarray:
    """Связный штриховой рисунок (ломаная толщиной 7 px) в рамке ``DRAWING``."""
    rng = np.random.default_rng(0)
    points = rng.integers([DRAWING.x0, DRAWING.y0], [DRAWING.x1, DRAWING.y1], size=(60, 2))
    cv2.polylines(page, [points.reshape(-1, 1, 2).astype(np.int32)], False, 0, 7)
    return page


# --- Слияние затравок --------------------------------------------------------------------


def test_пересекающиеся_затравки_сливаются_в_одну_с_меткой_surya() -> None:
    surya = (Box(600, 2000, 3400, 5600), "surya:Figure")
    deepseek = (Box(700, 1900, 3500, 5500), "deepseek:image")
    merged = merge_seeds([deepseek, surya])
    assert merged == [(Box(600, 1900, 3500, 5600), "surya:Figure")]


def test_далёкие_затравки_не_сливаются() -> None:
    first = (Box(0, 0, 1000, 1000), "surya:Figure")
    second = (Box(2000, 2000, 3000, 3000), "deepseek:image")
    assert merge_seeds([first, second]) == [first, second]


def test_слияние_транзитивно() -> None:
    """Цепочка A~B~C — одна затравка, даже если A и C не пересекаются."""
    seeds = [
        (Box(0, 0, 1000, 1000), "deepseek:image"),
        (Box(400, 0, 1400, 1000), "surya:Figure"),
        (Box(800, 0, 1800, 1000), "tables:схема"),
    ]
    assert merge_seeds(seeds) == [(Box(0, 0, 1800, 1000), "tables:схема")]


def test_seeds_from_отдаёт_одну_затравку_на_рисунок() -> None:
    hints = [
        HintBox(*DRAWING.as_tuple(), source="surya:Figure", confidence=0.9),
        HintBox(*DRAWING.padded(40).as_tuple(), source="deepseek:image"),
    ]
    assert len(seeds_from(LineArtInputs(blank(), DPI, [], hints))) == 1


def test_surya_и_deepseek_на_одном_рисунке_дают_одну_область_с_обоими_источниками() -> None:
    hints = [
        HintBox(*DRAWING.as_tuple(), source="surya:Figure", confidence=0.9),
        HintBox(DRAWING.x0 + 100, DRAWING.y0 - 50, DRAWING.x1 + 80, DRAWING.y1 - 100, source="deepseek:image"),
    ]
    regions = detect_line_art(LineArtInputs(with_drawing(blank()), DPI, [], hints))
    assert len(regions) == 1
    assert {"surya:Figure", "deepseek:image"} <= set(regions[0].info["sources"])


# --- Затравки DeepSeek ------------------------------------------------------------------


def test_затравки_deepseek_только_из_блоков_image() -> None:
    blocks = from_boxes(
        [
            ("image", 1.0, Box(10, 10, 200, 200)),
            ("text", 1.0, Box(10, 220, 200, 260)),
            ("table", 1.0, Box(10, 300, 200, 500)),
            ("figure_title", 1.0, Box(10, 510, 200, 530)),
        ],
        833,
        1470,
    )
    hints = hints_from_deepseek(blocks)
    assert [h.source for h in hints] == ["deepseek:image"]
    assert hints[0].as_tuple() == (10, 10, 200, 200)


def test_без_разбора_deepseek_затравок_нет() -> None:
    assert hints_from_deepseek(None) == []


def test_элементы_воркера_в_блоки_кэша() -> None:
    elements = [
        {"x0": 50, "y0": 296, "x1": 400, "y1": 710, "label": "image", "text": ""},
        {"x0": 446, "y0": 300, "x1": 900, "y1": 706, "label": "image", "text": ""},  # вылез за кадр
        {"x0": 10, "y0": 10, "x1": 10, "y1": 50, "label": "text", "text": "пусто"},  # нулевая ширина
    ]
    blocks = blocks_from_elements(elements, 833, 1470)
    assert [(b.label, b.box.as_tuple(), b.confidence) for b in blocks.blocks] == [
        ("image", (50, 296, 400, 710), 1.0),
        ("image", (446, 300, 833, 706), 1.0),
    ]


# --- Правила A и B ------------------------------------------------------------------------


def test_правило_b_открытая_ломаная_из_линеек_не_находка() -> None:
    """Обломок тонкой рамки выходных данных: вертикаль и короткая горизонталь снизу (1977/01 0050_1L, ×4 к 150 dpi)."""
    page = blank()
    cv2.line(page, (800, 3400), (800, 5580), 0, 10)
    cv2.line(page, (800, 5580), (1360, 5580), 0, 10)
    findings = analyse_gray(page, params_for_dpi(DPI))
    assert findings.boxes == []
    assert findings.dropped.get("открытая ломаная из линеек") == 1


def test_правило_b_рисунок_не_задевает() -> None:
    findings = analyse_gray(with_drawing(blank()), params_for_dpi(DPI))
    assert len(findings.boxes) == 1
    assert "открытая ломаная из линеек" not in findings.dropped


def test_правило_a_слипшаяся_строка_заголовка_под_текстом_не_кандидат() -> None:
    """Пятно высотой меньше полутора строк, накрытое текстовым блоком, — слипшиеся буквы, а не штрих."""
    page = blank()
    word = (1000, 1000, 1436, 1100)
    cv2.rectangle(page, word[:2], word[2:], 0, 20)
    params = params_for_dpi(DPI)
    assert len(analyse_gray(page, params).candidates) == 1, "без текстового блока пятно проходит ворота"
    findings = analyse_gray(page, params, text_boxes=[(900, 950, 2000, 1150)])
    assert findings.candidates == []
    assert findings.dropped.get("слипшаяся строка заголовка") == 1


def test_правило_a_высокое_пятно_под_текстом_остаётся() -> None:
    """Логотип рубрики выше полутора строк накрыт текстовым блоком surya, но это штрих."""
    page = blank()
    cv2.rectangle(page, (1000, 1000), (1600, 1400), 0, 20)
    findings = analyse_gray(page, params_for_dpi(DPI), text_boxes=[(900, 950, 2000, 1450)])
    assert len(findings.candidates) == 1
