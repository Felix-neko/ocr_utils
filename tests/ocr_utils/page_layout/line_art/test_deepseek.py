"""DeepSeek-OCR-2 как классификатор кандидатов line art: разбор вывода, правила, решение по двум проходам — на синтетике, без модели."""

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.line_art.deepseek.decide import Outcome, decide, needs_pass2
from ocr_utils.page_layout.line_art.deepseek.parse import PROMPTS, parse
from ocr_utils.page_layout.line_art.deepseek.pass2 import despeckle, fill_words, strip_rules, verdict_pass2
from ocr_utils.page_layout.line_art.deepseek.rules import (
    CROP_DPI,
    Crop,
    block_class,
    crop_region,
    crop_to_page,
    is_hallucinated,
    on_region,
)

# Кандидат: на полосе 600 dpi рамка (100, 100, 700, 400); вырезка 300 dpi с полем 10 px.
CROP = Crop(Box(100, 100, 700, 400), 600, (10, 10, 310, 160), (320, 170))


def block(label: str, x0: int, y0: int, x1: int, y1: int, text: str = "") -> dict:
    """Блок DeepSeek в пикселях вырезки."""
    return {"label": label, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text}


def test_промпты_без_хвостового_пробела() -> None:
    assert all(not prompt.endswith(" ") for prompt in PROMPTS.values())


def test_parse_переводит_координаты_999_в_пиксели() -> None:
    raw = "<|ref|>sub_title<|/ref|><|det|>[[0, 0, 999, 499]]<|/det|>\n## ОТДЕЛ\n\n<|ref|>image<|/ref|><|det|>[[100, 500, 300, 999]]<|/det|>"
    elements = parse(raw, 1000, 200)
    assert [e["label"] for e in elements] == ["sub_title", "image"]
    assert (elements[0]["x1"], elements[0]["y1"]) == (1000, 100) and elements[0]["text"] == "## ОТДЕЛ"


def test_класс_блока_формула_по_latex_но_не_по_выдумке() -> None:
    assert block_class(block("text", 0, 0, 1, 1, r"\( B_{M}=\frac{y}{Q} \) ，此C. py6.")).value == "формула"
    assert block_class(block("sub_title", 0, 0, 1, 1, r"## 1. 已知 \( f(x) \) ，求 \( f(2) \) 的值。")) is None
    assert is_hallucinated("已知求的值") and not is_hallucinated("此C. py6.")
    assert block_class(block("image", 0, 0, 1, 1)).value == "рисунок"


def test_блок_относится_к_области_симметрично() -> None:
    assert on_region(block("equation", 0, 0, 320, 170), CROP.inner), "большой блок, накрывший область"
    assert not on_region(block("image", 311, 0, 320, 170), CROP.inner), "мелкий блок в поле"


def test_вырезка_и_обратный_перевод_в_пиксели_полосы() -> None:
    gray = np.full((1000, 1000), 255, np.uint8)
    crop_gray, crop = crop_region(gray, Box(100, 100, 700, 400), 600)
    assert crop.size == (crop_gray.shape[1], crop_gray.shape[0])
    back = crop_to_page({"x0": crop.inner[0], "y0": crop.inner[1], "x1": crop.inner[2], "y1": crop.inner[3]}, crop)
    assert abs(back.x0 - 100) <= 2 and abs(back.y1 - 400) <= 2


def test_второй_проход_нужен_при_тексте_и_при_рисунке_с_текстом() -> None:
    assert needs_pass2(CROP, [block("sub_title", 10, 10, 300, 150)])
    assert needs_pass2(CROP, [block("image", 10, 10, 200, 150), block("text", 200, 10, 310, 150)])
    assert not needs_pass2(CROP, [block("image", 10, 10, 300, 150)])
    assert not needs_pass2(CROP, [block("equation", 10, 10, 300, 150), block("text", 10, 10, 300, 150)])


def test_один_класс_рамка_детектора_без_второго_прохода() -> None:
    gray = np.full((170, 320), 255, np.uint8)
    result = decide(CROP, gray, [block("image", 20, 20, 300, 150)], np.zeros((800, 800), bool), [])
    assert result.outcome is Outcome.OBJECTS and result.objects[0]["box"] == list(CROP.box.as_tuple())


def test_рисунок_второго_прохода_заменяет_рамку_первого() -> None:
    gray = np.full((170, 320), 255, np.uint8)
    binary = np.full((170, 320), 255, np.uint8)
    cv2.circle(binary, (60, 80), 40, 0, 6)  # рисунок слева, текст справа залит
    first = [block("image", 10, 10, 310, 160), block("text", 150, 10, 310, 160)]
    second = [block("image", 10, 10, 140, 160)]
    result = decide(CROP, gray, first, np.zeros((800, 800), bool), [], second, binary)
    assert result.pass2_used and result.objects[0]["class"] == "рисунок"
    assert result.objects[0]["box"][2] < 500, "рамка — по рисунку, без текста справа"


def test_без_объектов_надпись_если_текст_над_краской_и_невысоко() -> None:
    gray = np.full((170, 320), 255, np.uint8)
    cv2.putText(gray, "TITLE", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 0, 8)
    result = decide(CROP, gray, [block("sub_title", 10, 10, 310, 160)], np.zeros((800, 800), bool), [])
    assert result.outcome is Outcome.TITLE
    empty = decide(CROP, np.full((170, 320), 255, np.uint8), [], np.zeros((800, 800), bool), [])
    assert empty.outcome is Outcome.UNCLEAR


def test_заливка_стирает_слово_и_поле_но_не_буквицу() -> None:
    gray = np.full((200, 400), 255, np.uint8)
    cv2.rectangle(gray, (20, 20), (60, 140), 0, 6)
    cv2.putText(gray, "WORD", (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 4)
    cv2.rectangle(gray, (0, 190), (399, 199), 0, -1)
    binary, _ = fill_words(gray, (10, 10, 390, 180), [{"x0": 90, "y0": 55, "x1": 250, "y1": 95}])
    assert (binary[55:95, 95:240] == 255).all() and (binary[20:140, 20:60] == 0).any()
    assert (binary[190:, :] == 255).all()


def test_крапины_и_линейки_снимаются_а_рамка_из_линеек_не_объект() -> None:
    binary = np.full((200, 800), 255, np.uint8)
    for x in range(20, 380, 20):
        cv2.circle(binary, (x, 50), 3, 0, -1)
    cv2.line(binary, (20, 160), (780, 162), 0, 3)  # слегка наклонная линейка
    assert (despeckle(binary)[40:60, :] == 255).all()
    assert (strip_rules(binary)[150:170, 100:700] == 255).all()
    frame = np.full((600, 400), 255, np.uint8)
    cv2.line(frame, (20, 20), (380, 20), 0, 3)
    cv2.line(frame, (380, 20), (380, 580), 0, 3)
    assert verdict_pass2(frame, (0, 0, 400, 600), [block("image", 10, 10, 390, 590)], []) == []


def test_разрешение_вырезок() -> None:
    assert CROP_DPI == 300
