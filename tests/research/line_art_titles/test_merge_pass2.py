"""Слияние с вердиктом DeepSeek (0 / 1 / 2 класса) и второй проход (заливка слов, снятие линеек) — на синтетике."""

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box
from research.line_art_titles.merge import merge_region
from research.line_art_titles.pass2 import fill_words, strip_rules, verdict_pass2

# Область: на полосе 600 dpi рамка (100, 100, 700, 400); вырезка 300 dpi с полем 10 px.
ROW = {"id": "p_0", "page": "p", "box": [100, 100, 700, 400], "dpi": 600, "crop_inner": [10, 10, 310, 160]}
FEATURES_TITLE = {"ds_has_non_text_block": "0", "ds_text_block_ink_share": "0.8", "height_mm": "12.7"}


def block(label: str, x0: int, y0: int, x1: int, y1: int) -> dict:
    """Блок DeepSeek в пикселях вырезки."""
    return {"label": label, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": ""}


def test_без_нетекстовых_блоков_правило_надписи() -> None:
    result = merge_region(ROW, FEATURES_TITLE, [block("sub_title", 10, 10, 300, 150)], np.zeros((800, 800), bool), [])
    assert result["verdict"] == "надпись" and result["objects"] == []


def test_один_класс_рамка_детектора() -> None:
    blocks = [block("image", 20, 20, 150, 150), block("image", 160, 20, 300, 150)]
    result = merge_region(ROW, FEATURES_TITLE, blocks, np.zeros((800, 800), bool), [])
    assert result["verdict"] is None and len(result["objects"]) == 1
    assert result["objects"][0] == {"class": "рисунок", "box": ROW["box"], "box_source": "детектор", "failed": []}


def test_два_класса_рамки_deepseek_в_пикселях_полосы() -> None:
    ink = np.zeros((800, 800), bool)
    blocks = [block("image", 20, 20, 150, 150), block("table", 160, 20, 300, 150)]
    result = merge_region(ROW, FEATURES_TITLE, blocks, ink, [])
    assert sorted(o["class"] for o in result["objects"]) == ["рисунок", "таблица"]
    drawing = next(o for o in result["objects"] if o["class"] == "рисунок")
    # (20 - 10) * 2 + 100 = 120: сдвиг на поле вырезки и масштаб 300 → 600 dpi.
    assert drawing["box"][:2] == [120, 120] and drawing["box_source"] == "DeepSeek"


def test_блок_в_поле_вне_области_не_считается() -> None:
    result = merge_region(ROW, FEATURES_TITLE, [block("image", 311, 0, 330, 170)], np.zeros((800, 800), bool), [])
    assert result["verdict"] == "надпись"


def test_заливка_стирает_слово_и_поле_но_не_буквицу() -> None:
    gray = np.full((200, 400), 255, np.uint8)
    cv2.rectangle(gray, (20, 20), (60, 140), 0, 6)  # «буквица» — крупное пятно
    cv2.putText(gray, "WORD", (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 4)
    cv2.rectangle(gray, (0, 190), (399, 199), 0, -1)  # краска в поле вне области
    binary, boxes = fill_words(gray, [10, 10, 390, 180], [{"x0": 90, "y0": 55, "x1": 250, "y1": 95}])
    assert (binary[55:95, 95:240] == 255).all(), "слово залито"
    assert (binary[20:140, 20:60] == 0).any(), "буквица осталась"
    assert (binary[190:, :] == 255).all(), "поле вне области — бумага"


def test_линейка_не_входит_в_рамку_объекта() -> None:
    binary = np.full((200, 800), 255, np.uint8)
    cv2.rectangle(binary, (40, 30), (90, 150), 0, 5)  # буквица
    cv2.line(binary, (20, 160), (780, 160), 0, 3)  # подчёркивающая линейка на всю ширину
    stripped = strip_rules(binary)
    assert (stripped[158:163, 300:700] == 255).all() and (stripped[30:150, 40:90] == 0).any()
    result = verdict_pass2(binary, [0, 0, 800, 200], [block("image", 10, 10, 790, 190)], [])
    box = Box(*result["objects"][0]["box"])
    assert box.x1 <= 100, "рамка обнимает буквицу, а не всю ширину линейки"


def test_текст_с_latex_или_долларами_становится_формулой_с_уточнённой_рамкой() -> None:
    ink = np.zeros((800, 800), bool)
    ink[380:420, 300:400] = True  # индекс формулы, вылезающий за низ рамки детектора (y1 = 400)
    text = {"label": "text", "x0": 20, "y0": 20, "x1": 300, "y1": 150, "text": "цена $x = 1$ руб."}
    result = merge_region(ROW, FEATURES_TITLE, [text], ink, [])
    assert [o["class"] for o in result["objects"]] == ["формула"]
    assert result["objects"][0]["box"][3] >= 420, "рамка формулы достроена до края пятна"


def test_крапины_стираются_а_штрих_остаётся() -> None:
    from research.line_art_titles.pass2 import despeckle

    binary = np.full((200, 400), 255, np.uint8)
    for x in range(20, 380, 20):
        cv2.circle(binary, (x, 50), 3, 0, -1)  # отточие: точки ~0.6 мм
    cv2.line(binary, (20, 120), (200, 180), 0, 4)  # штрих рисунка
    cleaned = despeckle(binary)
    assert (cleaned[40:60, :] == 255).all() and (cleaned[110:190, :] == 0).any()


def test_все_согласные_рамки_классики_входят_в_объект() -> None:
    binary = np.full((400, 800), 255, np.uint8)
    for x0 in (20, 300, 580):
        cv2.rectangle(binary, (x0, 40), (x0 + 180, 340), 0, 4)  # три коробки схемы
    classic = [(20, 40, 205, 345)]  # классика нашла только первую коробку
    result = verdict_pass2(binary, [0, 0, 800, 400], [block("image", 10, 30, 790, 360)], classic)
    box = Box(*result["objects"][0]["box"])
    assert box.x0 <= 20 and box.x1 >= 760, "объект — вся схема, а не одна согласная коробка"


def test_рамка_из_линеек_без_тела_не_объект_а_формула_второго_прохода_отвергается() -> None:
    binary = np.full((600, 400), 255, np.uint8)
    cv2.line(binary, (20, 20), (380, 20), 0, 3)
    cv2.line(binary, (380, 20), (380, 580), 0, 3)  # угол из двух линеек
    frame = verdict_pass2(binary, [0, 0, 400, 600], [block("image", 10, 10, 390, 590)], [])
    assert frame["verdict"] == "надпись"
    formula = {"label": "text", "x0": 10, "y0": 10, "x1": 390, "y1": 590, "text": r"## \( x^{2} \)"}
    cv2.circle(binary, (200, 300), 40, 0, 6)
    assert verdict_pass2(binary, [0, 0, 400, 600], [formula], [])["verdict"] == "надпись"


def test_блок_больше_области_накрывший_её_целиком_относится_к_ней() -> None:
    # Блок на всю вырезку с полем: внутри рамки области меньше половины его площади, но область — под ним целиком.
    big = block("equation", 0, 0, 330, 175)
    result = merge_region(ROW, FEATURES_TITLE, [big], np.zeros((800, 800), bool), [])
    assert [o["class"] for o in result["objects"]] == ["формула"]


def test_выдуманный_китайский_текст_с_latex_не_формула() -> None:
    fake = {
        "label": "sub_title",
        "x0": 20,
        "y0": 20,
        "x1": 300,
        "y1": 150,
        "text": r"## 1. 已知 \( f(x) = x^{2} \) ，求 \( f(2) \) 的值。",
    }
    result = merge_region(ROW, FEATURES_TITLE, [fake], np.zeros((800, 800), bool), [])
    assert result["objects"] == []


def test_одиночный_иероглиф_в_настоящей_формуле_не_мешает() -> None:
    real = {"label": "text", "x0": 20, "y0": 20, "x1": 300, "y1": 150, "text": r"\( B_{M}=\frac{y}{Q} \) ，此C. py6."}
    result = merge_region(ROW, FEATURES_TITLE, [real], np.zeros((800, 800), bool), [])
    assert [o["class"] for o in result["objects"]] == ["формула"]
