"""Разбор grounding-вывода DeepSeek-OCR-2 и промпты — без зависимостей от ``ocr_utils``: модуль берёт и воркер vLLM в своём окружении.

Вывод модели в режиме ``<|grounding|>`` — теги ``<|ref|>метка<|/ref|><|det|>[[x1, y1, x2, y2], …]<|/det|>``,
за которыми идёт текст элемента. Координаты нормированы на 0–999 по каждой оси картинки (так их
переводит и штатная отрисовка модели: ``x / 999 * ширина``).

Промпты подаются **без хвостового пробела**: HF remote code его срезает сам (``format_messages``:
``strip()``), а в vLLM пробел — лишний токен после «.», на котором модель уходит в мусор
(reports/line_art_titles.md, раздел про vLLM).
"""

from __future__ import annotations

import ast
import re

MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"

# Промпты режима grounding. ``markdown`` — штатный промпт документа из карточки модели (рамки
# блоков вёрстки с метками title/text/image …); ``ocr`` — промпт «прочих картинок» из первой
# DeepSeek-OCR (у второй версии он даёт рамки слов и строк; кириллицу пишет латинскими двойниками).
PROMPTS = {
    "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
    "ocr": "<image>\n<|grounding|>OCR this image.",
}

# Один элемент вывода: метка, список рамок и текст до следующего тега.
ELEMENT = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>(.*?)(?=<\|ref\|>|\Z)", re.DOTALL)

# Служебные теги модели, которые в тексте элемента не нужны.
TAGS = re.compile(r"<\|[^|]*\|>")

# Шкала нормированных координат модели.
SCALE = 999


def parse(raw: str, width: int, height: int) -> list[dict]:
    """Разобрать вывод модели в элементы с рамками в пикселях картинки.

    Args:
        raw: Сырой вывод модели с тегами grounding.
        width: Ширина картинки, px.
        height: Высота картинки, px.

    Returns:
        По словарю на рамку: ``x0, y0, x1, y1`` (px, ``x1``/``y1`` — за последним пикселем),
        ``label`` — содержимое ``<|ref|>`` (тип блока или сам текст, смотря по промпту),
        ``text`` — текст после тега до следующего элемента (без служебных тегов).
        Элемент с несколькими рамками даёт несколько словарей с одним текстом.
    """
    elements = []
    for label, boxes, text in ELEMENT.findall(raw):
        try:
            coordinates = ast.literal_eval(boxes.strip())
        except (ValueError, SyntaxError):
            continue
        clean = TAGS.sub("", text).strip()
        for box in coordinates:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            elements.append(
                {
                    "x0": int(round(min(x0, x1) / SCALE * width)),
                    "y0": int(round(min(y0, y1) / SCALE * height)),
                    "x1": int(round(max(x0, x1) / SCALE * width)),
                    "y1": int(round(max(y0, y1) / SCALE * height)),
                    "label": label.strip(),
                    "text": clean,
                }
            )
    return elements


__all__ = ["MODEL_ID", "PROMPTS", "parse"]
