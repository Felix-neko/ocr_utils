"""Судья surya (детектор строк + распознаватель): полоса (c) — рамки символов и слов OCR накрывают кандидата; строка (b) — разностный по вырезке строки с кандидатом и без."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from research.edge_marks.candidates import load

# Партия распознавателя: строк полосы — сотни, на 16 ГБ хватает с запасом.
BATCH = 64


def boxes_of(result) -> tuple[list, list]:
    """Рамки символов (без пробелов) и слов из ответа surya по одной картинке.

    Args:
        result: ``OCRResult`` surya.

    Returns:
        ``(символы, слова)`` — списки ``[x0, y0, x1, y1, текст, уверенность]``.
    """
    chars, words = [], []
    for line in result.text_lines:
        for char in line.chars or []:
            if char.text.strip():
                chars.append([*map(float, char.bbox), char.text, float(char.confidence or 0.0)])
        for word in line.words or []:
            if word.text.strip():
                words.append([*map(float, word.bbox), word.text, float(word.confidence or 0.0)])
    return chars, words


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cand-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor

    recognizer, detector = RecognitionPredictor(FoundationPredictor()), DetectionPredictor()
    candidates = load(args.cand_dir)
    handle = args.out.open("w", encoding="utf-8")
    handle.write(json.dumps({"meta": {"load_seconds": time.monotonic() - started, "device": "cuda"}}) + "\n")
    by_page: dict[str, list] = {}
    for c in candidates:
        by_page.setdefault(c.key, []).append(c)
    for key, items in by_page.items():
        # (c) Полоса целиком: детектор строк + распознаватель с рамками символов и слов.
        tick = time.monotonic()
        page = Image.open(args.cand_dir / "pages" / f"{key}.png").convert("RGB")
        result = recognizer([page], det_predictor=detector, math_mode=False, return_words=True,
                            recognition_batch_size=BATCH)[0]  # fmt: skip
        chars, words = boxes_of(result)
        share = (time.monotonic() - tick) / len(items)
        for c in items:
            for scale, boxes in (("c_chars", chars), ("c_words", words)):
                record = {"id": c.id, "scale": scale, "kind": "boxes", "boxes": boxes, "seconds": share}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            # (b) Строка: вырезка целиком — одна строка (рамка на всю вырезку), с кандидатом и без.
            tick = time.monotonic()
            crops = [Image.open(args.cand_dir / f / f"{c.id}.png").convert("RGB") for f in ("b", "b_erased")]
            lines = recognizer(crops, bboxes=[[[0, 0, *crop.size]] for crop in crops], math_mode=False)
            texts = [" ".join(line.text for line in item.text_lines) for item in lines]
            record = {"id": c.id, "scale": "b", "kind": "diff", "with": texts[0], "erased": texts[1],
                      "seconds": time.monotonic() - tick}  # fmt: skip
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
    handle.close()


if __name__ == "__main__":
    main()
