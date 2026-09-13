"""Распознавание вырезок ячеек через PaddleOCR. Запускается в отдельном окружении.

Вход — папка с PNG, выход — JSON вида ``{"имя.png": {"lines": [...], "confidence": 0.93}}``.
Никакой предобработки: вырезки уже выпрямлены и дополнены полями основным пакетом.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="PaddleOCR по папке с вырезками ячеек")
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lang", default="ru")
    arguments = parser.parse_args()

    from paddleocr import PaddleOCR

    # ``enable_mkldnn=False`` — не украшательство: сборка paddlepaddle под CPU падает на
    # oneDNN с «ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute...]»
    # и возвращает пустой результат по каждой вырезке. Без oneDNN та же модель считает.
    engine = PaddleOCR(
        lang=arguments.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    payload: dict[str, dict] = {}
    for path in sorted(arguments.in_dir.glob("*.png")):
        started = time.time()
        try:
            result = engine.predict(str(path))
        except Exception as error:  # одна битая вырезка не должна ронять пакет
            payload[path.name] = {"lines": [], "confidence": 0.0, "note": str(error)[:200]}
            continue
        lines: list[str] = []
        scores: list[float] = []
        for page in result or []:
            texts = page.get("rec_texts") or []
            confidences = page.get("rec_scores") or []
            lines.extend(str(text) for text in texts)
            scores.extend(float(value) for value in confidences)
        payload[path.name] = {
            "lines": lines,
            "confidence": (sum(scores) / len(scores)) if scores else 0.0,
            "seconds": time.time() - started,
        }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
