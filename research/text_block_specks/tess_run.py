"""Tesseract по парам вырезок судьи (``<judge>/crops``) → ``<judge>/tesseract.jsonl`` ({"id", "text"}); пул процессов."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from research.text_block_specks.judge import tesseract_text


def run(path: Path) -> dict:
    """Текст одной вырезки."""
    return {"id": path.stem, "text": tesseract_text(path)}


if __name__ == "__main__":
    judge = Path(sys.argv[1])
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    crops = sorted((judge / "crops").glob("*.png"))
    with ProcessPoolExecutor(jobs) as pool, (judge / "tesseract.jsonl").open("w") as out:
        for record in pool.map(run, crops):
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
