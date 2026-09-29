"""Судья tesseract (разностный): текст строки с кандидатом и с закрашенным кандидатом, ``rus --psm 7``; вердикт — по краю текста (:mod:`interpret`)."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from research.edge_marks.candidates import load


def text_of(path: Path) -> str:
    """Текст строки по tesseract (русский, одна строка)."""
    result = subprocess.run(["tesseract", str(path), "-", "-l", "rus", "--psm", "7"], capture_output=True, text=True, check=False)
    return result.stdout.strip()


def judge_one(args: tuple[Path, str]) -> dict:
    """Пара текстов по кандидату ``cid`` и время на пару."""
    cand_dir, cid = args
    started = time.monotonic()
    with_text = text_of(cand_dir / "b" / f"{cid}.png")
    erased = text_of(cand_dir / "b_erased" / f"{cid}.png")
    return {"id": cid, "scale": "b", "kind": "diff", "with": with_text, "erased": erased, "seconds": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cand-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    started = time.monotonic()
    tasks = [(args.cand_dir, c.id) for c in load(args.cand_dir)]
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"meta": {"load_seconds": time.monotonic() - started, "device": "cpu"}}) + "\n")
        with ThreadPoolExecutor(args.jobs) as pool:
            for record in pool.map(judge_one, tasks):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
