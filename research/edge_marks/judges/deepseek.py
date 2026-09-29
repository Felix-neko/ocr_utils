"""Судья DeepSeek-OCR-2 (vLLM, grounding): вырезка (a) — рамки слов накрывают кандидата; строка (b) — разностный по промптам ``free`` и ``ocr``; полоса (c) — рамки слов grounding по всей странице."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from research.edge_marks.candidates import CROP_A_XH, MARGIN_PX, UPSCALE, load

PROJECT = Path(__file__).resolve().parents[3]
VLLM_ENV = PROJECT / "ocr_utils" / "page_layout" / "line_art" / "deepseek" / "vllm_env"
RUNNER = PROJECT / "research" / "text_block_specks" / "deepseek_run.py"
# Время по промптам из журнала воркера: «free: 376 вырезок за 123.4 с».
TIMING = re.compile(r"^(\w+): (\d+) вырезок за ([\d.]+) с", re.M)


def run_worker(jobs: list[dict], out_dir: Path, prompts: str, max_new_tokens: int) -> dict[str, float]:
    """Прогнать воркер vLLM по заданиям; вернуть секунды генерации по промптам (без загрузки модели).

    Args:
        jobs: ``[{"id", "crop"}]``.
        out_dir: Папка выхода воркера.
        prompts: Промпты через запятую.
        max_new_tokens: Предел длины ответа.

    Returns:
        ``промпт → секунды``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Ответы уже есть по всем промптам и на все задания (повторная сборка результатов) — воркер не гоняем,
    # время берём из прошлого журнала.
    done = [out_dir / f"{name}.jsonl" for name in prompts.split(",")]
    log = out_dir / "worker.log"
    if log.exists() and all(p.exists() and len(p.read_text().splitlines()) == len(jobs) for p in done):
        return {name: float(seconds) for name, _, seconds in TIMING.findall(log.read_text())}
    jobs_file = out_dir / "jobs.jsonl"
    jobs_file.write_text("\n".join(json.dumps(job) for job in jobs) + "\n")
    result = subprocess.run(
        ["uv", "run", "--project", str(VLLM_ENV), "python", str(RUNNER), "--jobs", str(jobs_file), "--out-dir", str(out_dir),
         "--prompts", prompts, "--max-new-tokens", str(max_new_tokens)],
        capture_output=True, text=True, check=False, cwd=str(PROJECT),
    )  # fmt: skip
    (out_dir / "worker.log").write_text(result.stdout + result.stderr)
    return {name: float(seconds) for name, _, seconds in TIMING.findall(result.stdout)}


def element_text(element: dict) -> str:
    """Текст элемента grounding: у промпта ``ocr`` он лежит в ``label`` (тег ``<|ref|>``), ``text`` пуст."""
    return element.get("text") or element.get("label", "")


def read(path: Path) -> dict[str, dict]:
    """Ответы воркера по ``id``."""
    return {json.loads(line)["id"]: json.loads(line) for line in path.read_text().splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cand-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidates = load(args.cand_dir)
    work = args.out.parent / f"{args.out.stem}_work"
    crops = [{"id": f"{c.id}|a", "crop": str(args.cand_dir / "a" / f"{c.id}.png")} for c in candidates]
    crops += [{"id": f"{c.id}|bw", "crop": str(args.cand_dir / "b" / f"{c.id}.png")} for c in candidates]
    crops += [{"id": f"{c.id}|be", "crop": str(args.cand_dir / "b_erased" / f"{c.id}.png")} for c in candidates]
    timing = run_worker(crops, work / "crops", "free,ocr", 256)
    pages = sorted({c.key for c in candidates})
    page_jobs = [{"id": key, "crop": str(args.cand_dir / "pages" / f"{key}.png")} for key in pages]
    timing_pages = run_worker(page_jobs, work / "pages", "ocr", 6000)
    free, ocr = read(work / "crops" / "free.jsonl"), read(work / "crops" / "ocr.jsonl")
    page_ocr = read(work / "pages" / "ocr.jsonl")
    per_crop = {name: seconds / max(1, len(crops)) for name, seconds in timing.items()}
    per_page = timing_pages.get("ocr", 0.0) / max(1, len(pages))
    with args.out.open("w", encoding="utf-8") as handle:
        meta = {"load_seconds": None, "timing": {**timing, **{f"pages_{k}": v for k, v in timing_pages.items()}}}
        handle.write(json.dumps({"meta": meta}) + "\n")
        for c in candidates:
            # (a) рамки слов grounding на вырезке → в пиксели полосы: вырезка = (рамка − поле) × UPSCALE + кайма.
            pad = int(CROP_A_XH * c.x_h)
            origin = (c.box[0] - pad, c.box[1] - pad)
            boxes = []
            for element in ocr.get(f"{c.id}|a", {}).get("elements", []):
                x0, y0 = (element["x0"] - MARGIN_PX) / UPSCALE + origin[0], (
                    element["y0"] - MARGIN_PX
                ) / UPSCALE + origin[1]
                x1, y1 = (element["x1"] - MARGIN_PX) / UPSCALE + origin[0], (
                    element["y1"] - MARGIN_PX
                ) / UPSCALE + origin[1]
                boxes.append([x0, y0, x1, y1, element_text(element)])
            record = {
                "id": c.id,
                "scale": "a",
                "kind": "boxes",
                "boxes": boxes,
                "seconds": per_crop.get("ocr", 0.0),
                "variant": "ocr",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            # (b) разностный по двум промптам.
            for name, answers in (("free", free), ("ocr", ocr)):
                with_text = answers.get(f"{c.id}|bw", {}).get("raw", "")
                erased = answers.get(f"{c.id}|be", {}).get("raw", "")
                if name == "ocr":
                    with_text = " ".join(element_text(e) for e in answers.get(f"{c.id}|bw", {}).get("elements", []))
                    erased = " ".join(element_text(e) for e in answers.get(f"{c.id}|be", {}).get("elements", []))
                record = {"id": c.id, "scale": f"b_{name}", "kind": "diff", "with": with_text, "erased": erased,
                          "seconds": 2 * per_crop.get(name, 0.0)}  # fmt: skip
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            # (c) рамки слов по полосе целиком (пиксели полосы).
            boxes = [
                [e["x0"], e["y0"], e["x1"], e["y1"], element_text(e)]
                for e in page_ocr.get(c.key, {}).get("elements", [])
            ]
            record = {"id": c.id, "scale": "c", "kind": "boxes", "boxes": boxes, "seconds": per_page}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
