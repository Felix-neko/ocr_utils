"""Буквы под переплётом через DeepSeek-OCR-2 (vLLM, локально): колонка у сгиба кусками, «Free OCR.», словарь по крайним словам.

Подбор промпта (2026-10-04, ЭСХ 1979/12): из «Free OCR.», «OCR this image.», «Free OCR. Keep cut-off words as they
are.», «Free OCR. Mark illegible letters with #.», русской инструкции и grounding-режима годится только штатный
«Free OCR.»: остальные возвращают пустоту, повторы или сочиняют связный текст (дописывая обрубки), grounding
выдаёт кириллицу латиницей. «Free OCR.» обрубки в основном оставляет как есть («величить», «словных», «льств») —
это и нужно словарю; иногда достраивает («дей» → «дней»), что лишь снижает полноту.

Кэш: вырезки PNG и ответы ``free.jsonl`` лежат в ``<out_root>/deepseek_fold/``; повторный запуск читает только
новые кадры.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from research.spread_defects.frames import Frame, read_rgb
from research.spread_defects.gutter import find_fold
from research.spread_defects.lines import LEFT, RIGHT, find_glyphs, fold_distance, side_lines
from research.spread_defects.truncated_words import EdgeCheck, check_lines

PROJECT = Path(__file__).resolve().parents[2]
VLLM_ENV = PROJECT / "ocr_utils" / "page_layout" / "line_art" / "deepseek" / "vllm_env"
RUNNER = Path(__file__).resolve().parent / "deepseek_runner.py"
# Тело полосы по высоте, доли кадра; высота куска и перекрытие, px полного кадра (кусок ~12–14 строк).
TOP, BOTTOM = 0.07, 0.94
CHUNK, OVERLAP = 760, 60
# Заступ за сгиб и запас за дальним краем колонки, в высотах буквы.
OVER_FOLD, FAR_PAD = 1.5, 1.5


def column_width(gray: np.ndarray, fold: np.ndarray, side: str) -> float:
    """Ширина колонки у сгиба: медиана дальнего края строк от сгиба, px.

    Args:
        gray: Полутоновый кадр float32.
        fold: Прямая сгиба.
        side: ``L``/``R``.

    Returns:
        Ширина колонки с запасом; 1600, если строк не нашлось.
    """
    glyphs = find_glyphs(gray)
    if not np.isfinite(glyphs.glyph_h):
        return 1600.0
    _, far = fold_distance(glyphs, fold, side)
    lines = side_lines(glyphs, fold, side)
    if len(lines) < 5:
        return 1600.0
    extent = float(np.median([far[line.glyphs].max() for line in lines]))
    return extent + FAR_PAD * glyphs.glyph_h


def make_crops(frame: Frame, crop_dir: Path) -> list[dict]:
    """Куски колонки у сгиба обеих полос кадра в PNG (CPU, воркер пула).

    Args:
        frame: Кадр.
        crop_dir: Папка кэша вырезок кадра.

    Returns:
        Задания воркера: ``{"id": "<ключ>|<сторона>|<номер куска>", "crop": путь}``; уже готовые вырезки не пишутся.
    """
    rgb = read_rgb(frame.path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grayf = gray.astype(np.float32)
    fold, _ = find_fold(grayf)
    height, width = gray.shape
    jobs = []
    for side in (LEFT, RIGHT):
        reach = column_width(grayf, fold, side)
        glyph_h = 25.0
        n = 0
        for y0 in range(int(TOP * height), int(BOTTOM * height) - CHUNK // 2, CHUNK - OVERLAP):
            y1 = min(int(BOTTOM * height), y0 + CHUNK)
            fx = float(np.polyval(fold, (y0 + y1) / 2))
            if side == RIGHT:
                x0, x1 = int(fx - OVER_FOLD * glyph_h), int(fx + reach)
            else:
                x0, x1 = int(fx - reach), int(fx + OVER_FOLD * glyph_h)
            x0, x1 = max(0, x0), min(width, x1)
            path = crop_dir / f"{side}_{n:02d}.png"
            if not path.exists():
                crop_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), gray[y0:y1, x0:x1])
            jobs.append({"id": f"{frame.key}|{side}|{n:02d}", "crop": str(path)})
            n += 1
    return jobs


def run_deepseek(jobs: list[dict], out_dir: Path) -> None:
    """Прогнать воркер vLLM по заданиям, которых ещё нет в ``free.jsonl`` (ответы дописываются).

    Args:
        jobs: Задания ``{"id", "crop"}``.
        out_dir: Папка ответов (``free.jsonl``, ``worker.log``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    answers = out_dir / "free.jsonl"
    done = set(read_answers(answers))
    todo = [job for job in jobs if job["id"] not in done]
    if not todo:
        return
    jobs_file = out_dir / "jobs_todo.jsonl"
    jobs_file.write_text("\n".join(json.dumps(job, ensure_ascii=False) for job in todo) + "\n")
    result = subprocess.run(
        ["uv", "run", "--project", str(VLLM_ENV), "python", str(RUNNER), "--jobs", str(jobs_file), "--out-dir",
         str(out_dir), "--prompts", "free", "--max-new-tokens", "1024"],
        capture_output=True, text=True, check=False, cwd=str(PROJECT),
    )  # fmt: skip
    with (out_dir / "worker.log").open("a") as log:
        log.write(result.stdout + result.stderr)


def read_answers(path: Path) -> dict[str, str]:
    """Ответы воркера: ``id`` → текст.

    Args:
        path: ``free.jsonl``.

    Returns:
        Словарь; пустой, если файла нет.
    """
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            out[record["id"]] = record.get("text") or record.get("raw") or ""
    return out


def frame_check(answers: dict[str, str], key: str, side: str) -> EdgeCheck:
    """Словарная проверка крайних у сгиба слов одной полосы кадра по ответам на её куски.

    Строки кусков склеиваются по порядку (перекрытие кусков даёт редкий повтор строки — на долю не влияет).

    Args:
        answers: Ответы воркера.
        key: Ключ кадра.
        side: ``L``/``R``.

    Returns:
        ``EdgeCheck`` полосы.
    """
    chunks = sorted(k for k in answers if k.startswith(f"{key}|{side}|"))
    texts = []
    for chunk in chunks:
        for raw in answers[chunk].replace("­", "-").splitlines():
            words = raw.split()
            if words:
                texts.append(words)
    return check_lines(texts, side)
