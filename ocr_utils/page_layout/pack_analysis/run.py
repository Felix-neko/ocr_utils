"""Оркестратор разбора пака: стадии по порядку, CPU — пулом по полосам, DeepSeek-OCR-2 — подпроцессом vLLM под сторожем памяти.

Прогон идемпотентен: каждая стадия пропускает то, что уже сделано (JSON полос, строки ``*.jsonl``),
и прерванный прогон продолжается с места. Выход:

* ``<out>/work/`` — рабочие файлы: ориентация, JSON полос стадии кандидатов, вырезки, вывод DeepSeek;
* ``<out>/pages/`` — итог полосы (JSON);
* ``<out>/overlays/только_текст/``, ``<out>/overlays/не_только_текст/<класс|несколько_классов>/``,
  ``<out>/overlays/ориентация_спорная/``; опись — ``<out>/index.csv``.
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from ocr_utils.page_layout.pack_analysis.final import final_page
from ocr_utils.page_layout.pack_analysis.stages import (
    PageTask,
    candidates_page,
    init_worker,
    load_image,
    orient_page,
    page_key,
    pass2_candidate,
    read_jsonl_map,
    rotated_name,
)

logger = logging.getLogger(__name__)

# Окружение и воркер vLLM (рядом с логикой DeepSeek).
DEEPSEEK_DIR = Path(__file__).resolve().parent.parent / "line_art" / "deepseek" / "vllm_env"
WATCHDOG = Path(__file__).resolve().parents[3] / "scripts" / "memory_watchdog.py"
# Сторож: потолок памяти сессии vLLM и минимум свободной памяти системы, ГБ.
WATCHDOG_LIMIT_GB = 90
WATCHDOG_MIN_AVAILABLE_GB = 12


def list_tasks(root: Path, pages_file: Path | None = None, limit: int | None = None) -> list[PageTask]:
    """Полосы пака: ``<root>/<год>/<выпуск>/<полоса>.jpg`` по порядку.

    Args:
        root: Корень заострённых копий.
        pages_file: Файл с именами полос «год/выпуск/полоса» по строке — только они.
        limit: Только первые N (после отбора).

    Returns:
        Задания.
    """
    tasks = [PageTask(str(p.relative_to(root).with_suffix("")), p) for p in sorted(root.rglob("*.jpg"))]
    if pages_file is not None:
        wanted = {line.strip() for line in pages_file.read_text().splitlines() if line.strip()}
        tasks = [t for t in tasks if t.name in wanted]
    return tasks[:limit] if limit else tasks


def _pool(jobs: int) -> ProcessPoolExecutor:
    """Пул forkserver с инициализатором (hugepage и потоки BLAS выключены)."""
    return ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker)


def _safe(fn, *args):
    """Вызов стадии в воркере: исключение — в словарь, одна битая полоса не валит прогон."""
    try:
        return fn(*args)
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}", "args": str(args[0])}


def stage_orientation(tasks: list[PageTask], work: Path, jobs: int) -> dict[str, dict]:
    """Стадия 1: ориентация всех полос → ``work/orientation.jsonl``."""
    path = work / "orientation.jsonl"
    done = {r["page"]: r for r in _read_jsonl(path)}
    todo = [t for t in tasks if t.name not in done]
    logger.info("Ориентация: полос %d, к обработке %d", len(tasks), len(todo))
    if todo:
        with _pool(jobs) as pool, path.open("a") as out:
            for result in pool.map(_safe, [orient_page] * len(todo), todo, chunksize=4):
                if "error" in result:
                    logger.error("ориентация: %s", result)
                    continue
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                done[result["page"]] = result
    applied = sum(r["apply"] for r in done.values())
    disputed = sum(r["disputed"] for r in done.values())
    logger.info("Ориентация: поворот применяется на %d полосах, спорных %d", applied, disputed)
    return done


def stage_rotated_surya(tasks: list[PageTask], orientation: dict[str, dict], cache_root: Path) -> None:
    """Стадия 1.1: surya для повёрнутых кадров (в родителе, модель грузится только при промахе)."""
    from ocr_utils.page_layout.surya.cache import SuryaCache

    cache = SuryaCache(cache_root)
    model = None
    for task in tasks:
        verdict = orientation.get(task.name)
        if not verdict or not verdict["apply"]:
            continue
        image = load_image(task, verdict["rotate_cw"])
        if cache.load(image) is not None:
            continue
        if model is None:
            from ocr_utils.page_layout.surya.model import SuryaLayoutModel

            model = SuryaLayoutModel()
        blocks = model.predict_one(image.surya_frame)
        cache.save(image, blocks)
        logger.info("surya для повёрнутого кадра %s", rotated_name(task.name, verdict["rotate_cw"]))
    if model is not None:
        model.close()


def stage_candidates(tasks, orientation, cache_root: Path, work: Path, jobs: int) -> None:
    """Стадия 2: растр, таблицы, кандидаты line art и их вырезки → ``work/pages``, ``work/jobs_pass1.jsonl``."""
    todo = [t for t in tasks if not (work / "pages" / f"{page_key(t.name)}.json").is_file()]
    logger.info("Кандидаты: полос %d, к обработке %d", len(tasks), len(todo))
    if todo:
        rotations = [orientation[t.name]["rotate_cw"] if orientation[t.name]["apply"] else 0 for t in todo]
        with _pool(jobs) as pool:
            for result in pool.map(
                _safe,
                [candidates_page] * len(todo),
                todo,
                rotations,
                [cache_root] * len(todo),
                [work] * len(todo),
                chunksize=2,
            ):
                if "error" in result:
                    logger.error("кандидаты: %s", result)
    lines = []
    for task in tasks:
        path = work / "pages" / f"{page_key(task.name)}.json"
        if path.is_file():
            for candidate in json.loads(path.read_text())["candidates"]:
                crop = work / "crops" / "pass1" / f"{candidate['id']}.png"
                lines.append(json.dumps({"id": candidate["id"], "crop": str(crop)}))
    (work / "jobs_pass1.jsonl").write_text("\n".join(lines) + "\n")
    logger.info("Кандидатов line art: %d", len(lines))


def stage_deepseek(jobs_file: Path, out_dir: Path, prompts: str) -> None:
    """Стадии 3 и 5: воркер vLLM подпроцессом (своя сессия) под сторожем памяти; ждёт окончания."""
    if not jobs_file.read_text().strip():
        return
    log = out_dir.parent / f"{out_dir.name}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "uv", "run", "--project", str(DEEPSEEK_DIR), "python", str(DEEPSEEK_DIR / "worker.py"),
        "--jobs", str(jobs_file), "--out-dir", str(out_dir), "--prompts", prompts,
    ]  # fmt: skip
    with log.open("a") as handle:
        worker = subprocess.Popen(
            command, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
        )
    watchdog = subprocess.Popen(
        [
            sys.executable, str(WATCHDOG), "--sid", str(worker.pid), "--limit-gb", str(WATCHDOG_LIMIT_GB),
            "--min-available-gb", str(WATCHDOG_MIN_AVAILABLE_GB), "--log", str(out_dir.parent / f"{out_dir.name}_watchdog.log"),
        ],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )  # fmt: skip
    code = worker.wait()
    watchdog.wait()
    if code != 0:
        raise RuntimeError(f"воркер DeepSeek завершился с кодом {code}, лог {log}")
    logger.info("DeepSeek %s: готово (%s)", out_dir.name, prompts)


def stage_pass2(work: Path, jobs: int) -> None:
    """Стадия 4: какие кандидаты нужны второму проходу, их залитые вырезки → ``work/jobs_pass2.jsonl``."""
    markdown = read_jsonl_map(work / "deepseek" / "pass1" / "markdown.jsonl")
    words = read_jsonl_map(work / "deepseek" / "pass1" / "ocr.jsonl")
    candidates = []
    for path in sorted((work / "pages").glob("*.json")):
        candidates += json.loads(path.read_text())["candidates"]
    with _pool(jobs) as pool:
        flags = list(
            pool.map(
                pass2_candidate,
                candidates,
                [markdown.get(c["id"], []) for c in candidates],
                [words.get(c["id"], []) for c in candidates],
                [work] * len(candidates),
                chunksize=8,
            )
        )
    chosen = [c for c, flag in zip(candidates, flags) if flag]
    lines = [json.dumps({"id": c["id"], "crop": str(work / "crops" / "pass2" / f"{c['id']}.png")}) for c in chosen]
    (work / "jobs_pass2.jsonl").write_text("\n".join(lines) + "\n")
    logger.info("Второй проход: %d из %d кандидатов", len(chosen), len(candidates))


def stage_final(tasks, orientation, work: Path, out: Path, jobs: int) -> None:
    """Стадия 6: итог полос, оверлеи по папкам, ``index.csv``."""
    pass1 = read_jsonl_map(work / "deepseek" / "pass1" / "markdown.jsonl")
    pass2 = read_jsonl_map(work / "deepseek" / "pass2" / "markdown.jsonl")
    ready = [t for t in tasks if (work / "pages" / f"{page_key(t.name)}.json").is_file()]
    todo = [t for t in ready if not (out / "pages" / f"{page_key(t.name)}.json").is_file()]
    logger.info("Итог: полос %d, к обработке %d", len(ready), len(todo))
    payloads = []
    for task in todo:
        prefix = page_key(task.name) + "_"
        payloads.append(
            {
                "pass1_markdown": {
                    k: v for k, v in pass1.items() if k.startswith(prefix) and k[len(prefix) :].isdigit()
                },
                "pass2_markdown": {
                    k: v for k, v in pass2.items() if k.startswith(prefix) and k[len(prefix) :].isdigit()
                },
            }
        )
    if todo:
        with _pool(jobs) as pool:
            for done, result in enumerate(
                pool.map(
                    _safe, [final_page] * len(todo), todo, [orientation[t.name] for t in todo], payloads,
                    [work] * len(todo), [out] * len(todo), chunksize=2,
                ),
                1,
            ):  # fmt: skip
                if "error" in result:
                    logger.error("итог: %s", result)
                if done % 500 == 0:
                    logger.info("итог: %d/%d", done, len(todo))
    rows = []
    for task in ready:
        path = out / "pages" / f"{page_key(task.name)}.json"
        if path.is_file():
            result = json.loads(path.read_text())
            rows.append(
                {
                    "page": result["page"],
                    "folder": result["folder"],
                    "classes": ",".join(sorted({o["class"] for o in result["objects"]})),
                    "text_blocks": result["text_blocks"]["count"],
                    "rotate_cw": result["orientation"]["rotate_cw"] if result["orientation"].get("apply") else 0,
                    "orientation_disputed": int(bool(result["orientation"].get("disputed"))),
                }
            )
    with (out / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["page"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Папки: %s", dict(Counter(r["folder"] for r in rows)))


def upright(tasks: list[PageTask]) -> dict[str, dict]:
    """Вердикты «прямая» для всех полос — когда ориентация известна заранее (``--no-orientation``)."""
    return {
        t.name: {"page": t.name, "rotate_cw": 0, "confidence": 1.0, "disputed": False, "axis_only": False,
                 "note": "ориентация не определялась: полосы заранее прямые", "apply": False}
        for t in tasks
    }  # fmt: skip


def run(
    sharpened: Path,
    cache_root: Path,
    out: Path,
    jobs: int,
    pages_file: Path | None,
    limit: int | None,
    detect_orientation: bool = True,
) -> None:
    """Весь разбор по стадиям.

    Args:
        sharpened: Корень заострённых копий пака.
        cache_root: Корень кэша surya.
        out: Корень выхода.
        jobs: Воркеров CPU-пула.
        pages_file: Только эти полосы (для пробных прогонов).
        limit: Только первые N полос.
        detect_orientation: Определять ли ориентацию; ``False`` — все полосы считаются прямыми
            (заострённые копии пака-1 экспортированы уже повёрнутыми, стадия там ничего не даёт).
    """
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    tasks = list_tasks(sharpened, pages_file, limit)
    if detect_orientation:
        orientation = stage_orientation(tasks, work, jobs)
        stage_rotated_surya(tasks, orientation, cache_root)
    else:
        logger.info("Ориентация не определяется: все %d полос считаются прямыми", len(tasks))
        orientation = upright(tasks)
    stage_candidates(tasks, orientation, cache_root, work, jobs)
    stage_deepseek(work / "jobs_pass1.jsonl", work / "deepseek" / "pass1", "markdown,ocr")
    stage_pass2(work, jobs)
    stage_deepseek(work / "jobs_pass2.jsonl", work / "deepseek" / "pass2", "markdown")
    stage_final(tasks, orientation, work, out, jobs)


def _read_jsonl(path: Path) -> list[dict]:
    """Строки JSONL (пусто, если файла нет)."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


__all__ = ["list_tasks", "run"]
