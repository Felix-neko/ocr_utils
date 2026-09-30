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
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from ocr_utils.page_layout.image import Variant
from ocr_utils.page_layout.pack_analysis.final import OverlayMode, final_page, reblock_page, render_for_maps
from ocr_utils.page_layout.pack_analysis.raster_db import load_raster, same_raster
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
from ocr_utils.page_layout.text_blocks.blocks import DEFAULT_BLOCKS_MODE, BlocksMode
from ocr_utils.page_layout.text_blocks.columns import GutterMode
from ocr_utils.page_layout.text_blocks.glyph_maps import MapJob, compute_maps
from ocr_utils.page_layout.text_blocks.page import AxisKind

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


def list_pdf_tasks(
    pdf_dir: Path, variant: Variant, pages_file: Path | None = None, limit: int | None = None
) -> list[PageTask]:
    """Страницы полных PDF пака FineReader: ``<pdf_dir>/<stem>.pdf``, все страницы по порядку.

    Имя страницы — ``<stem>/pNNNN`` (номер с нуля), как ключ кэша surya у ``PageImage.from_pdf_page``:
    так попадает набитый ``prefill-surya`` кэш.

    Args:
        pdf_dir: Каталог PDF.
        variant: ``Variant.FR_GEO`` или ``Variant.FR_NOGEO``.
        pages_file: Файл с именами страниц ``<stem>/pNNNN`` по строке — только они.
        limit: Только первые N (после отбора).

    Returns:
        Задания.
    """
    import fitz

    tasks = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        with fitz.open(str(pdf)) as document:
            count = document.page_count
        tasks += [PageTask(f"{pdf.stem}/p{index:04d}", pdf, index, variant) for index in range(count)]
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


# Промпты DeepSeek по проходам: какие файлы ``deepseek/pass{1,2}/<промпт>.jsonl`` пишет воркер.
PASS_PROMPTS = {"pass1": ("markdown", "ocr"), "pass2": ("markdown",)}


def stage_reuse(tasks: list[PageTask], previous: Path, work: Path, raster: dict[str, list[dict]]) -> None:
    """Стадия 1.5: полосы, чей растр не изменился против прошлого прогона, берутся из него целиком.

    Растр — запретная зона line art, таблиц и повёрнутого текста; если он тот же (:func:`same_raster`),
    стадия кандидатов дала бы то же самое, и DeepSeek по тем же вырезкам — тот же ответ. Такой полосе
    копируются запись стадии кандидатов, PNG-вырезки её кандидатов обоих проходов и строки вывода
    DeepSeek (воркер vLLM пропускает готовые id, так что по ним он не пойдёт). Остальные полосы
    остаются стадии кандидатов, их список — в ``work/raster_changed.txt``.

    Идемпотентна: полосы, у которых запись уже есть, не трогаются; строки DeepSeek не дублируются.

    Args:
        tasks: Полосы.
        previous: Корень прошлого прогона (его ``work/``).
        work: Рабочая папка нового прогона.
        raster: Растр полос из базы (:func:`raster_db.load_raster`).
    """
    old_work = previous / "work"
    reused_ids: set[str] = set()
    changed: list[str] = []
    reused = 0
    for task in tasks:
        key = page_key(task.name)
        target = work / "pages" / f"{key}.json"
        source = old_work / "pages" / f"{key}.json"
        if not source.is_file():
            changed.append(task.name)
            continue
        record = json.loads(source.read_text())
        if not same_raster(record["raster"], raster.get(task.name, [])):
            changed.append(task.name)
            continue
        ids = [c["id"] for c in record["candidates"]]
        reused_ids.update(ids)
        if target.is_file():
            reused += 1
            continue
        # Вырезки — до записи полосы: запись полосы означает «полоса готова», и прерванное копирование
        # не должно оставить полосу без вырезок.
        for crops in ("pass1", "pass2"):
            for candidate_id in ids:
                crop = old_work / "crops" / crops / f"{candidate_id}.png"
                if crop.is_file():
                    (work / "crops" / crops).mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(crop, work / "crops" / crops / f"{candidate_id}.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        reused += 1
    for stage, prompts in PASS_PROMPTS.items():
        for prompt in prompts:
            _copy_jsonl(
                old_work / "deepseek" / stage / f"{prompt}.jsonl",
                work / "deepseek" / stage / f"{prompt}.jsonl",
                reused_ids,
            )
    (work / "raster_changed.txt").write_text("".join(f"{name}\n" for name in changed))
    logger.info("Растр из базы: полос взято из прошлого прогона %d, пересчитывается %d", reused, len(changed))


def _copy_jsonl(source: Path, target: Path, ids: set[str]) -> None:
    """Дописать в ``target`` строки ``source`` с id из ``ids``, которых в ``target`` ещё нет."""
    if not source.is_file():
        return
    present = set(read_jsonl_map(target))
    lines = [
        line
        for line in source.read_text().splitlines()
        if line.strip() and (record_id := json.loads(line)["id"]) in ids and record_id not in present
    ]
    if lines:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as handle:
            handle.write("\n".join(lines) + "\n")


def stage_candidates(
    tasks, orientation, cache_root: Path, work: Path, jobs: int, raster: dict[str, list[dict]] | None = None
) -> None:
    """Стадия 2: растр, таблицы, кандидаты line art и их вырезки → ``work/pages``, ``work/jobs_pass1.jsonl``.

    ``raster`` — растр полос из базы (:func:`raster_db.load_raster`); задан — детектор растра не
    запускается, полоса без записи в словаре считается полосой без растра.
    """
    todo = [t for t in tasks if not (work / "pages" / f"{page_key(t.name)}.json").is_file()]
    logger.info("Кандидаты: полос %d, к обработке %d", len(tasks), len(todo))
    if todo:
        rotations = [orientation[t.name]["rotate_cw"] if orientation[t.name]["apply"] else 0 for t in todo]
        page_raster = [None if raster is None else raster.get(t.name, []) for t in todo]
        with _pool(jobs) as pool:
            for result in pool.map(
                _safe,
                [candidates_page] * len(todo),
                todo,
                rotations,
                [cache_root] * len(todo),
                [work] * len(todo),
                page_raster,
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


def stage_final(
    tasks,
    orientation,
    work: Path,
    out: Path,
    jobs: int,
    redo: bool = False,
    axis: AxisKind = AxisKind.CENTRE,
    gutter_mode: GutterMode = GutterMode.SHORT,
    blocks_mode: BlocksMode = DEFAULT_BLOCKS_MODE,
    overlay: OverlayMode = OverlayMode.ALL,
) -> None:
    """Стадия 6: итог полос, оверлеи по папкам, ``index.csv``.

    Args:
        tasks: Полосы.
        orientation: Вердикты ориентации по имени полосы.
        work: Рабочая папка (JSON стадии кандидатов, вывод DeepSeek).
        out: Корень выхода.
        jobs: Воркеров пула.
        redo: Пересчитать все полосы, а не только те, у которых итога ещё нет (после правки
            детектора текстовых блоков или оверлея — остальные стадии при этом не трогаются).
        axis: По какой оси строки собирать ряды и блоки (см. :func:`final.text_blocks`).
        gutter_mode: Как межколонники становятся запретами сцепки (``columns.GutterMode``).
        blocks_mode: Способ группировки строк в блоки и их границы (``text_blocks.blocks.BlocksMode``).
        overlay: Писать ли оверлеи полос.
    """
    pass1 = read_jsonl_map(work / "deepseek" / "pass1" / "markdown.jsonl")
    pass2 = read_jsonl_map(work / "deepseek" / "pass2" / "markdown.jsonl")
    # Слова первого прохода — для проверки «пометка на полях» у кандидатов без второго прохода.
    words = read_jsonl_map(work / "deepseek" / "pass1" / "ocr.jsonl")
    ready = [t for t in tasks if (work / "pages" / f"{page_key(t.name)}.json").is_file()]
    todo = ready if redo else [t for t in ready if not (out / "pages" / f"{page_key(t.name)}.json").is_file()]
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
                "pass1_ocr": {k: v for k, v in words.items() if k.startswith(prefix) and k[len(prefix) :].isdigit()},
            }
        )
    if todo:
        with _pool(jobs) as pool:
            for done, result in enumerate(
                pool.map(
                    _safe, [final_page] * len(todo), todo, [orientation[t.name] for t in todo], payloads,
                    [work] * len(todo), [out] * len(todo), [axis] * len(todo), [gutter_mode] * len(todo),
                    [blocks_mode] * len(todo), [overlay] * len(todo), chunksize=2,
                ),
                1,
            ):  # fmt: skip
                if "error" in result:
                    logger.error("итог: %s", result)
                if done % 500 == 0:
                    logger.info("итог: %d/%d", done, len(todo))
    write_index(ready, out)


def stage_reblock(
    tasks: list[PageTask],
    source: Path,
    out: Path,
    jobs: int,
    axis: AxisKind = AxisKind.BODY,
    gutter_mode: GutterMode = GutterMode.SHORT,
    redo: bool = False,
    blocks_mode: BlocksMode = DEFAULT_BLOCKS_MODE,
) -> None:
    """Пересчитать только текстовые блоки по готовому разбору пака: ``<out>/pages``, ``overlays``, ``index.csv``.

    Объекты, надписи, кандидаты и ориентация берутся из итоговых JSON ``<source>/pages``
    (:func:`final.reblock_page`); рабочая папка прошлого разбора не нужна. Идемпотентна: полосы с
    готовым итогом в ``out`` пропускаются, если не ``redo``.

    Args:
        tasks: Полосы.
        source: Корень прошлого разбора.
        out: Корень выхода (не тот же, что ``source``).
        jobs: Воркеров пула.
        axis: Ось строки для рядов и блоков.
        gutter_mode: Как межколонники становятся запретами сцепки.
        redo: Пересчитать и готовые полосы.
        blocks_mode: Способ группировки строк в блоки и их границы (``BlocksMode``).
    """
    if out.resolve() == source.resolve():
        raise ValueError("пересчёт текстовых блоков пишется в новую папку, а не поверх прошлого разбора")
    ready = [t for t in tasks if (source / "pages" / f"{page_key(t.name)}.json").is_file()]
    todo = ready if redo else [t for t in ready if not (out / "pages" / f"{page_key(t.name)}.json").is_file()]
    logger.info("Текстовые блоки по %s: полос %d, к обработке %d", source, len(ready), len(todo))
    if todo:
        with _pool(jobs) as pool:
            for done, result in enumerate(
                pool.map(
                    _safe, [reblock_page] * len(todo), todo, [source] * len(todo), [out] * len(todo),
                    [axis] * len(todo), [gutter_mode] * len(todo), [blocks_mode] * len(todo), chunksize=4,
                ),
                1,
            ):  # fmt: skip
                if "error" in result:
                    logger.error("блоки: %s", result)
                if done % 500 == 0:
                    logger.info("блоки: %d/%d", done, len(todo))
    write_index(ready, out)


def stage_edge_craft(
    tasks: list[PageTask],
    out: Path,
    jobs: int,
    axis: AxisKind,
    gutter_mode: GutterMode = GutterMode.SHORT,
    blocks_mode: BlocksMode = DEFAULT_BLOCKS_MODE,
    maps_dir: Path | None = None,
    overlay: OverlayMode = OverlayMode.ALL,
) -> None:
    """Второй проход защиты сторон по готовому итогу: полосы с выступами на выровненных сторонах разбираются заново
    с фильтром сора голосованием CRAFT + pero (``text_blocks.edge_guard``).

    Полосы берутся по итоговым JSON ``<out>/pages``: аномальные (``text_blocks.edge_guard.anomalous``), у которых
    второго прохода ещё не было. Рендеры полос пишутся пулом, карты CRAFT и pero считаются одним GPU-процессом на
    детектор (кэш ``maps_dir``), затем полосы пересчитываются пулом (:func:`final.reblock_page`) на месте.
    Идемпотентна: полосы со вторым проходом пропускаются.

    Args:
        tasks: Полосы.
        out: Корень итога (читается и перезаписывается).
        jobs: Воркеров пула.
        axis: Ось строки для рядов и блоков (как у основного прохода).
        gutter_mode: Как межколонники становятся запретами сцепки.
        blocks_mode: Способ группировки строк в блоки и их границы.
        maps_dir: Кэш карт; по умолчанию ``<out>/glyph_maps``.
        overlay: Писать ли оверлеи пересчитанных полос.
    """
    maps_dir = maps_dir or out / "glyph_maps"
    todo = []
    for task in tasks:
        path = out / "pages" / f"{page_key(task.name)}.json"
        if not path.is_file():
            continue
        guard = json.loads(path.read_text()).get("text_blocks", {}).get("edge_guard") or {}
        if guard.get("anomalous") and not guard.get("second_pass"):
            todo.append(task)
    logger.info("Защита сторон: полос с выступами без второго прохода %d", len(todo))
    if not todo:
        return
    png_dir = out / "work" / "glyph_maps_input"
    png_dir.mkdir(parents=True, exist_ok=True)
    with _pool(jobs) as pool:
        pngs = list(pool.map(render_for_maps, todo, [out] * len(todo), [png_dir] * len(todo), chunksize=4))
    # Карты — GPU, не в пуле: детекторы строго по очереди, каждый одним процессом на все полосы.
    compute_maps([MapJob(page_key(t.name), png=Path(p)) for t, p in zip(todo, pngs)], maps_dir)
    with _pool(jobs) as pool:
        for done, result in enumerate(
            pool.map(
                _safe, [reblock_page] * len(todo), todo, [out] * len(todo), [out] * len(todo), [axis] * len(todo),
                [gutter_mode] * len(todo), [blocks_mode] * len(todo), [maps_dir] * len(todo), [overlay] * len(todo),
                chunksize=4,
            ),
            1,
        ):  # fmt: skip
            if "error" in result:
                logger.error("защита сторон: %s", result)
            if done % 200 == 0:
                logger.info("защита сторон: %d/%d", done, len(todo))
    for path in png_dir.glob("*.png"):
        path.unlink()
    write_index(tasks, out)


def write_index(tasks: list[PageTask], out: Path) -> None:
    """Опись ``<out>/index.csv`` по итоговым JSON полос: папка, классы, число блоков, ориентация."""
    rows = []
    for task in tasks:
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
    sharpened: Path | None,
    cache_root: Path,
    out: Path,
    jobs: int,
    pages_file: Path | None,
    limit: int | None,
    detect_orientation: bool = True,
    redo_final: bool = False,
    raster_db: Path | None = None,
    pack_name: str = "пак-1",
    reuse_from: Path | None = None,
    axis: AxisKind = AxisKind.CENTRE,
    gutter_mode: GutterMode = GutterMode.SHORT,
    blocks_mode: BlocksMode = DEFAULT_BLOCKS_MODE,
    edge_craft: bool = True,
    pdf_dir: Path | None = None,
    variant: Variant = Variant.SHARPENED,
    no_raster: bool = False,
    overlay: OverlayMode = OverlayMode.ALL,
) -> None:
    """Весь разбор по стадиям.

    Args:
        sharpened: Корень заострённых копий пака; ``None`` — разбираются страницы PDF (``pdf_dir``).
        cache_root: Корень кэша surya.
        out: Корень выхода.
        jobs: Воркеров CPU-пула.
        pages_file: Только эти полосы (для пробных прогонов).
        limit: Только первые N полос.
        detect_orientation: Определять ли ориентацию; ``False`` — все полосы считаются прямыми
            (заострённые копии пака-1 экспортированы уже повёрнутыми, стадия там ничего не даёт).
        redo_final: Пересчитать итоговую стадию у всех полос (см. :func:`stage_final`).
        raster_db: База разметки после ревью: растр берётся из неё, детектор растра не запускается.
        pack_name: Имя пака в базе.
        reuse_from: Корень прошлого прогона: полосы с тем же растром берутся из него (:func:`stage_reuse`);
            только вместе с ``raster_db``.
        axis: Ось строки, по которой детектор текстовых блоков собирает ряды и блоки.
        gutter_mode: Как межколонники становятся запретами сцепки (``columns.GutterMode``).
        blocks_mode: Способ группировки строк в блоки и их границы (``BlocksMode``).
        edge_craft: Второй проход защиты сторон (карты CRAFT + pero на GPU) по полосам с выступами на выровненных
            сторонах (:func:`stage_edge_craft`); выступы, оставшиеся по итогу, помечаются недостоверными всегда.
        pdf_dir: Каталог полных PDF FineReader: разбираются их страницы (вместо ``sharpened``).
        variant: Вариант страниц PDF (``FR_GEO`` / ``FR_NOGEO``) — от него ключ кэша surya.
        no_raster: Детектор растра не запускать и растра не брать: полосы считаются без растра (бинарные PDF
            FineReader — растр там всё равно бинаризован).
        overlay: Писать ли оверлеи полос (``NONE`` — только JSON и сайдкары).
    """
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    if pdf_dir is not None:
        tasks = list_pdf_tasks(pdf_dir, variant, pages_file, limit)
    elif sharpened is not None:
        tasks = list_tasks(sharpened, pages_file, limit)
    else:
        raise ValueError("нужен корень заострённых копий или каталог PDF")
    if detect_orientation:
        orientation = stage_orientation(tasks, work, jobs)
        stage_rotated_surya(tasks, orientation, cache_root)
    else:
        logger.info("Ориентация не определяется: все %d полос считаются прямыми", len(tasks))
        orientation = upright(tasks)
    raster = None
    if no_raster:
        if raster_db is not None:
            raise ValueError("--no-raster и --raster-db несовместимы")
        # Пустой растр у каждой полосы: детектор не запускается (см. ``stage_candidates``).
        raster = {}
        logger.info("Растр не ищется: полосы считаются без растра")
    elif raster_db is not None:
        raster = load_raster(raster_db, pack_name)
        logger.info("Растр из базы %s: полос с растром %d", raster_db, len(raster))
        if reuse_from is not None:
            stage_reuse(tasks, reuse_from, work, raster)
    elif reuse_from is not None:
        raise ValueError("--reuse-from имеет смысл только с растром из базы (--raster-db)")
    stage_candidates(tasks, orientation, cache_root, work, jobs, raster)
    stage_deepseek(work / "jobs_pass1.jsonl", work / "deepseek" / "pass1", "markdown,ocr")
    stage_pass2(work, jobs)
    stage_deepseek(work / "jobs_pass2.jsonl", work / "deepseek" / "pass2", "markdown")
    stage_final(
        tasks,
        orientation,
        work,
        out,
        jobs,
        redo=redo_final,
        axis=axis,
        gutter_mode=gutter_mode,
        blocks_mode=blocks_mode,
        overlay=overlay,
    )
    if edge_craft:
        stage_edge_craft(tasks, out, jobs, axis, gutter_mode, blocks_mode, overlay=overlay)


def _read_jsonl(path: Path) -> list[dict]:
    """Строки JSONL (пусто, если файла нет)."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


__all__ = ["list_pdf_tasks", "list_tasks", "run", "stage_edge_craft", "stage_reblock", "stage_reuse", "write_index"]
