"""Пересчёт разбора пака со стадии кандидатов (таблицы, линейки-сироты, кандидаты line art) в новый каталог с переносом ответов DeepSeek прошлого прогона.

Нужен, когда меняется детектор таблиц (``page_layout.TABLES_VERSION``), а растр — тот же (из базы после ревью):
кандидаты считаются заново по всем полосам, ответы DeepSeek прошлого прогона переносятся только для кандидатов с
тем же id И той же рамкой вырезки (та же картинка — тот же ответ), остальное досчитывает обычный ход
``pack_analysis.run.run`` (DeepSeek на GPU, второй проход, итог по всем полосам, защита сторон CRAFT).

Обобщение ``ai_slop/recompute_v3_candidates.py`` (пересчёт v3 2026-09-28): тот писал поверх v3, этот — в новый
каталог, прошлый прогон только читается.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from ocr_utils.page_layout.pack_analysis.raster_db import load_raster
from ocr_utils.page_layout.pack_analysis.run import list_tasks, run, stage_candidates, upright
from ocr_utils.page_layout.text_blocks.page import AxisKind

logger = logging.getLogger("recompute")


def _crop_boxes(pages_dir: Path) -> dict[str, list]:
    """Рамки вырезок кандидатов по id из JSON стадии кандидатов.

    Args:
        pages_dir: Каталог ``work/pages`` прогона.

    Returns:
        ``{id кандидата: рамка вырезки}``.
    """
    boxes: dict[str, list] = {}
    for path in pages_dir.glob("*.json"):
        for candidate in json.loads(path.read_text())["candidates"]:
            boxes[candidate["id"]] = candidate["crop"]
    return boxes


def transfer_deepseek(previous_work: Path, work: Path) -> int:
    """Перенести ответы DeepSeek прошлого прогона для кандидатов с тем же id и той же рамкой вырезки.

    Args:
        previous_work: ``work/`` прошлого прогона (только чтение).
        work: ``work/`` нового прогона (кандидаты уже посчитаны).

    Returns:
        Сколько кандидатов совпало (их ответы перенесены по всем проходам и промптам).
    """
    old_boxes = _crop_boxes(previous_work / "pages")
    new_boxes = _crop_boxes(work / "pages")
    same = {key for key, box in new_boxes.items() if old_boxes.get(key) == box}
    logger.info("кандидатов: прежде %d, теперь %d, с той же вырезкой %d", len(old_boxes), len(new_boxes), len(same))
    # Только первый проход: вырезки второго залиты с учётом соседних объектов полосы (таблиц, растра), и при новых
    # таблицах та же рамка может дать другую картинку — второй проход досчитывается заново.
    for stage, prompts in (("pass1", ("markdown", "ocr")),):
        for prompt in prompts:
            source = previous_work / "deepseek" / stage / f"{prompt}.jsonl"
            if not source.is_file():
                continue
            # Только ответы совпавших кандидатов; уже записанные в новом прогоне (повторный запуск) не дублируются.
            target = work / "deepseek" / stage / f"{prompt}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            done = set()
            if target.is_file():
                done = {json.loads(line)["id"] for line in target.read_text().splitlines() if line.strip()}
            lines = [
                line
                for line in source.read_text().splitlines()
                if line.strip() and json.loads(line)["id"] in same and json.loads(line)["id"] not in done
            ]
            with target.open("a") as handle:
                handle.write("".join(f"{line}\n" for line in lines))
            logger.info("%s/%s: перенесено %d", stage, prompt, len(lines))
    return len(same)


@click.command()
@click.option("--sharpened-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--cache", type=click.Path(path_type=Path, file_okay=False), required=True, help="Корень кэша surya.")
@click.option("--previous", type=click.Path(path_type=Path, file_okay=False), required=True, help="Прошлый прогон.")
@click.option("--out-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--raster-db", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--pack-name", default="пак-1", show_default=True)
@click.option("--jobs", type=int, default=12, show_default=True, help="Воркеров CPU-пула.")
@click.option("--axis", type=click.Choice([kind.value for kind in AxisKind]), default=AxisKind.BODY.value)
def main(
    sharpened_dir: Path,
    cache: Path,
    previous: Path,
    out_dir: Path,
    raster_db: Path,
    pack_name: str,
    jobs: int,
    axis: str,
) -> None:
    """Кандидаты заново в ``<out-dir>/work``, перенос ответов DeepSeek из ``<previous>/work``, досчёт обычным ходом.

    Args:
        sharpened_dir: Корень заострённых копий пака.
        cache: Корень кэша surya.
        previous: Корень прошлого прогона (читается только его ``work/``).
        out_dir: Корень нового прогона.
        raster_db: База разметки после ревью (растр берётся из неё).
        pack_name: Имя пака в базе.
        jobs: Воркеров CPU-пула.
        axis: Ось строки для рядов и блоков.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if out_dir.resolve() == previous.resolve():
        raise click.UsageError("новый прогон пишется в свой каталог, не в прошлый")
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    tasks = list_tasks(sharpened_dir)
    raster = load_raster(raster_db, pack_name)
    stage_candidates(tasks, upright(tasks), cache, work, jobs, raster)
    transfer_deepseek(previous / "work", work)
    run(
        sharpened_dir,
        cache,
        out_dir,
        jobs,
        None,
        None,
        detect_orientation=False,
        redo_final=True,
        raster_db=raster_db,
        pack_name=pack_name,
        axis=AxisKind(axis),
    )


if __name__ == "__main__":
    main()
