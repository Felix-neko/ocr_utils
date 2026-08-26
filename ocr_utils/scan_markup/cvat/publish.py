"""Команда ``to-cvat``: база -> уменьшенные картинки в share -> проект/задачи/джобы.

Два шага, и оба идемпотентные:

1. уменьшенные копии полос в ``--share-root`` (готовое пропускается);
2. проект пака, задачи-годы, джобы-выпуски и заливка предразметки.

Уже существующая задача повторно НЕ создаётся; её предразметка перезаливается только по
``--force-annotations``, потому что заливка заменяет разметку целиком и затёрла бы ручную
правку разметчика.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from ocr_utils.scan_markup.cvat.client import CvatSettings, check_share_root, make_cvat_client, share_prefix
from ocr_utils.scan_markup.cvat.images import ImageJob, cvat_rel_path, prepare_images
from ocr_utils.scan_markup.cvat.project import (
    assign_annotator,
    assign_jobs_to_issues,
    create_year_task,
    ensure_project,
    find_task,
    frame_index_by_name,
    project_label_ids,
    raster_shapes,
    upload_preannotations,
)
from ocr_utils.scan_markup.db.repo import require_pack

logger = logging.getLogger(__name__)


@dataclass
class PublishParams:
    """Параметры прогона ``to-cvat``."""

    db_path: Path
    pack_name: str
    share_root: Path
    pack_dir: Path | None = None  # по умолчанию берётся из базы
    only_year: str | None = None
    workers: int = 8
    skip_images: bool = False
    force_images: bool = False
    force_annotations: bool = False
    annotator: str | None = None
    settings: CvatSettings | None = None


@dataclass
class PublishStats:
    """Итоги прогона."""

    images_done: int = 0
    images_skipped: int = 0
    images_failed: int = 0
    tasks_created: int = 0
    tasks_existing: int = 0
    shapes: int = 0


def _prepare_year_images(session: Session, pack, params: PublishParams, stats: PublishStats) -> None:
    """Готовит уменьшенные копии для всех полос пака и записывает их параметры в базу."""
    pack_dir = params.pack_dir or Path(pack.root_path)
    # Путь, который уйдёт в server_files, отсчитывается от IMAGES_DIR, а не от --share-root:
    # именно его сервер ищет внутри /home/django/share. Он же становится именем кадра,
    # поэтому хранится в базе как есть и служит ключом при обратном сопоставлении.
    prefix = share_prefix(params.share_root)
    jobs, pages_by_id = [], {}
    for year in pack.year_packages:
        if params.only_year is not None and year.name != params.only_year:
            continue
        for issue in year.issues:
            for page in issue.pages:
                if page.divisor is None:
                    logger.warning("Полоса %s без делителя (не прошла detect), пропускаю", page.rel_path)
                    continue
                rel = cvat_rel_path(pack.name, page.rel_path)
                page.cvat_rel_path = (prefix / rel).as_posix()
                pages_by_id[page.id] = page
                jobs.append(
                    ImageJob(
                        src=pack_dir / page.rel_path, dst=params.share_root / rel, divisor=page.divisor, page_id=page.id
                    )
                )
    session.commit()

    if params.skip_images:
        logger.info("--skip-images: уменьшение пропущено, беру уже готовые файлы")
        return

    for result in prepare_images(jobs, params.share_root, params.workers, params.force_images):
        page = pages_by_id.get(result.page_id)
        if result.status == "failed":
            stats.images_failed += 1
            continue
        if page is not None and result.cvat_width:
            # Размер кадра берётся из РЕАЛЬНОГО файла, а не из расчёта по делителю:
            # если файл готовили прошлым прогоном с другим делителем, расчёт соврал бы,
            # а весь пересчёт координат опирается именно на эти числа.
            page.cvat_width, page.cvat_height = result.cvat_width, result.cvat_height
        stats.images_done += result.status == "done"
        stats.images_skipped += result.status == "skipped"
    session.commit()


def run_publish(params: PublishParams, session_factory) -> PublishStats:
    """Полный прогон ``to-cvat``."""
    stats = PublishStats()
    settings = params.settings or CvatSettings()

    warning = check_share_root(params.share_root)
    if warning:
        logger.warning("%s", warning)

    with session_factory() as session:  # type: Session
        pack = require_pack(session, params.pack_name)
        _prepare_year_images(session, pack, params, stats)

        with make_cvat_client(settings) as client:
            project = ensure_project(client, pack.name)
            pack.cvat_project_id = project.id
            label_ids = project_label_ids(client, project.id)
            session.commit()

            for year in pack.year_packages:
                if params.only_year is not None and year.name != params.only_year:
                    continue

                issues = [issue for issue in year.issues if issue.pages]
                job_files, pages = [], []
                for issue in issues:
                    files = [page.cvat_rel_path for page in issue.pages if page.cvat_rel_path]
                    if not files:
                        continue
                    job_files.append(files)
                    pages.extend(issue.pages)
                if not job_files:
                    logger.warning("Год %s: нет подготовленных картинок, пропускаю", year.name)
                    continue

                task = find_task(client, project.id, year.name)
                if task is None:
                    task = create_year_task(client, project.id, year.name, job_files)
                    stats.tasks_created += 1
                    upload = True
                else:
                    logger.info("Задача %r уже есть (id=%s)", year.name, task.id)
                    stats.tasks_existing += 1
                    upload = params.force_annotations

                year.cvat_task_id = task.id
                frames = frame_index_by_name(task)
                matched = 0
                for page in pages:
                    page.cvat_frame = frames.get(page.cvat_rel_path)
                    matched += page.cvat_frame is not None
                if matched != len(pages):
                    # Обычно это значит, что задача заводилась при другом --share-root:
                    # имя кадра на сервере осталось прежним, а в базе путь уже новый. Полосы
                    # без номера кадра молча выпадут из from-cvat, поэтому говорим сразу.
                    logger.warning(
                        "Задача %r: с кадрами сопоставилось %d полос из %d. Похоже, задача "
                        "заводилась при другом --share-root — имена кадров на сервере не совпали "
                        "с путями в базе. Несопоставленные полосы выпадут из from-cvat.",
                        year.name,
                        matched,
                        len(pages),
                    )
                assign_jobs_to_issues(task, issues)
                session.commit()

                if upload:
                    shapes = raster_shapes(((page, page.raster_regions) for page in pages), frames, label_ids)
                    stats.shapes += upload_preannotations(task, shapes)
                    logger.info("Задача %r: залито шейпов %d", year.name, len(shapes))
                else:
                    logger.info(
                        "Задача %r: предразметка НЕ трогается (заливка заменяет разметку "
                        "целиком и затёрла бы ручную правку); нужна — укажите --force-annotations",
                        year.name,
                    )

                if params.annotator:
                    assign_annotator(client, task, params.annotator)

        session.commit()
    return stats
