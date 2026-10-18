"""Команда ``delete-project``: снести проект пака в CVAT с JSON-бэкапом и отвязать от него базу.

Нужна, когда проект заводят заново с нуля (другой набор предразметки, другой масштаб копий), а
ручной правки в старом нет или она уже забрана ``from-cvat``. Порядок шагов выбран так, чтобы
сбой на любом из них ничего не терял:

1. разметка каждой задачи проекта пишется в JSON (тот же формат, что у ``--recreate-stale``);
2. проект удаляется на сервере — вместе с задачами, джобами и разметкой;
3. только после этого в базе обнуляются ссылки на CVAT: id проекта, задач, джобов, номера
   кадров и отпечаток залитого файла. Без обнуления следующий ``to-cvat`` искал бы задачу по
   мёртвому id, а полосы считались бы залитыми.

Уменьшенные копии в share, пути к ним и делители полос не трогаются: новый проект заводится
из тех же картинок, и разметка в базе остаётся в их масштабе.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ocr_utils.db.repo import require_pack
from ocr_utils.scan_markup.cvat.client import CvatSettings, make_cvat_client
from ocr_utils.scan_markup.cvat.project import fetch_shapes_by_frame, fetch_tags_by_frame, project_label_ids
from ocr_utils.scan_markup.cvat.publish import _backup_annotations

logger = logging.getLogger(__name__)


@dataclass
class DeleteParams:
    """Параметры прогона ``delete-project``."""

    db_path: Path
    pack_name: str
    # False — сухой прогон: только посчитать и показать, что будет удалено.
    confirm: bool = False
    backup_dir: Path | None = None  # по умолчанию — cvat_backup рядом с базой
    settings: CvatSettings | None = None


@dataclass
class DeleteStats:
    """Что нашлось в проекте и что сделано."""

    project_id: int | None = None
    tasks: int = 0
    jobs: int = 0
    # Шейпы и теги по именам меток — чтобы перед удалением было видно, есть ли там ручная работа.
    shapes_by_label: Counter = field(default_factory=Counter)
    tags_by_label: Counter = field(default_factory=Counter)
    # Сколько джобов разметчик уже трогал (стадия или состояние не начальные).
    jobs_touched: int = 0
    backups: list[Path] = field(default_factory=list)
    deleted: bool = False


def find_pack_project(client, name: str, project_id: int | None):
    """Проект пака: по id из базы, иначе по имени. ``None`` — проекта нет.

    Args:
        client: клиент ``cvat_sdk``.
        name: имя проекта (оно же имя пака).
        project_id: ``pack.cvat_project_id`` из базы или ``None``.

    Returns:
        Объект проекта ``cvat_sdk`` или ``None``.
    """
    projects = list(client.projects.list())
    # По id — в первую очередь: имя проекта разметчик мог переименовать руками.
    for project in projects:
        if project_id is not None and project.id == project_id:
            return project
    for project in projects:
        if project.name == name:
            return project
    return None


def unlink_pack(pack) -> int:
    """Обнуляет в базе ссылки пака на CVAT. Возвращает число отвязанных полос.

    Args:
        pack: ORM-объект ``Pack`` с подгружаемыми годами, выпусками и полосами.

    Returns:
        Число полос, у которых был номер кадра или отпечаток залитого файла.
    """
    pack.cvat_project_id = None
    unlinked = 0
    for year in pack.year_packages:
        year.cvat_task_id = None
        for issue in year.issues:
            issue.cvat_job_id = None
            for page in issue.pages:
                # Считаем полосы, которые и правда были связаны с кадром, — для отчёта.
                unlinked += page.cvat_frame is not None or page.cvat_file_hash is not None
                page.cvat_frame = None
                # Отпечаток залитого файла — признак «полоса в CVAT»; без проекта его нет.
                page.cvat_file_hash = None
    return unlinked


def run_delete(params: DeleteParams, session_factory) -> DeleteStats:
    """Полный прогон ``delete-project``: опись, бэкап, удаление, отвязка базы.

    Args:
        params: параметры прогона; без ``confirm`` ничего не удаляется и не пишется.
        session_factory: фабрика сессий SQLAlchemy базы разметки.

    Returns:
        ``DeleteStats`` с описью проекта и признаком, удалён ли он.
    """
    stats = DeleteStats()
    settings = params.settings or CvatSettings()
    backup_dir = params.backup_dir or params.db_path.parent / "cvat_backup"

    with session_factory() as session, make_cvat_client(settings) as client:
        pack = require_pack(session, params.pack_name)
        project = find_pack_project(client, pack.name, pack.cvat_project_id)
        if project is None:
            logger.warning("Проекта %r в CVAT нет — удалять нечего", pack.name)
            if params.confirm:
                # Ссылки в базе всё равно обнуляем: они смотрят в никуда.
                unlink_pack(pack)
                session.commit()
            return stats
        stats.project_id = project.id

        # Опись: имена меток по id, задачи проекта, джобы, шейпы и теги.
        label_names = {label_id: label for label, label_id in project_label_ids(client, project.id).items()}
        tasks = [task for task in client.tasks.list() if task.project_id == project.id]
        stats.tasks = len(tasks)
        for task in tasks:
            jobs = task.get_jobs()
            stats.jobs += len(jobs)
            # Джоб, который не в начальном состоянии, кто-то открывал и правил.
            stats.jobs_touched += sum(1 for job in jobs if job.state != "new" or job.stage != "annotation")
            by_frame = fetch_shapes_by_frame(task)
            tags_by_frame = fetch_tags_by_frame(task)
            for shapes in by_frame.values():
                stats.shapes_by_label.update(label_names.get(shape.label_id, "?") for shape in shapes)
            for tags in tags_by_frame.values():
                stats.tags_by_label.update(label_names.get(tag.label_id, "?") for tag in tags)
            if params.confirm:
                # Бэкап каждой задачи — до удаления, и только в настоящем прогоне.
                stats.backups.append(
                    _backup_annotations(backup_dir, pack.name, task.name, task, by_frame, tags_by_frame)
                )

        if not params.confirm:
            return stats

        # Удаление проекта уносит задачи, джобы и разметку; после этого база отвязывается.
        client.api_client.projects_api.destroy(project.id)
        stats.deleted = True
        logger.warning("Проект %r (id=%s) удалён из CVAT", project.name, project.id)
        unlinked = unlink_pack(pack)
        session.commit()
        logger.info("В базе отвязано от CVAT полос: %d", unlinked)
    return stats
