"""Команда ``to-cvat``: база -> уменьшенные картинки в share -> проект/задачи/джобы.

Два шага, и оба идемпотентные:

1. уменьшенные копии полос в ``--share-root`` (готовое пропускается);
2. проект пака, задачи-годы, джобы-выпуски и заливка предразметки.

Уже существующая задача повторно НЕ создаётся; её предразметка перезаливается только по
``--force-annotations``, потому что заливка заменяет разметку целиком и затёрла бы ручную
правку разметчика.

Обновление пака по частям
-------------------------
Сканы пересчитывают: выпуск перескали, полосу переобрезали в ScanTailor, кривой кадр
заменили. Такие полосы находятся по отпечатку файла: ``page.file_hash`` — что лежит на
диске сейчас, ``page.cvat_file_hash`` — что было залито в CVAT. Разошлись — кадр в CVAT
показывает не тот файл, и разметка на нём недействительна.

Дальше начинается ограничение CVAT: **обычный джоб удалить нельзя**. Сервер отвечает
``"Only ground truth jobs can be removed"`` (``cvat/apps/engine/views.py``,
``JobViewSet.perform_destroy``), а границы джобов задаются один раз при создании задачи
через ``job_file_mapping``. Наименьшее, что вообще можно пересоздать, — задача, то есть
целый годовой комплект.

Пересоздавать год целиком означало бы выбросить ручную разметку всех его выпусков из-за
одной подменённой полосы. Поэтому ``--recreate-stale`` делает не это, а следующее:

1. забирает разметку старой задачи и раскладывает её по ИМЕНАМ кадров;
2. пишет её в файл-бэкап;
3. создаёт новую задачу под временным именем;
4. заливает в неё перенесённую разметку — со всех кадров, КРОМЕ изменившихся, плюс
   свежую автоматическую предразметку на сами изменившиеся;
5. и только теперь удаляет старую задачу и переименовывает новую.

Для разметчика результат неотличим от «удалили только те джобы, где менялись картинки»:
вся его работа на неизменившихся полосах остаётся на месте. Порядок шагов выбран так,
чтобы сбой на любом из них оставлял старую задачу нетронутой: сначала создаём, потом
удаляем.

Геометрию при переносе не пересчитываем: уменьшение делается тем же делителем из базы,
значит кадр в новой задаче попиксельно совпадает со старым.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from ocr_utils.scan_markup.cvat.client import CvatSettings, check_share_root, make_cvat_client, share_prefix
from ocr_utils.scan_markup.cvat.images import ImageJob, cvat_rel_path, prepare_images
from ocr_utils.scan_markup.cvat.project import (
    TEMP_SUFFIX,
    apply_job_states,
    assign_annotator,
    assign_jobs_to_issues,
    create_year_task,
    ensure_project,
    fetch_shapes_by_frame,
    find_task,
    find_year_task,
    frame_index_by_name,
    job_states,
    project_label_ids,
    raster_shapes,
    rename_task,
    shapes_to_requests,
    upload_preannotations,
    year_task_name,
)
from ocr_utils.scan_markup.db.repo import require_pack
from ocr_utils.scan_markup.geometry import CVAT_DPI, crop_size, cvat_size, divisor_for_dpi
from ocr_utils.scan_markup.hashing import is_stale_in_cvat

logger = logging.getLogger(__name__)


@dataclass
class PublishParams:
    """Параметры прогона ``to-cvat``."""

    db_path: Path
    pack_name: str
    share_root: Path
    pack_dir: Path | None = None  # по умолчанию берётся из базы
    only_year: str | None = None
    cvat_dpi: int = CVAT_DPI
    workers: int = 8
    skip_images: bool = False
    force_images: bool = False
    force_annotations: bool = False
    recreate_stale: bool = False
    backup_dir: Path | None = None  # по умолчанию — рядом с базой
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
    tasks_rebuilt: int = 0
    pages_changed: int = 0
    pages_unpublished: int = 0
    shapes: int = 0
    shapes_carried: int = 0
    stale_years: list[str] = field(default_factory=list)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def issue_drift(issue) -> tuple[list, list]:
    """Расхождение выпуска с тем, что залито в CVAT.

    Возвращает две группы полос:

    * **изменившиеся** — файл на диске уже не тот, под которым размечали. Их разметку
      переносить нельзя;
    * **ни разу не залитые** — полоса появилась в выпуске после того, как задача была
      создана. Дописать кадр в существующую задачу CVAT не даёт, поэтому такая полоса
      остаётся невидимой разметчику, пока задачу не пересоздадут.

    Вторая группа считается расхождением только для СУЩЕСТВУЮЩЕЙ задачи: у новой задачи
    не залито ничего, и это нормальное состояние, а не проблема.
    """
    changed = [page for page in issue.pages if is_stale_in_cvat(page)]
    added = [page for page in issue.pages if page.cvat_file_hash is None and page.cvat_rel_path]
    return changed, added


def year_drift(issues) -> list[tuple[object, list, list]]:
    """Расхождения по всем выпускам года; выпуски без расхождений отбрасываются."""
    drift = []
    for issue in issues:
        changed, added = issue_drift(issue)
        if changed or added:
            drift.append((issue, changed, added))
    return drift


def report_drift(year_name: str, drift: list[tuple[object, list, list]]) -> None:
    """Печатает, какие джобы задеты и чем. Это и есть ответ на вопрос «что переделывать»."""
    logger.warning(
        "Задача %r разошлась с диском: затронуто выпусков (джобов) %d из уже созданной задачи.", year_name, len(drift)
    )
    for issue, changed, added in drift:
        parts = []
        if changed:
            parts.append(f"изменилось полос {len(changed)}")
        if added:
            parts.append(f"добавилось полос {len(added)}")
        logger.warning("  выпуск %s (job id=%s): %s", issue.name, issue.cvat_job_id, ", ".join(parts))
        for page in changed[:5]:
            logger.warning("    изменилась: %s", page.rel_path)
        if len(changed) > 5:
            logger.warning("    ... и ещё %d", len(changed) - 5)


def _backup_annotations(backup_dir: Path, pack_name: str, year_name: str, task, by_frame: dict) -> Path:
    """Складывает разметку задачи в JSON перед тем, как задачу удалят.

    Страховка на случай, если перенос разметки окажется неполным: восстановить из этого
    файла можно и руками. Стоит он копейки, а альтернатива — потерянные недели обводки.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utcnow().strftime("%Y%m%d-%H%M%S")
    safe_year = year_name.replace("/", "_")
    path = backup_dir / f"{pack_name}-{safe_year}-task{task.id}-{stamp}.json"
    payload = {
        "pack": pack_name,
        "year": year_name,
        "task_id": task.id,
        "saved_at": stamp,
        "frames": {name: [shape.to_dict() for shape in shapes] for name, shapes in by_frame.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Разметка задачи %r сохранена в %s", year_name, path)
    return path


def _drop_stale_temp_task(client, project_id: int, year_name: str) -> None:
    """Сносит недоделанного двойника, оставшегося от прерванного пересоздания."""
    leftover = find_task(client, project_id, year_name + TEMP_SUFFIX)
    if leftover is not None:
        logger.warning(
            "Нашлась незавершённая задача %r (id=%s) от прерванного пересоздания — удаляю.", leftover.name, leftover.id
        )
        leftover.remove()


def rebuild_year_task(
    client, project_id: int, year_name: str, title: str, old_task, job_files, carry, reset_jobs=None
) -> object:
    """Пересоздаёт задачу-год, перенося в неё разметку с неизменившихся кадров.

    ``carry`` — функция ``(frames) -> список шейпов``: она получает нумерацию кадров НОВОЙ
    задачи и возвращает всё, что в неё надо залить. Так вся работа с базой остаётся в
    вызывающем, а здесь — только порядок операций с сервером.

    Порядок принципиален: новая задача создаётся и наполняется ДО удаления старой. Сбой на
    любом шаге оставляет старую задачу целой, и прогон можно просто повторить.

    ``reset_jobs`` — позиции джобов, которым «завершён» больше не подходит: в выпуск
    добавилась полоса, которой разметчик там не видел.
    """
    temp_name = year_name + TEMP_SUFFIX
    states = job_states(old_task)
    new_task = create_year_task(client, project_id, temp_name, job_files)
    frames = frame_index_by_name(new_task)

    shapes = carry(frames)
    uploaded = upload_preannotations(new_task, shapes)
    logger.info("Новая задача %r (id=%s): залито шейпов %d", year_name, new_task.id, uploaded)

    # Состояния джобов не входят в разметку и при пересоздании обнулились бы: год, размеченный
    # наполовину, выглядел бы нетронутым.
    restored = apply_job_states(new_task, states, reset_jobs)
    if restored:
        logger.info("Новая задача %r: восстановлено состояний джобов %d", year_name, restored)

    old_id = old_task.id
    old_task.remove()
    rename_task(new_task, title)
    logger.warning("Задача %r пересоздана: id %s -> %s", year_name, old_id, new_task.id)
    return new_task


def _prepare_year_images(session: Session, pack, params: PublishParams, stats: PublishStats) -> None:
    """Готовит уменьшенные копии для всех полос пака и записывает их параметры в базу.

    Здесь же выбирается делитель: ``round(dpi полосы / --cvat-dpi)``. Он считается на этом
    шаге, а не на ``detect``, чтобы разрешение разметки можно было сменить, не перечитывая
    пак заново. Цена — лишняя запись в базу на каждом прогоне ``to-cvat``.

    Полосе, уже залитой в CVAT, делитель не меняется: её разметка нарисована в прежнем
    масштабе, и новый делитель сдвинул бы готовые рамки и маски. Такие полосы остаются со
    своим коэффициентом, о чём прогон предупреждает.
    """
    pack_dir = params.pack_dir or Path(pack.root_path)
    # Путь, который уйдёт в server_files, отсчитывается от IMAGES_DIR, а не от --share-root:
    # именно его сервер ищет внутри /home/django/share. Он же становится именем кадра,
    # поэтому хранится в базе как есть и служит ключом при обратном сопоставлении.
    prefix = share_prefix(params.share_root)
    jobs, pages_by_id = [], {}
    frozen = 0
    for year in pack.year_packages:
        if params.only_year is not None and year.name != params.only_year:
            continue
        for issue in year.issues:
            for page in issue.pages:
                if page.dpi is None:
                    logger.warning("Полоса %s без DPI (не прошла detect), пропускаю", page.rel_path)
                    continue
                divisor = divisor_for_dpi(page.dpi, params.cvat_dpi)
                rescaled = divisor != page.divisor
                if rescaled and page.cvat_file_hash is not None:
                    frozen += 1
                    divisor, rescaled = page.divisor, False
                if rescaled:
                    page.divisor = divisor
                    page.crop_width, page.crop_height = crop_size(page.width, page.height, divisor)
                    page.cvat_width, page.cvat_height = cvat_size(page.width, page.height, divisor)
                rel = cvat_rel_path(pack.name, page.rel_path)
                page.cvat_rel_path = (prefix / rel).as_posix()
                pages_by_id[page.id] = page
                jobs.append(
                    ImageJob(
                        src=pack_dir / page.rel_path,
                        dst=params.share_root / rel,
                        divisor=divisor,
                        page_id=page.id,
                        # Переделать старую уменьшенную копию: либо изменился исходник (иначе
                        # в CVAT останется кадр от прежнего файла), либо сменился делитель
                        # (иначе останется кадр прежнего масштаба).
                        force=is_stale_in_cvat(page) or rescaled,
                    )
                )
    if frozen:
        logger.warning(
            "Делитель не менялся у %d полос: они уже залиты в CVAT, и разметка на них "
            "нарисована в прежнем масштабе. Чтобы сменить разрешение и у них, задачу-год "
            "придётся завести заново.",
            frozen,
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


def _mark_published(pages) -> None:
    """Фиксирует, какой именно файл сейчас лежит под кадром CVAT.

    Вызывается только после того, как задача действительно создана и наполнена: иначе
    полоса считалась бы залитой, а её в CVAT нет, и расхождение никогда бы не всплыло.
    """
    for page in pages:
        if page.file_hash is not None:
            page.cvat_file_hash = page.file_hash


def run_publish(params: PublishParams, session_factory) -> PublishStats:
    """Полный прогон ``to-cvat``."""
    stats = PublishStats()
    settings = params.settings or CvatSettings()
    backup_dir = params.backup_dir or params.db_path.parent / "cvat_backup"

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

                _drop_stale_temp_task(client, project.id, year.name)
                title = year_task_name(year.name, len(job_files), len(pages))
                task = find_year_task(client, project.id, year.name, year.cvat_task_id)
                rebuilt = False

                if task is None:
                    task = create_year_task(client, project.id, title, job_files)
                    stats.tasks_created += 1
                    upload = True
                else:
                    stats.tasks_existing += 1
                    if task.name != title:
                        # Имя несёт число выпусков и полос, а они меняются при обновлении
                        # пака. Найдена задача по id, так что переименование безопасно.
                        rename_task(task, title)
                    upload = params.force_annotations
                    drift = year_drift(issues)
                    if drift:
                        changed = sum(len(pair[1]) for pair in drift)
                        added = sum(len(pair[2]) for pair in drift)
                        stats.pages_changed += changed
                        stats.pages_unpublished += added
                        stats.stale_years.append(year.name)
                        report_drift(year.name, drift)
                        if params.recreate_stale:
                            changed_names = {
                                page.cvat_rel_path for _issue, pages_changed, _a in drift for page in pages_changed
                            }
                            by_frame = fetch_shapes_by_frame(task)
                            _backup_annotations(backup_dir, pack.name, year.name, task, by_frame)

                            def carry(frames, _by=by_frame, _skip=changed_names, _pages=pages):
                                # Переносим ручную разметку со всех кадров, кроме
                                # изменившихся, и добавляем свежую автоматическую на сами
                                # изменившиеся — их-то detect уже пересчитал.
                                carried = shapes_to_requests(_by, frames, _skip)
                                stats.shapes_carried += len(carried)
                                fresh = raster_shapes(
                                    ((p, p.raster_regions) for p in _pages if p.cvat_rel_path in _skip),
                                    frames,
                                    label_ids,
                                )
                                return carried + fresh

                            # Выпуски, куда добавились полосы, «завершёнными» больше не
                            # считаются: разметчик этих кадров не видел.
                            reset_jobs = {
                                index
                                for index, issue in enumerate(issues)
                                for drift_issue, _c, added in drift
                                if drift_issue is issue and added
                            }
                            task = rebuild_year_task(
                                client, project.id, year.name, title, task, job_files, carry, reset_jobs
                            )
                            stats.tasks_rebuilt += 1
                            rebuilt = True
                            upload = False  # разметка уже залита при пересоздании
                        else:
                            logger.warning(
                                "Задача %r НЕ пересоздаётся. CVAT не умеет удалять обычные джобы "
                                "(только ground truth), поэтому обновить эти выпуски можно лишь "
                                "пересозданием всей задачи-года. Флаг --recreate-stale делает это, "
                                "перенося ручную разметку со всех неизменившихся полос.",
                                year.name,
                            )
                            continue

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

                if upload:
                    shapes = raster_shapes(((page, page.raster_regions) for page in pages), frames, label_ids)
                    stats.shapes += upload_preannotations(task, shapes)
                    logger.info("Задача %r: залито шейпов %d", year.name, len(shapes))
                elif not rebuilt:
                    logger.info(
                        "Задача %r: предразметка НЕ трогается (заливка заменяет разметку "
                        "целиком и затёрла бы ручную правку); нужна — укажите --force-annotations",
                        year.name,
                    )

                # Только те, что реально сопоставились с кадром: полоса, не доехавшая до
                # задачи, не должна считаться залитой — иначе её пропажа больше нигде
                # не всплывёт.
                _mark_published([page for page in pages if page.cvat_frame is not None])
                session.commit()

                if params.annotator:
                    assign_annotator(client, task, params.annotator)

        session.commit()
    return stats
