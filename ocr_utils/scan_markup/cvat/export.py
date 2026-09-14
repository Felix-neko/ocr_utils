"""Команда ``from-cvat``: уточнённая разметка из CVAT -> вторая база той же схемы.

Аннотации читаются шейпами через ``task.get_annotations()`` — без выгрузки датасета в
``CVAT for images 1.1`` или Datumaro. Файл формата пришлось бы заказывать, ждать, качать,
распаковывать и парсить XML ради тех же самых ``points``, которые API отдаёт сразу.

Дерево пак/год/выпуск/полоса копируется из исходной базы как есть: там уже лежат
идентификаторы CVAT и параметры уменьшения, без которых пересчёт координат невозможен.
Заново заполняются только растровые области и маски, с ``source='cvat'``.

Пересчёт координат — в ``scan_markup.geometry``: умножение на делитель плюс
распространение в полоску, обрезанную при уменьшении.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session
from tqdm import tqdm

from ocr_utils.scan_markup.cvat.client import CvatSettings, make_cvat_client
from ocr_utils.scan_markup.cvat.project import (
    FIELD_BY_TOC_LABEL,
    KIND_BY_LABEL,
    MASK_KIND_BY_LABEL,
    POINT_KIND_BY_LABEL,
    ROTATION_BY_LABEL,
)
from ocr_utils.scan_markup.db.models import (
    KIND_COLOR,
    KIND_COLOR_TEXT,
    KIND_GRAYSCALE,
    KIND_LINE_ART_SCHEMA,
    KIND_TABLE,
    SOURCE_CVAT,
    Issue,
    MaskAnnotation,
    Pack,
    Page,
    PointAnnotation,
    RectRegion,
    YearPackage,
)
from ocr_utils.scan_markup.db.repo import get_pack, require_pack
from ocr_utils.scan_markup.detection.boxes import FULL_PAGE_FRAC
from ocr_utils.scan_markup.geometry import mask_to_original, point_to_original, rect_to_original

logger = logging.getLogger(__name__)


@dataclass
class ExportParams:
    """Параметры прогона ``from-cvat``."""

    db_path: Path
    out_db_path: Path
    pack_name: str
    only_year: str | None = None
    full_page_frac: float = FULL_PAGE_FRAC
    settings: CvatSettings | None = None


@dataclass
class ExportStats:
    """Итоги прогона."""

    pages: int = 0
    regions: int = 0
    color: int = 0
    grayscale: int = 0
    # Цветной набор считается отдельно от растра: это ручная метка, и её счётчик показывает,
    # сколько ручной работы вообще сделано. Без него такие области видны только в общей сумме
    # ``regions``, и слагаемые в скобках с ней не сходятся.
    color_text: int = 0
    # Таблицы и схемы — тоже отдельно: это другой детектор, и его счётчики отвечают на свой
    # вопрос — сколько из автоматических находок разметчик оставил.
    table: int = 0
    line_art: int = 0
    full_page: int = 0
    masks: int = 0
    points: int = 0
    rotated: int = 0
    # Кадры, где разметчик оставил сразу несколько взаимоисключающих тегов поворота.
    conflicting_rotations: int = 0
    # Полосы с тегами оглавления: «Содержание» выпуска и указатель за год.
    toc_pages: int = 0
    year_index_pages: int = 0
    unknown_labels: int = 0
    unmatched_frames: int = 0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def copy_tree(src_session: Session, dst_session: Session, pack_name: str) -> Pack:
    """Копирует дерево пака (без разметки) в целевую базу и возвращает копию пака.

    Если пак в целевой базе уже есть, он ПЕРЕСОЗДАЁТСЯ: ``from-cvat`` — это снимок
    состояния разметки на данный момент, и подмешивать в него остатки прошлого снимка
    (области, которые разметчик с тех пор удалил) было бы прямой ошибкой.

    ВНИМАНИЕ ПРИ ДОБАВЛЕНИИ КОЛОНОК. Поля пака, выпуска и полосы перечисляются здесь
    ПОИМЁННО, и колонка, забытая в этом списке, потеряется при следующем же ``from-cvat``
    молча: пак-то пересоздаётся с нуля. Поэтому корни путей пака, имена собранных PDF и
    номера страниц в них копируются наравне с разметкой, хотя к CVAT отношения не имеют.
    """
    source = require_pack(src_session, pack_name)

    existing = get_pack(dst_session, pack_name)
    if existing is not None:
        dst_session.delete(existing)
        dst_session.flush()

    pack = Pack(
        name=source.name,
        source_pics_root=source.source_pics_root,
        cvat_project_id=source.cvat_project_id,
        cleaned_pics_root=source.cleaned_pics_root,
        sharpened_text_pics_root=source.sharpened_text_pics_root,
        full_intermediate_pdf_root=source.full_intermediate_pdf_root,
        pages_with_pics_only_intermediate_pdf_root=source.pages_with_pics_only_intermediate_pdf_root,
        final_pdfs_root=source.final_pdfs_root,
        allowed_rotations=source.allowed_rotations,
    )
    dst_session.add(pack)
    dst_session.flush()

    for src_year in source.year_packages:
        year = YearPackage(
            pack_id=pack.id,
            name=src_year.name,
            year=src_year.year,
            rel_path=src_year.rel_path,
            cvat_task_id=src_year.cvat_task_id,
        )
        dst_session.add(year)
        dst_session.flush()
        for src_issue in src_year.issues:
            issue = Issue(
                year_package_id=year.id,
                name=src_issue.name,
                number=src_issue.number,
                rel_path=src_issue.rel_path,
                cvat_job_id=src_issue.cvat_job_id,
                full_intermediate_pdf_name=src_issue.full_intermediate_pdf_name,
                pages_with_pics_only_intermediate_pdf_name=src_issue.pages_with_pics_only_intermediate_pdf_name,
                final_pdf_name=src_issue.final_pdf_name,
                allowed_rotations=src_issue.allowed_rotations,
            )
            dst_session.add(issue)
            dst_session.flush()
            for src_page in src_issue.pages:
                dst_session.add(
                    Page(
                        issue_id=issue.id,
                        source_file_name=src_page.source_file_name,
                        source_rel_path=src_page.source_rel_path,
                        order_index=src_page.order_index,
                        width=src_page.width,
                        height=src_page.height,
                        dpi=src_page.dpi,
                        divisor=src_page.divisor,
                        crop_width=src_page.crop_width,
                        crop_height=src_page.crop_height,
                        file_size=src_page.file_size,
                        file_mtime=src_page.file_mtime,
                        file_hash=src_page.file_hash,
                        hash_algo=src_page.hash_algo,
                        cvat_file_hash=src_page.cvat_file_hash,
                        cvat_rel_path=src_page.cvat_rel_path,
                        cvat_width=src_page.cvat_width,
                        cvat_height=src_page.cvat_height,
                        cvat_frame=src_page.cvat_frame,
                        cleaned_file_name=src_page.cleaned_file_name,
                        cleaned_rel_path=src_page.cleaned_rel_path,
                        cleaned_grayscale=src_page.cleaned_grayscale,
                        sharpened_text_pic_file_name=src_page.sharpened_text_pic_file_name,
                        sharpened_text_pic_rel_path=src_page.sharpened_text_pic_rel_path,
                        full_pdf_page_idx=src_page.full_pdf_page_idx,
                        pages_with_pics_only_pdf_page_idx=src_page.pages_with_pics_only_pdf_page_idx,
                        rotate_cw=src_page.rotate_cw,
                        orientation_confidence=src_page.orientation_confidence,
                        orientation_version=src_page.orientation_version,
                        orientation_detected_at=src_page.orientation_detected_at,
                        orientation_source=src_page.orientation_source,
                        cleaned_rotate_cw=src_page.cleaned_rotate_cw,
                        detected_at=src_page.detected_at,
                        detector_version=src_page.detector_version,
                        table_detector_version=src_page.table_detector_version,
                        tables_detected_at=src_page.tables_detected_at,
                        is_toc=src_page.is_toc,
                        is_year_index=src_page.is_year_index,
                        toc_score=src_page.toc_score,
                        toc_source=src_page.toc_source,
                        toc_version=src_page.toc_version,
                        toc_detected_at=src_page.toc_detected_at,
                    )
                )
            dst_session.flush()

    dst_session.commit()
    return pack


def shape_to_region(shape, page: Page, full_page_frac: float) -> RectRegion:
    """Прямоугольный шейп CVAT -> строка ``rect_regions`` в координатах оригинала."""
    x1, y1, x2, y2 = rect_to_original(*shape.points[:4], page.divisor, page.width, page.height)
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return RectRegion(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        kind=KIND_BY_LABEL[shape._label_name],
        full_page=area >= full_page_frac * page.width * page.height,
        source=SOURCE_CVAT,
        cvat_shape_id=getattr(shape, "id", None),
    )


def shape_to_mask(shape, page: Page) -> MaskAnnotation | None:
    """Маска-шейп CVAT -> строка ``mask_annotations`` в координатах оригинала.

    ``points`` маски — это длины пробегов, а следом четыре числа: ``left, top, right,
    bottom`` охватывающего прямоугольника (правый и нижний края ВКЛЮЧИТЕЛЬНО). Декодируем
    в кадре CVAT, апскейлим с распространением в обрезанную полоску и кодируем заново уже
    в разрешении оригинала — чтобы потребителю не пришлось помнить ни про делитель, ни
    про обрезку.
    """
    from cvat_sdk.masks import decode_mask, encode_mask

    small = decode_mask(
        [int(round(value)) for value in shape.points], image_width=page.cvat_width, image_height=page.cvat_height
    )
    full = mask_to_original(small, page.divisor, page.width, page.height)
    if not full.any():
        return None

    encoded = encode_mask(full)
    runs, (left, top, right, bottom) = encoded[:-4], encoded[-4:]
    return MaskAnnotation(
        kind=MASK_KIND_BY_LABEL.get(shape._label_name, shape._label_name),
        left=int(left),
        top=int(top),
        width=int(right) - int(left) + 1,
        height=int(bottom) - int(top) + 1,
        rle=",".join(str(int(run)) for run in runs),
        source_divisor=page.divisor,
        source=SOURCE_CVAT,
        cvat_shape_id=getattr(shape, "id", None),
    )


def shape_to_point(shape, page: Page) -> PointAnnotation:
    """Точечный шейп CVAT -> строка ``point_annotations`` в координатах оригинала.

    У шейпа типа ``points`` в ``points`` лежат пары координат; берём первую. Разметчик может
    поставить несколько точек одним объектом, но метка «Экслибрис» означает одно место, и
    трактовать вторую точку было бы гаданием.
    """
    x, y = point_to_original(shape.points[0], shape.points[1], page.divisor, page.width, page.height)
    return PointAnnotation(
        kind=POINT_KIND_BY_LABEL.get(shape._label_name, shape._label_name),
        x=x,
        y=y,
        source_divisor=page.divisor,
        source=SOURCE_CVAT,
        cvat_shape_id=getattr(shape, "id", None),
    )


def mask_from_row(row: MaskAnnotation, width: int, height: int) -> np.ndarray:
    """Обратное чтение маски из базы — bool-массив во весь кадр оригинала.

    Держится здесь, а не у потребителя, чтобы формат хранения знал ровно один модуль.
    """
    from cvat_sdk.masks import decode_mask

    points = [int(value) for value in row.rle.split(",")]
    points += [row.left, row.top, row.left + row.width - 1, row.top + row.height - 1]
    return decode_mask(points, image_width=width, image_height=height)


def _shape_type(shape) -> str:
    """Тип шейпа строкой: SDK отдаёт его enum'ом, а сравнивать удобнее со строкой."""
    value = getattr(shape, "type", None)
    return getattr(value, "value", value)


def _rotation_from_tags(tags: list, label_names: dict[int, str], stats: "ExportStats") -> int:
    """Угол поворота по тегам кадра. Нет тегов — 0, «поворот не нужен».

    Несколько взаимоисключающих тегов на одном кадре — это ошибка разметки, а не повод
    молча выбрать любой: берётся наибольший угол и пишется предупреждение. Наибольший, а не
    первый попавшийся, чтобы результат хотя бы не зависел от порядка выдачи сервера.
    """
    angles = sorted(
        {ROTATION_BY_LABEL[name] for tag in tags if (name := label_names.get(tag.label_id)) in ROTATION_BY_LABEL}
    )
    if not angles:
        return 0
    if len(angles) > 1:
        stats.conflicting_rotations += 1
        logger.warning("На кадре несколько тегов поворота (%s) — беру %d", angles, angles[-1])
    return angles[-1]


def _toc_from_tags(tags: list, label_names: dict[int, str]) -> dict[str, bool]:
    """Признаки оглавления по тегам кадра: ``{"is_toc": ..., "is_year_index": ...}``.
    Нет тега — False, «не оглавление»; правило то же, что у поворота."""
    flags = {field_name: False for field_name in FIELD_BY_TOC_LABEL.values()}
    for tag in tags:
        field_name = FIELD_BY_TOC_LABEL.get(label_names.get(tag.label_id, ""))
        if field_name is not None:
            flags[field_name] = True
    return flags


def import_task(
    task,
    label_names: dict[int, str],
    pages_by_frame: dict[int, Page],
    session: Session,
    params: ExportParams,
    stats: ExportStats,
) -> None:
    """Переносит разметку одной задачи-года в целевую базу.

    Разметка полосы ЗАМЕНЯЕТСЯ целиком: выгрузка — это снимок состояния, а не добавка к
    прошлому. Полосы задачи, на которых разметчик ничего не нарисовал, получают пустой
    список и отметку ``reviewed_at``: «здесь растра нет» — осмысленный результат, и
    отличать его от «сюда ещё не смотрели» нужно обязательно.
    """
    annotations = task.get_annotations()
    shapes_by_frame: dict[int, list] = {}
    for shape in annotations.shapes:
        if shape.frame not in pages_by_frame:
            stats.unmatched_frames += 1
            continue
        shapes_by_frame.setdefault(shape.frame, []).append(shape)

    tags_by_frame: dict[int, list] = {}
    for tag in annotations.tags:
        if tag.frame in pages_by_frame:
            tags_by_frame.setdefault(tag.frame, []).append(tag)

    for frame, page in pages_by_frame.items():
        regions: list[RectRegion] = []
        masks: list[MaskAnnotation] = []
        points: list[PointAnnotation] = []

        for shape in shapes_by_frame.get(frame, []):
            name = label_names.get(shape.label_id)
            shape_type = _shape_type(shape)
            shape._label_name = name

            if shape_type == "rectangle" and name in KIND_BY_LABEL:
                region = shape_to_region(shape, page, params.full_page_frac)
                regions.append(region)
                stats.regions += 1
                stats.color += region.kind == KIND_COLOR
                stats.grayscale += region.kind == KIND_GRAYSCALE
                stats.color_text += region.kind == KIND_COLOR_TEXT
                stats.table += region.kind == KIND_TABLE
                stats.line_art += region.kind == KIND_LINE_ART_SCHEMA
                stats.full_page += bool(region.full_page)
            elif shape_type == "mask" and name in MASK_KIND_BY_LABEL:
                mask = shape_to_mask(shape, page)
                if mask is not None:
                    masks.append(mask)
                    stats.masks += 1
            elif shape_type == "points" and name in POINT_KIND_BY_LABEL:
                points.append(shape_to_point(shape, page))
                stats.points += 1
            else:
                stats.unknown_labels += 1
                logger.debug("Пропускаю шейп типа %r с меткой %r", shape_type, name)

        # Ориентация — по ТЕГАМ кадра, и отсутствие тега здесь осмысленно.
        #
        # Полоса без тега получает rotate_cw = 0, то есть «поворот не нужен», а НЕ NULL. Это
        # то же правило, по которому полоса без шейпов получает пустой список: выгрузка —
        # снимок состояния. Благодаря ему снятый разметчиком лишний тег обрабатывается сам
        # собой, без единой отдельной строчки кода.
        page.rotate_cw = _rotation_from_tags(tags_by_frame.get(frame, []), label_names, stats)
        stats.rotated += bool(page.rotate_cw)
        page.orientation_source = SOURCE_CVAT
        page.orientation_detected_at = _utcnow()
        # Уверенность и версия — свойства АВТОМАТИЧЕСКОГО вердикта, и к решению человека
        # отношения не имеют. Оставить их от прошлого прогона значило бы приписать ручной
        # правке уверенность детектора.
        page.orientation_confidence = None
        page.orientation_version = None

        # Оглавление — те же правила, что у поворота: снимок тегов, нет тега — False, и
        # свойства автоматического решения (сила, версия) к ручному не относятся.
        toc_flags = _toc_from_tags(tags_by_frame.get(frame, []), label_names)
        page.is_toc = toc_flags["is_toc"]
        page.is_year_index = toc_flags["is_year_index"]
        stats.toc_pages += page.is_toc
        stats.year_index_pages += page.is_year_index
        page.toc_source = SOURCE_CVAT
        page.toc_detected_at = _utcnow()
        page.toc_score = None
        page.toc_version = None

        # Через коллекции связей: delete-orphan сам уберёт вытесненные строки, а объекты
        # остаются согласованы с тем, что увидит следующий обход этой же сессии.
        page.rect_regions = regions
        page.masks = masks
        page.points = points
        page.reviewed_at = _utcnow()
        stats.pages += 1

    session.commit()


@dataclass
class CopyStats:
    """Итоги ``copy-regions``."""

    pages: int = 0
    regions: int = 0
    missing_pages: int = 0


def copy_regions(src_session: Session, dst_session: Session, pack_name: str, kinds: "tuple[str, ...]") -> CopyStats:
    """Переносит прямоугольники заданных видов из одной базы в другую, по пути полосы.

    ЗАЧЕМ. Уточнённая база (``from-cvat``) — снимок разметки CVAT, и находки нового детектора
    попадут в неё только после того, как разметчик их отсмотрит. Потребителям ниже по
    конвейеру они нужны раньше, поэтому автоматические находки копируются сюда как есть,
    с ``source = auto``: ``from-cvat`` потом заменит их уточнёнными.

    Переносятся ТОЛЬКО заданные виды, и в целевой базе заменяются только они
    (:func:`replace_rect_regions` с ``kinds``): ручной растр остаётся нетронутым. Полоса
    ищется по ``source_rel_path``; полосы, которых в целевой базе нет, считаются и
    пропускаются — пересоздавать дерево не наше дело, это делает ``from-cvat``.
    """
    from ocr_utils.scan_markup.db.repo import replace_rect_regions

    source = require_pack(src_session, pack_name)
    target = require_pack(dst_session, pack_name)
    by_path = {
        page.source_rel_path: page for year in target.year_packages for issue in year.issues for page in issue.pages
    }
    stats = CopyStats()
    for year in source.year_packages:
        for issue in year.issues:
            for page in issue.pages:
                twin = by_path.get(page.source_rel_path)
                if twin is None:
                    stats.missing_pages += 1
                    continue
                regions = [
                    RectRegion(
                        x1=region.x1,
                        y1=region.y1,
                        x2=region.x2,
                        y2=region.y2,
                        kind=region.kind,
                        full_page=region.full_page,
                        detector_info=region.detector_info,
                        source=region.source,
                    )
                    for region in page.rect_regions
                    if region.kind in kinds
                ]
                replace_rect_regions(dst_session, twin, regions, kinds=kinds)
                twin.table_detector_version = page.table_detector_version
                twin.tables_detected_at = page.tables_detected_at
                stats.pages += 1
                stats.regions += len(regions)
    dst_session.commit()
    return stats


def run_export(params: ExportParams, session_factory, out_session_factory) -> ExportStats:
    """Полный прогон ``from-cvat``: дерево из исходной базы + разметка из CVAT."""
    stats = ExportStats()
    settings = params.settings or CvatSettings()

    with session_factory() as src_session, out_session_factory() as dst_session:
        pack = copy_tree(src_session, dst_session, params.pack_name)

        with make_cvat_client(settings) as client:
            label_names = {
                label.id: label.name
                for label in client.api_client.labels_api.list(project_id=pack.cvat_project_id, page_size=100)[
                    0
                ].results
            }

            years = [year for year in pack.year_packages if params.only_year is None or year.name == params.only_year]
            for year in tqdm(years, desc="выгрузка", unit="год"):
                if year.cvat_task_id is None:
                    logger.warning("Год %s не заведён в CVAT (нет cvat_task_id), пропускаю", year.name)
                    continue
                task = client.tasks.retrieve(year.cvat_task_id)
                pages_by_frame = {
                    page.cvat_frame: page
                    for issue in year.issues
                    for page in issue.pages
                    if page.cvat_frame is not None
                }
                if not pages_by_frame:
                    logger.warning("Год %s: ни у одной полосы нет номера кадра, пропускаю", year.name)
                    continue
                import_task(task, label_names, pages_by_frame, dst_session, params, stats)

        dst_session.commit()
    return stats
