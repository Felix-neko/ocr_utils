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
from ocr_utils.scan_markup.cvat.project import KIND_BY_LABEL, MASK_KIND_BY_LABEL
from ocr_utils.scan_markup.db.models import (
    KIND_COLOR,
    KIND_GRAYSCALE,
    SOURCE_CVAT,
    Issue,
    MaskAnnotation,
    Pack,
    Page,
    RasterRegion,
    YearPackage,
)
from ocr_utils.scan_markup.db.repo import get_pack, require_pack
from ocr_utils.scan_markup.detection.raster import FULL_PAGE_FRAC
from ocr_utils.scan_markup.geometry import mask_to_original, rect_to_original

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
    full_page: int = 0
    masks: int = 0
    unknown_labels: int = 0
    unmatched_frames: int = 0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def copy_tree(src_session: Session, dst_session: Session, pack_name: str) -> Pack:
    """Копирует дерево пака (без разметки) в целевую базу и возвращает копию пака.

    Если пак в целевой базе уже есть, он ПЕРЕСОЗДАЁТСЯ: ``from-cvat`` — это снимок
    состояния разметки на данный момент, и подмешивать в него остатки прошлого снимка
    (области, которые разметчик с тех пор удалил) было бы прямой ошибкой.
    """
    source = require_pack(src_session, pack_name)

    existing = get_pack(dst_session, pack_name)
    if existing is not None:
        dst_session.delete(existing)
        dst_session.flush()

    pack = Pack(name=source.name, root_path=source.root_path, cvat_project_id=source.cvat_project_id)
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
            )
            dst_session.add(issue)
            dst_session.flush()
            for src_page in src_issue.pages:
                dst_session.add(
                    Page(
                        issue_id=issue.id,
                        file_name=src_page.file_name,
                        rel_path=src_page.rel_path,
                        order_index=src_page.order_index,
                        width=src_page.width,
                        height=src_page.height,
                        dpi=src_page.dpi,
                        divisor=src_page.divisor,
                        crop_width=src_page.crop_width,
                        crop_height=src_page.crop_height,
                        cvat_rel_path=src_page.cvat_rel_path,
                        cvat_width=src_page.cvat_width,
                        cvat_height=src_page.cvat_height,
                        cvat_frame=src_page.cvat_frame,
                        detected_at=src_page.detected_at,
                    )
                )
            dst_session.flush()

    dst_session.commit()
    return pack


def shape_to_region(shape, page: Page, full_page_frac: float) -> RasterRegion:
    """Прямоугольный шейп CVAT -> строка ``raster_regions`` в координатах оригинала."""
    x1, y1, x2, y2 = rect_to_original(*shape.points[:4], page.divisor, page.width, page.height)
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return RasterRegion(
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
    shapes_by_frame: dict[int, list] = {}
    for shape in task.get_annotations().shapes:
        if shape.frame not in pages_by_frame:
            stats.unmatched_frames += 1
            continue
        shapes_by_frame.setdefault(shape.frame, []).append(shape)

    for frame, page in pages_by_frame.items():
        regions: list[RasterRegion] = []
        masks: list[MaskAnnotation] = []

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
                stats.full_page += bool(region.full_page)
            elif shape_type == "mask" and name in MASK_KIND_BY_LABEL:
                mask = shape_to_mask(shape, page)
                if mask is not None:
                    masks.append(mask)
                    stats.masks += 1
            else:
                stats.unknown_labels += 1
                logger.debug("Пропускаю шейп типа %r с меткой %r", shape_type, name)

        # Через коллекции связей: delete-orphan сам уберёт вытесненные строки, а объекты
        # остаются согласованы с тем, что увидит следующий обход этой же сессии.
        page.raster_regions = regions
        page.masks = masks
        page.reviewed_at = _utcnow()
        stats.pages += 1

    session.commit()


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
