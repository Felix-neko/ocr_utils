"""Пересчёт по УЖЕ найденным областям, без повторной детекции.

ЗАЧЕМ ОТДЕЛЬНАЯ КОМАНДА. Полный прогон ``detect`` по паку-1 — это полтерабайта чтения с
NTFS-3G, полтора-два часа в лучшем случае. А правка порога color/grayscale не меняет ни
одной координаты: области известны, надо лишь перекрасить их. Читать ради этого весь пак
незачем — достаточно тех полос, у которых области есть (в паке-1 их 693 из 12 135, то есть
28 ГБ вместо 480).

``mark-covers`` не читает и этого: полоса 0 каждого выпуска помечается обложкой по данным,
которые уже лежат в базе (порядковый номер и размеры кадра).

Обе команды правят разметку ``source='auto'`` и не трогают ручную из CVAT: перекрашивать
то, что человек уже проверил глазами, нельзя. ``recolor`` вдобавок обходит стороной области
``stamp_suspect``: у них ``kind`` означает не цвет, а происхождение.
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from ocr_utils.background_smoothing.processing import HALFTONE_DOWNSCALE
from ocr_utils.scan_markup.db.models import (
    KIND_COLOR,
    KIND_GRAYSCALE,
    SOURCE_AUTO,
    Issue,
    Page,
    RasterRegion,
    YearPackage,
)
from ocr_utils.scan_markup.db.repo import require_pack
from ocr_utils.scan_markup.detection.color_kind import (
    CHROMA_SELF_FRAC_THR,
    CHROMA_SPREAD_THR,
    CHROMA_THR,
    COLOR_FRAC_THR,
    classify,
    paper_color,
)
from ocr_utils.scan_markup.detection.cover import cover_region

logger = logging.getLogger(__name__)


@dataclass
class RecolorParams:
    """Параметры ``recolor`` и ``mark-covers``."""

    db_path: Path
    pack_name: str
    pack_dir: Path | None = None
    jobs: int = 8
    dry_run: bool = False
    chroma_thr: float = CHROMA_THR
    color_frac_thr: float = COLOR_FRAC_THR
    chroma_spread_thr: float = CHROMA_SPREAD_THR
    chroma_self_frac_thr: float | None = CHROMA_SELF_FRAC_THR


@dataclass
class RecolorStats:
    """Итоги: сколько полос прочитано и сколько областей сменило решение."""

    pages: int = 0
    regions: int = 0
    changed: int = 0
    failed: int = 0


@dataclass(frozen=True)
class _Job:
    """Полоса и её области в координатах оригинала."""

    path: Path
    page_id: int
    boxes: list[tuple[int, int, int, int]]
    params: RecolorParams


@dataclass
class _Result:
    page_id: int
    kinds: list[tuple[str, float, float, float]]
    error: str = ""


def _recolor_one(job: _Job) -> _Result:
    """Перекрашивает области одной полосы. Ошибка возвращается значением, не исключением."""
    try:
        bgr = cv2.imread(str(job.path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("не читается как изображение")
        height, width = bgr.shape[:2]
        work = cv2.resize(
            bgr,
            (max(1, width // HALFTONE_DOWNSCALE), max(1, height // HALFTONE_DOWNSCALE)),
            interpolation=cv2.INTER_AREA,
        )
        paper = paper_color(work)

        kinds = []
        for x1, y1, x2, y2 in job.boxes:
            color = classify(
                bgr[max(0, y1) : min(y2, height), max(0, x1) : min(x2, width)],
                paper,
                job.params.chroma_thr,
                job.params.color_frac_thr,
                job.params.chroma_spread_thr,
                job.params.chroma_self_frac_thr,
            )
            kinds.append((color.kind, color.chroma_frac, color.chroma_spread, color.chroma_self_frac))
        return _Result(job.page_id, kinds)
    except Exception as exc:  # noqa: BLE001 — одна битая полоса не должна валить прогон
        return _Result(job.page_id, [], str(exc))


def run_recolor(params: RecolorParams, session_factory) -> RecolorStats:
    """Пересчитывает ``kind`` у автоматических областей, не трогая координаты."""
    stats = RecolorStats()
    with session_factory() as session:  # type: Session
        pack = require_pack(session, params.pack_name)
        root = params.pack_dir or Path(pack.root_path)

        jobs: list[_Job] = []
        regions_by_page: dict[int, list[RasterRegion]] = {}
        for page in _pages_with_regions(session, pack.id):
            # Ровно два вида, и они перечислены прямо здесь, а не взяты из ``PICTURE_KINDS``:
            # перекраска умеет решать только «цветная или серая», и всё, что означает не цвет,
            # она бы испортила. У ``stamp_suspect`` вид означает происхождение (мелкий цветной
            # штрих) — перекраска превратила бы печать в картинку. У ``color_text`` — решение
            # разметчика, и перезаписывать его измерением тем более нельзя.
            regions = [
                region
                for region in page.raster_regions
                if region.source == SOURCE_AUTO and region.kind in (KIND_COLOR, KIND_GRAYSCALE)
            ]
            if not regions:
                continue
            regions_by_page[page.id] = regions
            jobs.append(_Job(root / page.rel_path, page.id, [(r.x1, r.y1, r.x2, r.y2) for r in regions], params))

        for result in _iter_recolor(jobs, params.jobs):
            if result.error:
                stats.failed += 1
                tqdm.write(f"ОШИБКА page_id={result.page_id}: {result.error}")
                continue
            stats.pages += 1
            for region, (kind, frac, spread, self_frac) in zip(regions_by_page[result.page_id], result.kinds):
                stats.regions += 1
                if region.kind != kind:
                    stats.changed += 1
                if not params.dry_run:
                    region.kind = kind
                    region.chroma_frac = frac
                    region.chroma_spread = spread
                    region.chroma_self_frac = self_frac
        if not params.dry_run:
            session.commit()
    return stats


def _iter_recolor(jobs: list[_Job], workers: int):
    """Результаты по полосам — в пуле процессов либо последовательно при ``--jobs 1``."""
    if not jobs:
        return
    if workers <= 1:
        for job in tqdm(jobs, desc="перекраска", unit="полоса"):
            yield _recolor_one(job)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from tqdm(pool.map(_recolor_one, jobs, chunksize=4), total=len(jobs), desc="перекраска", unit="полоса")


def _pages_with_regions(session: Session, pack_id: int):
    """Полосы пака, у которых есть хоть одна растровая область."""
    return session.scalars(
        select(Page)
        .join(Issue, Issue.id == Page.issue_id)
        .join(YearPackage, YearPackage.id == Issue.year_package_id)
        .where(YearPackage.pack_id == pack_id)
        .where(Page.raster_regions.any())
        .order_by(Page.id)
    ).all()


def run_mark_covers(params: RecolorParams, session_factory) -> RecolorStats:
    """Помечает первую полосу каждого выпуска обложкой во весь кадр. Пикселей не читает.

    Автоматическая разметка такой полосы заменяется целиком: детектор на обложке всё равно
    ничего осмысленного не находил (замер по паку-1: на 1969/09 IMG_0103_2R он выдал синюю
    плашку заголовка, две тени разворота по краям и цифру «9»).
    """
    stats = RecolorStats()
    with session_factory() as session:  # type: Session
        pack = require_pack(session, params.pack_name)
        for year in pack.year_packages:
            for issue in year.issues:
                page = next((p for p in issue.pages if p.order_index == 0), None)
                if page is None:
                    continue
                if page.width is None or page.height is None:
                    tqdm.write(f"ПРОПУСК {page.rel_path}: нет размеров, сначала нужен detect")
                    stats.failed += 1
                    continue
                if any(region.source != SOURCE_AUTO for region in page.raster_regions):
                    continue  # полосу уже правили руками — не трогаем
                stats.pages += 1
                if _already_cover(page):
                    continue
                stats.changed += 1
                if params.dry_run:
                    continue
                x1, y1, x2, y2 = cover_region(page.width, page.height)
                page.raster_regions = [
                    RasterRegion(x1=x1, y1=y1, x2=x2, y2=y2, kind=KIND_COLOR, full_page=True, source=SOURCE_AUTO)
                ]
        if not params.dry_run:
            session.commit()
        stats.regions = stats.pages
    return stats


def _already_cover(page: Page) -> bool:
    """Помечена ли полоса ровно одной цветной областью во весь кадр."""
    if len(page.raster_regions) != 1:
        return False
    region = page.raster_regions[0]
    return (
        region.full_page
        and region.kind == KIND_COLOR
        and (region.x2 - region.x1, region.y2 - region.y1) == (page.width, page.height)
    )
