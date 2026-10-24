"""Слои страниц ``scan_cropping.v2 --output-pages`` в базе разметки: зона контента (способом D — отступы с защитой контента; у обложки — вся страница) и зоны закраса пальцев из паспортов страниц; ручная зона из CVAT не перезаписывается."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.db.models import SOURCE_AUTO, SOURCE_CVAT, ContentBox, FingerZone
from ocr_utils.db.repo import iter_pages, require_pack
from ocr_utils.scan_cropping.v2.pipeline import init_worker, read_bgr
from ocr_utils.scan_cropping.v2.rectify.passport import PASSPORT_SUFFIX, PagePassport, read_passport
from ocr_utils.scan_markup.content_box.content import Box, page_content
from ocr_utils.scan_markup.content_box.methods import Method, TrimParams, box_of
from ocr_utils.units import MM_PER_INCH

logger = logging.getLogger(__name__)

# Версия расчёта зоны контента: поднимать при смене способа или чисел — по ней видно, что зону пора пересчитать.
CONTENT_BOX_VERSION = 1
# Способ по умолчанию — выбор стенда research/page_trim: без потерь контента, обрезка не меньше отступов.
DEFAULT_METHOD = Method.HYBRID


@dataclass(frozen=True)
class LayersParams:
    """Параметры ``page-layers``.

    Поля:
        db_path: База разметки.
        pack_name: Пак.
        pack_dir: Папка страниц пака (``--output-pages`` v2; относительные пути полос — от неё).
        meta_dir: Папка паспортов (``--output-pages-meta``) с той же раскладкой.
        method: Способ зоны контента.
        trim: Числа способа (отступы, запас), мм.
        redo: Пересчитать и полосы, у которых зона этой версии уже есть (ручную из CVAT — никогда).
        jobs: Процессов пула.
    """

    db_path: Path
    pack_name: str
    pack_dir: Path
    meta_dir: Path | None
    method: Method = DEFAULT_METHOD
    trim: TrimParams = TrimParams()
    redo: bool = False
    jobs: int = 12


@dataclass(frozen=True)
class PageLayers:
    """Слои одной полосы: зона контента и закрас пальцев.

    Поля:
        rel: Путь полосы относительно пака.
        box: Зона контента px полосы или None (ошибка).
        fingers: Контуры закраса пальцев и площадь, мм², по паспорту.
        message: Ошибка или пусто.
    """

    rel: str
    box: Box | None
    fingers: tuple[tuple[list, float], ...] = ()
    message: str = ""


def part_of(rel: str) -> str:
    """Часть кадра по имени файла страницы v2 (``_L``, ``_R``, ``_LR``, ``_P`` — одна страница); не страница v2 — ``LR`` (без стороны разреза)."""
    stem = Path(rel).stem
    for suffix, part in (("_LR", "LR"), ("_L", "L"), ("_R", "R"), ("_P", "P")):
        if stem.endswith(suffix):
            return part
    return "LR"


def passport_for(meta_dir: Path | None, rel: str) -> PagePassport | None:
    """Паспорт полосы ``meta_dir/<rel без расширения>.json`` или None."""
    if meta_dir is None:
        return None
    path = meta_dir / Path(rel).with_suffix(PASSPORT_SUFFIX)
    return read_passport(path) if path.is_file() else None


def page_layers(args: tuple[Path, str, float, Path | None, Method, TrimParams]) -> PageLayers:
    """Слои одной полосы (воркер пула).

    Args:
        args: (папка пака, путь полосы, dpi из базы, папка паспортов, способ, числа способа).

    Returns:
        Слои полосы.
    """
    pack_dir, rel, dpi, meta_dir, method, trim = args
    try:
        passport = passport_for(meta_dir, rel)
        part = passport.part if passport else part_of(rel)
        cover = bool(passport and passport.cover)
        page = read_bgr(pack_dir / rel)
        if cover:
            box = (0, 0, page.shape[1], page.shape[0])
        else:
            content = page_content(page, passport.dpi if passport else dpi, part)
            box = box_of(method, content, trim, cover=False)
        fingers = ()
        if passport and passport.finger_zones:
            # Площадь каждой зоны — по её многоугольнику (в паспорте площадь общая на страницу).
            px2_per_mm2 = (passport.dpi / MM_PER_INCH) ** 2
            fingers = tuple(
                (zone, float(cv2.contourArea(np.array(zone, np.float32))) / px2_per_mm2)
                for zone in passport.finger_zones
            )
        return PageLayers(rel, box, fingers)
    except Exception as exc:  # noqa: BLE001 — полоса с ошибкой не роняет прогон
        logger.exception("%s: слои не посчитаны", rel)
        return PageLayers(rel, None, message=repr(exc))


@dataclass
class LayersStats:
    """Итоги ``page-layers``."""

    pages: int = 0
    boxes: int = 0
    covers: int = 0
    finger_pages: int = 0
    kept_manual: int = 0
    skipped: int = 0
    failed: int = 0


def run_layers(params: LayersParams, session_factory) -> LayersStats:
    """Считает слои страниц пака и пишет их в базу.

    Args:
        params: Параметры.
        session_factory: Фабрика сессий базы (``open_db``).

    Returns:
        Итоги.
    """
    stats = LayersStats()
    with session_factory() as session:
        pack = require_pack(session, params.pack_name)
        pages = {page.source_rel_path: page for _y, _i, page in iter_pages(pack)}
        todo = []
        for rel, page in pages.items():
            box = page.content_box
            if box is not None and box.source == SOURCE_CVAT:
                stats.kept_manual += 1  # ручную зону не трогаем; закрас пальцев всё равно обновим ниже
            elif box is not None and box.version == CONTENT_BOX_VERSION and not params.redo:
                stats.skipped += 1
                continue
            todo.append((params.pack_dir, rel, float(page.dpi or 0), params.meta_dir, params.method, params.trim))
        ctx = mp.get_context("forkserver")
        with ProcessPoolExecutor(params.jobs, mp_context=ctx, initializer=init_worker) as pool:
            for result in pool.map(page_layers, todo, chunksize=2):
                stats.pages += 1
                if result.box is None:
                    stats.failed += 1
                    continue
                page = pages[result.rel]
                if page.content_box is None or page.content_box.source != SOURCE_CVAT:
                    x0, y0, x1, y1 = result.box
                    full = (x0, y0) == (0, 0) and (x1, y1) == (page.width, page.height)
                    page.content_box = ContentBox(x1=x0, y1=y0, x2=x1, y2=y1, version=CONTENT_BOX_VERSION,
                                                  method="cover" if full else params.method.value,
                                                  source=SOURCE_AUTO)  # fmt: skip
                    stats.boxes += 1
                    stats.covers += full
                page.finger_zones = [FingerZone(polygon=json.dumps(z), area_mm2=round(a, 1)) for z, a in result.fingers]
                stats.finger_pages += bool(result.fingers)
                if stats.pages % 50 == 0:
                    session.commit()
                    logger.info("полос %d/%d", stats.pages, len(todo))
        session.commit()
    return stats
