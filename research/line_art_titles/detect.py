"""Прогон единого детектора line art по паку: области, вырезки для OCR и блоки surya над ними.

Растр и таблицы берутся из проверенной базы разметки как известные (``known``), ровно как у
``scan_markup detect``: детектор line art их исключает, а сам детектор таблиц прогоняется
только ради затравок-схем. Surya — только из кэша (без модели): вариант ``sharpened`` полон
на все полосы пака-1.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.analysis import Find, LayoutOptions, PageLayout
from ocr_utils.page_layout.geometry import Box, intersection
from ocr_utils.page_layout.image import PageImage, Variant
from ocr_utils.page_layout.regions import Region, RegionKind
from ocr_utils.page_layout.surya.cache import SuryaCache

logger = logging.getLogger(__name__)

# Разрешение вырезок для OCR и контактных листов: у tesseract кегль текста 10 pt при 300 dpi
# уже читается, а крупный заголовок при 600 dpi только замедляет его.
CROP_DPI = 300

# Поле вокруг области при вырезке, мм, со всех сторон: буква на самом краю рамки читается хуже.
# Раньше по бокам поле было на высоту области (до 20 мм) — ради обрывков титула («ИЧНОЕ»), которые
# tesseract всё равно не дочитывал (слово срезано краем полосы), а DeepSeek-OCR-2 в широком поле
# читал соседнюю колонку и зацикливался на ней (1968/03 с.134). Прежние выходы — в ``pad20/``.
CROP_PAD_MM = 2.0


@dataclass(frozen=True)
class PageJob:
    """Одна полоса пака: где лежит заострённая копия и что о ней уже известно из базы.

    Attributes:
        name: Имя полосы без расширения относительно корня пака («1973/06/IMG_0103_2R»);
            оно же ключ кэша surya.
        path: Путь к заострённому JPEG.
        known_raster: Растровые области из базы ``(x0, y0, x1, y1, вид)`` в пикселях скана.
        known_tables: Таблицы из базы в том же виде.
    """

    name: str
    path: Path
    known_raster: tuple[tuple[int, int, int, int, str], ...]
    known_tables: tuple[tuple[int, int, int, int, str], ...]


def rotate_box(box: tuple[int, int, int, int, str], width: int, height: int, rotate_cw: int):
    """Рамка из координат скана → координаты копии, повёрнутой на ``rotate_cw`` градусов по часовой.

    Заострённые копии полос с боковым содержимым (таблица или схема стоят боком) экспортированы
    уже повёрнутыми, а рамки в базе записаны по исходному скану. Без пересчёта известные таблицы
    ложатся мимо, и их линейки уходят в line art (1974/07 с.16: таблица 332…3196 × 460…5736 на
    скане 3448 × 5963 — на копии 227…5503 × 332…3196).

    Args:
        box: ``(x0, y0, x1, y1, вид)`` в пикселях скана.
        width: Ширина скана, px.
        height: Высота скана, px.
        rotate_cw: Угол поворота копии относительно скана, градусы по часовой: 0, 90, 180 или 270.

    Returns:
        Та же рамка ``(x0, y0, x1, y1, вид)`` в пикселях повёрнутой копии.
    """
    x0, y0, x1, y1, kind = box
    if rotate_cw == 90:
        return height - y1, x0, height - y0, x1, kind
    if rotate_cw == 180:
        return width - x1, height - y1, width - x0, height - y0, kind
    if rotate_cw == 270:
        return y0, width - x1, y1, width - x0, kind
    return box


def load_jobs(db_path: Path, sharpened_dir: Path) -> list[PageJob]:
    """Все полосы пака из базы разметки с известными растром и таблицами в координатах заострённой копии.

    Args:
        db_path: Проверенная база разметки пака (``pack1_reviewed.sqlite``), открывается только на чтение.
        sharpened_dir: Корень заострённых копий: ``<год>/<выпуск>/<полоса>.jpg``.

    Returns:
        Задания по полосам в порядке пака; полосы без заострённой копии пропускаются с
        предупреждением. У копии, повёрнутой относительно скана (стороны переставлены, угол —
        ``pages.rotate_cw``), рамки из базы повёрнуты вслед за ней (:func:`rotate_box`).
    """
    from PIL import Image

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    pages = connection.execute(
        "select id, source_rel_path, width, height, coalesce(rotate_cw, 0) from pages order by source_rel_path"
    ).fetchall()
    regions: dict[int, list[tuple[int, int, int, int, str]]] = {}
    for page_id, x1, y1, x2, y2, kind in connection.execute("select page_id, x1, y1, x2, y2, kind from rect_regions"):
        regions.setdefault(page_id, []).append((int(x1), int(y1), int(x2), int(y2), str(kind)))
    connection.close()
    jobs: list[PageJob] = []
    missing = rotated = unknown = 0
    for page_id, rel_path, width, height, rotate_cw in pages:
        name = str(Path(rel_path).with_suffix(""))
        path = sharpened_dir / f"{name}.jpg"
        if not path.is_file():
            missing += 1
            continue
        own = regions.get(page_id, [])
        # Размер копии читается из заголовка JPEG (без декодирования): переставленные стороны —
        # копия повёрнута, и рамки из базы надо повернуть вслед за ней.
        size = Image.open(path).size
        if size != (width, height) and own:
            if size == (height, width) and rotate_cw in (90, 270):
                own = [rotate_box(box, width, height, rotate_cw) for box in own]
                rotated += 1
            else:
                # Поворот не восстановить — известные области не подаются вовсе (лучше без
                # исключений, чем с исключениями не на своём месте).
                own = []
                unknown += 1
        raster = tuple(r for r in own if r[4] in RASTER_KINDS)
        tables = tuple(r for r in own if r[4] == RegionKind.TABLE.value)
        jobs.append(PageJob(name, path, raster, tables))
    if missing:
        logger.warning("Нет заострённой копии у %d полос — пропущены", missing)
    if rotated or unknown:
        logger.info(
            "Копия повёрнута относительно скана: %d полос (рамки повёрнуты), поворот не ясен: %d", rotated, unknown
        )
    return jobs


# Виды растровых областей в базе.
RASTER_KINDS = tuple(kind.value for kind in RegionKind if kind.is_raster)


def _known(item: tuple[int, int, int, int, str]) -> Region:
    """Область из базы → ``Region`` с источником «db» (как ``scan_markup.detection.page._known_region``)."""
    x0, y0, x1, y1, kind = item
    return Region(Box(x0, y0, max(x0, x1), max(y0, y1)), RegionKind(kind), None, "db")


def surya_over(box: Box, cache: SuryaCache, name: str, width: int, height: int) -> list[dict]:
    """Блоки surya полосы, задевающие рамку, с долей рамки, которую каждый накрывает.

    Args:
        box: Рамка области в пикселях скана.
        cache: Кэш surya (только чтение).
        name: Ключ полосы в кэше.
        width: Ширина скана, px — к ней приводятся блоки кэша.
        height: Высота скана, px.

    Returns:
        Список ``{"label", "confidence", "share"}``; ``share`` — доля площади рамки под блоком.
        Пусто, если записи в кэше нет.
    """
    blocks = cache.blocks_of(Variant.SHARPENED, name)
    if blocks is None:
        return []
    blocks = blocks.scaled_to(width, height)
    found = []
    for block in blocks.blocks:
        common = intersection(block.box, box)
        if common is None:
            continue
        found.append(
            {
                "label": block.label,
                "confidence": round(float(block.confidence), 3),
                "share": round(common.area / max(1, box.area), 3),
            }
        )
    return found


def crop_region(
    gray: np.ndarray, box: Box, dpi: int, crop_path: Path, pad_mm: float = CROP_PAD_MM, side_pad_mm: float | None = None
) -> list[int]:
    """Вырезать область с полем, ужать до ``CROP_DPI`` и записать серым PNG.

    Args:
        gray: Серая полоса (заострённая копия) в родном разрешении.
        box: Рамка области в пикселях полосы.
        dpi: Разрешение полосы.
        crop_path: Куда писать PNG.
        pad_mm: Поле сверху и снизу, мм.
        side_pad_mm: Поле слева и справа, мм; ``None`` — как сверху и снизу.

    Returns:
        Рамка самой области внутри вырезки ``[x0, y0, x1, y1]`` в пикселях вырезки (``crop_inner``).
    """
    height, width = gray.shape[:2]
    pad = int(round(pad_mm / 25.4 * dpi))
    side = int(round((pad_mm if side_pad_mm is None else side_pad_mm) / 25.4 * dpi))
    padded = Box(box.x0 - side, box.y0 - pad, box.x1 + side, box.y1 + pad).clipped(width, height)
    scale = CROP_DPI / dpi
    crop = cv2.resize(gray[padded.slice], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(crop_path), crop)
    return [
        int(round((box.x0 - padded.x0) * scale)),
        int(round((box.y0 - padded.y0) * scale)),
        int(round((box.x1 - padded.x0) * scale)),
        int(round((box.y1 - padded.y0) * scale)),
    ]


def recrop_page(
    page: str, rows: list[dict], sharpened_dir: Path, crops_dir: Path, pad_mm: float, side_pad_mm: float
) -> list[dict]:
    """Пересоздать вырезки областей одной полосы с другим полем (детектор не перезапускается).

    Args:
        page: Имя полосы («год/выпуск/полоса»).
        rows: Строки ``regions.jsonl`` этой полосы.
        sharpened_dir: Корень заострённых копий.
        crops_dir: Куда писать вырезки (имена те же).
        pad_mm: Поле сверху и снизу, мм.
        side_pad_mm: Поле слева и справа, мм.

    Returns:
        Те же строки с новым ``crop_inner``.
    """
    image = PageImage.from_file(sharpened_dir / f"{page}.jpg", Variant.SHARPENED, page, default_dpi=600)
    gray = image.gray
    updated = []
    for row in rows:
        inner = crop_region(gray, Box(*row["box"]), image.dpi, crops_dir / row["crop"], pad_mm, side_pad_mm)
        updated.append({**row, "crop_inner": inner, "crop_pad_mm": [pad_mm, side_pad_mm]})
    return updated


def detect_page(job: PageJob, cache_root: Path, crops_dir: Path) -> list[dict]:
    """Области line art одной полосы; вырезка каждой пишется в ``crops_dir`` серым PNG при ``CROP_DPI``.

    Args:
        job: Полоса и её известные области.
        cache_root: Корень кэша surya ``page_layout``.
        crops_dir: Куда класть вырезки: ``<crops_dir>/<id>.png``.

    Returns:
        По словарю на область: ``id`` (имя полосы и номер области), рамка в пикселях скана,
        размер полосы, уверенность и ``info`` детектора, блоки surya над рамкой, путь вырезки
        и её поле в пикселях вырезки.
    """
    image = PageImage.from_file(job.path, Variant.SHARPENED, job.name, default_dpi=600)
    known = {
        Find.RASTER: [_known(item) for item in job.known_raster],
        Find.TABLES: [_known(item) for item in job.known_tables],
    }
    cache = SuryaCache(cache_root, readonly=True)
    from ocr_utils.page_layout.surya.source import SuryaSource

    layout = PageLayout(image, {Find.LINE_ART}, LayoutOptions(), known=known).process(SuryaSource(cache, None))
    rows = []
    gray = image.gray
    for index, region in enumerate(layout.line_arts):
        box = region.box
        region_id = f"{job.name.replace('/', '_')}_{index}"
        crop_path = crops_dir / f"{region_id}.png"
        inner = crop_region(gray, box, image.dpi, crop_path)
        rows.append(
            {
                "id": region_id,
                "page": job.name,
                "box": list(box.as_tuple()),
                "page_size": [image.width, image.height],
                "dpi": image.dpi,
                "confidence": region.confidence,
                "info": region.info,
                "surya": surya_over(box, cache, job.name, image.width, image.height),
                "crop": crop_path.name,
                # Где сама рамка внутри вырезки (с полем), в пикселях вырезки.
                "crop_inner": inner,
                "crop_pad_mm": [CROP_PAD_MM, CROP_PAD_MM],
            }
        )
    return rows


def page_hints(job: PageJob, cache_root: Path) -> dict:
    """Что детектор line art получил на вход на полосе: подсказки и исключения с источниками, и что из них вышло.

    Прогоняет тот же ``PageLayout``, что :func:`detect_page`, перехватывая вызов ``detect_line_art``
    (детектор сам подсказки не сохраняет).

    Args:
        job: Полоса и её известные области.
        cache_root: Корень кэша surya ``page_layout``.

    Returns:
        ``{"page", "hints": [{"source", "box", "agrees"}], "excluded": {источник: число}, "regions": n}``;
        рамки — в пикселях рабочей копии детектора; ``agrees`` — легла ли подсказка на итоговую
        область (пересечение не меньше половины меньшей рамки, как считает сам детектор).
    """
    from ocr_utils.page_layout import analysis
    from ocr_utils.page_layout.geometry import hint_source
    from ocr_utils.page_layout.line_art import detector
    from ocr_utils.page_layout.surya.source import SuryaSource

    captured: dict = {}
    original = analysis.detect_line_art

    def recording(inputs):
        # Перехват единственного вызова детектора на полосе: вход и выход в рабочих пикселях.
        regions = original(inputs)
        captured["inputs"], captured["regions"] = inputs, regions
        return regions

    analysis.detect_line_art = recording
    try:
        image = PageImage.from_file(job.path, Variant.SHARPENED, job.name, default_dpi=600)
        known = {
            Find.RASTER: [_known(item) for item in job.known_raster],
            Find.TABLES: [_known(item) for item in job.known_tables],
        }
        cache = SuryaCache(cache_root, readonly=True)
        layout = PageLayout(image, {Find.LINE_ART}, LayoutOptions(), known=known).process(SuryaSource(cache, None))
    finally:
        analysis.detect_line_art = original
    # Исходные рамки surya Equation (до достройки) и итоговые формулы — в пикселях полосы.
    blocks = cache.blocks_of(Variant.SHARPENED, job.name)
    equations = []
    if blocks is not None:
        native_blocks = blocks.scaled_to(image.width, image.height)
        equations = [
            {
                "box": list(b.box.clipped(image.width, image.height).as_tuple()),
                "confidence": round(float(b.confidence), 4),
            }
            for b in native_blocks.by_label(("Equation",))
        ]
    inputs, regions = captured["inputs"], captured["regions"]
    hints = [
        {
            "source": hint_source(hint),
            "box": list(hint.as_tuple()),
            "likely": [c.value for c in getattr(hint, "likely", ())],
            "agrees": any(detector._agrees(region.box, hint) for region in regions),
        }
        for hint in inputs.hinted_boxes
    ]
    return {
        "page": job.name,
        "hints": hints,
        "excluded": len(inputs.excluded_boxes),
        "excluded_raster": len(job.known_raster),
        "excluded_tables": len(job.known_tables),
        "regions": len(regions),
        "work_dpi": inputs.dpi,
        "page_size": [image.width, image.height],
        "native_dpi": image.dpi,
        "equations_surya": equations,
        "formulas": [
            {"box": list(r.box.as_tuple()), "info": r.info, "confidence": r.confidence} for r in layout.formulas
        ],
    }


def init_worker() -> None:
    """Инициализатор воркера: без hugepage и с одним потоком OpenCV и BLAS (см. ``.claude/rules/gpu_and_pools.md``)."""
    np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
    cv2.setNumThreads(1)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # noqa: BLE001 — без threadpoolctl остаётся только окружение
        pass
