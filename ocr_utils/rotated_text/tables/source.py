"""Откуда берутся таблицы: строки ``rect_regions`` базы разметки и вырезки из полос.

ВХОД — ЗАОСТРЁННЫЕ КОПИИ, а не оригиналы с /mnt/dump3. Во-первых, именно они уходят в
промежуточные PDF, то есть именно на них FineReader и спотыкается. Во-вторых, фон у них
уже выровнен: закраска ячейки уровнем бумаги не оставляет заплат. В-третьих, 40 МБ TIFF
со шпиндельного NTFS-3G читались бы дольше, чем считается вся таблица.

КООРДИНАТЫ. Рамка в базе лежит в пикселях ИСХОДНОГО скана, полоса может требовать поворота
(``pages.rotate_cw``). Вырезка режется из неповёрнутого кадра и поворачивается сама — это
дешевле, чем поворачивать полосу 21 Мпк ради одной таблицы.

ВЫРАВНИВАНИЕ. Наклон линеек в паке доходит до 1.7°, и на высоте ячейки в 500 px край
линейки уходит вбок на 15 px — этого достаточно, чтобы закраска задела линейку, а сетка
ошиблась колонкой. Поэтому вырезка выравнивается по своим же линейкам. Угол меряется, а не
берётся из базы, и результат ПРОВЕРЯЕТСЯ: после поворота остаточный наклон должен упасть;
если вырос — знак угла был перепутан, и поворот делается в другую сторону. Так ошибка
знака, которую иначе видно только глазами и только на рендере, не может дожить до него.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ocr_utils.db.models import KIND_TABLE
from ocr_utils.db.repo import iter_pages, require_pack
from ocr_utils.db.session import open_db
from ocr_utils.scan_markup.rotation import rotate_box
from ocr_utils.scan_markup.rotation import rotate_cw as rotate_image
from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.tables.ruling import find_lines, mm_to_px, skew_of

logger = logging.getLogger(__name__)

# Поле вокруг рамки детектора: рамка подвинута до чистого просвета, но линейки таблицы
# бывают на самом краю, и без поля морфология их не видит.
CROP_PAD_MM = 3.0

# Наклон меньше этого не правится: поворот на доли градуса пересемплирует штрих впустую.
MIN_DESKEW_DEG = 0.3

# Разрешение, на котором меряется наклон: линейки на нём находятся все, а стоит вчетверо
# дешевле исходных 600 dpi.
SKEW_DPI = 150


@dataclass(frozen=True)
class TableRef:
    """Одна таблица из базы: где лежит полоса и где на ней рамка."""

    table_id: str
    year: str
    issue: str
    page_rel_path: str
    image_path: str
    box: tuple[int, int, int, int]
    page_rotate_cw: int
    dpi: int
    skew_deg: float
    region_id: int
    page_id: int


@dataclass
class TableCrop:
    """Вырезка таблицы: повёрнута как полоса, выровнена по линейкам, в исходном dpi."""

    gray: np.ndarray
    dpi: int
    box_in_page: tuple[int, int, int, int]
    page_rotate_cw: int
    deskew_deg: float


def select_tables(
    db_path: Path,
    pack_name: str,
    images_root: "Path | None" = None,
    only_year: "str | None" = None,
    only_issue: "str | None" = None,
    limit: "int | None" = None,
    only_table: "str | None" = None,
) -> list[TableRef]:
    """Все таблицы пака из базы, в порядке год → выпуск → полоса → id находки."""
    factory = open_db(Path(db_path), create=False)
    refs: list[TableRef] = []
    with factory() as session:
        pack = require_pack(session, pack_name)
        root = Path(images_root) if images_root else Path(pack.sharpened_text_pics_root or "")
        if not root.is_dir():
            raise FileNotFoundError(f"нет корня с картинками: {root}")
        for year, issue, page in iter_pages(pack, only_year, only_issue):
            tables = sorted((r for r in page.rect_regions if r.kind == KIND_TABLE), key=lambda r: r.id)
            if not tables:
                continue
            rel = page.sharpened_text_pic_rel_path or page.source_rel_path
            path = root / rel
            if not path.is_file():
                logger.warning("нет картинки %s — таблицы полосы пропущены", path)
                continue
            stem = Path(page.source_rel_path).stem
            for index, region in enumerate(tables):
                info = json.loads(region.detector_info or "{}")
                table_id = f"{year.name}_{issue.name}_{stem}_t{index:02d}"
                if only_table and only_table not in table_id:
                    continue
                refs.append(
                    TableRef(
                        table_id=table_id,
                        year=year.name,
                        issue=issue.name,
                        page_rel_path=page.source_rel_path,
                        image_path=str(path),
                        box=(region.x1, region.y1, region.x2, region.y2),
                        page_rotate_cw=int(page.rotate_cw or 0),
                        dpi=int(page.dpi or 600),
                        skew_deg=float(info.get("skew_deg", 0.0) or 0.0),
                        region_id=region.id,
                        page_id=page.id,
                    )
                )
    if limit is not None:
        refs = refs[:limit]
    return refs


def load_gray(path: Path) -> np.ndarray:
    """Полоса целиком в градациях серого. ``draft`` заставляет libjpeg сразу декодировать
    в серое, минуя RGB — это заметно быстрее на 21-мегапиксельном кадре."""
    image = Image.open(path)
    image.draft("L", image.size)
    return np.asarray(image.convert("L"))


def load_table_crop(ref: TableRef, pad_mm: float = CROP_PAD_MM, deskew: bool = True) -> TableCrop:
    """Вырезка таблицы из полосы: с полем, повёрнутая как полоса, выровненная по линейкам."""
    page = load_gray(Path(ref.image_path))
    height, width = page.shape[:2]
    box = Box(*ref.box).padded(mm_to_px(pad_mm, ref.dpi)).clipped(width, height)
    crop = np.ascontiguousarray(page[box.slice])
    box_in_page = box.as_tuple()
    if ref.page_rotate_cw:
        crop = rotate_image(crop, ref.page_rotate_cw)
        box_in_page = rotate_box(box_in_page, width, height, ref.page_rotate_cw)
    angle = 0.0
    if deskew:
        crop, angle = deskew_by_rules(crop, ref.dpi)
    return TableCrop(
        gray=crop, dpi=ref.dpi, box_in_page=box_in_page, page_rotate_cw=ref.page_rotate_cw, deskew_deg=angle
    )


def measure_skew(gray: np.ndarray, dpi: int) -> float:
    """Медианный наклон горизонтальных линеек, градусы по часовой (валюта ``rotation``)."""
    factor = SKEW_DPI / dpi
    small = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) if factor < 1 else gray
    angle, _ = skew_of(find_lines(small, SKEW_DPI))
    return angle


def rotate_small(gray: np.ndarray, degrees_cw: float) -> np.ndarray:
    """Поворот на малый угол по часовой, поля — цветом бумаги."""
    height, width = gray.shape[:2]
    paper = int(np.percentile(gray, 90))
    # У cv2 положительный угол — ПРОТИВ часовой, поэтому знак обратный.
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), -degrees_cw, 1.0)
    return cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=paper)


def deskew_by_rules(gray: np.ndarray, dpi: int) -> tuple[np.ndarray, float]:
    """Выравнивание по линейкам с проверкой знака. Возвращает картинку и применённый угол."""
    angle = measure_skew(gray, dpi)
    if abs(angle) < MIN_DESKEW_DEG:
        return gray, 0.0
    candidate = rotate_small(gray, angle)
    residual = measure_skew(candidate, dpi)
    if abs(residual) <= abs(angle) * 0.5:
        return candidate, angle
    # Остаточный наклон не упал — знак перепутан или измерение шумное: пробуем другую сторону.
    other = rotate_small(gray, -angle)
    if abs(measure_skew(other, dpi)) < abs(residual):
        return other, -angle
    return (candidate, angle) if abs(residual) < abs(angle) else (gray, 0.0)
