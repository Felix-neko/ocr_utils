"""Точность и полнота источников регионов line art против ручной разметки пака.

Эталон — 221 область ``line_art_schema`` в ``pack1_reviewed.sqlite`` (пиксели скана). Сверка
идёт на PDF БЕЗ коррекции геометрии: там растр страницы = скан + поля выпуска (12 × 6 мм),
и рамка разметки ложится на страницу простым сдвигом (расхождение ~13 px, см.
``line_art_detection/markup.py``). Четыре источника рамок: картинки FineReader (то, что он
сам вырезал как иллюстрацию), детектор таблиц с видом «схема»/«рисунок», связные пятна и
скопления линеек ``line_art_detection`` и блоки Figure/Picture surya из кэша разметки
по сканам. Пятый — объединение всех. Метрика — по областям при IoU ≥ 0.5 и по покрытию
(эталон накрыт предсказаниями хотя бы наполовину).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import numpy as np

from ocr_utils.pdf_utils.intermediate_pdfs import DEFAULT_MARGIN_X_MM, DEFAULT_MARGIN_Y_MM
from ocr_utils.page_layout.image import Variant
from ocr_utils.page_layout.surya.blocks import LayoutBlocks
from ocr_utils.page_layout.surya.cache import SuryaCache, scan_cache_name
from ocr_utils.scan_markup.rotation import rotate_box
from ocr_utils.page_layout.geometry import KIND_TABLE, Box, iou, union
from ocr_utils.page_layout.surya.blocks import FIGURE_LABELS

from ocr_utils.text_layer_fix import mm_to_px
from ocr_utils.text_layer_fix.raster import page_raster, render_gray
from ocr_utils.text_layer_fix.zones import detect_tables

# Источники: figures — картинки FineReader; detector — схема/рисунок детектора таблиц (без surya);
# ink — связные пятна line_art_detection по полному растру; surya — Figure/Picture из кэша по СКАНАМ
# (сдвиг на поля); surya_render — Figure/Picture surya по РЕНДЕРУ no-geo (кэш fr_nogeo);
# page_layout — единый детектор page_layout (детектор таблиц + surya по рендеру + пятна, вне таблиц);
# union — объединение первых четырёх (как до page_layout).
SOURCES = ("figures", "detector", "ink", "surya", "surya_render", "page_layout", "union")
MATCH_IOU = 0.5
COVER_SHARE = 0.5


@dataclass(frozen=True)
class TruthPage:
    """Полоса с эталонными областями: где она в PDF и как её пиксели переводятся в страницу."""

    pdf: str
    page: int
    rel_path: str
    scan_width: int
    scan_height: int
    rotate_cw: int
    boxes: tuple[Box, ...]


def truth_pages(
    markup_db: Path, index: dict, dpi: int = 600, kinds: tuple[str, ...] = ("line_art_schema",)
) -> dict[tuple[str, int], TruthPage]:
    """Области заданных видов из базы по страницам no-geo PDF (эталон line art либо растр-исключения).

    Args:
        markup_db: База разметки (только чтение).
        index: ``(год, выпуск, полоса) → PageRef`` из :func:`pages.page_index_map`.
        dpi: Разрешение сканов (поля переводятся в пиксели по нему).
        kinds: Виды ``rect_regions``; по умолчанию — эталон line art.

    Returns:
        Словарь ``(pdf, page) → TruthPage`` с рамками в пикселях страницы no-geo PDF.
    """
    connection = sqlite3.connect(f"file:{markup_db}?mode=ro", uri=True)
    marks = ",".join("?" for _ in kinds)
    rows = connection.execute(
        f"""
        select y.year, i.name, p.source_file_name, p.source_rel_path, p.width, p.height, coalesce(p.rotate_cw, 0),
               r.x1, r.y1, r.x2, r.y2
        from rect_regions r join pages p on p.id = r.page_id
        join issues i on i.id = p.issue_id join year_packages y on y.id = i.year_package_id
        where r.kind in ({marks})
        """,
        kinds,
    ).fetchall()
    connection.close()
    dx, dy = mm_to_px(DEFAULT_MARGIN_X_MM, dpi), mm_to_px(DEFAULT_MARGIN_Y_MM, dpi)
    grouped: dict[tuple[str, int], dict] = {}
    for year, issue, file_name, rel_path, width, height, rotate, x1, y1, x2, y2 in rows:
        ref = index.get((str(year), str(issue), Path(file_name).stem))
        if ref is None:
            continue
        box = (int(x1), int(y1), int(x2), int(y2))
        if rotate:
            box = rotate_box(box, int(width), int(height), int(rotate))
            width, height = (height, width) if rotate % 180 else (width, height)
        entry = grouped.setdefault(
            ref.key, {"rel": rel_path, "w": int(width), "h": int(height), "rot": int(rotate), "boxes": []}
        )
        entry["boxes"].append(Box(box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy))
    return {
        key: TruthPage(key[0], key[1], e["rel"], e["w"], e["h"], e["rot"], tuple(e["boxes"]))
        for key, e in grouped.items()
    }


def _merge_overlapping(boxes: list[Box]) -> list[Box]:
    """Объединить пересекающиеся рамки в одну (для источника «union»)."""
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for a in range(len(merged)):
            for b in range(a + 1, len(merged)):
                first, second = merged[a], merged[b]
                if first.x0 < second.x1 and second.x0 < first.x1 and first.y0 < second.y1 and second.y0 < first.y1:
                    merged[a] = union([first, second])
                    del merged[b]
                    changed = True
                    break
            if changed:
                break
    return merged


def source_boxes(
    page: fitz.Page,
    gray: np.ndarray,
    dpi: int,
    cached: "LayoutBlocks | None",
    scan_size: "tuple[int, int] | None",
    structure=None,
) -> dict[str, list[Box]]:
    """Рамки line art страницы от каждого источника (пиксели растра страницы).

    Args:
        page: Страница no-geo PDF.
        gray: Её растр.
        dpi: Разрешение растра.
        cached: Разметка surya полосы из кэша (в пикселях скана) или None.
        scan_size: Размер скана ``(width, height)`` для масштабирования разметки surya.
        structure: Готовый ``page_layout.PageLayout`` по рендеру этой страницы или None.

    Returns:
        Словарь ``источник → рамки``, включая объединение ``union``.
    """
    from ocr_utils.page_layout.line_art.features import analyse_gray, params_for_dpi

    raster = page_raster(page)
    px = raster.to_px()
    height, width = gray.shape[:2]
    result: dict[str, list[Box]] = {}
    result["figures"] = [Box(*[int(round(v)) for v in (f * px)]).clipped(width, height) for f in raster.figures]
    result["detector"] = [t.box for t in detect_tables(gray, dpi) if t.kind != KIND_TABLE]
    findings = analyse_gray(gray, params_for_dpi(dpi))
    result["ink"] = [Box(*b).clipped(width, height) for b in findings.boxes]
    surya: list[Box] = []
    if cached is not None and scan_size is not None:
        dx, dy = mm_to_px(DEFAULT_MARGIN_X_MM, dpi), mm_to_px(DEFAULT_MARGIN_Y_MM, dpi)
        layout = cached.scaled_to(scan_size[0], scan_size[1])
        surya = [b.box.shifted(dx, dy).clipped(width, height) for b in layout.blocks if b.label in FIGURE_LABELS]
    result["surya"] = surya
    result["surya_render"] = []
    result["page_layout"] = []
    if structure is not None:
        if structure.raw_surya_content is not None:
            result["surya_render"] = [
                b.box.clipped(width, height) for b in structure.raw_surya_content.by_label(FIGURE_LABELS)
            ]
        result["page_layout"] = [r.box.clipped(width, height) for r in structure.line_arts]
    result["union"] = _merge_overlapping([b for name in ("figures", "detector", "ink", "surya") for b in result[name]])
    return result


@dataclass
class SourceScore:
    """Счётчики одного источника по всем страницам."""

    matched: int = 0  # предсказаний с IoU ≥ 0.5 к какому-то эталону
    predicted: int = 0
    truth: int = 0
    truth_hit: int = 0  # эталонов с IoU ≥ 0.5
    truth_covered: int = 0  # эталонов, накрытых предсказаниями хотя бы наполовину
    false_pages: int = 0  # предсказаний на страницах без эталона
    ious: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.matched / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.truth_hit / self.truth if self.truth else 0.0

    @property
    def coverage(self) -> float:
        return self.truth_covered / self.truth if self.truth else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def to_json(self) -> dict:
        return {
            "predicted": self.predicted,
            "matched": self.matched,
            "truth": self.truth,
            "truth_hit": self.truth_hit,
            "truth_covered": self.truth_covered,
            "false_on_pages_without_truth": self.false_pages,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "coverage": round(self.coverage, 3),
            "f1": round(self.f1, 3),
            "iou_median": round(float(np.median(self.ious)), 3) if self.ious else 0.0,
        }


def _covered_share(truth: Box, predictions: list[Box]) -> float:
    """Какая доля площади эталона накрыта объединением предсказаний (по маске)."""
    if not predictions or truth.area <= 0:
        return 0.0
    scale = 0.05
    w, h = max(1, int(truth.width * scale)), max(1, int(truth.height * scale))
    mask = np.zeros((h, w), bool)
    for box in predictions:
        x0 = max(0, int((box.x0 - truth.x0) * scale))
        y0 = max(0, int((box.y0 - truth.y0) * scale))
        x1 = min(w, int(np.ceil((box.x1 - truth.x0) * scale)))
        y1 = min(h, int(np.ceil((box.y1 - truth.y0) * scale)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return float(mask.mean())


def score_page(truth: list[Box], predictions: dict[str, list[Box]], scores: dict[str, SourceScore]) -> list[dict]:
    """Зачесть страницу в счётчики источников; вернуть строки для CSV (по предсказанию и по эталону).

    Args:
        truth: Эталонные рамки страницы (пусто — страница без line art).
        predictions: Рамки источников.
        scores: Накопительные счётчики по источникам (изменяются на месте).

    Returns:
        Строки: ``kind`` = ``prediction``/``truth``, источник, рамка, лучший IoU, накрытие.
    """
    rows: list[dict] = []
    for name in SOURCES:
        score = scores[name]
        boxes = predictions.get(name, [])
        score.predicted += len(boxes)
        if not truth:
            score.false_pages += len(boxes)
        for box in boxes:
            best = max((iou(box, t) for t in truth), default=0.0)
            if best >= MATCH_IOU:
                score.matched += 1
                score.ious.append(best)
            rows.append(
                {"kind": "prediction", "source": name, "box": box.as_tuple(), "iou": round(best, 3), "covered": ""}
            )
        for t in truth:
            best = max((iou(box, t) for box in boxes), default=0.0)
            covered = _covered_share(t, boxes)
            score.truth += 1
            score.truth_hit += int(best >= MATCH_IOU)
            score.truth_covered += int(covered >= COVER_SHARE)
            rows.append(
                {
                    "kind": "truth",
                    "source": name,
                    "box": t.as_tuple(),
                    "iou": round(best, 3),
                    "covered": round(covered, 3),
                }
            )
    return rows


def evaluate_page(
    doc: fitz.Document,
    index: int,
    truth: "TruthPage | None",
    layout_dir: "Path | None",
    rel_path: "str | None",
    raster_truth: "TruthPage | None" = None,
) -> tuple[dict[str, list[Box]], list[Box]]:
    """Предсказания всех источников и эталон одной страницы no-geo PDF.

    ``layout_dir`` — корень кэша surya page_layout: вариант ``scan`` даёт источник ``surya``
    (по сканам, сдвиг на поля), вариант ``fr_nogeo`` — ``surya_render`` и ``page_layout`` (по
    рендеру; кэш набивается заранее, промах — без surya). ``raster_truth`` — растровые области полосы из
    базы: ``page_layout`` получает их как известные исключения, ровно как на ``detect``.
    """
    from ocr_utils.page_layout.analysis import Find, LayoutOptions, PageLayout
    from ocr_utils.page_layout.image import PageImage, SourceStat
    from ocr_utils.page_layout.regions import Region, RegionKind
    from ocr_utils.page_layout.surya.source import OnMiss, SuryaSourceConfig

    page = doc[index]
    raster = page_raster(page)
    gray = render_gray(page, raster)
    layout = None
    if layout_dir:
        path = Path(doc.name)
        image = PageImage.from_array(
            gray, int(round(raster.dpi)), Variant.FR_NOGEO, f"{path.stem}/p{index:04d}", SourceStat.of(path, index)
        )
        surya = SuryaSourceConfig(Path(layout_dir), OnMiss.SKIP).open()
        known = (
            {Find.RASTER: [Region(b, RegionKind.GRAYSCALE, None, "db") for b in raster_truth.boxes]}
            if raster_truth
            else None
        )
        layout = PageLayout(image, {Find.TABLES, Find.LINE_ART}, LayoutOptions(), known=known).process(surya)
    cached = (
        SuryaCache(layout_dir, readonly=True).blocks_of(Variant.SCAN, scan_cache_name(rel_path))
        if (layout_dir and rel_path)
        else None
    )
    scan_size = (truth.scan_width, truth.scan_height) if truth else None
    predictions = source_boxes(page, gray, int(round(raster.dpi)), cached, scan_size, layout)
    return predictions, list(truth.boxes) if truth else []
