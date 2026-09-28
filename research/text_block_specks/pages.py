"""Полосы пака «только текст» и их страницы в бинаризованных PDF FineReader (сопоставление через базу разметки)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ocr_utils.final_pdfs.plan import load_plans
from ocr_utils.page_layout.pack_analysis.stages import page_key

# Итог разбора пака v3 (по заострённым сканам): по нему известно, какие полосы — «только текст».
V3_DIR = Path("/mnt/system/raw/mts/pack1_page_analysis_v3")
# Папка оверлеев v3 с полосами без объектов (растр, таблицы, line art и т. п.).
ONLY_TEXT = "только_текст"
# Бинаризованные PDF FineReader: без коррекции геометрии (основной вариант стенда) и с ней.
NOGEO_DIR = Path("/mnt/system/raw/mts/pack1_pdf/full_pdfs_binary_no_bg_brightening_no_geometry_correction")
GEO_DIR = Path("/mnt/system/raw/mts/pack1_pdf/full_pdfs_binary_no_bg_brightening")
# База разметки после ревью в CVAT и имя пака: из неё берётся порядок полос выпуска = номер страницы PDF.
DB_REVIEWED = Path.home() / "Projects" / "mts_markup" / "pack1_reviewed.sqlite"
PACK_NAME = "пак-1"


@dataclass(frozen=True)
class PdfPage:
    """Полоса пака как страница бинаризованного PDF.

    Attributes:
        key: Ключ полосы ``ГГГГ_ВВ_IMG_xxxx`` (как ``stages.page_key``).
        pdf: Имя PDF без каталога (``full_1966_01.pdf``).
        page: Номер страницы PDF с единицы.
    """

    key: str
    pdf: str
    page: int


def only_text_keys(v3_dir: Path = V3_DIR) -> set[str]:
    """Ключи полос, у которых в разборе v3 нет ни одного объекта и поворота.

    Args:
        v3_dir: Корень разбора v3 (``pages/*.json``).

    Returns:
        Множество ключей ``ГГГГ_ВВ_IMG_xxxx``. Полосы с поворотом (``rotate_cw`` ≠ 0) выброшены: в PDF они
        лежат повёрнутыми, и координаты с v3 не сходятся.
    """
    keys = set()
    for path in sorted((v3_dir / "pages").glob("*.json")):
        record = json.loads(path.read_text())
        # Папка оверлея «только текст» — у полосы нет объектов; поворот берётся из вердикта ориентации.
        if record.get("folder") != ONLY_TEXT:
            continue
        if (record.get("orientation") or {}).get("rotate_cw", 0) not in (0, None):
            continue
        keys.add(path.stem)
    return keys


def pdf_pages(keys: set[str] | None = None, db: Path = DB_REVIEWED, pack: str = PACK_NAME) -> list[PdfPage]:
    """Страницы PDF для полос пака, по порядку выпусков.

    Args:
        keys: Какие полосы оставить (``None`` — все).
        db: База разметки: порядок полос выпуска (``full_pdf_page_idx``).
        pack: Имя пака в базе.

    Returns:
        Список :class:`PdfPage`, отсортированный по PDF и номеру страницы.
    """
    out = []
    for plan in load_plans(db, pack):
        for page in plan.pages:
            # Имя полосы — путь заострённой копии без расширения («1966/01/IMG_0004_2R.jpg»).
            key = page_key(str(Path(page.sharpened_rel_path).with_suffix("")))
            if keys is not None and key not in keys:
                continue
            out.append(PdfPage(key=key, pdf=plan.full_pdf_name, page=page.full_pdf_page_idx + 1))
    return out


__all__ = ["GEO_DIR", "NOGEO_DIR", "PdfPage", "V3_DIR", "only_text_keys", "pdf_pages"]
