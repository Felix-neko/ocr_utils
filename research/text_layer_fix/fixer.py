"""Исправленные копии PDF из кэша прогона: удаление, усечение и вставка по вердиктам.

Слой перечитывается заново (по MCID слова находятся в свежем разборе), поэтому кэш
хранит только решения, а не байты потока. Вставляются зоны с принятым чтением, кроме
тех, где FineReader уже написал слово повёрнутым (иначе дубль в поиске).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import pikepdf

from research.text_layer_fix import VERSION
from research.text_layer_fix.classify import Verdict
from research.text_layer_fix.raster import page_raster
from research.text_layer_fix.rewrite import Insert, InsertFont, VerifyReport, apply_edits, verify_page
from research.text_layer_fix.text_layer import load_layer, page_content


@dataclass
class PageFix:
    """Что сделано с одной страницей и как прошла сверка."""

    page: int
    blanked: int = 0
    trimmed: int = 0
    inserted: int = 0
    skipped_duplicates: int = 0
    verify: VerifyReport = field(default_factory=VerifyReport)
    error: str = ""


def load_cache(cache_dir: Path, pdf_name: str) -> dict[int, dict]:
    """JSON кэша всех страниц одного PDF (только текущей версии).

    Args:
        cache_dir: Каталог ``cache/`` прогона.
        pdf_name: Имя PDF.

    Returns:
        Словарь ``номер страницы → JSON``.
    """
    result: dict[int, dict] = {}
    for path in sorted((cache_dir / pdf_name).glob("p*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") == VERSION and not payload.get("error"):
            result[int(payload["page"])] = payload
    return result


def inserts_for(payload: dict, to_pt: fitz.Matrix) -> tuple[list[Insert], int]:
    """Вставки страницы по принятым чтениям зон.

    Args:
        payload: JSON страницы.
        to_pt: Матрица «пиксели растра → pt».

    Returns:
        Список вставок и число зон, пропущенных из-за уже написанного FineReader повёрнутого слова.
    """
    rotated_zones = {
        w["zone_index"]
        for w in payload["words"]
        if w["verdict"] == Verdict.KEEP_ROTATED.value and w.get("zone_index") is not None
    }
    inserts: list[Insert] = []
    skipped = 0
    for index, zone in enumerate(payload["zones"]):
        reading = payload["readings"].get(str(index))
        if not reading or not reading.get("accepted") or not reading.get("text"):
            continue
        if index in rotated_zones:
            skipped += 1
            continue
        rect = fitz.Rect(*zone["box"]) * to_pt
        inserts.append(Insert(reading["text"], rect, int(reading.get("rotate_cw") or 0)))
    return inserts, skipped


def fix_pdf(src: Path, cache: dict[int, dict], out_path: Path, font: "InsertFont | None" = None) -> list[PageFix]:
    """Исправленная копия PDF по кэшу страниц.

    Args:
        src: Исходный PDF (не изменяется).
        cache: JSON страниц (:func:`load_cache`).
        out_path: Куда сохранить копию.
        font: Шрифт вставок.

    Returns:
        Отчёт по каждой странице из кэша.
    """
    font = font or InsertFont()
    original = fitz.open(str(src))
    pdf = pikepdf.open(str(src))
    doc = fitz.open(str(src))
    fixes: list[PageFix] = []
    # Что сверять после сохранения: (страница, оставленные, удалённые, вставки, xref картинки).
    checks: list[tuple[int, list, list, list, "int | None"]] = []
    for index in sorted(cache):
        payload = cache[index]
        fix = PageFix(index)
        try:
            page = original[index]
            layer = load_layer(page, pdf)
            by_mcid = {w.mcid: w for w in layer.words if w.mcid is not None}
            blanks, trims, kept = [], {}, []
            for record in payload["words"]:
                word = by_mcid.get(record["mcid"])
                if word is None:
                    continue
                verdict = Verdict(record["verdict"])
                if verdict == Verdict.DELETE:
                    blanks.append(word)
                elif verdict == Verdict.SANITIZE and record["keep_glyphs"]:
                    trims[id(word)] = (word, tuple(record["keep_glyphs"]))
                else:
                    kept.append(word)
            raster = page_raster(page)
            inserts, fix.skipped_duplicates = inserts_for(payload, raster.to_pt())
            stats = apply_edits(doc, index, page_content(page), blanks, trims, inserts, font)
            fix.blanked, fix.trimmed, fix.inserted = stats.blanked, stats.trimmed, stats.inserted
            checks.append((index, kept, blanks, inserts, raster.main_xref or None))
        except Exception as error:  # noqa: BLE001
            fix.error = f"{type(error).__name__}: {error}"
        fixes.append(fix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path), garbage=0, deflate=True)
    doc.close()
    # Сверка — по сохранённому файлу: так проверяется то, что увидит просмотрщик, а не память MuPDF.
    saved = fitz.open(str(out_path))
    by_page = {fix.page: fix for fix in fixes}
    for index, kept, blanks, inserts, xref in checks:
        try:
            by_page[index].verify = verify_page(original[index], saved[index], kept, blanks, inserts, xref)
        except Exception as error:  # noqa: BLE001
            by_page[index].error = f"сверка: {type(error).__name__}: {error}"
    return fixes
