"""Правка страницы по JSON кэша: удаление, усечение и вставка по вердиктам; сверка копии.

Слой перечитывается заново (по MCID слова находятся в свежем разборе), поэтому кэш хранит
только решения, а не байты потока. Вставляются зоны с принятым чтением, кроме тех, где
FineReader уже написал слово повёрнутым (иначе дубль в поиске).

Две формы использования: :func:`fix_pdf` — исправленная копия целого PDF (стенд
``research.text_layer_fix fix``); :func:`plan_page_edits` + :func:`apply_page_edits` +
:func:`verify_saved` — постранично, для сборщика финальных PDF, который копирует страницу в
новый документ и правит её там.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz
import pikepdf

from ocr_utils.text_layer_fix.cache import load_page
from ocr_utils.text_layer_fix.classify import Verdict
from ocr_utils.text_layer_fix.raster import page_raster
from ocr_utils.text_layer_fix.rewrite import Insert, InsertFont, VerifyReport, apply_edits, verify_page
from ocr_utils.text_layer_fix.text_layer import Word, load_layer, page_content


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


@dataclass
class PageEdits:
    """План правок одной страницы по её JSON: что удалить, усечь, оставить и вписать."""

    blanks: list[Word] = field(default_factory=list)  # слова DELETE
    trims: dict[int, tuple[Word, tuple[int, ...]]] = field(default_factory=dict)  # SANITIZE: слово и глифы
    kept: list[Word] = field(default_factory=list)  # всё остальное — должно остаться на месте
    inserts: list[Insert] = field(default_factory=list)  # своё чтение зон
    skipped_duplicates: int = 0  # зон не вписано из-за повёрнутого слова FineReader
    raw: bytes = b""  # поток содержимого страницы-источника до правок
    main_xref: int | None = None  # основной образ страницы в источнике (для md5-сверки)

    @property
    def empty(self) -> bool:
        """Правок нет — страницу можно не трогать."""
        return not (self.blanks or self.trims or self.inserts)


def load_cache(cache_dir: Path, pdf_name: str) -> dict[int, dict]:
    """JSON кэша всех страниц одного PDF (только текущей версии, без ошибок).

    Args:
        cache_dir: Каталог ``cache/`` прогона.
        pdf_name: Имя PDF.

    Returns:
        Словарь ``номер страницы → JSON``.
    """
    result: dict[int, dict] = {}
    for path in sorted((cache_dir / pdf_name).glob("p*.json")):
        payload = load_page(path)
        if payload is not None:
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


def has_edits(payload: dict) -> bool:
    """Есть ли в JSON страницы хоть одна правка: слово на удаление/усечение или принятое чтение.

    Дешёвая проверка до разбора слоя (0,3 с на страницу): на большинстве страниц править нечего.

    Args:
        payload: JSON страницы.

    Returns:
        ``True``, если план правок будет непустым.
    """
    if any(w.get("verdict") in (Verdict.DELETE.value, Verdict.SANITIZE.value) for w in payload.get("words", ())):
        return True
    return any(r.get("accepted") and r.get("text") for r in payload.get("readings", {}).values())


def plan_page_edits(page: fitz.Page, pdf: pikepdf.Pdf | None, payload: dict) -> PageEdits:
    """План правок страницы: слова по MCID из свежего разбора слоя и вставки по чтениям.

    Args:
        page: Страница ИСХОДНОГО (нетронутого) документа.
        pdf: Тот же документ, открытый pikepdf (метрики шрифтов); ``None`` — ширины по умолчанию.
        payload: JSON страницы из кэша.

    Returns:
        :class:`PageEdits`; слова, которых в свежем разборе нет, пропускаются.
    """
    layer = load_layer(page, pdf)
    by_mcid = {w.mcid: w for w in layer.words if w.mcid is not None}
    edits = PageEdits(raw=page_content(page))
    for record in payload["words"]:
        word = by_mcid.get(record["mcid"])
        if word is None:
            continue
        verdict = Verdict(record["verdict"])
        if verdict == Verdict.DELETE:
            edits.blanks.append(word)
        elif verdict == Verdict.SANITIZE and record["keep_glyphs"]:
            edits.trims[id(word)] = (word, tuple(record["keep_glyphs"]))
        else:
            edits.kept.append(word)
    raster = page_raster(page)
    edits.inserts, edits.skipped_duplicates = inserts_for(payload, raster.to_pt())
    edits.main_xref = raster.main_xref or None
    return edits


def apply_page_edits(doc: fitz.Document, index: int, edits: PageEdits, font: InsertFont) -> PageFix:
    """Применить план к странице документа, который будет сохранён.

    Args:
        doc: Документ, куда пишем (тот же файл, что источник, или новый со скопированной страницей).
        index: Номер страницы в ``doc`` с нуля.
        edits: План (:func:`plan_page_edits`) — его ``raw`` должен совпадать с потоком страницы.
        font: Шрифт вставок.

    Returns:
        :class:`PageFix` со счётчиками (сверка — отдельно, :func:`verify_saved`).
    """
    fix = PageFix(index, skipped_duplicates=edits.skipped_duplicates)
    if edits.empty:
        return fix
    stats = apply_edits(doc, index, edits.raw, edits.blanks, edits.trims, edits.inserts, font)
    fix.blanked, fix.trimmed, fix.inserted = stats.blanked, stats.trimmed, stats.inserted
    return fix


def verify_saved(
    source_page: fitz.Page, saved_page: fitz.Page, edits: PageEdits, saved_main_xref: int | None
) -> VerifyReport:
    """Сверить сохранённую страницу с планом: оставленное на месте, удалённого нет, вставки ищутся.

    Args:
        source_page: Страница исходного документа.
        saved_page: Та же страница в сохранённом файле (то, что увидит просмотрщик).
        edits: План правок.
        saved_main_xref: xref основного образа в сохранённом файле для md5-сверки; ``None`` —
            образ не сверять (например, он снят намеренно).

    Returns:
        Отчёт сверки.
    """
    image_xref = edits.main_xref if saved_main_xref is not None else None
    return verify_page(source_page, saved_page, edits.kept, edits.blanks, edits.inserts, image_xref, saved_main_xref)


def fix_pdf(src: Path, cache: dict[int, dict], out_path: Path, font: InsertFont | None = None) -> list[PageFix]:
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
    planned: dict[int, PageEdits] = {}
    for index in sorted(cache):
        try:
            edits = plan_page_edits(original[index], pdf, cache[index])
            fix = apply_page_edits(doc, index, edits, font)
            planned[index] = edits
        except Exception as error:  # noqa: BLE001
            fix = PageFix(index, error=f"{type(error).__name__}: {error}")
        fixes.append(fix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path), garbage=0, deflate=True)
    doc.close()
    # Сверка — по сохранённому файлу: так проверяется то, что увидит просмотрщик, а не память MuPDF.
    saved = fitz.open(str(out_path))
    by_page = {fix.page: fix for fix in fixes}
    for index, edits in planned.items():
        try:
            by_page[index].verify = verify_saved(original[index], saved[index], edits, edits.main_xref)
        except Exception as error:  # noqa: BLE001
            by_page[index].error = f"сверка: {type(error).__name__}: {error}"
    return fixes
