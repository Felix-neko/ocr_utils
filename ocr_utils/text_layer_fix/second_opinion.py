"""Второе мнение surya по зонам, где tesseract ненадёжен: GPU только в родительском процессе.

Берётся из ``ocr_utils.rotated_text.tables.second_opinion``: чтение блоком без детектора
строк, фильтр выдумок (чужая письменность, зацикливание, длинная латиница) и приём ответа
только при согласии с правдоподобным tesseract. Здесь — обход кэша прогона: вырезки зон
пересчитываются из PDF в родителе, ответы записываются обратно в JSON страниц с сохранением
исходного чтения tesseract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np

from ocr_utils.rotated_text.tables import second_opinion as surya
from ocr_utils.rotated_text.tables.ocr import CellText
from ocr_utils.rotated_text.tables.structure import strip_stubs
from ocr_utils.scan_markup.rotation import rotate_cw as rotate_image
from ocr_utils.scan_markup.table_detection.geometry import Box

from ocr_utils.text_layer_fix.cache import load_page, save_page
from ocr_utils.text_layer_fix.ocr import TABLE_KINDS, acceptable
from ocr_utils.text_layer_fix.raster import page_raster, render_gray
from ocr_utils.text_layer_fix.zones import rotate_crop


@dataclass
class SuryaStats:
    """Итог второго мнения по прогону."""

    asked: int = 0
    accepted: int = 0
    newly_accepted: int = 0  # зоны, которые стали пригодными для вставки только благодаря surya
    pages: int = 0


def _needs_opinion(reading: dict) -> bool:
    """Зона идёт на второе мнение, если чтение не принято или принято неуверенно."""
    if reading.get("rotate_cw") == 0:
        return False
    return not reading.get("accepted") or float(reading.get("confidence", 0.0)) < surya.RELIABLE_CONFIDENCE


def _crop(gray: np.ndarray, zone: dict, rotate: int, dpi: int) -> np.ndarray:
    box = Box(*zone["box"])
    if zone["kind"] in [str(k) for k in TABLE_KINDS]:
        crop = strip_stubs(rotate_crop(gray, box, 0, pad_mm=0.0, dpi=dpi), dpi)
        return rotate_image(crop, rotate) if rotate else crop
    return rotate_crop(gray, box, rotate, dpi=dpi)


def revise(pages: list[tuple[Path, Path]], batch: int = surya.DEFAULT_BATCH) -> SuryaStats:
    """Дочитать surya зоны с ненадёжным чтением и обновить JSON кэша.

    Args:
        pages: Пары «PDF-источник страницы → JSON её разбора». Источник у каждой страницы
            свой (сборщик финальных PDF берёт страницу то с коррекцией геометрии, то без).
        batch: Размер партии surya.

    Returns:
        Статистика.
    """
    stats = SuryaStats()
    requests: list[tuple[Path, str, np.ndarray, dict]] = []
    by_pdf: dict[Path, list[Path]] = {}
    for pdf_path, cache in pages:
        by_pdf.setdefault(Path(pdf_path), []).append(Path(cache))
    for pdf_path, caches in sorted(by_pdf.items()):
        if not pdf_path.is_file():
            continue
        with fitz.open(str(pdf_path)) as doc:
            for cache in sorted(caches):
                payload = load_page(cache)
                if payload is None:
                    continue
                wanted = [(i, r) for i, r in payload["readings"].items() if _needs_opinion(r)]
                if not wanted:
                    continue
                index = int(payload["page"])
                raster = page_raster(doc[index])
                gray = render_gray(doc[index], raster)
                dpi = int(round(raster.dpi))
                for key, reading in wanted:
                    zone = payload["zones"][int(key)]
                    rotate = reading.get("rotate_cw") or zone.get("rotate_cw") or 90
                    requests.append((cache, key, _crop(gray, zone, int(rotate), dpi), reading))
                stats.pages += 1
    if not requests:
        return stats
    stats.asked = len(requests)
    answers = surya.read_cells([r[2] for r in requests], batch)
    updates: dict[Path, dict[str, tuple[dict, CellText]]] = {}
    for (cache, key, _, reading), answer in zip(requests, answers):
        updates.setdefault(cache, {})[key] = (reading, answer)
    for cache, items in updates.items():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        for key, (reading, answer) in items.items():
            ours = CellText(
                lines=[reading.get("text", "")], confidence=float(reading.get("confidence", 0.0)), engine="tesseract"
            )
            accepted, why = surya.accept(
                ours, answer, ours_reliable=False, ours_plausible=bool(reading.get("accepted"))
            )
            entry = payload["readings"][key]
            entry.setdefault("text_tesseract", reading.get("text", ""))
            entry.setdefault("confidence_tesseract", reading.get("confidence", 0.0))
            entry["text_surya"], entry["confidence_surya"], entry["surya_note"] = (
                surya.cyrillic(answer.text),
                round(answer.confidence, 3),
                why,
            )
            if accepted:
                stats.accepted += 1
                text = surya.cyrillic(answer.text)
                ok, reason = acceptable(text, answer.confidence)
                # У surya пороги свои: её 0.5–0.6 на курсивном бланке — верные чтения.
                if not ok and answer.confidence >= surya.SURYA_MIN_CONFIDENCE and reason.startswith("уверенность"):
                    ok, reason = True, ""
                was = bool(entry.get("accepted"))
                entry.update(
                    {
                        "text": text,
                        "confidence": round(answer.confidence, 3),
                        "engine": "surya",
                        "accepted": ok,
                        "reason": reason,
                    }
                )
                if ok and not was:
                    stats.newly_accepted += 1
        save_page(cache, payload)
    return stats
