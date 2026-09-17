"""Выгрузка найденных полос оглавления в списки для внешнего OCR.

На выпуск — файл ``toc_pages.txt`` в папке выпуска ``<out-dir>/<год>/<выпуск>/``: по
полосе на строку, имя файла с заданным расширением (внешний OCR читает заострённые
``.jpg``, а в базе имена оригиналов ``.tif``), после «#» — вид и сила. Тот же формат
понимают ``--pages`` и ``--skip-pages`` у ``research.external_ocr_models run``: один и тот
же файл отправляет оглавление отдельным запросом и вычитает его из основного прогона.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ocr_utils.db.repo import require_pack
from ocr_utils.scan_markup.toc import KINDS, TOC_VERSION, kind_from_flags

LIST_NAME = "toc_pages.txt"


@dataclass
class ExportStats:
    issues: int = 0
    pages: int = 0
    # Выпуски, где детектор ещё не прогонялся (нет ``toc_version``) — их списки не пишутся.
    not_detected: list[str] = field(default_factory=list)
    # Выпуски без единой полосы оглавления: файл пишется пустым, чтобы было видно.
    empty: list[str] = field(default_factory=list)


def export_lists(
    session_factory,
    pack_name: str,
    out_dir: Path,
    suffix: str = ".jpg",
    kinds: tuple[str, ...] = KINDS,
    only_year: str | None = None,
    only_issue: str | None = None,
) -> ExportStats:
    stats = ExportStats()
    with session_factory() as session:
        pack = require_pack(session, pack_name)
        for year in pack.year_packages:
            if only_year is not None and year.name != only_year:
                continue
            for issue in year.issues:
                if only_issue is not None and issue.name != only_issue:
                    continue
                rel_dir = f"{year.name}/{issue.name}"
                pages = sorted(issue.pages, key=lambda p: p.order_index)
                if not pages or all(page.toc_version is None for page in pages):
                    stats.not_detected.append(rel_dir)
                    continue
                if any(page.toc_version != TOC_VERSION for page in pages):
                    stats.not_detected.append(rel_dir)
                    continue
                lines = [
                    f"{Path(page.source_file_name).with_suffix(suffix)}  # {kind} {page.toc_score or 0:.2f}"
                    f" {page.toc_source}"
                    for page in pages
                    # Вето человека «Не оглавление» сильнее любого признака.
                    if not page.force_is_not_toc and (kind := kind_from_flags(page.is_toc, page.is_year_index)) in kinds
                ]
                target = out_dir / rel_dir / LIST_NAME
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
                stats.issues += 1
                stats.pages += len(lines)
                if not lines:
                    stats.empty.append(rel_dir)
    return stats


__all__ = ["LIST_NAME", "ExportStats", "export_lists"]
