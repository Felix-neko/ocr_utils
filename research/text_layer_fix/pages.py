"""Выбор страниц для прогона и привязка «страница PDF ↔ полоса пака».

Номер страницы в полном PDF выпуска записан в ``pages.full_pdf_page_idx`` базы-зонда
``_margin_probe/db.sqlite`` (в рабочих базах разметки он пуст); в самих базах разметки лежат
области (таблицы, схемы, растр) и поворот полосы ``rotate_cw``. Оба источника — только чтение.
"""

from __future__ import annotations

import csv
import random
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class SampleKind(StrEnum):
    """Категория страницы в выборке — по какому признаку она попала в прогон."""

    ROTATED_TABLE = "rotated_table"  # таблица с боковыми ячейками по прогону rotated_text
    LINE_ART = "line_art"  # ручная разметка line_art_schema
    PAGE_ROTATED = "page_rotated"  # полоса повёрнута целиком (rotate_cw != 0)
    HIGH_COVERAGE = "high_coverage"  # крупный line art по CSV line_art_detection
    CONTROL = "control"  # случайная контрольная страница


@dataclass(frozen=True)
class PageRef:
    """Страница выпуска: файл PDF, номер страницы и полоса пака, которой она отвечает."""

    pdf: str  # имя файла, например full_1966_01.pdf
    index: int  # номер страницы с нуля
    year: str = ""
    issue: str = ""
    sheet: str = ""  # имя полосы без расширения: IMG_0044_2R
    rotate_cw: int = 0  # поворот полосы целиком по базе разметки

    @property
    def key(self) -> tuple[str, int]:
        return (self.pdf, self.index)

    @property
    def label(self) -> str:
        """Короткая подпись для файлов и отчёта: ``full_1966_01_p0042``."""
        return f"{Path(self.pdf).stem}_p{self.index:04d}"


def page_index_map(probe_db: Path) -> dict[tuple[str, str, str], PageRef]:
    """Все полосы пака с номерами страниц в полных PDF.

    Args:
        probe_db: База-зонд с заполненным ``full_pdf_page_idx`` и именами PDF выпусков.

    Returns:
        Словарь ``(год, выпуск, имя полосы без расширения) → PageRef`` (без ``rotate_cw``).
    """
    connection = sqlite3.connect(f"file:{probe_db}?mode=ro", uri=True)
    rows = connection.execute(
        """
        select y.year, i.name, p.source_file_name, i.full_intermediate_pdf_name, p.full_pdf_page_idx
        from pages p join issues i on i.id = p.issue_id join year_packages y on y.id = i.year_package_id
        where p.full_pdf_page_idx is not null and i.full_intermediate_pdf_name is not null
        """
    ).fetchall()
    connection.close()
    result: dict[tuple[str, str, str], PageRef] = {}
    for year, issue, file_name, pdf_name, index in rows:
        sheet = Path(file_name).stem
        result[(str(year), str(issue), sheet)] = PageRef(pdf_name, int(index), str(year), str(issue), sheet)
    return result


def page_rotations(markup_db: Path) -> dict[tuple[str, str, str], int]:
    """Поворот каждой полосы целиком по базе разметки.

    Args:
        markup_db: База разметки (``pack1_reviewed.sqlite``), только чтение.

    Returns:
        Словарь ``(год, выпуск, полоса) → rotate_cw``.
    """
    connection = sqlite3.connect(f"file:{markup_db}?mode=ro", uri=True)
    rows = connection.execute(
        """
        select y.year, i.name, p.source_file_name, coalesce(p.rotate_cw, 0)
        from pages p join issues i on i.id = p.issue_id join year_packages y on y.id = i.year_package_id
        """
    ).fetchall()
    connection.close()
    return {(str(y), str(i), Path(f).stem): int(r) for y, i, f, r in rows}


def sheets_with_regions(markup_db: Path, kinds: tuple[str, ...]) -> set[tuple[str, str, str]]:
    """Полосы, на которых в базе есть области указанных видов.

    Args:
        markup_db: База разметки, только чтение.
        kinds: Виды областей (``table``, ``line_art_schema`` …).

    Returns:
        Множество ключей ``(год, выпуск, полоса)``.
    """
    connection = sqlite3.connect(f"file:{markup_db}?mode=ro", uri=True)
    marks = ",".join("?" for _ in kinds)
    rows = connection.execute(
        f"""
        select distinct y.year, i.name, p.source_file_name
        from rect_regions r join pages p on p.id = r.page_id
        join issues i on i.id = p.issue_id join year_packages y on y.id = i.year_package_id
        where r.kind in ({marks})
        """,
        kinds,
    ).fetchall()
    connection.close()
    return {(str(y), str(i), Path(f).stem) for y, i, f in rows}


def sheets_with_rotated_tables(summary_csv: Path) -> set[tuple[str, str, str]]:
    """Полосы с таблицами, где прогон ``rotated_text`` нашёл боковые ячейки.

    Args:
        summary_csv: ``summary.csv`` прогона (``/mnt/SYSTEM/raw/mts/pack1_rotated_tables``).

    Returns:
        Множество ключей ``(год, выпуск, полоса)``.
    """
    result: set[tuple[str, str, str]] = set()
    with open(summary_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row.get("rotated") or 0) > 0:
                sheet = Path(row["page"]).stem
                result.add((row["year"], row["issue"], sheet))
    return result


def pages_with_coverage(line_art_csv: Path, min_coverage: float) -> set[tuple[str, int]]:
    """Страницы с покрытием line art не ниже порога по CSV ``line_art_detection``.

    Args:
        line_art_csv: CSV прогона ``line_art_detection scan``.
        min_coverage: Порог доли площади полосы.

    Returns:
        Множество ``(имя PDF, номер страницы с нуля)``.
    """
    result: set[tuple[str, int]] = set()
    with open(line_art_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                coverage = float(row.get("coverage") or 0)
                page = int(row.get("page") or 0)
            except ValueError:
                continue
            if coverage >= min_coverage:
                # В CSV лежит полный путь к PDF другого прогона FineReader; нужен только файл.
                pdf = Path(row.get("pdf") or row.get("file") or "").name
                if not pdf.endswith(".pdf"):
                    pdf += ".pdf"
                result.add((pdf, page - 1))
    return result


@dataclass(frozen=True)
class Sample:
    """Страница выборки с категорией (первой, по которой она была отобрана)."""

    ref: PageRef
    kind: SampleKind


def sample_pages(
    index: dict[tuple[str, str, str], PageRef],
    rotations: dict[tuple[str, str, str], int],
    rotated_tables: set[tuple[str, str, str]],
    line_art: set[tuple[str, str, str]],
    high_coverage: set[tuple[str, int]],
    controls: int,
    seed: int = 7,
) -> list[Sample]:
    """Собрать выборку страниц по категориям (страница считается один раз, в первой категории).

    Args:
        index: Все полосы с номерами страниц (:func:`page_index_map`).
        rotations: Поворот полос по базе разметки.
        rotated_tables: Полосы с боковыми ячейками.
        line_art: Полосы с ручной разметкой схем.
        high_coverage: Страницы с крупным line art по CSV.
        controls: Сколько случайных контрольных страниц добавить.
        seed: Зерно генератора для воспроизводимости.

    Returns:
        Список страниц выборки, отсортированный по PDF и номеру страницы.
    """
    chosen: dict[tuple[str, int], Sample] = {}

    def add(ref: PageRef, kind: SampleKind) -> None:
        chosen.setdefault(ref.key, Sample(ref, kind))

    refs = {key: PageRef(r.pdf, r.index, r.year, r.issue, r.sheet, rotations.get(key, 0)) for key, r in index.items()}
    for key in sorted(rotated_tables):
        if key in refs:
            add(refs[key], SampleKind.ROTATED_TABLE)
    for key in sorted(line_art):
        if key in refs:
            add(refs[key], SampleKind.LINE_ART)
    for key, ref in sorted(refs.items()):
        if ref.rotate_cw:
            add(ref, SampleKind.PAGE_ROTATED)
    by_page = {ref.key: ref for ref in refs.values()}
    for key in sorted(high_coverage):
        if key in by_page:
            add(by_page[key], SampleKind.HIGH_COVERAGE)
    rest = [ref for key, ref in sorted(by_page.items()) if key not in chosen]
    random.Random(seed).shuffle(rest)
    for ref in rest[:controls]:
        add(ref, SampleKind.CONTROL)
    return sorted(chosen.values(), key=lambda s: s.ref.key)
