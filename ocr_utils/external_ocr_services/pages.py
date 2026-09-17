"""Полосы на входе и что о них знает база: обход папки, списки, флаги оглавления.

Раскладка входа — ``{год}/{выпуск}/{полоса}`` (как у ``sharpened`` и ``blurred``). Флаги
``is_toc`` / ``is_year_index`` / ``force_is_not_toc`` читаются из SQLite разметки напрямую
через ``sqlite3`` в режиме только-чтение: ORM при открытии дописывает недостающие колонки, а
этому пакету в базу писать нельзя вовсе. Полоса в базе значится под именем оригинала
(``.tif``), а на входе лежит заострённая копия (``.jpg``) — сопоставление по пути без суффикса.
Запасной вход без базы — списки ``toc_pages.txt`` из ``scan_markup toc-pages``.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
LIST_NAME = "toc_pages.txt"


@dataclass(frozen=True)
class PageFlags:
    """Что база знает о полосе. ``known=False`` — записи нет, полоса считается обычной."""

    is_toc: bool = False
    is_year_index: bool = False
    force_is_not_toc: bool = False
    known: bool = True

    @property
    def toc_kind(self) -> str | None:
        """``index`` / ``contents`` / ``None``; вето человека сильнее любого признака."""
        if self.force_is_not_toc:
            return None
        if self.is_year_index:
            return "index"
        if self.is_toc:
            return "contents"
        return None


UNKNOWN = PageFlags(known=False)


def page_key(rel: Path | str) -> str:
    """Ключ полосы для сопоставления входа с базой: путь без суффикса, через ``/``."""
    return Path(rel).with_suffix("").as_posix()


def read_page_list(path: Path) -> list[Path]:
    """Список относительных путей из файла: по одному на строку, после «#» — комментарий."""
    wanted = [line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [Path(item) for item in wanted if item]


def list_pages(
    in_dir: Path,
    pages_file: Path | None = None,
    only_year: str | None = None,
    only_issue: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    """Относительные пути полос: все картинки под ``in_dir`` или только из файла-списка.

    Папки с «_» в начале имени — служебные и пропускаются. ``only_year`` / ``only_issue``
    режут по первым двум компонентам пути.
    """
    if pages_file is not None:
        rels = read_page_list(pages_file)
        missing = [rel for rel in rels if not (in_dir / rel).is_file()]
        if missing:
            raise FileNotFoundError(f"в {in_dir} нет полос из списка: {', '.join(map(str, missing))}")
    else:
        rels = sorted(
            path.relative_to(in_dir)
            for path in in_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and not any(part.startswith("_") for part in path.relative_to(in_dir).parts[:-1])
        )
    if only_year is not None:
        rels = [rel for rel in rels if rel.parts[:1] == (only_year,)]
    if only_issue is not None:
        rels = [rel for rel in rels if len(rel.parts) >= 2 and rel.parts[1] == only_issue]
    if limit is not None:
        rels = rels[:limit]
    return rels


def group_by_issue(rels: list[Path]) -> dict[str, list[Path]]:
    """``{"год/выпуск": [полосы по имени]}`` в порядке выпусков. Полоса в корне — выпуск «.»."""
    groups: dict[str, list[Path]] = {}
    for rel in rels:
        groups.setdefault(rel.parent.as_posix(), []).append(rel)
    return {issue: sorted(pages) for issue, pages in groups.items()}


def flags_from_db(db_path: Path, pack_name: str) -> dict[str, PageFlags]:
    """Флаги оглавления всех полос пака из базы разметки, ключ — :func:`page_key`.

    База открывается только на чтение; колонки ``force_is_not_toc`` в старой базе может не быть —
    тогда вето считается не поставленным.
    """
    uri = f"file:{db_path}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(pages)")}
        veto = "p.force_is_not_toc" if "force_is_not_toc" in columns else "NULL"
        rows = connection.execute(
            f"""
            SELECT p.source_rel_path, p.is_toc, p.is_year_index, {veto}
            FROM pages p
            JOIN issues i ON p.issue_id = i.id
            JOIN year_packages y ON i.year_package_id = y.id
            JOIN packs k ON y.pack_id = k.id
            WHERE k.name = ?
            """,
            (pack_name,),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise LookupError(f"в базе {db_path} нет пака {pack_name!r} или у него нет полос")
    return {
        page_key(rel): PageFlags(bool(is_toc), bool(is_year_index), bool(force))
        for rel, is_toc, is_year_index, force in rows
    }


def flags_from_lists(lists_dir: Path) -> dict[str, PageFlags]:
    """Флаги из списков ``<год>/<выпуск>/toc_pages.txt`` (после «#» — вид: contents или index).

    Списки знают только полосы оглавления; полоса, которой в списке нет, считается обычной, а
    вето в этом формате не передаётся.
    """
    flags: dict[str, PageFlags] = {}
    for path in sorted(lists_dir.rglob(LIST_NAME)):
        issue_dir = path.parent.relative_to(lists_dir)
        for line in path.read_text(encoding="utf-8").splitlines():
            name, _, comment = line.partition("#")
            name = name.strip()
            if not name:
                continue
            kind = (comment.split() or ["contents"])[0]
            flags[page_key(issue_dir / name)] = PageFlags(is_toc=kind == "contents", is_year_index=kind == "index")
    return flags


def flags_for(rel: Path, table: dict[str, PageFlags] | None) -> PageFlags:
    """Флаги полосы; без таблицы или без записи — :data:`UNKNOWN`."""
    if table is None:
        return UNKNOWN
    return table.get(page_key(rel), UNKNOWN)
