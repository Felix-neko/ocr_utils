"""Полосы на входе и что о них знает база: обход папки, списки, флаги оглавления.

Раскладка входа — ``{год}/{выпуск}/{полоса}`` (как у ``sharpened`` и ``blurred``). Флаги
``is_toc`` / ``is_year_index`` / ``force_is_not_toc`` читаются из базы разметки через ORM
``ocr_utils.db`` с ``open_db(create=False)``: без создания таблиц и дописывания колонок, одним
``select`` и без ``commit`` — этому пакету в базу писать нельзя вовсе. Полоса в базе значится
под именем оригинала (``.tif``), а на входе лежит заострённая копия (``.jpg``) — сопоставление
по пути без суффикса. Запасной вход без базы — списки ``toc_pages.txt`` из ``scan_markup toc-pages``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import inspect, select

from ocr_utils.db.models import Issue, Page, YearPackage
from ocr_utils.db.repo import require_pack
from ocr_utils.db.session import open_db
from ocr_utils.external_ocr_services.schema import TocKind

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
LIST_NAME = "toc_pages.txt"


@dataclass(frozen=True)
class PageFlags:
    """Что база знает о полосе. ``known=False`` — записи нет, полоса считается обычной.

    Args:
        is_toc: Тег CVAT «Оглавление» — полоса «Содержания» выпуска.
        is_year_index: Тег «Годовой указатель» — полоса указателя статей за год.
        force_is_not_toc: Вето человека «Не оглавление»: сильнее любого признака и ответа модели.
        known: В базе (или списках) есть запись об этой полосе; ``False`` — источник тегов есть,
            а полосы в нём нет.
    """

    is_toc: bool = False
    is_year_index: bool = False
    force_is_not_toc: bool = False
    known: bool = True

    @property
    def toc_kind(self) -> TocKind | None:
        """``INDEX`` / ``CONTENTS`` / ``None`` (обычная полоса); вето человека сильнее любого признака."""
        if self.force_is_not_toc:
            return None
        if self.is_year_index:
            return TocKind.INDEX
        if self.is_toc:
            return TocKind.CONTENTS
        return None


UNKNOWN = PageFlags(known=False)


def page_key(rel: Path | str) -> str:
    """Ключ полосы для сопоставления входа с базой: путь без суффикса, через ``/``.

    В базе полоса значится как ``1966/03/IMG_0104_2R.tif``, на входе лежит ``….jpg`` — суффикс
    отбрасывается, разделитель приводится к ``/``.

    Args:
        rel: Путь полосы относительно корня входа или ``source_rel_path`` из базы.
    """
    return Path(rel).with_suffix("").as_posix()


def read_page_list(path: Path) -> list[Path]:
    """Список относительных путей из файла: по одному на строку, после «#» — комментарий.

    Args:
        path: Текстовый файл (``--pages``); пустые строки и комментарии пропускаются.
    """
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

    Args:
        in_dir: Корень входа с раскладкой ``{год}/{выпуск}/{полоса}``.
        pages_file: Файл со списком относительных путей вместо обхода; путь, которого нет на
            диске, — ``FileNotFoundError`` (опечатка в списке не должна молча пропасть).
        only_year: Оставить только полосы этого годового комплекта (первая папка пути).
        only_issue: Оставить только полосы этого выпуска (вторая папка пути).
        limit: Взять первые N полос после всех отборов — для проб.
    """
    if pages_file is not None:
        rels = read_page_list(pages_file)
        missing = [rel for rel in rels if not (in_dir / rel).is_file()]
        if missing:
            raise FileNotFoundError(f"в {in_dir} нет полос из списка: {', '.join(map(str, missing))}")
    else:
        # Обход всего дерева: только файлы-картинки, минус служебные папки вида «_overlays».
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
    """``{"год/выпуск": [полосы по имени]}`` в порядке выпусков. Полоса в корне — выпуск «.».

    Args:
        rels: Относительные пути полос (из ``list_pages``), уже отсортированные.
    """
    groups: dict[str, list[Path]] = {}
    for rel in rels:
        groups.setdefault(rel.parent.as_posix(), []).append(rel)
    return {issue: sorted(pages) for issue, pages in groups.items()}


def flags_from_db(db_path: Path, pack_name: str) -> dict[str, PageFlags]:
    """Флаги оглавления всех полос пака из базы разметки, ключ — :func:`page_key`.

    База открывается через :func:`open_db` с ``create=False``: ни ``create_all``, ни дописывания
    колонок, только чтение. Колонки ``force_is_not_toc`` в старой базе может не быть — тогда она
    не запрашивается, и вето считается не поставленным.

    Args:
        db_path: Файл SQLite базы разметки (обычно ``pack1_reviewed.sqlite``).
        pack_name: Имя пака в базе (``packs.name``); нет такого — ``LookupError`` с перечнем паков.
    """
    factory = open_db(db_path, create=False)
    with factory() as session:
        pack = require_pack(session, pack_name)  # LookupError с перечнем паков, если имя не то
        # Только нужные колонки, а не целые Page: полос в паке 12 тыс., а колонок у полосы за 40.
        columns = [Page.source_rel_path, Page.is_toc, Page.is_year_index]
        present = {column["name"] for column in inspect(session.get_bind()).get_columns(Page.__tablename__)}
        has_veto = "force_is_not_toc" in present
        if has_veto:
            columns.append(Page.force_is_not_toc)
        query = (
            select(*columns)
            .join(Issue, Page.issue_id == Issue.id)
            .join(YearPackage, Issue.year_package_id == YearPackage.id)
            .where(YearPackage.pack_id == pack.id)
        )
        rows = session.execute(query).all()
    if not rows:
        raise LookupError(f"в базе {db_path} у пака {pack_name!r} нет полос")
    return {
        page_key(row[0]): PageFlags(bool(row[1]), bool(row[2]), bool(row[3]) if has_veto else False) for row in rows
    }


def flags_from_lists(lists_dir: Path) -> dict[str, PageFlags]:
    """Флаги из списков ``<год>/<выпуск>/toc_pages.txt`` (после «#» — вид: contents или index).

    Списки знают только полосы оглавления; полоса, которой в списке нет, считается обычной, а
    вето в этом формате не передаётся.

    Args:
        lists_dir: Корень списков той же раскладки, что вход: ``<год>/<выпуск>/toc_pages.txt``.
    """
    flags: dict[str, PageFlags] = {}
    for path in sorted(lists_dir.rglob(LIST_NAME)):
        issue_dir = path.parent.relative_to(lists_dir)
        for line in path.read_text(encoding="utf-8").splitlines():
            # «IMG_0002.jpg  # contents 1.00 cvat»: имя до «#», первое слово после — вид.
            name, _, comment = line.partition("#")
            name = name.strip()
            if not name:
                continue
            kind = (comment.split() or [TocKind.CONTENTS.value])[0]
            flags[page_key(issue_dir / name)] = PageFlags(
                is_toc=kind == TocKind.CONTENTS, is_year_index=kind == TocKind.INDEX
            )
    return flags


def flags_for(rel: Path, table: dict[str, PageFlags] | None) -> PageFlags:
    """Флаги полосы; без таблицы или без записи — :data:`UNKNOWN`.

    Args:
        rel: Путь полосы относительно корня входа.
        table: Таблица флагов из ``flags_from_db`` / ``flags_from_lists``; ``None`` — источника нет.
    """
    if table is None:
        return UNKNOWN
    return table.get(page_key(rel), UNKNOWN)
