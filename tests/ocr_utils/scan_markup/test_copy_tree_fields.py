"""Копирование дерева в reviewed-базу не должно терять поля.

`cvat/export.py:copy_tree` перечисляет поля ПОИМЁННО, и колонка, забытая в этом списке,
теряется молча: reviewed-база выглядит целой, а значение в ней NULL. Этот тест ловит такую
забывчивость по факту — сравнивает значения всех колонок, а не читает исходник.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect

from ocr_utils.scan_markup.cvat.export import copy_tree
from ocr_utils.db.models import Issue, Page, Pack, YearPackage
from ocr_utils.db.session import open_db
from ocr_utils.db.repo import require_pack
from ocr_utils.scan_markup.scan_tree import ScannedIssue, ScannedPage, ScannedYear
from ocr_utils.db.repo import upsert_pack

# Колонки, которые копировать НЕ надо, и почему.
SKIP = {
    Pack: {"id", "created_at"},
    YearPackage: {"id", "pack_id"},
    Issue: {"id", "year_package_id"},
    # reviewed_at проставляет сам импорт разметки — копировать его из исходной базы значило бы
    # объявить проверенным то, что ещё не смотрели.
    Page: {"id", "issue_id", "reviewed_at"},
}


def _sample(kind):
    """Непустое значение нужного типа: NULL совпал бы сам с собой и скрыл потерю."""
    if kind is bool:
        return True
    if kind is int:
        return 7
    if kind is float:
        return 1.5
    if kind is datetime:
        return datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc).replace(tzinfo=None)
    return "x"


def build_source(tmp_path):
    db = tmp_path / "src.sqlite"
    factory = open_db(db)
    with factory() as session:
        pages = [ScannedPage(tmp_path / "IMG_0001_1L.tif", "IMG_0001_1L.tif", "1967/01/IMG_0001_1L.tif", 0)]
        years = [ScannedYear("1967", 1967, "1967", [ScannedIssue("01", 1, "1967/01", pages)])]
        pack = upsert_pack(session, "пак-тест", tmp_path, years)
        pack.allowed_rotations = "0,90"
        pack.year_packages[0].cvat_task_id = 42
        issue = pack.year_packages[0].issues[0]
        issue.allowed_rotations = "0,90,180"
        page = issue.pages[0]
        # Заполняем ВСЕ необязательные колонки, иначе забытое поле совпало бы по NULL.
        for column in inspect(Page).columns:
            if column.key in SKIP[Page] or getattr(page, column.key) is not None:
                continue
            setattr(page, column.key, _sample(column.type.python_type))
        session.commit()
    return db


@pytest.mark.parametrize("model", [Pack, YearPackage, Issue, Page])
def test_copy_tree_carries_every_column(tmp_path, model):
    src_db = build_source(tmp_path)
    dst_db = tmp_path / "dst.sqlite"
    src_factory, dst_factory = open_db(src_db), open_db(dst_db)
    with src_factory() as src, dst_factory() as dst:
        copy_tree(src, dst, "пак-тест")
    with src_factory() as src, dst_factory() as dst:
        source = require_pack(src, "пак-тест")
        target = require_pack(dst, "пак-тест")
        pairs = {
            Pack: (source, target),
            YearPackage: (source.year_packages[0], target.year_packages[0]),
            Issue: (source.year_packages[0].issues[0], target.year_packages[0].issues[0]),
            Page: (source.year_packages[0].issues[0].pages[0], target.year_packages[0].issues[0].pages[0]),
        }[model]
        lost = [
            column.key
            for column in inspect(model).columns
            if column.key not in SKIP[model] and getattr(pairs[0], column.key) != getattr(pairs[1], column.key)
        ]
        assert not lost, f"copy_tree потерял колонки: {lost}"
