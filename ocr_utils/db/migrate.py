"""Переименование колонок и таблиц в уже заполненных базах разметки.

Зачем отдельный скрипт. ``db.session.add_missing_columns`` умеет только ДОПИСЫВАТЬ
колонки, и это правильно: дописать пустую колонку безопасно при любом исходе, поэтому
такое можно делать молча при каждом открытии базы. Переименование безопасным не бывает —
оно либо прошло целиком, либо нет, — и делать его украдкой посреди чужого прогона нельзя.
Тем более что в базе лежит ручная разметка из CVAT, которой на диске больше нигде нет:
она собиралась неделями и пересоздать её невозможно.

Порядок операций внутри — единственно возможный. Сначала ПЕРЕИМЕНОВАНИЕ через сырой
``sqlite3``, и только потом открытие базы через ``open_db``. Наоборот нельзя: открытие
дописало бы пустую ``source_file_name`` РЯДОМ со старой ``file_name``, данные остались бы
в старой колонке, а на вид база выглядела бы мигрированной.

Скрипт идемпотентен: колонка (и таблица) переименовывается, только если старое имя в файле
есть, а нового ещё нет. Повторный прогон по мигрированной базе ничего не делает и ничего
не портит.

Таблицы переименовываются раньше колонок: список колонок назван новыми именами таблиц.
Индексы SQLite при ``ALTER TABLE ... RENAME TO`` сохраняют старые имена
(``ix_raster_regions_page_id``); такой индекс удаляется, а нужный заводит ``open_db`` через
``add_missing_columns``/``create_all`` под именем из моделей.

Запуск::

    python -m ocr_utils.db.migrate путь/к/base.sqlite [ещё.sqlite ...]
    python -m ocr_utils.db.migrate --dry-run base.sqlite
"""

import logging
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import click

logger = logging.getLogger(__name__)

# Суффикс копии, снимаемой перед переименованием. Отдельный от ``.bak``, который делает
# run_scripts/scan_markup/pack1/common.sh перед каждым шагом: тот перезаписывается на
# следующем же запуске, а эта копия должна пережить всю миграцию.
BACKUP_SUFFIX = ".bak-до-переименования"

# Переименованные таблицы: (старое имя, новое имя, индексы старой таблицы под снос).
TABLE_RENAMES: "tuple[tuple[str, str, tuple[str, ...]], ...]" = (
    ("raster_regions", "rect_regions", ("ix_raster_regions_page_id",)),
)

# Что во что переименовывается: таблица -> (старое имя, новое имя).
RENAMES: "tuple[tuple[str, str, str], ...]" = (
    ("packs", "root_path", "source_pics_root"),
    ("pages", "file_name", "source_file_name"),
    ("pages", "rel_path", "source_rel_path"),
    # Детектор крупного штриха (stroke_*) заменён единым line art из page_layout; пара
    # версия/время переименована под него, находки прежних видов удаляются (DELETES).
    ("pages", "stroke_detector_version", "line_art_version"),
    ("pages", "strokes_detected_at", "line_art_detected_at"),
)

# Строки, УБРАННЫЕ из данных: (таблица, колонка, значения). Виды ``stroke_table`` и
# ``stroke_drawing`` жили один прогон (детектор крупного штриха на detect) и заменены
# единым ``line_art_schema``; их автоматические находки в базе ничего не значат.
DELETES: "tuple[tuple[str, str, tuple[str, ...]], ...]" = (
    ("rect_regions", "kind", ("stroke_table", "stroke_drawing")),
)

# Колонки, УБРАННЫЕ из схемы: (таблица, колонка). Сиротская колонка базе не мешает —
# SQLAlchemy её не видит, — но и лежать пустой ей незачем. ``pages.toc_kind`` жил один
# прогон: вид оглавления строкой заменили двумя признаками ``is_toc`` и ``is_year_index``
# (в CVAT им отвечают два независимых тега).
DROPS: "tuple[tuple[str, str], ...]" = (("pages", "toc_kind"),)


@dataclass
class MigrationReport:
    """Что случилось с одной базой."""

    path: Path
    backup: "Path | None" = None
    renamed: "list[str]" = field(default_factory=list)
    dropped: "list[str]" = field(default_factory=list)
    deleted: "list[str]" = field(default_factory=list)
    already: "list[str]" = field(default_factory=list)
    added: "list[str]" = field(default_factory=list)
    missing_tables: "list[str]" = field(default_factory=list)

    def lines(self) -> "list[str]":
        out = [f"{self.path}:"]
        if self.backup is not None:
            out.append(f"  копия: {self.backup.name}")
        out.append(f"  переименовано: {', '.join(self.renamed) if self.renamed else '(нечего)'}")
        if self.dropped:
            out.append(f"  убрано колонок: {', '.join(self.dropped)}")
        if self.deleted:
            out.append(f"  удалено строк: {'; '.join(self.deleted)}")
        if self.already:
            out.append(f"  уже было переименовано: {', '.join(self.already)}")
        if self.missing_tables:
            out.append(f"  таблиц нет вовсе: {', '.join(self.missing_tables)}")
        out.append(f"  дописано колонок: {', '.join(self.added) if self.added else '(нечего)'}")
        return out


def _table_columns(connection: sqlite3.Connection, table: str) -> "set[str] | None":
    """Имена колонок таблицы; ``None``, если таблицы в файле нет."""
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchall()
    if not rows:
        return None
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _has_table(connection: sqlite3.Connection, table: str) -> bool:
    return _table_columns(connection, table) is not None


def rename_columns(db_path: Path, dry_run: bool = False) -> MigrationReport:
    """Переименовывает таблицы по :data:`TABLE_RENAMES` и колонки по :data:`RENAMES`, убирает
    колонки по :data:`DROPS`; НЕ дописывает новых.

    Работает сырым ``sqlite3`` и ``ALTER TABLE ... RENAME COLUMN`` (SQLite 3.25+): он же
    сам переписывает и UNIQUE-констрейнт ``uq_page_in_issue``, который назван колонкой
    ``file_name`` по имени. ``ALTER TABLE ... RENAME TO`` заодно правит внешние ключи,
    ссылающиеся на таблицу (``PRAGMA legacy_alter_table`` выключена по умолчанию).
    """
    report = MigrationReport(path=db_path)
    with sqlite3.connect(db_path) as connection:
        for old, new, indexes in TABLE_RENAMES:
            if _has_table(connection, new):
                report.already.append(f"{old} -> {new}")
                continue
            if not _has_table(connection, old):
                if old not in report.missing_tables:
                    report.missing_tables.append(old)
                continue
            if not dry_run:
                connection.execute(f'ALTER TABLE "{old}" RENAME TO "{new}"')
                for index in indexes:
                    connection.execute(f'DROP INDEX IF EXISTS "{index}"')
            report.renamed.append(f"{old} -> {new}")
        for table, old, new in RENAMES:
            columns = _table_columns(connection, table)
            if columns is None:
                if table not in report.missing_tables:
                    report.missing_tables.append(table)
                continue
            if new in columns:
                report.already.append(f"{table}.{old} -> {new}")
                continue
            if old not in columns:
                # Ни старого, ни нового имени: таблица есть, а колонки нет вовсе. Молчать
                # тут нельзя — это не «уже мигрировано», это чужая или битая схема.
                logger.warning("В таблице %s нет ни %s, ни %s — пропускаю", table, old, new)
                continue
            if not dry_run:
                connection.execute(f'ALTER TABLE "{table}" RENAME COLUMN "{old}" TO "{new}"')
            report.renamed.append(f"{table}.{old} -> {new}")
        for table, column in DROPS:
            columns = _table_columns(connection, table)
            if columns is None or column not in columns:
                continue
            if not dry_run:
                connection.execute(f'ALTER TABLE "{table}" DROP COLUMN "{column}"')  # SQLite 3.35+
            report.dropped.append(f"{table}.{column}")
        for table, column, values in DELETES:
            columns = _table_columns(connection, table)
            if columns is None or column not in columns:
                continue
            marks = ",".join("?" for _ in values)
            count = connection.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" IN ({marks})', values
            ).fetchone()[0]
            if not count:
                continue
            if not dry_run:
                connection.execute(f'DELETE FROM "{table}" WHERE "{column}" IN ({marks})', values)
            report.deleted.append(f"{table}.{column} in {values}: {count}")
    return report


def migrate_db(db_path: Path, backup: bool = True, dry_run: bool = False) -> MigrationReport:
    """Полная миграция одной базы: копия, переименование, дописывание новых колонок."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"базы нет: {db_path}")

    backup_path: "Path | None" = None
    if backup and not dry_run:
        backup_path = db_path.with_name(db_path.name + BACKUP_SUFFIX)
        shutil.copyfile(db_path, backup_path)

    report = rename_columns(db_path, dry_run=dry_run)
    report.backup = backup_path

    if not dry_run:
        # Импорт внутри функции: ``session`` тянет за собой модели и SQLAlchemy, а
        # переименование обязано отработать ДО того, как схема из моделей коснётся файла.
        from sqlalchemy import create_engine

        from ocr_utils.db.session import add_missing_columns

        engine = create_engine(f"sqlite:///{db_path}")
        report.added = add_missing_columns(engine)
        engine.dispose()

    return report


@click.command()
@click.argument("databases", nargs=-1, required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--backup/--no-backup", default=True, show_default=True, help="Снять копию базы перед правкой.")
@click.option("--dry-run", is_flag=True, help="Только показать, что было бы сделано.")
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
def main(databases: "tuple[Path, ...]", backup: bool, dry_run: bool, log_level: str) -> None:
    """Мигрирует БАЗЫ на текущую схему: переименованные колонки и таблицы, новые колонки."""
    logging.basicConfig(level=log_level.upper(), format="%(levelname)s: %(message)s")
    for db_path in databases:
        if not db_path.exists():
            click.echo(f"{db_path}: базы нет, пропускаю")
            continue
        report = migrate_db(db_path, backup=backup, dry_run=dry_run)
        for line in report.lines():
            click.echo(line)


if __name__ == "__main__":
    main()
