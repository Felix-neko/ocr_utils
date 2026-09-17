"""Открытие SQLite-базы: движок, ``PRAGMA foreign_keys``, создание недостающих таблиц."""

import logging
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from ocr_utils.db.models import Base

logger = logging.getLogger(__name__)


@event.listens_for(Engine, "connect")
def _enable_foreign_keys(dbapi_connection, connection_record) -> None:
    """Включает проверку внешних ключей: в SQLite она по умолчанию ВЫКЛЮЧЕНА.

    Без этого ``ondelete="CASCADE"`` на уровне БД молча не работает, и удаление пака
    оставляет висячие годы, выпуски и полосы. ORM-каскад ``delete-orphan`` спасает только
    когда объекты загружены в сессию, а массовые ``delete()`` идут мимо него.

    Хук вешается на все движки процесса, включая чужие: pragma для не-SQLite соединения
    просто не выполнится, поэтому проверяем тип по факту.
    """
    if not hasattr(dbapi_connection, "execute"):  # не DB-API 2.0 — не наше дело
        return
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception:  # noqa: BLE001 — движок не SQLite, pragma неизвестна
        pass


def open_db(path: Path, create: bool = True) -> sessionmaker[Session]:
    """Фабрика сессий для базы ``path``; создаёт недостающие таблицы.

    Существующая база НЕ переписывается: ``create_all`` заводит только те таблицы, которых
    в файле ещё нет. Именно поэтому один и тот же файл спокойно накапливает несколько
    паков — новый прогон ``detect`` просто дописывает свой.

    Про новые колонки в старых таблицах ``create_all`` не знает вовсе, поэтому следом идёт
    :func:`add_missing_columns` — см. её докстринг.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}")
    check_schema_is_current(engine, path)
    if create:
        Base.metadata.create_all(engine)
        add_missing_columns(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


# Колонки, ПЕРЕИМЕНОВАННЫЕ в схеме: таблица -> (старое имя, новое имя). Держатся здесь, а
# не импортируются из ``db.migrate``, чтобы зависимость шла в одну сторону: миграция знает
# про открытие базы, открытие про миграцию — только по имени команды в тексте ошибки.
RENAMED_COLUMNS: "tuple[tuple[str, str, str], ...]" = (
    ("packs", "root_path", "source_pics_root"),
    ("pages", "file_name", "source_file_name"),
    ("pages", "rel_path", "source_rel_path"),
)

# ПЕРЕИМЕНОВАННЫЕ ТАБЛИЦЫ: (старое имя, новое имя). Признак старой базы тот же — старая
# есть, новой нет. Опасность та же, что у колонок, только крупнее: ``create_all`` завёл бы
# пустую ``rect_regions`` РЯДОМ с заполненной ``raster_regions``, и вся разметка растра
# пропала бы из виду, оставаясь в файле.
RENAMED_TABLES: "tuple[tuple[str, str], ...]" = (("raster_regions", "rect_regions"),)


def check_schema_is_current(engine: Engine, path: Path) -> None:
    """Отказывается открывать базу прошлой схемы, называя команду миграции.

    Без этой проверки старая база открылась бы как ни в чём не бывало:
    :func:`add_missing_columns` дописал бы пустую ``source_file_name`` РЯДОМ со старой
    ``file_name``, все прогоны увидели бы пустые имена файлов, а падение случилось бы
    где-то далеко и по другому поводу. Ловим ровно тот признак, который отличает старую
    базу от новой, — старое имя есть, нового нет.
    """
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    stale: "list[str]" = []
    for old, new in RENAMED_TABLES:
        if old in tables and new not in tables:
            stale.append(f"таблица {old} (должна быть {new})")
    for table, old, new in RENAMED_COLUMNS:
        if table not in tables:
            continue
        present = {column["name"] for column in inspector.get_columns(table)}
        if old in present and new not in present:
            stale.append(f"{table}.{old} (должна быть {new})")
    if stale:
        raise RuntimeError(
            f"база {path} — прошлой схемы: {', '.join(stale)}. "
            f"Выполните: uv run python -m ocr_utils.db.migrate {path}"
        )


def add_missing_columns(engine: Engine) -> list[str]:
    """Дописывает в уже существующие таблицы колонки, появившиеся в моделях позже.

    ``create_all`` заводит только недостающие ТАБЛИЦЫ и ничего не знает про новые колонки
    в старых: база, заведённая прошлой версией схемы, после обновления моделей падала бы
    на ``no such column``. Цена такого падения здесь несоразмерна — прогон ``detect`` по
    паку-1 это около четырёх часов GPU и полтерабайта чтения с медленного диска, и терять
    его из-за одной новой колонки нельзя.

    Добавляются только колонки, допускающие NULL: у существующих строк нового значения
    взяться неоткуда, и старая строка обязана оставаться читаемой. Колонка NOT NULL — это
    уже настоящая миграция с заполнением, и делать её украдкой при открытии базы нельзя;
    такие пропускаются с предупреждением.

    Возвращает список добавленного, в виде ``таблица.колонка``.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    added: list[str] = []

    with engine.begin() as connection:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # её только что создал create_all, колонки уже все на месте
            present = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                if not column.nullable:
                    logger.warning(
                        "Колонка %s.%s объявлена NOT NULL — автоматически добавить её нельзя, "
                        "старым строкам нечего в неё положить. Базу придётся пересоздать.",
                        table.name,
                        column.name,
                    )
                    continue
                column_type = column.type.compile(engine.dialect)
                connection.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column_type}'))
                added.append(f"{table.name}.{column.name}")

    if added:
        logger.warning("В базу дописаны новые колонки: %s", ", ".join(added))
        # Индексы на новых колонках create_all тоже не заводит: таблица уже существует.
        for table in Base.metadata.sorted_tables:
            if table.name in existing_tables:
                for index in table.indexes:
                    index.create(bind=engine, checkfirst=True)
    return added
