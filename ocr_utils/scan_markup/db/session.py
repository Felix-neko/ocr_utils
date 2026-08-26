"""Открытие SQLite-базы: движок, ``PRAGMA foreign_keys``, создание недостающих таблиц."""

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ocr_utils.scan_markup.db.models import Base


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
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}")
    if create:
        Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
